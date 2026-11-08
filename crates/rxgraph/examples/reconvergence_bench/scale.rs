//! Larger graphs and search spaces, with bounded result retention.
use super::*;
use rxgraph::OwnedGraphId;

const BATCH_SIZE: usize = 4096;

#[derive(Clone, Copy)]
pub(crate) enum Shape {
    Diamonds {
        components: u64,
        depth: u64,
        bits: u64,
    },
    Funnel {
        components: u64,
        branches: u64,
        tail: u64,
    },
    Tree {
        depth: u64,
    },
}

pub(crate) struct Dataset {
    pub(crate) case: Case,
    pub(crate) starts: Vec<OwnedGraphId>,
    pub(crate) shape: Shape,
}

impl Shape {
    pub(crate) fn build(self) -> Result<Dataset> {
        let (case, starts) = match self {
            Self::Diamonds {
                components,
                depth,
                bits,
            } => {
                let terminal_from = components * 3 * depth;
                let mut edges = Vec::with_capacity((components * depth * 4) as usize);
                for component in 0..components {
                    for stage in 0..depth {
                        let merge = component * 3 * depth + stage * 3;
                        let next = if stage + 1 == depth {
                            terminal_from + component
                        } else {
                            merge + 3
                        };
                        edges.extend([
                            (merge, merge + 1, 2 * stage, 1),
                            (merge, merge + 2, 2 * stage, 2),
                            (merge + 1, next, 2 * stage + 1, 0),
                            (merge + 2, next, 2 * stage + 1, 0),
                        ]);
                    }
                }
                (
                    case(
                        terminal_from + components,
                        terminal_from,
                        &edges,
                        bits,
                        (components << depth) as usize,
                    )?,
                    (0..components).map(|c| (c * 3 * depth).into()).collect(),
                )
            }
            Self::Funnel {
                components,
                branches,
                tail,
            } => {
                let width = branches + tail + 1;
                let terminal_from = components * width;
                let mut edges = Vec::with_capacity((components * (2 * branches + tail)) as usize);
                for component in 0..components {
                    let base = component * width;
                    let merge = base + branches + 1;
                    for branch in 1..=branches {
                        edges.extend([(base, base + branch, 0, 0), (base + branch, merge, 1, 0)]);
                    }
                    for step in 0..tail {
                        let next = if step + 1 == tail {
                            terminal_from + component
                        } else {
                            merge + step + 1
                        };
                        edges.push((merge + step, next, step + 2, 0));
                    }
                }
                (
                    case(
                        terminal_from + components,
                        terminal_from,
                        &edges,
                        0,
                        (components * branches) as usize,
                    )?,
                    (0..components).map(|c| (c * width).into()).collect(),
                )
            }
            Self::Tree { depth } => (tree(depth)?, vec![0_u64.into()]),
        };
        Ok(Dataset {
            case,
            starts,
            shape: self,
        })
    }

    pub(crate) fn depth(self) -> u64 {
        match self {
            Self::Diamonds { depth, .. } => 2 * depth,
            Self::Funnel { tail, .. } => tail + 2,
            Self::Tree { depth } => depth,
        }
    }
}

fn id(value: GraphId<'_>) -> u64 {
    match value {
        GraphId::U64(n) => n,
        GraphId::Str(_) => unreachable!(),
    }
}

// Exhaustive, exact validation without retaining millions of output paths.
// Every valid path has a unique integer identity; a bitset checks full coverage.
pub(crate) struct Validator {
    shape: Shape,
    seen: Vec<u64>,
    score: u64,
}
impl Validator {
    pub(crate) fn new(data: &Dataset, payload: &[u64]) -> Self {
        let mut score = 7_u64;
        for step in 0..data.shape.depth() {
            score = score.wrapping_add(step + 1);
            if !payload.is_empty() {
                score = fold(score, payload);
            }
        }
        Self {
            shape: data.shape,
            seen: vec![0; data.case.expected_paths.div_ceil(64)],
            score,
        }
    }

    pub(crate) fn check(&mut self, path: &GraphPath<'_, State<0>>) {
        assert_eq!(path.edges.len(), self.shape.depth() as usize);
        assert_eq!(path.nodes.len(), path.edges.len() + 1);
        assert!(path.intermediate_states.is_none());
        let mut route = 0;
        let identity = match self.shape {
            Shape::Diamonds {
                components,
                depth,
                bits,
            } => {
                let base = id(path.nodes[0]);
                let component = base / (3 * depth);
                assert!(component < components);
                assert_eq!(base, component * 3 * depth);
                let mut identity = 0;
                for stage in 0..depth {
                    let merge = base + stage * 3;
                    let branch = id(path.nodes[2 * stage as usize + 1]) - merge;
                    assert!(branch == 1 || branch == 2);
                    let next = if stage + 1 == depth {
                        components * 3 * depth + component
                    } else {
                        merge + 3
                    };
                    assert_eq!(id(path.nodes[2 * stage as usize]), merge);
                    assert_eq!(id(path.nodes[2 * stage as usize + 2]), next);
                    let edge = component * 4 * depth + stage * 4;
                    assert_eq!(id(path.edges[2 * stage as usize]), edge + branch - 1);
                    assert_eq!(id(path.edges[2 * stage as usize + 1]), edge + branch + 1);
                    identity = identity * 2 + branch - 1;
                    if stage < bits {
                        route = route * 2 + branch;
                    }
                }
                (component << depth) + identity
            }
            Shape::Funnel {
                components,
                branches,
                tail,
            } => {
                let width = branches + tail + 1;
                let base = id(path.nodes[0]);
                let component = base / width;
                assert!(component < components);
                assert_eq!(base, component * width);
                let branch = id(path.nodes[1]) - base;
                assert!((1..=branches).contains(&branch));
                let merge = base + branches + 1;
                assert_eq!(id(path.nodes[2]), merge);
                let edge_base = component * (2 * branches + tail);
                assert_eq!(id(path.edges[0]), edge_base + 2 * (branch - 1));
                assert_eq!(id(path.edges[1]), edge_base + 2 * (branch - 1) + 1);
                for step in 0..tail {
                    let next = if step + 1 == tail {
                        components * width + component
                    } else {
                        merge + step + 1
                    };
                    assert_eq!(id(path.nodes[step as usize + 3]), next);
                    assert_eq!(
                        id(path.edges[step as usize + 2]),
                        edge_base + 2 * branches + step
                    );
                }
                component * branches + branch - 1
            }
            Shape::Tree { depth } => {
                assert_eq!(id(path.nodes[0]), 0);
                for step in 0..depth as usize {
                    let node = id(path.nodes[step]);
                    let child = id(path.nodes[step + 1]);
                    assert!(child == 2 * node + 1 || child == 2 * node + 2);
                    assert_eq!(id(path.edges[step]), 2 * node + child - (2 * node + 1));
                }
                id(*path.nodes.last().unwrap()) - ((1 << depth) - 1)
            }
        } as usize;
        let bit = 1_u64 << (identity % 64);
        assert_eq!(self.seen[identity / 64] & bit, 0, "duplicate output path");
        self.seen[identity / 64] |= bit;
        assert_eq!(
            path.state,
            State {
                depth: self.shape.depth(),
                route,
                score: self.score,
                carried: []
            }
        );
    }
}

struct Summary {
    paths: usize,
    stats: SearchStats,
    calls: usize,
    entries: usize,
    capacity: usize,
}

fn execute_large<const COUNT: bool>(
    data: &Dataset,
    payload: &[u64],
    parallel: bool,
    eager: bool,
    cached: bool,
) -> Result<Summary> {
    let calls = AtomicUsize::new(0);
    let kernel = WorkKernel::<0, COUNT> {
        case: &data.case,
        payload,
        calls: &calls,
        reject_depth: None,
    };
    let run = RunOptions {
        start_nodes: data.starts.clone(),
        strategy: TraversalStrategy::DepthFirst,
        parallel,
        ..Default::default()
    };
    let mut validator = COUNT.then(|| Validator::new(data, payload));
    let mut paths = 0;
    let mut consume = |batch: &[GraphPath<'_, State<0>>]| {
        paths += batch.len();
        black_box(batch);
        if let Some(validator) = &mut validator {
            for path in batch {
                validator.check(path);
            }
        }
    };
    let generation = if cached {
        NEXT_GENERATION.fetch_add(1, Ordering::Relaxed)
    } else {
        0
    };
    macro_rules! search {
        ($kernel:expr) => {{
            if eager {
                let result = data.case.graph.search_with($kernel, run)?;
                consume(&result.paths);
                result.stats
            } else {
                let mut stream = data
                    .case
                    .graph
                    .search_batches_with($kernel, run, BATCH_SIZE)?;
                for batch in stream.by_ref() {
                    consume(&batch?);
                }
                stream.stats()
            }
        }};
    }
    let stats = if cached {
        search!(WorkerMemo {
            inner: kernel,
            generation
        })
    } else {
        search!(kernel)
    };
    let (entries, capacity) = if cached {
        clear_workers::<0, COUNT>(generation)
    } else {
        (0, 0)
    };
    assert_eq!(paths, data.case.expected_paths);
    Ok(Summary {
        paths,
        stats,
        calls: if COUNT {
            calls.load(Ordering::Relaxed)
        } else {
            0
        },
        entries,
        capacity,
    })
}

fn verify_large() -> Result<()> {
    for shape in [
        Shape::Diamonds {
            components: 4,
            depth: 5,
            bits: 0,
        },
        Shape::Diamonds {
            components: 4,
            depth: 5,
            bits: 3,
        },
        Shape::Diamonds {
            components: 4,
            depth: 5,
            bits: 5,
        },
        Shape::Funnel {
            components: 4,
            branches: 8,
            tail: 5,
        },
        Shape::Tree { depth: 6 },
    ] {
        let data = shape.build()?;
        for eager in [false, true] {
            for parallel in [false, true] {
                let a = execute_large::<true>(&data, &[1, 2, 3], parallel, eager, false)?;
                let b = execute_large::<true>(&data, &[1, 2, 3], parallel, eager, true)?;
                assert_eq!(a.stats, b.stats);
                assert_eq!(a.calls, a.stats.evaluated_edges);
                assert_eq!(b.calls, b.entries);
            }
        }
    }
    eprintln!(
        "scale correctness: every path, edge, state and unique path identity checked exactly"
    );
    Ok(())
}

fn measure(
    data: &Dataset,
    payload: &[u64],
    parallel: bool,
    eager: bool,
    cached: bool,
    minimum: f64,
) -> Result<(f64, usize)> {
    let start = Instant::now();
    let mut searches = 0;
    loop {
        black_box(execute_large::<false>(
            data, payload, parallel, eager, cached,
        )?);
        searches += 1;
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= minimum {
            return Ok((elapsed, searches));
        }
    }
}

struct Job {
    name: &'static str,
    shape: Shape,
    work: usize,
    parallel: bool,
    eager: bool,
}

fn benchmark(job: &Job, samples: usize, minimum: f64) -> Result<Json> {
    let data = job.shape.build()?;
    let payload = (0..job.work)
        .map(|i| (i as u64 + 1).wrapping_mul(0xd6e8_feb8_6659_fd93))
        .collect::<Vec<_>>();
    eprintln!(
        "{}: {} nodes, {} edges, {} paths; validating",
        job.name,
        data.case.graph.node_count(),
        data.case.graph.edge_count(),
        data.case.expected_paths
    );
    let a = execute_large::<true>(&data, &payload, job.parallel, job.eager, false)?;
    let b = execute_large::<true>(&data, &payload, job.parallel, job.eager, true)?;
    assert_eq!(a.stats, b.stats);
    assert_eq!(a.calls, a.stats.evaluated_edges);
    assert_eq!(b.calls, b.entries);
    if job.parallel {
        assert!(a.stats.parallel_edges > 0);
    }
    for cached in [false, true] {
        measure(&data, &payload, job.parallel, job.eager, cached, minimum)?;
    }
    let mut seconds = [Vec::new(), Vec::new()];
    let mut counts = [Vec::new(), Vec::new()];
    for sample in 0..samples {
        for mode in [sample % 2, 1 - sample % 2] {
            let (elapsed, count) =
                measure(&data, &payload, job.parallel, job.eager, mode == 1, minimum)?;
            seconds[mode].push(elapsed);
            counts[mode].push(count);
        }
        eprintln!(
            "{} sample {}/{}: {:.3}s -> {:.3}s per search",
            job.name,
            sample + 1,
            samples,
            seconds[0][sample] / counts[0][sample] as f64,
            seconds[1][sample] / counts[1][sample] as f64
        );
    }
    let normalized: [Vec<f64>; 2] = std::array::from_fn(|mode| {
        seconds[mode]
            .iter()
            .zip(&counts[mode])
            .map(|(&s, &n)| s / n as f64)
            .collect()
    });
    let baseline = median(&normalized[0]);
    let cached = median(&normalized[1]);
    eprintln!(
        "{}: {:.3}s -> {:.3}s ({:.2}x), {} -> {} computed transitions",
        job.name,
        baseline,
        cached,
        baseline / cached,
        a.calls,
        b.calls
    );
    Ok(json!({
        "name": job.name, "graph_nodes": data.case.graph.node_count(), "graph_edges": data.case.graph.edge_count(),
        "paths": a.paths, "path_depth": data.shape.depth(), "evaluated_edges": a.calls,
        "computed_transitions_cached": b.calls, "cache_hits": a.calls - b.calls,
        "hit_rate": 1.0 - b.calls as f64 / a.calls as f64,
        "cache_entries": b.entries, "cache_capacity_entries": b.capacity,
        "live_entry_payload_bytes": b.entries * size_of::<((EdgeId, State<0>), Transition<State<0>>)>(),
        "work_units": job.work, "state_bytes": size_of::<State<0>>(), "parallel": job.parallel,
        "parallel_edges": a.stats.parallel_edges, "output": if job.eager {"eager"} else {"streaming"},
        "batch_size": if job.eager {0} else {BATCH_SIZE}, "cache_mode": "worker-local",
        "baseline_median_seconds": baseline, "cached_median_seconds": cached, "speedup": baseline / cached,
        "samples_seconds": {"baseline": normalized[0], "cached": normalized[1]},
        "measurement_window_seconds": {"baseline": seconds[0], "cached": seconds[1]},
        "searches_per_window": {"baseline": counts[0], "cached": counts[1]},
        "exact_output_validation": true,
    }))
}

pub(super) fn run() -> Result<()> {
    anyhow::ensure!(
        !cfg!(debug_assertions),
        "scale benchmarks require --release"
    );
    let samples = std::env::var("RX_RECONV_SAMPLES")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(7);
    let minimum = std::env::var("RX_RECONV_SAMPLE_SECONDS")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(2.0);
    anyhow::ensure!(
        samples >= 3 && minimum >= 1.0,
        "use >=3 samples and >=1 second per window"
    );
    let filter = std::env::var("RX_RECONV_FILTER").unwrap_or_default();
    let many = Shape::Diamonds {
        components: 1024,
        depth: 12,
        bits: 0,
    };
    let funnel = Shape::Funnel {
        components: 4,
        branches: 65536,
        tail: 64,
    };
    let jobs = [
        Job {
            name: "scale/diamonds-many/cheap",
            shape: many,
            work: 0,
            parallel: false,
            eager: false,
        },
        Job {
            name: "scale/diamonds-many/cpu256",
            shape: many,
            work: 256,
            parallel: false,
            eager: false,
        },
        Job {
            name: "scale/diamonds-many/parallel-cpu256",
            shape: many,
            work: 256,
            parallel: true,
            eager: false,
        },
        Job {
            name: "scale/diamonds-partial/cpu256",
            shape: Shape::Diamonds {
                components: 64,
                depth: 14,
                bits: 11,
            },
            work: 256,
            parallel: false,
            eager: false,
        },
        Job {
            name: "scale/diamonds-unique/cpu256",
            shape: Shape::Diamonds {
                components: 64,
                depth: 14,
                bits: 14,
            },
            work: 256,
            parallel: false,
            eager: false,
        },
        Job {
            name: "scale/funnel/cpu256",
            shape: funnel,
            work: 256,
            parallel: false,
            eager: false,
        },
        Job {
            name: "scale/funnel/parallel-cpu256",
            shape: funnel,
            work: 256,
            parallel: true,
            eager: false,
        },
        Job {
            name: "scale/tree/cpu256",
            shape: Shape::Tree { depth: 20 },
            work: 256,
            parallel: false,
            eager: false,
        },
        Job {
            name: "scale/eager-diamonds/cpu256",
            shape: Shape::Diamonds {
                components: 256,
                depth: 12,
                bits: 0,
            },
            work: 256,
            parallel: false,
            eager: true,
        },
    ];
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let rows = pool.install(|| -> Result<Vec<Json>> {
        verify_large()?;
        let mut rows = Vec::new();
        for job in jobs.iter().filter(|j| j.name.contains(&filter)) {
            rows.push(benchmark(job, samples, minimum)?);
        }
        Ok(rows)
    })?;
    anyhow::ensure!(!rows.is_empty(), "filter matched no cases");
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "rxgraph_version": env!("CARGO_PKG_VERSION"), "arch": std::env::consts::ARCH,
            "os": std::env::consts::OS, "rayon_threads": 4, "samples": samples,
            "minimum_measurement_window_seconds": minimum,
            "method": "release; cold per-search worker caches; exact exhaustive output audits outside timing; no timed kernel counters; full path materialization and teardown; >=minimum-second windows of independent searches; alternating arm order; synthetic kernels",
            "memory_note": "live_entry_payload_bytes is a lower bound, excluding empty buckets and allocator overhead",
            "cases": rows,
        }))?
    );
    Ok(())
}
