//! Measurements for the native per-search cache. Set RX_NATIVE_CASE and RX_NATIVE_MODE.

#[path = "reconvergence_bench.rs"]
#[allow(dead_code)]
mod fixture;
#[path = "reconvergence_bench/heap.rs"]
#[allow(dead_code)]
mod heap;

use anyhow::Result;
use fixture::{
    State, WorkKernel,
    scale::{Dataset, Shape, Validator},
};
use heap::{Buffer, HeapKernel, HeapState};
use rxgraph::{GraphPath, MemoizableKernel, RunOptions, SearchStats, TargetOptions, TraversalStrategy};
use serde_json::json;
use std::{
    hint::black_box,
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

impl<const COUNT: bool> MemoizableKernel for WorkKernel<'_, 0, COUNT> {
    type Key = State<0>;
    fn state_key(&self, state: &Self::State) -> Self::Key {
        *state
    }
}

// This fixture's buffer is fully determined by its metadata and fixed constants.
impl<B: Buffer, const COUNT: bool> MemoizableKernel for HeapKernel<'_, B, COUNT> {
    type Key = State<0>;
    fn state_key(&self, state: &Self::State) -> Self::Key {
        state.meta
    }
}

trait Inspect {
    fn meta(&self) -> State<0>;
    fn validate(&self, words: usize);
}
impl Inspect for State<0> {
    fn meta(&self) -> State<0> {
        *self
    }
    fn validate(&self, _: usize) {}
}
impl<B: Buffer> Inspect for HeapState<B> {
    fn meta(&self) -> State<0> {
        self.meta
    }
    fn validate(&self, words: usize) {
        let values = self.buffer.values();
        assert_eq!(values.len(), words);
        assert_eq!(values[0], self.meta.score);
        for (i, &value) in values.iter().enumerate().skip(1) {
            assert_eq!(value, i as u64 + 19);
        }
    }
}

#[derive(Clone, Copy)]
struct Spec {
    shape: Shape,
    words: usize,
    cpu: usize,
    bfs: bool,
    parallel: bool,
    eager: bool,
}

fn consume<S: Inspect>(
    paths: Vec<GraphPath<'_, S>>,
    validator: &mut Option<Validator>,
    words: usize,
    count: &mut usize,
) {
    *count += paths.len();
    black_box(&paths);
    if let Some(validator) = validator {
        for path in paths {
            path.state.validate(words);
            validator.check(&GraphPath {
                nodes: path.nodes,
                edges: path.edges,
                state: path.state.meta(),
                intermediate_states: path
                    .intermediate_states
                    .map(|states| states.iter().map(Inspect::meta).collect()),
            });
        }
    }
}

fn execute<K>(
    data: &Dataset,
    kernel: K,
    spec: Spec,
    arm: usize,
    audit: bool,
    payload: &[u64],
) -> Result<SearchStats>
where
    K: MemoizableKernel + Sync,
    K::State: Inspect + Send + Sync,
{
    let mut validator = audit.then(|| Validator::new(data, payload));
    let run = RunOptions {
        start_nodes: data.starts.clone(),
        strategy: if spec.bfs {
            TraversalStrategy::BreadthFirst
        } else {
            TraversalStrategy::DepthFirst
        },
        parallel: spec.parallel,
        ..Default::default()
    };
    let mut count = 0;
    macro_rules! consume_stream {
        ($stream:expr) => {{
            let mut stream = $stream?;
            for batch in stream.by_ref() {
                consume(batch?, &mut validator, spec.words, &mut count);
            }
            stream.stats()
        }};
    }
    let stats = if arm >= 3 {
        consume_stream!(data.case.graph.search_batches_with(
            kernel,
            run.with_optimizations(TargetOptions {
                release_expanded: arm == 4,
                ..Default::default()
            }),
            1024,
        ))
    } else if spec.eager {
        let result = if arm != 0 {
            data.case.graph.search_paths_memoized_with(
                kernel,
                run,
                if arm == 1 { 0 } else { 4096 },
            )?
        } else {
            data.case.graph.search_paths_with(kernel, run)?
        };
        let stats = result.stats;
        consume(result.paths, &mut validator, spec.words, &mut count);
        stats
    } else if arm != 0 {
        consume_stream!(data.case.graph.search_batches_memoized_with(
            kernel,
            run,
            1024,
            if arm == 1 { 0 } else { 4096 }
        ))
    } else {
        consume_stream!(data.case.graph.search_batches_with(kernel, run, 1024))
    };
    assert_eq!(count, data.case.expected_paths);
    if spec.parallel {
        assert!(stats.parallel_edges > 0);
    }
    Ok(stats)
}

fn run<const COUNT: bool>(
    data: &Dataset,
    spec: Spec,
    arm: usize,
    audit: bool,
    calls: &AtomicUsize,
    payload: &[u64],
) -> Result<SearchStats> {
    calls.store(0, Ordering::Relaxed);
    let inner = WorkKernel::<0, COUNT> {
        case: &data.case,
        payload,
        calls,
        reject_depth: None,
    };
    if spec.words == 0 {
        execute(data, inner, spec, arm, audit, payload)
    } else {
        execute(
            data,
            HeapKernel::<Vec<u64>, COUNT> {
                inner,
                words: spec.words,
                marker: PhantomData,
            },
            spec,
            arm,
            audit,
            payload,
        )
    }
}

fn window(
    data: &Dataset,
    spec: Spec,
    arm: usize,
    minimum: f64,
    calls: &AtomicUsize,
    payload: &[u64],
) -> Result<(f64, usize)> {
    let start = Instant::now();
    let mut count = 0;
    loop {
        black_box(run::<false>(data, spec, arm, false, calls, payload)?);
        count += 1;
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= minimum {
            return Ok((elapsed, count));
        }
    }
}

fn main() -> Result<()> {
    anyhow::ensure!(!cfg!(debug_assertions), "run with --release");
    let name = std::env::var("RX_NATIVE_CASE").unwrap_or_else(|_| "vec-bfs".into());
    let mode = std::env::var("RX_NATIVE_MODE").unwrap_or_else(|_| "timing".into());
    let spec = match name.as_str() {
        "small-bfs" | "small-bfs-parallel" => Spec {
            shape: Shape::Diamonds {
                components: 16,
                depth: 12,
                bits: 0,
            },
            words: 0,
            cpu: 0,
            bfs: true,
            parallel: name == "small-bfs-parallel",
            eager: false,
        },
        "vec-bfs" => Spec {
            shape: Shape::Diamonds {
                components: 16,
                depth: 12,
                bits: 0,
            },
            words: 512,
            cpu: 0,
            bfs: true,
            parallel: false,
            eager: false,
        },
        "vec-bfs-parallel" => Spec {
            shape: Shape::Diamonds {
                components: 16,
                depth: 12,
                bits: 0,
            },
            words: 512,
            cpu: 0,
            bfs: true,
            parallel: true,
            eager: false,
        },
        "vec-dfs" => Spec {
            shape: Shape::Diamonds {
                components: 64,
                depth: 12,
                bits: 0,
            },
            words: 512,
            cpu: 0,
            bfs: false,
            parallel: false,
            eager: false,
        },
        "vec-dfs-parallel" => Spec {
            shape: Shape::Diamonds {
                components: 64,
                depth: 12,
                bits: 0,
            },
            words: 512,
            cpu: 0,
            bfs: false,
            parallel: true,
            eager: false,
        },
        "vec-eager" => Spec {
            shape: Shape::Diamonds {
                components: 32,
                depth: 12,
                bits: 0,
            },
            words: 512,
            cpu: 0,
            bfs: false,
            parallel: false,
            eager: true,
        },
        "vec-unique" => Spec {
            shape: Shape::Diamonds {
                components: 1,
                depth: 16,
                bits: 16,
            },
            words: 512,
            cpu: 0,
            bfs: false,
            parallel: false,
            eager: false,
        },
        "cpu-dfs" => Spec {
            shape: Shape::Diamonds {
                components: 16,
                depth: 14,
                bits: 0,
            },
            words: 0,
            cpu: 512,
            bfs: false,
            parallel: false,
            eager: false,
        },
        "cpu-bfs" => Spec {
            shape: Shape::Diamonds {
                components: 4,
                depth: 14,
                bits: 0,
            },
            words: 0,
            cpu: 512,
            bfs: true,
            parallel: false,
            eager: false,
        },
        _ => anyhow::bail!("unknown case {name}"),
    };
    let data = spec.shape.build()?;
    let payload = (0..spec.cpu)
        .map(|i| (i as u64).wrapping_mul(17).wrapping_add(1))
        .collect::<Vec<_>>();
    let calls = AtomicUsize::new(0);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let result = pool.install(|| -> Result<_> {
        let arm: usize = std::env::var("RX_NATIVE_ARM").unwrap_or_else(|_| "0".into()).parse()?;
        anyhow::ensure!(arm <= 4, "unknown arm");
        if mode == "rss" {
            let stats = run::<true>(&data, spec, arm, true, &calls, &payload)?;
            let computed = calls.load(Ordering::Relaxed);
            Ok(json!({"case": name, "arm": arm, "paths": data.case.expected_paths, "evaluated_edges": stats.evaluated_edges, "computed": computed, "hits": stats.evaluated_edges - computed, "parallel_edges": stats.parallel_edges, "exact_output_validation": true}))
        } else {
            let seconds: f64 = std::env::var("RX_NATIVE_SECONDS").ok().map(|s| s.parse()).transpose()?.unwrap_or(2.0);
            anyhow::ensure!(seconds >= 1.0, "need at least a one-second window");
            window(&data, spec, arm, 0.5, &calls, &payload)?;
            let (duration, count) = window(&data, spec, arm, seconds, &calls, &payload)?;
            Ok(json!({"case":name, "arm":arm, "paths":data.case.expected_paths, "buffer_bytes":spec.words * 8, "parallel":spec.parallel, "eager":spec.eager, "seconds_per_search": duration / count as f64, "window_seconds":duration, "searches_per_window":count, "method":"uninstrumented release, isolated process, fresh search cache, 4-thread Rayon pool"}))
        }
    })?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
