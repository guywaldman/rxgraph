//! End-to-end experiment; does not modify the traversal engine.
//! cargo run --release -p rxgraph --example reconvergence_bench --offline > results.json
//! Optional: RX_RECONV_FILTER=<case substring>, RX_RECONV_SAMPLES=9,
//! RX_RECONV_CACHE=worker (default: shared).
//! RX_RECONV_SCALE=1 selects the million-path suite (7 samples, >=2s per window).
//! Each timed search starts with an empty cache and materializes/drops all paths.

#[path = "reconvergence_bench/scale.rs"]
pub(crate) mod scale;

use std::{
    any::Any,
    cell::RefCell,
    collections::HashMap,
    hint::black_box,
    mem::size_of,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Instant,
};

use anyhow::Result;
use arrow::array::{ArrayRef, BooleanArray, RecordBatch, UInt64Array};
use rxgraph::{
    EdgeCtx, EdgeField, EdgeId, Graph, GraphId, GraphPath, Kernel, NodeId, RunOptions, SearchStats,
    Transition, TraversalStrategy,
};
use serde_json::{Value as Json, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct State<const WORDS: usize> {
    pub(crate) depth: u64,
    pub(crate) route: u64,
    pub(crate) score: u64,
    pub(crate) carried: [u64; WORDS],
}

type Map<const W: usize> = HashMap<(EdgeId, State<W>), Transition<State<W>>>;
type Cache<const W: usize> = Mutex<Map<W>>;
type Row<const W: usize> = (Vec<u64>, Vec<u64>, State<W>, Option<Vec<State<W>>>);

#[derive(Clone, Copy, PartialEq, Eq)]
enum CacheMode {
    Disabled,
    Shared,
    Worker,
}

struct WorkerCache {
    generation: u64,
    map: Box<dyn Any>,
}
thread_local! {
    static WORKER_CACHE: RefCell<Option<WorkerCache>> = const { RefCell::new(None) };
}
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

pub(crate) struct Case {
    pub(crate) graph: Graph,
    pub(crate) step: EdgeField<u64>,
    pub(crate) choice: EdgeField<u64>,
    pub(crate) terminal: EdgeField<bool>,
    pub(crate) route_bits: u64,
    pub(crate) expected_paths: usize,
}

// (source, destination, step, branch choice). Payloads at equal depths have equal cost.
fn case(
    nodes: u64,
    terminal_from: u64,
    edges: &[(u64, u64, u64, u64)],
    route_bits: u64,
    paths: usize,
) -> Result<Case> {
    let column = |v: Vec<u64>| Arc::new(UInt64Array::from(v)) as ArrayRef;
    let graph = Graph::new(
        RecordBatch::try_from_iter([("id", column((0..nodes).collect()))])?,
        RecordBatch::try_from_iter([
            ("id", column((0..edges.len() as u64).collect())),
            ("src", column(edges.iter().map(|e| e.0).collect())),
            ("dest", column(edges.iter().map(|e| e.1).collect())),
            ("step", column(edges.iter().map(|e| e.2).collect())),
            ("choice", column(edges.iter().map(|e| e.3).collect())),
            (
                "terminal",
                Arc::new(BooleanArray::from(
                    edges
                        .iter()
                        .map(|e| e.1 >= terminal_from)
                        .collect::<Vec<_>>(),
                )),
            ),
        ])?,
    )?;
    Ok(Case {
        step: graph.edge_field("step")?,
        choice: graph.edge_field("choice")?,
        terminal: graph.edge_field("terminal")?,
        graph,
        route_bits,
        expected_paths: paths,
    })
}

fn diamonds(depth: u64, route_bits: u64) -> Result<Case> {
    let mut edges = Vec::new();
    for i in 0..depth {
        let merge = 3 * i;
        edges.extend([
            (merge, merge + 1, 2 * i, 1),
            (merge, merge + 2, 2 * i, 2),
            (merge + 1, merge + 3, 2 * i + 1, 0),
            (merge + 2, merge + 3, 2 * i + 1, 0),
        ]);
    }
    case(3 * depth + 1, 3 * depth, &edges, route_bits, 1 << depth)
}

fn funnel(branches: u64, tail: u64) -> Result<Case> {
    let merge = branches + 1;
    let mut edges = Vec::new();
    for i in 1..=branches {
        edges.extend([(0, i, 0, 0), (i, merge, 1, 0)]);
    }
    edges.extend((0..tail).map(|i| (merge + i, merge + i + 1, i + 2, 0)));
    case(merge + tail + 1, merge + tail, &edges, 0, branches as usize)
}

fn tree(depth: u64) -> Result<Case> {
    let mut edges = Vec::new();
    for step in 0..depth {
        for node in ((1 << step) - 1)..((1 << (step + 1)) - 1) {
            edges.extend([(node, 2 * node + 1, step, 1), (node, 2 * node + 2, step, 2)]);
        }
    }
    case(
        (1 << (depth + 1)) - 1,
        (1 << depth) - 1,
        &edges,
        0,
        1 << depth,
    )
}

// Synthetic state-dependent CPU work. Its result is returned in every path's state.
// No black_box or counters in this loop; the dependency chain cannot be eliminated.
#[inline(never)]
fn fold(mut score: u64, payload: &[u64]) -> u64 {
    for &value in payload {
        score = score
            .rotate_left(9)
            .wrapping_add(value)
            .wrapping_mul(0x9e37_79b1_85eb_ca87);
    }
    score
}

pub(crate) struct WorkKernel<'a, const W: usize, const COUNT: bool> {
    pub(crate) case: &'a Case,
    pub(crate) payload: &'a [u64],
    pub(crate) calls: &'a AtomicUsize,
    pub(crate) reject_depth: Option<u64>,
}

impl<const W: usize, const COUNT: bool> Kernel for WorkKernel<'_, W, COUNT> {
    type State = State<W>;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<Self::State> {
        Ok(State {
            depth: 0,
            route: 0,
            score: 7,
            carried: [19; W],
        })
    }
    fn transition(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Transition<Self::State>> {
        if COUNT {
            self.calls.fetch_add(1, Ordering::Relaxed);
        }
        let step = cx.edge_field(&self.case.step).unwrap();
        let choice = cx.edge_field(&self.case.choice).unwrap();
        let mut state = *cx.state();
        state.depth += 1;
        if step < self.case.route_bits * 2 && choice != 0 {
            state.route = state.route * 2 + choice;
        }
        state.score = state.score.wrapping_add(step + 1);
        if !self.payload.is_empty() {
            state.score = fold(state.score, self.payload);
        }
        Ok(if self.reject_depth == Some(state.depth) {
            Transition::Reject
        } else if cx.edge_field(&self.case.terminal).unwrap() {
            Transition::Complete(state)
        } else {
            Transition::Continue(state)
        })
    }
}

// Valid only for a deterministic kernel, one fixed graph/parameter set, and one search.
// Cache equality compares the entire state, not just a hash or the destination node.
struct Memo<'a, const W: usize, const COUNT: bool> {
    inner: WorkKernel<'a, W, COUNT>,
    shards: &'a [Cache<W>],
}
impl<const W: usize, const COUNT: bool> Kernel for Memo<'_, W, COUNT> {
    type State = State<W>;
    fn initial_state(&self, graph: &Graph, start: NodeId) -> Result<Self::State> {
        self.inner.initial_state(graph, start)
    }
    fn transition(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Transition<Self::State>> {
        let key = (cx.edge(), *cx.state());
        let mut cache = self.shards[cx.edge() as usize % self.shards.len()]
            .lock()
            .unwrap();
        if let Some(transition) = cache.get(&key) {
            return Ok(transition.clone());
        }
        let transition = self.inner.transition(cx)?;
        cache.insert(key, transition.clone());
        Ok(transition)
    }
}

// A lock-free cache per worker, identified by a unique search generation.
// Cleanup below runs on every pool worker before the timed search returns.
struct WorkerMemo<'a, const W: usize, const COUNT: bool> {
    inner: WorkKernel<'a, W, COUNT>,
    generation: u64,
}
impl<const W: usize, const COUNT: bool> Kernel for WorkerMemo<'_, W, COUNT> {
    type State = State<W>;
    fn initial_state(&self, graph: &Graph, start: NodeId) -> Result<Self::State> {
        self.inner.initial_state(graph, start)
    }
    fn transition(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Transition<Self::State>> {
        WORKER_CACHE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot
                .as_ref()
                .is_none_or(|c| c.generation != self.generation)
            {
                *slot = Some(WorkerCache {
                    generation: self.generation,
                    map: Box::new(Map::<W>::new()),
                });
            }
            let cache = slot.as_mut().unwrap().map.downcast_mut::<Map<W>>().unwrap();
            let key = (cx.edge(), *cx.state());
            if let Some(transition) = cache.get(&key) {
                return Ok(transition.clone());
            }
            let transition = self.inner.transition(cx)?;
            cache.insert(key, transition.clone());
            Ok(transition)
        })
    }
}

fn clear_workers<const W: usize, const COUNT: bool>(generation: u64) -> (usize, usize) {
    rayon::broadcast(|_| {
        WORKER_CACHE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.as_ref().is_none_or(|c| c.generation != generation) {
                return (0, 0);
            }
            let cache = slot.take().unwrap();
            if COUNT {
                let map = cache.map.downcast_ref::<Map<W>>().unwrap();
                (map.len(), map.capacity())
            } else {
                (0, 0)
            }
        })
    })
    .into_iter()
    .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
}

struct Audit<const W: usize> {
    rows: Vec<Row<W>>,
    stats: SearchStats,
    calls: usize,
    entries: usize,
    capacity: usize,
}

fn normalize<const W: usize>(paths: Vec<GraphPath<'_, State<W>>>) -> Vec<Row<W>> {
    let id = |v| match v {
        GraphId::U64(n) => n,
        GraphId::Str(_) => unreachable!(),
    };
    let mut rows = paths
        .into_iter()
        .map(|p| {
            (
                p.nodes.into_iter().map(id).collect(),
                p.edges.into_iter().map(id).collect(),
                p.state,
                p.intermediate_states,
            )
        })
        .collect::<Vec<Row<W>>>();
    rows.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    rows
}

fn execute<const W: usize, const COUNT: bool>(
    case: &Case,
    payload: &[u64],
    run: &RunOptions,
    mode: CacheMode,
    stream: bool,
    reject_depth: Option<u64>,
) -> Result<Audit<W>> {
    let calls = AtomicUsize::new(0);
    let kernel = WorkKernel::<W, COUNT> {
        case,
        payload,
        calls: &calls,
        reject_depth,
    };
    let shards = (0..if mode == CacheMode::Shared {
        if run.parallel { 64 } else { 1 }
    } else {
        0
    })
        .map(|_| Mutex::new(HashMap::new()))
        .collect::<Vec<_>>();
    macro_rules! search {
        ($kernel:expr) => {{
            if stream {
                let mut cursor = case.graph.search_batches_with($kernel, run.clone(), 1)?;
                let mut paths = Vec::new();
                for batch in cursor.by_ref() {
                    paths.extend(batch?);
                }
                (paths, cursor.stats())
            } else {
                let result = case.graph.search_with($kernel, run.clone())?;
                (result.paths, result.stats)
            }
        }};
    }
    let generation = if mode == CacheMode::Worker {
        NEXT_GENERATION.fetch_add(1, Ordering::Relaxed)
    } else {
        0
    };
    let (paths, stats) = match mode {
        CacheMode::Shared => search!(Memo {
            inner: kernel,
            shards: &shards
        }),
        CacheMode::Worker => search!(WorkerMemo {
            inner: kernel,
            generation
        }),
        CacheMode::Disabled => search!(kernel),
    };
    black_box(&paths);
    let mut audit = Audit {
        rows: Vec::new(),
        stats,
        calls: 0,
        entries: 0,
        capacity: 0,
    };
    if mode == CacheMode::Worker {
        (audit.entries, audit.capacity) = clear_workers::<W, COUNT>(generation);
    }
    if COUNT {
        audit.rows = normalize(paths);
        audit.calls = calls.load(Ordering::Relaxed);
        for shard in &shards {
            let cache = shard.lock().unwrap();
            audit.entries += cache.len();
            audit.capacity += cache.capacity();
        }
    }
    // All paths and the cold cache are dropped before a timed invocation returns.
    Ok(audit)
}

fn options(strategy: TraversalStrategy, parallel: bool, limit: Option<usize>) -> RunOptions {
    RunOptions {
        start_nodes: vec![0_u64.into()],
        strategy,
        parallel,
        max_paths: limit,
        ..Default::default()
    }
}

fn verify() -> Result<()> {
    let cyclic = case(
        5,
        4,
        &[
            (0, 1, 0, 0),
            (0, 2, 0, 0),
            (1, 3, 1, 0),
            (2, 3, 1, 0),
            (3, 1, 2, 0),
            (1, 4, 3, 0),
        ],
        0,
        0,
    )?;
    for graph in [&cyclic, &diamonds(5, 0)?, &diamonds(5, 5)?] {
        for strategy in [
            TraversalStrategy::DepthFirst,
            TraversalStrategy::BreadthFirst,
        ] {
            for parallel in [false, true] {
                for stream in [false, true] {
                    for visits in [1, 2] {
                        for rejection in [None, Some(4)] {
                            let mut run = options(strategy, parallel, None);
                            run.max_depth = Some(12);
                            run.max_visits_per_node = visits;
                            run.intermediate_states = true;
                            let a = execute::<0, true>(
                                graph,
                                &[1, 2, 3],
                                &run,
                                CacheMode::Disabled,
                                stream,
                                rejection,
                            )?;
                            for mode in [CacheMode::Shared, CacheMode::Worker] {
                                let b = execute::<0, true>(
                                    graph,
                                    &[1, 2, 3],
                                    &run,
                                    mode,
                                    stream,
                                    rejection,
                                )?;
                                assert_eq!(a.rows, b.rows);
                                assert_eq!(a.stats, b.stats);
                                assert_eq!(a.calls, a.stats.evaluated_edges);
                                assert_eq!(b.calls, b.entries);
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "correctness: paths, edges, states, history, revisits, rejects, DFS/BFS and streaming agree"
    );
    Ok(())
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn bench<const W: usize>(
    name: &str,
    case: &Case,
    work: usize,
    run: RunOptions,
    samples: usize,
    mode: CacheMode,
) -> Result<Json> {
    let payload = (0..work)
        .map(|i| (i as u64 + 1).wrapping_mul(0xd6e8_feb8_6659_fd93))
        .collect::<Vec<_>>();
    let baseline = execute::<W, true>(case, &payload, &run, CacheMode::Disabled, false, None)?;
    let cached = execute::<W, true>(case, &payload, &run, mode, false, None)?;
    assert_eq!(baseline.rows, cached.rows);
    assert_eq!(baseline.stats, cached.stats);
    assert_eq!(
        baseline.rows.len(),
        case.expected_paths.min(run.max_paths.unwrap_or(usize::MAX))
    );
    assert_eq!(baseline.calls, baseline.stats.evaluated_edges);
    assert_eq!(cached.calls, cached.entries);
    if run.parallel {
        assert!(baseline.stats.parallel_edges > 0);
    }

    let mut iterations = [1_usize; 2];
    for (index, count) in iterations.iter_mut().enumerate() {
        for _ in 0..2 {
            let start = Instant::now();
            execute::<W, false>(
                case,
                &payload,
                &run,
                if index == 1 {
                    mode
                } else {
                    CacheMode::Disabled
                },
                false,
                None,
            )?;
            *count = (0.015 / start.elapsed().as_secs_f64())
                .ceil()
                .clamp(1.0, 512.0) as usize;
        }
    }
    let mut times = [Vec::new(), Vec::new()];
    for sample in 0..samples {
        // Alternate baseline/cache ordering to reduce clock/thermal bias.
        for index in [sample % 2, 1 - sample % 2] {
            let start = Instant::now();
            for _ in 0..iterations[index] {
                black_box(execute::<W, false>(
                    case,
                    &payload,
                    &run,
                    if index == 1 {
                        mode
                    } else {
                        CacheMode::Disabled
                    },
                    false,
                    None,
                )?);
            }
            times[index].push(start.elapsed().as_secs_f64() / iterations[index] as f64);
        }
    }
    let a = median(&times[0]);
    let b = median(&times[1]);
    let hit_rate = 1.0 - cached.calls as f64 / baseline.calls as f64;
    eprintln!(
        "{name}: {:.3} -> {:.3} ms ({:.2}x), {:.1}% hits",
        a * 1e3,
        b * 1e3,
        a / b,
        hit_rate * 100.0
    );
    Ok(json!({
        "name": name, "work_units": work, "state_bytes": size_of::<State<W>>(),
        "paths": baseline.rows.len(), "evaluated_edges": baseline.calls,
        "computed_transitions_cached": cached.calls, "cache_hits": baseline.calls - cached.calls,
        "hit_rate": hit_rate, "cache_entries": cached.entries, "cache_capacity_entries": cached.capacity,
        "live_entry_payload_bytes": cached.entries * size_of::<((EdgeId, State<W>), Transition<State<W>>)>(),
        "parallel_edges": baseline.stats.parallel_edges,
        "cache_mode": if mode == CacheMode::Worker {"worker-local"} else {"shared"},
        "cache_shards": if mode == CacheMode::Worker {0} else if run.parallel {64} else {1},
        "baseline_median_seconds": a, "cached_median_seconds": b, "speedup": a / b,
        "samples_seconds": {"baseline": times[0], "cached": times[1]},
        "iterations_per_sample": iterations,
    }))
}

fn main() -> Result<()> {
    if std::env::var("RX_RECONV_SCALE").as_deref() == Ok("1") {
        return scale::run();
    }
    let samples = std::env::var("RX_RECONV_SAMPLES")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(9);
    anyhow::ensure!(samples >= 3, "at least three samples required");
    let filter = std::env::var("RX_RECONV_FILTER").unwrap_or_default();
    let mode = match std::env::var("RX_RECONV_CACHE")
        .as_deref()
        .unwrap_or("shared")
    {
        "shared" => CacheMode::Shared,
        "worker" => CacheMode::Worker,
        value => anyhow::bail!("unknown cache mode {value}"),
    };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let rows = pool.install(|| -> Result<Vec<Json>> {
        verify()?;
        let cases = [
            ("diamonds-shared", diamonds(12, 0)?),
            ("diamonds-512-states", diamonds(12, 9)?),
            ("diamonds-unique", diamonds(12, 12)?),
            ("tree-no-reconvergence", tree(12)?),
            ("funnel-1024-tail32", funnel(1024, 32)?),
        ];
        let mut rows = Vec::new();
        for (label, case) in &cases {
            for work in [0, 16, 32, 64, 128, 256, 2048] {
                let name = format!("dfs/{label}/work-{work}/state-24");
                if name.contains(&filter) {
                    rows.push(bench::<0>(
                        &name,
                        case,
                        work,
                        options(TraversalStrategy::DepthFirst, false, None),
                        samples,
                        mode,
                    )?);
                }
            }
        }
        for (label, case) in [&cases[0], &cases[2], &cases[4]] {
            for parallel in [false, true] {
                for work in [0, 256] {
                    let name = format!("bfs-parallel-{parallel}/{label}/work-{work}/state-24");
                    if name.contains(&filter) {
                        rows.push(bench::<0>(
                            &name,
                            case,
                            work,
                            options(TraversalStrategy::BreadthFirst, parallel, None),
                            samples,
                            mode,
                        )?);
                    }
                }
            }
        }
        for (label, case) in [&cases[0], &cases[2]] {
            for work in [0, 256] {
                let name = format!("dfs/{label}/work-{work}/state-280");
                if name.contains(&filter) {
                    rows.push(bench::<32>(
                        &name,
                        case,
                        work,
                        options(TraversalStrategy::DepthFirst, false, None),
                        samples,
                        mode,
                    )?);
                }
            }
        }
        for limit in [1, 64] {
            for work in [0, 256] {
                let name = format!("dfs/diamonds-shared/work-{work}/max-paths-{limit}");
                if name.contains(&filter) {
                    rows.push(bench::<0>(
                        &name,
                        &cases[0].1,
                        work,
                        options(TraversalStrategy::DepthFirst, false, Some(limit)),
                        samples,
                        mode,
                    )?);
                }
            }
        }
        Ok(rows)
    })?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "rxgraph_version": env!("CARGO_PKG_VERSION"), "arch": std::env::consts::ARCH,
            "os": std::env::consts::OS, "rayon_threads": 4, "samples": samples,
            "cache_mode": if mode == CacheMode::Worker {"worker-local"} else {"shared"},
            "method": "release; cold cache per search; full path output and teardown included; counters and equality checks outside timing; alternating paired samples; synthetic CPU kernels",
            "memory_note": "live_entry_payload_bytes excludes empty buckets, control bytes, allocator and mutex overhead",
            "cases": rows,
        }))?
    );
    Ok(())
}
