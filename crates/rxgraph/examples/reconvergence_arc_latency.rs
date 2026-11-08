//! Uninstrumented latency runs using the same kernel/cache as the memory probe.
//! RX_ARC_LAT_CASE=eager cargo run --release -p rxgraph --example reconvergence_arc_latency --offline

#[path = "reconvergence_bench.rs"]
#[allow(dead_code)]
mod fixture;
#[path = "reconvergence_bench/heap.rs"]
mod heap;

use std::{
    hash::Hash,
    hint::black_box,
    marker::PhantomData,
    sync::{Arc, atomic::AtomicUsize},
    time::Instant,
};

use anyhow::Result;
use fixture::{
    WorkKernel,
    scale::{Dataset, Shape},
};
use heap::{CacheMode, HeapKernel, Memo};
use rxgraph::{Kernel, RunOptions, SearchStats, TraversalStrategy};
use serde_json::{Value as Json, json};

#[derive(Clone, Copy)]
struct Spec {
    shape: Shape,
    words: usize,
    eager: bool,
    bfs: bool,
    parallel: bool,
    clone_key: bool,
}

#[derive(Clone, Copy)]
struct Arm {
    shared: bool,
    mode: CacheMode,
}
impl Arm {
    fn label(self) -> String {
        format!(
            "{}-{}",
            if self.shared { "arc" } else { "vec" },
            self.mode.label()
        )
    }
}

fn execute<K>(data: &Dataset, kernel: K, spec: Spec, mode: CacheMode) -> Result<SearchStats>
where
    K: Kernel + Sync,
    K::State: Clone + Eq + Hash + Send + Sync + 'static,
{
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
    macro_rules! search {
        ($kernel:expr) => {{
            if spec.eager {
                let result = data.case.graph.search_with($kernel, run)?;
                count = result.paths.len();
                black_box(&result.paths);
                (Some(result.paths), result.stats)
            } else {
                let mut stream = data.case.graph.search_batches_with($kernel, run, 1024)?;
                for batch in stream.by_ref() {
                    let batch = batch?;
                    count += batch.len();
                    black_box(&batch);
                }
                (None, stream.stats())
            }
        }};
    }
    let (result, stats) = match mode {
        CacheMode::None => search!(kernel),
        CacheMode::CloneKey => search!(Memo::<_, false>(kernel)),
        CacheMode::Borrowed => search!(Memo::<_, true>(kernel)),
    };
    heap::clear::<K::State, false>(mode, spec.parallel);
    drop(result);
    assert_eq!(count, data.case.expected_paths);
    if spec.parallel {
        assert!(stats.parallel_edges > 0);
    }
    Ok(stats)
}

fn run_one(data: &Dataset, spec: Spec, arm: Arm) -> Result<SearchStats> {
    let calls = AtomicUsize::new(0);
    let inner = WorkKernel::<0, false> {
        case: &data.case,
        payload: &[],
        calls: &calls,
        reject_depth: None,
    };
    if arm.shared {
        execute(
            data,
            HeapKernel::<Arc<Vec<u64>>, false> {
                inner,
                words: spec.words,
                marker: PhantomData,
            },
            spec,
            arm.mode,
        )
    } else {
        execute(
            data,
            HeapKernel::<Vec<u64>, false> {
                inner,
                words: spec.words,
                marker: PhantomData,
            },
            spec,
            arm.mode,
        )
    }
}

fn window(data: &Dataset, spec: Spec, arm: Arm, minimum: f64) -> Result<(f64, usize, SearchStats)> {
    let start = Instant::now();
    let mut searches = 0;
    loop {
        let stats = black_box(run_one(data, spec, arm)?);
        searches += 1;
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= minimum {
            return Ok((elapsed, searches, stats));
        }
    }
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn main() -> Result<()> {
    anyhow::ensure!(!cfg!(debug_assertions), "run with --release");
    let name = std::env::var("RX_ARC_LAT_CASE").unwrap_or_else(|_| "eager".into());
    let samples: usize = std::env::var("RX_ARC_LAT_SAMPLES")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(5);
    let minimum: f64 = std::env::var("RX_ARC_LAT_SECONDS")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(2.0);
    anyhow::ensure!(
        samples >= 3 && minimum >= 1.0,
        "use >=3 samples and >=1s per window"
    );
    let many = Shape::Diamonds {
        components: 64,
        depth: 12,
        bits: 0,
    };
    let bfs = Shape::Diamonds {
        components: 16,
        depth: 12,
        bits: 0,
    };
    let spec = match name.as_str() {
        "eager" => Spec {
            shape: many,
            words: 512,
            eager: true,
            bfs: false,
            parallel: false,
            clone_key: true,
        },
        "stream" => Spec {
            shape: many,
            words: 512,
            eager: false,
            bfs: false,
            parallel: false,
            clone_key: true,
        },
        "bfs-stream" => Spec {
            shape: bfs,
            words: 512,
            eager: false,
            bfs: true,
            parallel: false,
            clone_key: false,
        },
        "unique-eager" => Spec {
            shape: Shape::Diamonds {
                components: 1,
                depth: 16,
                bits: 16,
            },
            words: 512,
            eager: true,
            bfs: false,
            parallel: false,
            clone_key: false,
        },
        "small-stream" => Spec {
            shape: many,
            words: 8,
            eager: false,
            bfs: false,
            parallel: false,
            clone_key: false,
        },
        "parallel-bfs-stream" => Spec {
            shape: bfs,
            words: 512,
            eager: false,
            bfs: true,
            parallel: true,
            clone_key: false,
        },
        value => anyhow::bail!("unknown case {value}"),
    };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    pool.install(|| heap::prime(spec.parallel));
    let output = pool.install(|| -> Result<Json> {
        let data = spec.shape.build()?;
        let mut arms = Vec::new();
        for shared in [false, true] {
            for mode in [CacheMode::None, CacheMode::Borrowed] { arms.push(Arm {shared, mode}); }
            if spec.clone_key { arms.push(Arm {shared, mode:CacheMode::CloneKey}); }
        }
        eprintln!("{name}: {} paths, {} nodes, {} edges; warming {} arms", data.case.expected_paths, data.case.graph.node_count(), data.case.graph.edge_count(), arms.len());
        for &arm in &arms { window(&data, spec, arm, 1.0)?; }
        let mut durations = vec![Vec::new(); arms.len()];
        let mut counts = vec![Vec::new(); arms.len()];
        let mut stats = vec![None; arms.len()];
        for sample in 0..samples {
            for position in 0..arms.len() {
                let position = if sample % 2 == 0 {position} else {arms.len() - 1 - position};
                let i = (position + sample) % arms.len();
                let (seconds, n, result) = window(&data, spec, arms[i], minimum)?;
                durations[i].push(seconds);
                counts[i].push(n);
                stats[i] = Some(result);
            }
            eprintln!("{name}: sample {}/{} complete", sample + 1, samples);
        }
        let mut rows = Vec::new();
        for (i, arm) in arms.iter().enumerate() {
            let per_search = durations[i].iter().zip(&counts[i]).map(|(&elapsed, &n)| elapsed / n as f64).collect::<Vec<_>>();
            let median = median(&per_search);
            let stat = stats[i].unwrap();
            eprintln!("{name}/{}: {:.3}s per search", arm.label(), median);
            rows.push(json!({"arm":arm.label(), "median_seconds":median, "samples_seconds":per_search,
                "measurement_window_seconds":durations[i], "searches_per_window":counts[i],
                "evaluated_edges":stat.evaluated_edges, "parallel_edges":stat.parallel_edges}));
        }
        Ok(json!({"case":name, "words":spec.words, "buffer_bytes":spec.words*8,
            "paths":data.case.expected_paths, "nodes":data.case.graph.node_count(), "edges":data.case.graph.edge_count(),
            "output":if spec.eager {"eager"} else {"streaming-1024"},
            "strategy":if spec.bfs {"bfs"} else {"dfs"}, "parallel":spec.parallel,
            "samples":samples, "minimum_window_seconds":minimum, "arms":rows,
            "method":"release; default system allocator; no allocation instrumentation or timed kernel counters; cold cache and full result materialization/drop included; graph construction excluded; rotating arm order"}))
    })?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
