//! Isolated memory experiment. Each invocation runs exactly one cold search.
//! RX_MEM_CASE=arc-eager RX_MEM_CACHE=1 cargo run --release -p rxgraph --example reconvergence_memory --offline
//! This allocator-instrumented executable is separate from the timing benchmark.

#[path = "reconvergence_bench.rs"]
#[allow(dead_code)]
mod fixture;
#[path = "reconvergence_bench/heap.rs"]
mod heap;

use std::{
    alloc::{GlobalAlloc, Layout},
    hash::Hash,
    hint::black_box,
    mem::size_of,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use anyhow::Result;
use fixture::{
    State, WorkKernel,
    scale::{Dataset, Shape, Validator},
};
use heap::{Buffer, CacheMode, HeapKernel, HeapState, Memo};
use rxgraph::{GraphPath, Kernel, RunOptions, TraversalStrategy};
use serde_json::json;
use stats_alloc::INSTRUMENTED_SYSTEM;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Meter;
impl Meter {
    fn grow(bytes: usize) {
        let live = LIVE.fetch_add(bytes, Ordering::SeqCst) + bytes;
        PEAK.fetch_max(live, Ordering::SeqCst);
    }
}
// Records requested Rust heap bytes, independently of allocator-retained RSS.
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { INSTRUMENTED_SYSTEM.alloc(layout) };
        if !ptr.is_null() {
            Self::grow(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { INSTRUMENTED_SYSTEM.alloc_zeroed(layout) };
        if !ptr.is_null() {
            Self::grow(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            INSTRUMENTED_SYSTEM.dealloc(ptr, layout);
        }
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let ptr = unsafe { INSTRUMENTED_SYSTEM.realloc(ptr, layout, new_size) };
        if !ptr.is_null() {
            if new_size > layout.size() {
                Self::grow(new_size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::SeqCst);
            }
        }
        ptr
    }
}
#[global_allocator]
static ALLOCATOR: Meter = Meter;

trait Inspect {
    fn meta(&self) -> State<0>;
    fn validate_buffer(&self, words: usize);
}
impl Inspect for State<0> {
    fn meta(&self) -> State<0> {
        *self
    }
    fn validate_buffer(&self, _: usize) {}
}
impl<B: Buffer> Inspect for HeapState<B> {
    fn meta(&self) -> State<0> {
        self.meta
    }
    fn validate_buffer(&self, words: usize) {
        let values = self.buffer.values();
        assert_eq!(values.len(), words);
        assert_eq!(values[0], self.meta.score);
        for (i, &value) in values.iter().enumerate().skip(1) {
            assert_eq!(value, i as u64 + 19);
        }
    }
}

fn consume<S: Inspect>(
    paths: Vec<GraphPath<'_, S>>,
    validator: &mut Validator,
    count: &mut usize,
    words: usize,
) {
    *count += paths.len();
    black_box(&paths);
    for path in paths {
        path.state.validate_buffer(words);
        validator.check(&GraphPath {
            nodes: path.nodes,
            edges: path.edges,
            state: path.state.meta(),
            intermediate_states: None,
        });
        assert!(path.intermediate_states.is_none());
    }
}

#[derive(Clone, Copy)]
struct Options {
    mode: CacheMode,
    parallel: bool,
    eager: bool,
    bfs: bool,
    words: usize,
    check_cleanup: bool,
}

fn measure<K>(
    name: &str,
    data: &Dataset,
    kernel: K,
    calls: &AtomicUsize,
    options: Options,
) -> Result<serde_json::Value>
where
    K: Kernel + Sync,
    K::State: Inspect + Eq + Hash + Send + Sync + 'static,
{
    // Bitset and graph exist before the allocation region; neither arm retains audit copies.
    let mut validator = Validator::new(data, &[]);
    let before = INSTRUMENTED_SYSTEM.stats();
    let live_before = LIVE.load(Ordering::SeqCst);
    PEAK.store(live_before, Ordering::SeqCst);
    let run = RunOptions {
        start_nodes: data.starts.clone(),
        strategy: if options.bfs {
            TraversalStrategy::BreadthFirst
        } else {
            TraversalStrategy::DepthFirst
        },
        parallel: options.parallel,
        ..Default::default()
    };
    let mut paths = 0;
    macro_rules! search {
        ($kernel:expr) => {{
            if options.eager {
                let result = data.case.graph.search_with($kernel, run)?;
                (Some(result.paths), result.stats)
            } else {
                let mut stream = data.case.graph.search_batches_with($kernel, run, 1024)?;
                for batch in stream.by_ref() {
                    consume(batch?, &mut validator, &mut paths, options.words);
                }
                (None, stream.stats())
            }
        }};
    }
    let (result, stats) = match options.mode {
        CacheMode::None => search!(kernel),
        CacheMode::CloneKey => search!(Memo::<_, false>(kernel)),
        CacheMode::Borrowed => search!(Memo::<_, true>(kernel)),
    };
    let live_with_cache = LIVE.load(Ordering::SeqCst).saturating_sub(live_before);
    let (entries, capacity) = heap::clear::<K::State, true>(options.mode, options.parallel);
    let peak_search = PEAK.load(Ordering::SeqCst).saturating_sub(live_before);
    let live_after_cache_drop = LIVE.load(Ordering::SeqCst).saturating_sub(live_before);
    if let Some(result) = result {
        consume(result, &mut validator, &mut paths, options.words);
    }
    assert_eq!(paths, data.case.expected_paths);
    let alloc = INSTRUMENTED_SYSTEM.stats() - before;
    let net_live = LIVE.load(Ordering::SeqCst) as i64 - live_before as i64;
    if options.check_cleanup {
        assert_eq!(net_live, 0, "all search-owned allocations must be released");
    }
    let computed = calls.load(Ordering::Relaxed);
    assert_eq!(
        computed,
        if options.mode != CacheMode::None {
            entries
        } else {
            stats.evaluated_edges
        }
    );
    Ok(json!({
        "case": name, "cached": options.mode != CacheMode::None, "cache_mode": options.mode.label(),
        "output": if options.eager {"eager"} else {"streaming-1024"},
        "strategy": if options.bfs {"bfs"} else {"dfs"}, "parallel": options.parallel,
        "buffer_bytes": options.words * size_of::<u64>(), "state_inline_bytes": size_of::<K::State>(),
        "paths": paths, "evaluated_edges": stats.evaluated_edges, "computed_transitions": computed,
        "cache_entries": entries, "cache_capacity_entries": capacity,
        "peak_search_live_heap_bytes": peak_search, "live_at_result_with_cache_bytes": live_with_cache,
        "live_result_after_cache_drop_bytes": live_after_cache_drop,
        "bytes_allocated": alloc.bytes_allocated, "bytes_deallocated": alloc.bytes_deallocated,
        "allocations": alloc.allocations, "reallocations": alloc.reallocations,
        "net_live_after_teardown_bytes": net_live, "exact_output_validation": true,
        "parallel_edges": stats.parallel_edges,
        "measurement": "requested Rust heap bytes; graph and audit bitset excluded; cache setup/growth, traversal, materialized output and teardown included; memory-only instrumented executable",
    }))
}

fn warm_parallel(mode: CacheMode, storage: &str) -> Result<()> {
    let data = Shape::Diamonds {
        components: 16,
        depth: 8,
        bits: 0,
    }
    .build()?;
    let calls = AtomicUsize::new(0);
    let inner = WorkKernel::<0, true> {
        case: &data.case,
        payload: &[],
        calls: &calls,
        reject_depth: None,
    };
    let options = Options {
        mode,
        parallel: true,
        eager: false,
        bfs: true,
        words: 8,
        check_cleanup: false,
    };
    let result = match storage {
        "inline" => measure("worker-warmup", &data, inner, &calls, options)?,
        "vec" => measure(
            "worker-warmup",
            &data,
            HeapKernel::<Vec<u64>, true> {
                inner,
                words: 8,
                marker: std::marker::PhantomData,
            },
            &calls,
            options,
        )?,
        _ => measure(
            "worker-warmup",
            &data,
            HeapKernel::<Arc<Vec<u64>>, true> {
                inner,
                words: 8,
                marker: std::marker::PhantomData,
            },
            &calls,
            options,
        )?,
    };
    assert!(result["parallel_edges"].as_u64().unwrap() > 0);
    Ok(())
}

fn main() -> Result<()> {
    anyhow::ensure!(!cfg!(debug_assertions), "run with --release");
    let name = std::env::var("RX_MEM_CASE").unwrap_or_else(|_| "arc-eager".into());
    let mode = match std::env::var("RX_MEM_CACHE").as_deref().unwrap_or("0") {
        "0" => CacheMode::None,
        "1" => CacheMode::CloneKey,
        "2" => CacheMode::Borrowed,
        value => anyhow::bail!("unknown cache mode {value}"),
    };
    let words = std::env::var("RX_MEM_WORDS")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(512);
    anyhow::ensure!(words > 0, "buffer must have at least one word");
    let parallel = std::env::var("RX_MEM_PARALLEL").as_deref() == Ok("1");
    let diagnostic = std::env::var("RX_MEM_DIAG").as_deref() == Ok("1");
    let (components, depth, bits, eager, bfs) = match name.as_str() {
        "inline-stream" => (1024, 12, 0, false, false),
        "inline-eager" => (256, 12, 0, true, false),
        "inline-unique-stream" => (64, 14, 14, false, false),
        "vec-eager" | "arc-eager" => (64, 12, 0, true, false),
        "vec-stream" | "arc-stream" => (64, 12, 0, false, false),
        "vec-one-stream" | "arc-one-stream" => (1, 18, 0, false, false),
        "vec-bfs-stream" | "arc-bfs-stream" => (16, 12, 0, false, true),
        "vec-unique-eager" | "arc-unique-eager" => (1, 16, 16, true, false),
        value => anyhow::bail!("unknown case {value}"),
    };
    let options = Options {
        mode,
        parallel,
        eager,
        bfs,
        words,
        check_cleanup: !diagnostic,
    };
    let data = Shape::Diamonds {
        components,
        depth,
        bits,
    }
    .build()?;
    let calls = AtomicUsize::new(0);
    let execute = || -> Result<_> {
        calls.store(0, Ordering::Relaxed);
        let inner = WorkKernel::<0, true> {
            case: &data.case,
            payload: &[],
            calls: &calls,
            reject_depth: None,
        };
        match name.split('-').next().unwrap() {
            "inline" => measure(&name, &data, inner, &calls, options),
            "vec" => measure(
                &name,
                &data,
                HeapKernel::<Vec<u64>, true> {
                    inner,
                    words,
                    marker: std::marker::PhantomData,
                },
                &calls,
                options,
            ),
            _ => measure(
                &name,
                &data,
                HeapKernel::<Arc<Vec<u64>>, true> {
                    inner,
                    words,
                    marker: std::marker::PhantomData,
                },
                &calls,
                options,
            ),
        }
    };
    let result = if parallel {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
        pool.install(|| heap::prime(true));
        pool.install(|| warm_parallel(mode, name.split('-').next().unwrap()))?;
        if diagnostic {
            pool.install(|| {
                Ok::<_, anyhow::Error>(json!({"first":execute()?, "second":execute()?}))
            })?
        } else {
            pool.install(execute)?
        }
    } else {
        heap::prime(false);
        execute()?
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
