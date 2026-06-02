//! Memory benchmarking for graph construction, holding, and from-source search.
//!
//! Unlike the criterion benches (which measure time), this binary uses `stats_alloc` to
//! report allocation deltas at each stage. Run with:
//!
//! ```sh
//! cargo bench -p rxgraph --bench memory
//! ```
//!
//! It builds a large, sparse graph where only a small subset is reachable from the source,
//! and reports bytes allocated/RSS.

use std::{
    alloc::System,
    hint::black_box,
    sync::Arc,
    time::Instant,
};

use arrow::{
    array::{ArrayRef, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use rxgraph::Graph;
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// Number of nodes in the synthetic graph.
const NODES: u64 = 5_000_000;
/// Length of the single reachable chain from node 0 (the "working set").
const REACHABLE_CHAIN: u64 = 5_000;

fn batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

/// Builds a graph with `NODES` contiguous u64 node ids and a single linear chain of
/// `REACHABLE_CHAIN` edges from node 0. Everything past the chain is unreachable, so a
/// search from node 0 only ever needs a tiny working set.
fn tables() -> (RecordBatch, RecordBatch) {
    let node_ids: Vec<u64> = (0..NODES).collect();
    let nodes = batch(
        vec![Field::new("id", DataType::UInt64, false)],
        vec![Arc::new(UInt64Array::from(node_ids)) as ArrayRef],
    );

    let edge_count = REACHABLE_CHAIN;
    let edge_ids: Vec<u64> = (0..edge_count).collect();
    let srcs: Vec<u64> = (0..edge_count).collect();
    let dests: Vec<u64> = (1..=edge_count).collect();
    let edges = batch(
        vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("src", DataType::UInt64, false),
            Field::new("dest", DataType::UInt64, false),
        ],
        vec![
            Arc::new(UInt64Array::from(edge_ids)) as ArrayRef,
            Arc::new(UInt64Array::from(srcs)),
            Arc::new(UInt64Array::from(dests)),
        ],
    );

    (nodes, edges)
}

fn mib(bytes: isize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Runs the received callback inside an allocation-tracking region and reports the net change.
fn measure<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let region = Region::new(GLOBAL);
    let started = Instant::now();
    let value = f();
    let stats = region.change();
    let elapsed = started.elapsed();
    let resident = stats.bytes_allocated as isize - stats.bytes_deallocated as isize;
    eprintln!(
        "{label:<28} resident={:>9.2} MiB  allocations={:<8} (alloc={:.2} MiB, dealloc={:.2} MiB)  {elapsed:?}",
        mib(resident),
        stats.allocations,
        mib(stats.bytes_allocated as isize),
        mib(stats.bytes_deallocated as isize),
    );
    value
}

fn main() {
    eprintln!(
        "memory profile: nodes={NODES} reachable_chain={REACHABLE_CHAIN}\n"
    );

    let (nodes, edges) = tables();

    // Construction: forward CSR + identity only. Reverse CSR is NOT built here (lazy).
    let graph = measure("construct", || Graph::new(nodes, edges).unwrap());

    // Forward-only BFS from node 0: touches only the reachable chain.
    measure("bfs_from_source", || {
        black_box(graph.bfs_u64(0, None).unwrap());
    });

    // First degree query forces the lazy reverse CSR to materialize.
    measure("in_degrees (builds rev CSR)", || {
        black_box(graph.in_degrees());
    });

    // Subsequent reverse-adjacency use is free (cached).
    measure("weakly_connected (rev cached)", || {
        black_box(graph.weakly_connected_components_u64());
    });

    // Keep the graph alive so its footprint is attributed to the stages above.
    black_box(&graph);
}
