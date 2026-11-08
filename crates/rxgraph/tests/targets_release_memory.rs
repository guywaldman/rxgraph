//! Checks that expanded BFS prefixes do not retain spilled target bounds.
use std::{alloc::System, sync::Arc};

use anyhow::Result;
use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use rxgraph::{
    EdgeCtx, Evaluation, Graph, Kernel, NodeId, OwnedGraphId, RunOptions, TargetCtx, TargetOptions,
    Transition, TraversalStrategy,
};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

const CHAIN: u64 = 1024;
const TARGETS: u64 = 16;

struct ManyTargets;
impl Kernel for ManyTargets {
    type State = ();
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<()> {
        Ok(())
    }
    fn transition(&self, cx: &EdgeCtx<'_, ()>) -> Result<Transition<()>> {
        Ok(if u64::from(cx.dest()) > CHAIN {
            Transition::Complete(())
        } else {
            Transition::Continue(())
        })
    }
    fn possible_targets(&self, cx: &TargetCtx<'_, ()>) -> Result<Option<Vec<OwnedGraphId>>> {
        Ok(match cx {
            TargetCtx::Start { .. } => Some((1..=TARGETS).map(|n| (CHAIN + n).into()).collect()),
            TargetCtx::Edge { .. } => None,
        })
    }
}

fn graph() -> Result<Graph> {
    let mut src = Vec::new();
    let mut dest = Vec::new();
    for node in 0..CHAIN {
        src.push(node);
        dest.push(node + 1);
    }
    for n in 1..=TARGETS {
        src.push(CHAIN);
        dest.push(CHAIN + n);
    }
    let column = |values| Arc::new(UInt64Array::from(values)) as ArrayRef;
    let nodes =
        RecordBatch::try_from_iter([("id", column((0..=CHAIN + TARGETS).collect::<Vec<_>>()))])?;
    let edges = RecordBatch::try_from_iter([
        ("id", column((0..src.len() as u64).collect::<Vec<_>>())),
        ("src", column(src)),
        ("dest", column(dest)),
    ])?;
    Graph::new(nodes, edges)
}

fn retained_search_bytes(graph: &Graph, release_expanded: bool) -> Result<i64> {
    let region = Region::new(GLOBAL);
    let run = RunOptions {
        start_nodes: vec![0u64.into()],
        strategy: TraversalStrategy::BreadthFirst,
        parallel: false,
        ..Default::default()
    };
    let options = TargetOptions {
        evaluation: Evaluation::Incremental,
        reverse: false,
        release_expanded,
    };
    let run = run.with_optimizations(options);
    let mut stream = graph.search_batches_with(ManyTargets, run, 1)?;
    let first = stream.next().expect("first result")?;
    assert_eq!(first[0].nodes.len(), CHAIN as usize + 2);
    drop(first);
    let stats = region.change();
    let retained = stats.bytes_allocated as i64 - stats.bytes_deallocated as i64;
    stream.close();
    Ok(retained)
}

#[test]
fn releasing_expanded_prefixes_drops_large_target_bounds() -> Result<()> {
    let graph = graph()?;
    let keep = retained_search_bytes(&graph, false)?;
    let release = retained_search_bytes(&graph, true)?;
    assert!(
        keep - release >= 32 * 1024,
        "expanded target bounds should be freed: keep={keep}, release={release}"
    );
    Ok(())
}
