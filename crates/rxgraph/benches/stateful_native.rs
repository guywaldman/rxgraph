use std::{hint::black_box, sync::Arc};

use anyhow::Result;
use arrow::{
    array::{ArrayRef, BooleanArray, RecordBatch, UInt64Array},
    datatypes::{DataType, Field, Schema},
};
use criterion::{Criterion, criterion_group, criterion_main};
use rxgraph::{EdgeCtx, Graph, Kernel, NodeId, RunOptions, StateRow, TraversalStrategy, Value};

const CASES: usize = 128;
const DEPTH: usize = 12;
const FANOUT: usize = 64;
const DECOYS: usize = 1_024;

#[derive(Clone, Copy)]
struct SearchState {
    spent: u64,
}

#[derive(Clone, Copy)]
struct SearchKernel {
    target: u64,
}

impl Kernel for SearchKernel {
    type State = SearchState;

    fn initial_state(&self, _graph: &Graph, _start: NodeId) -> Self::State {
        SearchState { spent: 0 }
    }

    fn visit(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<bool> {
        Ok(cx.edge_bool("allowed")?.unwrap_or(false)
            && cx.state().spent + cx.edge_u64("cost")?.unwrap_or(0) <= DEPTH as u64)
    }

    fn next_state(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Self::State> {
        Ok(SearchState {
            spent: cx.state().spent + cx.edge_u64("cost")?.unwrap_or(0),
        })
    }

    fn stop(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<bool> {
        Ok(cx.dest_id() == Some(rxgraph::GraphId::U64(self.target)))
    }

    fn state_row(&self, state: &Self::State) -> StateRow {
        vec![("spent".into(), Value::U64(state.spent))]
    }
}

struct Workload {
    graph: Graph,
    starts: Vec<rxgraph::OwnedGraphId>,
    kernel: SearchKernel,
}

impl Workload {
    fn new() -> Self {
        let chain_nodes = CASES * DEPTH;
        let target = (chain_nodes + DECOYS) as u64;
        let node_count = target as usize + 1;
        let nodes = batch(
            vec![Field::new("id", DataType::UInt64, false)],
            vec![Arc::new(UInt64Array::from_iter_values(
                0..node_count as u64,
            ))],
        );

        let edge_count = CASES * DEPTH * (FANOUT + 1);
        let mut ids = Vec::with_capacity(edge_count);
        let mut srcs = Vec::with_capacity(edge_count);
        let mut dests = Vec::with_capacity(edge_count);
        let mut costs = Vec::with_capacity(edge_count);
        let mut allowed = Vec::with_capacity(edge_count);

        for case in 0..CASES {
            for depth in 0..DEPTH {
                let src = (case * DEPTH + depth) as u64;
                let dest = if depth + 1 == DEPTH { target } else { src + 1 };
                push_edge(
                    &mut ids,
                    &mut srcs,
                    &mut dests,
                    &mut costs,
                    &mut allowed,
                    src,
                    dest,
                    1,
                    true,
                );
                for decoy in 0..FANOUT {
                    push_edge(
                        &mut ids,
                        &mut srcs,
                        &mut dests,
                        &mut costs,
                        &mut allowed,
                        src,
                        (chain_nodes + (case * 131 + depth * 67 + decoy) % DECOYS) as u64,
                        DEPTH as u64 + 1,
                        false,
                    );
                }
            }
        }

        let edges = batch(
            vec![
                Field::new("id", DataType::UInt64, false),
                Field::new("src", DataType::UInt64, false),
                Field::new("dest", DataType::UInt64, false),
                Field::new("cost", DataType::UInt64, false),
                Field::new("allowed", DataType::Boolean, false),
            ],
            vec![
                Arc::new(UInt64Array::from(ids)),
                Arc::new(UInt64Array::from(srcs)),
                Arc::new(UInt64Array::from(dests)),
                Arc::new(UInt64Array::from(costs)),
                Arc::new(BooleanArray::from(allowed)),
            ],
        );
        let graph = Graph::new(nodes, edges).unwrap();
        let starts = (0..CASES)
            .map(|case| ((case * DEPTH) as u64).into())
            .collect();
        Self {
            graph,
            starts,
            kernel: SearchKernel { target },
        }
    }

    fn run(&self, strategy: TraversalStrategy, parallel: bool, first: bool) -> RunOptions {
        RunOptions {
            start_nodes: if first {
                self.starts[..1].to_vec()
            } else {
                self.starts.clone()
            },
            max_depth: Some(DEPTH),
            max_paths: Some(if first { 1 } else { CASES }),
            strategy,
            parallel,
            ..RunOptions::default()
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_edge(
    ids: &mut Vec<u64>,
    srcs: &mut Vec<u64>,
    dests: &mut Vec<u64>,
    costs: &mut Vec<u64>,
    allowed: &mut Vec<bool>,
    src: u64,
    dest: u64,
    cost: u64,
    is_allowed: bool,
) {
    ids.push(ids.len() as u64);
    srcs.push(src);
    dests.push(dest);
    costs.push(cost);
    allowed.push(is_allowed);
}

fn batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn bench_stateful_native(c: &mut Criterion) {
    let workload = Workload::new();
    let mut group = c.benchmark_group("stateful_native");

    group.bench_function("paths_dfs_serial", |b| {
        b.iter(|| {
            black_box(
                workload
                    .graph
                    .search_with(
                        workload.kernel,
                        workload.run(TraversalStrategy::DepthFirst, false, false),
                    )
                    .unwrap(),
            )
        })
    });
    group.bench_function("paths_bfs_parallel", |b| {
        b.iter(|| {
            black_box(
                workload
                    .graph
                    .search_with(
                        workload.kernel,
                        workload.run(TraversalStrategy::BreadthFirst, true, false),
                    )
                    .unwrap(),
            )
        })
    });
    group.bench_function("first_bfs_serial", |b| {
        b.iter(|| {
            black_box(
                workload
                    .graph
                    .search_with(
                        workload.kernel,
                        workload.run(TraversalStrategy::BreadthFirst, false, true),
                    )
                    .unwrap(),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_stateful_native);
criterion_main!(benches);
