use std::{env, hint::black_box, sync::Arc, time::Instant};

use anyhow::Result;
use arrow::{
    array::{ArrayRef, BooleanArray, RecordBatch, UInt64Array},
    datatypes::{DataType, Field, Schema},
};
use criterion::Criterion;
use rxgraph::{EdgeCtx, Graph, Kernel, NodeId, RunOptions, Transition, TraversalStrategy};

const DEFAULT_CASES: usize = 128;
const DEFAULT_DEPTH: usize = 12;
const DEFAULT_FANOUT: usize = 64;

#[derive(Clone, Copy)]
struct SearchState {
    spent: u64,
}

#[derive(Clone, Copy)]
struct SearchKernel {
    target: u64,
    budget: u64,
}

impl Kernel for SearchKernel {
    type State = SearchState;

    fn initial_state(&self, _graph: &Graph, _start: NodeId) -> Result<Self::State> {
        Ok(SearchState { spent: 0 })
    }

    fn transition(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Transition<Self::State>> {
        if !cx.edge_bool("allowed")?.unwrap_or(false) {
            return Ok(Transition::Reject);
        }
        let state = SearchState {
            spent: cx.state().spent + cx.edge_u64("cost")?.unwrap_or(0),
        };
        Ok(if state.spent > self.budget {
            Transition::Reject
        } else if cx.dest_id() == Some(rxgraph::GraphId::U64(self.target)) {
            Transition::Complete(state)
        } else {
            Transition::Continue(state)
        })
    }
}

struct Workload {
    graph: Graph,
    starts: Vec<rxgraph::OwnedGraphId>,
    kernel: SearchKernel,
    cases: usize,
    depth: usize,
}

impl Workload {
    fn new() -> Self {
        let cases = env_usize("RXGRAPH_NATIVE_CASES", DEFAULT_CASES);
        let depth = env_usize("RXGRAPH_NATIVE_DEPTH", DEFAULT_DEPTH);
        let fanout = env_usize("RXGRAPH_NATIVE_FANOUT", DEFAULT_FANOUT);
        let decoys = env_usize("RXGRAPH_NATIVE_DECOYS", fanout * 16);
        let chain_nodes = cases * depth;
        let target = (chain_nodes + decoys) as u64;
        let node_count = target as usize + 1;
        let nodes = batch(
            vec![Field::new("id", DataType::UInt64, false)],
            vec![Arc::new(UInt64Array::from_iter_values(
                0..node_count as u64,
            ))],
        );

        let edge_count = cases * depth * (fanout + 1);
        let mut ids = Vec::with_capacity(edge_count);
        let mut srcs = Vec::with_capacity(edge_count);
        let mut dests = Vec::with_capacity(edge_count);
        let mut costs = Vec::with_capacity(edge_count);
        let mut allowed = Vec::with_capacity(edge_count);

        for case in 0..cases {
            for level in 0..depth {
                let src = (case * depth + level) as u64;
                let dest = if level + 1 == depth { target } else { src + 1 };
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
                for decoy in 0..fanout {
                    push_edge(
                        &mut ids,
                        &mut srcs,
                        &mut dests,
                        &mut costs,
                        &mut allowed,
                        src,
                        (chain_nodes + (case * 131 + level * 67 + decoy) % decoys) as u64,
                        depth as u64 + 1,
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
        let starts = (0..cases)
            .map(|case| ((case * depth) as u64).into())
            .collect();
        Self {
            graph,
            starts,
            kernel: SearchKernel {
                target,
                budget: depth as u64,
            },
            cases,
            depth,
        }
    }

    fn run(&self, strategy: TraversalStrategy, parallel: bool, first: bool) -> RunOptions {
        RunOptions {
            start_nodes: self.starts.clone(),
            max_depth: Some(self.depth),
            max_paths: Some(if first { 1 } else { self.cases }),
            strategy,
            parallel,
            ..RunOptions::default()
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
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

fn measure_large(label: &str, run: impl FnOnce()) {
    let started = Instant::now();
    run();
    eprintln!("stateful_native/{label} {:?}", started.elapsed());
}

fn main() {
    if env::var_os("RXGRAPH_NATIVE_LARGE").is_some() {
        let workload = Workload::new();
        eprintln!(
            "stateful_native large: nodes={} edges={} cases={} depth={}",
            workload.graph.node_count(),
            workload.graph.edge_count(),
            workload.cases,
            workload.depth,
        );
        measure_large("paths_dfs_serial", || {
            black_box(
                workload
                    .graph
                    .search_with(
                        workload.kernel,
                        workload.run(TraversalStrategy::DepthFirst, false, false),
                    )
                    .unwrap(),
            );
        });
        measure_large("paths_bfs_parallel", || {
            black_box(
                workload
                    .graph
                    .search_with(
                        workload.kernel,
                        workload.run(TraversalStrategy::BreadthFirst, true, false),
                    )
                    .unwrap(),
            );
        });
        measure_large("first_bfs_serial", || {
            black_box(
                workload
                    .graph
                    .search_with(
                        workload.kernel,
                        workload.run(TraversalStrategy::BreadthFirst, false, true),
                    )
                    .unwrap(),
            );
        });
        return;
    }

    let mut criterion = Criterion::default().configure_from_args();
    bench_stateful_native(&mut criterion);
    criterion.final_summary();
}
