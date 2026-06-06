use std::{env, hint::black_box, sync::Arc, time::Instant};

use arrow::{
    array::{ArrayRef, BooleanArray, Int32Array, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use criterion::{Criterion, criterion_group, criterion_main};
use petgraph::{
    Direction::Outgoing,
    graph::{DiGraph, NodeIndex},
    visit::EdgeRef,
};
use rxgraph::{
    DslExpr as e, DslKernel, Graph, RustEdgeContext, RustSearchKernel, RustTraversalConfig,
    RustTraversalConfigBuilder, TraversalConfig, TraversalConfigBuilder, TraversalStrategy, Value,
};

type PetGraph = DiGraph<NodeData, EdgeData, u32>;

fn bench_petgraph_search(c: &mut Criterion) {
    let workload = SearchWorkload::new(search_scale());
    verify_search(&workload);

    let started = Instant::now();
    let rxgraph = workload.rxgraph();
    let rx_build = started.elapsed();
    let started = Instant::now();
    let petgraph = workload.petgraph();
    let pet_build = started.elapsed();

    eprintln!(
        "petgraph_search setup: nodes={} edges={} cases={} depth={} fanout={} rxgraph_build={rx_build:?} petgraph_build={pet_build:?}",
        workload.nodes.len(),
        workload.edges.len(),
        workload.starts.len(),
        workload.depth,
        workload.fanout,
    );

    let mut group = c.benchmark_group("petgraph_search");
    group.sample_size(10);

    group.bench_function("rxgraph/build", |b| {
        b.iter(|| black_box(workload.rxgraph()))
    });
    group.bench_function("petgraph/build", |b| {
        b.iter(|| black_box(workload.petgraph()))
    });

    group.bench_function("rxgraph/search_serial", |b| {
        b.iter(|| black_box(rxgraph.search(workload.traversal(false)).unwrap()))
    });
    group.bench_function("rxgraph/search_parallel", |b| {
        b.iter(|| black_box(rxgraph.search(workload.traversal(true)).unwrap()))
    });
    group.bench_function("rxgraph/search_rust", |b| {
        b.iter(|| black_box(rxgraph.search_rust(workload.rust_traversal()).unwrap()))
    });
    group.bench_function("petgraph/search_dfs", |b| {
        b.iter(|| black_box(petgraph_search(&petgraph, &workload)))
    });

    group.finish();
}

criterion_group!(benches, bench_petgraph_search);
criterion_main!(benches);

fn search_scale() -> usize {
    env::var("RXGRAPH_PETGRAPH_SEARCH_SCALE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(120_000)
}

#[derive(Debug, Clone, Copy)]
struct NodeData {
    risk: i32,
    frozen: bool,
    cashout: bool,
}

#[derive(Debug, Clone, Copy)]
struct EdgeData {
    amount: u64,
    time: u64,
    noise: bool,
    flagged: bool,
    success: bool,
}

#[derive(Debug, Clone, Copy)]
struct EdgeSpec {
    src: usize,
    dest: usize,
    data: EdgeData,
}

#[derive(Debug)]
struct SearchWorkload {
    nodes: Vec<NodeData>,
    edges: Vec<EdgeSpec>,
    starts: Vec<u64>,
    depth: usize,
    fanout: usize,
    limit: u64,
}

impl SearchWorkload {
    fn new(scale: usize) -> Self {
        let scale = scale.max(4_096);
        let cases = (scale / 1_500).clamp(64, 128);
        let depth = 8;
        let fanout = (scale / cases / depth / 4).clamp(16, 64);
        let limit = depth as u64 * 180;

        let mut nodes = (0..scale)
            .map(|i| NodeData {
                risk: ((i * 13) % 40) as i32,
                frozen: i != 0 && i % 997 == 0,
                cashout: false,
            })
            .collect::<Vec<_>>();
        let mut edges = Vec::with_capacity(cases * depth * (fanout + 1));
        let mut starts = Vec::with_capacity(cases);

        for case in 0..cases {
            let chain = chain(scale, cases, depth, case);
            starts.push(chain[0] as u64);

            for &node in &chain {
                nodes[node].risk = 2;
                nodes[node].frozen = false;
                nodes[node].cashout = false;
            }
            nodes[*chain.last().unwrap()].cashout = true;

            for (hop, pair) in chain.windows(2).enumerate() {
                edges.push(EdgeSpec {
                    src: pair[0],
                    dest: pair[1],
                    data: EdgeData {
                        amount: 120 + hop as u64,
                        time: hop as u64 * 10,
                        noise: false,
                        flagged: false,
                        success: true,
                    },
                });
            }

            for &src in chain.iter().take(chain.len() - 1) {
                for n in 0..fanout {
                    let dest = (src + n * 97 + case * 131 + 31) % scale;
                    if dest == src {
                        continue;
                    }
                    edges.push(EdgeSpec {
                        src,
                        dest,
                        data: EdgeData {
                            amount: 5 + (n % 30) as u64,
                            time: n as u64,
                            noise: true,
                            flagged: true,
                            success: n % 11 != 0,
                        },
                    });
                }
            }
        }

        Self {
            nodes,
            edges,
            starts,
            depth,
            fanout,
            limit,
        }
    }

    fn rxgraph(&self) -> Graph {
        Graph::new(self.nodes_table(), self.edges_table()).unwrap()
    }

    fn petgraph(&self) -> PetGraph {
        let mut graph = PetGraph::with_capacity(self.nodes.len(), self.edges.len());
        for &node in &self.nodes {
            graph.add_node(node);
        }
        for edge in &self.edges {
            graph.add_edge(
                NodeIndex::new(edge.src),
                NodeIndex::new(edge.dest),
                edge.data,
            );
        }
        graph
    }

    fn traversal(&self, parallel: bool) -> TraversalConfig {
        TraversalConfigBuilder::new(self.kernel())
            .with_start_nodes(self.starts.iter().copied())
            .with_max_depth(self.depth)
            .with_max_paths(self.starts.len())
            .with_strategy(TraversalStrategy::DepthFirst)
            .with_parallelism(parallel)
            .build()
    }

    fn rust_traversal(&self) -> RustTraversalConfig<RxNativeKernel<'_>> {
        RustTraversalConfigBuilder::new(RxNativeKernel { workload: self })
            .with_start_nodes(self.starts.iter().copied())
            .with_max_depth(self.depth)
            .with_max_paths(self.starts.len())
            .with_strategy(TraversalStrategy::DepthFirst)
            .build()
    }

    fn kernel(&self) -> DslKernel {
        let visit = e::edge("noise")
            .not()
            .and(e::edge("flagged").not())
            .and(e::edge("success"))
            .and(e::dest("frozen").not())
            .and(e::state("hops").lt(e::uint_lit(self.depth as u64)))
            .and(
                e::state("amount")
                    .plus(e::edge("amount"))
                    .le(e::uint_lit(self.limit)),
            )
            .and(e::edge("time").ge(e::state("time")))
            .and(e::state("risk").plus(e::dest("risk")).le(e::int_lit(85)));

        DslKernel::new(
            visit,
            [
                (
                    "amount".to_string(),
                    e::state("amount").plus(e::edge("amount")),
                ),
                ("hops".to_string(), e::state("hops").plus(e::uint_lit(1))),
                ("time".to_string(), e::edge("time").plus(e::uint_lit(1))),
                ("risk".to_string(), e::state("risk").plus(e::dest("risk"))),
            ],
            e::dest("cashout"),
            [
                ("amount".to_string(), Value::U64(0)),
                ("hops".to_string(), Value::U64(0)),
                ("time".to_string(), Value::U64(0)),
                ("risk".to_string(), Value::I64(0)),
            ],
        )
    }

    fn nodes_table(&self) -> RecordBatch {
        let ids = (0..self.nodes.len() as u64).collect::<Vec<_>>();
        let risk = self.nodes.iter().map(|node| node.risk).collect::<Vec<_>>();
        let frozen = self
            .nodes
            .iter()
            .map(|node| node.frozen)
            .collect::<Vec<_>>();
        let cashout = self
            .nodes
            .iter()
            .map(|node| node.cashout)
            .collect::<Vec<_>>();

        batch(
            vec![
                Field::new("id", DataType::UInt64, false),
                Field::new("risk", DataType::Int32, false),
                Field::new("frozen", DataType::Boolean, false),
                Field::new("cashout", DataType::Boolean, false),
            ],
            vec![
                Arc::new(UInt64Array::from(ids)) as ArrayRef,
                Arc::new(Int32Array::from(risk)),
                Arc::new(BooleanArray::from(frozen)),
                Arc::new(BooleanArray::from(cashout)),
            ],
        )
    }

    fn edges_table(&self) -> RecordBatch {
        let ids = (0..self.edges.len() as u64).collect::<Vec<_>>();
        let src = self
            .edges
            .iter()
            .map(|edge| edge.src as u64)
            .collect::<Vec<_>>();
        let dest = self
            .edges
            .iter()
            .map(|edge| edge.dest as u64)
            .collect::<Vec<_>>();
        let amount = self
            .edges
            .iter()
            .map(|edge| edge.data.amount)
            .collect::<Vec<_>>();
        let time = self
            .edges
            .iter()
            .map(|edge| edge.data.time)
            .collect::<Vec<_>>();
        let noise = self
            .edges
            .iter()
            .map(|edge| edge.data.noise)
            .collect::<Vec<_>>();
        let flagged = self
            .edges
            .iter()
            .map(|edge| edge.data.flagged)
            .collect::<Vec<_>>();
        let success = self
            .edges
            .iter()
            .map(|edge| edge.data.success)
            .collect::<Vec<_>>();

        batch(
            vec![
                Field::new("id", DataType::UInt64, false),
                Field::new("src", DataType::UInt64, false),
                Field::new("dest", DataType::UInt64, false),
                Field::new("amount", DataType::UInt64, false),
                Field::new("time", DataType::UInt64, false),
                Field::new("noise", DataType::Boolean, false),
                Field::new("flagged", DataType::Boolean, false),
                Field::new("success", DataType::Boolean, false),
            ],
            vec![
                Arc::new(UInt64Array::from(ids)) as ArrayRef,
                Arc::new(UInt64Array::from(src)),
                Arc::new(UInt64Array::from(dest)),
                Arc::new(UInt64Array::from(amount)),
                Arc::new(UInt64Array::from(time)),
                Arc::new(BooleanArray::from(noise)),
                Arc::new(BooleanArray::from(flagged)),
                Arc::new(BooleanArray::from(success)),
            ],
        )
    }
}

fn chain(scale: usize, cases: usize, depth: usize, case: usize) -> Vec<usize> {
    let block = scale / cases;
    let base = case * block;
    let step = (block / (depth + 1)).max(1);
    (0..=depth)
        .map(|hop| (base + hop * step).min(scale - 1))
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct PetState {
    amount: u64,
    hops: u64,
    time: u64,
    risk: i64,
}

#[derive(Debug, Clone, Copy)]
struct PetEntry {
    node: NodeIndex<u32>,
    parent: Option<usize>,
    depth: usize,
    state: PetState,
}

#[derive(Debug, Clone, Copy)]
struct RxNativeKernel<'a> {
    workload: &'a SearchWorkload,
}

impl RustSearchKernel for RxNativeKernel<'_> {
    type State = PetState;

    fn initial_state(&self) -> Self::State {
        PetState {
            amount: 0,
            hops: 0,
            time: 0,
            risk: 0,
        }
    }

    fn visit(&self, ctx: RustEdgeContext<'_, Self::State>) -> bool {
        let edge = self.workload.edges[ctx.edge as usize].data;
        let dest = self.workload.nodes[ctx.dest as usize];
        petgraph_visit(*ctx.state, dest, edge, self.workload)
    }

    fn next_state(&self, ctx: RustEdgeContext<'_, Self::State>) -> Self::State {
        let edge = self.workload.edges[ctx.edge as usize].data;
        let dest = self.workload.nodes[ctx.dest as usize];
        PetState {
            amount: ctx.state.amount + edge.amount,
            hops: ctx.state.hops + 1,
            time: edge.time + 1,
            risk: ctx.state.risk + dest.risk as i64,
        }
    }

    fn stop(&self, ctx: RustEdgeContext<'_, Self::State>) -> bool {
        self.workload.nodes[ctx.dest as usize].cashout
    }
}

#[derive(Debug)]
struct PetSearchResult {
    paths: Vec<Vec<u64>>,
    evaluated_edges: usize,
    accepted_edges: usize,
}

fn petgraph_search(graph: &PetGraph, workload: &SearchWorkload) -> PetSearchResult {
    let mut arena = Vec::with_capacity(workload.starts.len() * (workload.depth + 1));
    let mut frontier = Vec::with_capacity(workload.starts.len());
    let mut paths = Vec::with_capacity(workload.starts.len());
    let mut evaluated_edges = 0;
    let mut accepted_edges = 0;

    for &start in &workload.starts {
        frontier.push(arena.len());
        arena.push(PetEntry {
            node: NodeIndex::new(start as usize),
            parent: None,
            depth: 0,
            state: PetState {
                amount: 0,
                hops: 0,
                time: 0,
                risk: 0,
            },
        });
    }

    while let Some(parent) = frontier.pop() {
        let parent_entry = arena[parent];
        if parent_entry.depth >= workload.depth {
            continue;
        }

        for edge in graph.edges_directed(parent_entry.node, Outgoing) {
            evaluated_edges += 1;
            let edge_data = *edge.weight();
            let dest = edge.target();
            let dest_data = graph[dest];
            let state = parent_entry.state;

            if !petgraph_visit(state, dest_data, edge_data, workload) {
                continue;
            }

            let next_state = PetState {
                amount: state.amount + edge_data.amount,
                hops: state.hops + 1,
                time: edge_data.time + 1,
                risk: state.risk + dest_data.risk as i64,
            };
            let child = arena.len();
            arena.push(PetEntry {
                node: dest,
                parent: Some(parent),
                depth: parent_entry.depth + 1,
                state: next_state,
            });
            accepted_edges += 1;

            if dest_data.cashout {
                paths.push(materialize_petgraph_path(&arena, child));
                if paths.len() >= workload.starts.len() {
                    return PetSearchResult {
                        paths,
                        evaluated_edges,
                        accepted_edges,
                    };
                }
            } else {
                frontier.push(child);
            }
        }
    }

    PetSearchResult {
        paths,
        evaluated_edges,
        accepted_edges,
    }
}

fn petgraph_visit(
    state: PetState,
    dest: NodeData,
    edge: EdgeData,
    workload: &SearchWorkload,
) -> bool {
    !edge.noise
        && !edge.flagged
        && edge.success
        && !dest.frozen
        && state.hops < workload.depth as u64
        && state.amount + edge.amount <= workload.limit
        && edge.time >= state.time
        && state.risk + dest.risk as i64 <= 85
}

fn materialize_petgraph_path(arena: &[PetEntry], mut index: usize) -> Vec<u64> {
    let mut path = Vec::new();
    loop {
        let entry = arena[index];
        path.push(entry.node.index() as u64);
        let Some(parent) = entry.parent else {
            break;
        };
        index = parent;
    }
    path.reverse();
    path
}

fn verify_search(workload: &SearchWorkload) {
    let rxgraph = workload.rxgraph();
    let petgraph = workload.petgraph();
    let serial = rxgraph.search(workload.traversal(false)).unwrap();
    let parallel = rxgraph.search(workload.traversal(true)).unwrap();
    let rust = rxgraph.search_rust(workload.rust_traversal()).unwrap();
    let pet = petgraph_search(&petgraph, workload);

    assert_eq!(rxgraph.node_count(), petgraph.node_count());
    assert_eq!(rxgraph.edge_count(), petgraph.edge_count());
    assert_eq!(serial.paths.len(), workload.starts.len());
    assert_eq!(parallel.paths.len(), serial.paths.len());
    assert_eq!(rust.paths.len(), serial.paths.len());
    assert_eq!(pet.paths.len(), serial.paths.len());

    let mut rx_lengths = serial
        .paths
        .iter()
        .map(|path| path.nodes.len())
        .collect::<Vec<_>>();
    let mut rust_lengths = rust
        .paths
        .iter()
        .map(|path| path.nodes.len())
        .collect::<Vec<_>>();
    let mut pet_lengths = pet.paths.iter().map(Vec::len).collect::<Vec<_>>();
    rx_lengths.sort_unstable();
    rust_lengths.sort_unstable();
    pet_lengths.sort_unstable();
    assert_eq!(rx_lengths, pet_lengths);
    assert_eq!(rx_lengths, rust_lengths);
    assert_eq!(rust.stats.accepted_edges, serial.stats.accepted_edges);
    assert_eq!(pet.accepted_edges, serial.stats.accepted_edges);
    assert!(pet.evaluated_edges >= pet.accepted_edges);
}

fn batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}
