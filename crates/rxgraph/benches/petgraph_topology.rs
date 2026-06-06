use std::{collections::VecDeque, hint::black_box, time::Instant};

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use petgraph::{
    Direction::{Incoming, Outgoing},
    graph::{DiGraph, NodeIndex},
    visit::Bfs,
};
use rxgraph::Graph;

type PetGraph = DiGraph<(), (), u32>;

fn bench_petgraph_topology(c: &mut Criterion) {
    let workload = TopologyWorkload::new(100_000, 4);
    verify_topology(&workload);

    let started = Instant::now();
    let rxgraph = workload.rxgraph();
    let rx_build = started.elapsed();
    let started = Instant::now();
    let petgraph = workload.petgraph();
    let pet_build = started.elapsed();

    eprintln!(
        "petgraph_topology setup: nodes={} edges={} rxgraph_build={rx_build:?} petgraph_build={pet_build:?}",
        workload.nodes,
        workload.edges.len(),
    );

    let mut group = c.benchmark_group("petgraph_topology");
    group.sample_size(10);

    group.bench_function("rxgraph/build", |b| {
        b.iter(|| black_box(workload.rxgraph()))
    });
    group.bench_function("petgraph/build", |b| {
        b.iter(|| black_box(workload.petgraph()))
    });

    group.bench_function("rxgraph/bfs", |b| {
        b.iter(|| black_box(rxgraph.bfs_u64(0, None).unwrap().unwrap().len()))
    });
    group.bench_function("petgraph/bfs", |b| {
        b.iter(|| black_box(petgraph_bfs(&petgraph, 0).len()))
    });

    group.bench_function("rxgraph/shortest_path", |b| {
        b.iter(|| {
            black_box(
                rxgraph
                    .shortest_path_u64(0, (workload.nodes - 1) as u64)
                    .unwrap()
                    .unwrap()
                    .unwrap()
                    .len(),
            )
        })
    });
    group.bench_function("petgraph/shortest_path", |b| {
        b.iter(|| {
            black_box(
                petgraph_shortest_path(&petgraph, 0, workload.nodes - 1)
                    .unwrap()
                    .len(),
            )
        })
    });

    group.bench_function("rxgraph/degrees", |b| {
        b.iter_batched(
            || workload.rxgraph(),
            |graph| black_box(graph.degrees()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("petgraph/degrees", |b| {
        b.iter(|| black_box(petgraph_degrees(&petgraph)))
    });

    group.bench_function("rxgraph/weak_components", |b| {
        b.iter_batched(
            || workload.rxgraph(),
            |graph| black_box(graph.weakly_connected_components_u64().unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("petgraph/weak_components", |b| {
        b.iter(|| black_box(petgraph_weak_components(&petgraph)))
    });

    group.finish();
}

criterion_group!(benches, bench_petgraph_topology);
criterion_main!(benches);

#[derive(Debug)]
struct TopologyWorkload {
    nodes: usize,
    edges: Vec<(u64, u64)>,
}

impl TopologyWorkload {
    fn new(nodes: usize, fanout: usize) -> Self {
        let nodes = nodes.max(2);
        let mut edges = Vec::with_capacity(nodes.saturating_sub(1) + nodes * fanout);

        for src in 0..nodes - 1 {
            edges.push((src as u64, (src + 1) as u64));
        }

        for src in 0..nodes {
            for n in 0..fanout {
                let dest = (src * 37 + n * 101 + 17) % nodes;
                if dest != src {
                    edges.push((src as u64, dest as u64));
                }
            }
        }

        Self { nodes, edges }
    }

    fn rxgraph(&self) -> Graph {
        Graph::from_u64_edges(self.nodes, self.edges.iter().copied()).unwrap()
    }

    fn petgraph(&self) -> PetGraph {
        let mut graph = PetGraph::with_capacity(self.nodes, self.edges.len());
        for _ in 0..self.nodes {
            graph.add_node(());
        }
        for &(src, dest) in &self.edges {
            graph.add_edge(
                NodeIndex::new(src as usize),
                NodeIndex::new(dest as usize),
                (),
            );
        }
        graph
    }
}

fn verify_topology(workload: &TopologyWorkload) {
    let rxgraph = workload.rxgraph();
    let petgraph = workload.petgraph();

    assert_eq!(rxgraph.node_count(), petgraph.node_count());
    assert_eq!(rxgraph.edge_count(), petgraph.edge_count());
    assert_eq!(
        rxgraph.bfs_u64(0, None).unwrap().unwrap().len(),
        petgraph_bfs(&petgraph, 0).len()
    );
    assert_eq!(
        rxgraph
            .shortest_path_u64(0, (workload.nodes - 1) as u64)
            .unwrap()
            .unwrap()
            .map(|path| path.len()),
        petgraph_shortest_path(&petgraph, 0, workload.nodes - 1).map(|path| path.len())
    );
    assert_eq!(rxgraph.degrees(), petgraph_degrees(&petgraph));

    let mut rx_components = rxgraph
        .weakly_connected_components_u64()
        .unwrap()
        .into_iter()
        .map(|component| component.len())
        .collect::<Vec<_>>();
    let mut pet_components = petgraph_weak_components(&petgraph)
        .into_iter()
        .map(|component| component.len())
        .collect::<Vec<_>>();
    rx_components.sort_unstable();
    pet_components.sort_unstable();
    assert_eq!(rx_components, pet_components);
}

fn petgraph_bfs(graph: &PetGraph, start: usize) -> Vec<usize> {
    let mut bfs = Bfs::new(graph, NodeIndex::new(start));
    let mut nodes = Vec::new();
    while let Some(node) = bfs.next(graph) {
        nodes.push(node.index());
    }
    nodes
}

fn petgraph_shortest_path(graph: &PetGraph, source: usize, target: usize) -> Option<Vec<usize>> {
    if source == target {
        return Some(vec![source]);
    }

    let mut visited = vec![false; graph.node_count()];
    let mut parent = vec![usize::MAX; graph.node_count()];
    let mut frontier = VecDeque::new();

    frontier.push_back(NodeIndex::new(source));
    visited[source] = true;

    while let Some(node) = frontier.pop_front() {
        for dest in graph.neighbors_directed(node, Outgoing) {
            let dest_index = dest.index();
            if visited[dest_index] {
                continue;
            }
            visited[dest_index] = true;
            parent[dest_index] = node.index();
            if dest_index == target {
                let mut path = vec![target];
                let mut cursor = target;
                while cursor != source {
                    cursor = parent[cursor];
                    path.push(cursor);
                }
                path.reverse();
                return Some(path);
            }
            frontier.push_back(dest);
        }
    }

    None
}

fn petgraph_degrees(graph: &PetGraph) -> Vec<usize> {
    (0..graph.node_count())
        .map(|node| {
            let node = NodeIndex::new(node);
            graph.neighbors_directed(node, Outgoing).count()
                + graph.neighbors_directed(node, Incoming).count()
        })
        .collect()
}

fn petgraph_weak_components(graph: &PetGraph) -> Vec<Vec<usize>> {
    let mut visited = vec![false; graph.node_count()];
    let mut components = Vec::new();
    let mut frontier = Vec::new();

    for start in 0..graph.node_count() {
        if visited[start] {
            continue;
        }

        let mut component = Vec::new();
        frontier.clear();
        frontier.push(NodeIndex::new(start));
        visited[start] = true;

        while let Some(node) = frontier.pop() {
            component.push(node.index());
            for next in graph
                .neighbors_directed(node, Outgoing)
                .chain(graph.neighbors_directed(node, Incoming))
            {
                let index = next.index();
                if !visited[index] {
                    visited[index] = true;
                    frontier.push(next);
                }
            }
        }

        components.push(component);
    }

    components
}
