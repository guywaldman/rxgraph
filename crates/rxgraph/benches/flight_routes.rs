use std::{hint::black_box, time::Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use rxgraph::TraversalStrategy;

mod flight_routes {
    #![allow(dead_code)]

    include!("../examples/flight_routes.rs");

    fn workload() -> Workload {
        Workload {
            airports: 25_000,
            max_paths: 1_000,
            decoys: 2_048,
            branches: 512,
        }
    }

    pub(super) fn graph() -> rxgraph::Graph {
        workload().graph()
    }

    pub(super) fn traversal(
        strategy: rxgraph::TraversalStrategy,
        parallel: bool,
    ) -> rxgraph::TraversalConfig {
        workload().traversal(strategy, parallel)
    }
}

fn bench_flight_routes(c: &mut Criterion) {
    let started = Instant::now();
    let graph = flight_routes::graph();
    let build = started.elapsed();
    let result = graph
        .search(flight_routes::traversal(
            TraversalStrategy::DepthFirst,
            true,
        ))
        .unwrap();

    eprintln!(
        "rxgraph flight_routes setup: nodes={} edges={} build={build:?} paths={} evaluated_edges={} accepted_edges={} rejected_edges={} stopped_paths={}",
        graph.node_count(),
        graph.edge_count(),
        result.paths.len(),
        result.stats.evaluated_edges,
        result.stats.accepted_edges,
        result.stats.rejected_edges,
        result.stats.stopped_paths,
    );

    c.bench_function("flight_routes/search", |b| {
        b.iter(|| {
            black_box(
                graph
                    .search(flight_routes::traversal(
                        TraversalStrategy::DepthFirst,
                        true,
                    ))
                    .unwrap(),
            )
        })
    });

    c.bench_function("flight_routes/build_graph", |b| {
        b.iter(|| black_box(flight_routes::graph()))
    });
}

criterion_group!(benches, bench_flight_routes);
criterion_main!(benches);
