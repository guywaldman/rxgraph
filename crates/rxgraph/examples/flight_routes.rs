use std::{env, sync::Arc, time::Instant};

use anyhow::{Result, bail};
use arrow::{
    array::{ArrayRef, BooleanArray, Int32Array, StringArray, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use rxgraph::{
    DslExpr as e, DslKernel, Graph, GraphId, Scalar, TraversalConfigBuilder, TraversalStrategy,
};

fn main() -> Result<()> {
    let airports = arg(1)?.unwrap_or(10_000).max(2);
    let max_paths = arg(2)?.unwrap_or(8);
    let decoys = arg(3)?.unwrap_or(16_384);
    let branches = arg(4)?.unwrap_or(2_048);
    let strategy = env::args()
        .nth(5)
        .map(|arg| parse_strategy(&arg))
        .transpose()?
        .unwrap_or(TraversalStrategy::DepthFirst);
    let parallel = env::args()
        .nth(6)
        .map(|arg| parse_parallel(&arg))
        .transpose()?
        .unwrap_or(true);

    let workload = Workload {
        airports,
        max_paths,
        decoys,
        branches,
    };
    let started = Instant::now();
    let graph = workload.graph();
    let build = started.elapsed();

    let started = Instant::now();
    let result = graph.search(workload.traversal(strategy, parallel))?;
    let search = started.elapsed();

    println!(
        "airports={} flights={} budget={} max_paths={} strategy={strategy:?} parallel={parallel}",
        graph.node_count(),
        graph.edge_count(),
        workload.budget(),
        max_paths,
    );
    println!(
        "build={build:?} search={search:?} paths={} evaluated_edges={} accepted_edges={} rejected_edges={} stopped_paths={} max_depth={}",
        result.paths.len(),
        result.stats.evaluated_edges,
        result.stats.accepted_edges,
        result.stats.rejected_edges,
        result.stats.stopped_paths,
        result.stats.max_depth,
    );

    for (i, path) in result.paths.iter().take(5).enumerate() {
        println!("path[{i}] nodes={}", summarize_nodes(&path.nodes));
    }

    Ok(())
}

fn arg(index: usize) -> Result<Option<usize>> {
    env::args()
        .nth(index)
        .map(|arg| arg.parse::<usize>())
        .transpose()
        .map_err(Into::into)
}

fn parse_strategy(value: &str) -> Result<TraversalStrategy> {
    Ok(match value {
        "dfs" => TraversalStrategy::DepthFirst,
        "bfs" => TraversalStrategy::BreadthFirst,
        other => bail!("unknown strategy {other:?}; expected 'dfs' or 'bfs'"),
    })
}

fn parse_parallel(value: &str) -> Result<bool> {
    Ok(match value {
        "auto" | "on" => true,
        "off" => false,
        other => bail!("unknown parallel mode {other:?}; expected 'auto', 'on', or 'off'"),
    })
}

#[derive(Clone, Copy)]
struct Workload {
    airports: usize,
    max_paths: usize,
    decoys: usize,
    branches: usize,
}

impl Workload {
    fn graph(self) -> Graph {
        Graph::new(self.airports_table(), self.flights_table()).unwrap()
    }

    fn traversal(self, strategy: TraversalStrategy, parallel: bool) -> rxgraph::TraversalConfig {
        TraversalConfigBuilder::new(self.kernel())
            .with_start_nodes([0_u64])
            .with_max_depth(Self::MAX_HOPS)
            .with_max_paths(self.max_paths)
            .with_strategy(strategy)
            .with_parallelism(parallel)
            .build()
    }

    fn kernel(self) -> DslKernel {
        let visit = e::state("detours")
            .eq(e::uint(0))
            .and(e::dest("closed").not())
            .and(e::edge("reliability").ge(e::int(70)))
            .and(e::edge("route_kind").ne(e::string("decoy")))
            .and(e::state("hops").lt(e::uint(Self::MAX_HOPS as u64)))
            .and(
                e::state("spent")
                    .plus(e::edge("price"))
                    .le(e::uint(self.budget())),
            )
            .and(e::edge("departure").ge(e::state("ready_at")))
            .and(e::state("risk").plus(e::dest("risk")).le(e::int(90)));

        DslKernel::new(
            visit,
            [
                ("spent".into(), e::state("spent").plus(e::edge("price"))),
                ("hops".into(), e::state("hops").plus(e::uint(1))),
                (
                    "ready_at".into(),
                    e::edge("arrival").plus(e::dest("min_connection")),
                ),
                ("risk".into(), e::state("risk").plus(e::dest("risk"))),
                (
                    "detours".into(),
                    e::state("detours").plus(e::edge("detour_cost")),
                ),
            ],
            e::dest_id().eq(e::uint((self.airports - 1) as u64)),
            [
                ("spent".into(), Scalar::U64(0)),
                ("hops".into(), Scalar::U64(0)),
                ("ready_at".into(), Scalar::U64(0)),
                ("risk".into(), Scalar::I64(0)),
                ("detours".into(), Scalar::U64(0)),
            ],
        )
    }

    fn airports_table(self) -> RecordBatch {
        let ids = (0..self.airports as u64).collect::<Vec<_>>();
        let codes = (0..self.airports)
            .map(|i| format!("AP{i:06}"))
            .collect::<Vec<_>>();
        let risks = (0..self.airports)
            .map(|i| ((i * 7) % 9) as i32)
            .collect::<Vec<_>>();
        let min_connections = (0..self.airports)
            .map(|i| 35 + ((i * 11) % 50) as u64)
            .collect::<Vec<_>>();
        let closed = (0..self.airports)
            .map(|i| i != 0 && i + 1 != self.airports && i % 23 == 0 && i % self.step() != 0)
            .collect::<Vec<_>>();

        batch(
            vec![
                Field::new("id", DataType::UInt64, false),
                Field::new("code", DataType::Utf8, false),
                Field::new("risk", DataType::Int32, false),
                Field::new("min_connection", DataType::UInt64, false),
                Field::new("closed", DataType::Boolean, false),
            ],
            vec![
                Arc::new(UInt64Array::from(ids)) as ArrayRef,
                Arc::new(StringArray::from(codes)),
                Arc::new(Int32Array::from(risks)),
                Arc::new(UInt64Array::from(min_connections)),
                Arc::new(BooleanArray::from(closed)),
            ],
        )
    }

    fn flights_table(self) -> RecordBatch {
        let mut flights = Flights::default();
        for from in 0..self.airports.saturating_sub(1) {
            for stride in self.strides() {
                let to = (from + stride).min(self.airports - 1);
                if to != from {
                    flights.push(
                        from,
                        to,
                        25 + ((stride as u64 * 3 + from as u64) % 110),
                        if self.is_corridor(stride) {
                            92
                        } else {
                            45 + ((from * 5 + stride * 3) % 20) as i32
                        },
                        "route",
                        0,
                    );
                }
            }

            if self.is_hub(from) {
                for n in 0..self.decoys {
                    let to = 1 + ((from + n * 37 + 17) % self.airports.saturating_sub(1));
                    if to != from {
                        flights.push(
                            from,
                            to,
                            15 + (n as u64 % 50),
                            35 + (n as i32 % 30),
                            "decoy",
                            0,
                        );
                    }
                }
                for n in 0..self.branches {
                    let to = 1 + ((from + n * 53 + 29) % self.airports.saturating_sub(1));
                    if to != from && to + 1 != self.airports {
                        flights.push(from, to, 20 + (n as u64 % 35), 95, "branch", 1);
                    }
                }
            }
        }
        flights.into_batch()
    }

    const MAX_HOPS: usize = 18;

    fn step(self) -> usize {
        (self.airports / Self::MAX_HOPS).max(1)
    }

    fn budget(self) -> u64 {
        750 + (self.airports as u64 / 500).min(2_500)
    }

    fn strides(self) -> Vec<usize> {
        let mut strides = vec![
            1,
            2,
            3,
            5,
            8,
            13,
            21,
            self.step().saturating_sub(1).max(1),
            self.step(),
            self.step() + 1,
            (self.airports / 7).max(1),
            (self.airports / 5).max(1),
        ];
        strides.sort_unstable();
        strides.dedup();
        strides
    }

    fn is_corridor(self, stride: usize) -> bool {
        stride == self.step()
            || stride == self.step() + 1
            || stride == (self.airports / 5).max(1)
            || stride == (self.airports / 7).max(1)
    }

    fn is_hub(self, airport: usize) -> bool {
        airport % self.step() == 0
            || airport % (self.airports / 5).max(1) == 0
            || airport % (self.airports / 7).max(1) == 0
    }
}

#[derive(Default)]
struct Flights {
    src: Vec<u64>,
    dest: Vec<u64>,
    price: Vec<u64>,
    departure: Vec<u64>,
    arrival: Vec<u64>,
    reliability: Vec<i32>,
    route_kind: Vec<&'static str>,
    detour_cost: Vec<u64>,
}

impl Flights {
    fn push(
        &mut self,
        from: usize,
        to: usize,
        fare: u64,
        reliability: i32,
        kind: &'static str,
        detour: u64,
    ) {
        let depart = (from as u64 * 120) + ((to as u64 % 9) * 7);
        let flight_time = 45 + ((to as u64 * 13 + from as u64) % 240);
        self.src.push(from as u64);
        self.dest.push(to as u64);
        self.price.push(fare);
        self.departure.push(depart);
        self.arrival.push(depart + flight_time);
        self.reliability.push(reliability);
        self.route_kind.push(kind);
        self.detour_cost.push(detour);
    }

    fn into_batch(self) -> RecordBatch {
        let ids = (0..self.src.len() as u64).collect::<Vec<_>>();
        batch(
            vec![
                Field::new("id", DataType::UInt64, false),
                Field::new("src", DataType::UInt64, false),
                Field::new("dest", DataType::UInt64, false),
                Field::new("price", DataType::UInt64, false),
                Field::new("departure", DataType::UInt64, false),
                Field::new("arrival", DataType::UInt64, false),
                Field::new("reliability", DataType::Int32, false),
                Field::new("route_kind", DataType::Utf8, false),
                Field::new("detour_cost", DataType::UInt64, false),
            ],
            vec![
                Arc::new(UInt64Array::from(ids)) as ArrayRef,
                Arc::new(UInt64Array::from(self.src)),
                Arc::new(UInt64Array::from(self.dest)),
                Arc::new(UInt64Array::from(self.price)),
                Arc::new(UInt64Array::from(self.departure)),
                Arc::new(UInt64Array::from(self.arrival)),
                Arc::new(Int32Array::from(self.reliability)),
                Arc::new(StringArray::from(self.route_kind)),
                Arc::new(UInt64Array::from(self.detour_cost)),
            ],
        )
    }
}

fn batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn summarize_nodes(nodes: &[GraphId<'_>]) -> String {
    let ids = nodes
        .iter()
        .map(|id| match id {
            GraphId::U64(value) => value.to_string(),
            GraphId::Str(value) => value.to_string(),
        })
        .collect::<Vec<_>>();
    if ids.len() <= 10 {
        return format!("[{}]", ids.join(", "));
    }
    format!(
        "[{}, ... {} more ..., {}]",
        ids.iter().take(5).cloned().collect::<Vec<_>>().join(", "),
        ids.len() - 8,
        ids.iter()
            .skip(ids.len() - 3)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    )
}
