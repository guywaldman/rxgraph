use std::{env, sync::Arc, time::Instant};

use anyhow::{Result, bail};
use arrow::{
    array::{ArrayRef, BooleanArray, Int32Array, StringArray, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use rxgraph::{
    DslKernel as Kernel, DslTraversalBuilder as TraversalBuilder, GraphBuilder, Parallelism,
    Scalar, TraversalStrategy,
};

fn main() -> Result<()> {
    let airport_count = env::args()
        .nth(1)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(10_000)
        .max(2);
    let max_paths = env::args()
        .nth(2)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(8);
    let hub_decoys = env::args()
        .nth(3)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(16_384);
    let hub_branches = env::args()
        .nth(4)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(2_048);
    let strategy = env::args()
        .nth(5)
        .map(|arg| parse_strategy(&arg))
        .transpose()?
        .unwrap_or(TraversalStrategy::DepthFirst);
    let parallelism = env::args()
        .nth(6)
        .map(|arg| parse_parallelism(&arg))
        .transpose()?
        .unwrap_or(Parallelism::Auto);

    let destination = (airport_count - 1) as u64;
    let max_hops = 18usize;
    let budget = budget_for(airport_count);
    let step = (airport_count / max_hops.max(1)).max(1);

    let build_started = Instant::now();
    let airports = airport_table(airport_count, step);
    let flights = flight_table(airport_count, step, hub_decoys, hub_branches);
    let graph = GraphBuilder::new()
        .with_node_table("airport", airports)
        .with_edge_table("flight", flights)
        .build()?;
    let build_elapsed = build_started.elapsed();

    let traversal = TraversalBuilder::new(kernel(destination, budget, max_hops))
        .with_start_nodes([0])
        .with_max_depth(max_hops)
        .with_max_paths(max_paths)
        .with_strategy(strategy)
        .with_parallelism(parallelism)
        .build();

    let search_started = Instant::now();
    let result = graph.search(traversal)?;
    let search_elapsed = search_started.elapsed();

    println!(
        "airports={} flights={} budget={} max_hops={} max_paths={} hub_decoys={} hub_branches={} strategy={strategy:?} parallelism={parallelism:?}",
        graph.node_count(),
        graph.edge_count(),
        budget,
        max_hops,
        max_paths,
        hub_decoys,
        hub_branches
    );
    println!(
        "build={:?} search={:?} visited_entries={} evaluated_edges={} accepted_edges={} rejected_edges={} stopped_paths={} max_depth={}",
        build_elapsed,
        search_elapsed,
        result.stats.visited_path_entries,
        result.stats.evaluated_edges,
        result.stats.accepted_edges,
        result
            .stats
            .evaluated_edges
            .saturating_sub(result.stats.accepted_edges + result.stats.skipped_errors),
        result.stats.stopped_paths,
        result.stats.max_depth
    );

    for (i, path) in result.paths.iter().take(5).enumerate() {
        let state = path.state.as_object().expect("state is an object");
        let spent = state
            .get("spent")
            .and_then(|value| value.as_u64())
            .unwrap_or_default();
        let hops = state
            .get("hops")
            .and_then(|value| value.as_u64())
            .unwrap_or_default();

        println!(
            "path[{i}] hops={} spent={} nodes={}",
            hops,
            spent,
            summarize_nodes(&path.nodes)
        );
    }

    Ok(())
}

fn parse_strategy(value: &str) -> Result<TraversalStrategy> {
    Ok(match value {
        "dfs" => TraversalStrategy::DepthFirst,
        "bfs" => TraversalStrategy::BreadthFirst,
        other => bail!("unknown strategy {other:?}; expected 'dfs' or 'bfs'"),
    })
}

fn parse_parallelism(value: &str) -> Result<Parallelism> {
    Ok(match value {
        "auto" => Parallelism::Auto,
        "off" => Parallelism::Disabled,
        "on" => Parallelism::Enabled {
            min_frontier: 0,
            min_edges: 0,
        },
        other => bail!("unknown parallel mode {other:?}; expected 'auto', 'off', or 'on'"),
    })
}

fn airport_table(count: usize, protected_step: usize) -> RecordBatch {
    let ids = (0..count as u64).collect::<Vec<_>>();
    let codes = (0..count).map(|i| format!("AP{i:06}")).collect::<Vec<_>>();
    let regions = (0..count)
        .map(|i| match i % 5 {
            0 => "NA",
            1 => "EU",
            2 => "APAC",
            3 => "LATAM",
            _ => "MEA",
        })
        .collect::<Vec<_>>();
    let risks = (0..count).map(|i| ((i * 7) % 9) as i32).collect::<Vec<_>>();
    let min_connections = (0..count)
        .map(|i| 35 + ((i * 11) % 50) as u64)
        .collect::<Vec<_>>();
    let closed = (0..count)
        .map(|i| i != 0 && i + 1 != count && i % 23 == 0 && i % protected_step != 0)
        .collect::<Vec<_>>();

    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("code", DataType::Utf8, false),
            Field::new("region", DataType::Utf8, false),
            Field::new("risk", DataType::Int32, false),
            Field::new("min_connection", DataType::UInt64, false),
            Field::new("closed", DataType::Boolean, false),
        ])),
        vec![
            Arc::new(UInt64Array::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(codes)),
            Arc::new(StringArray::from(regions)),
            Arc::new(Int32Array::from(risks)),
            Arc::new(UInt64Array::from(min_connections)),
            Arc::new(BooleanArray::from(closed)),
        ],
    )
    .expect("airport table")
}

fn flight_table(
    count: usize,
    protected_step: usize,
    hub_decoys: usize,
    hub_branches: usize,
) -> RecordBatch {
    let strides = route_strides(count, protected_step);
    let mut src = Vec::new();
    let mut dest = Vec::new();
    let mut carrier = Vec::new();
    let mut price = Vec::new();
    let mut duration = Vec::new();
    let mut departure = Vec::new();
    let mut arrival = Vec::new();
    let mut reliability = Vec::new();
    let mut route_kind = Vec::new();
    let mut detour_cost = Vec::new();

    for from in 0..count.saturating_sub(1) {
        for &stride in &strides {
            let to = (from + stride).min(count - 1);

            if to == from {
                continue;
            }

            let depart = (from as u64 * 120) + ((stride as u64 % 9) * 7);
            let flight_time = 45 + ((stride as u64 * 13 + from as u64) % 240);

            push_flight(
                &mut src,
                &mut dest,
                &mut carrier,
                &mut price,
                &mut duration,
                &mut departure,
                &mut arrival,
                &mut reliability,
                &mut route_kind,
                &mut detour_cost,
                from,
                to,
                depart,
                flight_time,
                25 + ((stride as u64 * 3 + from as u64) % 110),
                if is_main_corridor_stride(stride, count, protected_step) {
                    92
                } else {
                    45 + ((from * 5 + stride * 3) % 20) as i32
                },
                "route",
                0,
            );
        }

        if is_stress_hub(from, count, protected_step) {
            for decoy in 0..hub_decoys {
                let to = 1 + ((from + decoy * 37 + 17) % count.saturating_sub(1));

                if to == from
                    || is_main_corridor_stride(to.saturating_sub(from), count, protected_step)
                {
                    continue;
                }

                let depart = (from as u64 * 120) + (decoy as u64 % 90);
                let flight_time = 60 + (decoy as u64 % 180);

                push_flight(
                    &mut src,
                    &mut dest,
                    &mut carrier,
                    &mut price,
                    &mut duration,
                    &mut departure,
                    &mut arrival,
                    &mut reliability,
                    &mut route_kind,
                    &mut detour_cost,
                    from,
                    to,
                    depart,
                    flight_time,
                    15 + (decoy as u64 % 50),
                    35 + (decoy as i32 % 30),
                    "decoy",
                    0,
                );
            }

            for branch in 0..hub_branches {
                let to = 1 + ((from + branch * 53 + 29) % count.saturating_sub(1));

                if to == from
                    || to + 1 == count
                    || is_main_corridor_stride(to.saturating_sub(from), count, protected_step)
                {
                    continue;
                }

                let depart = (from as u64 * 120) + (branch as u64 % 60);
                let flight_time = 50 + (branch as u64 % 120);

                push_flight(
                    &mut src,
                    &mut dest,
                    &mut carrier,
                    &mut price,
                    &mut duration,
                    &mut departure,
                    &mut arrival,
                    &mut reliability,
                    &mut route_kind,
                    &mut detour_cost,
                    from,
                    to,
                    depart,
                    flight_time,
                    20 + (branch as u64 % 35),
                    95,
                    "branch",
                    1,
                );
            }
        }
    }

    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::UInt64, false),
            Field::new("dest", DataType::UInt64, false),
            Field::new("carrier", DataType::Utf8, false),
            Field::new("price", DataType::UInt64, false),
            Field::new("duration", DataType::UInt64, false),
            Field::new("departure", DataType::UInt64, false),
            Field::new("arrival", DataType::UInt64, false),
            Field::new("reliability", DataType::Int32, false),
            Field::new("route_kind", DataType::Utf8, false),
            Field::new("detour_cost", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(UInt64Array::from(src)) as ArrayRef,
            Arc::new(UInt64Array::from(dest)),
            Arc::new(StringArray::from(carrier)),
            Arc::new(UInt64Array::from(price)),
            Arc::new(UInt64Array::from(duration)),
            Arc::new(UInt64Array::from(departure)),
            Arc::new(UInt64Array::from(arrival)),
            Arc::new(Int32Array::from(reliability)),
            Arc::new(StringArray::from(route_kind)),
            Arc::new(UInt64Array::from(detour_cost)),
        ],
    )
    .expect("flight table")
}

#[allow(clippy::too_many_arguments)]
fn push_flight(
    src: &mut Vec<u64>,
    dest: &mut Vec<u64>,
    carrier: &mut Vec<String>,
    price: &mut Vec<u64>,
    duration: &mut Vec<u64>,
    departure: &mut Vec<u64>,
    arrival: &mut Vec<u64>,
    reliability: &mut Vec<i32>,
    route_kind: &mut Vec<&'static str>,
    detour_cost: &mut Vec<u64>,
    from: usize,
    to: usize,
    depart: u64,
    flight_time: u64,
    fare: u64,
    reliability_score: i32,
    kind: &'static str,
    detour: u64,
) {
    src.push(from as u64);
    dest.push(to as u64);
    carrier.push(format!("C{:02}", (from + to) % 17));
    price.push(fare);
    duration.push(flight_time);
    departure.push(depart);
    arrival.push(depart + flight_time);
    reliability.push(reliability_score);
    route_kind.push(kind);
    detour_cost.push(detour);
}

fn route_strides(count: usize, protected_step: usize) -> Vec<usize> {
    let mut strides = vec![
        1,
        2,
        3,
        5,
        8,
        13,
        21,
        protected_step.saturating_sub(1).max(1),
        protected_step,
        protected_step + 1,
        (count / 7).max(1),
        (count / 5).max(1),
    ];

    strides.sort_unstable();
    strides.dedup();
    strides
}

fn is_main_corridor_stride(stride: usize, count: usize, protected_step: usize) -> bool {
    stride == protected_step
        || stride == protected_step + 1
        || stride == (count / 5).max(1)
        || stride == (count / 7).max(1)
}

fn is_stress_hub(airport: usize, count: usize, protected_step: usize) -> bool {
    airport % protected_step == 0
        || airport % (count / 5).max(1) == 0
        || airport % (count / 7).max(1) == 0
}

fn kernel(destination: u64, budget: u64, max_hops: usize) -> Kernel {
    let visit = and_all([
        binary(col("state.detours"), "Eq", int(0)),
        not(col("dest.closed")),
        binary(col("edge.reliability"), "GtEq", int(70)),
        binary(col("edge.route_kind"), "NotEq", string("decoy")),
        binary(col("state.hops"), "Lt", int(max_hops as i64)),
        binary(
            binary(col("state.spent"), "Plus", col("edge.price")),
            "LtEq",
            int(budget as i64),
        ),
        binary(col("edge.departure"), "GtEq", col("state.ready_at")),
        binary(
            binary(col("state.risk"), "Plus", col("dest.risk")),
            "LtEq",
            int(90),
        ),
    ]);
    let stop = binary(col("dest.id"), "Eq", int(destination as i64));

    Kernel::new(
        &visit,
        [
            (
                "spent".to_string(),
                binary(col("state.spent"), "Plus", col("edge.price")),
            ),
            (
                "hops".to_string(),
                binary(col("state.hops"), "Plus", int(1)),
            ),
            (
                "ready_at".to_string(),
                binary(col("edge.arrival"), "Plus", col("dest.min_connection")),
            ),
            (
                "risk".to_string(),
                binary(col("state.risk"), "Plus", col("dest.risk")),
            ),
            (
                "detours".to_string(),
                binary(col("state.detours"), "Plus", col("edge.detour_cost")),
            ),
        ],
        &stop,
        [
            ("spent".to_string(), Scalar::U64(0)),
            ("hops".to_string(), Scalar::U64(0)),
            ("ready_at".to_string(), Scalar::U64(0)),
            ("risk".to_string(), Scalar::I64(0)),
            ("detours".to_string(), Scalar::U64(0)),
        ],
    )
    .unwrap()
}

fn and_all(exprs: impl IntoIterator<Item = String>) -> String {
    exprs
        .into_iter()
        .reduce(|left, right| binary(left, "And", right))
        .unwrap()
}

fn binary(left: String, op: &str, right: String) -> String {
    format!(r#"{{"BinaryExpr":{{"left":{left},"op":"{op}","right":{right}}}}}"#)
}

fn not(input: String) -> String {
    format!(r#"{{"Function":{{"input":[{input}],"function":{{"Boolean":"Not"}}}}}}"#)
}

fn col(name: &str) -> String {
    format!(r#"{{"Column":"{name}"}}"#)
}

fn int(value: i64) -> String {
    format!(r#"{{"Literal":{{"Dyn":{{"Int":{value}}}}}}}"#)
}

fn string(value: &str) -> String {
    format!(r#"{{"Literal":{{"Scalar":{{"String":"{value}"}}}}}}"#)
}

fn budget_for(count: usize) -> u64 {
    750 + (count as u64 / 500).min(2_500)
}

fn summarize_nodes(nodes: &[u64]) -> String {
    if nodes.len() <= 10 {
        return format!("{nodes:?}");
    }

    let head = nodes
        .iter()
        .take(5)
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let tail = nodes
        .iter()
        .rev()
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");

    format!("[{head}, ... {} more ..., {tail}]", nodes.len() - 8)
}
