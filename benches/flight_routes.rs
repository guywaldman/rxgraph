use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

use arrow::{
    array::{ArrayRef, BooleanArray, Int32Array, StringArray, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use criterion::{Criterion, criterion_group, criterion_main};
use rxgraph::{
    DslKernel as Kernel, DslTraversal as Traversal, DslTraversalBuilder as TraversalBuilder,
    GraphBuilder, Scalar, TraversalStrategy,
};

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static BASELINE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };

        if !ptr.is_null() {
            record_alloc(layout.size());
        }

        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, old_layout: Layout, new_size: usize) -> *mut u8 {
        let ptr = unsafe { System.realloc(ptr, old_layout, new_size) };

        if !ptr.is_null() {
            if new_size >= old_layout.size() {
                record_alloc(new_size - old_layout.size());
            } else {
                CURRENT_BYTES.fetch_sub(old_layout.size() - new_size, Ordering::Relaxed);
            }
        }

        ptr
    }
}

fn record_alloc(bytes: usize) {
    let current = CURRENT_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    let mut peak = PEAK_BYTES.load(Ordering::Relaxed);

    while current > peak {
        match PEAK_BYTES.compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(next) => peak = next,
        }
    }
}

fn reset_peak() {
    let current = CURRENT_BYTES.load(Ordering::Relaxed);

    BASELINE_BYTES.store(current, Ordering::Relaxed);
    PEAK_BYTES.store(current, Ordering::Relaxed);
}

fn peak_delta() -> usize {
    PEAK_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(BASELINE_BYTES.load(Ordering::Relaxed))
}

fn bench_flight_routes(c: &mut Criterion) {
    let workload = Workload::new(25_000, 2_048, 512, 1_000);

    reset_peak();
    let build_started = Instant::now();
    let graph = workload.graph();
    let build_elapsed = build_started.elapsed();
    let build_peak = peak_delta();

    reset_peak();
    let result = graph.search(workload.traversal()).unwrap();
    let search_peak = peak_delta();

    eprintln!(
        "rxgraph flight_routes setup: nodes={} edges={} build={:?} build_peak={} search_peak={} evaluated_edges={} accepted_edges={} stopped_paths={}",
        graph.node_count(),
        graph.edge_count(),
        build_elapsed,
        format_bytes(build_peak),
        format_bytes(search_peak),
        result.stats.evaluated_edges,
        result.stats.accepted_edges,
        result.stats.stopped_paths,
    );

    c.bench_function("flight_routes/search", |b| {
        b.iter(|| {
            let result = graph.search(workload.traversal()).unwrap();
            std::hint::black_box(result);
        })
    });

    c.bench_function("flight_routes/build_graph", |b| {
        b.iter(|| {
            let graph = workload.graph();
            std::hint::black_box(graph);
        })
    });
}

criterion_group!(benches, bench_flight_routes);
criterion_main!(benches);

struct Workload {
    airport_count: usize,
    hub_decoys: usize,
    hub_branches: usize,
    max_paths: usize,
}

impl Workload {
    fn new(airport_count: usize, hub_decoys: usize, hub_branches: usize, max_paths: usize) -> Self {
        Self {
            airport_count,
            hub_decoys,
            hub_branches,
            max_paths,
        }
    }

    fn graph(&self) -> rxgraph::Graph {
        let step = self.step();

        GraphBuilder::new()
            .with_node_table("airport", airport_table(self.airport_count, step))
            .with_edge_table(
                "flight",
                flight_table(self.airport_count, step, self.hub_decoys, self.hub_branches),
            )
            .build()
            .unwrap()
    }

    fn traversal(&self) -> Traversal {
        TraversalBuilder::new(kernel(
            self.destination(),
            budget_for(self.airport_count),
            18,
        ))
        .with_start_nodes([0])
        .with_max_depth(18)
        .with_max_paths(self.max_paths)
        .with_strategy(TraversalStrategy::DepthFirst)
        .build()
    }

    fn step(&self) -> usize {
        (self.airport_count / 18).max(1)
    }

    fn destination(&self) -> u64 {
        (self.airport_count - 1) as u64
    }
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

fn airport_table(count: usize, protected_step: usize) -> RecordBatch {
    let ids = (0..count as u64).collect::<Vec<_>>();
    let codes = (0..count).map(|i| format!("AP{i:06}")).collect::<Vec<_>>();
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
            Field::new("risk", DataType::Int32, false),
            Field::new("min_connection", DataType::UInt64, false),
            Field::new("closed", DataType::Boolean, false),
        ])),
        vec![
            Arc::new(UInt64Array::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(codes)),
            Arc::new(Int32Array::from(risks)),
            Arc::new(UInt64Array::from(min_connections)),
            Arc::new(BooleanArray::from(closed)),
        ],
    )
    .unwrap()
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
    let mut price = Vec::new();
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

            push_flight(
                &mut src,
                &mut dest,
                &mut price,
                &mut departure,
                &mut arrival,
                &mut reliability,
                &mut route_kind,
                &mut detour_cost,
                from,
                to,
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

                if to == from {
                    continue;
                }

                push_flight(
                    &mut src,
                    &mut dest,
                    &mut price,
                    &mut departure,
                    &mut arrival,
                    &mut reliability,
                    &mut route_kind,
                    &mut detour_cost,
                    from,
                    to,
                    15 + (decoy as u64 % 50),
                    35 + (decoy as i32 % 30),
                    "decoy",
                    0,
                );
            }

            for branch in 0..hub_branches {
                let to = 1 + ((from + branch * 53 + 29) % count.saturating_sub(1));

                if to == from || to + 1 == count {
                    continue;
                }

                push_flight(
                    &mut src,
                    &mut dest,
                    &mut price,
                    &mut departure,
                    &mut arrival,
                    &mut reliability,
                    &mut route_kind,
                    &mut detour_cost,
                    from,
                    to,
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
            Field::new("price", DataType::UInt64, false),
            Field::new("departure", DataType::UInt64, false),
            Field::new("arrival", DataType::UInt64, false),
            Field::new("reliability", DataType::Int32, false),
            Field::new("route_kind", DataType::Utf8, false),
            Field::new("detour_cost", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(UInt64Array::from(src)) as ArrayRef,
            Arc::new(UInt64Array::from(dest)),
            Arc::new(UInt64Array::from(price)),
            Arc::new(UInt64Array::from(departure)),
            Arc::new(UInt64Array::from(arrival)),
            Arc::new(Int32Array::from(reliability)),
            Arc::new(StringArray::from(route_kind)),
            Arc::new(UInt64Array::from(detour_cost)),
        ],
    )
    .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn push_flight(
    src: &mut Vec<u64>,
    dest: &mut Vec<u64>,
    price: &mut Vec<u64>,
    departure: &mut Vec<u64>,
    arrival: &mut Vec<u64>,
    reliability: &mut Vec<i32>,
    route_kind: &mut Vec<&'static str>,
    detour_cost: &mut Vec<u64>,
    from: usize,
    to: usize,
    fare: u64,
    reliability_score: i32,
    kind: &'static str,
    detour: u64,
) {
    let depart = (from as u64 * 120) + ((to as u64 % 9) * 7);
    let flight_time = 45 + ((to as u64 * 13 + from as u64) % 240);

    src.push(from as u64);
    dest.push(to as u64);
    price.push(fare);
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

fn budget_for(count: usize) -> u64 {
    750 + (count as u64 / 500).min(2_500)
}

fn format_bytes(bytes: usize) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    format!("{:.1}MiB", bytes as f64 / MIB)
}
