use std::{hint::black_box, sync::Arc, time::Instant};

use arrow::{
    array::{ArrayRef, BooleanArray, Int32Array, StringArray, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use criterion::{Criterion, criterion_group, criterion_main};
use rxgraph::{
    DslExpr as e, DslKernel, Graph, TraversalConfig, TraversalConfigBuilder, TraversalStrategy,
    Value,
};

fn bench_payment_risk(c: &mut Criterion) {
    let workload = Workload::new(500_000);

    let started = Instant::now();
    let graph = workload.graph();
    eprintln!(
        "rxgraph payment_risk setup: accounts={} transactions={} cases={} depth={} fanout={} build={:?}",
        workload.accounts(),
        workload.transactions(),
        workload.cases(),
        workload.depth(),
        workload.fanout(),
        started.elapsed(),
    );

    let result = graph.search(workload.traversal(true)).unwrap();
    eprintln!(
        "rxgraph payment_risk warmup: paths={} evaluated_edges={} accepted_edges={} rejected_edges={} skipped_revisits={}",
        result.paths.len(),
        result.stats.evaluated_edges,
        result.stats.accepted_edges,
        result.stats.rejected_edges,
        result.stats.skipped_revisits,
    );

    c.bench_function("payment_risk/search_serial", |b| {
        b.iter(|| black_box(graph.search(workload.traversal(false)).unwrap()))
    });

    c.bench_function("payment_risk/search_parallel", |b| {
        b.iter(|| black_box(graph.search(workload.traversal(true)).unwrap()))
    });

    c.bench_function("payment_risk/build_graph", |b| {
        b.iter(|| black_box(workload.graph()))
    });
}

criterion_group!(benches, bench_payment_risk);
criterion_main!(benches);

#[derive(Debug, Clone, Copy)]
struct Workload {
    scale: usize,
}

impl Workload {
    fn new(scale: usize) -> Self {
        Self {
            scale: scale.max(128),
        }
    }

    fn graph(self) -> Graph {
        Graph::new(self.accounts_table(), self.transactions_table()).unwrap()
    }

    fn traversal(self, parallel: bool) -> TraversalConfig {
        TraversalConfigBuilder::new(self.kernel())
            .with_start_nodes((0..self.cases()).map(|case| account_id(self.chain(case)[0])))
            .with_max_depth(self.depth())
            .with_max_paths(self.cases())
            .with_strategy(TraversalStrategy::BreadthFirst)
            .with_parallelism(parallel)
            .build()
    }

    fn kernel(self) -> DslKernel {
        let visit = e::edge("kind")
            .ne(e::string_lit("noise"))
            .and(e::edge("flagged").eq(e::bool_lit(false)))
            .and(e::edge("success").eq(e::bool_lit(true)))
            .and(e::dest("frozen").eq(e::bool_lit(false)))
            .and(e::state("hops").lt(e::int_lit(self.depth() as i64)))
            .and(
                e::state("amount")
                    .plus(e::edge("amount"))
                    .le(e::int_lit(self.limit() as i64)),
            )
            .and(e::edge("time").ge(e::state("time")))
            .and(e::state("risk").plus(e::dest("risk")).le(e::int_lit(85)));
        let stop = e::dest("segment").eq(e::string_lit("cashout"));

        DslKernel::new(
            visit,
            [
                (
                    "amount".to_string(),
                    e::state("amount").plus(e::edge("amount")),
                ),
                ("hops".to_string(), e::state("hops").plus(e::int_lit(1))),
                ("time".to_string(), e::edge("time").plus(e::int_lit(1))),
                ("risk".to_string(), e::state("risk").plus(e::dest("risk"))),
            ],
            stop,
            [
                ("amount".to_string(), Value::U64(0)),
                ("hops".to_string(), Value::U64(0)),
                ("time".to_string(), Value::U64(0)),
                ("risk".to_string(), Value::I64(0)),
            ],
        )
    }

    fn accounts_table(self) -> RecordBatch {
        let count = self.accounts();
        let ids = (0..count).map(account_id).collect::<Vec<_>>();
        let mut segment = vec!["retail"; count];
        let mut risk = (0..count)
            .map(|i| ((i * 13) % 40) as i32)
            .collect::<Vec<_>>();
        let mut frozen = (0..count)
            .map(|i| i % self.frozen_every() == 0)
            .collect::<Vec<_>>();

        for case in 0..self.cases() {
            let chain = self.chain(case);
            for &account in &chain {
                segment[account] = "mule";
                risk[account] = 2;
                frozen[account] = false;
            }
            segment[*chain.last().unwrap()] = "cashout";
        }

        batch(
            vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("segment", DataType::Utf8, false),
                Field::new("risk", DataType::Int32, false),
                Field::new("frozen", DataType::Boolean, false),
            ],
            vec![
                Arc::new(StringArray::from(ids)) as ArrayRef,
                Arc::new(StringArray::from(segment)),
                Arc::new(Int32Array::from(risk)),
                Arc::new(BooleanArray::from(frozen)),
            ],
        )
    }

    fn transactions_table(self) -> RecordBatch {
        let mut edges = Edges::default();

        for case in 0..self.cases() {
            let chain = self.chain(case);

            for (hop, pair) in chain.windows(2).enumerate() {
                edges.push(
                    pair[0],
                    pair[1],
                    140 + hop as u64,
                    hop as u64 * 10,
                    "transfer",
                    false,
                    true,
                );
            }

            // Wide rejected fanout models irrelevant transaction history and gives parallel BFS enough edge work per layer.
            for &from in chain.iter().take(chain.len() - 1) {
                for n in 0..self.fanout() {
                    let to = (from + n * 97 + case * 131 + 31) % self.accounts();
                    edges.push(
                        from,
                        to,
                        5 + (n % 30) as u64,
                        n as u64,
                        "noise",
                        true,
                        n % 11 != 0,
                    );
                }
            }
        }

        edges.into_batch()
    }

    fn accounts(self) -> usize {
        self.scale
    }

    fn depth(self) -> usize {
        ((usize::BITS - self.scale.leading_zeros()) as usize / 2).clamp(6, 10)
    }

    fn fanout(self) -> usize {
        (self.scale / self.cases() / self.depth() / 4).clamp(64, 512)
    }

    fn transactions(self) -> usize {
        self.cases() * self.depth() * (self.fanout() + 1)
    }

    fn limit(self) -> u64 {
        self.depth() as u64 * 180
    }

    fn frozen_every(self) -> usize {
        (self.scale / 200).max(7)
    }

    fn cases(self) -> usize {
        (self.scale / 4_000).clamp(64, 512)
    }

    fn chain(self, case: usize) -> Vec<usize> {
        let block = self.accounts() / self.cases();
        let base = case * block;
        let step = (block / (self.depth() + 1)).max(1);
        (0..=self.depth())
            .map(|i| (base + i * step).min(self.accounts() - 1))
            .collect()
    }
}

#[derive(Default)]
struct Edges {
    id: Vec<String>,
    src: Vec<String>,
    dest: Vec<String>,
    amount: Vec<u64>,
    time: Vec<u64>,
    kind: Vec<&'static str>,
    flagged: Vec<bool>,
    success: Vec<bool>,
}

impl Edges {
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        src: usize,
        dest: usize,
        amount: u64,
        time: u64,
        kind: &'static str,
        flagged: bool,
        success: bool,
    ) {
        self.id.push(format!("tx{:08}", self.id.len()));
        self.src.push(account_id(src));
        self.dest.push(account_id(dest));
        self.amount.push(amount);
        self.time.push(time);
        self.kind.push(kind);
        self.flagged.push(flagged);
        self.success.push(success);
    }

    fn into_batch(self) -> RecordBatch {
        batch(
            vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("src", DataType::Utf8, false),
                Field::new("dest", DataType::Utf8, false),
                Field::new("amount", DataType::UInt64, false),
                Field::new("time", DataType::UInt64, false),
                Field::new("kind", DataType::Utf8, false),
                Field::new("flagged", DataType::Boolean, false),
                Field::new("success", DataType::Boolean, false),
            ],
            vec![
                Arc::new(StringArray::from(self.id)) as ArrayRef,
                Arc::new(StringArray::from(self.src)),
                Arc::new(StringArray::from(self.dest)),
                Arc::new(UInt64Array::from(self.amount)),
                Arc::new(UInt64Array::from(self.time)),
                Arc::new(StringArray::from(self.kind)),
                Arc::new(BooleanArray::from(self.flagged)),
                Arc::new(BooleanArray::from(self.success)),
            ],
        )
    }
}

fn batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn account_id(i: usize) -> String {
    format!("acct{i:08}")
}
