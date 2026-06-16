//! Integration tests for the example native kernel, exercised through the
//! public `rxgraph` API exactly as an external Rust consumer would.

use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, Int64Array, ListArray, RecordBatch, StringArray,
        StructArray, UInt64Array, record_batch,
    },
    datatypes::{DataType, Field as ArrowField, Int64Type, Schema},
};
use parquet::arrow::ArrowWriter;
use pretty_assertions::assert_eq;
use rxgraph::{
    ArrowRow, ArrowStruct, DslExpr as e, DslKernel, Graph, GraphId, OwnedGraphId, ParquetPaths,
    PayloadField, RunOptions, StateRow, TraversalConfigBuilder, TraversalStrategy, TypedKernel,
    Value,
    examples::kernels::{BudgetState, WeightedBudget},
    traversal::native,
};

/// Builds a small weighted graph with u64 IDs:
///
/// ```text
///   a(0) --3--> b(1)         a --10--> c
///   b --4--> d(3)            b --2--> c(2)
///   c --1--> d
/// ```
///
/// Target is `d`(3). Reaching it from `a`:
/// - a->b->d       spends 3 + 4 = 7
/// - a->b->c->d    spends 3 + 2 + 1 = 6
/// - a->c (10) is rejected by any budget < 10.
fn weighted_graph() -> Result<Graph> {
    let nodes = record_batch!((ID_COL, UInt64, [0, 1, 2, 3]))?;
    let edges = record_batch!(
        (ID_COL, UInt64, [10, 11, 12, 13, 14]),
        (EDGE_SRC_COL, UInt64, [0, 0, 1, 2, 1]),
        (EDGE_DEST_COL, UInt64, [1, 2, 3, 3, 2]),
        ("cost", UInt64, [3, 10, 4, 1, 2])
    )?;
    Graph::new(nodes, edges)
}

// Mirror the topology column names used by the crate's own tests.
const ID_COL: &str = "id";
const EDGE_SRC_COL: &str = "src";
const EDGE_DEST_COL: &str = "dest";

/// `(node ids, edge ids, spent)` for each returned path, sorted for stable
/// comparison regardless of traversal order.
fn summarize(result: &rxgraph::SearchResult<'_>) -> Vec<(Vec<u64>, Vec<u64>, u64)> {
    let mut rows = result
        .paths
        .iter()
        .map(|p| {
            (
                p.nodes.iter().map(u64_id).collect::<Vec<_>>(),
                p.edges.iter().map(u64_id).collect::<Vec<_>>(),
                spent(&p.state),
            )
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

fn u64_id(id: &GraphId<'_>) -> u64 {
    match id {
        GraphId::U64(value) => *value,
        GraphId::Str(value) => panic!("expected u64 id, got {value:?}"),
    }
}

fn spent(state: &[(String, Value)]) -> u64 {
    match state.iter().find(|(k, _)| k == "spent").map(|(_, v)| v) {
        Some(Value::U64(value)) => *value,
        other => panic!("expected 'spent' to be U64, got {other:?}"),
    }
}

/// Serial BFS from node `a`(0); serial keeps path order deterministic.
fn run_opts() -> RunOptions {
    RunOptions {
        start_nodes: vec![0_u64.into()],
        strategy: TraversalStrategy::BreadthFirst,
        parallel: false,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// (a) STATIC path: construct WeightedBudget directly.
// ---------------------------------------------------------------------------

#[test]
fn static_kernel_reaches_target_within_budget() -> Result<()> {
    let graph = weighted_graph()?;
    let kernel = WeightedBudget {
        weight_col: "cost".to_string(),
        budget: 8,
        target: 3_u64.into(),
    };

    let result = graph.search_with(kernel, run_opts())?;

    // Both routes to d fit in budget 8: a->b->d (7) and a->b->c->d (6).
    assert_eq!(
        summarize(&result),
        vec![
            (vec![0, 1, 2, 3], vec![10, 14, 13], 6),
            (vec![0, 1, 3], vec![10, 12], 7),
        ]
    );
    assert_eq!(result.stats.stopped_paths, 2);
    Ok(())
}

#[test]
fn static_kernel_initial_state_is_zero() -> Result<()> {
    let graph = weighted_graph()?;
    let kernel = WeightedBudget {
        weight_col: "cost".to_string(),
        budget: 8,
        target: 3_u64.into(),
    };
    // initial_state ignores graph/start and starts at 0.
    use rxgraph::Kernel;
    assert_eq!(kernel.initial_state(&graph, 0), BudgetState { spent: 0 });
    Ok(())
}

#[test]
fn static_kernel_budget_too_small_yields_no_paths() -> Result<()> {
    let graph = weighted_graph()?;
    let kernel = WeightedBudget {
        weight_col: "cost".to_string(),
        // 5 can't reach d: a->b->d needs 7, a->b->c->d needs 6.
        budget: 5,
        target: 3_u64.into(),
    };

    let result = graph.search_with(kernel, run_opts())?;

    assert_eq!(summarize(&result), Vec::<(Vec<u64>, Vec<u64>, u64)>::new());
    assert_eq!(result.stats.stopped_paths, 0);
    Ok(())
}

// ---------------------------------------------------------------------------
// (b) REGISTRY / by-name path: build_kernel("weighted_budget", ..).
// ---------------------------------------------------------------------------

#[test]
fn registry_kernel_matches_static_kernel() -> Result<()> {
    let graph = weighted_graph()?;

    let static_kernel = WeightedBudget {
        weight_col: "cost".to_string(),
        budget: 8,
        target: 3_u64.into(),
    };
    let static_result = graph.search_with(static_kernel, run_opts())?;

    let boxed = rxgraph::build_kernel(
        "weighted_budget",
        &serde_json::json!({ "weight_col": "cost", "budget": 8, "target": 3 }),
    )?;
    let registry_result = boxed.run(&graph, run_opts())?;

    assert_eq!(summarize(&registry_result), summarize(&static_result));
    Ok(())
}

#[test]
fn registry_kernel_accepts_string_target_on_string_graph() -> Result<()> {
    // String-ID variant to prove target parsing handles JSON strings too.
    let nodes = record_batch!((ID_COL, Utf8, ["a", "b", "d"]))?;
    let edges = record_batch!(
        (ID_COL, Utf8, ["ab", "bd"]),
        (EDGE_SRC_COL, Utf8, ["a", "b"]),
        (EDGE_DEST_COL, Utf8, ["b", "d"]),
        ("cost", UInt64, [2, 2])
    )?;
    let graph = Graph::new(nodes, edges)?;

    let boxed = rxgraph::build_kernel(
        "weighted_budget",
        &serde_json::json!({ "weight_col": "cost", "budget": 10, "target": "d" }),
    )?;
    let result = boxed.run(
        &graph,
        RunOptions {
            start_nodes: vec!["a".into()],
            strategy: TraversalStrategy::BreadthFirst,
            parallel: false,
            ..Default::default()
        },
    )?;

    assert_eq!(result.paths.len(), 1);
    assert_eq!(
        result.paths[0].nodes,
        vec![GraphId::Str("a"), GraphId::Str("b"), GraphId::Str("d")]
    );
    assert_eq!(spent(&result.paths[0].state), 4);
    Ok(())
}

// ---------------------------------------------------------------------------
// (c) Equivalence to an equivalent DSL kernel.
// ---------------------------------------------------------------------------

#[test]
fn dsl_equivalent_kernel_matches_native_node_sequences() -> Result<()> {
    let graph = weighted_graph()?;

    let native = WeightedBudget {
        weight_col: "cost".to_string(),
        budget: 8,
        target: 3_u64.into(),
    };
    let native_result = graph.search_with(native, run_opts())?;

    // visit: spent + cost <= budget   (budget held as a literal here)
    // next_state: spent = spent + cost
    // stop: dest_id == target
    let dsl = DslKernel::new(
        e::state("spent").plus(e::edge("cost")).le(e::uint_lit(8)),
        [("spent".into(), e::state("spent").plus(e::edge("cost")))],
        e::dest_id().eq(e::uint_lit(3)),
        [("spent".into(), Value::U64(0))],
    );
    let dsl_result = graph.search(
        TraversalConfigBuilder::new(dsl)
            .with_start_nodes([0_u64])
            .with_strategy(TraversalStrategy::BreadthFirst)
            .with_parallelism(false)
            .build(),
    )?;

    assert_eq!(summarize(&dsl_result), summarize(&native_result));
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct RowMeta {
    name: String,
    score: Option<i64>,
}

impl TryFrom<ArrowStruct<'_>> for RowMeta {
    type Error = anyhow::Error;

    fn try_from(row: ArrowStruct<'_>) -> Result<Self> {
        Ok(Self {
            name: row.string("name")?.context("meta.name is required")?,
            score: row.i64("score")?,
        })
    }
}

#[test]
fn arrow_row_reads_nested_values() -> Result<()> {
    let items =
        ListArray::from_iter_primitive::<Int64Type, _, _>(vec![Some(vec![Some(1), Some(2)]), None]);
    let meta = StructArray::from(vec![
        (
            Arc::new(ArrowField::new("name", DataType::Utf8, true)),
            Arc::new(StringArray::from(vec![Some("a"), Some("b")])) as ArrayRef,
        ),
        (
            Arc::new(ArrowField::new("score", DataType::Int64, true)),
            Arc::new(Int64Array::from(vec![Some(7), None])) as ArrayRef,
        ),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            ArrowField::new("items", items.data_type().clone(), true),
            ArrowField::new("meta", meta.data_type().clone(), true),
        ])),
        vec![Arc::new(items), Arc::new(meta)],
    )?;

    let row = ArrowRow::new(&batch, 0);
    let item_list = row.list_items("items")?.context("items is required")?;
    assert_eq!(item_list.len(), 2);
    assert_eq!(item_list.u64(0)?, Some(1));
    assert_eq!(item_list.u64(1)?, Some(2));
    assert_eq!(item_list.values()?, vec![Value::I64(1), Value::I64(2)]);
    assert_eq!(
        row.struct_as::<RowMeta>("meta")?,
        Some(RowMeta {
            name: "a".to_string(),
            score: Some(7),
        })
    );
    assert_eq!(
        row.value("items")?,
        Value::List(vec![Value::I64(1), Value::I64(2)])
    );
    assert_eq!(row.list("items")?, Some(vec![Value::I64(1), Value::I64(2)]));
    assert_eq!(
        row.struct_fields("meta")?,
        Some(vec![
            ("name".to_string(), Value::Str(Arc::from("a"))),
            ("score".to_string(), Value::I64(7)),
        ])
    );

    let row = ArrowRow::new(&batch, 1);
    assert_eq!(row.list("items")?, None);
    assert!(row.list_items("items")?.is_none());
    assert_eq!(
        row.struct_as::<RowMeta>("meta")?,
        Some(RowMeta {
            name: "b".to_string(),
            score: None,
        })
    );
    assert_eq!(
        row.struct_fields("meta")?,
        Some(vec![
            ("name".to_string(), Value::Str(Arc::from("b"))),
            ("score".to_string(), Value::Null),
        ])
    );
    Ok(())
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CompoundState {
    spent: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompoundPolicy {
    enabled: bool,
    limit: u64,
}

impl TryFrom<ArrowStruct<'_>> for CompoundPolicy {
    type Error = anyhow::Error;

    fn try_from(row: ArrowStruct<'_>) -> Result<Self> {
        Ok(Self {
            enabled: row.bool("enabled")?.unwrap_or(false),
            limit: row.u64("limit")?.unwrap_or(0),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompoundEdge {
    charges: Vec<u64>,
    policy: CompoundPolicy,
}

impl CompoundEdge {
    fn total_charge(&self) -> u64 {
        self.charges.iter().sum()
    }
}

impl TryFrom<ArrowRow<'_>> for CompoundEdge {
    type Error = anyhow::Error;

    fn try_from(row: ArrowRow<'_>) -> Result<Self> {
        let charges = row.list_items("charges")?.context("charges is required")?;
        let charges = (0..charges.len())
            .map(|index| {
                charges
                    .u64(index)?
                    .with_context(|| format!("charges[{index}] cannot be null"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            charges,
            policy: row.struct_as("policy")?.context("policy is required")?,
        })
    }
}

#[derive(Clone, Debug)]
struct CompoundBudget;

impl TypedKernel for CompoundBudget {
    type Node = ();
    type Edge = CompoundEdge;
    type State = CompoundState;

    fn edge_fields(&self) -> Vec<PayloadField> {
        vec![PayloadField::new("charges"), PayloadField::new("policy")]
    }

    fn initial_state(
        &self,
        _cx: &native::StartCtx<'_, Self::Node, Self::Edge>,
    ) -> Result<Self::State> {
        Ok(CompoundState::default())
    }

    fn visit(
        &self,
        cx: &native::EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<bool> {
        let edge = cx.edge()?;
        let next_spent = cx.state().spent.saturating_add(edge.total_charge());
        Ok(edge.policy.enabled && next_spent <= edge.policy.limit)
    }

    fn next_state(
        &self,
        cx: &native::EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<Self::State> {
        Ok(CompoundState {
            spent: cx.state().spent.saturating_add(cx.edge()?.total_charge()),
        })
    }

    fn stop(
        &self,
        cx: &native::EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<bool> {
        Ok(cx.dest_external_id()? == Some(GraphId::U64(2)))
    }

    fn state_row(&self, state: &Self::State) -> StateRow {
        vec![("spent".to_string(), Value::U64(state.spent))]
    }
}

fn compound_graph_batches() -> Result<(RecordBatch, RecordBatch)> {
    let nodes = record_batch!((ID_COL, UInt64, [0, 1, 2, 3]))?;
    let charges = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
        Some(vec![Some(2), Some(3)]),
        Some(vec![Some(4)]),
        Some(vec![Some(9), Some(9)]),
        Some(vec![Some(1)]),
    ]);
    let policy = StructArray::from(vec![
        (
            Arc::new(ArrowField::new("enabled", DataType::Boolean, true)),
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(true),
                Some(true),
                Some(false),
            ])) as ArrayRef,
        ),
        (
            Arc::new(ArrowField::new("limit", DataType::Int64, true)),
            Arc::new(Int64Array::from(vec![
                Some(10),
                Some(10),
                Some(10),
                Some(10),
            ])) as ArrayRef,
        ),
    ]);
    let edges = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            ArrowField::new(ID_COL, DataType::UInt64, false),
            ArrowField::new(EDGE_SRC_COL, DataType::UInt64, false),
            ArrowField::new(EDGE_DEST_COL, DataType::UInt64, false),
            ArrowField::new("charges", charges.data_type().clone(), true),
            ArrowField::new("policy", policy.data_type().clone(), true),
        ])),
        vec![
            Arc::new(UInt64Array::from(vec![10, 11, 12, 13])),
            Arc::new(UInt64Array::from(vec![0, 1, 0, 0])),
            Arc::new(UInt64Array::from(vec![1, 2, 2, 3])),
            Arc::new(charges),
            Arc::new(policy),
        ],
    )?;
    Ok((nodes, edges))
}

#[test]
fn typed_kernel_decodes_compound_edge_payloads() -> Result<()> {
    let (nodes, edges) = compound_graph_batches()?;
    let graph = Graph::new(nodes, edges)?;

    let result = rxgraph::boxed_typed_run(CompoundBudget).run_eager(&graph, run_opts())?;

    assert_eq!(result.paths.len(), 1);
    assert_eq!(
        result.paths[0].nodes,
        vec![
            OwnedGraphId::U64(0),
            OwnedGraphId::U64(1),
            OwnedGraphId::U64(2),
        ]
    );
    assert_eq!(
        result.paths[0].edges,
        vec![OwnedGraphId::U64(10), OwnedGraphId::U64(11)]
    );
    assert_eq!(spent(&result.paths[0].state), 9);
    Ok(())
}

#[test]
fn typed_kernel_decodes_compound_parquet_payloads() -> Result<()> {
    let (nodes, edges) = compound_graph_batches()?;
    let nodes_path = temp_parquet_path("compound-nodes");
    let edges_path = temp_parquet_path("compound-edges");
    write_parquet(&nodes_path, &nodes)?;
    write_parquet(&edges_path, &edges)?;

    let eager_graph = Graph::from_parquet(nodes_path.clone(), edges_path.clone())?;
    let eager = rxgraph::boxed_typed_run(CompoundBudget).run_eager(&eager_graph, run_opts())?;

    let lazy_graph = Graph::from_parquet_topology(nodes_path.clone(), edges_path.clone())?;
    let lazy = rxgraph::boxed_typed_run(CompoundBudget).run_parquet_lazy(
        &lazy_graph,
        ParquetPaths {
            nodes: nodes_path.clone(),
            edges: edges_path.clone(),
        },
        run_opts(),
    )?;

    let _ = std::fs::remove_file(nodes_path);
    let _ = std::fs::remove_file(edges_path);

    assert_eq!(eager.paths.len(), 1);
    assert_eq!(lazy.paths.len(), 1);
    assert_eq!(eager.paths[0].nodes, lazy.paths[0].nodes);
    assert_eq!(eager.paths[0].edges, lazy.paths[0].edges);
    assert_eq!(spent(&eager.paths[0].state), 9);
    assert_eq!(spent(&lazy.paths[0].state), 9);
    Ok(())
}

fn write_parquet(path: &Path, batch: &RecordBatch) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}

fn temp_parquet_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before UNIX_EPOCH")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "rxgraph-{name}-{}-{nanos}.parquet",
        std::process::id()
    ))
}

// ---------------------------------------------------------------------------
// (d) Error paths.
// ---------------------------------------------------------------------------

// `BoxedRun` is not `Debug`, so inspect the error via `Result::err`.
fn build_err(name: &str, params: serde_json::Value) -> String {
    match rxgraph::build_kernel(name, &params) {
        Ok(_) => panic!("expected build_kernel({name:?}) to fail"),
        Err(err) => err.to_string(),
    }
}

#[test]
fn unknown_kernel_name_errors() {
    let err = build_err("does_not_exist", serde_json::json!({}));
    assert!(err.contains("unknown kernel"), "got: {err}");
}

#[test]
fn bad_params_error() {
    let err = build_err(
        "weighted_budget",
        serde_json::json!({ "weight_col": "cost", "budget": "oops", "target": 3 }),
    );
    assert!(err.contains("budget"), "got: {err}");
}
