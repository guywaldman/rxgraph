//! Integration tests for the example native kernel, exercised through the
//! public `rxgraph` API exactly as an external Rust consumer would.

use anyhow::Result;
use arrow::array::record_batch;
use pretty_assertions::assert_eq;
use rxgraph::{
    DslExpr as e, DslKernel, Graph, GraphId, RunOptions, TraversalConfigBuilder, TraversalStrategy,
    Value,
    examples::kernels::{BudgetState, WeightedBudget},
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
