//! Traversal expression DSL.
//!
//! A [`DslKernel`] is the predicate/state machine used by graph traversal. For
//! every candidate edge `(src)-[edge]->(dest)`, traversal evaluates:
//!
//! 1. `visit`: whether the edge may be accepted.
//! 2. `next_state`: how named state changes after accepting the edge.
//! 3. `stop`: whether the newly accepted path should be emitted.
//!
//! Prefer the method-based [`DslExpr`] API in Rust. [`DslKernel::from_polars_json`]
//! exists for callers that already have Polars expression JSON.

mod arrow_value;
pub(crate) mod bind;
pub(crate) mod eval;
mod expr;
mod ops;
mod polars_json;
mod value;

use std::sync::Arc;

use anyhow::{Context, Result};
use smallvec::SmallVec;

pub use expr::DslExpr;
use expr::{ColumnRef, Expr};
pub use value::{Scalar, Value};

use crate::graph::Graph;

pub(crate) use bind::BoundKernel;
pub(crate) use eval::EvalCtx;

/// A traversal predicate/state kernel.
///
/// -   `visit` and `stop` must evaluate to booleans.
/// -   `next_state` expressions run only after `visit` accepts an edge,
///     and each `(name, expr)` pair replaces that named state value for the child path.
#[derive(Debug, Clone)]
pub struct DslKernel {
    pub(crate) visit: Expr<ColumnRef>,
    pub(crate) next_state: Vec<(String, Expr<ColumnRef>)>,
    pub(crate) stop: Expr<ColumnRef>,
    pub(crate) initial_state: StateRow,
}

impl DslKernel {
    /// Creates a kernel from typed DSL expressions.
    pub fn new(
        visit: DslExpr,
        next_state: impl IntoIterator<Item = (String, DslExpr)>,
        stop: DslExpr,
        initial_state: impl IntoIterator<Item = (String, Value)>,
    ) -> Self {
        let mut initial_state = initial_state.into_iter().collect::<StateRow>();
        initial_state.sort_by(|a, b| a.0.cmp(&b.0));

        Self {
            visit: visit.into_inner(),
            next_state: next_state
                .into_iter()
                .map(|(name, expr)| (name, expr.into_inner()))
                .collect(),
            stop: stop.into_inner(),
            initial_state,
        }
    }

    /// Creates a kernel from supported Polars expression JSON.
    ///
    /// This is a compatibility path - it maps JSON into the same expression tree
    /// used by [`DslKernel::new`].
    pub fn from_polars_json(
        visit_json: &str,
        next_state: impl IntoIterator<Item = (String, String)>,
        stop_json: &str,
        initial_state: impl IntoIterator<Item = (String, Value)>,
    ) -> Result<Self> {
        let mut initial_state = initial_state.into_iter().collect::<StateRow>();
        initial_state.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(Self {
            visit: DslExpr::from_polars_json(visit_json)
                .context("invalid visit expression")?
                .into_inner(),
            next_state: next_state
                .into_iter()
                .map(|(name, json)| Ok((name, DslExpr::from_polars_json(&json)?.into_inner())))
                .collect::<Result<_>>()
                .context("invalid next_state expression")?,
            stop: DslExpr::from_polars_json(stop_json)
                .context("invalid stop expression")?
                .into_inner(),
            initial_state,
        })
    }

    pub(crate) fn bind(self, graph: &Graph) -> Result<bind::BoundKernel> {
        bind::BoundKernel::bind(graph, self)
    }
}

/// Named state carried by each path.
pub type StateRow = Vec<(String, Value)>;

#[derive(Debug, Clone)]
pub(crate) enum StateValue {
    Inline(Value),
    Shared(Arc<Value>),
}

impl StateValue {
    pub(crate) fn new(value: Value) -> Self {
        match value {
            Value::List(_) | Value::Struct(_) => Self::Shared(Arc::new(value)),
            value => Self::Inline(value),
        }
    }

    pub(crate) fn as_value(&self) -> &Value {
        match self {
            Self::Inline(value) => value,
            Self::Shared(value) => value,
        }
    }

    pub(crate) fn to_value(&self) -> Value {
        self.as_value().clone()
    }
}

// Bound state drops names so every state read/write is an index lookup.
pub(crate) type StateValues = SmallVec<[StateValue; 8]>;

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::record_batch;

    use crate::{
        graph::{EDGE_DEST_COL, EDGE_SRC_COL, Graph, GraphId, ID_COL},
        traversal::{TraversalConfigBuilder, TraversalStrategy},
    };

    fn string_graph() -> Graph {
        Graph::new(
            record_batch!(
                (ID_COL, Utf8, ["a", "b", "c"]),
                ("kind", Utf8, ["start", "target", "target"])
            )
            .unwrap(),
            record_batch!(
                (ID_COL, Utf8, ["ab", "ac"]),
                (EDGE_SRC_COL, Utf8, ["a", "a"]),
                (EDGE_DEST_COL, Utf8, ["b", "c"]),
                ("active", Boolean, [true, false]),
                ("cost", UInt64, [5, 7])
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn typed_api_filters_updates_state_and_reads_string_ids() {
        let kernel = DslKernel::new(
            DslExpr::edge("active").and(DslExpr::state("budget").ge(DslExpr::edge("cost"))),
            [
                (
                    "budget".into(),
                    DslExpr::state("budget").minus(DslExpr::edge("cost")),
                ),
                (
                    "spent".into(),
                    DslExpr::state("spent").plus(DslExpr::edge("cost")),
                ),
            ],
            DslExpr::dest("kind")
                .eq(DslExpr::string_lit("target"))
                .and(DslExpr::edge_id().eq(DslExpr::string_lit("ab"))),
            [
                ("budget".into(), Value::U64(6)),
                ("spent".into(), Value::U64(0)),
            ],
        );

        let graph = string_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(kernel)
                    .with_start_nodes(["a"])
                    .with_strategy(TraversalStrategy::BreadthFirst)
                    .with_parallelism(false)
                    .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 1);
        assert_eq!(
            result.paths[0].nodes,
            vec![GraphId::Str("a"), GraphId::Str("b")]
        );
        assert_eq!(result.paths[0].edges, vec![GraphId::Str("ab")]);
        assert_eq!(result.stats.evaluated_edges, 2);
        assert_eq!(result.stats.accepted_edges, 1);
        assert_eq!(result.stats.rejected_edges, 1);
    }

    #[test]
    fn reserved_topology_fields_read_from_identity() {
        let kernel = DslKernel::new(
            DslExpr::src("id")
                .eq(DslExpr::string_lit("a"))
                .and(DslExpr::edge("src").eq(DslExpr::string_lit("a")))
                .and(DslExpr::edge("dest").eq(DslExpr::string_lit("b"))),
            std::iter::empty::<(String, DslExpr)>(),
            DslExpr::dest("id")
                .eq(DslExpr::string_lit("b"))
                .and(DslExpr::edge("id").eq(DslExpr::string_lit("ab"))),
            std::iter::empty::<(String, Value)>(),
        );

        let graph = string_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(kernel)
                    .with_start_nodes(["a"])
                    .with_strategy(TraversalStrategy::BreadthFirst)
                    .with_parallelism(false)
                    .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].edges, vec![GraphId::Str("ab")]);
    }

    #[test]
    fn u64_id_columns_and_mask_if_any_work() {
        let graph = Graph::new(
            record_batch!((ID_COL, UInt64, [1, 2, 3])).unwrap(),
            record_batch!(
                (ID_COL, UInt64, [9, 10]),
                (EDGE_SRC_COL, UInt64, [1, 2]),
                (EDGE_DEST_COL, UInt64, [2, 3]),
                ("from_mask", UInt64, [0b010, 0b100]),
                ("to_mask", UInt64, [0b100, 0b001])
            )
            .unwrap(),
        )
        .unwrap();
        let kernel = DslKernel::new(
            DslExpr::src_id()
                .eq(DslExpr::state("last"))
                .and(DslExpr::edge_id().eq(DslExpr::uint_lit(9))),
            [
                (
                    "bits".into(),
                    DslExpr::state("bits")
                        .mask_if_any(DslExpr::edge("from_mask"), DslExpr::edge("to_mask")),
                ),
                ("last".into(), DslExpr::dest_id()),
            ],
            DslExpr::state("bits").eq(DslExpr::uint_lit(0b100)),
            [
                ("bits".into(), Value::U64(0b010)),
                ("last".into(), Value::U64(1)),
            ],
        );

        let result = graph
            .search(
                TraversalConfigBuilder::new(kernel)
                    .with_start_nodes([1_u64])
                    .with_parallelism(false)
                    .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 1);
        assert_eq!(
            result.paths[0].nodes,
            vec![GraphId::U64(1), GraphId::U64(2)]
        );
        assert_eq!(result.paths[0].edges, vec![GraphId::U64(9)]);
    }

    #[test]
    fn primitive_column_literal_comparisons_work() {
        let kernel = DslKernel::new(
            DslExpr::edge("active")
                .eq(DslExpr::bool_lit(true))
                .and(DslExpr::edge("cost").lt(DslExpr::uint_lit(6)))
                .and(DslExpr::dest("kind").eq(DslExpr::string_lit("target"))),
            std::iter::empty::<(String, DslExpr)>(),
            DslExpr::bool_lit(true),
            std::iter::empty::<(String, Value)>(),
        );

        let graph = string_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(kernel)
                    .with_start_nodes(["a"])
                    .with_strategy(TraversalStrategy::BreadthFirst)
                    .with_parallelism(false)
                    .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].edges, vec![GraphId::Str("ab")]);
    }

    #[test]
    fn polars_json_compat_path_uses_same_expression_engine() {
        let visit = r#"{"Column":"edge.active"}"#;
        let stop = r#"{"BinaryExpr":{"left":{"Column":"dest.kind"},"op":"Eq","right":{"Literal":{"Dyn":{"String":"target"}}}}}"#;
        let next_state = [(
            "hops".into(),
            r#"{"BinaryExpr":{"left":{"Column":"state.hops"},"op":"Plus","right":{"Literal":{"Dyn":{"UInt":1}}}}}"#
                .into(),
        )];
        let kernel =
            DslKernel::from_polars_json(visit, next_state, stop, [("hops".into(), Value::U64(0))])
                .unwrap();

        let graph = string_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(kernel)
                    .with_start_nodes(["a"])
                    .with_parallelism(false)
                    .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].edges, vec![GraphId::Str("ab")]);
    }

    #[test]
    fn string_predicates_work() {
        let kernel = DslKernel::new(
            DslExpr::bool_lit(true),
            std::iter::empty::<(String, DslExpr)>(),
            DslExpr::dest("kind")
                .str_starts_with(DslExpr::string_lit("tar"))
                .and(DslExpr::dest("kind").str_contains(DslExpr::string_lit("get")))
                .and(DslExpr::dest("kind").str_ends_with(DslExpr::string_lit("et"))),
            std::iter::empty::<(String, Value)>(),
        );

        let graph = string_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(kernel)
                    .with_start_nodes(["a"])
                    .with_parallelism(false)
                    .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 2);
    }

    #[test]
    fn typed_api_boolean_ops_are_lazy() {
        let bad_rhs = DslExpr::edge("active")
            .plus(DslExpr::uint_lit(1))
            .eq(DslExpr::uint_lit(2));
        let graph = string_graph();

        let and_result = graph
            .search(
                TraversalConfigBuilder::new(DslKernel::new(
                    DslExpr::bool_lit(false).and(bad_rhs.clone()),
                    std::iter::empty::<(String, DslExpr)>(),
                    DslExpr::bool_lit(true),
                    std::iter::empty::<(String, Value)>(),
                ))
                .with_start_nodes(["a"])
                .with_parallelism(false)
                .build(),
            )
            .unwrap();
        assert_eq!(and_result.paths.len(), 0);

        let or_result = graph
            .search(
                TraversalConfigBuilder::new(DslKernel::new(
                    DslExpr::bool_lit(true).or(bad_rhs),
                    std::iter::empty::<(String, DslExpr)>(),
                    DslExpr::dest("kind").eq(DslExpr::string_lit("target")),
                    std::iter::empty::<(String, Value)>(),
                ))
                .with_start_nodes(["a"])
                .with_parallelism(false)
                .build(),
            )
            .unwrap();
        assert_eq!(or_result.paths.len(), 2);
    }

    #[test]
    fn typed_api_conditionals_are_lazy() {
        let kernel = DslKernel::new(
            DslExpr::bool_lit(true),
            [(
                "score".into(),
                DslExpr::when(
                    DslExpr::bool_lit(true),
                    DslExpr::uint_lit(7),
                    DslExpr::string_lit("unused").plus(DslExpr::uint_lit(1)),
                ),
            )],
            DslExpr::state("score").eq(DslExpr::uint_lit(7)),
            [("score".into(), Value::U64(0))],
        );

        let graph = string_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(kernel)
                    .with_start_nodes(["a"])
                    .with_parallelism(false)
                    .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 2);
    }

    #[test]
    fn unsupported_json_ops_fail_clearly() {
        let err = DslExpr::from_polars_json(
            r#"{"BinaryExpr":{"left":{"Column":"dest.kind"},"op":"Pow","right":{"Literal":{"Dyn":{"String":"target"}}}}}"#,
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("unsupported binary op"));
    }

    #[test]
    fn list_ops_handle_nested_runtime_values() {
        use crate::dsl::ops::list::{ListOp, SetOp};

        let values = Value::List(vec![
            Value::I64(3),
            Value::Null,
            Value::I64(1),
            Value::I64(3),
        ]);

        assert_eq!(
            ListOp::DropNulls
                .eval(std::slice::from_ref(&values))
                .unwrap(),
            Value::List(vec![Value::I64(3), Value::I64(1), Value::I64(3)])
        );
        assert_eq!(
            ListOp::Unique.eval(std::slice::from_ref(&values)).unwrap(),
            Value::List(vec![Value::I64(3), Value::Null, Value::I64(1)])
        );
        assert_eq!(
            ListOp::Sort {
                descending: false,
                nulls_last: true,
            }
            .eval(std::slice::from_ref(&values))
            .unwrap(),
            Value::List(vec![
                Value::I64(1),
                Value::I64(3),
                Value::I64(3),
                Value::Null
            ])
        );
        assert_eq!(
            ListOp::Set(SetOp::Difference)
                .eval(&[values, Value::List(vec![Value::I64(1)])])
                .unwrap(),
            Value::List(vec![Value::I64(3), Value::Null])
        );
        assert_eq!(
            ListOp::Explode {
                empty_as_null: true,
                keep_nulls: true,
            }
            .eval(&[Value::List(vec![
                Value::List(vec![Value::I64(1), Value::I64(2)]),
                Value::I64(3),
                Value::Null,
                Value::List(vec![]),
            ])])
            .unwrap(),
            Value::List(vec![
                Value::I64(1),
                Value::I64(2),
                Value::I64(3),
                Value::Null,
                Value::Null,
            ])
        );
        assert_eq!(
            ListOp::Set(SetOp::Intersection)
                .eval(&[
                    Value::List(vec![Value::Struct(vec![
                        ("protocol".into(), Value::Str("tcp".into())),
                        ("from_port".into(), Value::U64(80)),
                    ])]),
                    Value::List(vec![Value::Struct(vec![
                        ("from_port".into(), Value::U64(80)),
                        ("protocol".into(), Value::Str("tcp".into())),
                    ])]),
                ])
                .unwrap(),
            Value::List(vec![Value::Struct(vec![
                ("protocol".into(), Value::Str("tcp".into())),
                ("from_port".into(), Value::U64(80)),
            ])])
        );
    }

    #[test]
    fn struct_ops_handle_runtime_values() {
        use crate::dsl::ops::struct_::StructOp;

        let value = Value::Struct(vec![
            ("score".into(), Value::I64(9)),
            ("label".into(), Value::Str("b".into())),
        ]);

        assert_eq!(
            StructOp::FieldByName("score".into())
                .eval(std::slice::from_ref(&value))
                .unwrap(),
            Value::I64(9)
        );
        assert_eq!(
            StructOp::RenameFields(vec!["points".into(), "name".into()])
                .eval(std::slice::from_ref(&value))
                .unwrap(),
            Value::Struct(vec![
                ("points".into(), Value::I64(9)),
                ("name".into(), Value::Str("b".into())),
            ])
        );
        assert_eq!(
            value.to_value(),
            serde_json::json!({"score": 9, "label": "b"})
        );
    }

    #[test]
    fn polars_json_parser_accepts_list_and_struct_shapes() {
        for json in [
            r#"{"Function":{"input":[{"Column":"state.x"}],"function":{"ListExpr":"Length"}}}"#,
            r#"{"Function":{"input":[{"Column":"state.x"},{"Literal":{"Dyn":{"Int":2}}}],"function":{"ListExpr":{"Contains":{"nulls_equal":true}}}}}"#,
            r#"{"Eval":{"expr":{"Column":"state.x"},"evaluation":{"BinaryExpr":{"left":"Element","op":"Plus","right":{"Literal":{"Dyn":{"Int":1}}}}},"variant":"List"}}"#,
            r#"{"Eval":{"expr":{"Column":"state.x"},"evaluation":{"Filter":{"input":{"Column":""},"by":{"BinaryExpr":{"left":"Element","op":"Gt","right":{"Literal":{"Dyn":{"Int":1}}}}}}},"variant":"List"}}"#,
            r#"{"Ternary":{"predicate":{"Column":"state.ok"},"truthy":{"Literal":{"Dyn":{"Int":1}}},"falsy":{"Literal":{"Dyn":{"Int":0}}}}}"#,
            r#"{"Explode":{"input":{"Column":"state.x"},"options":{"empty_as_null":true,"keep_nulls":true}}}"#,
            r#"{"Function":{"input":[{"Column":"state.s"}],"function":{"StructExpr":{"FieldByName":"score"}}}}"#,
            r#"{"StructEval":{"expr":{"Column":"state.s"},"evaluation":[{"Alias":[{"Literal":{"Dyn":{"Int":3}}},"extra"]}]}}"#,
        ] {
            DslExpr::from_polars_json(json).unwrap();
        }
    }
}
