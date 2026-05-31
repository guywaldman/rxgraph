//! Traversal expression DSL.
//!
//! A [`DslKernel`] is the predicate/state machine used by graph traversal. For
//! every candidate edge `(src)-[edge]->(dest)`, traversal evaluates:
//!
//! 1. `visit`: whether the edge may be accepted.
//! 2. `next_state`: how named scalar state changes after accepting the edge.
//! 3. `stop`: whether the newly accepted path should be emitted.
//!
//! Expressions can read source-node fields, destination-node fields, edge fields,
//! traversal state, and graph identity columns via [`DslExpr::src_id`],
//! [`DslExpr::dest_id`], and [`DslExpr::edge_id`]. Identity expressions follow
//! the graph's ID mode: [`Value::U64`] for integer-ID graphs and [`Value::Str`]
//! for string-ID graphs.
//!
//! Prefer the method-based [`DslExpr`] API in Rust. [`DslKernel::from_polars_json`]
//! exists for callers that already have Polars expression JSON.
//!
//! A typical kernel accepts only enabled edges, increments path state, and emits
//! paths that reach a target node:
//!
//! ```
//! use rxgraph::{DslExpr as e, DslKernel, Value};
//!
//! let kernel = DslKernel::new(
//!     e::edge("enabled"),
//!     [("hops".into(), e::state("hops").plus(e::uint(1)))],
//!     e::dest("is_target"),
//!     [("hops".into(), Value::U64(0))],
//! );
//! ```

// TODO: Think of a better IR than JSON, and consider our own typed DSL.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use arrow::{
    array::{
        Array, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
        Int64Array, LargeListArray, LargeStringArray, ListArray, StringArray, StringViewArray,
        UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::DataType,
    record_batch::RecordBatch,
};
use serde_json::{Number as JsonNumber, Value as Json};
use smallvec::SmallVec;

use crate::graph::{EdgeId, Graph, GraphId, GraphRepo, NodeId};

/// A traversal predicate/state kernel.
///
/// `visit` and `stop` must evaluate to booleans. `next_state` expressions run
/// only after `visit` accepts an edge, and each `(name, expr)` pair replaces that
/// named state value for the child path.
#[derive(Debug, Clone)]
pub struct DslKernel {
    visit: Expr<ColumnRef>,
    next_state: Vec<(String, Expr<ColumnRef>)>,
    stop: Expr<ColumnRef>,
    initial_state: StateRow,
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
            visit: visit.0,
            next_state: next_state
                .into_iter()
                .map(|(name, expr)| (name, expr.0))
                .collect(),
            stop: stop.0,
            initial_state,
        }
    }

    /// Creates a kernel from supported Polars expression JSON.
    ///
    /// This is a compatibility path; it maps JSON into the same expression tree
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
                .0,
            next_state: next_state
                .into_iter()
                .map(|(name, json)| Ok((name, DslExpr::from_polars_json(&json)?.0)))
                .collect::<Result<_>>()
                .context("invalid next_state expression")?,
            stop: DslExpr::from_polars_json(stop_json)
                .context("invalid stop expression")?
                .0,
            initial_state,
        })
    }

    pub(crate) fn bind(self, graph: &Graph) -> Result<BoundKernel> {
        BoundKernel::bind(graph, self)
    }
}

/// Named scalar state carried by each path.
///
/// Names are normalized once when the kernel is bound to a graph. During search,
/// state reads and writes use compact indexes.
pub type StateRow = Vec<(String, Value)>;
// Bound state drops names so every state read/write is an index lookup.
pub(crate) type StateValues = SmallVec<[Value; 8]>;

/// Typed traversal expression.
///
/// Expressions are immutable builders. Methods consume `self` and return a new
/// expression, so predicates can be composed without string operation names.
#[derive(Debug, Clone)]
pub struct DslExpr(Expr<ColumnRef>);

impl DslExpr {
    /// Reads a field from the source node of the candidate edge.
    pub fn src(field: impl Into<String>) -> Self {
        Self(Expr::Column(ColumnRef::SrcField(field.into())))
    }

    /// Reads a field from the destination node of the candidate edge.
    pub fn dest(field: impl Into<String>) -> Self {
        Self(Expr::Column(ColumnRef::DestField(field.into())))
    }

    /// Reads a field from the candidate edge.
    pub fn edge(field: impl Into<String>) -> Self {
        Self(Expr::Column(ColumnRef::EdgeField(field.into())))
    }

    /// Reads a named value from the current path state.
    pub fn state(field: impl Into<String>) -> Self {
        Self(Expr::Column(ColumnRef::State(field.into())))
    }

    /// Reads the external source-node ID.
    pub fn src_id() -> Self {
        Self(Expr::Column(ColumnRef::SrcId))
    }

    /// Reads the external destination-node ID.
    pub fn dest_id() -> Self {
        Self(Expr::Column(ColumnRef::DestId))
    }

    /// Reads the external edge ID.
    pub fn edge_id() -> Self {
        Self(Expr::Column(ColumnRef::EdgeId))
    }

    /// Null literal.
    pub fn null() -> Self {
        Self::literal(Value::Null)
    }

    /// Boolean literal.
    pub fn bool(value: bool) -> Self {
        Self::literal(Value::Bool(value))
    }

    /// Signed integer literal.
    pub fn int(value: i64) -> Self {
        Self::literal(Value::I64(value))
    }

    /// Unsigned integer literal.
    pub fn uint(value: u64) -> Self {
        Self::literal(Value::U64(value))
    }

    /// Floating point literal.
    pub fn float(value: f64) -> Self {
        Self::literal(Value::F64(value))
    }

    /// String literal.
    pub fn string(value: impl Into<Arc<str>>) -> Self {
        Self::literal(Value::Str(value.into()))
    }

    /// Literal value.
    pub fn literal(value: impl Into<Value>) -> Self {
        Self(Expr::Literal(value.into()))
    }

    /// Parses the supported Polars JSON expression subset.
    pub fn from_polars_json(json: &str) -> Result<Self> {
        Ok(Self(parse_polars_json(json)?))
    }

    /// Boolean `AND`.
    pub fn and(self, rhs: Self) -> Self {
        self.binary(BinaryOp::And, rhs)
    }

    /// Boolean `OR`.
    pub fn or(self, rhs: Self) -> Self {
        self.binary(BinaryOp::Or, rhs)
    }

    /// Equality comparison.
    pub fn eq(self, rhs: Self) -> Self {
        self.binary(BinaryOp::Eq, rhs)
    }

    /// Inequality comparison.
    pub fn ne(self, rhs: Self) -> Self {
        self.binary(BinaryOp::NotEq, rhs)
    }

    /// Less-than comparison.
    pub fn lt(self, rhs: Self) -> Self {
        self.binary(BinaryOp::Lt, rhs)
    }

    /// Less-than-or-equal comparison.
    pub fn le(self, rhs: Self) -> Self {
        self.binary(BinaryOp::LtEq, rhs)
    }

    /// Greater-than comparison.
    pub fn gt(self, rhs: Self) -> Self {
        self.binary(BinaryOp::Gt, rhs)
    }

    /// Greater-than-or-equal comparison.
    pub fn ge(self, rhs: Self) -> Self {
        self.binary(BinaryOp::GtEq, rhs)
    }

    /// Numeric addition.
    pub fn plus(self, rhs: Self) -> Self {
        self.binary(BinaryOp::Plus, rhs)
    }

    /// Numeric subtraction.
    pub fn minus(self, rhs: Self) -> Self {
        self.binary(BinaryOp::Minus, rhs)
    }

    /// Numeric multiplication.
    pub fn multiply(self, rhs: Self) -> Self {
        self.binary(BinaryOp::Multiply, rhs)
    }

    /// Numeric division. Always returns [`Value::F64`].
    pub fn divide(self, rhs: Self) -> Self {
        self.binary(BinaryOp::Divide, rhs)
    }

    /// Integer remainder.
    pub fn modulo(self, rhs: Self) -> Self {
        self.binary(BinaryOp::Modulus, rhs)
    }

    /// Unsigned integer bitwise `AND`.
    pub fn bit_and(self, rhs: Self) -> Self {
        self.binary(BinaryOp::BitAnd, rhs)
    }

    /// Unsigned integer bitwise `OR`.
    pub fn bit_or(self, rhs: Self) -> Self {
        self.binary(BinaryOp::BitOr, rhs)
    }

    /// Unsigned integer bitwise `XOR`.
    pub fn bit_xor(self, rhs: Self) -> Self {
        self.binary(BinaryOp::BitXor, rhs)
    }

    /// Returns `then_value` when any bit in `mask` is set in `self`; otherwise
    /// returns `0`.
    pub fn mask_if_any(self, mask: Self, then_value: Self) -> Self {
        Self(Expr::MaskIfAny {
            value: Box::new(self.0),
            mask: Box::new(mask.0),
            then_value: Box::new(then_value.0),
        })
    }

    /// Boolean negation.
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(Expr::Not(Box::new(self.0)))
    }

    /// Null predicate.
    pub fn is_null(self) -> Self {
        Self(Expr::IsNull(Box::new(self.0)))
    }

    /// Non-null predicate.
    pub fn is_not_null(self) -> Self {
        Self(Expr::IsNotNull(Box::new(self.0)))
    }

    /// String containment predicate.
    pub fn contains(self, needle: Self) -> Self {
        self.str_pred(StrOp::Contains, needle)
    }

    /// String prefix predicate.
    pub fn starts_with(self, prefix: Self) -> Self {
        self.str_pred(StrOp::StartsWith, prefix)
    }

    /// String suffix predicate.
    pub fn ends_with(self, suffix: Self) -> Self {
        self.str_pred(StrOp::EndsWith, suffix)
    }

    /// Concatenates list expressions, appending scalar expressions as single items.
    pub fn concat_list(values: impl IntoIterator<Item = Self>) -> Self {
        Self(Expr::ListConcat(
            values.into_iter().map(|expr| expr.0).collect(),
        ))
    }

    fn binary(self, op: BinaryOp, rhs: Self) -> Self {
        Self(Expr::Binary(Box::new(self.0), op, Box::new(rhs.0)))
    }

    fn str_pred(self, op: StrOp, rhs: Self) -> Self {
        Self(Expr::StrPred(Box::new(self.0), op, Box::new(rhs.0)))
    }
}

/// Scalar value used by DSL expressions and traversal state.
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    /// Null value.
    Null,
    /// Boolean value.
    Bool(bool),
    /// Signed integer value.
    I64(i64),
    /// Unsigned integer value.
    U64(u64),
    /// Floating point value.
    F64(f64),
    /// Shared string value.
    Str(Arc<str>),
}

impl From<Scalar> for Value {
    fn from(value: Scalar) -> Self {
        match value {
            Scalar::Null => Self::Null,
            Scalar::Bool(value) => Self::Bool(value),
            Scalar::I64(value) => Self::I64(value),
            Scalar::U64(value) => Self::U64(value),
            Scalar::F64(value) => Self::F64(value),
            Scalar::Str(value) => Self::Str(value),
        }
    }
}

/// Runtime value used by DSL expressions and traversal state.
///
/// Arrow columns are converted to this compact runtime representation when read
/// by the DSL.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Null value.
    Null,
    /// Boolean value.
    Bool(bool),
    /// Signed integer value.
    I64(i64),
    /// Unsigned integer value.
    U64(u64),
    /// Floating point value.
    F64(f64),
    /// Shared string value.
    Str(Arc<str>),
    /// Ordered value list.
    List(Vec<Value>),
}

impl Value {
    /// Converts the scalar to JSON for callers that need a loosely typed value.
    pub fn to_value(&self) -> Json {
        match self {
            Self::Null => Json::Null,
            Self::Bool(value) => Json::Bool(*value),
            Self::I64(value) => Json::Number(JsonNumber::from(*value)),
            Self::U64(value) => Json::Number(JsonNumber::from(*value)),
            Self::F64(value) => JsonNumber::from_f64(*value)
                .map(Json::Number)
                .unwrap_or(Json::Null),
            Self::Str(value) => Json::String(value.to_string()),
            Self::List(values) => Json::Array(values.iter().map(Value::to_value).collect()),
        }
    }

    fn truthy(&self) -> Result<bool> {
        match self {
            Self::Bool(value) => Ok(*value),
            Self::Null => Ok(false),
            other => bail!("expected boolean expression, got {other:?}"),
        }
    }

    fn as_i128(&self) -> Option<i128> {
        match self {
            Self::I64(value) => Some(*value as i128),
            Self::U64(value) => Some(*value as i128),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::I64(value) if *value >= 0 => Some(*value as u64),
            Self::U64(value) => Some(*value),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::I64(value) => Some(*value as f64),
            Self::U64(value) => Some(*value as f64),
            Self::F64(value) => Some(*value),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
enum Expr<C> {
    Column(C),
    Literal(Value),
    Binary(Box<Expr<C>>, BinaryOp, Box<Expr<C>>),
    MaskIfAny {
        value: Box<Expr<C>>,
        mask: Box<Expr<C>>,
        then_value: Box<Expr<C>>,
    },
    ListConcat(Vec<Expr<C>>),
    Not(Box<Expr<C>>),
    IsNull(Box<Expr<C>>),
    IsNotNull(Box<Expr<C>>),
    StrPred(Box<Expr<C>>, StrOp, Box<Expr<C>>),
}

impl<C> Expr<C> {
    fn try_map_column<D>(self, f: &mut impl FnMut(C) -> Result<D>) -> Result<Expr<D>> {
        Ok(match self {
            Self::Column(column) => Expr::Column(f(column)?),
            Self::Literal(value) => Expr::Literal(value),
            Self::Binary(left, op, right) => Expr::Binary(
                Box::new(left.try_map_column(f)?),
                op,
                Box::new(right.try_map_column(f)?),
            ),
            Self::MaskIfAny {
                value,
                mask,
                then_value,
            } => Expr::MaskIfAny {
                value: Box::new(value.try_map_column(f)?),
                mask: Box::new(mask.try_map_column(f)?),
                then_value: Box::new(then_value.try_map_column(f)?),
            },
            Self::ListConcat(values) => Expr::ListConcat(
                values
                    .into_iter()
                    .map(|expr| expr.try_map_column(f))
                    .collect::<Result<_>>()?,
            ),
            Self::Not(expr) => Expr::Not(Box::new(expr.try_map_column(f)?)),
            Self::IsNull(expr) => Expr::IsNull(Box::new(expr.try_map_column(f)?)),
            Self::IsNotNull(expr) => Expr::IsNotNull(Box::new(expr.try_map_column(f)?)),
            Self::StrPred(left, op, right) => Expr::StrPred(
                Box::new(left.try_map_column(f)?),
                op,
                Box::new(right.try_map_column(f)?),
            ),
        })
    }
}

#[derive(Debug, Clone)]
enum ColumnRef {
    SrcId,
    DestId,
    EdgeId,
    SrcField(String),
    DestField(String),
    EdgeField(String),
    State(String),
}

#[derive(Debug, Clone, Copy)]
enum BinaryOp {
    And,
    Or,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulus,
    BitAnd,
    BitOr,
    BitXor,
}

#[derive(Debug, Clone, Copy)]
enum StrOp {
    Contains,
    StartsWith,
    EndsWith,
}

#[derive(Debug)]
pub(crate) struct BoundKernel {
    visit: Expr<BoundColumn>,
    next_state: Vec<(usize, Expr<BoundColumn>)>,
    stop: Expr<BoundColumn>,
    names: Vec<String>,
    initial_state: StateValues,
}

impl BoundKernel {
    fn bind(graph: &Graph, kernel: DslKernel) -> Result<Self> {
        let names = state_names(&kernel.initial_state, &kernel.next_state);
        let mut bind = |column| BoundColumn::bind(graph, column, &names);

        Ok(Self {
            // Column binding removes per-edge schema lookups from DSL evaluation.
            visit: kernel.visit.try_map_column(&mut bind)?,
            next_state: kernel
                .next_state
                .into_iter()
                .map(|(name, expr)| {
                    Ok((
                        state_index(&names, &name).unwrap(),
                        expr.try_map_column(&mut bind)?,
                    ))
                })
                .collect::<Result<_>>()?,
            stop: kernel.stop.try_map_column(&mut bind)?,
            names: names.clone(),
            initial_state: normalize_state(kernel.initial_state, &names),
        })
    }

    pub(crate) fn initial_state(&self) -> &StateValues {
        &self.initial_state
    }

    pub(crate) fn visit(&self, ctx: &EvalCtx<'_>) -> Result<bool> {
        self.visit.eval(ctx)?.truthy()
    }

    pub(crate) fn next_state(&self, current: &[Value], ctx: &EvalCtx<'_>) -> Result<StateValues> {
        let mut next = current.iter().cloned().collect::<StateValues>();
        for (index, expr) in &self.next_state {
            next[*index] = expr.eval(ctx)?;
        }
        Ok(next)
    }

    pub(crate) fn stop(&self, ctx: &EvalCtx<'_>) -> Result<bool> {
        self.stop.eval(ctx)?.truthy()
    }

    pub(crate) fn state_row(&self, state: &[Value]) -> StateRow {
        self.names
            .iter()
            .cloned()
            .zip(state.iter().cloned())
            .collect()
    }
}

#[derive(Debug, Clone)]
enum BoundColumn {
    SrcId,
    DestId,
    EdgeId,
    Src(ColumnReader),
    Dest(ColumnReader),
    Edge(ColumnReader),
    State(usize),
    MissingState,
}

impl BoundColumn {
    fn bind(graph: &Graph, column: ColumnRef, names: &[String]) -> Result<Self> {
        Ok(match column {
            ColumnRef::SrcId => Self::SrcId,
            ColumnRef::DestId => Self::DestId,
            ColumnRef::EdgeId => Self::EdgeId,
            ColumnRef::SrcField(name) => Self::Src(ColumnReader::bind(&graph.repo.nodes, &name)?),
            ColumnRef::DestField(name) => Self::Dest(ColumnReader::bind(&graph.repo.nodes, &name)?),
            ColumnRef::EdgeField(name) => Self::Edge(ColumnReader::bind(&graph.repo.edges, &name)?),
            ColumnRef::State(name) => state_index(names, &name)
                .map(Self::State)
                .unwrap_or(Self::MissingState),
        })
    }

    fn eval(&self, ctx: &EvalCtx<'_>) -> Result<Value> {
        match self {
            Self::SrcId => graph_id_scalar(
                ctx.graph
                    .repo
                    .external_node(ctx.src)
                    .context("missing src id")?,
            ),
            Self::DestId => graph_id_scalar(
                ctx.graph
                    .repo
                    .external_node(ctx.dest)
                    .context("missing dest id")?,
            ),
            Self::EdgeId => graph_id_scalar(
                ctx.graph
                    .repo
                    .external_edge(ctx.edge)
                    .context("missing edge id")?,
            ),
            Self::Src(reader) => reader.value(ctx.src as usize),
            Self::Dest(reader) => reader.value(ctx.dest as usize),
            Self::Edge(reader) => reader.value(ctx.edge as usize),
            Self::State(index) => Ok(ctx.state[*index].clone()),
            Self::MissingState => Ok(Value::Null),
        }
    }

    fn str_value<'a>(&'a self, ctx: &'a EvalCtx<'_>) -> Result<Option<&'a str>> {
        Ok(match self {
            Self::SrcId => graph_id_str(ctx.graph.repo.external_node(ctx.src))?,
            Self::DestId => graph_id_str(ctx.graph.repo.external_node(ctx.dest))?,
            Self::EdgeId => graph_id_str(ctx.graph.repo.external_edge(ctx.edge))?,
            Self::Src(reader) => reader.str_value(ctx.src as usize)?,
            Self::Dest(reader) => reader.str_value(ctx.dest as usize)?,
            Self::Edge(reader) => reader.str_value(ctx.edge as usize)?,
            Self::State(index) => match &ctx.state[*index] {
                Value::Null => None,
                Value::Str(value) => Some(value),
                _ => bail!("string predicate expected strings"),
            },
            Self::MissingState => None,
        })
    }

    fn try_str_value<'a>(&'a self, ctx: &'a EvalCtx<'_>) -> Result<Option<Option<&'a str>>> {
        Ok(match self {
            Self::SrcId => graph_id_try_str(ctx.graph.repo.external_node(ctx.src))?,
            Self::DestId => graph_id_try_str(ctx.graph.repo.external_node(ctx.dest))?,
            Self::EdgeId => graph_id_try_str(ctx.graph.repo.external_edge(ctx.edge))?,
            Self::MissingState => Some(None),
            Self::Src(reader) => reader.try_str_value(ctx.src as usize)?,
            Self::Dest(reader) => reader.try_str_value(ctx.dest as usize)?,
            Self::Edge(reader) => reader.try_str_value(ctx.edge as usize)?,
            Self::State(index) => match &ctx.state[*index] {
                Value::Null => Some(None),
                Value::Str(value) => Some(Some(value)),
                _ => None,
            },
        })
    }
}

fn graph_id_scalar(id: GraphId<'_>) -> Result<Value> {
    Ok(match id {
        GraphId::U64(value) => Value::U64(value),
        GraphId::Str(value) => Value::Str(Arc::from(value)),
    })
}

fn graph_id_str(id: Option<GraphId<'_>>) -> Result<Option<&str>> {
    Ok(match id {
        Some(GraphId::Str(value)) => Some(value),
        Some(GraphId::U64(_)) => bail!("string predicate expected strings"),
        None => None,
    })
}

fn graph_id_try_str(id: Option<GraphId<'_>>) -> Result<Option<Option<&str>>> {
    Ok(match id {
        Some(GraphId::Str(value)) => Some(Some(value)),
        Some(GraphId::U64(_)) => None,
        None => Some(None),
    })
}

impl Expr<BoundColumn> {
    fn eval(&self, ctx: &EvalCtx<'_>) -> Result<Value> {
        Ok(match self {
            Self::Column(column) => column.eval(ctx)?,
            Self::Literal(value) => value.clone(),
            Self::Binary(left, BinaryOp::And, right) => {
                Value::Bool(left.eval(ctx)?.truthy()? && right.eval(ctx)?.truthy()?)
            }
            Self::Binary(left, BinaryOp::Or, right) => {
                Value::Bool(left.eval(ctx)?.truthy()? || right.eval(ctx)?.truthy()?)
            }
            Self::Binary(left, op, right) => {
                if let Some(value) = eval_str_binary(left, *op, right, ctx)? {
                    value
                } else {
                    eval_binary(left.eval(ctx)?, *op, right.eval(ctx)?)?
                }
            }
            Self::MaskIfAny {
                value,
                mask,
                then_value,
            } => {
                let value = value
                    .eval(ctx)?
                    .as_u64()
                    .context("mask_if_any value must be an unsigned integer")?;
                let mask = mask
                    .eval(ctx)?
                    .as_u64()
                    .context("mask_if_any mask must be an unsigned integer")?;
                if value & mask == 0 {
                    Value::U64(0)
                } else {
                    then_value.eval(ctx)?
                }
            }
            Self::ListConcat(values) => eval_list_concat(values, ctx)?,
            Self::Not(expr) => Value::Bool(!expr.eval(ctx)?.truthy()?),
            Self::IsNull(expr) => Value::Bool(expr.eval(ctx)? == Value::Null),
            Self::IsNotNull(expr) => Value::Bool(expr.eval(ctx)? != Value::Null),
            Self::StrPred(left, op, right) => eval_str_pred(left, *op, right, ctx)?,
        })
    }

    fn str_value<'a>(&'a self, ctx: &'a EvalCtx<'_>) -> Result<Option<&'a str>> {
        Ok(match self {
            Self::Column(column) => column.str_value(ctx)?,
            Self::Literal(Value::Null) => None,
            Self::Literal(Value::Str(value)) => Some(value),
            _ => bail!("string predicate expected strings"),
        })
    }

    fn try_str_value<'a>(&'a self, ctx: &'a EvalCtx<'_>) -> Result<Option<Option<&'a str>>> {
        Ok(match self {
            Self::Column(column) => column.try_str_value(ctx)?,
            Self::Literal(Value::Null) => Some(None),
            Self::Literal(Value::Str(value)) => Some(Some(value)),
            _ => None,
        })
    }
}

fn eval_list_concat(values: &[Expr<BoundColumn>], ctx: &EvalCtx<'_>) -> Result<Value> {
    let mut out = Vec::new();
    for expr in values {
        match expr.eval(ctx)? {
            Value::List(values) => out.extend(values),
            Value::Null => out.push(Value::Null),
            value => out.push(value),
        }
    }
    Ok(Value::List(out))
}

pub(crate) struct EvalCtx<'a> {
    pub(crate) graph: &'a Graph,
    pub(crate) src: NodeId,
    pub(crate) dest: NodeId,
    pub(crate) edge: EdgeId,
    pub(crate) state: &'a [Value],
}

#[derive(Debug, Clone)]
enum ColumnReader {
    Bool(BooleanArray),
    I8(Int8Array),
    I16(Int16Array),
    I32(Int32Array),
    I64(Int64Array),
    U8(UInt8Array),
    U16(UInt16Array),
    U32(UInt32Array),
    U64(UInt64Array),
    F32(Float32Array),
    F64(Float64Array),
    Utf8(StringArray),
    LargeUtf8(LargeStringArray),
    Utf8View(StringViewArray),
    List(ListArray),
    LargeList(LargeListArray),
}

impl ColumnReader {
    fn bind(batch: &RecordBatch, name: &str) -> Result<Self> {
        let column = batch
            .column_by_name(name)
            .with_context(|| format!("column {name:?} is missing"))?;

        macro_rules! typed {
            ($variant:ident, $array:ty) => {
                column
                    .as_any()
                    .downcast_ref::<$array>()
                    .with_context(|| format!("column {name:?} does not match its Arrow type"))?
                    .clone()
            };
        }

        Ok(match column.data_type() {
            DataType::Boolean => Self::Bool(typed!(Bool, BooleanArray)),
            DataType::Int8 => Self::I8(typed!(I8, Int8Array)),
            DataType::Int16 => Self::I16(typed!(I16, Int16Array)),
            DataType::Int32 => Self::I32(typed!(I32, Int32Array)),
            DataType::Int64 => Self::I64(typed!(I64, Int64Array)),
            DataType::UInt8 => Self::U8(typed!(U8, UInt8Array)),
            DataType::UInt16 => Self::U16(typed!(U16, UInt16Array)),
            DataType::UInt32 => Self::U32(typed!(U32, UInt32Array)),
            DataType::UInt64 => Self::U64(typed!(U64, UInt64Array)),
            DataType::Float32 => Self::F32(typed!(F32, Float32Array)),
            DataType::Float64 => Self::F64(typed!(F64, Float64Array)),
            DataType::Utf8 => Self::Utf8(typed!(Utf8, StringArray)),
            DataType::LargeUtf8 => Self::LargeUtf8(typed!(LargeUtf8, LargeStringArray)),
            DataType::Utf8View => Self::Utf8View(typed!(Utf8View, StringViewArray)),
            DataType::List(_) => Self::List(typed!(List, ListArray)),
            DataType::LargeList(_) => Self::LargeList(typed!(LargeList, LargeListArray)),
            typ => bail!("unsupported DSL column type for {name:?}: {typ:?}"),
        })
    }

    fn value(&self, row: usize) -> Result<Value> {
        macro_rules! nullable {
            ($array:expr, $value:expr) => {
                if $array.is_null(row) {
                    Value::Null
                } else {
                    $value
                }
            };
        }

        Ok(match self {
            Self::Bool(array) => nullable!(array, Value::Bool(array.value(row))),
            Self::I8(array) => nullable!(array, Value::I64(array.value(row) as i64)),
            Self::I16(array) => nullable!(array, Value::I64(array.value(row) as i64)),
            Self::I32(array) => nullable!(array, Value::I64(array.value(row) as i64)),
            Self::I64(array) => nullable!(array, Value::I64(array.value(row))),
            Self::U8(array) => nullable!(array, Value::U64(array.value(row) as u64)),
            Self::U16(array) => nullable!(array, Value::U64(array.value(row) as u64)),
            Self::U32(array) => nullable!(array, Value::U64(array.value(row) as u64)),
            Self::U64(array) => nullable!(array, Value::U64(array.value(row))),
            Self::F32(array) => nullable!(array, Value::F64(array.value(row) as f64)),
            Self::F64(array) => nullable!(array, Value::F64(array.value(row))),
            Self::Utf8(array) => nullable!(array, Value::Str(Arc::from(array.value(row)))),
            Self::LargeUtf8(array) => nullable!(array, Value::Str(Arc::from(array.value(row)))),
            Self::Utf8View(array) => nullable!(array, Value::Str(Arc::from(array.value(row)))),
            Self::List(array) => {
                nullable!(array, Value::List(array_to_scalars(&array.value(row))?))
            }
            Self::LargeList(array) => {
                nullable!(array, Value::List(array_to_scalars(&array.value(row))?))
            }
        })
    }

    fn str_value(&self, row: usize) -> Result<Option<&str>> {
        Ok(match self {
            Self::Utf8(array) => (!array.is_null(row)).then(|| array.value(row)),
            Self::LargeUtf8(array) => (!array.is_null(row)).then(|| array.value(row)),
            Self::Utf8View(array) => (!array.is_null(row)).then(|| array.value(row)),
            _ => bail!("string predicate expected strings"),
        })
    }

    fn try_str_value(&self, row: usize) -> Result<Option<Option<&str>>> {
        Ok(match self {
            Self::Utf8(_) | Self::LargeUtf8(_) | Self::Utf8View(_) => Some(self.str_value(row)?),
            _ => None,
        })
    }
}

fn array_to_scalars(array: &dyn Array) -> Result<Vec<Value>> {
    macro_rules! primitive {
        ($array:ty, $scalar:expr) => {
            if let Some(array) = array.as_any().downcast_ref::<$array>() {
                return (0..array.len())
                    .map(|row| {
                        Ok(if array.is_null(row) {
                            Value::Null
                        } else {
                            $scalar(array.value(row))
                        })
                    })
                    .collect();
            }
        };
    }

    primitive!(BooleanArray, Value::Bool);
    primitive!(Int8Array, |value| Value::I64(value as i64));
    primitive!(Int16Array, |value| Value::I64(value as i64));
    primitive!(Int32Array, |value| Value::I64(value as i64));
    primitive!(Int64Array, Value::I64);
    primitive!(UInt8Array, |value| Value::U64(value as u64));
    primitive!(UInt16Array, |value| Value::U64(value as u64));
    primitive!(UInt32Array, |value| Value::U64(value as u64));
    primitive!(UInt64Array, Value::U64);
    primitive!(Float32Array, |value| Value::F64(value as f64));
    primitive!(Float64Array, Value::F64);

    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        return string_array_to_scalars(array);
    }
    if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        return string_array_to_scalars(array);
    }
    if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
        return string_array_to_scalars(array);
    }
    if let Some(array) = array.as_any().downcast_ref::<ListArray>() {
        return (0..array.len())
            .map(|row| {
                Ok(if array.is_null(row) {
                    Value::Null
                } else {
                    Value::List(array_to_scalars(&array.value(row))?)
                })
            })
            .collect();
    }
    if let Some(array) = array.as_any().downcast_ref::<LargeListArray>() {
        return (0..array.len())
            .map(|row| {
                Ok(if array.is_null(row) {
                    Value::Null
                } else {
                    Value::List(array_to_scalars(&array.value(row))?)
                })
            })
            .collect();
    }

    bail!("unsupported list value type: {:?}", array.data_type())
}

fn string_array_to_scalars<T: ArrayString>(array: &T) -> Result<Vec<Value>> {
    Ok((0..array.len())
        .map(|row| {
            if array.is_null(row) {
                Value::Null
            } else {
                Value::Str(Arc::from(array.str_value(row)))
            }
        })
        .collect())
}

trait ArrayString: Array {
    fn str_value(&self, row: usize) -> &str;
}

impl ArrayString for StringArray {
    fn str_value(&self, row: usize) -> &str {
        self.value(row)
    }
}

impl ArrayString for LargeStringArray {
    fn str_value(&self, row: usize) -> &str {
        self.value(row)
    }
}

impl ArrayString for StringViewArray {
    fn str_value(&self, row: usize) -> &str {
        self.value(row)
    }
}

fn eval_str_pred(
    left: &Expr<BoundColumn>,
    op: StrOp,
    right: &Expr<BoundColumn>,
    ctx: &EvalCtx<'_>,
) -> Result<Value> {
    let Some(left) = left.str_value(ctx)? else {
        return Ok(Value::Null);
    };
    let Some(right) = right.str_value(ctx)? else {
        return Ok(Value::Null);
    };

    // Borrow Arrow string values directly; string predicates are common and should not allocate per edge.
    Ok(Value::Bool(match op {
        StrOp::Contains => left.contains(right),
        StrOp::StartsWith => left.starts_with(right),
        StrOp::EndsWith => left.ends_with(right),
    }))
}

fn eval_str_binary(
    left: &Expr<BoundColumn>,
    op: BinaryOp,
    right: &Expr<BoundColumn>,
    ctx: &EvalCtx<'_>,
) -> Result<Option<Value>> {
    if !matches!(
        op,
        BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
    ) {
        return Ok(None);
    }

    let Some(left) = left.try_str_value(ctx)? else {
        return Ok(None);
    };
    let Some(right) = right.try_str_value(ctx)? else {
        return Ok(None);
    };

    // Simple string comparisons borrow Arrow/literal values instead of materializing Value::Str.
    Ok(Some(match (left, right) {
        (None, None) => match op {
            BinaryOp::Eq => Value::Bool(true),
            BinaryOp::NotEq => Value::Bool(false),
            _ => Value::Null,
        },
        (None, _) | (_, None) => match op {
            BinaryOp::Eq => Value::Bool(false),
            BinaryOp::NotEq => Value::Bool(true),
            _ => Value::Null,
        },
        (Some(left), Some(right)) => match op {
            BinaryOp::Eq => Value::Bool(left == right),
            BinaryOp::NotEq => Value::Bool(left != right),
            BinaryOp::Lt => Value::Bool(left < right),
            BinaryOp::LtEq => Value::Bool(left <= right),
            BinaryOp::Gt => Value::Bool(left > right),
            BinaryOp::GtEq => Value::Bool(left >= right),
            _ => return Ok(None),
        },
    }))
}

fn eval_binary(left: Value, op: BinaryOp, right: Value) -> Result<Value> {
    if matches!(left, Value::Null) || matches!(right, Value::Null) {
        return Ok(match op {
            BinaryOp::Eq => Value::Bool(left == right),
            BinaryOp::NotEq => Value::Bool(left != right),
            _ => Value::Null,
        });
    }

    Ok(match op {
        BinaryOp::Eq => Value::Bool(scalar_eq(&left, &right)),
        BinaryOp::NotEq => Value::Bool(!scalar_eq(&left, &right)),
        BinaryOp::Lt => Value::Bool(compare(&left, &right)?.is_lt()),
        BinaryOp::LtEq => Value::Bool(compare(&left, &right)?.is_le()),
        BinaryOp::Gt => Value::Bool(compare(&left, &right)?.is_gt()),
        BinaryOp::GtEq => Value::Bool(compare(&left, &right)?.is_ge()),
        BinaryOp::Plus => numeric(left, right, |a, b| a + b, |a, b| a + b)?,
        BinaryOp::Minus => numeric(left, right, |a, b| a - b, |a, b| a - b)?,
        BinaryOp::Multiply => numeric(left, right, |a, b| a * b, |a, b| a * b)?,
        BinaryOp::Divide => Value::F64(number(&left)? / number(&right)?),
        BinaryOp::Modulus => Value::I64((integer(&left)? % integer(&right)?) as i64),
        BinaryOp::BitAnd => bitwise(left, right, |a, b| a & b)?,
        BinaryOp::BitOr => bitwise(left, right, |a, b| a | b)?,
        BinaryOp::BitXor => bitwise(left, right, |a, b| a ^ b)?,
        BinaryOp::And | BinaryOp::Or => unreachable!("logical ops are short-circuited"),
    })
}

fn scalar_eq(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Bool(left), Value::Bool(right)) => left == right,
        (Value::Str(left), Value::Str(right)) => left == right,
        (Value::List(left), Value::List(right)) => left == right,
        _ => left
            .as_f64()
            .zip(right.as_f64())
            .is_some_and(|(left, right)| left == right),
    }
}

fn integer(value: &Value) -> Result<i128> {
    value.as_i128().context("expected integer")
}

fn number(value: &Value) -> Result<f64> {
    value.as_f64().context("expected number")
}

fn numeric(
    left: Value,
    right: Value,
    int: impl FnOnce(i128, i128) -> i128,
    float: impl FnOnce(f64, f64) -> f64,
) -> Result<Value> {
    if matches!(left, Value::F64(_)) || matches!(right, Value::F64(_)) {
        Ok(Value::F64(float(number(&left)?, number(&right)?)))
    } else {
        let value = int(integer(&left)?, integer(&right)?);
        Ok(if value >= 0 {
            Value::U64(value as u64)
        } else {
            Value::I64(value as i64)
        })
    }
}

fn bitwise(left: Value, right: Value, op: impl FnOnce(u64, u64) -> u64) -> Result<Value> {
    Ok(Value::U64(op(
        left.as_u64().context("expected unsigned integer")?,
        right.as_u64().context("expected unsigned integer")?,
    )))
}

fn compare(left: &Value, right: &Value) -> Result<std::cmp::Ordering> {
    match (left, right) {
        (Value::Str(left), Value::Str(right)) => Ok(left.cmp(right)),
        _ => number(left)?
            .partial_cmp(&number(right)?)
            .context("cannot compare NaN"),
    }
}

fn parse_polars_json(json: &str) -> Result<Expr<ColumnRef>> {
    parse_expr(&serde_json::from_str(json)?)
}

fn parse_expr(value: &Json) -> Result<Expr<ColumnRef>> {
    let object = value.as_object().context("expected expression object")?;
    if let Some(column) = object.get("Column") {
        return parse_column(column.as_str().context("Column must be a string")?);
    }
    if let Some(literal) = object.get("Literal") {
        return Ok(Expr::Literal(parse_literal(literal)?));
    }
    if let Some(binary) = object.get("BinaryExpr").and_then(Json::as_object) {
        return Ok(Expr::Binary(
            Box::new(parse_expr(
                binary.get("left").context("BinaryExpr missing left")?,
            )?),
            parse_binary_op(binary.get("op").context("BinaryExpr missing op")?)?,
            Box::new(parse_expr(
                binary.get("right").context("BinaryExpr missing right")?,
            )?),
        ));
    }
    if let Some(rx) = object.get("Rx") {
        return parse_rx_expr(rx);
    }
    if let Some(function) = object.get("Function") {
        return parse_function(function);
    }
    bail!("unsupported Polars expression JSON: {value}")
}

fn parse_column(name: &str) -> Result<Expr<ColumnRef>> {
    Ok(Expr::Column(match name {
        "src.id" => ColumnRef::SrcId,
        "dest.id" => ColumnRef::DestId,
        "edge.id" => ColumnRef::EdgeId,
        _ => {
            let (scope, field) = name
                .split_once('.')
                .ok_or_else(|| anyhow!("column {name:?} must use scope.field"))?;
            match scope {
                "src" => ColumnRef::SrcField(field.to_owned()),
                "dest" => ColumnRef::DestField(field.to_owned()),
                "edge" => ColumnRef::EdgeField(field.to_owned()),
                "state" => ColumnRef::State(field.to_owned()),
                _ => bail!("unknown column scope {scope:?} in {name:?}"),
            }
        }
    }))
}

fn parse_binary_op(value: &Json) -> Result<BinaryOp> {
    Ok(
        match value.as_str().context("binary op must be a string")? {
            "And" => BinaryOp::And,
            "Or" => BinaryOp::Or,
            "Eq" => BinaryOp::Eq,
            "NotEq" => BinaryOp::NotEq,
            "Lt" => BinaryOp::Lt,
            "LtEq" => BinaryOp::LtEq,
            "Gt" => BinaryOp::Gt,
            "GtEq" => BinaryOp::GtEq,
            "Plus" => BinaryOp::Plus,
            "Minus" => BinaryOp::Minus,
            "Multiply" => BinaryOp::Multiply,
            "Divide" => BinaryOp::Divide,
            "Modulus" => BinaryOp::Modulus,
            "BitAnd" => BinaryOp::BitAnd,
            "BitOr" => BinaryOp::BitOr,
            "BitXor" => BinaryOp::BitXor,
            op => bail!("unsupported binary op {op:?}"),
        },
    )
}

fn parse_rx_expr(value: &Json) -> Result<Expr<ColumnRef>> {
    let object = value
        .as_object()
        .context("Rx expression must be an object")?;
    let value = object
        .get("MaskIfAny")
        .context("unsupported Rx expression")?
        .as_object()
        .context("MaskIfAny expression must be an object")?;
    Ok(Expr::MaskIfAny {
        value: Box::new(parse_expr(
            value.get("value").context("MaskIfAny missing value")?,
        )?),
        mask: Box::new(parse_expr(
            value.get("mask").context("MaskIfAny missing mask")?,
        )?),
        then_value: Box::new(parse_expr(
            value
                .get("then")
                .or_else(|| value.get("then_value"))
                .context("MaskIfAny missing then")?,
        )?),
    })
}

fn parse_literal(value: &Json) -> Result<Value> {
    parse_scalar_literal(
        value
            .pointer("/Value")
            .or_else(|| value.pointer("/Scalar"))
            .or_else(|| value.pointer("/Dyn"))
            .context("unsupported literal")?,
    )
}

fn parse_scalar_literal(value: &Json) -> Result<Value> {
    let object = value.as_object().context("literal must be object")?;
    if let Some(value) = object.get("String") {
        return Ok(Value::Str(Arc::from(
            value.as_str().context("String literal must be string")?,
        )));
    }
    if let Some(value) = object.get("Boolean") {
        return Ok(Value::Bool(
            value.as_bool().context("Boolean literal must be bool")?,
        ));
    }
    if object.contains_key("Null") {
        return Ok(Value::Null);
    }
    if let Some(value) = object.get("Int") {
        return Ok(Value::I64(
            value.as_i64().context("Int literal must be i64")?,
        ));
    }
    if let Some(value) = object.get("UInt") {
        return Ok(Value::U64(
            value.as_u64().context("UInt literal must be u64")?,
        ));
    }
    if let Some(value) = object.get("Float") {
        return Ok(Value::F64(
            value.as_f64().context("Float literal must be f64")?,
        ));
    }
    bail!("unsupported literal {value}")
}

fn parse_function(value: &Json) -> Result<Expr<ColumnRef>> {
    let object = value.as_object().context("Function must be an object")?;
    let inputs = object
        .get("input")
        .and_then(Json::as_array)
        .context("Function missing input array")?;
    let function = object
        .get("function")
        .context("Function missing function")?;

    if function.pointer("/Boolean").and_then(Json::as_str) == Some("Not") {
        return unary(inputs, Expr::Not);
    }
    if function.pointer("/Boolean").and_then(Json::as_str) == Some("IsNull") {
        return unary(inputs, Expr::IsNull);
    }
    if function.pointer("/Boolean").and_then(Json::as_str) == Some("IsNotNull") {
        return unary(inputs, Expr::IsNotNull);
    }
    if function.pointer("/StringExpr/Contains").is_some() {
        return binary_fn(inputs, StrOp::Contains);
    }
    if function.pointer("/StringExpr/StartsWith").is_some() {
        return binary_fn(inputs, StrOp::StartsWith);
    }
    if function.pointer("/StringExpr/EndsWith").is_some() {
        return binary_fn(inputs, StrOp::EndsWith);
    }
    if function.pointer("/ListExpr").and_then(Json::as_str) == Some("Concat") {
        return Ok(Expr::ListConcat(
            inputs.iter().map(parse_expr).collect::<Result<_>>()?,
        ));
    }
    bail!("unsupported Polars function {function}")
}

fn unary(
    inputs: &[Json],
    wrap: impl FnOnce(Box<Expr<ColumnRef>>) -> Expr<ColumnRef>,
) -> Result<Expr<ColumnRef>> {
    ensure_arity(inputs, 1)?;
    Ok(wrap(Box::new(parse_expr(&inputs[0])?)))
}

fn binary_fn(inputs: &[Json], op: StrOp) -> Result<Expr<ColumnRef>> {
    ensure_arity(inputs, 2)?;
    Ok(Expr::StrPred(
        Box::new(parse_expr(&inputs[0])?),
        op,
        Box::new(parse_expr(&inputs[1])?),
    ))
}

fn ensure_arity(inputs: &[Json], len: usize) -> Result<()> {
    if inputs.len() == len {
        Ok(())
    } else {
        bail!("expected {len} function inputs, got {}", inputs.len())
    }
}

fn state_names(initial: &StateRow, next: &[(String, Expr<ColumnRef>)]) -> Vec<String> {
    let mut names = initial
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    names.extend(next.iter().map(|(name, _)| name.clone()));
    names.sort();
    names.dedup();
    names
}

fn normalize_state(state: StateRow, names: &[String]) -> StateValues {
    names
        .iter()
        .map(|name| {
            state
                .binary_search_by(|(key, _)| key.as_str().cmp(name))
                .ok()
                .map(|i| state[i].1.clone())
                .unwrap_or(Value::Null)
        })
        .collect::<StateValues>()
}

fn state_index(names: &[String], name: &str) -> Option<usize> {
    names.binary_search_by(|key| key.as_str().cmp(name)).ok()
}

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
                .eq(DslExpr::string("target"))
                .and(DslExpr::edge_id().eq(DslExpr::string("ab"))),
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
                .and(DslExpr::edge_id().eq(DslExpr::uint(9))),
            [
                (
                    "bits".into(),
                    DslExpr::state("bits")
                        .mask_if_any(DslExpr::edge("from_mask"), DslExpr::edge("to_mask")),
                ),
                ("last".into(), DslExpr::dest_id()),
            ],
            DslExpr::state("bits").eq(DslExpr::uint(0b100)),
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
    fn string_predicates_borrow_arrow_values() {
        let kernel = DslKernel::new(
            DslExpr::bool(true),
            std::iter::empty::<(String, DslExpr)>(),
            DslExpr::dest("kind")
                .starts_with(DslExpr::string("tar"))
                .and(DslExpr::dest("kind").contains(DslExpr::string("get")))
                .and(DslExpr::dest("kind").ends_with(DslExpr::string("et"))),
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
    fn unsupported_json_ops_fail_clearly() {
        let err = DslExpr::from_polars_json(
            r#"{"BinaryExpr":{"left":{"Column":"dest.kind"},"op":"Pow","right":{"Literal":{"Dyn":{"String":"target"}}}}}"#,
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("unsupported binary op"));
    }
}
