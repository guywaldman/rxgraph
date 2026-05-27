//! Expression kernel parsing and evaluation.
//!
//! The DSL consumes serialized Polars expression JSON for three traversal
//! hooks: whether an edge may be visited, how traversal state changes after an
//! accepted edge, and whether an accepted edge stops a path.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, Int32Array, Int64Array, LargeStringArray, StringArray,
        StringViewArray, UInt64Array,
    },
    datatypes::DataType,
};
use serde_json::Value as Json;
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};

use crate::{
    Graph,
    graph::{EdgeId, NodeId},
};

/// Traversal kernel built from serialized expression JSON.
///
/// Kernels are independent of any particular graph until search time. Binding a
/// kernel to a graph resolves Arrow column readers and state indexes once, so
/// edge evaluation can stay tight inside traversal loops.
#[derive(Debug, Clone)]
pub struct DslKernel {
    visit: Expr,
    next_state: Vec<(String, Expr)>,
    stop: Expr,
    initial_state: StateRow,
}

impl DslKernel {
    /// Creates a traversal kernel from serialized expressions.
    ///
    /// `visit_json` must evaluate to a boolean. If it returns `false`, the edge
    /// is rejected. `next_state` contains named expressions used to update the
    /// per-path state after an edge is accepted. `stop_json` must evaluate to a
    /// boolean and determines whether the accepted edge materializes a path.
    ///
    /// State fields are referenced as `state.<name>`, node fields as
    /// `src.<column>` / `dest.<column>`, edge fields as `edge.<column>`, and IDs
    /// as `src.id`, `dest.id`, and `edge.id`.
    pub fn new(
        visit_json: &str,
        next_state: impl IntoIterator<Item = (String, String)>,
        stop_json: &str,
        initial_state: impl IntoIterator<Item = (String, Scalar)>,
    ) -> Result<Self> {
        let mut initial_state = initial_state.into_iter().collect::<StateRow>();
        initial_state.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(Self {
            visit: parse_polars_json(visit_json).context("invalid visit expression")?,
            next_state: next_state
                .into_iter()
                .map(|(name, json)| Ok((name, parse_polars_json(&json)?)))
                .collect::<Result<_>>()
                .context("invalid next_state expression")?,
            stop: parse_polars_json(stop_json).context("invalid stop expression")?,
            initial_state,
        })
    }
}

/// Sorted state row carried by a path during traversal.
pub type StateRow = Vec<(String, Scalar)>;

/// Scalar value type supported by expression evaluation.
///
/// This deliberately mirrors only the Arrow/Polars scalar subset needed by the
/// traversal engine today.
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    /// Missing or null value.
    Null,
    /// Boolean value.
    Bool(bool),
    /// Signed 64-bit integer.
    I64(i64),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// 64-bit floating point number.
    F64(f64),
    /// UTF-8 string.
    Str(Arc<str>),
}

impl Scalar {
    /// Converts the scalar into a JSON value for path result state.
    pub fn to_value(&self) -> JsonValue {
        match self {
            Self::Null => JsonValue::Null,
            Self::Bool(value) => JsonValue::Bool(*value),
            Self::I64(value) => JsonValue::Number(JsonNumber::from(*value)),
            Self::U64(value) => JsonValue::Number(JsonNumber::from(*value)),
            Self::F64(value) => JsonNumber::from_f64(*value)
                .map(JsonValue::Number)
                .unwrap_or(JsonValue::Null),
            Self::Str(value) => JsonValue::String(value.to_string()),
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
            Self::I64(value) => (*value >= 0).then_some(*value as u64),
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
enum Expr {
    Column(ColumnRef),
    Literal(Scalar),
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    MaskIfAny {
        value: Box<Expr>,
        mask: Box<Expr>,
        then_value: Box<Expr>,
    },
    Not(Box<Expr>),
    IsNull(Box<Expr>),
    IsNotNull(Box<Expr>),
    StrContains(Box<Expr>, Box<Expr>),
    StrStartsWith(Box<Expr>, Box<Expr>),
    StrEndsWith(Box<Expr>, Box<Expr>),
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

#[derive(Debug)]
pub(crate) struct BoundKernel {
    visit: BoundExpr,
    next_state: Vec<(usize, BoundExpr)>,
    stop: BoundExpr,
    initial_state: StateRow,
}

#[derive(Debug, Clone)]
enum BoundExpr {
    Column(BoundColumn),
    Literal(Scalar),
    Binary(Box<BoundExpr>, BinaryOp, Box<BoundExpr>),
    MaskIfAny {
        value: Box<BoundExpr>,
        mask: Box<BoundExpr>,
        then_value: Box<BoundExpr>,
    },
    Not(Box<BoundExpr>),
    IsNull(Box<BoundExpr>),
    IsNotNull(Box<BoundExpr>),
    StrContains(Box<BoundExpr>, Box<BoundExpr>),
    StrStartsWith(Box<BoundExpr>, Box<BoundExpr>),
    StrEndsWith(Box<BoundExpr>, Box<BoundExpr>),
}

#[derive(Debug, Clone)]
enum BoundColumn {
    SrcId,
    DestId,
    EdgeId,
    Src(Vec<Option<ColumnReader>>),
    Dest(Vec<Option<ColumnReader>>),
    Edge(Vec<Option<ColumnReader>>),
    State(usize),
    MissingState,
}

#[derive(Debug, Clone)]
enum ColumnReader {
    Bool(BooleanArray),
    I32(Int32Array),
    I64(Int64Array),
    U64(UInt64Array),
    Utf8(StringArray),
    LargeUtf8(LargeStringArray),
    Utf8View(StringViewArray),
}

impl DslKernel {
    pub(crate) fn bind(self, graph: &Graph) -> Result<BoundKernel> {
        BoundKernel::bind(graph, self)
    }
}

impl BoundKernel {
    fn bind(graph: &Graph, kernel: DslKernel) -> Result<Self> {
        let state_names = state_names(&kernel.initial_state, &kernel.next_state);
        let initial_state = normalize_state(kernel.initial_state, &state_names);

        Ok(Self {
            visit: BoundExpr::bind(graph, kernel.visit, &state_names)?,
            next_state: kernel
                .next_state
                .into_iter()
                .map(|(name, expr)| {
                    let index = state_index(&state_names, &name).unwrap();
                    Ok((index, BoundExpr::bind(graph, expr, &state_names)?))
                })
                .collect::<Result<_>>()?,
            stop: BoundExpr::bind(graph, kernel.stop, &state_names)?,
            initial_state,
        })
    }

    pub(crate) fn initial_state(&self) -> &StateRow {
        &self.initial_state
    }

    pub(crate) fn visit(&self, ctx: &EvalCtx<'_>) -> Result<bool> {
        self.visit.eval(ctx)?.truthy()
    }

    pub(crate) fn next_state(&self, current: &StateRow, ctx: &EvalCtx<'_>) -> Result<StateRow> {
        let mut next_state = current.clone();
        for (index, expr) in &self.next_state {
            next_state[*index].1 = expr.eval(ctx)?;
        }
        Ok(next_state)
    }

    pub(crate) fn stop(&self, ctx: &EvalCtx<'_>) -> Result<bool> {
        self.stop.eval(ctx)?.truthy()
    }
}

impl BoundExpr {
    fn bind(graph: &Graph, expr: Expr, state_names: &[String]) -> Result<Self> {
        Ok(match expr {
            Expr::Column(column) => Self::Column(BoundColumn::bind(graph, column, state_names)?),
            Expr::Literal(value) => Self::Literal(value),
            Expr::Binary(left, op, right) => Self::Binary(
                Box::new(Self::bind(graph, *left, state_names)?),
                op,
                Box::new(Self::bind(graph, *right, state_names)?),
            ),
            Expr::MaskIfAny {
                value,
                mask,
                then_value,
            } => Self::MaskIfAny {
                value: Box::new(Self::bind(graph, *value, state_names)?),
                mask: Box::new(Self::bind(graph, *mask, state_names)?),
                then_value: Box::new(Self::bind(graph, *then_value, state_names)?),
            },
            Expr::Not(expr) => Self::Not(Box::new(Self::bind(graph, *expr, state_names)?)),
            Expr::IsNull(expr) => Self::IsNull(Box::new(Self::bind(graph, *expr, state_names)?)),
            Expr::IsNotNull(expr) => {
                Self::IsNotNull(Box::new(Self::bind(graph, *expr, state_names)?))
            }
            Expr::StrContains(left, right) => Self::StrContains(
                Box::new(Self::bind(graph, *left, state_names)?),
                Box::new(Self::bind(graph, *right, state_names)?),
            ),
            Expr::StrStartsWith(left, right) => Self::StrStartsWith(
                Box::new(Self::bind(graph, *left, state_names)?),
                Box::new(Self::bind(graph, *right, state_names)?),
            ),
            Expr::StrEndsWith(left, right) => Self::StrEndsWith(
                Box::new(Self::bind(graph, *left, state_names)?),
                Box::new(Self::bind(graph, *right, state_names)?),
            ),
        })
    }

    fn eval(&self, ctx: &EvalCtx<'_>) -> Result<Scalar> {
        Ok(match self {
            Self::Column(column) => column.eval(ctx)?,
            Self::Literal(value) => value.clone(),
            Self::Binary(left, BinaryOp::And, right) => {
                if !left.eval(ctx)?.truthy()? {
                    Scalar::Bool(false)
                } else {
                    Scalar::Bool(right.eval(ctx)?.truthy()?)
                }
            }
            Self::Binary(left, BinaryOp::Or, right) => {
                if left.eval(ctx)?.truthy()? {
                    Scalar::Bool(true)
                } else {
                    Scalar::Bool(right.eval(ctx)?.truthy()?)
                }
            }
            Self::Binary(left, op, right) => eval_binary(left.eval(ctx)?, *op, right.eval(ctx)?)?,
            Self::MaskIfAny {
                value,
                mask,
                then_value,
            } => {
                let value = value
                    .eval(ctx)?
                    .as_u64()
                    .ok_or_else(|| anyhow!("mask_if_any value must be an unsigned integer"))?;
                let mask = mask
                    .eval(ctx)?
                    .as_u64()
                    .ok_or_else(|| anyhow!("mask_if_any mask must be an unsigned integer"))?;
                if value & mask == 0 {
                    Scalar::U64(0)
                } else {
                    then_value.eval(ctx)?
                }
            }
            Self::Not(expr) => Scalar::Bool(!expr.eval(ctx)?.truthy()?),
            Self::IsNull(expr) => Scalar::Bool(expr.eval(ctx)? == Scalar::Null),
            Self::IsNotNull(expr) => Scalar::Bool(expr.eval(ctx)? != Scalar::Null),
            Self::StrContains(left, right) => {
                eval_str_pred(left.eval(ctx)?, right.eval(ctx)?, |a, b| a.contains(b))?
            }
            Self::StrStartsWith(left, right) => {
                eval_str_pred(left.eval(ctx)?, right.eval(ctx)?, |a, b| a.starts_with(b))?
            }
            Self::StrEndsWith(left, right) => {
                eval_str_pred(left.eval(ctx)?, right.eval(ctx)?, |a, b| a.ends_with(b))?
            }
        })
    }
}

impl BoundColumn {
    fn bind(graph: &Graph, column: ColumnRef, state_names: &[String]) -> Result<Self> {
        Ok(match column {
            ColumnRef::SrcId => Self::SrcId,
            ColumnRef::DestId => Self::DestId,
            ColumnRef::EdgeId => Self::EdgeId,
            ColumnRef::State(name) => state_index(state_names, &name)
                .map(Self::State)
                .unwrap_or(Self::MissingState),
            ColumnRef::SrcField(name) => {
                Self::Src(bind_readers(graph.node_tables(), &name, "node")?)
            }
            ColumnRef::DestField(name) => {
                Self::Dest(bind_readers(graph.node_tables(), &name, "node")?)
            }
            ColumnRef::EdgeField(name) => {
                Self::Edge(bind_readers(graph.edge_tables(), &name, "edge")?)
            }
        })
    }

    fn eval(&self, ctx: &EvalCtx<'_>) -> Result<Scalar> {
        match self {
            Self::SrcId => Ok(Scalar::U64(ctx.graph.external_node(ctx.src))),
            Self::DestId => Ok(Scalar::U64(ctx.graph.external_node(ctx.dest))),
            Self::EdgeId => Ok(Scalar::U64(ctx.edge as u64)),
            Self::State(index) => Ok(ctx.state[*index].1.clone()),
            Self::MissingState => Ok(Scalar::Null),
            Self::Src(readers) => read_node(ctx.graph, readers, ctx.src),
            Self::Dest(readers) => read_node(ctx.graph, readers, ctx.dest),
            Self::Edge(readers) => {
                let row = ctx.graph.edge_row(ctx.edge);
                let reader = readers
                    .get(row.table)
                    .and_then(Option::as_ref)
                    .with_context(|| "edge table missing bound reader".to_string())?;
                reader.value(row.row)
            }
        }
    }
}

pub(crate) struct EvalCtx<'a> {
    pub(crate) graph: &'a Graph,
    pub(crate) src: NodeId,
    pub(crate) dest: NodeId,
    pub(crate) edge: EdgeId,
    pub(crate) state: &'a StateRow,
}

fn bind_readers(
    tables: &[crate::graph::Table],
    name: &str,
    kind: &str,
) -> Result<Vec<Option<ColumnReader>>> {
    let readers = tables
        .iter()
        .map(|table| match table.batch.schema().index_of(name) {
            Ok(index) => Ok(Some(ColumnReader::new(name, table.batch.column(index))?)),
            Err(_) => Ok(None),
        })
        .collect::<Result<Vec<_>>>()?;

    if readers.iter().any(Option::is_some) {
        Ok(readers)
    } else {
        bail!("column {name:?} is not present in any {kind} table")
    }
}

fn read_node(graph: &Graph, readers: &[Option<ColumnReader>], node: NodeId) -> Result<Scalar> {
    let row = graph.node_row(node);
    let reader = readers
        .get(row.table)
        .and_then(Option::as_ref)
        .with_context(|| "node table missing bound reader".to_string())?;
    reader.value(row.row)
}

impl ColumnReader {
    fn new(name: &str, column: &ArrayRef) -> Result<Self> {
        macro_rules! reader {
            ($variant:ident, $array:ty) => {{
                let array = column
                    .as_any()
                    .downcast_ref::<$array>()
                    .with_context(|| format!("column {name:?} does not match its Arrow type"))?;
                Self::$variant(array.clone())
            }};
        }

        Ok(match column.data_type() {
            DataType::Boolean => reader!(Bool, BooleanArray),
            DataType::Int32 => reader!(I32, Int32Array),
            DataType::Int64 => reader!(I64, Int64Array),
            DataType::UInt64 => reader!(U64, UInt64Array),
            DataType::Utf8 => reader!(Utf8, StringArray),
            DataType::LargeUtf8 => reader!(LargeUtf8, LargeStringArray),
            DataType::Utf8View => reader!(Utf8View, StringViewArray),
            typ => bail!("unsupported DSL column type for {name:?}: {typ:?}"),
        })
    }

    fn value(&self, row: usize) -> Result<Scalar> {
        macro_rules! primitive {
            ($array:expr, $variant:ident) => {{
                if $array.is_null(row) {
                    Ok(Scalar::Null)
                } else {
                    Ok(Scalar::$variant($array.value(row).into()))
                }
            }};
        }

        match self {
            Self::Bool(array) => primitive!(array, Bool),
            Self::I32(array) => {
                if array.is_null(row) {
                    Ok(Scalar::Null)
                } else {
                    Ok(Scalar::I64(array.value(row) as i64))
                }
            }
            Self::I64(array) => primitive!(array, I64),
            Self::U64(array) => primitive!(array, U64),
            Self::Utf8(array) => {
                if array.is_null(row) {
                    Ok(Scalar::Null)
                } else {
                    Ok(Scalar::Str(Arc::from(array.value(row))))
                }
            }
            Self::LargeUtf8(array) => {
                if array.is_null(row) {
                    Ok(Scalar::Null)
                } else {
                    Ok(Scalar::Str(Arc::from(array.value(row))))
                }
            }
            Self::Utf8View(array) => {
                if array.is_null(row) {
                    Ok(Scalar::Null)
                } else {
                    Ok(Scalar::Str(Arc::from(array.value(row))))
                }
            }
        }
    }
}

fn eval_binary(left: Scalar, op: BinaryOp, right: Scalar) -> Result<Scalar> {
    if matches!(left, Scalar::Null) || matches!(right, Scalar::Null) {
        return Ok(match op {
            BinaryOp::And => Scalar::Bool(left.truthy()? && right.truthy()?),
            BinaryOp::Or => Scalar::Bool(left.truthy()? || right.truthy()?),
            BinaryOp::Eq => Scalar::Bool(scalar_eq(&left, &right)?),
            BinaryOp::NotEq => Scalar::Bool(!scalar_eq(&left, &right)?),
            _ => Scalar::Null,
        });
    }

    Ok(match op {
        BinaryOp::And => Scalar::Bool(left.truthy()? && right.truthy()?),
        BinaryOp::Or => Scalar::Bool(left.truthy()? || right.truthy()?),
        BinaryOp::Eq => Scalar::Bool(scalar_eq(&left, &right)?),
        BinaryOp::NotEq => Scalar::Bool(!scalar_eq(&left, &right)?),
        BinaryOp::Lt => Scalar::Bool(compare(&left, &right)?.is_lt()),
        BinaryOp::LtEq => Scalar::Bool(compare(&left, &right)?.is_le()),
        BinaryOp::Gt => Scalar::Bool(compare(&left, &right)?.is_gt()),
        BinaryOp::GtEq => Scalar::Bool(compare(&left, &right)?.is_ge()),
        BinaryOp::Plus => numeric(left, right, |a, b| a + b, |a, b| a + b)?,
        BinaryOp::Minus => numeric(left, right, |a, b| a - b, |a, b| a - b)?,
        BinaryOp::Multiply => numeric(left, right, |a, b| a * b, |a, b| a * b)?,
        BinaryOp::Divide => Scalar::F64(left.as_f64().unwrap() / right.as_f64().unwrap()),
        BinaryOp::Modulus => {
            let left = left
                .as_i128()
                .ok_or_else(|| anyhow!("modulus requires integers"))?;
            let right = right
                .as_i128()
                .ok_or_else(|| anyhow!("modulus requires integers"))?;
            Scalar::I64((left % right) as i64)
        }
        BinaryOp::BitAnd => bitwise(left, right, |a, b| a & b)?,
        BinaryOp::BitOr => bitwise(left, right, |a, b| a | b)?,
        BinaryOp::BitXor => bitwise(left, right, |a, b| a ^ b)?,
    })
}

fn scalar_eq(left: &Scalar, right: &Scalar) -> Result<bool> {
    match (left, right) {
        (Scalar::Null, Scalar::Null) => Ok(true),
        (Scalar::Bool(left), Scalar::Bool(right)) => Ok(left == right),
        (Scalar::Str(left), Scalar::Str(right)) => Ok(left == right),
        _ if left.as_f64().is_some() && right.as_f64().is_some() => {
            Ok(left.as_f64().unwrap() == right.as_f64().unwrap())
        }
        _ => Ok(false),
    }
}

fn numeric(
    left: Scalar,
    right: Scalar,
    int: impl FnOnce(i128, i128) -> i128,
    float: impl FnOnce(f64, f64) -> f64,
) -> Result<Scalar> {
    if matches!(left, Scalar::F64(_)) || matches!(right, Scalar::F64(_)) {
        Ok(Scalar::F64(float(
            left.as_f64()
                .ok_or_else(|| anyhow!("numeric op expected number"))?,
            right
                .as_f64()
                .ok_or_else(|| anyhow!("numeric op expected number"))?,
        )))
    } else {
        let value = int(
            left.as_i128()
                .ok_or_else(|| anyhow!("numeric op expected integer"))?,
            right
                .as_i128()
                .ok_or_else(|| anyhow!("numeric op expected integer"))?,
        );
        if value >= 0 {
            Ok(Scalar::U64(value as u64))
        } else {
            Ok(Scalar::I64(value as i64))
        }
    }
}

fn bitwise(left: Scalar, right: Scalar, op: impl FnOnce(u64, u64) -> u64) -> Result<Scalar> {
    let left = left
        .as_u64()
        .ok_or_else(|| anyhow!("bitwise op expected unsigned integer"))?;
    let right = right
        .as_u64()
        .ok_or_else(|| anyhow!("bitwise op expected unsigned integer"))?;
    Ok(Scalar::U64(op(left, right)))
}

fn compare(left: &Scalar, right: &Scalar) -> Result<std::cmp::Ordering> {
    match (left, right) {
        (Scalar::Str(left), Scalar::Str(right)) => Ok(left.cmp(right)),
        _ => {
            let left = left
                .as_f64()
                .ok_or_else(|| anyhow!("comparison expected number"))?;
            let right = right
                .as_f64()
                .ok_or_else(|| anyhow!("comparison expected number"))?;
            left.partial_cmp(&right)
                .ok_or_else(|| anyhow!("cannot compare NaN"))
        }
    }
}

fn eval_str_pred(
    left: Scalar,
    right: Scalar,
    pred: impl FnOnce(&str, &str) -> bool,
) -> Result<Scalar> {
    match (left, right) {
        (Scalar::Null, _) | (_, Scalar::Null) => Ok(Scalar::Null),
        (Scalar::Str(left), Scalar::Str(right)) => Ok(Scalar::Bool(pred(&left, &right))),
        _ => bail!("string predicate expected strings"),
    }
}

fn parse_polars_json(json: &str) -> Result<Expr> {
    let value = serde_json::from_str(json)?;
    parse_expr(&value)
}

fn parse_expr(value: &Json) -> Result<Expr> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("expected expression object"))?;

    if let Some(column) = object.get("Column") {
        return parse_column(column.as_str().context("Column must be a string")?);
    }
    if let Some(literal) = object.get("Literal") {
        return Ok(Expr::Literal(parse_literal(literal)?));
    }
    if let Some(binary) = object.get("BinaryExpr") {
        let binary = binary.as_object().context("BinaryExpr must be an object")?;
        let left = parse_expr(binary.get("left").context("BinaryExpr missing left")?)?;
        let right = parse_expr(binary.get("right").context("BinaryExpr missing right")?)?;
        let op = parse_binary_op(binary.get("op").context("BinaryExpr missing op")?)?;

        return Ok(Expr::Binary(Box::new(left), op, Box::new(right)));
    }
    if let Some(rx) = object.get("Rx") {
        return parse_rx_expr(rx);
    }
    if let Some(function) = object.get("Function") {
        return parse_function(function);
    }

    bail!("unsupported Polars expression JSON: {value}")
}

fn parse_column(name: &str) -> Result<Expr> {
    Ok(Expr::Column(match name {
        "src.id" => ColumnRef::SrcId,
        "dest.id" => ColumnRef::DestId,
        "edge.id" => ColumnRef::EdgeId,
        _ => {
            let (scope, field) = name
                .split_once('.')
                .ok_or_else(|| anyhow!("column {name:?} must use scope.field"))?;
            match scope {
                "src" => ColumnRef::SrcField(field.to_string()),
                "dest" => ColumnRef::DestField(field.to_string()),
                "edge" => ColumnRef::EdgeField(field.to_string()),
                "state" => ColumnRef::State(field.to_string()),
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

fn parse_rx_expr(value: &Json) -> Result<Expr> {
    let object = value
        .as_object()
        .context("Rx expression must be an object")?;
    if let Some(value) = object.get("MaskIfAny") {
        let value = value
            .as_object()
            .context("MaskIfAny expression must be an object")?;
        return Ok(Expr::MaskIfAny {
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
        });
    }

    bail!("unsupported Rx expression {value}")
}

fn parse_literal(value: &Json) -> Result<Scalar> {
    if let Some(scalar) = value.pointer("/Scalar") {
        return parse_scalar_literal(scalar);
    }
    if let Some(dyn_value) = value.pointer("/Dyn") {
        return parse_dyn_literal(dyn_value);
    }
    bail!("unsupported literal {value}")
}

fn parse_scalar_literal(value: &Json) -> Result<Scalar> {
    let object = value.as_object().context("Scalar literal must be object")?;
    if let Some(value) = object.get("String") {
        return Ok(Scalar::Str(Arc::from(
            value.as_str().context("String literal must be string")?,
        )));
    }
    if let Some(value) = object.get("Boolean") {
        return Ok(Scalar::Bool(
            value.as_bool().context("Boolean literal must be bool")?,
        ));
    }
    if object.contains_key("Null") {
        return Ok(Scalar::Null);
    }
    parse_dyn_literal(value)
}

fn parse_dyn_literal(value: &Json) -> Result<Scalar> {
    let object = value.as_object().context("Dyn literal must be object")?;
    if let Some(value) = object.get("Int") {
        return Ok(Scalar::I64(
            value.as_i64().context("Int literal must be i64")?,
        ));
    }
    if let Some(value) = object.get("UInt") {
        return Ok(Scalar::U64(
            value.as_u64().context("UInt literal must be u64")?,
        ));
    }
    if let Some(value) = object.get("Float") {
        return Ok(Scalar::F64(
            value.as_f64().context("Float literal must be f64")?,
        ));
    }
    if let Some(value) = object.get("String") {
        return Ok(Scalar::Str(Arc::from(
            value.as_str().context("String literal must be string")?,
        )));
    }
    if let Some(value) = object.get("Boolean") {
        return Ok(Scalar::Bool(
            value.as_bool().context("Boolean literal must be bool")?,
        ));
    }
    bail!("unsupported dynamic literal {value}")
}

fn parse_function(value: &Json) -> Result<Expr> {
    let object = value.as_object().context("Function must be an object")?;
    let inputs = object
        .get("input")
        .and_then(Json::as_array)
        .context("Function missing input array")?;
    let function = object
        .get("function")
        .context("Function missing function")?;

    if function.pointer("/Boolean").and_then(Json::as_str) == Some("Not") {
        ensure_arity(inputs, 1)?;
        return Ok(Expr::Not(Box::new(parse_expr(&inputs[0])?)));
    }
    if function.pointer("/Boolean").and_then(Json::as_str) == Some("IsNull") {
        ensure_arity(inputs, 1)?;
        return Ok(Expr::IsNull(Box::new(parse_expr(&inputs[0])?)));
    }
    if function.pointer("/Boolean").and_then(Json::as_str) == Some("IsNotNull") {
        ensure_arity(inputs, 1)?;
        return Ok(Expr::IsNotNull(Box::new(parse_expr(&inputs[0])?)));
    }
    if function.pointer("/StringExpr/Contains").is_some() {
        ensure_arity(inputs, 2)?;
        return Ok(Expr::StrContains(
            Box::new(parse_expr(&inputs[0])?),
            Box::new(parse_expr(&inputs[1])?),
        ));
    }
    if function.pointer("/StringExpr/StartsWith").is_some() {
        ensure_arity(inputs, 2)?;
        return Ok(Expr::StrStartsWith(
            Box::new(parse_expr(&inputs[0])?),
            Box::new(parse_expr(&inputs[1])?),
        ));
    }
    if function.pointer("/StringExpr/EndsWith").is_some() {
        ensure_arity(inputs, 2)?;
        return Ok(Expr::StrEndsWith(
            Box::new(parse_expr(&inputs[0])?),
            Box::new(parse_expr(&inputs[1])?),
        ));
    }

    bail!("unsupported Polars function {function}")
}

fn ensure_arity(inputs: &[Json], len: usize) -> Result<()> {
    if inputs.len() == len {
        Ok(())
    } else {
        bail!("expected {len} function inputs, got {}", inputs.len())
    }
}

fn state_names(initial_state: &StateRow, next_state: &[(String, Expr)]) -> Vec<String> {
    let mut names = initial_state
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();

    for (name, _) in next_state {
        if state_index(&names, name).is_none() {
            names.push(name.clone());
        }
    }

    names.sort();
    names.dedup();
    names
}

fn normalize_state(state: StateRow, names: &[String]) -> StateRow {
    names
        .iter()
        .map(|name| {
            (
                name.clone(),
                get_state(&state, name).cloned().unwrap_or(Scalar::Null),
            )
        })
        .collect()
}

fn state_index(state_names: &[String], name: &str) -> Option<usize> {
    state_names
        .binary_search_by(|key| key.as_str().cmp(name))
        .ok()
}

fn get_state<'a>(state: &'a StateRow, name: &str) -> Option<&'a Scalar> {
    state
        .binary_search_by(|(key, _)| key.as_str().cmp(name))
        .ok()
        .map(|index| &state[index].1)
}

pub(crate) fn state_to_value(state: &StateRow) -> JsonValue {
    JsonValue::Object(
        state
            .iter()
            .map(|(name, value)| (name.clone(), value.to_value()))
            .collect::<JsonMap<_, _>>(),
    )
}
