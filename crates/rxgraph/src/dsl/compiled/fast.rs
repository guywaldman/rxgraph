use anyhow::{Context, Result, bail};
use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    PrimitiveArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::ArrowPrimitiveType;

use crate::dsl::{
    StateValue, StateValues, Value, arrow_value::ColumnReader, bind::BoundColumn, eval::EvalCtx,
    expr::Expr, ops::scalar::ScalarOp,
};
use crate::graph::{EdgeId, Graph, GraphId, GraphRepo, NodeId};

pub(crate) type FastStateValues = smallvec::SmallVec<[FastScalar; 8]>;

#[derive(Debug, Clone)]
pub(crate) enum FastBool {
    Literal(bool),
    Column(FastBoolReader),
    NotColumn(FastBoolReader),
    Not(Box<FastBool>),
    All(Vec<FastBool>),
    Any(Vec<FastBool>),
    NumCmp(NumCmp, FastNum, FastNum),
}

#[derive(Debug, Clone)]
pub(crate) enum FastNum {
    Literal(FastScalar),
    Column(FastNumReader),
    Binary(NumOp, Box<FastNum>, Box<FastNum>),
}

#[derive(Debug, Clone)]
pub(crate) enum FastBoolReader {
    Array {
        scope: RowScope,
        nullable: bool,
        array: BooleanArray,
    },
    MissingState,
}

#[derive(Debug, Clone)]
pub(crate) enum FastNumReader {
    Array {
        scope: RowScope,
        array: FastNumArray,
    },
    State(usize),
    MissingState,
    SrcId,
    DestId,
    EdgeId,
}

// The fast numeric reader wants one enum dispatch per Arrow type, not one
// hand-written arm for every (src/dest/edge) x type pair.
macro_rules! fast_num_arrays {
    (
        signed { $($signed_variant:ident($signed_array:ty)),* $(,)? }
        unsigned { $($unsigned_variant:ident($unsigned_array:ty)),* $(,)? }
        float { $($float_variant:ident($float_array:ty)),* $(,)? }
    ) => {
        #[derive(Debug, Clone)]
        pub(crate) enum FastNumArray {
            $($signed_variant($signed_array),)*
            $($unsigned_variant($unsigned_array),)*
            $($float_variant($float_array),)*
        }

        impl FastNumArray {
            fn compile(reader: &ColumnReader) -> Option<Self> {
                Some(match reader {
                    $(ColumnReader::$signed_variant(array) => Self::$signed_variant(array.clone()),)*
                    $(ColumnReader::$unsigned_variant(array) => Self::$unsigned_variant(array.clone()),)*
                    $(ColumnReader::$float_variant(array) => Self::$float_variant(array.clone()),)*
                    _ => return None,
                })
            }

            fn read(&self, row: usize) -> Option<FastScalar> {
                match self {
                    $(Self::$signed_variant(array) => scalar_i64(array, row, |value| value as i64),)*
                    $(Self::$unsigned_variant(array) => scalar_u64(array, row, |value| value as u64),)*
                    $(Self::$float_variant(array) => scalar_f64(array, row, |value| value as f64),)*
                }
            }
        }
    };
}

fast_num_arrays! {
    signed {
        I8(Int8Array),
        I16(Int16Array),
        I32(Int32Array),
        I64(Int64Array),
    }
    unsigned {
        U8(UInt8Array),
        U16(UInt16Array),
        U32(UInt32Array),
        U64(UInt64Array),
    }
    float {
        F32(Float32Array),
        F64(Float64Array),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum FastScalar {
    I64(i64),
    U64(u64),
    F64(f64),
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum NumOp {
    Plus,
    Minus,
    Multiply,
    Divide,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum NumCmp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

#[derive(Debug, Clone)]
pub(crate) struct FastScalarKernel {
    visit: FastBool,
    next_state: Vec<(usize, FastNum)>,
    stop: FastBool,
    initial_state: FastStateValues,
}

#[derive(Debug, Clone)]
pub(crate) struct FastEdgeDecision {
    pub(crate) state: FastStateValues,
    pub(crate) stop: bool,
}

pub(crate) struct FastEvalCtx<'a> {
    graph: &'a Graph,
    src: NodeId,
    dest: NodeId,
    edge: EdgeId,
    state: &'a [FastScalar],
}

trait FastInputCtx {
    fn graph(&self) -> &Graph;
    fn src(&self) -> NodeId;
    fn dest(&self) -> NodeId;
    fn edge(&self) -> EdgeId;
    fn state_slot(&self, index: usize) -> Result<Option<FastScalar>>;
}

impl FastInputCtx for EvalCtx<'_> {
    fn graph(&self) -> &Graph {
        self.graph
    }

    fn src(&self) -> NodeId {
        self.src
    }

    fn dest(&self) -> NodeId {
        self.dest
    }

    fn edge(&self) -> EdgeId {
        self.edge
    }

    fn state_slot(&self, index: usize) -> Result<Option<FastScalar>> {
        let value = self.state[index].as_value();
        if value.is_null() {
            Ok(None)
        } else {
            FastScalar::from_value(value)
                .context("expected number")
                .map(Some)
        }
    }
}

impl FastInputCtx for FastEvalCtx<'_> {
    fn graph(&self) -> &Graph {
        self.graph
    }

    fn src(&self) -> NodeId {
        self.src
    }

    fn dest(&self) -> NodeId {
        self.dest
    }

    fn edge(&self) -> EdgeId {
        self.edge
    }

    fn state_slot(&self, index: usize) -> Result<Option<FastScalar>> {
        Ok(Some(self.state[index]))
    }
}

impl FastBool {
    pub(super) fn compile(expr: &Expr<BoundColumn>) -> Option<Self> {
        match expr {
            Expr::Literal(Value::Bool(value)) => Some(Self::Literal(*value)),
            Expr::Column(column) => Some(Self::Column(FastBoolReader::compile(column)?)),
            Expr::Alias(expr, _name) => Self::compile(expr),
            Expr::Scalar(ScalarOp::Not, args) if args.len() == 1 => {
                if let Expr::Column(column) = &args[0] {
                    return Some(Self::NotColumn(FastBoolReader::compile(column)?));
                }
                Some(Self::Not(Box::new(Self::compile(&args[0])?)))
            }
            Expr::Scalar(ScalarOp::And, args) if args.len() == 2 => {
                let mut terms = Vec::new();
                collect_bool_terms(ScalarOp::And, &args[0], &mut terms)?;
                collect_bool_terms(ScalarOp::And, &args[1], &mut terms)?;
                Some(Self::All(terms))
            }
            Expr::Scalar(ScalarOp::Or, args) if args.len() == 2 => {
                let mut terms = Vec::new();
                collect_bool_terms(ScalarOp::Or, &args[0], &mut terms)?;
                collect_bool_terms(ScalarOp::Or, &args[1], &mut terms)?;
                Some(Self::Any(terms))
            }
            Expr::Scalar(op, args) if args.len() == 2 => {
                let cmp = NumCmp::from_scalar(*op)?;
                Some(Self::NumCmp(
                    cmp,
                    FastNum::compile(&args[0])?,
                    FastNum::compile(&args[1])?,
                ))
            }
            _ => None,
        }
    }

    pub(super) fn eval(&self, ctx: &EvalCtx<'_>) -> Result<bool> {
        self.eval_in(ctx)
    }

    fn eval_fast(&self, ctx: &FastEvalCtx<'_>) -> Result<bool> {
        self.eval_in(ctx)
    }

    fn eval_in(&self, ctx: &impl FastInputCtx) -> Result<bool> {
        Ok(match self {
            Self::Literal(value) => *value,
            Self::Column(reader) => reader.read(ctx),
            Self::NotColumn(reader) => !reader.read(ctx),
            Self::Not(expr) => !expr.eval_in(ctx)?,
            Self::All(terms) => {
                for term in terms {
                    if !term.eval_in(ctx)? {
                        return Ok(false);
                    }
                }
                true
            }
            Self::Any(terms) => {
                for term in terms {
                    if term.eval_in(ctx)? {
                        return Ok(true);
                    }
                }
                false
            }
            Self::NumCmp(op, left, right) => {
                let Some(left) = left.eval_in(ctx)? else {
                    return Ok(false);
                };
                let Some(right) = right.eval_in(ctx)? else {
                    return Ok(false);
                };
                op.eval(left, right)?
            }
        })
    }
}

impl FastNum {
    fn compile(expr: &Expr<BoundColumn>) -> Option<Self> {
        match expr {
            Expr::Literal(value) => FastScalar::from_value(value).map(Self::Literal),
            Expr::Column(column) => Some(Self::Column(FastNumReader::compile(column)?)),
            Expr::Alias(expr, _name) => Self::compile(expr),
            Expr::Scalar(op, args) if args.len() == 2 => {
                let op = NumOp::from_scalar(*op)?;
                Some(Self::Binary(
                    op,
                    Box::new(Self::compile(&args[0])?),
                    Box::new(Self::compile(&args[1])?),
                ))
            }
            _ => None,
        }
    }

    fn eval_fast(&self, ctx: &FastEvalCtx<'_>) -> Result<Option<FastScalar>> {
        self.eval_in(ctx)
    }

    fn eval_in(&self, ctx: &impl FastInputCtx) -> Result<Option<FastScalar>> {
        Ok(match self {
            Self::Literal(value) => Some(*value),
            Self::Column(reader) => reader.read(ctx)?,
            Self::Binary(op, left, right) => {
                let Some(left) = left.eval_in(ctx)? else {
                    return Ok(None);
                };
                let Some(right) = right.eval_in(ctx)? else {
                    return Ok(None);
                };
                Some(op.eval(left, right))
            }
        })
    }
}

impl FastScalarKernel {
    pub(crate) fn compile(
        visit: &Expr<BoundColumn>,
        next_state: &[(usize, Expr<BoundColumn>)],
        stop: &Expr<BoundColumn>,
        initial_state: &[StateValue],
    ) -> Option<Self> {
        let visit = FastBool::compile(visit)?;
        let stop = FastBool::compile(stop)?;
        let mut initial_values = FastStateValues::new();
        for value in initial_state {
            let value = value.as_value();
            initial_values.push(FastScalar::from_value(value)?);
        }

        let next_state = next_state
            .iter()
            .map(|(index, expr)| Some((*index, FastNum::compile(expr)?)))
            .collect::<Option<Vec<_>>>()?;

        Some(Self {
            visit,
            next_state,
            stop,
            initial_state: initial_values,
        })
    }

    pub(crate) fn initial_state(&self) -> &FastStateValues {
        &self.initial_state
    }

    pub(crate) fn evaluate_edge(
        &self,
        graph: &Graph,
        src: NodeId,
        dest: NodeId,
        edge: EdgeId,
        current: &[FastScalar],
    ) -> Result<Option<FastEdgeDecision>> {
        let ctx = FastEvalCtx {
            graph,
            src,
            dest,
            edge,
            state: current,
        };
        if !self.visit.eval_fast(&ctx)? {
            return Ok(None);
        }

        let mut next = current.iter().copied().collect::<FastStateValues>();
        for (index, expr) in &self.next_state {
            next[*index] = expr
                .eval_fast(&ctx)?
                .context("numeric fast state expression produced null")?;
        }
        let stop = self.stop.eval_fast(&FastEvalCtx {
            graph,
            src,
            dest,
            edge,
            state: &next,
        })?;
        Ok(Some(FastEdgeDecision { state: next, stop }))
    }

    pub(crate) fn state_values(&self, state: &[FastScalar]) -> StateValues {
        state
            .iter()
            .copied()
            .map(|value| StateValue::new(value.to_value()))
            .collect()
    }
}

impl NumOp {
    fn from_scalar(op: ScalarOp) -> Option<Self> {
        match op {
            ScalarOp::Plus => Some(Self::Plus),
            ScalarOp::Minus => Some(Self::Minus),
            ScalarOp::Multiply => Some(Self::Multiply),
            ScalarOp::Divide => Some(Self::Divide),
            _ => None,
        }
    }

    fn eval(self, left: FastScalar, right: FastScalar) -> FastScalar {
        if matches!(self, Self::Divide) || left.is_float() || right.is_float() {
            let left = left.as_f64();
            let right = right.as_f64();
            return FastScalar::F64(match self {
                Self::Plus => left + right,
                Self::Minus => left - right,
                Self::Multiply => left * right,
                Self::Divide => left / right,
            });
        }

        let left = left.as_i128();
        let right = right.as_i128();
        let value = match self {
            Self::Plus => left + right,
            Self::Minus => left - right,
            Self::Multiply => left * right,
            Self::Divide => unreachable!(),
        };
        if value >= 0 {
            FastScalar::U64(value as u64)
        } else {
            FastScalar::I64(value as i64)
        }
    }
}

impl NumCmp {
    fn from_scalar(op: ScalarOp) -> Option<Self> {
        match op {
            ScalarOp::Eq => Some(Self::Eq),
            ScalarOp::NotEq => Some(Self::NotEq),
            ScalarOp::Lt => Some(Self::Lt),
            ScalarOp::LtEq => Some(Self::LtEq),
            ScalarOp::Gt => Some(Self::Gt),
            ScalarOp::GtEq => Some(Self::GtEq),
            _ => None,
        }
    }

    fn eval(self, left: FastScalar, right: FastScalar) -> Result<bool> {
        let left = left.as_f64();
        let right = right.as_f64();
        Ok(match self {
            Self::Eq => left == right,
            Self::NotEq => left != right,
            Self::Lt | Self::LtEq | Self::Gt | Self::GtEq => {
                let ordering = left.partial_cmp(&right).context("cannot compare values")?;
                match self {
                    Self::Lt => ordering.is_lt(),
                    Self::LtEq => ordering.is_le(),
                    Self::Gt => ordering.is_gt(),
                    Self::GtEq => ordering.is_ge(),
                    Self::Eq | Self::NotEq => unreachable!(),
                }
            }
        })
    }
}

fn collect_bool_terms(
    op: ScalarOp,
    expr: &Expr<BoundColumn>,
    out: &mut Vec<FastBool>,
) -> Option<()> {
    match expr {
        Expr::Alias(expr, _name) => collect_bool_terms(op, expr, out),
        Expr::Scalar(expr_op, args) if *expr_op == op && args.len() == 2 => {
            collect_bool_terms(op, &args[0], out)?;
            collect_bool_terms(op, &args[1], out)
        }
        _ => {
            out.push(FastBool::compile(expr)?);
            Some(())
        }
    }
}

impl FastBoolReader {
    fn compile(column: &BoundColumn) -> Option<Self> {
        match column {
            BoundColumn::Src(ColumnReader::Bool(array)) => Some(Self::array(RowScope::Src, array)),
            BoundColumn::Dest(ColumnReader::Bool(array)) => {
                Some(Self::array(RowScope::Dest, array))
            }
            BoundColumn::Edge(ColumnReader::Bool(array)) => {
                Some(Self::array(RowScope::Edge, array))
            }
            BoundColumn::MissingState => Some(Self::MissingState),
            _ => None,
        }
    }

    fn array(scope: RowScope, array: &BooleanArray) -> Self {
        Self::Array {
            scope,
            nullable: array.null_count() != 0,
            array: array.clone(),
        }
    }

    fn read(&self, ctx: &impl FastInputCtx) -> bool {
        match self {
            Self::Array {
                scope,
                nullable,
                array,
            } => {
                let row = scope.row(ctx.src(), ctx.dest(), ctx.edge());
                if *nullable {
                    bool_value(array, row)
                } else {
                    array.value(row)
                }
            }
            Self::MissingState => false,
        }
    }
}

impl FastNumReader {
    fn compile(column: &BoundColumn) -> Option<Self> {
        Some(match column {
            BoundColumn::Src(reader) => Self::array(RowScope::Src, reader)?,
            BoundColumn::Dest(reader) => Self::array(RowScope::Dest, reader)?,
            BoundColumn::Edge(reader) => Self::array(RowScope::Edge, reader)?,
            BoundColumn::State(index) => Self::State(*index),
            BoundColumn::MissingState => Self::MissingState,
            BoundColumn::SrcId => Self::SrcId,
            BoundColumn::DestId => Self::DestId,
            BoundColumn::EdgeId => Self::EdgeId,
        })
    }

    fn array(scope: RowScope, reader: &ColumnReader) -> Option<Self> {
        Some(Self::Array {
            scope,
            array: FastNumArray::compile(reader)?,
        })
    }

    fn read(&self, ctx: &impl FastInputCtx) -> Result<Option<FastScalar>> {
        Ok(match self {
            Self::Array { scope, array } => {
                array.read(scope.row(ctx.src(), ctx.dest(), ctx.edge()))
            }
            Self::State(index) => ctx.state_slot(*index)?,
            Self::MissingState => None,
            Self::SrcId => graph_id_scalar(
                ctx.graph()
                    .repo
                    .external_node(ctx.src())
                    .context("missing src id")?,
            )?,
            Self::DestId => graph_id_scalar(
                ctx.graph()
                    .repo
                    .external_node(ctx.dest())
                    .context("missing dest id")?,
            )?,
            Self::EdgeId => graph_id_scalar(
                ctx.graph()
                    .repo
                    .external_edge(ctx.edge())
                    .context("missing edge id")?,
            )?,
        })
    }
}

impl FastScalar {
    fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::I64(value) => Some(Self::I64(*value)),
            Value::U64(value) => Some(Self::U64(*value)),
            Value::F64(value) => Some(Self::F64(*value)),
            _ => None,
        }
    }

    fn to_value(self) -> Value {
        match self {
            Self::I64(value) => Value::I64(value),
            Self::U64(value) => Value::U64(value),
            Self::F64(value) => Value::F64(value),
        }
    }

    fn as_f64(self) -> f64 {
        match self {
            Self::I64(value) => value as f64,
            Self::U64(value) => value as f64,
            Self::F64(value) => value,
        }
    }

    fn as_i128(self) -> i128 {
        match self {
            Self::I64(value) => value as i128,
            Self::U64(value) => value as i128,
            Self::F64(_) => unreachable!("float state should use float arithmetic"),
        }
    }

    fn is_float(self) -> bool {
        matches!(self, Self::F64(_))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RowScope {
    Src,
    Dest,
    Edge,
}

impl RowScope {
    fn row(self, src: NodeId, dest: NodeId, edge: EdgeId) -> usize {
        match self {
            Self::Src => src as usize,
            Self::Dest => dest as usize,
            Self::Edge => edge as usize,
        }
    }
}

fn bool_value(array: &BooleanArray, row: usize) -> bool {
    !array.is_null(row) && array.value(row)
}

fn scalar_i64<T>(
    array: &PrimitiveArray<T>,
    row: usize,
    value: impl FnOnce(T::Native) -> i64,
) -> Option<FastScalar>
where
    T: ArrowPrimitiveType,
{
    if array.is_null(row) {
        None
    } else {
        Some(FastScalar::I64(value(array.value(row))))
    }
}

fn scalar_u64<T>(
    array: &PrimitiveArray<T>,
    row: usize,
    value: impl FnOnce(T::Native) -> u64,
) -> Option<FastScalar>
where
    T: ArrowPrimitiveType,
{
    if array.is_null(row) {
        None
    } else {
        Some(FastScalar::U64(value(array.value(row))))
    }
}

fn scalar_f64<T>(
    array: &PrimitiveArray<T>,
    row: usize,
    value: impl FnOnce(T::Native) -> f64,
) -> Option<FastScalar>
where
    T: ArrowPrimitiveType,
{
    if array.is_null(row) {
        None
    } else {
        Some(FastScalar::F64(value(array.value(row))))
    }
}

fn graph_id_scalar(id: GraphId<'_>) -> Result<Option<FastScalar>> {
    match id {
        GraphId::U64(value) => Ok(Some(FastScalar::U64(value))),
        GraphId::Str(_) => bail!("string graph IDs cannot be used as numbers"),
    }
}
