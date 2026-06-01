use std::sync::Arc;

use anyhow::Result;

use crate::dsl::{
    Value,
    ops::{list::ListOp, scalar::ScalarOp, string::StringOp, struct_::StructOp},
    polars_json::parse_polars_json,
};

/// Typed traversal expression.
#[derive(Debug, Clone)]
pub struct DslExpr(pub(crate) Expr<ColumnRef>);

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

    pub fn null_lit() -> Self {
        Self::lit(Value::Null)
    }

    pub fn bool_lit(value: bool) -> Self {
        Self::lit(Value::Bool(value))
    }

    pub fn int_lit(value: i64) -> Self {
        Self::lit(Value::I64(value))
    }

    pub fn uint_lit(value: u64) -> Self {
        Self::lit(Value::U64(value))
    }

    pub fn float_lit(value: f64) -> Self {
        Self::lit(Value::F64(value))
    }

    pub fn string_lit(value: impl Into<Arc<str>>) -> Self {
        Self::lit(Value::Str(value.into()))
    }

    pub fn lit(value: impl Into<Value>) -> Self {
        Self(Expr::Literal(value.into()))
    }

    /// Parses the supported Polars JSON expression subset.
    pub fn from_polars_json(json: &str) -> Result<Self> {
        Ok(Self(parse_polars_json(json)?))
    }

    pub fn and(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::And, [rhs])
    }

    pub fn or(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::Or, [rhs])
    }

    pub fn eq(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::Eq, [rhs])
    }

    pub fn ne(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::NotEq, [rhs])
    }

    pub fn lt(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::Lt, [rhs])
    }

    pub fn le(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::LtEq, [rhs])
    }

    pub fn gt(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::Gt, [rhs])
    }

    pub fn ge(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::GtEq, [rhs])
    }

    pub fn plus(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::Plus, [rhs])
    }

    pub fn minus(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::Minus, [rhs])
    }

    pub fn multiply(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::Multiply, [rhs])
    }

    /// Numeric division. Always returns [`Value::F64`].
    pub fn divide(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::Divide, [rhs])
    }

    pub fn modulo(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::Modulus, [rhs])
    }

    pub fn bit_and(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::BitAnd, [rhs])
    }

    pub fn bit_or(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::BitOr, [rhs])
    }

    pub fn bit_xor(self, rhs: Self) -> Self {
        self.scalar(ScalarOp::BitXor, [rhs])
    }

    /// Returns `then_value` when any bit in `mask` is set in `self`; otherwise
    /// returns `0`.
    pub fn mask_if_any(self, mask: Self, then_value: Self) -> Self {
        self.scalar(ScalarOp::MaskIfAny, [mask, then_value])
    }

    /// Boolean negation.
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(Expr::Scalar(ScalarOp::Not, vec![self.0]))
    }

    pub fn is_null(self) -> Self {
        Self(Expr::Scalar(ScalarOp::IsNull, vec![self.0]))
    }

    pub fn is_not_null(self) -> Self {
        Self(Expr::Scalar(ScalarOp::IsNotNull, vec![self.0]))
    }

    pub fn str_contains(self, needle: Self) -> Self {
        self.string_op(StringOp::Contains, [needle])
    }

    pub fn str_starts_with(self, prefix: Self) -> Self {
        self.string_op(StringOp::StartsWith, [prefix])
    }

    pub fn str_ends_with(self, suffix: Self) -> Self {
        self.string_op(StringOp::EndsWith, [suffix])
    }

    /// Concatenates list expressions, appending scalar expressions as single items.
    pub fn concat_list(values: impl IntoIterator<Item = Self>) -> Self {
        Self(Expr::List(
            ListOp::Concat,
            values.into_iter().map(Self::into_inner).collect(),
        ))
    }

    pub(crate) fn into_inner(self) -> Expr<ColumnRef> {
        self.0
    }

    fn scalar<const N: usize>(self, op: ScalarOp, rhs: [Self; N]) -> Self {
        Self(Expr::Scalar(
            op,
            std::iter::once(self.0)
                .chain(rhs.into_iter().map(Self::into_inner))
                .collect(),
        ))
    }

    fn string_op<const N: usize>(self, op: StringOp, rhs: [Self; N]) -> Self {
        Self(Expr::String(
            op,
            std::iter::once(self.0)
                .chain(rhs.into_iter().map(Self::into_inner))
                .collect(),
        ))
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Expr<C> {
    Column(C),
    Element,
    Literal(Value),
    Alias(Box<Expr<C>>, String),
    Scalar(ScalarOp, Vec<Expr<C>>),
    String(StringOp, Vec<Expr<C>>),
    List(ListOp, Vec<Expr<C>>),
    Struct(StructOp, Vec<Expr<C>>),
}

impl<C> Expr<C> {
    pub(crate) fn try_map_column<D>(self, f: &mut impl FnMut(C) -> Result<D>) -> Result<Expr<D>> {
        Ok(match self {
            Self::Column(column) => Expr::Column(f(column)?),
            Self::Element => Expr::Element,
            Self::Literal(value) => Expr::Literal(value),
            Self::Alias(expr, name) => Expr::Alias(Box::new(expr.try_map_column(f)?), name),
            Self::Scalar(op, args) => Expr::Scalar(op, map_args(args, f)?),
            Self::String(op, args) => Expr::String(op, map_args(args, f)?),
            Self::List(op, args) => Expr::List(op, map_args(args, f)?),
            Self::Struct(op, args) => Expr::Struct(op, map_args(args, f)?),
        })
    }
}

fn map_args<C, D>(args: Vec<Expr<C>>, f: &mut impl FnMut(C) -> Result<D>) -> Result<Vec<Expr<D>>> {
    args.into_iter()
        .map(|expr| expr.try_map_column(f))
        .collect()
}

#[derive(Debug, Clone)]
pub(crate) enum ColumnRef {
    SrcId,
    DestId,
    EdgeId,
    SrcField(String),
    DestField(String),
    EdgeField(String),
    State(String),
}
