use anyhow::{Context, Result};

use crate::{
    dsl::{StateValue, Value, bind::BoundColumn, expr::Expr, ops::scalar::ScalarOp},
    graph::{EdgeId, Graph, NodeId},
};

pub(crate) struct EvalCtx<'a> {
    pub(crate) graph: &'a Graph,
    pub(crate) src: NodeId,
    pub(crate) dest: NodeId,
    pub(crate) edge: EdgeId,
    pub(crate) state: &'a [StateValue],
    element: Option<&'a Value>,
}

impl<'a> EvalCtx<'a> {
    pub(crate) fn new(
        graph: &'a Graph,
        src: NodeId,
        dest: NodeId,
        edge: EdgeId,
        state: &'a [StateValue],
    ) -> Self {
        Self {
            graph,
            src,
            dest,
            edge,
            state,
            element: None,
        }
    }

    pub(crate) fn with_state<'b>(&'b self, state: &'b [StateValue]) -> EvalCtx<'b> {
        EvalCtx {
            graph: self.graph,
            src: self.src,
            dest: self.dest,
            edge: self.edge,
            state,
            element: self.element,
        }
    }

    pub(crate) fn with_element<'b>(&'b self, element: &'b Value) -> EvalCtx<'b> {
        EvalCtx {
            graph: self.graph,
            src: self.src,
            dest: self.dest,
            edge: self.edge,
            state: self.state,
            element: Some(element),
        }
    }
}

impl Expr<BoundColumn> {
    pub(crate) fn eval(&self, ctx: &EvalCtx<'_>) -> Result<Value> {
        match self {
            Self::Column(column) => column.value(ctx),
            Self::Element => ctx
                .element
                .cloned()
                .context("pl.element() is only valid inside list.eval/list.filter"),
            Self::Literal(value) => Ok(value.clone()),
            Self::Alias(expr, _) => expr.eval(ctx),
            Self::Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                if predicate.eval(ctx)?.truthy()? {
                    truthy.eval(ctx)
                } else {
                    falsy.eval(ctx)
                }
            }
            Self::Scalar(ScalarOp::And, args) => eval_and(args, ctx),
            Self::Scalar(ScalarOp::Or, args) => eval_or(args, ctx),
            Self::Scalar(op, args) => {
                if let Some(value) = try_eval_scalar_fast_path(*op, args, ctx)? {
                    return Ok(value);
                }
                let args = eval_args(args, ctx)?;
                op.eval(&args)
            }
            Self::String(op, args) => {
                let args = eval_args(args, ctx)?;
                op.eval(&args)
            }
            Self::List(op, args) => op.eval_with_exprs(args, ctx),
            Self::Struct(op, args) => op.eval_with_exprs(args, ctx),
        }
    }
}

fn eval_args(args: &[Expr<BoundColumn>], ctx: &EvalCtx<'_>) -> Result<Vec<Value>> {
    args.iter().map(|expr| expr.eval(ctx)).collect()
}

fn eval_and(args: &[Expr<BoundColumn>], ctx: &EvalCtx<'_>) -> Result<Value> {
    let left = expr_arg(args, 0)?;
    let right = expr_arg(args, 1)?;
    // Short circuit (optimization)
    if !left.eval(ctx)?.truthy()? {
        return Ok(Value::Bool(false));
    }
    Ok(Value::Bool(right.eval(ctx)?.truthy()?))
}

fn eval_or(args: &[Expr<BoundColumn>], ctx: &EvalCtx<'_>) -> Result<Value> {
    let left = expr_arg(args, 0)?;
    let right = expr_arg(args, 1)?;
    // Short circuit (optimization)
    if left.eval(ctx)?.truthy()? {
        return Ok(Value::Bool(true));
    }
    Ok(Value::Bool(right.eval(ctx)?.truthy()?))
}

// Handles primitive column/literal comparisons without boxing column values (optimization).
fn try_eval_scalar_fast_path(
    op: ScalarOp,
    args: &[Expr<BoundColumn>],
    ctx: &EvalCtx<'_>,
) -> Result<Option<Value>> {
    if !matches!(
        op,
        ScalarOp::Eq
            | ScalarOp::NotEq
            | ScalarOp::Lt
            | ScalarOp::LtEq
            | ScalarOp::Gt
            | ScalarOp::GtEq
    ) {
        return Ok(None);
    }

    let left = expr_arg(args, 0)?;
    let right = expr_arg(args, 1)?;
    if let (Some(column), Some(literal)) = (column_expr(left), literal_expr(right)) {
        return column.eval_scalar_literal(ctx, op, literal, false);
    }
    if let (Some(literal), Some(column)) = (literal_expr(left), column_expr(right)) {
        return column.eval_scalar_literal(ctx, op, literal, true);
    }
    Ok(None)
}

fn expr_arg(args: &[Expr<BoundColumn>], index: usize) -> Result<&Expr<BoundColumn>> {
    args.get(index)
        .with_context(|| format!("missing scalar op argument {index}"))
}

fn column_expr(expr: &Expr<BoundColumn>) -> Option<&BoundColumn> {
    match expr {
        Expr::Column(column) => Some(column),
        Expr::Alias(expr, _) => column_expr(expr),
        _ => None,
    }
}

fn literal_expr(expr: &Expr<BoundColumn>) -> Option<&Value> {
    match expr {
        Expr::Literal(value) => Some(value),
        Expr::Alias(expr, _) => literal_expr(expr),
        _ => None,
    }
}
