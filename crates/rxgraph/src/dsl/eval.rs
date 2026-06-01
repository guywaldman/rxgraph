use anyhow::{Context, Result};

use crate::{
    dsl::{Value, bind::BoundColumn, expr::Expr},
    graph::{EdgeId, Graph, NodeId},
};

pub(crate) struct EvalCtx<'a> {
    pub(crate) graph: &'a Graph,
    pub(crate) src: NodeId,
    pub(crate) dest: NodeId,
    pub(crate) edge: EdgeId,
    pub(crate) state: &'a [Value],
    element: Option<&'a Value>,
}

impl<'a> EvalCtx<'a> {
    pub(crate) fn new(
        graph: &'a Graph,
        src: NodeId,
        dest: NodeId,
        edge: EdgeId,
        state: &'a [Value],
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

    pub(crate) fn with_state<'b>(&'b self, state: &'b [Value]) -> EvalCtx<'b> {
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
            Self::Scalar(op, args) => {
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
