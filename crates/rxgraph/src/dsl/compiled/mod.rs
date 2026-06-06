use anyhow::{Context, Result, bail};

mod fast;

pub(crate) use fast::{FastBool, FastScalar, FastScalarKernel, FastStateValues};

use crate::dsl::{
    Value,
    bind::BoundColumn,
    eval::EvalCtx,
    expr::Expr,
    ops::{list::ListOp, scalar::ScalarOp, string::StringOp, struct_::StructOp},
};

#[derive(Debug, Clone)]
pub(crate) enum CompiledExpr {
    Column(BoundColumn),
    Element,
    Literal(Value),
    Alias(Box<CompiledExpr>),
    Ternary {
        predicate: Box<CompiledBool>,
        truthy: Box<CompiledExpr>,
        falsy: Box<CompiledExpr>,
    },
    Scalar(ScalarOp, Vec<CompiledExpr>),
    String(StringOp, Vec<CompiledExpr>),
    List(ListOp, Vec<CompiledExpr>),
    Struct(StructOp, Vec<CompiledExpr>),
}

#[derive(Debug, Clone)]
pub(crate) enum CompiledBool {
    Fast(FastBool),
    Dynamic(CompiledExpr),
}

impl CompiledExpr {
    pub(crate) fn compile(expr: Expr<BoundColumn>) -> Result<Self> {
        Ok(match expr {
            Expr::Column(column) => Self::Column(column),
            Expr::Element => Self::Element,
            Expr::Literal(value) => Self::Literal(value),
            Expr::Alias(expr, _name) => Self::Alias(Box::new(Self::compile(*expr)?)),
            Expr::Ternary {
                predicate,
                truthy,
                falsy,
            } => Self::Ternary {
                predicate: Box::new(CompiledBool::compile(*predicate)?),
                truthy: Box::new(Self::compile(*truthy)?),
                falsy: Box::new(Self::compile(*falsy)?),
            },
            Expr::Scalar(op, args) => Self::Scalar(op, compile_args(args)?),
            Expr::String(op, args) => Self::String(op, compile_args(args)?),
            Expr::List(op, args) => Self::List(op, compile_args(args)?),
            Expr::Struct(op, args) => Self::Struct(op, compile_args(args)?),
        })
    }

    pub(crate) fn eval(&self, ctx: &EvalCtx<'_>) -> Result<Value> {
        match self {
            Self::Column(column) => column.value(ctx),
            Self::Element => ctx
                .element()
                .cloned()
                .context("pl.element() is only valid inside list.eval/list.filter"),
            Self::Literal(value) => Ok(value.clone()),
            Self::Alias(expr) => expr.eval(ctx),
            Self::Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                if predicate.eval(ctx)? {
                    truthy.eval(ctx)
                } else {
                    falsy.eval(ctx)
                }
            }
            Self::Scalar(ScalarOp::And, args) => {
                let left = expr_arg(args, 0)?;
                if !left.eval(ctx)?.truthy()? {
                    return Ok(Value::Bool(false));
                }
                Ok(Value::Bool(expr_arg(args, 1)?.eval(ctx)?.truthy()?))
            }
            Self::Scalar(ScalarOp::Or, args) => {
                let left = expr_arg(args, 0)?;
                if left.eval(ctx)?.truthy()? {
                    return Ok(Value::Bool(true));
                }
                Ok(Value::Bool(expr_arg(args, 1)?.eval(ctx)?.truthy()?))
            }
            Self::Scalar(op, args) => {
                let values = eval_args(args, ctx)?;
                op.eval(&values)
            }
            Self::String(op, args) => {
                let values = eval_args(args, ctx)?;
                op.eval(&values)
            }
            Self::List(op, args) => eval_list(op, args, ctx),
            Self::Struct(op, args) => eval_struct(op, args, ctx),
        }
    }
}

impl CompiledBool {
    pub(crate) fn compile(expr: Expr<BoundColumn>) -> Result<Self> {
        if let Some(fast) = FastBool::compile(&expr) {
            return Ok(Self::Fast(fast));
        }
        Ok(Self::Dynamic(CompiledExpr::compile(expr)?))
    }

    pub(crate) fn eval(&self, ctx: &EvalCtx<'_>) -> Result<bool> {
        match self {
            Self::Fast(expr) => expr.eval(ctx),
            Self::Dynamic(expr) => expr.eval(ctx)?.truthy(),
        }
    }
}

fn compile_args(args: Vec<Expr<BoundColumn>>) -> Result<Vec<CompiledExpr>> {
    args.into_iter().map(CompiledExpr::compile).collect()
}

fn eval_args(args: &[CompiledExpr], ctx: &EvalCtx<'_>) -> Result<Vec<Value>> {
    args.iter().map(|expr| expr.eval(ctx)).collect()
}

fn expr_arg(args: &[CompiledExpr], index: usize) -> Result<&CompiledExpr> {
    args.get(index)
        .with_context(|| format!("missing scalar op argument {index}"))
}

fn eval_list(op: &ListOp, args: &[CompiledExpr], ctx: &EvalCtx<'_>) -> Result<Value> {
    match op {
        ListOp::Eval => eval_list_map(args, ctx),
        ListOp::Filter => eval_list_filter(args, ctx),
        _ => {
            let values = eval_args(args, ctx)?;
            op.eval(&values)
        }
    }
}

fn eval_list_map(args: &[CompiledExpr], ctx: &EvalCtx<'_>) -> Result<Value> {
    ensure_expr_args(args, 2)?;
    let list = args[0].eval(ctx)?;
    let Some(values) = list.as_list()? else {
        return Ok(Value::Null);
    };
    values
        .iter()
        .map(|value| args[1].eval(&ctx.with_element(value)))
        .collect::<Result<Vec<_>>>()
        .map(Value::List)
}

fn eval_list_filter(args: &[CompiledExpr], ctx: &EvalCtx<'_>) -> Result<Value> {
    ensure_expr_args(args, 2)?;
    let list = args[0].eval(ctx)?;
    let Some(values) = list.as_list()? else {
        return Ok(Value::Null);
    };
    let mut out = Vec::new();
    for value in values {
        if args[1].eval(&ctx.with_element(value))?.truthy()? {
            out.push(value.clone());
        }
    }
    Ok(Value::List(out))
}

fn eval_struct(op: &StructOp, args: &[CompiledExpr], ctx: &EvalCtx<'_>) -> Result<Value> {
    match op {
        StructOp::WithFields(names) => {
            let base = args.first().context("missing base struct")?.eval(ctx)?;
            let Some(mut fields) = base.into_struct()? else {
                return Ok(Value::Null);
            };
            for (index, name) in names.iter().enumerate() {
                let value = args
                    .get(index + 1)
                    .with_context(|| format!("missing struct field expression {name:?}"))?
                    .eval(ctx)?;
                if let Some((_, existing)) = fields.iter_mut().find(|(field, _)| field == name) {
                    *existing = value;
                } else {
                    fields.push((name.clone(), value));
                }
            }
            Ok(Value::Struct(fields))
        }
        _ => {
            let values = eval_args(args, ctx)?;
            op.eval(&values)
        }
    }
}

fn ensure_expr_args(args: &[CompiledExpr], len: usize) -> Result<()> {
    if args.len() == len {
        Ok(())
    } else {
        bail!("expected {len} list expression inputs, got {}", args.len())
    }
}
