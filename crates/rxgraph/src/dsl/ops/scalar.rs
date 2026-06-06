use std::sync::Arc;

use anyhow::{Context, Result, bail};

use crate::dsl::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CastType {
    Bool,
    Int64,
    UInt64,
    Float64,
    String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScalarOp {
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
    MaskIfAny,
    Not,
    IsNull,
    IsNotNull,
    Cast(CastType),
}

impl ScalarOp {
    pub(crate) fn eval(self, args: &[Value]) -> Result<Value> {
        Ok(match self {
            Self::And => Value::Bool(arg(args, 0)?.truthy()? && arg(args, 1)?.truthy()?),
            Self::Or => Value::Bool(arg(args, 0)?.truthy()? || arg(args, 1)?.truthy()?),
            Self::Eq => Value::Bool(arg(args, 0)?.eq_value(arg(args, 1)?)),
            Self::NotEq => Value::Bool(!arg(args, 0)?.eq_value(arg(args, 1)?)),
            Self::Lt => compare(arg(args, 0)?, arg(args, 1)?, |ord| ord.is_lt())?,
            Self::LtEq => compare(arg(args, 0)?, arg(args, 1)?, |ord| ord.is_le())?,
            Self::Gt => compare(arg(args, 0)?, arg(args, 1)?, |ord| ord.is_gt())?,
            Self::GtEq => compare(arg(args, 0)?, arg(args, 1)?, |ord| ord.is_ge())?,
            Self::Plus => numeric_binary(
                arg(args, 0)?.clone(),
                arg(args, 1)?.clone(),
                |a, b| a + b,
                |a, b| a + b,
            )?,
            Self::Minus => numeric_binary(
                arg(args, 0)?.clone(),
                arg(args, 1)?.clone(),
                |a, b| a - b,
                |a, b| a - b,
            )?,
            Self::Multiply => numeric_binary(
                arg(args, 0)?.clone(),
                arg(args, 1)?.clone(),
                |a, b| a * b,
                |a, b| a * b,
            )?,
            Self::Divide => Value::F64(number(arg(args, 0)?)? / number(arg(args, 1)?)?),
            Self::Modulus => Value::I64((integer(arg(args, 0)?)? % integer(arg(args, 1)?)?) as i64),
            Self::BitAnd => bitwise(arg(args, 0)?, arg(args, 1)?, |a, b| a & b)?,
            Self::BitOr => bitwise(arg(args, 0)?, arg(args, 1)?, |a, b| a | b)?,
            Self::BitXor => bitwise(arg(args, 0)?, arg(args, 1)?, |a, b| a ^ b)?,
            Self::MaskIfAny => {
                let value = arg(args, 0)?
                    .as_u64()
                    .context("mask_if_any value must be an unsigned integer")?;
                let mask = arg(args, 1)?
                    .as_u64()
                    .context("mask_if_any mask must be an unsigned integer")?;
                if value & mask == 0 {
                    Value::U64(0)
                } else {
                    arg(args, 2)?.clone()
                }
            }
            Self::Not => Value::Bool(!arg(args, 0)?.truthy()?),
            Self::IsNull => Value::Bool(arg(args, 0)?.is_null()),
            Self::IsNotNull => Value::Bool(!arg(args, 0)?.is_null()),
            Self::Cast(dtype) => cast(arg(args, 0)?, dtype)?,
        })
    }
}

fn arg(args: &[Value], index: usize) -> Result<&Value> {
    args.get(index)
        .with_context(|| format!("missing scalar op argument {index}"))
}

fn cast(value: &Value, dtype: CastType) -> Result<Value> {
    if value.is_null() {
        return Ok(Value::Null);
    }
    Ok(match dtype {
        CastType::Bool => Value::Bool(value.truthy()?),
        CastType::Int64 => Value::I64(integer(value)? as i64),
        CastType::UInt64 => Value::U64(value.as_u64().context("expected unsigned integer")?),
        CastType::Float64 => Value::F64(number(value)?),
        CastType::String => Value::Str(Arc::from(match value {
            Value::Str(value) => value.to_string(),
            Value::Bool(value) => value.to_string(),
            Value::I64(value) => value.to_string(),
            Value::U64(value) => value.to_string(),
            Value::F64(value) => value.to_string(),
            other => other.to_value().to_string(),
        })),
    })
}

fn compare(
    left: &Value,
    right: &Value,
    f: impl FnOnce(std::cmp::Ordering) -> bool,
) -> Result<Value> {
    if left.is_null() || right.is_null() {
        Ok(Value::Null)
    } else {
        Ok(Value::Bool(f(left.compare(right)?)))
    }
}

fn integer(value: &Value) -> Result<i128> {
    value.as_i128().context("expected integer")
}

fn number(value: &Value) -> Result<f64> {
    value.as_f64().context("expected number")
}

fn numeric_binary(
    left: Value,
    right: Value,
    int: impl FnOnce(i128, i128) -> i128,
    float: impl FnOnce(f64, f64) -> f64,
) -> Result<Value> {
    if left.is_null() || right.is_null() {
        return Ok(Value::Null);
    }
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

fn bitwise(left: &Value, right: &Value, op: impl FnOnce(u64, u64) -> u64) -> Result<Value> {
    if left.is_null() || right.is_null() {
        return Ok(Value::Null);
    }
    Ok(Value::U64(op(
        left.as_u64().context("expected unsigned integer")?,
        right.as_u64().context("expected unsigned integer")?,
    )))
}

pub(crate) fn parse_binary_op(op: &str) -> Result<ScalarOp> {
    Ok(match op {
        "And" => ScalarOp::And,
        "Or" => ScalarOp::Or,
        "Eq" => ScalarOp::Eq,
        "NotEq" => ScalarOp::NotEq,
        "Lt" => ScalarOp::Lt,
        "LtEq" => ScalarOp::LtEq,
        "Gt" => ScalarOp::Gt,
        "GtEq" => ScalarOp::GtEq,
        "Plus" => ScalarOp::Plus,
        "Minus" => ScalarOp::Minus,
        "Multiply" => ScalarOp::Multiply,
        "Divide" => ScalarOp::Divide,
        "Modulus" => ScalarOp::Modulus,
        "BitAnd" => ScalarOp::BitAnd,
        "BitOr" => ScalarOp::BitOr,
        "BitXor" => ScalarOp::BitXor,
        op => bail!("unsupported binary op {op:?}"),
    })
}
