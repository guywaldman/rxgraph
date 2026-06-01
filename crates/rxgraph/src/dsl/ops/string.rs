use anyhow::{Context, Result, bail};

use crate::dsl::Value;

#[derive(Debug, Clone, Copy)]
pub(crate) enum StringOp {
    Contains,
    StartsWith,
    EndsWith,
}

impl StringOp {
    pub(crate) fn eval(self, args: &[Value]) -> Result<Value> {
        let left = string_arg(args, 0)?;
        let right = string_arg(args, 1)?;
        let (Some(left), Some(right)) = (left, right) else {
            return Ok(Value::Null);
        };
        Ok(Value::Bool(match self {
            Self::Contains => left.contains(right),
            Self::StartsWith => left.starts_with(right),
            Self::EndsWith => left.ends_with(right),
        }))
    }
}

fn string_arg(args: &[Value], index: usize) -> Result<Option<&str>> {
    match args
        .get(index)
        .with_context(|| format!("missing string op argument {index}"))?
    {
        Value::Null => Ok(None),
        Value::Str(value) => Ok(Some(value)),
        other => bail!("string operation expected strings, got {other:?}"),
    }
}
