use std::{cmp::Ordering, collections::HashSet, sync::Arc};

use anyhow::{Context, Result, bail};

use crate::dsl::{Value, bind::BoundColumn, eval::EvalCtx, expr::Expr};

#[derive(Debug, Clone, Copy)]
pub(crate) enum SetOp {
    Union,
    Intersection,
    Difference,
    SymmetricDifference,
}

#[derive(Debug, Clone)]
pub(crate) enum ListOp {
    Concat,
    Len,
    Contains {
        nulls_equal: bool,
    },
    Get {
        null_on_oob: bool,
    },
    Slice,
    Reverse,
    Sort {
        descending: bool,
        nulls_last: bool,
    },
    Unique,
    DropNulls,
    Sum,
    Min,
    Max,
    Mean,
    Median,
    Any {
        ignore_nulls: bool,
    },
    All {
        ignore_nulls: bool,
    },
    CountMatches,
    NUnique,
    Join {
        ignore_nulls: bool,
    },
    Shift,
    GatherEvery,
    Set(SetOp),
    Eval,
    Filter,
    Explode {
        empty_as_null: bool,
        keep_nulls: bool,
    },
    ToStruct(Vec<String>),
}

impl ListOp {
    pub(crate) fn eval(&self, args: &[Value]) -> Result<Value> {
        Ok(match self {
            Self::Concat => concat(args),
            Self::Len => unary_list(args, |values| Value::U64(values.len() as u64))?,
            Self::Contains { nulls_equal } => contains(args, *nulls_equal)?,
            Self::Get { null_on_oob } => get(args, *null_on_oob)?,
            Self::Slice => slice(args)?,
            Self::Reverse => map_list(args, |mut values| {
                values.reverse();
                Ok(Value::List(values))
            })?,
            Self::Sort {
                descending,
                nulls_last,
            } => map_list(args, |mut values| {
                sort_values(&mut values, *descending, *nulls_last)?;
                Ok(Value::List(values))
            })?,
            Self::Unique => map_list(args, |values| Ok(Value::List(unique(values))))?,
            Self::DropNulls => map_list(args, |values| {
                Ok(Value::List(
                    values
                        .into_iter()
                        .filter(|value| !value.is_null())
                        .collect(),
                ))
            })?,
            Self::Sum => aggregate_sum(args)?,
            Self::Min => aggregate_order(args, Ordering::Less)?,
            Self::Max => aggregate_order(args, Ordering::Greater)?,
            Self::Mean => aggregate_mean(args)?,
            Self::Median => aggregate_median(args)?,
            Self::Any { ignore_nulls } => aggregate_bool(args, *ignore_nulls, true)?,
            Self::All { ignore_nulls } => aggregate_bool(args, *ignore_nulls, false)?,
            Self::CountMatches => count_matches(args)?,
            Self::NUnique => unary_list(args, |values| {
                Value::U64(unique(values.to_vec()).len() as u64)
            })?,
            Self::Join { ignore_nulls } => join(args, *ignore_nulls)?,
            Self::Shift => shift(args)?,
            Self::GatherEvery => gather_every(args)?,
            Self::Set(op) => set_operation(args, *op)?,
            Self::Explode {
                empty_as_null,
                keep_nulls,
            } => explode(args, *empty_as_null, *keep_nulls)?,
            Self::ToStruct(names) => to_struct(args, names)?,
            Self::Eval | Self::Filter => bail!("list op requires expression context"),
        })
    }

    pub(crate) fn eval_with_exprs(
        &self,
        args: &[Expr<BoundColumn>],
        ctx: &EvalCtx<'_>,
    ) -> Result<Value> {
        match self {
            Self::Eval => eval_list(args, ctx),
            Self::Filter => filter_list(args, ctx),
            _ => {
                let values = args
                    .iter()
                    .map(|expr| expr.eval(ctx))
                    .collect::<Result<Vec<_>>>()?;
                self.eval(&values)
            }
        }
    }
}

fn concat(args: &[Value]) -> Value {
    let mut out = Vec::new();
    for value in args {
        match value {
            Value::List(values) => out.extend(values.iter().cloned()),
            Value::Null => out.push(Value::Null),
            value => out.push(value.clone()),
        }
    }
    Value::List(out)
}

fn unary_list(args: &[Value], f: impl FnOnce(&[Value]) -> Value) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    Ok(f(values))
}

fn map_list(args: &[Value], f: impl FnOnce(Vec<Value>) -> Result<Value>) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    f(values.to_vec())
}

fn list_arg(args: &[Value], index: usize) -> Result<Option<&[Value]>> {
    args.get(index)
        .with_context(|| format!("missing list op argument {index}"))?
        .as_list()
}

fn contains(args: &[Value], nulls_equal: bool) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    let needle = arg(args, 1)?;
    if needle.is_null() && !nulls_equal {
        return Ok(Value::Null);
    }
    Ok(Value::Bool(values.iter().any(|value| {
        if value.is_null() || needle.is_null() {
            nulls_equal && value.is_null() && needle.is_null()
        } else {
            value.eq_value(needle)
        }
    })))
}

fn get(args: &[Value], null_on_oob: bool) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    let index = integer_arg(args, 1)?;
    let index = if index < 0 {
        values.len() as i128 + index
    } else {
        index
    };
    if !(0..values.len() as i128).contains(&index) {
        if null_on_oob {
            return Ok(Value::Null);
        }
        bail!(
            "list index {index} is out of bounds for length {}",
            values.len()
        );
    }
    Ok(values[index as usize].clone())
}

fn slice(args: &[Value]) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    let offset = integer_arg(args, 1)?;
    let len = integer_arg(args, 2)?.max(0) as usize;
    let start = if offset < 0 {
        values.len().saturating_sub((-offset) as usize)
    } else {
        (offset as usize).min(values.len())
    };
    let end = start.saturating_add(len).min(values.len());
    Ok(Value::List(values[start..end].to_vec()))
}

fn sort_values(values: &mut [Value], descending: bool, nulls_last: bool) -> Result<()> {
    validate_sortable(values)?;
    values.sort_by(|left, right| match (left.is_null(), right.is_null()) {
        (true, true) => Ordering::Equal,
        (true, false) => {
            if nulls_last {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        (false, true) => {
            if nulls_last {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (false, false) => {
            let ordering = left.compare(right).unwrap_or(Ordering::Equal);
            if descending {
                ordering.reverse()
            } else {
                ordering
            }
        }
    });
    Ok(())
}

fn validate_sortable(values: &[Value]) -> Result<()> {
    let mut first: Option<&Value> = None;
    for value in values.iter().filter(|value| !value.is_null()) {
        if let Some(first) = first {
            first.compare(value)?;
        } else {
            first = Some(value);
        }
    }
    Ok(())
}

fn compare_values(left: &Value, right: &Value) -> Ordering {
    left.compare(right).unwrap_or(Ordering::Equal)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ScalarKey {
    Null,
    Bool(bool),
    Number(u64),
    Str(Arc<str>),
}

fn scalar_key(value: &Value) -> Option<ScalarKey> {
    match value {
        Value::Null => Some(ScalarKey::Null),
        Value::Bool(value) => Some(ScalarKey::Bool(*value)),
        Value::Str(value) => Some(ScalarKey::Str(value.clone())),
        _ => {
            let number = value.as_f64()?;
            if number.is_nan() {
                return None;
            }
            let number = if number == 0.0 { 0.0 } else { number };
            Some(ScalarKey::Number(number.to_bits()))
        }
    }
}

fn unique(values: Vec<Value>) -> Vec<Value> {
    if let Some(keys) = values.iter().map(scalar_key).collect::<Option<Vec<_>>>() {
        return unique_by_scalar_key(values, keys);
    }
    unique_slow(values)
}

fn unique_by_scalar_key(values: Vec<Value>, keys: Vec<ScalarKey>) -> Vec<Value> {
    let mut seen = HashSet::with_capacity(keys.len());
    let mut out = Vec::new();
    for (value, key) in values.into_iter().zip(keys) {
        if seen.insert(key) {
            out.push(value);
        }
    }
    out
}

fn unique_slow(values: Vec<Value>) -> Vec<Value> {
    let mut out = Vec::new();
    for value in values {
        if !out.iter().any(|existing: &Value| existing.eq_value(&value)) {
            out.push(value);
        }
    }
    out
}

fn aggregate_sum(args: &[Value]) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    let mut has_float = false;
    let mut int_sum = 0i128;
    let mut float_sum = 0.0f64;
    for value in values.iter().filter(|value| !value.is_null()) {
        if matches!(value, Value::F64(_)) || has_float {
            has_float = true;
            float_sum += value.as_f64().context("list sum expected numbers")?;
        } else {
            int_sum += value.as_i128().context("list sum expected numbers")?;
        }
    }
    if has_float {
        Ok(Value::F64(float_sum + int_sum as f64))
    } else if int_sum >= 0 {
        Ok(Value::U64(int_sum as u64))
    } else {
        Ok(Value::I64(int_sum as i64))
    }
}

fn aggregate_order(args: &[Value], preferred: Ordering) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    let mut out: Option<&Value> = None;
    for value in values.iter().filter(|value| !value.is_null()) {
        if out.is_none_or(|current| value.compare(current).is_ok_and(|ord| ord == preferred)) {
            out = Some(value);
        }
    }
    Ok(out.cloned().unwrap_or(Value::Null))
}

fn aggregate_mean(args: &[Value]) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values.iter().filter(|value| !value.is_null()) {
        sum += value.as_f64().context("list mean expected numbers")?;
        count += 1;
    }
    Ok(if count == 0 {
        Value::Null
    } else {
        Value::F64(sum / count as f64)
    })
}

fn aggregate_median(args: &[Value]) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    let mut values = values
        .iter()
        .filter(|value| !value.is_null())
        .cloned()
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Ok(Value::Null);
    }
    validate_sortable(&values)?;
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        let (_, median, _) = values.select_nth_unstable_by(mid, compare_values);
        return Ok(median.clone());
    }

    let upper = {
        let (_, upper, _) = values.select_nth_unstable_by(mid, compare_values);
        upper.clone()
    };
    let lower = values[..mid]
        .iter()
        .max_by(|left, right| compare_values(left, right))
        .expect("even-length median has a lower midpoint");
    Ok(Value::F64(
        (lower.as_f64().context("list median expected numbers")?
            + upper.as_f64().context("list median expected numbers")?)
            / 2.0,
    ))
}

fn aggregate_bool(args: &[Value], ignore_nulls: bool, any: bool) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    let mut saw_null = false;
    if any {
        for value in values {
            match value {
                Value::Bool(true) => return Ok(Value::Bool(true)),
                Value::Bool(false) => {}
                Value::Null if ignore_nulls => {}
                Value::Null => saw_null = true,
                other => bail!("list any expected booleans, got {other:?}"),
            }
        }
        Ok(if saw_null {
            Value::Null
        } else {
            Value::Bool(false)
        })
    } else {
        for value in values {
            match value {
                Value::Bool(true) => {}
                Value::Bool(false) => return Ok(Value::Bool(false)),
                Value::Null if ignore_nulls => {}
                Value::Null => saw_null = true,
                other => bail!("list all expected booleans, got {other:?}"),
            }
        }
        Ok(if saw_null {
            Value::Null
        } else {
            Value::Bool(true)
        })
    }
}

fn count_matches(args: &[Value]) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    let needle = arg(args, 1)?;
    Ok(Value::U64(
        values.iter().filter(|value| value.eq_value(needle)).count() as u64,
    ))
}

fn join(args: &[Value], ignore_nulls: bool) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    let separator = match arg(args, 1)? {
        Value::Str(value) => value.as_ref(),
        Value::Null => return Ok(Value::Null),
        other => bail!("list join separator must be a string, got {other:?}"),
    };
    let mut parts = Vec::new();
    for value in values {
        match value {
            Value::Str(value) => parts.push(value.to_string()),
            Value::Null if ignore_nulls => {}
            Value::Null => return Ok(Value::Null),
            other => bail!("list join expected strings, got {other:?}"),
        }
    }
    Ok(Value::Str(Arc::from(parts.join(separator))))
}

fn shift(args: &[Value]) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    let n = integer_arg(args, 1)?;
    let mut out = vec![Value::Null; values.len()];
    for (index, value) in values.iter().enumerate() {
        let shifted = index as i128 + n;
        if (0..values.len() as i128).contains(&shifted) {
            out[shifted as usize] = value.clone();
        }
    }
    Ok(Value::List(out))
}

fn gather_every(args: &[Value]) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    let n = integer_arg(args, 1)?;
    let offset = integer_arg(args, 2)?;
    if n <= 0 {
        bail!("gather_every step must be positive");
    }
    let mut out = Vec::new();
    let mut index = offset.max(0) as usize;
    while index < values.len() {
        out.push(values[index].clone());
        index += n as usize;
    }
    Ok(Value::List(out))
}

fn explode(args: &[Value], empty_as_null: bool, keep_nulls: bool) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    let mut out = Vec::new();
    for value in values {
        match value {
            Value::List(inner) if inner.is_empty() && empty_as_null => out.push(Value::Null),
            Value::List(inner) => out.extend(inner.iter().cloned()),
            Value::Null if keep_nulls => out.push(Value::Null),
            Value::Null => {}
            value => out.push(value.clone()),
        }
    }
    if out.is_empty() && empty_as_null {
        out.push(Value::Null);
    }
    Ok(Value::List(out))
}

fn set_operation(args: &[Value], op: SetOp) -> Result<Value> {
    let Some(left) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    let Some(right) = list_arg(args, 1)? else {
        return Ok(Value::Null);
    };
    if let Some(out) = set_operation_scalar(left, right, op) {
        return Ok(Value::List(out));
    }
    let left = unique_slow(left.to_vec());
    let right = unique_slow(right.to_vec());
    let contains =
        |values: &[Value], needle: &Value| values.iter().any(|value| value.eq_value(needle));
    let out = match op {
        SetOp::Union => unique_slow(left.into_iter().chain(right).collect()),
        SetOp::Intersection => left
            .into_iter()
            .filter(|value| contains(&right, value))
            .collect(),
        SetOp::Difference => left
            .into_iter()
            .filter(|value| !contains(&right, value))
            .collect(),
        SetOp::SymmetricDifference => {
            let mut out = left
                .iter()
                .filter(|value| !contains(&right, value))
                .cloned()
                .collect::<Vec<_>>();
            out.extend(right.into_iter().filter(|value| !contains(&left, value)));
            out
        }
    };
    Ok(Value::List(out))
}

fn set_operation_scalar(left: &[Value], right: &[Value], op: SetOp) -> Option<Vec<Value>> {
    let left = keyed_unique(left)?;
    let right = keyed_unique(right)?;
    let left_keys = left
        .iter()
        .map(|(key, _)| key.clone())
        .collect::<HashSet<_>>();
    let right_keys = right
        .iter()
        .map(|(key, _)| key.clone())
        .collect::<HashSet<_>>();

    let out = match op {
        SetOp::Union => {
            let mut seen = HashSet::with_capacity(left.len() + right.len());
            left.into_iter()
                .chain(right)
                .filter_map(|(key, value)| seen.insert(key).then_some(value))
                .collect()
        }
        SetOp::Intersection => left
            .into_iter()
            .filter_map(|(key, value)| right_keys.contains(&key).then_some(value))
            .collect(),
        SetOp::Difference => left
            .into_iter()
            .filter_map(|(key, value)| (!right_keys.contains(&key)).then_some(value))
            .collect(),
        SetOp::SymmetricDifference => {
            let mut out = left
                .into_iter()
                .filter_map(|(key, value)| (!right_keys.contains(&key)).then_some(value))
                .collect::<Vec<_>>();
            out.extend(
                right
                    .into_iter()
                    .filter_map(|(key, value)| (!left_keys.contains(&key)).then_some(value)),
            );
            out
        }
    };
    Some(out)
}

fn keyed_unique(values: &[Value]) -> Option<Vec<(ScalarKey, Value)>> {
    let mut seen = HashSet::with_capacity(values.len());
    let mut out = Vec::new();
    for value in values {
        let key = scalar_key(value)?;
        if seen.insert(key.clone()) {
            out.push((key, value.clone()));
        }
    }
    Some(out)
}

fn to_struct(args: &[Value], names: &[String]) -> Result<Value> {
    let Some(values) = list_arg(args, 0)? else {
        return Ok(Value::Null);
    };
    Ok(Value::Struct(
        names
            .iter()
            .enumerate()
            .map(|(index, name)| {
                (
                    name.clone(),
                    values.get(index).cloned().unwrap_or(Value::Null),
                )
            })
            .collect(),
    ))
}

fn eval_list(args: &[Expr<BoundColumn>], ctx: &EvalCtx<'_>) -> Result<Value> {
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

fn filter_list(args: &[Expr<BoundColumn>], ctx: &EvalCtx<'_>) -> Result<Value> {
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

fn ensure_expr_args(args: &[Expr<BoundColumn>], len: usize) -> Result<()> {
    if args.len() == len {
        Ok(())
    } else {
        bail!("expected {len} list expression inputs, got {}", args.len())
    }
}

fn arg(args: &[Value], index: usize) -> Result<&Value> {
    args.get(index)
        .with_context(|| format!("missing list op argument {index}"))
}

fn integer_arg(args: &[Value], index: usize) -> Result<i128> {
    arg(args, index)?.as_i128().context("expected integer")
}
