use std::sync::Arc;

use anyhow::{Context, Result, bail};

use crate::dsl::Value;

#[derive(Debug, Clone)]
pub(crate) enum StructOp {
    FieldByName(String),
    RenameFields(Vec<String>),
    WithFields(Vec<String>),
    JsonEncode,
}

impl StructOp {
    pub(crate) fn eval(&self, args: &[Value]) -> Result<Value> {
        Ok(match self {
            Self::FieldByName(name) => {
                let Some(fields) = struct_arg(args, 0)? else {
                    return Ok(Value::Null);
                };
                fields
                    .iter()
                    .find_map(|(field, value)| (field == name).then(|| value.clone()))
                    .unwrap_or(Value::Null)
            }
            Self::RenameFields(names) => {
                let Some(fields) = struct_arg(args, 0)? else {
                    return Ok(Value::Null);
                };
                Value::Struct(
                    fields
                        .iter()
                        .enumerate()
                        .map(|(index, (_, value))| {
                            (
                                names
                                    .get(index)
                                    .cloned()
                                    .unwrap_or_else(|| format!("field_{index}")),
                                value.clone(),
                            )
                        })
                        .collect(),
                )
            }
            Self::JsonEncode => {
                let value = args.first().context("missing struct argument")?;
                if value.is_null() {
                    Value::Null
                } else {
                    Value::Str(Arc::from(value.to_value().to_string()))
                }
            }
            Self::WithFields(_) => bail!("struct with_fields requires expression context"),
        })
    }
}

fn struct_arg(args: &[Value], index: usize) -> Result<Option<&[(String, Value)]>> {
    match args
        .get(index)
        .with_context(|| format!("missing struct op argument {index}"))?
    {
        Value::Null => Ok(None),
        Value::Struct(fields) => Ok(Some(fields)),
        other => bail!("expected struct, got {other:?}"),
    }
}
