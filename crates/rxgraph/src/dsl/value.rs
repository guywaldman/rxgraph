use std::{cmp::Ordering, collections::HashMap, sync::Arc};

use anyhow::{Context, Result, bail};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as Json};

/// Scalar value used by DSL expressions and traversal state.
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    /// Null value.
    Null,
    /// Boolean value.
    Bool(bool),
    /// Signed integer value.
    I64(i64),
    /// Unsigned integer value.
    U64(u64),
    /// Floating point value.
    F64(f64),
    /// Shared string value.
    Str(Arc<str>),
}

impl From<Scalar> for Value {
    fn from(value: Scalar) -> Self {
        match value {
            Scalar::Null => Self::Null,
            Scalar::Bool(value) => Self::Bool(value),
            Scalar::I64(value) => Self::I64(value),
            Scalar::U64(value) => Self::U64(value),
            Scalar::F64(value) => Self::F64(value),
            Scalar::Str(value) => Self::Str(value),
        }
    }
}

/// Runtime value used by DSL expressions and traversal state.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Null value.
    Null,
    /// Boolean value.
    Bool(bool),
    /// Signed integer value.
    I64(i64),
    /// Unsigned integer value.
    U64(u64),
    /// Floating point value.
    F64(f64),
    /// Shared string value.
    Str(Arc<str>),
    /// Ordered value list.
    List(Vec<Value>),
    /// Ordered named struct fields.
    Struct(Vec<(String, Value)>),
}

impl Value {
    /// Converts the value to JSON for callers that need a loosely typed value.
    pub fn to_value(&self) -> Json {
        match self {
            Self::Null => Json::Null,
            Self::Bool(value) => Json::Bool(*value),
            Self::I64(value) => Json::Number(JsonNumber::from(*value)),
            Self::U64(value) => Json::Number(JsonNumber::from(*value)),
            Self::F64(value) => JsonNumber::from_f64(*value)
                .map(Json::Number)
                .unwrap_or(Json::Null),
            Self::Str(value) => Json::String(value.to_string()),
            Self::List(values) => Json::Array(values.iter().map(Value::to_value).collect()),
            Self::Struct(fields) => Json::Object(
                fields
                    .iter()
                    .map(|(name, value)| (name.clone(), value.to_value()))
                    .collect::<JsonMap<_, _>>(),
            ),
        }
    }

    pub(crate) fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    pub(crate) fn truthy(&self) -> Result<bool> {
        match self {
            Self::Bool(value) => Ok(*value),
            Self::Null => Ok(false),
            other => bail!("expected boolean expression, got {other:?}"),
        }
    }

    pub(crate) fn as_i128(&self) -> Option<i128> {
        match self {
            Self::I64(value) => Some(*value as i128),
            Self::U64(value) => Some(*value as i128),
            _ => None,
        }
    }

    pub(crate) fn as_u64(&self) -> Option<u64> {
        match self {
            Self::I64(value) if *value >= 0 => Some(*value as u64),
            Self::U64(value) => Some(*value),
            _ => None,
        }
    }

    pub(crate) fn as_f64(&self) -> Option<f64> {
        match self {
            Self::I64(value) => Some(*value as f64),
            Self::U64(value) => Some(*value as f64),
            Self::F64(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Result<Option<&[Value]>> {
        match self {
            Self::Null => Ok(None),
            Self::List(values) => Ok(Some(values)),
            other => bail!("expected list, got {other:?}"),
        }
    }

    pub fn into_list(self) -> Result<Option<Vec<Value>>> {
        match self {
            Self::Null => Ok(None),
            Self::List(values) => Ok(Some(values)),
            other => bail!("expected list, got {other:?}"),
        }
    }

    pub fn as_struct(&self) -> Result<Option<&[(String, Value)]>> {
        match self {
            Self::Null => Ok(None),
            Self::Struct(fields) => Ok(Some(fields)),
            other => bail!("expected struct, got {other:?}"),
        }
    }

    pub fn into_struct(self) -> Result<Option<Vec<(String, Value)>>> {
        match self {
            Self::Null => Ok(None),
            Self::Struct(fields) => Ok(Some(fields)),
            other => bail!("expected struct, got {other:?}"),
        }
    }

    pub(crate) fn eq_value(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(left), Self::Bool(right)) => left == right,
            (Self::Str(left), Self::Str(right)) => left == right,
            (Self::List(left), Self::List(right)) => {
                left.len() == right.len() && left.iter().zip(right).all(|(l, r)| l.eq_value(r))
            }
            (Self::Struct(left), Self::Struct(right)) => struct_fields_eq(left, right),
            _ => self
                .as_f64()
                .zip(rhs.as_f64())
                .is_some_and(|(left, right)| left == right),
        }
    }

    pub(crate) fn compare(&self, rhs: &Self) -> Result<Ordering> {
        match (self, rhs) {
            (Self::Bool(left), Self::Bool(right)) => Ok(left.cmp(right)),
            (Self::Str(left), Self::Str(right)) => Ok(left.cmp(right)),
            _ => self
                .as_f64()
                .zip(rhs.as_f64())
                .and_then(|(left, right)| left.partial_cmp(&right))
                .context("cannot compare values"),
        }
    }
}

fn struct_fields_eq(left: &[(String, Value)], right: &[(String, Value)]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    if right.len() < 8 {
        return struct_fields_eq_linear(left, right);
    }

    let mut right_by_name = HashMap::with_capacity(right.len());
    for (name, value) in right {
        right_by_name.entry(name.as_str()).or_insert(value);
    }
    left.iter().all(|(name, value)| {
        right_by_name
            .get(name.as_str())
            .is_some_and(|right_value| value.eq_value(right_value))
    })
}

fn struct_fields_eq_linear(left: &[(String, Value)], right: &[(String, Value)]) -> bool {
    left.iter().all(|(name, value)| {
        right
            .iter()
            .find(|(right_name, _)| right_name == name)
            .is_some_and(|(_, right_value)| value.eq_value(right_value))
    })
}
