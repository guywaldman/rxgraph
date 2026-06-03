use std::{cmp::Ordering, io::Cursor, sync::Arc};

use anyhow::{Context, Result, bail};
use arrow::{
    array::{
        Array, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
        Int64Array, LargeListArray, LargeStringArray, ListArray, StringArray, StringViewArray,
        StructArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::DataType,
    ipc::reader::StreamReader,
    record_batch::RecordBatch,
};

use crate::dsl::{Value, ops::scalar::ScalarOp};

#[derive(Debug, Clone)]
pub(crate) enum ColumnReader {
    Bool(BooleanArray),
    I8(Int8Array),
    I16(Int16Array),
    I32(Int32Array),
    I64(Int64Array),
    U8(UInt8Array),
    U16(UInt16Array),
    U32(UInt32Array),
    U64(UInt64Array),
    F32(Float32Array),
    F64(Float64Array),
    Utf8(StringArray),
    LargeUtf8(LargeStringArray),
    Utf8View(StringViewArray),
    List(ListArray),
    LargeList(LargeListArray),
    Struct(StructArray),
}

enum ScalarValueRef<'a> {
    Null,
    Bool(bool),
    Number(f64),
    Str(&'a str),
}

impl ColumnReader {
    pub(crate) fn bind(batch: &RecordBatch, name: &str) -> Result<Self> {
        let column = batch
            .column_by_name(name)
            .with_context(|| format!("column {name:?} is missing"))?;

        macro_rules! typed {
            ($array:ty) => {
                column
                    .as_any()
                    .downcast_ref::<$array>()
                    .with_context(|| format!("column {name:?} does not match its Arrow type"))?
                    .clone()
            };
        }

        Ok(match column.data_type() {
            DataType::Boolean => Self::Bool(typed!(BooleanArray)),
            DataType::Int8 => Self::I8(typed!(Int8Array)),
            DataType::Int16 => Self::I16(typed!(Int16Array)),
            DataType::Int32 => Self::I32(typed!(Int32Array)),
            DataType::Int64 => Self::I64(typed!(Int64Array)),
            DataType::UInt8 => Self::U8(typed!(UInt8Array)),
            DataType::UInt16 => Self::U16(typed!(UInt16Array)),
            DataType::UInt32 => Self::U32(typed!(UInt32Array)),
            DataType::UInt64 => Self::U64(typed!(UInt64Array)),
            DataType::Float32 => Self::F32(typed!(Float32Array)),
            DataType::Float64 => Self::F64(typed!(Float64Array)),
            DataType::Utf8 => Self::Utf8(typed!(StringArray)),
            DataType::LargeUtf8 => Self::LargeUtf8(typed!(LargeStringArray)),
            DataType::Utf8View => Self::Utf8View(typed!(StringViewArray)),
            DataType::List(_) => Self::List(typed!(ListArray)),
            DataType::LargeList(_) => Self::LargeList(typed!(LargeListArray)),
            DataType::Struct(_) => Self::Struct(typed!(StructArray)),
            typ => bail!("unsupported DSL column type for {name:?}: {typ:?}"),
        })
    }

    pub(crate) fn value(&self, row: usize) -> Result<Value> {
        macro_rules! nullable {
            ($array:expr, $value:expr) => {
                if $array.is_null(row) {
                    Value::Null
                } else {
                    $value
                }
            };
        }

        Ok(match self {
            Self::Bool(array) => nullable!(array, Value::Bool(array.value(row))),
            Self::I8(array) => nullable!(array, Value::I64(array.value(row) as i64)),
            Self::I16(array) => nullable!(array, Value::I64(array.value(row) as i64)),
            Self::I32(array) => nullable!(array, Value::I64(array.value(row) as i64)),
            Self::I64(array) => nullable!(array, Value::I64(array.value(row))),
            Self::U8(array) => nullable!(array, Value::U64(array.value(row) as u64)),
            Self::U16(array) => nullable!(array, Value::U64(array.value(row) as u64)),
            Self::U32(array) => nullable!(array, Value::U64(array.value(row) as u64)),
            Self::U64(array) => nullable!(array, Value::U64(array.value(row))),
            Self::F32(array) => nullable!(array, Value::F64(array.value(row) as f64)),
            Self::F64(array) => nullable!(array, Value::F64(array.value(row))),
            Self::Utf8(array) => nullable!(array, Value::Str(Arc::from(array.value(row)))),
            Self::LargeUtf8(array) => nullable!(array, Value::Str(Arc::from(array.value(row)))),
            Self::Utf8View(array) => nullable!(array, Value::Str(Arc::from(array.value(row)))),
            Self::List(array) => nullable!(array, Value::List(array_to_values(&array.value(row))?)),
            Self::LargeList(array) => {
                nullable!(array, Value::List(array_to_values(&array.value(row))?))
            }
            Self::Struct(array) => nullable!(array, struct_row_to_value(array, row)?),
        })
    }

    pub(crate) fn eval_scalar_literal(
        &self,
        row: usize,
        op: ScalarOp,
        literal: &Value,
        reverse: bool,
    ) -> Result<Option<Value>> {
        let Some(value) = self.scalar_value(row) else {
            return Ok(None);
        };
        Ok(Some(match value {
            ScalarValueRef::Null => eval_null_literal(op, literal),
            ScalarValueRef::Bool(value) => eval_bool_literal(value, op, literal, reverse)?,
            ScalarValueRef::Number(value) => eval_number_literal(value, op, literal, reverse)?,
            ScalarValueRef::Str(value) => eval_str_literal(value, op, literal, reverse)?,
        }))
    }

    fn scalar_value(&self, row: usize) -> Option<ScalarValueRef<'_>> {
        macro_rules! nullable {
            ($array:expr, $value:expr) => {
                if $array.is_null(row) {
                    ScalarValueRef::Null
                } else {
                    $value
                }
            };
        }

        Some(match self {
            Self::Bool(array) => nullable!(array, ScalarValueRef::Bool(array.value(row))),
            Self::I8(array) => nullable!(array, ScalarValueRef::Number(array.value(row) as f64)),
            Self::I16(array) => nullable!(array, ScalarValueRef::Number(array.value(row) as f64)),
            Self::I32(array) => nullable!(array, ScalarValueRef::Number(array.value(row) as f64)),
            Self::I64(array) => nullable!(array, ScalarValueRef::Number(array.value(row) as f64)),
            Self::U8(array) => nullable!(array, ScalarValueRef::Number(array.value(row) as f64)),
            Self::U16(array) => nullable!(array, ScalarValueRef::Number(array.value(row) as f64)),
            Self::U32(array) => nullable!(array, ScalarValueRef::Number(array.value(row) as f64)),
            Self::U64(array) => nullable!(array, ScalarValueRef::Number(array.value(row) as f64)),
            Self::F32(array) => nullable!(array, ScalarValueRef::Number(array.value(row) as f64)),
            Self::F64(array) => nullable!(array, ScalarValueRef::Number(array.value(row))),
            Self::Utf8(array) => nullable!(array, ScalarValueRef::Str(array.value(row))),
            Self::LargeUtf8(array) => nullable!(array, ScalarValueRef::Str(array.value(row))),
            Self::Utf8View(array) => nullable!(array, ScalarValueRef::Str(array.value(row))),
            Self::List(_) | Self::LargeList(_) | Self::Struct(_) => return None,
        })
    }
}

fn eval_null_literal(op: ScalarOp, literal: &Value) -> Value {
    match op {
        ScalarOp::Eq => Value::Bool(literal.is_null()),
        ScalarOp::NotEq => Value::Bool(!literal.is_null()),
        ScalarOp::Lt | ScalarOp::LtEq | ScalarOp::Gt | ScalarOp::GtEq => Value::Null,
        _ => unreachable!("fast scalar literal only handles comparison ops"),
    }
}

fn eval_non_null_null_literal(op: ScalarOp) -> Value {
    match op {
        ScalarOp::Eq => Value::Bool(false),
        ScalarOp::NotEq => Value::Bool(true),
        ScalarOp::Lt | ScalarOp::LtEq | ScalarOp::Gt | ScalarOp::GtEq => Value::Null,
        _ => unreachable!("fast scalar literal only handles comparison ops"),
    }
}

fn eval_bool_literal(value: bool, op: ScalarOp, literal: &Value, reverse: bool) -> Result<Value> {
    if literal.is_null() {
        return Ok(eval_non_null_null_literal(op));
    }
    let Some(rhs) = literal_bool(literal) else {
        return eval_incomparable_literal(op);
    };
    Ok(eval_ordering_or_eq(
        op,
        value == rhs,
        value.cmp(&rhs),
        reverse,
    ))
}

fn eval_number_literal(value: f64, op: ScalarOp, literal: &Value, reverse: bool) -> Result<Value> {
    if literal.is_null() {
        return Ok(eval_non_null_null_literal(op));
    }
    let Some(rhs) = literal.as_f64() else {
        return eval_incomparable_literal(op);
    };
    match op {
        ScalarOp::Eq => Ok(Value::Bool(value == rhs)),
        ScalarOp::NotEq => Ok(Value::Bool(value != rhs)),
        ScalarOp::Lt | ScalarOp::LtEq | ScalarOp::Gt | ScalarOp::GtEq => {
            let ordering = value.partial_cmp(&rhs).context("cannot compare values")?;
            Ok(Value::Bool(apply_ordering(op, ordering, reverse)))
        }
        _ => unreachable!("fast scalar literal only handles comparison ops"),
    }
}

fn eval_str_literal(value: &str, op: ScalarOp, literal: &Value, reverse: bool) -> Result<Value> {
    if literal.is_null() {
        return Ok(eval_non_null_null_literal(op));
    }
    let Value::Str(rhs) = literal else {
        return eval_incomparable_literal(op);
    };
    Ok(eval_ordering_or_eq(
        op,
        value == rhs.as_ref(),
        value.cmp(rhs.as_ref()),
        reverse,
    ))
}

fn literal_bool(literal: &Value) -> Option<bool> {
    match literal {
        Value::Bool(value) => Some(*value),
        _ => None,
    }
}

fn eval_incomparable_literal(op: ScalarOp) -> Result<Value> {
    match op {
        ScalarOp::Eq => Ok(Value::Bool(false)),
        ScalarOp::NotEq => Ok(Value::Bool(true)),
        ScalarOp::Lt | ScalarOp::LtEq | ScalarOp::Gt | ScalarOp::GtEq => {
            bail!("cannot compare values")
        }
        _ => unreachable!("fast scalar literal only handles comparison ops"),
    }
}

fn eval_ordering_or_eq(op: ScalarOp, equal: bool, ordering: Ordering, reverse: bool) -> Value {
    match op {
        ScalarOp::Eq => Value::Bool(equal),
        ScalarOp::NotEq => Value::Bool(!equal),
        ScalarOp::Lt | ScalarOp::LtEq | ScalarOp::Gt | ScalarOp::GtEq => {
            Value::Bool(apply_ordering(op, ordering, reverse))
        }
        _ => unreachable!("fast scalar literal only handles comparison ops"),
    }
}

fn apply_ordering(op: ScalarOp, ordering: Ordering, reverse: bool) -> bool {
    let ordering = if reverse {
        ordering.reverse()
    } else {
        ordering
    };
    match op {
        ScalarOp::Lt => ordering.is_lt(),
        ScalarOp::LtEq => ordering.is_le(),
        ScalarOp::Gt => ordering.is_gt(),
        ScalarOp::GtEq => ordering.is_ge(),
        _ => unreachable!("fast scalar literal only handles ordering ops"),
    }
}

pub(crate) fn array_to_values(array: &dyn Array) -> Result<Vec<Value>> {
    (0..array.len())
        .map(|row| array_row_to_value(array, row))
        .collect()
}

pub(crate) fn ipc_list_literal_to_value(bytes: &[u8]) -> Result<Value> {
    let mut reader = StreamReader::try_new(Cursor::new(bytes), None)
        .context("invalid Polars Arrow IPC list literal")?;
    let batch = reader
        .next()
        .transpose()
        .context("invalid Polars Arrow IPC list literal batch")?
        .context("Polars Arrow IPC list literal is empty")?;
    if batch.num_columns() != 1 {
        bail!(
            "Polars Arrow IPC list literal expected one column, got {} columns",
            batch.num_columns(),
        );
    }
    let column = batch.column(0).as_ref();
    (0..batch.num_rows())
        .map(|row| array_row_to_value(column, row))
        .collect::<Result<Vec<_>>>()
        .map(Value::List)
}

fn array_row_to_value(array: &dyn Array, row: usize) -> Result<Value> {
    macro_rules! primitive {
        ($array:ty, $value:expr) => {
            if let Some(array) = array.as_any().downcast_ref::<$array>() {
                return Ok(if array.is_null(row) {
                    Value::Null
                } else {
                    $value(array.value(row))
                });
            }
        };
    }

    primitive!(BooleanArray, Value::Bool);
    primitive!(Int8Array, |value| Value::I64(value as i64));
    primitive!(Int16Array, |value| Value::I64(value as i64));
    primitive!(Int32Array, |value| Value::I64(value as i64));
    primitive!(Int64Array, Value::I64);
    primitive!(UInt8Array, |value| Value::U64(value as u64));
    primitive!(UInt16Array, |value| Value::U64(value as u64));
    primitive!(UInt32Array, |value| Value::U64(value as u64));
    primitive!(UInt64Array, Value::U64);
    primitive!(Float32Array, |value| Value::F64(value as f64));
    primitive!(Float64Array, Value::F64);

    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(if array.is_null(row) {
            Value::Null
        } else {
            Value::Str(Arc::from(array.value(row)))
        });
    }
    if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(if array.is_null(row) {
            Value::Null
        } else {
            Value::Str(Arc::from(array.value(row)))
        });
    }
    if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
        return Ok(if array.is_null(row) {
            Value::Null
        } else {
            Value::Str(Arc::from(array.value(row)))
        });
    }
    if let Some(array) = array.as_any().downcast_ref::<ListArray>() {
        return Ok(if array.is_null(row) {
            Value::Null
        } else {
            Value::List(array_to_values(&array.value(row))?)
        });
    }
    if let Some(array) = array.as_any().downcast_ref::<LargeListArray>() {
        return Ok(if array.is_null(row) {
            Value::Null
        } else {
            Value::List(array_to_values(&array.value(row))?)
        });
    }
    if let Some(array) = array.as_any().downcast_ref::<StructArray>() {
        return Ok(if array.is_null(row) {
            Value::Null
        } else {
            struct_row_to_value(array, row)?
        });
    }

    bail!(
        "unsupported list/struct value type: {:?}",
        array.data_type()
    )
}

fn struct_row_to_value(array: &StructArray, row: usize) -> Result<Value> {
    Ok(Value::Struct(
        array
            .fields()
            .iter()
            .zip(array.columns())
            .map(|(field, column)| Ok((field.name().clone(), array_row_to_value(column, row)?)))
            .collect::<Result<_>>()?,
    ))
}
