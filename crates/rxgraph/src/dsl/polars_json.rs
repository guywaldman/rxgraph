use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde_json::Value as Json;

use crate::dsl::{
    Value,
    expr::{ColumnRef, Expr},
    ops::{
        list::{ListOp, SetOp},
        scalar::{CastType, ScalarOp, parse_binary_op},
        string::StringOp,
        struct_::StructOp,
    },
};

pub(crate) fn parse_polars_json(json: &str) -> Result<Expr<ColumnRef>> {
    parse_expr(&serde_json::from_str(json)?)
}

fn parse_expr(value: &Json) -> Result<Expr<ColumnRef>> {
    if value.as_str() == Some("Element") {
        return Ok(Expr::Element);
    }

    let object = value.as_object().context("expected expression object")?;
    if let Some(column) = object.get("Column") {
        return parse_column(column.as_str().context("Column must be a string")?);
    }
    if let Some(literal) = object.get("Literal") {
        return Ok(Expr::Literal(parse_literal(literal)?));
    }
    if let Some(binary) = object.get("BinaryExpr").and_then(Json::as_object) {
        return Ok(Expr::Scalar(
            parse_binary_op(
                binary
                    .get("op")
                    .and_then(Json::as_str)
                    .context("binary op must be a string")?,
            )?,
            vec![
                parse_expr(binary.get("left").context("BinaryExpr missing left")?)?,
                parse_expr(binary.get("right").context("BinaryExpr missing right")?)?,
            ],
        ));
    }
    if let Some(cast) = object.get("Cast").and_then(Json::as_object) {
        return Ok(Expr::Scalar(
            ScalarOp::Cast(parse_cast_type(
                cast.get("dtype").context("Cast missing dtype")?,
            )?),
            vec![parse_expr(cast.get("expr").context("Cast missing expr")?)?],
        ));
    }
    if let Some(alias) = object.get("Alias").and_then(Json::as_array) {
        ensure_arity(alias, 2)?;
        let name = alias[1].as_str().context("Alias name must be a string")?;
        return Ok(Expr::Alias(
            Box::new(parse_expr(&alias[0])?),
            name.to_owned(),
        ));
    }
    if let Some(rx) = object.get("Rx") {
        return parse_rx_expr(rx);
    }
    if let Some(function) = object.get("Function") {
        return parse_function(function);
    }
    if let Some(eval) = object.get("Eval") {
        return parse_eval(eval);
    }
    if let Some(eval) = object.get("StructEval") {
        return parse_struct_eval(eval);
    }
    if let Some(filter) = object.get("Filter") {
        return parse_filter(filter);
    }
    bail!("unsupported Polars expression JSON: {value}")
}

fn parse_column(name: &str) -> Result<Expr<ColumnRef>> {
    if name.is_empty() {
        return Ok(Expr::Element);
    }
    Ok(Expr::Column(match name {
        "src.id" => ColumnRef::SrcId,
        "dest.id" => ColumnRef::DestId,
        "edge.id" => ColumnRef::EdgeId,
        _ => {
            let (scope, field) = name
                .split_once('.')
                .ok_or_else(|| anyhow::anyhow!("column {name:?} must use scope.field"))?;
            match scope {
                "src" => ColumnRef::SrcField(field.to_owned()),
                "dest" => ColumnRef::DestField(field.to_owned()),
                "edge" => ColumnRef::EdgeField(field.to_owned()),
                "state" => ColumnRef::State(field.to_owned()),
                _ => bail!("unknown column scope {scope:?} in {name:?}"),
            }
        }
    }))
}

fn parse_rx_expr(value: &Json) -> Result<Expr<ColumnRef>> {
    let object = value
        .as_object()
        .context("Rx expression must be an object")?;
    let value = object
        .get("MaskIfAny")
        .context("unsupported Rx expression")?
        .as_object()
        .context("MaskIfAny expression must be an object")?;
    Ok(Expr::Scalar(
        ScalarOp::MaskIfAny,
        vec![
            parse_expr(value.get("value").context("MaskIfAny missing value")?)?,
            parse_expr(value.get("mask").context("MaskIfAny missing mask")?)?,
            parse_expr(
                value
                    .get("then")
                    .or_else(|| value.get("then_value"))
                    .context("MaskIfAny missing then")?,
            )?,
        ],
    ))
}

fn parse_literal(value: &Json) -> Result<Value> {
    parse_scalar_literal(
        value
            .pointer("/Value")
            .or_else(|| value.pointer("/Scalar"))
            .or_else(|| value.pointer("/Dyn"))
            .context("unsupported literal")?,
    )
}

fn parse_scalar_literal(value: &Json) -> Result<Value> {
    let object = value.as_object().context("literal must be object")?;
    if let Some(value) = object.get("String") {
        return Ok(Value::Str(Arc::from(
            value.as_str().context("String literal must be string")?,
        )));
    }
    if let Some(value) = object.get("Boolean") {
        return Ok(Value::Bool(
            value.as_bool().context("Boolean literal must be bool")?,
        ));
    }
    if object.contains_key("Null") {
        return Ok(Value::Null);
    }
    if let Some(value) = object.get("Int") {
        return Ok(Value::I64(
            value.as_i64().context("Int literal must be i64")?,
        ));
    }
    if let Some(value) = object.get("UInt") {
        return Ok(Value::U64(
            value.as_u64().context("UInt literal must be u64")?,
        ));
    }
    if let Some(value) = object.get("Float") {
        return Ok(Value::F64(
            value.as_f64().context("Float literal must be f64")?,
        ));
    }
    if object.contains_key("List") {
        bail!("Polars JSON list literals are not supported; use list columns or state");
    }
    bail!("unsupported literal {value}")
}

fn parse_function(value: &Json) -> Result<Expr<ColumnRef>> {
    let object = value.as_object().context("Function must be an object")?;
    let inputs = object
        .get("input")
        .and_then(Json::as_array)
        .context("Function missing input array")?;
    let function = object
        .get("function")
        .context("Function missing function")?;

    if function.pointer("/Boolean").and_then(Json::as_str) == Some("Not") {
        return unary(inputs, |input| Expr::Scalar(ScalarOp::Not, vec![input]));
    }
    if function.pointer("/Boolean").and_then(Json::as_str) == Some("IsNull") {
        return unary(inputs, |input| Expr::Scalar(ScalarOp::IsNull, vec![input]));
    }
    if function.pointer("/Boolean").and_then(Json::as_str) == Some("IsNotNull") {
        return unary(inputs, |input| {
            Expr::Scalar(ScalarOp::IsNotNull, vec![input])
        });
    }
    if function.pointer("/StringExpr/Contains").is_some() {
        return op(inputs, |args| Expr::String(StringOp::Contains, args));
    }
    if function.pointer("/StringExpr/StartsWith").is_some() {
        return op(inputs, |args| Expr::String(StringOp::StartsWith, args));
    }
    if function.pointer("/StringExpr/EndsWith").is_some() {
        return op(inputs, |args| Expr::String(StringOp::EndsWith, args));
    }
    if let Some(list) = function.get("ListExpr") {
        return parse_list_function(list, inputs);
    }
    if let Some(struct_) = function.get("StructExpr") {
        return parse_struct_function(struct_, inputs);
    }
    bail!("unsupported Polars function {function}")
}

fn parse_list_function(value: &Json, inputs: &[Json]) -> Result<Expr<ColumnRef>> {
    let list_op = match value.as_str() {
        Some("Concat") => ListOp::Concat,
        Some("Length") => ListOp::Len,
        Some("Sum") => ListOp::Sum,
        Some("Min") => ListOp::Min,
        Some("Max") => ListOp::Max,
        Some("Mean") => ListOp::Mean,
        Some("Median") => ListOp::Median,
        Some("Reverse") => ListOp::Reverse,
        Some("DropNulls") => ListOp::DropNulls,
        Some("CountMatches") => ListOp::CountMatches,
        Some("NUnique") => ListOp::NUnique,
        Some("Shift") => ListOp::Shift,
        Some("GatherEvery") => ListOp::GatherEvery,
        Some("Slice") => ListOp::Slice,
        Some(other) => bail!("unsupported Polars list function {other:?}"),
        None => parse_list_object(value)?,
    };
    op(inputs, |args| Expr::List(list_op, args))
}

fn parse_list_object(value: &Json) -> Result<ListOp> {
    let object = value
        .as_object()
        .context("ListExpr must be string or object")?;
    if let Some(value) = object.get("Contains") {
        return Ok(ListOp::Contains {
            nulls_equal: value
                .get("nulls_equal")
                .and_then(Json::as_bool)
                .unwrap_or(true),
        });
    }
    if let Some(value) = object.get("Get") {
        return Ok(ListOp::Get {
            null_on_oob: value.as_bool().unwrap_or(false),
        });
    }
    if object.contains_key("Unique") {
        return Ok(ListOp::Unique);
    }
    if let Some(sort) = object.get("Sort").and_then(Json::as_object) {
        return Ok(ListOp::Sort {
            descending: sort
                .get("descending")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            nulls_last: sort
                .get("nulls_last")
                .and_then(Json::as_bool)
                .unwrap_or(false),
        });
    }
    if let Some(value) = object.get("SetOperation").and_then(Json::as_str) {
        return Ok(ListOp::Set(match value {
            "Union" => SetOp::Union,
            "Intersection" => SetOp::Intersection,
            "Difference" => SetOp::Difference,
            "SymmetricDifference" => SetOp::SymmetricDifference,
            other => bail!("unsupported list set operation {other:?}"),
        }));
    }
    if let Some(value) = object.get("Join") {
        return Ok(ListOp::Join {
            ignore_nulls: value.as_bool().unwrap_or(true),
        });
    }
    if let Some(fields) = object.get("ToStruct").and_then(Json::as_array) {
        return Ok(ListOp::ToStruct(
            fields
                .iter()
                .map(|field| {
                    field
                        .as_str()
                        .map(str::to_owned)
                        .context("ToStruct field names must be strings")
                })
                .collect::<Result<_>>()?,
        ));
    }
    bail!("unsupported Polars list function {value}")
}

fn parse_struct_function(value: &Json, inputs: &[Json]) -> Result<Expr<ColumnRef>> {
    let struct_op = if value.as_str() == Some("JsonEncode") {
        StructOp::JsonEncode
    } else if let Some(name) = value.pointer("/FieldByName").and_then(Json::as_str) {
        StructOp::FieldByName(name.to_owned())
    } else if let Some(names) = value.pointer("/RenameFields").and_then(Json::as_array) {
        StructOp::RenameFields(
            names
                .iter()
                .map(|name| {
                    name.as_str()
                        .map(str::to_owned)
                        .context("RenameFields names must be strings")
                })
                .collect::<Result<_>>()?,
        )
    } else {
        bail!("unsupported Polars struct function {value}")
    };
    op(inputs, |args| Expr::Struct(struct_op, args))
}

fn parse_eval(value: &Json) -> Result<Expr<ColumnRef>> {
    let object = value.as_object().context("Eval must be an object")?;
    let list = parse_expr(object.get("expr").context("Eval missing expr")?)?;
    let evaluation = object
        .get("evaluation")
        .context("Eval missing evaluation")?;
    match object
        .get("variant")
        .and_then(Json::as_str)
        .context("Eval missing variant")?
    {
        "List" => {
            if let Some(filter) = evaluation.get("Filter") {
                return parse_filter_with_list(list, filter);
            }
            Ok(Expr::List(
                ListOp::Eval,
                vec![list, parse_expr(evaluation)?],
            ))
        }
        "ListAgg" => {
            let op = list_agg_op(evaluation)?;
            Ok(Expr::List(op, vec![list]))
        }
        variant => bail!("unsupported Eval variant {variant:?}"),
    }
}

fn parse_filter(value: &Json) -> Result<Expr<ColumnRef>> {
    parse_filter_with_list(Expr::Element, value)
}

fn parse_filter_with_list(list: Expr<ColumnRef>, value: &Json) -> Result<Expr<ColumnRef>> {
    let object = value.as_object().context("Filter must be an object")?;
    Ok(Expr::List(
        ListOp::Filter,
        vec![
            list,
            parse_expr(object.get("by").context("Filter missing by")?)?,
        ],
    ))
}

fn list_agg_op(value: &Json) -> Result<ListOp> {
    let function = value
        .get("Function")
        .and_then(Json::as_object)
        .and_then(|object| object.get("function"))
        .context("ListAgg must contain a Function")?;
    if let Some(all) = function.pointer("/Boolean/All").and_then(Json::as_object) {
        return Ok(ListOp::All {
            ignore_nulls: all
                .get("ignore_nulls")
                .and_then(Json::as_bool)
                .unwrap_or(true),
        });
    }
    if let Some(any) = function.pointer("/Boolean/Any").and_then(Json::as_object) {
        return Ok(ListOp::Any {
            ignore_nulls: any
                .get("ignore_nulls")
                .and_then(Json::as_bool)
                .unwrap_or(true),
        });
    }
    bail!("unsupported ListAgg expression {value}")
}

fn parse_struct_eval(value: &Json) -> Result<Expr<ColumnRef>> {
    let object = value.as_object().context("StructEval must be an object")?;
    let mut args = vec![parse_expr(
        object.get("expr").context("StructEval missing expr")?,
    )?];
    let mut names = Vec::new();
    for expr in object
        .get("evaluation")
        .and_then(Json::as_array)
        .context("StructEval missing evaluation")?
    {
        let Expr::Alias(expr, name) = parse_expr(expr)? else {
            bail!("StructEval fields must be aliases")
        };
        names.push(name);
        args.push(*expr);
    }
    Ok(Expr::Struct(StructOp::WithFields(names), args))
}

fn parse_cast_type(value: &Json) -> Result<CastType> {
    let dtype = value
        .get("Literal")
        .and_then(Json::as_str)
        .context("only literal cast dtypes are supported")?;
    Ok(match dtype {
        "Boolean" => CastType::Bool,
        "Int64" | "Int32" | "Int16" | "Int8" => CastType::Int64,
        "UInt64" | "UInt32" | "UInt16" | "UInt8" => CastType::UInt64,
        "Float64" | "Float32" => CastType::Float64,
        "String" | "Utf8" => CastType::String,
        other => bail!("unsupported cast dtype {other:?}"),
    })
}

fn unary(
    inputs: &[Json],
    wrap: impl FnOnce(Expr<ColumnRef>) -> Expr<ColumnRef>,
) -> Result<Expr<ColumnRef>> {
    ensure_arity(inputs, 1)?;
    Ok(wrap(parse_expr(&inputs[0])?))
}

fn op(
    inputs: &[Json],
    wrap: impl FnOnce(Vec<Expr<ColumnRef>>) -> Expr<ColumnRef>,
) -> Result<Expr<ColumnRef>> {
    inputs
        .iter()
        .map(parse_expr)
        .collect::<Result<Vec<_>>>()
        .map(wrap)
}

fn ensure_arity(inputs: &[Json], len: usize) -> Result<()> {
    if inputs.len() == len {
        Ok(())
    } else {
        bail!("expected {len} function inputs, got {}", inputs.len())
    }
}
