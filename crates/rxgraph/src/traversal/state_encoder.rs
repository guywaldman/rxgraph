//! Serde directly into traversal values, preserving the former JSON conversion contract.
use crate::{StateRow, Value};
use serde::{
    Serialize,
    ser::{self, Error as _},
};
use std::collections::BTreeMap;
type Error = serde_json::Error;
type Result<T> = std::result::Result<T, Error>;

pub(crate) fn encode<T: Serialize>(state: &T) -> anyhow::Result<StateRow> {
    match state.serialize(Encoder)? {
        Value::Struct(fields) => Ok(fields),
        _ => anyhow::bail!("kernel state must serialize as an object"),
    }
}
struct Encoder;
impl ser::Serializer for Encoder {
    type Ok = Value;
    type Error = Error;
    type SerializeSeq = Sequence;
    type SerializeTuple = Sequence;
    type SerializeTupleStruct = Sequence;
    type SerializeTupleVariant = Sequence;
    type SerializeMap = Object;
    type SerializeStruct = Object;
    type SerializeStructVariant = Object;
    fn serialize_bool(self, v: bool) -> Result<Value> {
        Ok(Value::Bool(v))
    }
    fn serialize_i8(self, v: i8) -> Result<Value> {
        self.serialize_i64(v.into())
    }
    fn serialize_i16(self, v: i16) -> Result<Value> {
        self.serialize_i64(v.into())
    }
    fn serialize_i32(self, v: i32) -> Result<Value> {
        self.serialize_i64(v.into())
    }
    fn serialize_i64(self, v: i64) -> Result<Value> {
        Ok(if v >= 0 {
            Value::U64(v as u64)
        } else {
            Value::I64(v)
        })
    }
    fn serialize_i128(self, v: i128) -> Result<Value> {
        if let Ok(v) = i64::try_from(v) {
            self.serialize_i64(v)
        } else if let Ok(v) = u64::try_from(v) {
            self.serialize_u64(v)
        } else {
            Err(Error::custom("number out of range"))
        }
    }
    fn serialize_u8(self, v: u8) -> Result<Value> {
        self.serialize_u64(v.into())
    }
    fn serialize_u16(self, v: u16) -> Result<Value> {
        self.serialize_u64(v.into())
    }
    fn serialize_u32(self, v: u32) -> Result<Value> {
        self.serialize_u64(v.into())
    }
    fn serialize_u64(self, v: u64) -> Result<Value> {
        Ok(Value::U64(v))
    }
    fn serialize_u128(self, v: u128) -> Result<Value> {
        self.serialize_u64(v.try_into().map_err(Error::custom)?)
    }
    fn serialize_f32(self, v: f32) -> Result<Value> {
        self.serialize_f64(v.into())
    }
    fn serialize_f64(self, v: f64) -> Result<Value> {
        Ok(if v.is_finite() {
            Value::F64(v)
        } else {
            Value::Null
        })
    }
    fn serialize_char(self, v: char) -> Result<Value> {
        self.serialize_str(&v.to_string())
    }
    fn serialize_str(self, v: &str) -> Result<Value> {
        Ok(Value::Str(v.into()))
    }
    fn serialize_bytes(self, v: &[u8]) -> Result<Value> {
        Ok(Value::List(
            v.iter().map(|&v| Value::U64(v.into())).collect(),
        ))
    }
    fn serialize_none(self) -> Result<Value> {
        Ok(Value::Null)
    }
    fn serialize_some<T: ?Sized + Serialize>(self, v: &T) -> Result<Value> {
        v.serialize(self)
    }
    fn serialize_unit(self) -> Result<Value> {
        Ok(Value::Null)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<Value> {
        self.serialize_unit()
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<Value> {
        self.serialize_str(variant)
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        v: &T,
    ) -> Result<Value> {
        v.serialize(self)
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        v: &T,
    ) -> Result<Value> {
        Ok(Value::Struct(vec![(variant.into(), v.serialize(self)?)]))
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Sequence> {
        Ok(Sequence {
            values: Vec::with_capacity(len.unwrap_or(0)),
            variant: None,
        })
    }
    fn serialize_tuple(self, len: usize) -> Result<Sequence> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_struct(self, _: &'static str, len: usize) -> Result<Sequence> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Sequence> {
        Ok(Sequence {
            values: Vec::with_capacity(len),
            variant: Some(variant),
        })
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Object> {
        Ok(Object {
            values: BTreeMap::new(),
            key: None,
            variant: None,
        })
    }
    fn serialize_struct(self, _: &'static str, len: usize) -> Result<Object> {
        self.serialize_map(Some(len))
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Object> {
        Ok(Object {
            values: BTreeMap::new(),
            key: None,
            variant: Some(variant),
        })
    }
}
struct Sequence {
    values: Vec<Value>,
    variant: Option<&'static str>,
}
impl Sequence {
    fn push<T: ?Sized + Serialize>(&mut self, v: &T) -> Result<()> {
        self.values.push(v.serialize(Encoder)?);
        Ok(())
    }
    fn finish(self) -> Result<Value> {
        let value = Value::List(self.values);
        Ok(match self.variant {
            Some(name) => Value::Struct(vec![(name.into(), value)]),
            None => value,
        })
    }
}
macro_rules! sequence {
    ($trait:ident,$method:ident) => {
        impl ser::$trait for Sequence {
            type Ok = Value;
            type Error = Error;
            fn $method<T: ?Sized + Serialize>(&mut self, v: &T) -> Result<()> {
                self.push(v)
            }
            fn end(self) -> Result<Value> {
                self.finish()
            }
        }
    };
}
sequence!(SerializeSeq, serialize_element);
sequence!(SerializeTuple, serialize_element);
sequence!(SerializeTupleStruct, serialize_field);
sequence!(SerializeTupleVariant, serialize_field);
struct Object {
    values: BTreeMap<String, Value>,
    key: Option<String>,
    variant: Option<&'static str>,
}
impl Object {
    fn field<T: ?Sized + Serialize>(&mut self, key: &str, value: &T) -> Result<()> {
        self.values.insert(key.into(), value.serialize(Encoder)?);
        Ok(())
    }
    fn finish(self) -> Result<Value> {
        let value = Value::Struct(self.values.into_iter().collect());
        Ok(match self.variant {
            Some(name) => Value::Struct(vec![(name.into(), value)]),
            None => value,
        })
    }
}
impl ser::SerializeMap for Object {
    type Ok = Value;
    type Error = Error;
    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<()> {
        self.key = Some(key.serialize(KeyEncoder)?);
        Ok(())
    }
    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        let key = self
            .key
            .take()
            .ok_or_else(|| Error::custom("map value without key"))?;
        self.values.insert(key, value.serialize(Encoder)?);
        Ok(())
    }
    fn end(self) -> Result<Value> {
        self.finish()
    }
}
macro_rules! object {
    ($trait:ident) => {
        impl ser::$trait for Object {
            type Ok = Value;
            type Error = Error;
            fn serialize_field<T: ?Sized + Serialize>(
                &mut self,
                key: &'static str,
                value: &T,
            ) -> Result<()> {
                self.field(key, value)
            }
            fn end(self) -> Result<Value> {
                self.finish()
            }
        }
    };
}
object!(SerializeStruct);
object!(SerializeStructVariant);

// JSON-compatible object keys without constructing an intermediate object.
struct KeyEncoder;
impl ser::Serializer for KeyEncoder {
    type Ok = String;
    type Error = Error;
    type SerializeSeq = ser::Impossible<String, Error>;
    type SerializeTuple = ser::Impossible<String, Error>;
    type SerializeTupleStruct = ser::Impossible<String, Error>;
    type SerializeTupleVariant = ser::Impossible<String, Error>;
    type SerializeMap = ser::Impossible<String, Error>;
    type SerializeStruct = ser::Impossible<String, Error>;
    type SerializeStructVariant = ser::Impossible<String, Error>;
    fn serialize_bool(self, v: bool) -> Result<String> {
        Ok(v.to_string())
    }
    fn serialize_i8(self, v: i8) -> Result<String> {
        Ok(v.to_string())
    }
    fn serialize_i16(self, v: i16) -> Result<String> {
        Ok(v.to_string())
    }
    fn serialize_i32(self, v: i32) -> Result<String> {
        Ok(v.to_string())
    }
    fn serialize_i64(self, v: i64) -> Result<String> {
        Ok(v.to_string())
    }
    fn serialize_i128(self, v: i128) -> Result<String> {
        Ok(v.to_string())
    }
    fn serialize_u8(self, v: u8) -> Result<String> {
        Ok(v.to_string())
    }
    fn serialize_u16(self, v: u16) -> Result<String> {
        Ok(v.to_string())
    }
    fn serialize_u32(self, v: u32) -> Result<String> {
        Ok(v.to_string())
    }
    fn serialize_u64(self, v: u64) -> Result<String> {
        Ok(v.to_string())
    }
    fn serialize_u128(self, v: u128) -> Result<String> {
        Ok(v.to_string())
    }
    fn serialize_char(self, v: char) -> Result<String> {
        Ok(v.to_string())
    }
    fn serialize_str(self, v: &str) -> Result<String> {
        Ok(v.into())
    }
    fn serialize_f32(self, v: f32) -> Result<String> {
        if v.is_finite() {
            Ok(serde_json::to_string(&v)?)
        } else {
            Err(Error::custom("float key must be finite"))
        }
    }
    fn serialize_f64(self, v: f64) -> Result<String> {
        if v.is_finite() {
            Ok(serde_json::to_string(&v)?)
        } else {
            Err(Error::custom("float key must be finite"))
        }
    }
    fn serialize_some<T: ?Sized + Serialize>(self, v: &T) -> Result<String> {
        v.serialize(self)
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        v: &T,
    ) -> Result<String> {
        v.serialize(self)
    }
    fn serialize_unit_variant(self, _: &'static str, _: u32, v: &'static str) -> Result<String> {
        Ok(v.into())
    }
    fn serialize_bytes(self, _: &[u8]) -> Result<String> {
        Err(Error::custom("key must be a string"))
    }
    fn serialize_none(self) -> Result<String> {
        Err(Error::custom("key must be a string"))
    }
    fn serialize_unit(self) -> Result<String> {
        Err(Error::custom("key must be a string"))
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<String> {
        Err(Error::custom("key must be a string"))
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: &T,
    ) -> Result<String> {
        Err(Error::custom("key must be a string"))
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq> {
        Err(Error::custom("key must be a string"))
    }
    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple> {
        Err(Error::custom("key must be a string"))
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        Err(Error::custom("key must be a string"))
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        Err(Error::custom("key must be a string"))
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap> {
        Err(Error::custom("key must be a string"))
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self::SerializeStruct> {
        Err(Error::custom("key must be a string"))
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant> {
        Err(Error::custom("key must be a string"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Serialize)]
    enum Choice {
        Unit,
        New(u64),
        Tuple(i64, String),
        Fields { value: Vec<u8> },
    }
    #[derive(Serialize)]
    struct State {
        positive: i64,
        negative: i64,
        float: f64,
        nested: Vec<Choice>,
        map: BTreeMap<u64, Option<u64>>,
    }
    #[test]
    fn matches_json_contract() {
        let state = State {
            positive: 7,
            negative: -3,
            float: f64::NAN,
            nested: vec![
                Choice::Unit,
                Choice::New(2),
                Choice::Tuple(-1, "x".into()),
                Choice::Fields { value: vec![1, 2] },
            ],
            map: BTreeMap::from([(5, Some(3)), (9, None)]),
        };
        let actual = Value::Struct(encode(&state).unwrap()).to_value();
        assert_eq!(actual, serde_json::to_value(&state).unwrap());
        assert!(encode(&[1, 2, 3]).is_err());
    }
}
