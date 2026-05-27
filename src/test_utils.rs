use std::sync::Arc;

use arrow::{
    array::{ArrayRef, Int32Array, StringArray, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

pub(crate) fn nodes(ids: &[u64], kinds: &[&str], scores: &[i32]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("kind", DataType::Utf8, false),
            Field::new("score", DataType::Int32, false),
        ])),
        vec![
            Arc::new(UInt64Array::from(ids.to_vec())) as ArrayRef,
            Arc::new(StringArray::from(kinds.to_vec())),
            Arc::new(Int32Array::from(scores.to_vec())),
        ],
    )
    .unwrap()
}

pub(crate) fn edges(src: &[u64], dest: &[u64], rel: &[&str]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::UInt64, false),
            Field::new("dest", DataType::UInt64, false),
            Field::new("rel", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(UInt64Array::from(src.to_vec())) as ArrayRef,
            Arc::new(UInt64Array::from(dest.to_vec())),
            Arc::new(StringArray::from(rel.to_vec())),
        ],
    )
    .unwrap()
}
