use anyhow::{Result, anyhow};
use arrow::array::RecordBatch;
use arrow_schema::DataType;

pub(crate) fn validate_field_exists(batch: &RecordBatch, col: &str) -> Result<DataType> {
    let schema = batch.schema();
    schema
        .field_with_name(col)
        .map(|f| f.data_type().clone())
        .map_err(|_| anyhow!("Missing the '{col}' column"))
}
