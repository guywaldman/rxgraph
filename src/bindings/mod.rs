mod utils;
use pyo3::prelude::*;

use crate::py_dataclass;

#[pyclass]
pub struct Graph {
    #[pyo3()]
    node_schema: Schema,
}

#[pymethods]
impl Graph {
    #[new]
    fn new(node_schema: Schema) -> Self {
        Graph { node_schema }
    }
}

py_dataclass! {
    pub struct Schema {
        fields: Vec<SchemaField>,
    }
}

py_dataclass! {
    pub struct SchemaField {
        id: String,
        r#type: String,
    }
}
