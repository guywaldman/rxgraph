mod bindings;
mod graph;

// TODO: Scope down.
pub use graph::*;

use pyo3::prelude::*;

#[pymodule]
fn rxgraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bindings::Graph>()?;
    m.add_class::<bindings::Schema>()?;
    m.add_class::<bindings::SchemaField>()?;
    Ok(())
}
