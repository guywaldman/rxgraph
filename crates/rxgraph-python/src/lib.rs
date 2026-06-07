use pyo3::prelude::*;

/// The canonical `rxgraph` Python extension.
///
/// All binding logic lives in the `rxgraph` crate behind its `python` feature.
/// Downstream native-kernel extensions use `rxgraph::plugin!`; see
/// `examples/rust-kernel-plugin/`.
#[pymodule]
fn _rxgraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    rxgraph::register(m)
}
