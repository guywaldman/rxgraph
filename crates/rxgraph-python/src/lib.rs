use pyo3::prelude::*;

/// The canonical `rxgraph` Python extension.
///
/// All binding logic lives in the reusable `rxgraph-py` crate. Downstream
/// native-kernel extensions use `rxgraph_py::plugin!`; see
/// `examples/rust-kernel-plugin/`.
#[pymodule]
fn _rxgraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    rxgraph_py::register(m)
}
