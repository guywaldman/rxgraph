use super::*;

#[pyclass(name = "SearchStream", unsendable)]
pub(super) struct PySearchStream {
    pub(super) inner: Option<Box<dyn crate::NativeBatchStream>>,
    pub(super) batch_size: usize,
    pub(super) stats: SearchStats,
    pub(super) path_factory: Option<Py<PyAny>>,
}
#[pymethods]
impl PySearchStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Vec<Py<PyAny>>>> {
        let Some(stream) = self.inner.as_mut() else {
            return Ok(None);
        };
        let result = py.detach(|| stream.next_batch(self.batch_size));
        self.stats = stream.stats();
        match result {
            Ok(Some(result)) => {
                match owned_paths_into_py(py, result.paths, self.path_factory.as_ref()) {
                    Ok(paths) => Ok(Some(paths)),
                    Err(error) => {
                        self.close();
                        Err(error)
                    }
                }
            }
            Ok(None) => {
                self.close();
                Ok(None)
            }
            Err(error) => {
                self.close();
                Err(to_py_runtime_err(error))
            }
        }
    }
    fn close(&mut self) {
        self.path_factory = None;
        if let Some(mut stream) = self.inner.take() {
            self.stats = stream.stats();
            stream.close();
        }
    }
    fn _set_path_factory(&mut self, factory: Py<PyAny>) {
        self.path_factory = Some(factory);
    }
    #[getter]
    fn stats(&self) -> PySearchStats {
        self.stats.into()
    }
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __exit__(
        &mut self,
        _ty: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        self.close();
    }
}
impl Drop for PySearchStream {
    fn drop(&mut self) {
        self.close();
    }
}
