use super::*;

#[pyclass(name = "SearchStream", unsendable)]
pub(super) struct PySearchStream {
    pub(super) inner: Option<Box<dyn crate::NativeBatchStream>>,
    pub(super) batch_size: usize,
    pub(super) stats: SearchStats,
}
#[pymethods]
impl PySearchStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PySearchResult>> {
        let Some(stream) = self.inner.as_mut() else {
            return Ok(None);
        };
        let result = py.detach(|| stream.next_batch(self.batch_size));
        self.stats = stream.stats();
        match result {
            Ok(Some(result)) => PySearchResult::from_owned_result(py, result).map(Some),
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
        if let Some(mut stream) = self.inner.take() {
            self.stats = stream.stats();
            stream.close();
        }
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
