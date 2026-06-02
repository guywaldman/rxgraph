use pyo3::{
    Borrowed,
    conversion::IntoPyObjectExt,
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyDict, PyList, PyString},
};
use pyo3_arrow::PyTable;
use rayon::ThreadPoolBuilder;
use rxgraph::{
    DslKernel, Graph, GraphId, OwnedGraphId, SearchResult, SearchStats, StateRow,
    TraversalConfigBuilder, TraversalStrategy, Value,
};
use std::thread;

#[pymodule]
fn _rxgraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    initialize_rayon_pool();

    m.add_class::<PyGraph>()?;
    m.add_class::<PyKernel>()?;
    m.add_class::<PyTraversal>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PySearchStats>()?;
    m.add_class::<PySearchPath>()?;
    m.add_function(wrap_pyfunction!(rayon_thread_count, m)?)?;
    Ok(())
}

fn initialize_rayon_pool() {
    let threads = thread::available_parallelism().map_or(1, usize::from);
    let _ = ThreadPoolBuilder::new().num_threads(threads).build_global();
}

#[pyfunction]
fn rayon_thread_count() -> usize {
    rayon::current_num_threads()
}

#[pyclass(name = "Graph", unsendable)]
struct PyGraph {
    inner: Graph,
}

#[pymethods]
impl PyGraph {
    #[new]
    #[pyo3(signature = (nodes, edges))]
    fn new(nodes: PyTable, edges: PyTable) -> PyResult<Self> {
        Ok(Self {
            inner: Graph::new(one_batch(nodes, "nodes")?, one_batch(edges, "edges")?)
                .map_err(to_py_value_err)?,
        })
    }

    fn search(&self, py: Python<'_>, traversal: &PyTraversal) -> PyResult<PySearchResult> {
        let mut builder = TraversalConfigBuilder::new(traversal.kernel.clone())
            .with_start_nodes(traversal.start_nodes.clone())
            .with_strategy(traversal.strategy)
            .with_parallelism(traversal.parallel)
            .with_intermediate_states(traversal.intermediate_states)
            .with_progress(traversal.progress);

        if let Some(max_depth) = traversal.max_depth {
            builder = builder.with_max_depth(max_depth);
        }
        if let Some(max_paths) = traversal.max_paths {
            builder = builder.with_max_paths(max_paths);
        }

        PySearchResult::from_result(
            py,
            self.inner
                .search(builder.build())
                .map_err(to_py_runtime_err)?,
        )
    }

    #[pyo3(signature = (start, max_depth = None))]
    fn bfs(
        &self,
        py: Python<'_>,
        start: PyGraphId,
        max_depth: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        if let OwnedGraphId::U64(value) = &start.0
            && let Some(ids) = self
                .inner
                .bfs_u64(*value, max_depth)
                .map_err(to_py_value_err)?
        {
            return ids.into_py_any(py);
        }
        ids_to_py(
            py,
            self.inner
                .bfs(start.0, max_depth)
                .map_err(to_py_value_err)?,
        )
    }

    #[pyo3(signature = (start, max_depth = None))]
    fn dfs(
        &self,
        py: Python<'_>,
        start: PyGraphId,
        max_depth: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        if let OwnedGraphId::U64(value) = &start.0
            && let Some(ids) = self
                .inner
                .dfs_u64(*value, max_depth)
                .map_err(to_py_value_err)?
        {
            return ids.into_py_any(py);
        }
        ids_to_py(
            py,
            self.inner
                .dfs(start.0, max_depth)
                .map_err(to_py_value_err)?,
        )
    }

    fn reachable_nodes(&self, py: Python<'_>, start: PyGraphId) -> PyResult<Py<PyAny>> {
        if let OwnedGraphId::U64(value) = &start.0
            && let Some(ids) = self
                .inner
                .reachable_nodes_u64(*value)
                .map_err(to_py_value_err)?
        {
            return ids.into_py_any(py);
        }
        ids_to_py(
            py,
            self.inner
                .reachable_nodes(start.0)
                .map_err(to_py_value_err)?,
        )
    }

    fn shortest_path(
        &self,
        py: Python<'_>,
        source: PyGraphId,
        target: PyGraphId,
    ) -> PyResult<Option<Py<PyAny>>> {
        if let (OwnedGraphId::U64(source), OwnedGraphId::U64(target)) = (&source.0, &target.0)
            && let Some(path) = self
                .inner
                .shortest_path_u64(*source, *target)
                .map_err(to_py_value_err)?
        {
            return path.map(|path| path.into_py_any(py)).transpose();
        }
        self.inner
            .shortest_path(source.0, target.0)
            .map_err(to_py_value_err)?
            .map(|path| ids_to_py(py, path))
            .transpose()
    }

    fn out_degrees(&self) -> Vec<usize> {
        self.inner.out_degrees()
    }

    fn in_degrees(&self) -> Vec<usize> {
        self.inner.in_degrees()
    }

    fn degrees(&self) -> Vec<usize> {
        self.inner.degrees()
    }

    fn weakly_connected_components(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(components) = self.inner.weakly_connected_components_u64() {
            return components.into_py_any(py);
        }
        components_to_py(py, self.inner.weakly_connected_components())
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    #[getter]
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Replaces the payload tables in place, reusing the existing topology.
    ///
    /// Used by lazy graphs to have column-projected payload batches before a search.
    /// This is only used for optimizations, for example in cases where nodes/edges contain many columns
    /// and we want to lazily pull their payloads to reduce memory strain.
    /// Each table must have one row per node/edge in internal-ID order.
    fn set_payloads(&mut self, nodes: PyTable, edges: PyTable) -> PyResult<()> {
        self.inner
            .set_payloads(one_batch(nodes, "nodes")?, one_batch(edges, "edges")?)
            .map_err(to_py_value_err)
    }
}

fn one_batch(table: PyTable, label: &str) -> PyResult<arrow::record_batch::RecordBatch> {
    let (mut batches, _) = table.into_inner();
    if batches.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "{label} must be a single Arrow record batch"
        )));
    }
    Ok(batches.remove(0))
}

#[pyclass(name = "Kernel", unsendable)]
struct PyKernel {
    inner: DslKernel,
}

#[pymethods]
impl PyKernel {
    #[new]
    #[pyo3(signature = (visit, next_state, stop, initial_state))]
    fn new(
        visit: &Bound<'_, PyAny>,
        next_state: &Bound<'_, PyDict>,
        stop: &Bound<'_, PyAny>,
        initial_state: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let next_state = next_state
            .iter()
            .map(|(key, value)| {
                let key = key
                    .cast::<PyString>()
                    .map_err(|_| PyTypeError::new_err("next_state keys must be strings"))?
                    .to_str()?
                    .to_string();
                Ok((key, serialize_polars_expr(&value)?))
            })
            .collect::<PyResult<Vec<_>>>()?;

        let initial_state = initial_state
            .iter()
            .map(|(key, value)| {
                let key = key
                    .cast::<PyString>()
                    .map_err(|_| PyTypeError::new_err("initial_state keys must be strings"))?
                    .to_str()?
                    .to_string();
                Ok((key, py_to_value(&value)?))
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            inner: DslKernel::from_polars_json(
                &serialize_polars_expr(visit)?,
                next_state,
                &serialize_polars_expr(stop)?,
                initial_state,
            )
            .map_err(to_py_value_err)?,
        })
    }
}

fn serialize_polars_expr(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(json) = value.extract::<String>() {
        return Ok(json);
    }

    let meta = value.getattr("meta").map_err(|_| {
        PyTypeError::new_err("expected a Polars Expr or serialized Polars expression JSON")
    })?;
    let kwargs = PyDict::new(value.py());
    kwargs.set_item("format", "json")?;
    meta.call_method("serialize", (), Some(&kwargs))?.extract()
}

#[pyclass(name = "Traversal", unsendable)]
struct PyTraversal {
    kernel: DslKernel,
    start_nodes: Vec<OwnedGraphId>,
    max_depth: Option<usize>,
    max_paths: Option<usize>,
    strategy: TraversalStrategy,
    parallel: bool,
    intermediate_states: bool,
    progress: bool,
}

#[pymethods]
impl PyTraversal {
    #[new]
    #[pyo3(signature = (kernel, start_nodes, max_depth = None, max_paths = None, strategy = "dfs", parallel = true, intermediate_states = false, progress = false))]
    // Internal and mirrors the Python keyword API, so OK to have a lot of variables
    #[allow(clippy::too_many_arguments)]
    fn new(
        kernel: &PyKernel,
        start_nodes: Vec<PyGraphId>,
        max_depth: Option<usize>,
        max_paths: Option<usize>,
        strategy: &str,
        parallel: bool,
        intermediate_states: bool,
        progress: bool,
    ) -> PyResult<Self> {
        let strategy = match strategy {
            "dfs" => TraversalStrategy::DepthFirst,
            "bfs" => TraversalStrategy::BreadthFirst,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown traversal strategy {other:?}; expected 'dfs' or 'bfs'"
                )));
            }
        };

        Ok(Self {
            kernel: kernel.inner.clone(),
            start_nodes: start_nodes.into_iter().map(|id| id.0).collect(),
            max_depth,
            max_paths,
            strategy,
            parallel,
            intermediate_states,
            progress,
        })
    }
}

#[pyclass(name = "SearchResult")]
struct PySearchResult {
    #[pyo3(get)]
    paths: Vec<PySearchPath>,
    #[pyo3(get)]
    stats: PySearchStats,
}

impl PySearchResult {
    fn from_result(py: Python<'_>, result: SearchResult<'_>) -> PyResult<Self> {
        Ok(Self {
            paths: result
                .paths
                .into_iter()
                .map(|path| PySearchPath::from_path(py, path))
                .collect::<PyResult<_>>()?,
            stats: result.stats.into(),
        })
    }
}

#[pyclass(name = "SearchStats", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PySearchStats {
    #[pyo3(get)]
    start_nodes: usize,
    #[pyo3(get)]
    path_entries: usize,
    #[pyo3(get)]
    evaluated_edges: usize,
    #[pyo3(get)]
    accepted_edges: usize,
    #[pyo3(get)]
    rejected_edges: usize,
    #[pyo3(get)]
    skipped_revisits: usize,
    #[pyo3(get)]
    stopped_paths: usize,
    #[pyo3(get)]
    max_depth: usize,
}

impl From<SearchStats> for PySearchStats {
    fn from(stats: SearchStats) -> Self {
        Self {
            start_nodes: stats.start_nodes,
            path_entries: stats.path_entries,
            evaluated_edges: stats.evaluated_edges,
            accepted_edges: stats.accepted_edges,
            rejected_edges: stats.rejected_edges,
            skipped_revisits: stats.skipped_revisits,
            stopped_paths: stats.stopped_paths,
            max_depth: stats.max_depth,
        }
    }
}

#[pyclass(name = "SearchPath", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PySearchPath {
    nodes: Vec<OwnedGraphId>,
    edges: Vec<OwnedGraphId>,
    state: StateRow,
    intermediate_states: Option<Vec<StateRow>>,
}

impl PySearchPath {
    fn from_path(_py: Python<'_>, path: rxgraph::GraphPath<'_>) -> PyResult<Self> {
        Ok(Self {
            nodes: path.nodes.into_iter().map(GraphId::into_owned).collect(),
            edges: path.edges.into_iter().map(GraphId::into_owned).collect(),
            state: path.state,
            intermediate_states: path.intermediate_states,
        })
    }
}

#[pymethods]
impl PySearchPath {
    #[getter]
    fn nodes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        owned_ids_to_py(py, &self.nodes)
    }

    #[getter]
    fn edges(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        owned_ids_to_py(py, &self.edges)
    }

    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        state_to_py(py, self.state.clone())
    }

    #[getter]
    fn intermediate_states(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.intermediate_states
            .as_ref()
            .map(|states| states_to_py(py, states.clone()))
            .transpose()
    }
}

struct PyGraphId(OwnedGraphId);

impl FromPyObject<'_, '_> for PyGraphId {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<u64>() {
            return Ok(Self(OwnedGraphId::U64(value)));
        }
        if let Ok(value) = obj.extract::<String>() {
            return Ok(Self(OwnedGraphId::Str(value)));
        }
        Err(PyTypeError::new_err("graph IDs must be int or str"))
    }
}

fn ids_to_py(py: Python<'_>, ids: Vec<GraphId<'_>>) -> PyResult<Py<PyAny>> {
    if ids.iter().all(|id| matches!(id, GraphId::U64(_))) {
        ids.into_iter()
            .map(|id| match id {
                GraphId::U64(value) => value,
                GraphId::Str(_) => unreachable!(),
            })
            .collect::<Vec<_>>()
            .into_py_any(py)
    } else {
        ids.into_iter()
            .map(|id| match id {
                GraphId::Str(value) => value.to_owned(),
                GraphId::U64(_) => unreachable!(),
            })
            .collect::<Vec<_>>()
            .into_py_any(py)
    }
}

fn owned_ids_to_py(py: Python<'_>, ids: &[OwnedGraphId]) -> PyResult<Py<PyAny>> {
    if ids.iter().all(|id| matches!(id, OwnedGraphId::U64(_))) {
        ids.iter()
            .map(|id| match id {
                OwnedGraphId::U64(value) => *value,
                OwnedGraphId::Str(_) => unreachable!(),
            })
            .collect::<Vec<_>>()
            .into_py_any(py)
    } else {
        ids.iter()
            .map(|id| match id {
                OwnedGraphId::Str(value) => value.clone(),
                OwnedGraphId::U64(_) => unreachable!(),
            })
            .collect::<Vec<_>>()
            .into_py_any(py)
    }
}

fn states_to_py(py: Python<'_>, states: Vec<StateRow>) -> PyResult<Py<PyAny>> {
    let values = states
        .into_iter()
        .map(|state| state_to_py(py, state))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, values)?.into_any().unbind())
}

fn state_to_py(py: Python<'_>, state: StateRow) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    for (name, value) in state {
        dict.set_item(name, value_to_py(py, value)?)?;
    }
    Ok(dict.into_any().unbind())
}

fn value_to_py(py: Python<'_>, value: Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(value) => value.into_py_any(py),
        Value::I64(value) => value.into_py_any(py),
        Value::U64(value) => value.into_py_any(py),
        Value::F64(value) => value.into_py_any(py),
        Value::Str(value) => value.to_string().into_py_any(py),
        Value::List(values) => {
            let values = values
                .into_iter()
                .map(|value| value_to_py(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, values)?.into_any().unbind())
        }
        Value::Struct(fields) => {
            let dict = PyDict::new(py);
            for (name, value) in fields {
                dict.set_item(name, value_to_py(py, value)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

fn components_to_py(py: Python<'_>, components: Vec<Vec<GraphId<'_>>>) -> PyResult<Py<PyAny>> {
    let u64_mode = components
        .iter()
        .flat_map(|component| component.iter())
        .all(|id| matches!(id, GraphId::U64(_)));

    if u64_mode {
        components
            .into_iter()
            .map(|component| {
                component
                    .into_iter()
                    .map(|id| match id {
                        GraphId::U64(value) => value,
                        GraphId::Str(_) => unreachable!(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .into_py_any(py)
    } else {
        components
            .into_iter()
            .map(|component| {
                component
                    .into_iter()
                    .map(|id| match id {
                        GraphId::Str(value) => value.to_owned(),
                        GraphId::U64(_) => unreachable!(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .into_py_any(py)
    }
}

fn py_to_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(value) = value.extract::<bool>() {
        return Ok(Value::Bool(value));
    }
    if let Ok(value) = value.extract::<u64>() {
        return Ok(Value::U64(value));
    }
    if let Ok(value) = value.extract::<i64>() {
        return Ok(Value::I64(value));
    }
    if let Ok(value) = value.extract::<f64>() {
        return Ok(Value::F64(value));
    }
    if let Ok(value) = value.cast::<PyString>() {
        return Ok(Value::Str(std::sync::Arc::from(value.to_str()?)));
    }
    if let Ok(values) = value.cast::<PyList>() {
        return values
            .iter()
            .map(|value| py_to_value(&value))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::List);
    }
    if let Ok(fields) = value.cast::<PyDict>() {
        return fields
            .iter()
            .map(|(key, value)| {
                let key = key
                    .cast::<PyString>()
                    .map_err(|_| PyTypeError::new_err("struct keys must be strings"))?
                    .to_str()?
                    .to_string();
                Ok((key, py_to_value(&value)?))
            })
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Struct);
    }

    Err(PyTypeError::new_err(format!(
        "cannot convert {} to DSL value",
        value.get_type().name()?
    )))
}

fn to_py_value_err(err: anyhow::Error) -> PyErr {
    PyValueError::new_err(format!("{err:#}"))
}

fn to_py_runtime_err(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{err:#}"))
}
