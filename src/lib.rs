//! Graph traversal over Arrow-backed node and edge tables.
//!
//! `rxgraph` stores graph topology in compact internal IDs while keeping all
//! node and edge attributes in Arrow arrays. Traversals are driven by a small
//! expression kernel, usually serialized from Polars expressions on the Python
//! side.

mod dsl;
mod graph;
mod traversal;

#[cfg(test)]
mod test_utils;

pub use dsl::{DslKernel, Scalar};
pub use graph::{EdgeId, Graph, GraphBuilder, NodeId};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyDict, PyString},
};
use pyo3_arrow::PyTable;
pub use traversal::{
    DslTraversal, DslTraversalBuilder, Parallelism, SearchPath, SearchResult, SearchStats,
    TraversalStrategy,
};

#[pymodule]
fn rxgraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyKernel>()?;
    m.add_class::<PyTraversal>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PySearchStats>()?;
    m.add_class::<PySearchPath>()?;
    Ok(())
}

#[pyclass(name = "Graph", unsendable)]
struct PyGraph {
    inner: Graph,
}

#[pymethods]
impl PyGraph {
    #[new]
    #[pyo3(signature = (node_tables, edge_tables))]
    fn new(
        node_tables: Vec<(String, PyTable)>,
        edge_tables: Vec<(String, PyTable)>,
    ) -> PyResult<Self> {
        let mut builder = GraphBuilder::new();

        for (typ, table) in node_tables {
            let (batches, _) = table.into_inner();
            for batch in batches {
                builder = builder.with_node_table(typ.clone(), batch);
            }
        }
        for (typ, table) in edge_tables {
            let (batches, _) = table.into_inner();
            for batch in batches {
                builder = builder.with_edge_table(typ.clone(), batch);
            }
        }

        Ok(Self {
            inner: builder.build().map_err(to_py_value_err)?,
        })
    }

    fn search(&self, traversal: &PyTraversal) -> PyResult<PySearchResult> {
        let traversal = DslTraversalBuilder::new(traversal.kernel.clone())
            .with_start_nodes(traversal.start_nodes.clone())
            .with_max_depth(traversal.max_depth)
            .with_max_paths(traversal.max_paths)
            .with_strategy(traversal.strategy)
            .with_parallelism(traversal.parallelism)
            .build();
        let result = self.inner.search(traversal).map_err(to_py_runtime_err)?;

        Ok(result.into())
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    #[getter]
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }
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
                Ok((key, py_to_scalar(&value)?))
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            inner: DslKernel::new(
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
    start_nodes: Vec<u64>,
    max_depth: usize,
    max_paths: usize,
    strategy: TraversalStrategy,
    parallelism: Parallelism,
}

#[pymethods]
impl PyTraversal {
    #[new]
    #[pyo3(signature = (
        kernel,
        start_nodes,
        max_depth,
        max_paths,
        strategy = "dfs",
        parallel = "auto",
        parallel_min_frontier = 512,
        parallel_min_edges = 8192,
    ))]
    fn new(
        kernel: &PyKernel,
        start_nodes: Vec<u64>,
        max_depth: usize,
        max_paths: usize,
        strategy: &str,
        parallel: &str,
        parallel_min_frontier: usize,
        parallel_min_edges: usize,
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
        let parallelism = parse_parallelism(parallel, parallel_min_frontier, parallel_min_edges)?;

        Ok(Self {
            kernel: kernel.inner.clone(),
            start_nodes,
            max_depth,
            max_paths,
            strategy,
            parallelism,
        })
    }
}

fn parse_parallelism(
    parallel: &str,
    min_frontier: usize,
    min_edges: usize,
) -> PyResult<Parallelism> {
    match parallel {
        "auto" => Ok(Parallelism::Auto),
        "off" => Ok(Parallelism::Disabled),
        "on" => Ok(Parallelism::Enabled {
            min_frontier,
            min_edges,
        }),
        other => Err(PyValueError::new_err(format!(
            "unknown parallel mode {other:?}; expected 'auto', 'off', or 'on'"
        ))),
    }
}

#[pyclass(name = "SearchResult")]
struct PySearchResult {
    #[pyo3(get)]
    paths: Vec<PySearchPath>,
    #[pyo3(get)]
    stats: PySearchStats,
}

impl From<SearchResult> for PySearchResult {
    fn from(result: SearchResult) -> Self {
        Self {
            paths: result.paths.into_iter().map(Into::into).collect(),
            stats: result.stats.into(),
        }
    }
}

#[pyclass(name = "SearchStats", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PySearchStats {
    #[pyo3(get)]
    visited_path_entries: usize,
    #[pyo3(get)]
    evaluated_edges: usize,
    #[pyo3(get)]
    accepted_edges: usize,
    #[pyo3(get)]
    stopped_paths: usize,
    #[pyo3(get)]
    skipped_errors: usize,
    #[pyo3(get)]
    max_depth: usize,
}

impl From<SearchStats> for PySearchStats {
    fn from(stats: SearchStats) -> Self {
        Self {
            visited_path_entries: stats.visited_path_entries,
            evaluated_edges: stats.evaluated_edges,
            accepted_edges: stats.accepted_edges,
            stopped_paths: stats.stopped_paths,
            skipped_errors: stats.skipped_errors,
            max_depth: stats.max_depth,
        }
    }
}

#[pyclass(name = "SearchPath", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PySearchPath {
    #[pyo3(get)]
    nodes: Vec<u64>,
    #[pyo3(get)]
    edges: Vec<EdgeId>,
    #[pyo3(get)]
    state: String,
}

impl From<SearchPath> for PySearchPath {
    fn from(path: SearchPath) -> Self {
        Self {
            nodes: path.nodes,
            edges: path.edges,
            state: path.state.to_string(),
        }
    }
}

fn py_to_scalar(value: &Bound<'_, PyAny>) -> PyResult<Scalar> {
    if value.is_none() {
        return Ok(Scalar::Null);
    }
    if let Ok(value) = value.extract::<bool>() {
        return Ok(Scalar::Bool(value));
    }
    if let Ok(value) = value.extract::<u64>() {
        return Ok(Scalar::U64(value));
    }
    if let Ok(value) = value.extract::<i64>() {
        return Ok(Scalar::I64(value));
    }
    if let Ok(value) = value.extract::<f64>() {
        return Ok(Scalar::F64(value));
    }
    if let Ok(value) = value.cast::<PyString>() {
        return Ok(Scalar::Str(std::sync::Arc::from(value.to_str()?)));
    }

    Err(PyTypeError::new_err(format!(
        "cannot convert {} to DSL scalar",
        value.get_type().name()?
    )))
}

fn to_py_value_err(err: anyhow::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn to_py_runtime_err(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}
