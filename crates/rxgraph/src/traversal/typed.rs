//! Typed native traversal support.
//!
//! This layer keeps the user-facing native payload contract small:
//! implement `TryFrom<ArrowRow<'_>, Error = anyhow::Error>` for node/edge structs, then implement
//! [`TypedKernel`] over those structs. Stores may batch/projection-read however
//! they want before calling `TryFrom` per row.

use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use anyhow::{Context, Result, anyhow, bail};
use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, Float64Array, Int64Array, LargeStringArray, RecordBatch,
        StringArray, StringViewArray, UInt64Array,
    },
    datatypes::{DataType, Field, FieldRef, Schema},
};
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ParquetRecordBatchReaderBuilder, RowSelection,
};

use crate::{
    dsl::{StateRow, Value, arrow_value},
    graph::{EdgeId, Graph, GraphId, GraphRepo, NodeId, OwnedGraphId},
    traversal::{RunOptions, SearchStats, native},
};

/// A projected Arrow row handed to `TryFrom<ArrowRow<'_>>`.
pub struct ArrowRow<'a> {
    batch: &'a RecordBatch,
    row: usize,
}

impl<'a> ArrowRow<'a> {
    /// Builds a row view over `batch[row]`.
    pub fn new(batch: &'a RecordBatch, row: usize) -> Self {
        Self { batch, row }
    }

    /// Reads `col` as `u64`, accepting lossless integer coercions.
    pub fn u64(&self, col: &str) -> Result<Option<u64>> {
        let array = self.column(col)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::UInt64 => Ok(Some(
                array
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .context("UInt64 column downcast failed")?
                    .value(self.row),
            )),
            DataType::Int64 => {
                let value = array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .context("Int64 column downcast failed")?
                    .value(self.row);
                if value < 0 {
                    bail!("cannot read negative value {value} as u64");
                }
                Ok(Some(value as u64))
            }
            other => bail!("column {col:?} must be UInt64/Int64, got {other:?}"),
        }
    }

    /// Reads `col` as `i64`, accepting lossless integer coercions.
    pub fn i64(&self, col: &str) -> Result<Option<i64>> {
        let array = self.column(col)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::Int64 => Ok(Some(
                array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .context("Int64 column downcast failed")?
                    .value(self.row),
            )),
            DataType::UInt64 => {
                let value = array
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .context("UInt64 column downcast failed")?
                    .value(self.row);
                if value > i64::MAX as u64 {
                    bail!("cannot read u64 {value} as i64");
                }
                Ok(Some(value as i64))
            }
            other => bail!("column {col:?} must be Int64/UInt64, got {other:?}"),
        }
    }

    /// Reads `col` as `f64`.
    pub fn f64(&self, col: &str) -> Result<Option<f64>> {
        let array = self.column(col)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::Float64 => Ok(Some(
                array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .context("Float64 column downcast failed")?
                    .value(self.row),
            )),
            DataType::Int64 => Ok(Some(
                array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .context("Int64 column downcast failed")?
                    .value(self.row) as f64,
            )),
            DataType::UInt64 => Ok(Some(
                array
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .context("UInt64 column downcast failed")?
                    .value(self.row) as f64,
            )),
            other => bail!("column {col:?} must be numeric, got {other:?}"),
        }
    }

    /// Reads `col` as `bool`.
    pub fn bool(&self, col: &str) -> Result<Option<bool>> {
        let array = self.column(col)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::Boolean => Ok(Some(
                array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .context("Boolean column downcast failed")?
                    .value(self.row),
            )),
            other => bail!("column {col:?} must be Boolean, got {other:?}"),
        }
    }

    /// Reads `col` as an owned string.
    pub fn string(&self, col: &str) -> Result<Option<String>> {
        let array = self.column(col)?;
        if array.is_null(self.row) {
            return Ok(None);
        }
        match array.data_type() {
            DataType::Utf8 => Ok(Some(
                array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Utf8 column downcast failed")?
                    .value(self.row)
                    .to_string(),
            )),
            DataType::LargeUtf8 => Ok(Some(
                array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .context("LargeUtf8 column downcast failed")?
                    .value(self.row)
                    .to_string(),
            )),
            DataType::Utf8View => Ok(Some(
                array
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .context("Utf8View column downcast failed")?
                    .value(self.row)
                    .to_string(),
            )),
            other => bail!("column {col:?} must be Utf8, got {other:?}"),
        }
    }

    /// Reads `col` as a DSL value, including nested list and struct values.
    pub fn value(&self, col: &str) -> Result<Value> {
        let array = self.column(col)?;
        arrow_value::array_row_to_value(array.as_ref(), self.row)
    }

    /// Reads `col` as a list value.
    pub fn list(&self, col: &str) -> Result<Option<Vec<Value>>> {
        self.value(col)?.into_list()
    }

    /// Reads `col` as a struct value, preserving Arrow field order.
    pub fn struct_fields(&self, col: &str) -> Result<Option<Vec<(String, Value)>>> {
        self.value(col)?.into_struct()
    }

    fn column(&self, col: &str) -> Result<&ArrayRef> {
        let index = self
            .batch
            .schema()
            .index_of(col)
            .with_context(|| format!("missing payload column {col:?}"))?;
        Ok(self.batch.column(index))
    }
}

impl TryFrom<ArrowRow<'_>> for () {
    type Error = anyhow::Error;

    fn try_from(_row: ArrowRow<'_>) -> Result<Self> {
        Ok(())
    }
}

/// One source payload column and the name exposed to `TryFrom<ArrowRow<'_>>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PayloadField {
    pub source: String,
    pub alias: String,
}

impl PayloadField {
    pub fn new(source: impl Into<String>) -> Self {
        let source = source.into();
        Self {
            alias: source.clone(),
            source,
        }
    }

    pub fn aliased(source: impl Into<String>, alias: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            alias: alias.into(),
        }
    }
}

/// Native Rust traversal kernel over typed node and edge payloads.
pub trait TypedKernel: Clone {
    type Node: for<'row> TryFrom<ArrowRow<'row>, Error = anyhow::Error>;
    type Edge: for<'row> TryFrom<ArrowRow<'row>, Error = anyhow::Error>;
    type State: Clone;

    fn node_fields(&self) -> Vec<PayloadField> {
        Vec::new()
    }

    fn edge_fields(&self) -> Vec<PayloadField> {
        Vec::new()
    }

    fn initial_state(
        &self,
        cx: &native::StartCtx<'_, Self::Node, Self::Edge>,
    ) -> Result<Self::State>;

    fn visit(
        &self,
        cx: &native::EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<bool>;

    fn next_state(
        &self,
        cx: &native::EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<Self::State>;

    fn stop(
        &self,
        cx: &native::EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<bool>;

    fn state_row(&self, state: &Self::State) -> StateRow;
}

#[derive(Clone)]
pub(crate) struct TypedKernelAdapter<K> {
    kernel: K,
}

impl<K> TypedKernelAdapter<K> {
    pub(crate) fn new(kernel: K) -> Self {
        Self { kernel }
    }
}

impl<K> native::Kernel for TypedKernelAdapter<K>
where
    K: TypedKernel,
{
    type Node = K::Node;
    type Edge = K::Edge;
    type State = K::State;

    fn initial_state(
        &self,
        cx: &native::StartCtx<'_, Self::Node, Self::Edge>,
    ) -> Result<Self::State> {
        self.kernel.initial_state(cx)
    }

    fn visit(
        &self,
        cx: &native::EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<bool> {
        self.kernel.visit(cx)
    }

    fn next_state(
        &self,
        cx: &native::EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<Self::State> {
        self.kernel.next_state(cx)
    }

    fn stop(
        &self,
        cx: &native::EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<bool> {
        self.kernel.stop(cx)
    }

    fn state_row(&self, state: &Self::State) -> StateRow {
        self.kernel.state_row(state)
    }
}

/// Owned path result used by file-backed native kernels.
#[derive(Debug)]
pub struct OwnedSearchResult {
    pub paths: Vec<OwnedGraphPath>,
    pub stats: SearchStats,
}

#[derive(Debug)]
pub struct OwnedGraphPath {
    pub nodes: Vec<OwnedGraphId>,
    pub edges: Vec<OwnedGraphId>,
    pub state: StateRow,
    pub intermediate_states: Option<Vec<StateRow>>,
}

#[derive(Default)]
pub struct TypedPayloadCache {
    entries: RefCell<HashMap<TypedCacheKey, Box<dyn Any>>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TypedCacheKey {
    node_type: TypeId,
    edge_type: TypeId,
    node_fields: Vec<PayloadField>,
    edge_fields: Vec<PayloadField>,
}

struct DecodedPayloads<N, E> {
    nodes: Vec<N>,
    edges: Vec<E>,
}

impl TypedPayloadCache {
    fn payloads<K>(
        &self,
        graph: &Graph,
        kernel: &K,
    ) -> Result<Arc<DecodedPayloads<K::Node, K::Edge>>>
    where
        K: TypedKernel + 'static,
        K::Node: 'static,
        K::Edge: 'static,
    {
        let node_fields = kernel.node_fields();
        let edge_fields = kernel.edge_fields();
        let key = TypedCacheKey {
            node_type: TypeId::of::<K::Node>(),
            edge_type: TypeId::of::<K::Edge>(),
            node_fields,
            edge_fields,
        };

        if let Some(existing) = self.entries.borrow().get(&key) {
            let payloads = existing
                .downcast_ref::<Arc<DecodedPayloads<K::Node, K::Edge>>>()
                .context("typed payload cache stored an unexpected payload type")?;
            return Ok(Arc::clone(payloads));
        }

        let node_batch = project_batch(graph.node_payloads(), &key.node_fields)?;
        let edge_batch = project_batch(graph.edge_payloads(), &key.edge_fields)?;
        let payloads = Arc::new(DecodedPayloads {
            nodes: decode_all::<K::Node>(&node_batch)?,
            edges: decode_all::<K::Edge>(&edge_batch)?,
        });
        self.entries
            .borrow_mut()
            .insert(key, Box::new(Arc::clone(&payloads)));
        Ok(payloads)
    }
}

pub(crate) fn run_typed_eager<K>(
    graph: &Graph,
    kernel: K,
    run: RunOptions,
) -> Result<OwnedSearchResult>
where
    K: TypedKernel,
{
    let node_batch = project_batch(graph.node_payloads(), &kernel.node_fields())?;
    let edge_batch = project_batch(graph.edge_payloads(), &kernel.edge_fields())?;
    let nodes = decode_all::<K::Node>(&node_batch)?;
    let edges = decode_all::<K::Edge>(&edge_batch)?;
    run_typed_payloads(graph, kernel, &nodes, &edges, run)
}

pub(crate) fn run_typed_eager_cached<K>(
    graph: &Graph,
    cache: &TypedPayloadCache,
    kernel: K,
    run: RunOptions,
) -> Result<OwnedSearchResult>
where
    K: TypedKernel + 'static,
    K::Node: 'static,
    K::Edge: 'static,
{
    let payloads = cache.payloads(graph, &kernel)?;
    run_typed_payloads(graph, kernel, &payloads.nodes, &payloads.edges, run)
}

fn run_typed_payloads<K>(
    graph: &Graph,
    kernel: K,
    nodes: &[K::Node],
    edges: &[K::Edge],
    run: RunOptions,
) -> Result<OwnedSearchResult>
where
    K: TypedKernel,
{
    let store = native::EagerGraphStore::new(graph, nodes, edges)?;
    let adapter = TypedKernelAdapter::new(kernel);
    let result = native::search_native(&store, adapter.clone(), run)?;
    owned_result(result, &adapter)
}

pub(crate) fn run_typed_parquet_lazy<K>(
    graph: &Graph,
    paths: ParquetPaths,
    kernel: K,
    run: RunOptions,
) -> Result<OwnedSearchResult>
where
    K: TypedKernel,
{
    let store = ParquetGraphStore::<K::Node, K::Edge>::new(
        graph,
        paths,
        kernel.node_fields(),
        kernel.edge_fields(),
    )?;
    let adapter = TypedKernelAdapter::new(kernel);
    let result = native::search_native(&store, adapter.clone(), run)?;
    let mut owned = owned_result(result, &adapter)?;
    owned.stats.materialized_node_payloads = store.loaded_node_count();
    owned.stats.materialized_edge_payloads = store.loaded_edge_count();
    let io = store.io_stats();
    owned.stats.lazy_payload_read_calls = io.read_calls;
    owned.stats.lazy_payload_requested_rows = io.requested_rows;
    owned.stats.lazy_payload_selected_rows = io.selected_rows;
    owned.stats.lazy_payload_row_groups = io.row_groups;
    Ok(owned)
}

pub(crate) fn owned_result<N, E, S, K>(
    result: native::SearchResult<'_, N, E, S>,
    kernel: &K,
) -> Result<OwnedSearchResult>
where
    K: native::Kernel<Node = N, Edge = E, State = S> + ?Sized,
    S: Clone,
{
    let paths = result
        .paths
        .into_iter()
        .map(|path| {
            let final_state = kernel.state_row(&path.state);
            let intermediate_states = path
                .nodes
                .iter()
                .any(|node| node.state.is_some())
                .then(|| {
                    path.nodes
                        .iter()
                        .map(|node| {
                            node.state
                                .as_ref()
                                .map(|state| kernel.state_row(state))
                                .context("missing intermediate state")
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?;
            let nodes = path
                .nodes
                .into_iter()
                .map(|node| {
                    node.external_id
                        .context("path references missing node")?
                        .into_owned()
                        .pipe(Ok)
                })
                .collect::<Result<Vec<_>>>()?;
            let edges = path
                .edges
                .into_iter()
                .map(|edge| {
                    edge.external_id
                        .context("path references missing edge")?
                        .into_owned()
                        .pipe(Ok)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(OwnedGraphPath {
                nodes,
                edges,
                state: final_state,
                intermediate_states,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(OwnedSearchResult {
        paths,
        stats: result.stats,
    })
}

trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}

impl<T> Pipe for T {}

#[derive(Clone, Debug)]
pub struct ParquetPaths {
    pub nodes: PathBuf,
    pub edges: PathBuf,
}

pub(crate) fn read_parquet_topology(paths: &ParquetPaths) -> Result<(RecordBatch, RecordBatch)> {
    let nodes = read_projected_parquet(&paths.nodes, &[PayloadField::new("id")])?;
    let edges = read_projected_parquet(
        &paths.edges,
        &[
            PayloadField::new("id"),
            PayloadField::aliased(edge_src_col(&paths.edges)?, "src"),
            PayloadField::aliased(edge_dest_col(&paths.edges)?, "dest"),
        ],
    )?;
    Ok((nodes, edges))
}

pub(crate) fn read_parquet_tables(paths: &ParquetPaths) -> Result<(RecordBatch, RecordBatch)> {
    let nodes = read_projected_parquet(&paths.nodes, &all_parquet_fields(&paths.nodes)?)?;
    let edges = read_projected_parquet(&paths.edges, &edge_table_fields(&paths.edges)?)?;
    Ok((nodes, edges))
}

fn all_parquet_fields(path: &Path) -> Result<Vec<PayloadField>> {
    Ok(parquet_schema(path)?
        .fields()
        .iter()
        .map(|field| PayloadField::new(field.name()))
        .collect())
}

fn edge_table_fields(path: &Path) -> Result<Vec<PayloadField>> {
    let schema = parquet_schema(path)?;
    let src = edge_src_col_from_schema(&schema)?;
    let dest = edge_dest_col_from_schema(&schema)?;
    let mut fields = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let name = field.name().as_str();
        if name == src {
            fields.push(PayloadField::aliased(name, "src"));
        } else if name == dest {
            fields.push(PayloadField::aliased(name, "dest"));
        } else if matches!(name, "src" | "src_id" | "dest" | "dest_id") {
            continue;
        } else {
            fields.push(PayloadField::new(name));
        }
    }
    Ok(fields)
}

fn edge_src_col(path: &Path) -> Result<&'static str> {
    let schema = parquet_schema(path)?;
    edge_src_col_from_schema(&schema)
}

fn edge_src_col_from_schema(schema: &Schema) -> Result<&'static str> {
    if schema.index_of("src_id").is_ok() {
        Ok("src_id")
    } else if schema.index_of("src").is_ok() {
        Ok("src")
    } else {
        bail!("edges parquet must contain 'src_id' or 'src'")
    }
}

fn edge_dest_col(path: &Path) -> Result<&'static str> {
    let schema = parquet_schema(path)?;
    edge_dest_col_from_schema(&schema)
}

fn edge_dest_col_from_schema(schema: &Schema) -> Result<&'static str> {
    if schema.index_of("dest_id").is_ok() {
        Ok("dest_id")
    } else if schema.index_of("dest").is_ok() {
        Ok("dest")
    } else {
        bail!("edges parquet must contain 'dest_id' or 'dest'")
    }
}

fn parquet_schema(path: &Path) -> Result<std::sync::Arc<Schema>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("failed to read parquet metadata from {}", path.display()))?;
    Ok(builder.schema().clone())
}

fn decode_all<T>(batch: &RecordBatch) -> Result<Vec<T>>
where
    T: for<'row> TryFrom<ArrowRow<'row>, Error = anyhow::Error>,
{
    (0..batch.num_rows())
        .map(|row| T::try_from(ArrowRow::new(batch, row)))
        .collect()
}

fn project_batch(batch: &RecordBatch, fields: &[PayloadField]) -> Result<RecordBatch> {
    if fields.is_empty() {
        return empty_batch(batch.num_rows());
    }
    let mut out_fields = Vec::with_capacity(fields.len());
    let mut columns = Vec::with_capacity(fields.len());
    for field in fields {
        let index = batch
            .schema()
            .index_of(&field.source)
            .with_context(|| format!("missing payload column {:?}", field.source))?;
        let schema = batch.schema();
        let source = schema.field(index);
        out_fields.push(Field::new(
            field.alias.clone(),
            source.data_type().clone(),
            source.is_nullable(),
        ));
        columns.push(batch.column(index).clone());
    }
    RecordBatch::try_new(std::sync::Arc::new(Schema::new(out_fields)), columns)
        .context("failed to project payload batch")
}

fn empty_batch(rows: usize) -> Result<RecordBatch> {
    RecordBatch::try_new_with_options(
        std::sync::Arc::new(Schema::empty()),
        Vec::new(),
        &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .context("failed to build empty payload batch")
}

fn read_projected_parquet(path: &Path, fields: &[PayloadField]) -> Result<RecordBatch> {
    read_projected_parquet_range(path, fields, None, None)
}

fn read_selected_parquet_with_builder(
    path: &Path,
    builder: ParquetRecordBatchReaderBuilder<File>,
    fields: &[PayloadField],
    rows: &[usize],
    total_rows: usize,
) -> Result<RecordBatch> {
    if rows.is_empty() {
        return empty_batch(0);
    }

    if let Some(&row) = rows.last()
        && row >= total_rows
    {
        bail!("row {row} is out of range for {}", path.display());
    }
    if fields.is_empty() {
        return empty_batch(rows.len());
    }

    let schema = builder.schema().clone();
    let indexes = fields
        .iter()
        .map(|field| {
            schema
                .index_of(&field.source)
                .with_context(|| format!("missing parquet column {:?}", field.source))
        })
        .collect::<Result<Vec<_>>>()?;
    let projection = parquet::arrow::ProjectionMask::leaves(builder.parquet_schema(), indexes);
    let ranges = rows.iter().copied().map(|row| row..row + 1);
    let reader = builder
        .with_projection(projection)
        .with_row_selection(RowSelection::from_consecutive_ranges(ranges, total_rows))
        .build()
        .with_context(|| format!("failed to build parquet reader for {}", path.display()))?;

    let batches = reader.collect::<std::result::Result<Vec<_>, _>>()?;
    let batch = concat_batches(&batches)?;
    if batch.num_rows() != rows.len() {
        bail!(
            "selected {} rows from {} but parquet reader returned {}",
            rows.len(),
            path.display(),
            batch.num_rows()
        );
    }
    alias_batch(batch, fields)
}

fn read_projected_parquet_range(
    path: &Path,
    fields: &[PayloadField],
    offset: Option<usize>,
    limit: Option<usize>,
) -> Result<RecordBatch> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("failed to read parquet metadata from {}", path.display()))?;
    if fields.is_empty() {
        return empty_batch(
            limit.unwrap_or(builder.metadata().file_metadata().num_rows() as usize),
        );
    }

    let schema = builder.schema().clone();
    let indexes = fields
        .iter()
        .map(|field| {
            schema
                .index_of(&field.source)
                .with_context(|| format!("missing parquet column {:?}", field.source))
        })
        .collect::<Result<Vec<_>>>()?;
    let projection = parquet::arrow::ProjectionMask::leaves(builder.parquet_schema(), indexes);
    let mut builder = builder.with_projection(projection);
    if let Some(offset) = offset {
        builder = builder.with_offset(offset);
    }
    if let Some(limit) = limit {
        builder = builder.with_limit(limit);
    }
    let reader = builder
        .build()
        .with_context(|| format!("failed to build parquet reader for {}", path.display()))?;

    let batches = reader.collect::<std::result::Result<Vec<_>, _>>()?;
    let batch = concat_batches(&batches)?;
    alias_batch(batch, fields)
}

fn concat_batches(batches: &[RecordBatch]) -> Result<RecordBatch> {
    match batches {
        [] => empty_batch(0),
        [batch] => Ok(batch.clone()),
        _ => {
            let schema = batches[0].schema();
            let columns = (0..schema.fields().len())
                .map(|index| {
                    let arrays = batches
                        .iter()
                        .map(|batch| batch.column(index).as_ref())
                        .collect::<Vec<_>>();
                    arrow::compute::concat(&arrays).context("failed to concatenate parquet batches")
                })
                .collect::<Result<Vec<_>>>()?;
            RecordBatch::try_new(schema.clone(), columns).context("failed to concatenate batches")
        }
    }
}

fn alias_batch(batch: RecordBatch, fields: &[PayloadField]) -> Result<RecordBatch> {
    if fields.is_empty() {
        return Ok(batch);
    }
    let out_fields = batch
        .schema()
        .fields()
        .iter()
        .zip(fields)
        .map(|(source, field)| {
            FieldRef::from(Field::new(
                field.alias.clone(),
                source.data_type().clone(),
                source.is_nullable(),
            ))
        })
        .collect::<Vec<_>>();
    RecordBatch::try_new(Schema::new(out_fields).into(), batch.columns().to_vec())
        .context("failed to alias parquet batch")
}

fn decode_selected<T>(
    batch: RecordBatch,
    rows: impl IntoIterator<Item = usize>,
) -> Result<Vec<(usize, T)>>
where
    T: for<'row> TryFrom<ArrowRow<'row>, Error = anyhow::Error>,
{
    rows.into_iter()
        .enumerate()
        .map(|(offset, row)| {
            T::try_from(ArrowRow::new(&batch, offset)).map(|payload| (row, payload))
        })
        .collect()
}

fn read_many<T>(
    source: &ParquetPayloadFile,
    fields: &[PayloadField],
    rows: impl IntoIterator<Item = usize>,
) -> Result<Vec<(usize, T)>>
where
    T: for<'row> TryFrom<ArrowRow<'row>, Error = anyhow::Error>,
{
    let mut rows = rows.into_iter().collect::<Vec<_>>();
    rows.sort_unstable();
    rows.dedup();
    if rows.is_empty() {
        return Ok(Vec::new());
    }

    let batch = source.read_selected(fields, &rows)?;
    decode_selected(batch, rows)
}

struct ParquetPayloadFile {
    path: PathBuf,
    file: File,
    metadata: ArrowReaderMetadata,
    row_group_offsets: Vec<usize>,
    stats: LazyPayloadIoStats,
}

#[derive(Default)]
struct LazyPayloadIoStats {
    read_calls: AtomicUsize,
    requested_rows: AtomicUsize,
    selected_rows: AtomicUsize,
    row_groups: AtomicUsize,
}

#[derive(Default, Clone, Copy)]
struct LazyPayloadIoSnapshot {
    read_calls: usize,
    requested_rows: usize,
    selected_rows: usize,
    row_groups: usize,
}

impl LazyPayloadIoStats {
    fn record_read(&self, requested_rows: usize, selected_rows: usize, row_groups: usize) {
        self.read_calls.fetch_add(1, Ordering::Relaxed);
        self.requested_rows
            .fetch_add(requested_rows, Ordering::Relaxed);
        self.selected_rows
            .fetch_add(selected_rows, Ordering::Relaxed);
        self.row_groups.fetch_add(row_groups, Ordering::Relaxed);
    }

    fn snapshot(&self) -> LazyPayloadIoSnapshot {
        LazyPayloadIoSnapshot {
            read_calls: self.read_calls.load(Ordering::Relaxed),
            requested_rows: self.requested_rows.load(Ordering::Relaxed),
            selected_rows: self.selected_rows.load(Ordering::Relaxed),
            row_groups: self.row_groups.load(Ordering::Relaxed),
        }
    }
}

impl ParquetPayloadFile {
    fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let file =
            File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
        let metadata = ArrowReaderMetadata::load(&file, Default::default())
            .with_context(|| format!("failed to read parquet metadata from {}", path.display()))?;
        let row_group_offsets = row_group_offsets(&file, &path)?;
        Ok(Self {
            path,
            file,
            metadata,
            row_group_offsets,
            stats: LazyPayloadIoStats::default(),
        })
    }

    fn read_selected(&self, fields: &[PayloadField], rows: &[usize]) -> Result<RecordBatch> {
        let selection = self.selection(rows)?;
        if fields.is_empty() {
            return empty_batch(rows.len());
        }
        self.stats.record_read(
            rows.len(),
            selection.selected_rows,
            selection.row_groups.len(),
        );

        let file = self
            .file
            .try_clone()
            .with_context(|| format!("failed to clone file handle for {}", self.path.display()))?;
        let builder =
            ParquetRecordBatchReaderBuilder::new_with_metadata(file, self.metadata.clone())
                .with_row_groups(selection.row_groups);
        read_selected_parquet_with_builder(
            &self.path,
            builder,
            fields,
            &selection.relative_rows,
            selection.selected_rows,
        )
    }

    fn selection(&self, rows: &[usize]) -> Result<RowGroupSelection> {
        if rows.is_empty() {
            return Ok(RowGroupSelection {
                row_groups: Vec::new(),
                relative_rows: Vec::new(),
                selected_rows: 0,
            });
        }

        let total_rows = self.row_group_offsets.last().copied().unwrap_or(0);
        if let Some(&row) = rows.last()
            && row >= total_rows
        {
            bail!("row {row} is out of range for {}", self.path.display());
        }

        let mut row_groups = Vec::new();
        for &row in rows {
            let group = self.row_group(row);
            if row_groups.last().copied() != Some(group) {
                row_groups.push(group);
            }
        }

        let mut selected_offsets = HashMap::with_capacity(row_groups.len());
        let mut selected_rows = 0usize;
        for &group in &row_groups {
            selected_offsets.insert(group, selected_rows);
            selected_rows += self.row_group_offsets[group + 1] - self.row_group_offsets[group];
        }

        let relative_rows = rows
            .iter()
            .map(|&row| {
                let group = self.row_group(row);
                selected_offsets[&group] + row - self.row_group_offsets[group]
            })
            .collect();

        Ok(RowGroupSelection {
            row_groups,
            relative_rows,
            selected_rows,
        })
    }

    fn row_group(&self, row: usize) -> usize {
        self.row_group_offsets
            .partition_point(|&offset| offset <= row)
            - 1
    }

    fn stats(&self) -> LazyPayloadIoSnapshot {
        self.stats.snapshot()
    }
}

struct RowGroupSelection {
    row_groups: Vec<usize>,
    relative_rows: Vec<usize>,
    selected_rows: usize,
}

fn row_group_offsets(file: &File, path: &Path) -> Result<Vec<usize>> {
    let file = file
        .try_clone()
        .with_context(|| format!("failed to clone file handle for {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("failed to read parquet metadata from {}", path.display()))?;
    let metadata = builder.metadata();
    let mut offsets = Vec::with_capacity(metadata.num_row_groups() + 1);
    offsets.push(0);
    for group in 0..metadata.num_row_groups() {
        let next = offsets[group] + metadata.row_group(group).num_rows() as usize;
        offsets.push(next);
    }
    Ok(offsets)
}

struct ParquetGraphStore<'a, N, E> {
    graph: &'a Graph,
    node_file: ParquetPayloadFile,
    edge_file: ParquetPayloadFile,
    node_fields: Vec<PayloadField>,
    edge_fields: Vec<PayloadField>,
    nodes: Vec<OnceLock<Result<N, String>>>,
    edges: Vec<OnceLock<Result<E, String>>>,
    outgoing: Vec<OnceLock<Vec<native::OutgoingEdge>>>,
}

impl<'a, N, E> ParquetGraphStore<'a, N, E> {
    fn new(
        graph: &'a Graph,
        paths: ParquetPaths,
        node_fields: Vec<PayloadField>,
        edge_fields: Vec<PayloadField>,
    ) -> Result<Self> {
        Ok(Self {
            nodes: (0..graph.node_count()).map(|_| OnceLock::new()).collect(),
            edges: (0..graph.edge_count()).map(|_| OnceLock::new()).collect(),
            outgoing: (0..graph.node_count()).map(|_| OnceLock::new()).collect(),
            graph,
            node_file: ParquetPayloadFile::open(paths.nodes)?,
            edge_file: ParquetPayloadFile::open(paths.edges)?,
            node_fields,
            edge_fields,
        })
    }

    fn loaded_node_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|cell| cell.get().is_some())
            .count()
    }

    fn loaded_edge_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|cell| cell.get().is_some())
            .count()
    }

    fn io_stats(&self) -> LazyPayloadIoSnapshot {
        let node = self.node_file.stats();
        let edge = self.edge_file.stats();
        LazyPayloadIoSnapshot {
            read_calls: node.read_calls + edge.read_calls,
            requested_rows: node.requested_rows + edge.requested_rows,
            selected_rows: node.selected_rows + edge.selected_rows,
            row_groups: node.row_groups + edge.row_groups,
        }
    }
}

impl<N, E> ParquetGraphStore<'_, N, E>
where
    N: for<'row> TryFrom<ArrowRow<'row>, Error = anyhow::Error>,
    E: for<'row> TryFrom<ArrowRow<'row>, Error = anyhow::Error>,
{
    fn missing_node_rows(&self, rows: impl IntoIterator<Item = usize>) -> Result<Vec<usize>> {
        missing_rows(&self.nodes, rows, "node")
    }

    fn missing_edge_rows(&self, rows: impl IntoIterator<Item = usize>) -> Result<Vec<usize>> {
        missing_rows(&self.edges, rows, "edge")
    }

    fn fill_nodes(&self, rows: impl IntoIterator<Item = usize>) -> Result<()> {
        let rows = self.missing_node_rows(rows)?;
        for (row, payload) in read_many::<N>(&self.node_file, &self.node_fields, rows)? {
            let cell = self
                .nodes
                .get(row)
                .with_context(|| format!("node row {row} is out of range"))?;
            let _ = cell.set(Ok(payload));
        }
        Ok(())
    }

    fn fill_edges(&self, rows: impl IntoIterator<Item = usize>) -> Result<()> {
        let rows = self.missing_edge_rows(rows)?;
        for (row, payload) in read_many::<E>(&self.edge_file, &self.edge_fields, rows)? {
            let cell = self
                .edges
                .get(row)
                .with_context(|| format!("edge row {row} is out of range"))?;
            let _ = cell.set(Ok(payload));
        }
        Ok(())
    }
}

fn missing_rows<T>(
    cells: &[OnceLock<Result<T, String>>],
    rows: impl IntoIterator<Item = usize>,
    label: &str,
) -> Result<Vec<usize>> {
    let mut missing = Vec::new();
    for row in rows {
        let cell = cells
            .get(row)
            .with_context(|| format!("{label} row {row} is out of range"))?;
        if cell.get().is_none() {
            missing.push(row);
        }
    }
    missing.sort_unstable();
    missing.dedup();
    Ok(missing)
}

impl<N, E> native::GraphStore for ParquetGraphStore<'_, N, E>
where
    N: for<'row> TryFrom<ArrowRow<'row>, Error = anyhow::Error>,
    E: for<'row> TryFrom<ArrowRow<'row>, Error = anyhow::Error>,
{
    type Node = N;
    type Edge = E;

    fn resolve_node(&self, external: GraphId<'_>) -> Result<Option<NodeId>> {
        Ok(self.graph.repo.internal_node(external))
    }

    fn external_node(&self, internal: NodeId) -> Result<Option<GraphId<'_>>> {
        Ok(self.graph.repo.external_node(internal))
    }

    fn external_edge(&self, internal: EdgeId) -> Result<Option<GraphId<'_>>> {
        Ok(self.graph.repo.external_edge(internal))
    }

    fn outgoing(&self, src: NodeId) -> Result<&[native::OutgoingEdge]> {
        let slot = self
            .outgoing
            .get(src as usize)
            .with_context(|| format!("node row {src} is out of range"))?;
        Ok(slot
            .get_or_init(|| {
                let (edges, dests) = self.graph.repo.outgoing_slice(src);
                edges
                    .iter()
                    .zip(dests)
                    .map(|(&edge, &dest)| native::OutgoingEdge { edge, dest })
                    .collect()
            })
            .as_slice())
    }

    fn prefetch_outgoing(&self, nodes: &[NodeId]) -> Result<()> {
        let mut node_rows = Vec::new();
        let mut edge_rows = Vec::new();
        for &node in nodes {
            if !self.node_fields.is_empty() {
                node_rows.push(node as usize);
            }
            for &native::OutgoingEdge { edge, dest } in self.outgoing(node)? {
                if !self.edge_fields.is_empty() {
                    edge_rows.push(edge as usize);
                }
                if !self.node_fields.is_empty() {
                    node_rows.push(dest as usize);
                }
            }
        }
        self.fill_nodes(node_rows)?;
        self.fill_edges(edge_rows)?;
        Ok(())
    }

    fn node(&self, id: NodeId) -> Result<&Self::Node> {
        let row = id as usize;
        if self
            .nodes
            .get(row)
            .with_context(|| format!("node row {id} is out of range"))?
            .get()
            .is_none()
        {
            self.fill_nodes([row])?;
        }
        let value = self
            .nodes
            .get(row)
            .and_then(OnceLock::get)
            .with_context(|| format!("node row {id} was not materialized"))?;
        value
            .as_ref()
            .map_err(|err| anyhow!("failed to materialize node row {id}: {err}"))
    }

    fn edge(&self, id: EdgeId) -> Result<&Self::Edge> {
        let row = id as usize;
        if self
            .edges
            .get(row)
            .with_context(|| format!("edge row {id} is out of range"))?
            .get()
            .is_none()
        {
            self.fill_edges([row])?;
        }
        let value = self
            .edges
            .get(row)
            .and_then(OnceLock::get)
            .with_context(|| format!("edge row {id} was not materialized"))?;
        value
            .as_ref()
            .map_err(|err| anyhow!("failed to materialize edge row {id}: {err}"))
    }
}
