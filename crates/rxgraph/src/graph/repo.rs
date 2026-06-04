use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::{Arc, OnceLock},
};

use anyhow::{Context, Result, bail};
use arrow::array::{
    Array, LargeStringArray, RecordBatch, StringArray, StringViewArray, UInt64Array,
};
use arrow_schema::{DataType, FieldRef, Schema};

use crate::{
    arrow::validate_field_exists,
    graph::csr::{Csr, Offset, build_csr},
};

/// Compact internal node identifier used for traversal.
pub type NodeId = u32;

/// Compact internal edge identifier used for traversal.
pub type EdgeId = u32;

pub const ID_COL: &str = "id";
pub const TYPE_COL: &str = "type";

pub const EDGE_SRC_COL: &str = "src";
pub const EDGE_DEST_COL: &str = "dest";

const NODE_TOPOLOGY_COLS: &[&str] = &[ID_COL];
const EDGE_TOPOLOGY_COLS: &[&str] = &[ID_COL, EDGE_SRC_COL, EDGE_DEST_COL];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphId<'a> {
    U64(u64),
    Str(&'a str),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OwnedGraphId {
    U64(u64),
    Str(String),
}

impl OwnedGraphId {
    pub fn as_ref(&self) -> GraphId<'_> {
        match self {
            Self::U64(value) => GraphId::U64(*value),
            Self::Str(value) => GraphId::Str(value),
        }
    }
}

impl GraphId<'_> {
    pub fn into_owned(self) -> OwnedGraphId {
        match self {
            Self::U64(value) => OwnedGraphId::U64(value),
            Self::Str(value) => OwnedGraphId::Str(value.to_owned()),
        }
    }
}

impl From<u64> for OwnedGraphId {
    fn from(value: u64) -> Self {
        Self::U64(value)
    }
}

impl From<&str> for OwnedGraphId {
    fn from(value: &str) -> Self {
        Self::Str(value.to_owned())
    }
}

impl From<String> for OwnedGraphId {
    fn from(value: String) -> Self {
        Self::Str(value)
    }
}

impl fmt::Display for GraphId<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U64(value) => write!(f, "{value}"),
            Self::Str(value) => write!(f, "{value:?}"),
        }
    }
}

impl fmt::Display for OwnedGraphId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

/// Read-only graph storage operations used by traversal and algorithms.
pub trait GraphRepo {
    fn outgoing(&self, node: NodeId) -> impl Iterator<Item = (EdgeId, NodeId)>;
    fn outgoing_slice(&self, node: NodeId) -> (&[EdgeId], &[NodeId]);
    fn incoming(&self, node: NodeId) -> impl Iterator<Item = NodeId>;
    fn internal_node(&self, external: GraphId<'_>) -> Option<NodeId>;
    fn external_node(&self, internal: NodeId) -> Option<GraphId<'_>>;
    fn external_edge(&self, internal: EdgeId) -> Option<GraphId<'_>>;
    fn out_degree(&self, node: NodeId) -> usize;
    fn in_degree(&self, node: NodeId) -> usize;
}

#[derive(Debug)]
pub(crate) struct Repo {
    csr_offsets: Vec<Offset>,
    csr_dests: Vec<NodeId>,
    edge_ids: Vec<EdgeId>,
    node_count: usize,
    edge_count: usize,

    identity: Identity,
    pub nodes: RecordBatch,
    pub edges: RecordBatch,

    /// Reverse adjacency (incoming edges).
    /// Used for optimization - only some searches require it and it's built lazily on first use
    /// to keep construction memory and time low (and proportional) foraward only workloads
    /// (like BFS, as opposed to WCC or degrees).
    incoming: OnceLock<IncomingCsr>,
    /// Endpoints retained to build the reverse CSR lazily without re-reading Arrow columns.
    edge_endpoints: Vec<(NodeId, NodeId)>,

    /// Degree vectors, only used when whole-graph degree query and cached after.
    /// Search-only workloads never touch these, so construction stays cheap;
    /// degree-heavy workloads pay the O(n) build once instead of on every call.
    out_degrees: OnceLock<Vec<usize>>,
    in_degrees: OnceLock<Vec<usize>>,
    degrees: OnceLock<Vec<usize>>,
}

#[derive(Debug)]
struct IncomingCsr {
    offsets: Vec<Offset>,
    srcs: Vec<NodeId>,
}

#[derive(Debug)]
enum Identity {
    U64Contiguous {
        node_count: usize,
        edge_count: usize,
    },
    U64 {
        nodes: Vec<u64>,
        edges: Vec<u64>,
        node_lookup: HashMap<u64, NodeId>,
    },
    Str {
        nodes: Vec<Arc<str>>,
        edges: Vec<Arc<str>>,
        node_lookup: HashMap<Arc<str>, NodeId>,
    },
}

impl Identity {
    fn is_contiguous_u64(&self) -> bool {
        matches!(self, Self::U64Contiguous { .. })
    }

    fn internal_node_u64(&self, external: u64) -> Option<NodeId> {
        match self {
            Self::U64Contiguous { node_count, .. } if (external as usize) < *node_count => {
                Some(external as NodeId)
            }
            Self::U64 { node_lookup, .. } => node_lookup.get(&external).copied(),
            _ => None,
        }
    }

    fn external_node_u64(&self, internal: NodeId) -> Option<u64> {
        match self {
            Self::U64Contiguous { node_count, .. } if (internal as usize) < *node_count => {
                Some(internal as u64)
            }
            Self::U64 { nodes, .. } => nodes.get(internal as usize).copied(),
            _ => None,
        }
    }

    fn internal_node(&self, external: GraphId<'_>) -> Option<NodeId> {
        match (self, external) {
            (Self::U64Contiguous { node_count, .. }, GraphId::U64(value))
                if (value as usize) < *node_count =>
            {
                Some(value as NodeId)
            }
            (Self::U64 { node_lookup, .. }, GraphId::U64(value)) => {
                node_lookup.get(&value).copied()
            }
            (Self::Str { node_lookup, .. }, GraphId::Str(value)) => node_lookup.get(value).copied(),
            _ => None,
        }
    }

    fn external_node(&self, internal: NodeId) -> Option<GraphId<'_>> {
        match self {
            Self::U64Contiguous { node_count, .. } if (internal as usize) < *node_count => {
                Some(GraphId::U64(internal as u64))
            }
            Self::U64Contiguous { .. } => None,
            Self::U64 { nodes, .. } => nodes.get(internal as usize).copied().map(GraphId::U64),
            Self::Str { nodes, .. } => nodes
                .get(internal as usize)
                .map(|value| GraphId::Str(value.as_ref())),
        }
    }

    fn external_edge(&self, internal: EdgeId) -> Option<GraphId<'_>> {
        match self {
            Self::U64Contiguous { edge_count, .. } if (internal as usize) < *edge_count => {
                Some(GraphId::U64(internal as u64))
            }
            Self::U64Contiguous { .. } => None,
            Self::U64 { edges, .. } => edges.get(internal as usize).copied().map(GraphId::U64),
            Self::Str { edges, .. } => edges
                .get(internal as usize)
                .map(|value| GraphId::Str(value.as_ref())),
        }
    }
}

impl Repo {
    pub(crate) fn is_contiguous_u64(&self) -> bool {
        self.identity.is_contiguous_u64()
    }

    pub(crate) fn has_u64_ids(&self) -> bool {
        matches!(
            self.identity,
            Identity::U64Contiguous { .. } | Identity::U64 { .. }
        )
    }

    pub(crate) fn node_count(&self) -> usize {
        self.node_count
    }

    pub(crate) fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Replaces the payload (attribute) tables without rebuilding topology.
    ///
    /// Used by lazy graphs to swap in column-projected payload batches for a single search.
    /// The new batches must keep the original row order and count: DSL column reads index
    /// payload arrays by internal node/edge ID, which equals the Arrow row position.
    /// Identity (`id`/`src`/`dest`) is resolved from the precomputed mapping, not these
    /// batches, so topology columns are discarded if callers include them.
    pub(crate) fn set_payloads(&mut self, nodes: RecordBatch, edges: RecordBatch) -> Result<()> {
        if nodes.num_rows() != self.node_count {
            bail!(
                "projected nodes table has {} rows but topology expects {}",
                nodes.num_rows(),
                self.node_count
            );
        }
        if edges.num_rows() != self.edge_count {
            bail!(
                "projected edges table has {} rows but topology expects {}",
                edges.num_rows(),
                self.edge_count
            );
        }
        self.nodes = strip_topology_columns(nodes, NODE_TOPOLOGY_COLS)?;
        self.edges = strip_topology_columns(edges, EDGE_TOPOLOGY_COLS)?;
        Ok(())
    }

    pub(crate) fn internal_node_u64(&self, external: u64) -> Option<NodeId> {
        self.identity.internal_node_u64(external)
    }

    pub(crate) fn external_node_u64(&self, internal: NodeId) -> Option<u64> {
        self.identity.external_node_u64(internal)
    }
}

impl GraphRepo for Repo {
    fn outgoing(&self, node: NodeId) -> impl Iterator<Item = (EdgeId, NodeId)> {
        let i = node as usize;
        let start = self.csr_offsets[i] as usize;
        let end = self.csr_offsets[i + 1] as usize;

        self.edge_ids[start..end]
            .iter()
            .copied()
            .zip(self.csr_dests[start..end].iter().copied())
    }

    fn outgoing_slice(&self, node: NodeId) -> (&[EdgeId], &[NodeId]) {
        let i = node as usize;
        let start = self.csr_offsets[i] as usize;
        let end = self.csr_offsets[i + 1] as usize;
        (&self.edge_ids[start..end], &self.csr_dests[start..end])
    }

    fn incoming(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        let incoming = self.incoming();
        let i = node as usize;
        let start = incoming.offsets[i] as usize;
        let end = incoming.offsets[i + 1] as usize;
        incoming.srcs[start..end].iter().copied()
    }

    fn internal_node(&self, external: GraphId<'_>) -> Option<NodeId> {
        self.identity.internal_node(external)
    }

    fn external_node(&self, internal: NodeId) -> Option<GraphId<'_>> {
        self.identity.external_node(internal)
    }

    fn external_edge(&self, internal: EdgeId) -> Option<GraphId<'_>> {
        self.identity.external_edge(internal)
    }

    fn out_degree(&self, node: NodeId) -> usize {
        let i = node as usize;
        (self.csr_offsets[i + 1] - self.csr_offsets[i]) as usize
    }

    fn in_degree(&self, node: NodeId) -> usize {
        let incoming = self.incoming();
        let i = node as usize;
        (incoming.offsets[i + 1] - incoming.offsets[i]) as usize
    }
}

impl Repo {
    /// Returns the reverse-adjacency CSR, building it on first use.
    fn incoming(&self) -> &IncomingCsr {
        self.incoming
            .get_or_init(|| build_incoming_csr(self.node_count, &self.edge_endpoints))
    }

    pub(crate) fn out_degrees(&self) -> Vec<usize> {
        self.out_degrees
            .get_or_init(|| degrees_from_offsets(&self.csr_offsets))
            .clone()
    }

    pub(crate) fn in_degrees(&self) -> Vec<usize> {
        self.in_degrees
            .get_or_init(|| degrees_from_offsets(&self.incoming().offsets))
            .clone()
    }

    pub(crate) fn degrees(&self) -> Vec<usize> {
        self.degrees.get_or_init(|| self.compute_degrees()).clone()
    }

    fn compute_degrees(&self) -> Vec<usize> {
        let out = &self.csr_offsets;
        let incoming = &self.incoming().offsets;
        (0..self.node_count)
            .map(|i| {
                let out_deg = (out[i + 1] - out[i]) as usize;
                let in_deg = (incoming[i + 1] - incoming[i]) as usize;
                out_deg + in_deg
            })
            .collect()
    }
}

impl Repo {
    pub(crate) fn from_tables(nodes: RecordBatch, edges: RecordBatch) -> Result<Self> {
        let Preprocessed {
            identity,
            edge_endpoints,
        } = preprocess_graph(&nodes, &edges)?;
        let node_count = nodes.num_rows();

        let Csr {
            offsets: csr_offsets,
            edge_ids,
            dests: csr_dests,
        } = build_csr(node_count, &edge_endpoints).context("failed to construct CSR")?;

        Ok(Self {
            nodes: strip_topology_columns(nodes, NODE_TOPOLOGY_COLS)?,
            edges: strip_topology_columns(edges, EDGE_TOPOLOGY_COLS)?,
            node_count,
            edge_count: edge_endpoints.len(),
            csr_offsets,
            csr_dests,
            edge_ids,
            incoming: OnceLock::new(),
            edge_endpoints,
            identity,
            out_degrees: OnceLock::new(),
            in_degrees: OnceLock::new(),
            degrees: OnceLock::new(),
        })
    }
}

fn strip_topology_columns(batch: RecordBatch, topology_cols: &[&str]) -> Result<RecordBatch> {
    let row_count = batch.num_rows();
    let mut fields = Vec::<FieldRef>::with_capacity(batch.num_columns());
    let mut columns = Vec::with_capacity(batch.num_columns());

    for (index, field) in batch.schema().fields().iter().enumerate() {
        if topology_cols.contains(&field.name().as_str()) {
            continue;
        }
        fields.push(field.clone());
        columns.push(batch.column(index).clone());
    }

    RecordBatch::try_new_with_options(
        Schema::new(fields).into(),
        columns,
        &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(row_count)),
    )
    .context("failed to build payload table")
}

fn degrees_from_offsets(offsets: &[Offset]) -> Vec<usize> {
    offsets
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as usize)
        .collect()
}

fn build_incoming_csr(node_count: usize, edge_endpoints: &[(NodeId, NodeId)]) -> IncomingCsr {
    if edge_endpoints.len() > Offset::MAX as usize {
        panic!(
            "too many edges for u32 CSR offsets ({} > {})",
            edge_endpoints.len(),
            Offset::MAX
        );
    }

    let mut offsets = vec![0 as Offset; node_count + 1];
    for &(_, dest) in edge_endpoints {
        offsets[dest as usize + 1] += 1;
    }
    for i in 1..offsets.len() {
        offsets[i] += offsets[i - 1];
    }

    let mut srcs = vec![0; edge_endpoints.len()];
    let mut cursor = offsets.clone();
    for &(src, dest) in edge_endpoints {
        let pos = cursor[dest as usize] as usize;
        srcs[pos] = src;
        cursor[dest as usize] += 1;
    }
    IncomingCsr { offsets, srcs }
}

struct Preprocessed {
    identity: Identity,
    edge_endpoints: Vec<(NodeId, NodeId)>,
}

fn preprocess_graph(nodes: &RecordBatch, edges: &RecordBatch) -> Result<Preprocessed> {
    validate_type_col(nodes, "nodes")?;
    validate_type_col(edges, "edges")?;

    let mode = id_mode(nodes, ID_COL).context("validation failed for nodes table")?;
    require_mode(edges, ID_COL, mode).context("validation failed for edges table")?;
    require_mode(edges, EDGE_SRC_COL, mode).context("validation failed for edges table")?;
    require_mode(edges, EDGE_DEST_COL, mode).context("validation failed for edges table")?;

    match mode {
        IdMode::U64 => preprocess_u64(nodes, edges),
        IdMode::Str => preprocess_str(nodes, edges),
    }
}

fn preprocess_u64(nodes: &RecordBatch, edges: &RecordBatch) -> Result<Preprocessed> {
    let node_ids = u64_col(nodes, ID_COL)?;
    let edge_ids = u64_col(edges, ID_COL)?;
    let edge_srcs = u64_col(edges, EDGE_SRC_COL)?;
    let edge_dests = u64_col(edges, EDGE_DEST_COL)?;

    let mut node_lookup = HashMap::with_capacity(nodes.num_rows());
    let mut nodes_out = Vec::with_capacity(nodes.num_rows());
    for row in 0..nodes.num_rows() {
        if node_ids.is_null(row) {
            bail!("nodes row {row} has null id");
        }
        let id = node_ids.value(row);
        let internal = checked_id(row, "node")?;
        if node_lookup.insert(id, internal).is_some() {
            bail!("duplicate node id {id}");
        }
        nodes_out.push(id);
    }

    let mut edges_out = Vec::with_capacity(edges.num_rows());
    let mut edge_lookup = HashMap::with_capacity(edges.num_rows());
    let mut edge_endpoints = Vec::with_capacity(edges.num_rows());
    for row in 0..edges.num_rows() {
        if edge_ids.is_null(row) {
            bail!("edges row {row} has null id");
        }
        if edge_srcs.is_null(row) {
            bail!("edges row {row} has null src");
        }
        if edge_dests.is_null(row) {
            bail!("edges row {row} has null dest");
        }

        let id = edge_ids.value(row);
        if edge_lookup.insert(id, ()).is_some() {
            bail!("duplicate edge id {id}");
        }
        let src_external = edge_srcs.value(row);
        let dest_external = edge_dests.value(row);
        let src = *node_lookup
            .get(&src_external)
            .with_context(|| format!("edge row {row} references missing src {src_external}"))?;
        let dest = *node_lookup
            .get(&dest_external)
            .with_context(|| format!("edge row {row} references missing dest {dest_external}"))?;
        checked_id(row, "edge")?;
        edges_out.push(id);
        edge_endpoints.push((src, dest));
    }

    let identity = if nodes_out
        .iter()
        .enumerate()
        .all(|(row, &id)| id == row as u64)
        && edges_out
            .iter()
            .enumerate()
            .all(|(row, &id)| id == row as u64)
    {
        Identity::U64Contiguous {
            node_count: nodes_out.len(),
            edge_count: edges_out.len(),
        }
    } else {
        Identity::U64 {
            nodes: nodes_out,
            edges: edges_out,
            node_lookup,
        }
    };

    Ok(Preprocessed {
        identity,
        edge_endpoints,
    })
}

fn preprocess_str(nodes: &RecordBatch, edges: &RecordBatch) -> Result<Preprocessed> {
    let node_ids = str_col(nodes, ID_COL)?;
    let edge_ids = str_col(edges, ID_COL)?;
    let edge_srcs = str_col(edges, EDGE_SRC_COL)?;
    let edge_dests = str_col(edges, EDGE_DEST_COL)?;

    let mut node_lookup = HashMap::with_capacity(nodes.num_rows());
    let mut nodes_out = Vec::with_capacity(nodes.num_rows());
    for row in 0..nodes.num_rows() {
        let id = node_ids
            .value(row)
            .with_context(|| format!("nodes row {row} has null id"))?;
        let internal = checked_id(row, "node")?;
        let id = Arc::<str>::from(id);
        if node_lookup.insert(id.clone(), internal).is_some() {
            bail!("duplicate node id {id:?}");
        }
        nodes_out.push(id);
    }

    let mut edge_lookup = HashSet::with_capacity(edges.num_rows());
    let mut edges_out = Vec::with_capacity(edges.num_rows());
    let mut edge_endpoints = Vec::with_capacity(edges.num_rows());
    for row in 0..edges.num_rows() {
        let id = edge_ids
            .value(row)
            .with_context(|| format!("edges row {row} has null id"))?;
        let src_external = edge_srcs
            .value(row)
            .with_context(|| format!("edges row {row} has null src"))?;
        let dest_external = edge_dests
            .value(row)
            .with_context(|| format!("edges row {row} has null dest"))?;
        let id = Arc::<str>::from(id);
        if !edge_lookup.insert(id.clone()) {
            bail!("duplicate edge id {id:?}");
        }
        let src = *node_lookup
            .get(src_external)
            .with_context(|| format!("edge row {row} references missing src {src_external:?}"))?;
        let dest = *node_lookup
            .get(dest_external)
            .with_context(|| format!("edge row {row} references missing dest {dest_external:?}"))?;
        checked_id(row, "edge")?;
        edges_out.push(id);
        edge_endpoints.push((src, dest));
    }

    Ok(Preprocessed {
        identity: Identity::Str {
            nodes: nodes_out,
            edges: edges_out,
            node_lookup,
        },
        edge_endpoints,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdMode {
    U64,
    Str,
}

fn id_mode(batch: &RecordBatch, col: &str) -> Result<IdMode> {
    let ty = validate_field_exists(batch, col)?;
    match ty {
        DataType::UInt64 => Ok(IdMode::U64),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Ok(IdMode::Str),
        other => bail!("'{col}' must be UInt64 or string (actual type: {other})"),
    }
}

fn require_mode(batch: &RecordBatch, col: &str, mode: IdMode) -> Result<()> {
    let actual = id_mode(batch, col)?;
    if actual != mode {
        bail!("'{col}' must use the same ID type as nodes.id");
    }
    Ok(())
}

fn validate_type_col(batch: &RecordBatch, label: &str) -> Result<()> {
    if let Ok(ty) = validate_field_exists(batch, TYPE_COL)
        && !ty.is_string()
    {
        bail!("validation failed for {label} table: '{TYPE_COL}' must be a string when present");
    }
    Ok(())
}

fn u64_col<'a>(batch: &'a RecordBatch, col: &str) -> Result<&'a UInt64Array> {
    batch
        .column_by_name(col)
        .with_context(|| format!("missing '{col}' column"))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .with_context(|| format!("'{col}' must be UInt64"))
}

enum StrCol<'a> {
    Utf8(&'a StringArray),
    Large(&'a LargeStringArray),
    View(&'a StringViewArray),
}

impl StrCol<'_> {
    fn value(&self, row: usize) -> Option<&str> {
        match self {
            Self::Utf8(array) => (!array.is_null(row)).then(|| array.value(row)),
            Self::Large(array) => (!array.is_null(row)).then(|| array.value(row)),
            Self::View(array) => (!array.is_null(row)).then(|| array.value(row)),
        }
    }
}

fn str_col<'a>(batch: &'a RecordBatch, col: &str) -> Result<StrCol<'a>> {
    let array = batch
        .column_by_name(col)
        .with_context(|| format!("missing '{col}' column"))?;
    match array.data_type() {
        DataType::Utf8 => Ok(StrCol::Utf8(
            array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("validated Utf8 array"),
        )),
        DataType::LargeUtf8 => Ok(StrCol::Large(
            array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("validated LargeUtf8 array"),
        )),
        DataType::Utf8View => Ok(StrCol::View(
            array
                .as_any()
                .downcast_ref::<StringViewArray>()
                .expect("validated Utf8View array"),
        )),
        other => bail!("'{col}' must be a string column (actual type: {other})"),
    }
}

fn checked_id(index: usize, kind: &str) -> Result<u32> {
    u32::try_from(index).with_context(|| format!("too many {kind}s for u32 internal IDs"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::record_batch;

    fn outgoing_for<'a>(repo: &'a Repo, external_id: GraphId<'_>) -> Vec<GraphId<'a>> {
        let internal_id = repo.internal_node(external_id).unwrap();
        repo.outgoing(internal_id)
            .map(|(_, node)| repo.external_node(node).unwrap())
            .collect()
    }

    #[test]
    fn builds_string_ids() {
        let nodes = record_batch!(
            (ID_COL, Utf8, ["a", "b", "c", "d"]),
            ("age", UInt32, [Some(20), None, Some(54), Some(23)])
        )
        .unwrap();
        let edges = record_batch!(
            (ID_COL, Utf8, ["a->b", "c->a", "c->d"]),
            (EDGE_SRC_COL, Utf8, ["a", "c", "c"]),
            (EDGE_DEST_COL, Utf8, ["b", "a", "d"])
        )
        .unwrap();

        let repo = Repo::from_tables(nodes, edges).unwrap();
        assert_eq!(
            outgoing_for(&repo, GraphId::Str("a")),
            vec![GraphId::Str("b")]
        );
        assert_eq!(
            outgoing_for(&repo, GraphId::Str("b")),
            Vec::<GraphId<'_>>::new()
        );
        assert_eq!(
            outgoing_for(&repo, GraphId::Str("c")),
            vec![GraphId::Str("a"), GraphId::Str("d")]
        );
        assert_eq!(repo.external_edge(2), Some(GraphId::Str("c->d")));
    }

    #[test]
    fn builds_u64_ids() {
        let nodes = record_batch!((ID_COL, UInt64, [10, 20, 30])).unwrap();
        let edges = record_batch!(
            (ID_COL, UInt64, [100, 200]),
            (EDGE_SRC_COL, UInt64, [10, 20]),
            (EDGE_DEST_COL, UInt64, [20, 30])
        )
        .unwrap();

        let repo = Repo::from_tables(nodes, edges).unwrap();
        assert_eq!(
            outgoing_for(&repo, GraphId::U64(10)),
            vec![GraphId::U64(20)]
        );
        assert_eq!(repo.external_edge(1), Some(GraphId::U64(200)));
    }

    #[test]
    fn validates_optional_type_columns() {
        let nodes = record_batch!(
            (ID_COL, Utf8, ["a", "b"]),
            (TYPE_COL, UInt32, [Some(2), None])
        )
        .unwrap();
        let edges = record_batch!(
            (ID_COL, Utf8, ["a->b"]),
            (EDGE_SRC_COL, Utf8, ["a"]),
            (EDGE_DEST_COL, Utf8, ["b"])
        )
        .unwrap();

        let err = Repo::from_tables(nodes, edges).unwrap_err().to_string();
        assert!(err.contains("'type' must be a string"));
    }

    #[test]
    fn rejects_mixed_id_modes() {
        let nodes = record_batch!((ID_COL, UInt64, [1, 2])).unwrap();
        let edges = record_batch!(
            (ID_COL, UInt64, [10]),
            (EDGE_SRC_COL, Utf8, ["1"]),
            (EDGE_DEST_COL, UInt64, [2])
        )
        .unwrap();

        let err = format!("{:#}", Repo::from_tables(nodes, edges).unwrap_err());
        assert!(err.contains("same ID type"));
    }

    #[test]
    fn rejects_nulls_duplicates_and_missing_endpoints() {
        let nodes = record_batch!((ID_COL, UInt64, [Some(1), Some(1)])).unwrap();
        let edges = record_batch!(
            (ID_COL, UInt64, [10]),
            (EDGE_SRC_COL, UInt64, [1]),
            (EDGE_DEST_COL, UInt64, [2])
        )
        .unwrap();
        assert!(
            Repo::from_tables(nodes, edges)
                .unwrap_err()
                .to_string()
                .contains("duplicate node id")
        );

        let nodes = record_batch!((ID_COL, Utf8, ["a", "b"])).unwrap();
        let edges = record_batch!(
            (ID_COL, Utf8, ["ab"]),
            (EDGE_SRC_COL, Utf8, ["a"]),
            (EDGE_DEST_COL, Utf8, ["missing"])
        )
        .unwrap();
        assert!(
            Repo::from_tables(nodes, edges)
                .unwrap_err()
                .to_string()
                .contains("missing dest")
        );
    }

    #[test]
    fn set_payloads_swaps_columns_and_keeps_topology() {
        let nodes = record_batch!((ID_COL, UInt64, [0, 1, 2])).unwrap();
        let edges = record_batch!(
            (ID_COL, UInt64, [0, 1]),
            (EDGE_SRC_COL, UInt64, [0, 1]),
            (EDGE_DEST_COL, UInt64, [1, 2])
        )
        .unwrap();
        let mut repo = Repo::from_tables(nodes, edges).unwrap();

        // Project to a different set of payload columns (same row counts).
        let new_nodes =
            record_batch!((ID_COL, UInt64, [0, 1, 2]), ("score", Int64, [10, 20, 30])).unwrap();
        let new_edges = record_batch!(
            (ID_COL, UInt64, [0, 1]),
            (EDGE_SRC_COL, UInt64, [0, 1]),
            (EDGE_DEST_COL, UInt64, [1, 2])
        )
        .unwrap();
        repo.set_payloads(new_nodes, new_edges).unwrap();

        // Topology is unchanged after the swap.
        assert_eq!(outgoing_for(&repo, GraphId::U64(0)), vec![GraphId::U64(1)]);
        assert!(repo.nodes.column_by_name(ID_COL).is_none());
        assert!(repo.edges.column_by_name(ID_COL).is_none());
        assert!(repo.edges.column_by_name(EDGE_SRC_COL).is_none());
        assert!(repo.edges.column_by_name(EDGE_DEST_COL).is_none());
        assert!(repo.nodes.column_by_name("score").is_some());
    }

    #[test]
    fn set_payloads_rejects_row_count_mismatch() {
        let nodes = record_batch!((ID_COL, UInt64, [0, 1, 2])).unwrap();
        let edges = record_batch!(
            (ID_COL, UInt64, [0]),
            (EDGE_SRC_COL, UInt64, [0]),
            (EDGE_DEST_COL, UInt64, [1])
        )
        .unwrap();
        let mut repo = Repo::from_tables(nodes, edges).unwrap();

        let bad_nodes = record_batch!((ID_COL, UInt64, [0, 1])).unwrap();
        let same_edges = record_batch!(
            (ID_COL, UInt64, [0]),
            (EDGE_SRC_COL, UInt64, [0]),
            (EDGE_DEST_COL, UInt64, [1])
        )
        .unwrap();
        let err = repo
            .set_payloads(bad_nodes, same_edges)
            .unwrap_err()
            .to_string();
        assert!(err.contains("projected nodes table has 2 rows"));
    }
}
