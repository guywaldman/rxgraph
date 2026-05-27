//! Graph construction and storage.
//!
//! This module validates Arrow node/edge tables, maps external `u64` node IDs
//! to compact internal IDs, and stores outgoing edges in CSR form for traversal.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use arrow::{
    array::{Array, UInt64Array},
    record_batch::RecordBatch,
};

/// Compact internal node identifier used by traversal code.
pub type NodeId = u32;

/// Stable edge identifier matching edge insertion order across edge tables.
pub type EdgeId = u32;

/// Immutable Arrow-backed directed graph.
///
/// Nodes are supplied by one or more record batches with a required `id:
/// UInt64` column. Edges are supplied by one or more record batches with
/// required `src: UInt64` and `dest: UInt64` columns. Other columns remain in
/// their Arrow arrays and can be read by traversal kernels as `src.*`,
/// `dest.*`, or `edge.*`.
#[derive(Debug)]
pub struct Graph {
    node_tables: Vec<Table>,
    edge_tables: Vec<Table>,
    external_to_internal: HashMap<u64, NodeId>,
    internal_to_external: Vec<u64>,
    node_rows: Vec<RowRef>,
    edge_rows: Vec<RowRef>,
    offsets: Vec<usize>,
    edge_ids: Vec<EdgeId>,
    dests: Vec<NodeId>,
}

#[derive(Debug)]
pub(crate) struct Table {
    pub(crate) batch: RecordBatch,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RowRef {
    pub(crate) table: usize,
    pub(crate) row: usize,
}

/// Builder for an Arrow-backed graph.
///
/// Each table has a type label used only for diagnostics today; rows from all
/// node tables share one global node ID namespace, and rows from all edge tables
/// share one global edge ID namespace.
#[derive(Debug, Default)]
pub struct GraphBuilder {
    node_tables: Vec<(String, RecordBatch)>,
    edge_tables: Vec<(String, RecordBatch)>,
}

impl GraphBuilder {
    /// Creates an empty graph builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node table.
    ///
    /// The table must contain a non-null `id` column of Arrow type `UInt64`.
    pub fn with_node_table(mut self, typ: impl Into<String>, batch: RecordBatch) -> Self {
        self.node_tables.push((typ.into(), batch));
        self
    }

    /// Adds an edge table.
    ///
    /// The table must contain non-null `src` and `dest` columns of Arrow type
    /// `UInt64`, and every endpoint must reference an existing node ID.
    pub fn with_edge_table(mut self, typ: impl Into<String>, batch: RecordBatch) -> Self {
        self.edge_tables.push((typ.into(), batch));
        self
    }

    /// Validates all tables and returns an immutable graph.
    pub fn build(self) -> Result<Graph> {
        Graph::from_tables(self.node_tables, self.edge_tables)
    }
}

impl Graph {
    fn from_tables(
        node_batches: Vec<(String, RecordBatch)>,
        edge_batches: Vec<(String, RecordBatch)>,
    ) -> Result<Self> {
        let node_count_hint = node_batches.iter().map(|(_, batch)| batch.num_rows()).sum();
        let edge_count_hint = edge_batches.iter().map(|(_, batch)| batch.num_rows()).sum();
        let mut external_to_internal = HashMap::with_capacity(node_count_hint);
        let mut internal_to_external = Vec::with_capacity(node_count_hint);
        let mut node_rows = Vec::with_capacity(node_count_hint);
        let mut node_tables = Vec::with_capacity(node_batches.len());

        for (table_idx, (typ, batch)) in node_batches.into_iter().enumerate() {
            let ids = required_u64(&batch, "id", "node", &typ)?;

            for row in 0..batch.num_rows() {
                if ids.is_null(row) {
                    bail!("node table {typ:?} row {row} has null id");
                }

                let external = ids.value(row);
                let internal = checked_id(internal_to_external.len(), "node")?;

                if external_to_internal.insert(external, internal).is_some() {
                    bail!("duplicate node id {external} in table {typ:?} row {row}");
                }

                internal_to_external.push(external);
                node_rows.push(RowRef {
                    table: table_idx,
                    row,
                });
            }

            node_tables.push(Table { batch });
        }

        let node_count = internal_to_external.len();
        let mut edge_tables = Vec::with_capacity(edge_batches.len());
        let mut edge_rows = Vec::with_capacity(edge_count_hint);
        let mut edge_endpoints = Vec::with_capacity(edge_count_hint);

        for (table_idx, (typ, batch)) in edge_batches.into_iter().enumerate() {
            let srcs = required_u64(&batch, "src", "edge", &typ)?;
            let dests = required_u64(&batch, "dest", "edge", &typ)?;

            for row in 0..batch.num_rows() {
                if srcs.is_null(row) {
                    bail!("edge table {typ:?} row {row} has null src");
                }
                if dests.is_null(row) {
                    bail!("edge table {typ:?} row {row} has null dest");
                }

                let src_external = srcs.value(row);
                let dest_external = dests.value(row);
                let src = *external_to_internal.get(&src_external).with_context(|| {
                    format!("edge table {typ:?} row {row} references missing src {src_external}")
                })?;
                let dest = *external_to_internal.get(&dest_external).with_context(|| {
                    format!("edge table {typ:?} row {row} references missing dest {dest_external}")
                })?;

                checked_id(edge_rows.len(), "edge")?;
                edge_rows.push(RowRef {
                    table: table_idx,
                    row,
                });
                edge_endpoints.push((src, dest));
            }

            edge_tables.push(Table { batch });
        }

        let (offsets, edge_ids, csr_dests) = build_csr(node_count, &edge_endpoints)?;

        Ok(Self {
            node_tables,
            edge_tables,
            external_to_internal,
            internal_to_external,
            node_rows,
            edge_rows,
            offsets,
            edge_ids,
            dests: csr_dests,
        })
    }

    /// Executes a configured traversal against this graph.
    pub fn search(&self, traversal: crate::DslTraversal) -> Result<crate::SearchResult> {
        crate::traversal::search(self, traversal)
    }

    /// Returns the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.internal_to_external.len()
    }

    /// Returns the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edge_rows.len()
    }

    pub(crate) fn internal_node(&self, external: u64) -> Option<NodeId> {
        self.external_to_internal.get(&external).copied()
    }

    pub(crate) fn external_node(&self, node: NodeId) -> u64 {
        self.internal_to_external[node as usize]
    }

    pub(crate) fn outgoing(&self, node: NodeId) -> impl Iterator<Item = (EdgeId, NodeId)> + '_ {
        let i = node as usize;
        let start = self.offsets[i];
        let end = self.offsets[i + 1];

        self.edge_ids[start..end]
            .iter()
            .copied()
            .zip(self.dests[start..end].iter().copied())
    }

    pub(crate) fn out_degree(&self, node: NodeId) -> usize {
        let i = node as usize;
        self.offsets[i + 1] - self.offsets[i]
    }

    pub(crate) fn node_row(&self, node: NodeId) -> RowRef {
        self.node_rows[node as usize]
    }

    pub(crate) fn edge_row(&self, edge: EdgeId) -> RowRef {
        self.edge_rows[edge as usize]
    }

    pub(crate) fn node_tables(&self) -> &[Table] {
        &self.node_tables
    }

    pub(crate) fn edge_tables(&self) -> &[Table] {
        &self.edge_tables
    }
}

fn required_u64<'a>(
    batch: &'a RecordBatch,
    column: &str,
    kind: &str,
    typ: &str,
) -> Result<&'a UInt64Array> {
    let index = batch
        .schema()
        .index_of(column)
        .with_context(|| format!("{kind} table {typ:?} is missing required column {column:?}"))?;

    batch
        .column(index)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .with_context(|| {
            format!("{kind} table {typ:?} column {column:?} must have Arrow type UInt64")
        })
}

fn checked_id(id: usize, kind: &str) -> Result<u32> {
    u32::try_from(id).with_context(|| format!("too many {kind}s for u32 internal ids"))
}

fn build_csr(
    node_count: usize,
    edges: &[(NodeId, NodeId)],
) -> Result<(Vec<usize>, Vec<EdgeId>, Vec<NodeId>)> {
    let mut offsets = vec![0usize; node_count + 1];

    for &(src, _) in edges {
        offsets[src as usize + 1] += 1;
    }

    for i in 1..offsets.len() {
        offsets[i] += offsets[i - 1];
    }

    let mut edge_ids = vec![0; edges.len()];
    let mut dests = vec![0; edges.len()];
    let mut cursor = offsets.clone();

    for (edge_id, &(src, dest)) in edges.iter().enumerate() {
        let pos = cursor[src as usize];

        edge_ids[pos] = checked_id(edge_id, "edge")?;
        dests[pos] = dest;
        cursor[src as usize] += 1;
    }

    Ok((offsets, edge_ids, dests))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{edges, nodes};

    #[test]
    fn builds_graph_with_multiple_types() {
        let graph = GraphBuilder::new()
            .with_node_table("person", nodes(&[10, 20], &["a", "b"], &[1, 2]))
            .with_node_table("project", nodes(&[30], &["c"], &[3]))
            .with_edge_table(
                "knows",
                edges(&[10, 10, 20], &[20, 30, 30], &["x", "y", "z"]),
            )
            .build()
            .unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);

        let node = graph.internal_node(10).unwrap();
        let outgoing = graph.outgoing(node).collect::<Vec<_>>();

        assert_eq!(outgoing.len(), 2);
        assert_eq!(graph.external_node(outgoing[0].1), 20);
        assert_eq!(graph.external_node(outgoing[1].1), 30);
    }

    #[test]
    fn rejects_duplicate_node_ids() {
        let err = GraphBuilder::new()
            .with_node_table("person", nodes(&[10, 10], &["a", "b"], &[1, 2]))
            .build()
            .unwrap_err();

        assert!(err.to_string().contains("duplicate node id 10"));
    }

    #[test]
    fn rejects_missing_edge_endpoint() {
        let err = GraphBuilder::new()
            .with_node_table("person", nodes(&[10], &["a"], &[1]))
            .with_edge_table("knows", edges(&[10], &[99], &["x"]))
            .build()
            .unwrap_err();

        assert!(err.to_string().contains("missing dest 99"));
    }
}
