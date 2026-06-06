//! High-level graph API.
//!
//! [`Graph`] owns validated Arrow node/edge tables plus compact CSR adjacency.
//! Construction validates the identity schema once, maps external IDs to dense
//! internal `u32` IDs, and leaves all non-topology columns in Arrow for DSL
//! reads.
//!
//! The methods here are for topology-only queries: BFS/DFS reachability,
//! shortest path, degrees, and weakly connected components. Stateful path
//! enumeration lives in [`Graph::search`](Self::search), implemented by the
//! traversal module.

use anyhow::{Context, Result, anyhow};
use arrow::record_batch::RecordBatch;

use crate::graph::{GraphId, GraphRepo, NodeId, OwnedGraphId, repo::Repo};

pub struct Graph {
    pub(crate) repo: Repo,
}

impl Graph {
    /// Builds a graph from Arrow node and edge tables.
    ///
    /// Required columns:
    ///
    /// - nodes: `id`
    /// - edges: `id`, `src`, `dest`
    ///
    /// All identity columns must be uniformly `UInt64` or uniformly string.
    /// Additional columns remain available to traversal DSL expressions.
    pub fn new(nodes: RecordBatch, edges: RecordBatch) -> Result<Self> {
        Ok(Self {
            repo: Repo::from_tables(nodes, edges)?,
        })
    }

    /// Builds a topology-only graph from contiguous integer node IDs and edge pairs.
    ///
    /// Nodes are `0..node_count`, edge IDs are assigned by input order, and no
    /// node or edge payload columns are installed.
    pub fn from_u64_edges(
        node_count: usize,
        edges: impl IntoIterator<Item = (u64, u64)>,
    ) -> Result<Self> {
        Ok(Self {
            repo: Repo::from_u64_edges(node_count, edges)?,
        })
    }

    /// Replaces the payload (attribute) tables, reusing the existing topology.
    pub fn set_payloads(&mut self, nodes: RecordBatch, edges: RecordBatch) -> Result<()> {
        self.repo.set_payloads(nodes, edges)
    }

    /// Number of node rows.
    pub fn node_count(&self) -> usize {
        self.repo.node_count()
    }

    /// Number of edge rows.
    pub fn edge_count(&self) -> usize {
        self.repo.edge_count()
    }

    /// Topology-only breadth-first traversal from an external node ID.
    pub fn bfs(
        &self,
        start: impl Into<OwnedGraphId>,
        max_depth: Option<usize>,
    ) -> Result<Vec<GraphId<'_>>> {
        let start = start.into();
        let start = self.required_internal_node(start.as_ref())?;
        Ok(self
            .walk_breadth_first(start, max_depth)
            .into_iter()
            .map(|node| self.external_node(node))
            .collect())
    }

    /// Fast breadth-first traversal for integer-ID graphs.
    ///
    /// Returns `Ok(None)` when the graph uses string IDs or `start` is missing.
    pub fn bfs_u64(&self, start: u64, max_depth: Option<usize>) -> Result<Option<Vec<u64>>> {
        let Some(start) = self.repo.internal_node_u64(start) else {
            return Ok(None);
        };
        self.materialize_nodes_u64(self.walk_breadth_first(start, max_depth))
    }

    /// Batched breadth-first traversals for integer-ID graphs.
    ///
    /// Reuses generation-stamped scratch across starts, avoiding an O(nodes)
    /// visited-array clear per query.
    /// Returns `Ok(None)` when the graph uses string IDs.
    /// Missing start nodes produce `None` entries.
    pub fn bfs_many_u64(
        &self,
        starts: impl IntoIterator<Item = u64>,
        max_depth: Option<usize>,
    ) -> Result<Option<Vec<Option<Vec<u64>>>>> {
        if !self.repo.has_u64_ids() {
            return Ok(None);
        }

        let mut scratch = TopologyScratch::new(self.node_count());
        starts
            .into_iter()
            .map(|start| self.bfs_u64_with_scratch(start, max_depth, &mut scratch))
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    /// Topology-only depth-first traversal from an external node ID.
    pub fn dfs(
        &self,
        start: impl Into<OwnedGraphId>,
        max_depth: Option<usize>,
    ) -> Result<Vec<GraphId<'_>>> {
        let start = start.into();
        let start = self.required_internal_node(start.as_ref())?;
        Ok(self
            .walk_depth_first(start, max_depth)
            .into_iter()
            .map(|node| self.external_node(node))
            .collect())
    }

    /// Fast depth-first traversal for integer-ID graphs.
    ///
    /// Returns `Ok(None)` when the graph uses string IDs or `start` is missing.
    pub fn dfs_u64(&self, start: u64, max_depth: Option<usize>) -> Result<Option<Vec<u64>>> {
        let Some(start) = self.repo.internal_node_u64(start) else {
            return Ok(None);
        };
        self.materialize_nodes_u64(self.walk_depth_first(start, max_depth))
    }

    /// All nodes topologically reachable from `start` in BFS order.
    pub fn reachable_nodes(&self, start: impl Into<OwnedGraphId>) -> Result<Vec<GraphId<'_>>> {
        self.bfs(start, None)
    }

    /// Fast reachable-node query for integer-ID graphs.
    pub fn reachable_nodes_u64(&self, start: u64) -> Result<Option<Vec<u64>>> {
        self.bfs_u64(start, None)
    }

    /// Shortest unweighted directed path between two external node IDs.
    pub fn shortest_path(
        &self,
        source: impl Into<OwnedGraphId>,
        target: impl Into<OwnedGraphId>,
    ) -> Result<Option<Vec<GraphId<'_>>>> {
        let source = source.into();
        let target = target.into();
        let source = self.required_internal_node(source.as_ref())?;
        let target = self.required_internal_node(target.as_ref())?;

        if source == target {
            return Ok(Some(vec![self.external_node(source)]));
        }

        let mut visited = vec![0u8; self.node_count()];
        let mut parent = vec![None; self.node_count()];
        let mut frontier = vec![source];
        let mut head = 0;
        visited[source as usize] = 1;

        while let Some(&node) = frontier.get(head) {
            head += 1;
            let (_, dests) = self.repo.outgoing_slice(node);
            for &dest in dests {
                let dest_idx = dest as usize;
                if visited[dest_idx] != 0 {
                    continue;
                }

                visited[dest_idx] = 1;
                parent[dest_idx] = Some(node);
                if dest == target {
                    return Ok(Some(self.materialize_path(source, target, &parent)));
                }
                frontier.push(dest);
            }
        }

        Ok(None)
    }

    /// Fast shortest-path query for integer-ID graphs.
    ///
    /// Returns `Ok(None)` when either endpoint is missing or the graph uses
    /// string IDs. Returns `Ok(Some(None))` when both endpoints exist but no
    /// directed path connects them.
    pub fn shortest_path_u64(&self, source: u64, target: u64) -> Result<Option<Option<Vec<u64>>>> {
        let source_external = source;
        let Some(source) = self.repo.internal_node_u64(source) else {
            return Ok(None);
        };
        let Some(target) = self.repo.internal_node_u64(target) else {
            return Ok(None);
        };

        if source == target {
            return Ok(Some(Some(vec![source_external])));
        }

        let mut visited = vec![0u8; self.node_count()];
        let mut parent = vec![NodeId::MAX; self.node_count()];
        let mut frontier = vec![source];
        let mut head = 0;
        visited[source as usize] = 1;

        while let Some(&node) = frontier.get(head) {
            head += 1;
            let (_, dests) = self.repo.outgoing_slice(node);
            for &dest in dests {
                let dest_idx = dest as usize;
                if visited[dest_idx] != 0 {
                    continue;
                }

                visited[dest_idx] = 1;
                parent[dest_idx] = node;
                if dest == target {
                    return self.materialize_path_u64(source, target, &parent).map(Some);
                }
                frontier.push(dest);
            }
        }

        Ok(Some(None))
    }

    /// Batched shortest-path query for integer-ID graphs.
    ///
    /// Reuses generation-stamped visited/parent scratch across queries, which
    /// avoids O(nodes) zeroing for many small path queries. Returns `Ok(None)`
    /// when the graph uses string IDs. Missing endpoints or disconnected pairs
    /// produce `None` entries.
    pub fn shortest_paths_u64(
        &self,
        queries: impl IntoIterator<Item = (u64, u64)>,
    ) -> Result<Option<Vec<Option<Vec<u64>>>>> {
        if !self.repo.has_u64_ids() {
            return Ok(None);
        }

        let mut scratch = TopologyScratch::new(self.node_count());
        queries
            .into_iter()
            .map(|(source, target)| {
                self.shortest_path_u64_with_scratch(source, target, &mut scratch)
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    /// Out-degree per internal node row order.
    pub fn out_degrees(&self) -> Vec<usize> {
        self.repo.out_degrees()
    }

    /// In-degree per internal node row order.
    pub fn in_degrees(&self) -> Vec<usize> {
        self.repo.in_degrees()
    }

    /// In-degree plus out-degree per internal node row order.
    pub fn degrees(&self) -> Vec<usize> {
        self.repo.degrees()
    }

    /// Weakly connected components, materialized as external node IDs.
    pub fn weakly_connected_components(&self) -> Vec<Vec<GraphId<'_>>> {
        let mut visited = vec![0u8; self.node_count()];
        let mut components = Vec::new();
        // Reused across components to reduce allocations.
        let mut frontier = Vec::new();

        for start in 0..self.node_count() {
            if visited[start] != 0 {
                continue;
            }

            let mut component = Vec::new();
            frontier.clear();
            frontier.push(start as NodeId);
            let mut head = 0;
            visited[start] = 1;

            while let Some(&node) = frontier.get(head) {
                head += 1;
                component.push(self.external_node(node));

                for (_, dest) in self.repo.outgoing(node) {
                    if visited[dest as usize] == 0 {
                        visited[dest as usize] = 1;
                        frontier.push(dest);
                    }
                }

                for src in self.repo.incoming(node) {
                    if visited[src as usize] == 0 {
                        visited[src as usize] = 1;
                        frontier.push(src);
                    }
                }
            }

            components.push(component);
        }

        components
    }

    /// Fast weak-component query for integer-ID graphs.
    ///
    /// Returns `None` for string-ID graphs.
    pub fn weakly_connected_components_u64(&self) -> Option<Vec<Vec<u64>>> {
        let mut visited = vec![0u8; self.node_count()];
        let mut components = Vec::new();
        // Reused across components to reduce allocations.
        let mut frontier = Vec::new();

        for start in 0..self.node_count() {
            if visited[start] != 0 {
                continue;
            }

            let mut component = Vec::new();
            frontier.clear();
            frontier.push(start as NodeId);
            let mut head = 0;
            visited[start] = 1;

            while let Some(&node) = frontier.get(head) {
                head += 1;
                component.push(if self.repo.is_contiguous_u64() {
                    node as u64
                } else {
                    self.repo.external_node_u64(node)?
                });

                let (_, dests) = self.repo.outgoing_slice(node);
                for &dest in dests {
                    if visited[dest as usize] == 0 {
                        visited[dest as usize] = 1;
                        frontier.push(dest);
                    }
                }

                for src in self.repo.incoming(node) {
                    if visited[src as usize] == 0 {
                        visited[src as usize] = 1;
                        frontier.push(src);
                    }
                }
            }

            components.push(component);
        }

        Some(components)
    }

    fn required_internal_node(&self, external: GraphId<'_>) -> Result<NodeId> {
        self.repo
            .internal_node(external)
            .ok_or_else(|| anyhow!("node id {external} is not present in the graph"))
    }

    fn external_node(&self, node: NodeId) -> GraphId<'_> {
        self.repo
            .external_node(node)
            .expect("internal node must map to external id")
    }

    fn materialize_nodes_u64(&self, nodes: Vec<NodeId>) -> Result<Option<Vec<u64>>> {
        self.materialize_nodes_u64_slice(&nodes).map(Some)
    }

    fn materialize_nodes_u64_slice(&self, nodes: &[NodeId]) -> Result<Vec<u64>> {
        nodes
            .iter()
            .map(|&node| self.node_to_u64(node))
            .collect::<Result<Vec<_>>>()
    }

    fn node_to_u64(&self, node: NodeId) -> Result<u64> {
        if self.repo.is_contiguous_u64() {
            return Ok(node as u64);
        }
        self.repo
            .external_node_u64(node)
            .context("internal node must map to u64 id")
    }

    fn bfs_u64_with_scratch(
        &self,
        start: u64,
        max_depth: Option<usize>,
        scratch: &mut TopologyScratch,
    ) -> Result<Option<Vec<u64>>> {
        let Some(start) = self.repo.internal_node_u64(start) else {
            return Ok(None);
        };

        let generation = scratch.next_generation();
        if let Some(max_depth) = max_depth {
            scratch.frontier.clear();
            scratch.depth_frontier.clear();
            scratch.depth_frontier.push((start, 0));
            scratch.visited[start as usize] = generation;
            let mut head = 0;

            while let Some(&(node, depth)) = scratch.depth_frontier.get(head) {
                head += 1;
                scratch.frontier.push(node);
                if depth >= max_depth {
                    continue;
                }
                let (_, dests) = self.repo.outgoing_slice(node);
                for &dest in dests {
                    let dest_idx = dest as usize;
                    if scratch.visited[dest_idx] == generation {
                        continue;
                    }
                    scratch.visited[dest_idx] = generation;
                    scratch.depth_frontier.push((dest, depth + 1));
                }
            }
        } else {
            scratch.frontier.clear();
            scratch.frontier.push(start);
            scratch.visited[start as usize] = generation;
            let mut head = 0;

            while let Some(&node) = scratch.frontier.get(head) {
                head += 1;
                let (_, dests) = self.repo.outgoing_slice(node);
                for &dest in dests {
                    let dest_idx = dest as usize;
                    if scratch.visited[dest_idx] == generation {
                        continue;
                    }
                    scratch.visited[dest_idx] = generation;
                    scratch.frontier.push(dest);
                }
            }
        }

        self.materialize_nodes_u64_slice(&scratch.frontier)
            .map(Some)
    }

    fn shortest_path_u64_with_scratch(
        &self,
        source: u64,
        target: u64,
        scratch: &mut TopologyScratch,
    ) -> Result<Option<Vec<u64>>> {
        let Some(source) = self.repo.internal_node_u64(source) else {
            return Ok(None);
        };
        let Some(target) = self.repo.internal_node_u64(target) else {
            return Ok(None);
        };

        if source == target {
            return Ok(Some(vec![self.node_to_u64(source)?]));
        }

        let generation = scratch.next_generation();
        scratch.frontier.clear();
        scratch.frontier.push(source);
        scratch.visited[source as usize] = generation;
        let mut head = 0;

        while let Some(&node) = scratch.frontier.get(head) {
            head += 1;
            let (_, dests) = self.repo.outgoing_slice(node);
            for &dest in dests {
                let dest_idx = dest as usize;
                if scratch.visited[dest_idx] == generation {
                    continue;
                }

                scratch.visited[dest_idx] = generation;
                scratch.parent[dest_idx] = node;
                if dest == target {
                    return self.materialize_path_u64(source, target, &scratch.parent);
                }
                scratch.frontier.push(dest);
            }
        }

        Ok(None)
    }

    fn walk_breadth_first(&self, start: NodeId, max_depth: Option<usize>) -> Vec<NodeId> {
        if max_depth.is_none() {
            return self.walk_breadth_first_unbounded(start);
        }

        let mut visited = vec![0u8; self.node_count()];
        let mut order = Vec::new();
        let mut frontier = vec![(start, 0usize)];
        let mut head = 0;
        visited[start as usize] = 1;

        while let Some(&(node, depth)) = frontier.get(head) {
            head += 1;
            order.push(node);
            if max_depth.is_some_and(|max| depth >= max) {
                continue;
            }
            let (_, dests) = self.repo.outgoing_slice(node);
            for &dest in dests {
                if visited[dest as usize] == 0 {
                    visited[dest as usize] = 1;
                    frontier.push((dest, depth + 1));
                }
            }
        }

        order
    }

    fn walk_breadth_first_unbounded(&self, start: NodeId) -> Vec<NodeId> {
        let mut visited = vec![0u8; self.node_count()];
        let mut frontier = Vec::with_capacity(self.node_count().min(1024));
        let mut head = 0;
        frontier.push(start);
        visited[start as usize] = 1;

        while let Some(&node) = frontier.get(head) {
            head += 1;
            let (_, dests) = self.repo.outgoing_slice(node);
            for &dest in dests {
                if visited[dest as usize] == 0 {
                    visited[dest as usize] = 1;
                    frontier.push(dest);
                }
            }
        }

        frontier
    }

    fn walk_depth_first(&self, start: NodeId, max_depth: Option<usize>) -> Vec<NodeId> {
        let mut visited = vec![0u8; self.node_count()];
        let mut order = Vec::new();
        let mut stack = vec![(start, 0usize)];

        while let Some((node, depth)) = stack.pop() {
            if visited[node as usize] != 0 {
                continue;
            }
            visited[node as usize] = 1;
            order.push(node);
            if max_depth.is_some_and(|max| depth >= max) {
                continue;
            }

            let (_, dests) = self.repo.outgoing_slice(node);
            for &dest in dests.iter().rev() {
                if visited[dest as usize] == 0 {
                    stack.push((dest, depth + 1));
                }
            }
        }

        order
    }

    fn materialize_path(
        &self,
        source: NodeId,
        target: NodeId,
        parent: &[Option<NodeId>],
    ) -> Vec<GraphId<'_>> {
        let mut path = Vec::new();
        let mut node = target;

        while node != source {
            path.push(self.external_node(node));
            node = parent[node as usize].expect("target has a parent chain");
        }
        path.push(self.external_node(source));
        path.reverse();
        path
    }

    fn materialize_path_u64(
        &self,
        source: NodeId,
        target: NodeId,
        parent: &[NodeId],
    ) -> Result<Option<Vec<u64>>> {
        let mut path = Vec::new();
        let mut node = target;

        while node != source {
            path.push(if self.repo.is_contiguous_u64() {
                node as u64
            } else {
                self.repo
                    .external_node_u64(node)
                    .context("internal node must map to u64 id")?
            });
            node = parent[node as usize];
            debug_assert_ne!(node, NodeId::MAX);
        }
        path.push(if self.repo.is_contiguous_u64() {
            source as u64
        } else {
            self.repo
                .external_node_u64(source)
                .context("internal node must map to u64 id")?
        });
        path.reverse();
        Ok(Some(path))
    }
}

struct TopologyScratch {
    visited: Vec<u32>,
    generation: u32,
    parent: Vec<NodeId>,
    frontier: Vec<NodeId>,
    depth_frontier: Vec<(NodeId, usize)>,
}

impl TopologyScratch {
    fn new(node_count: usize) -> Self {
        Self {
            visited: vec![0; node_count],
            generation: 0,
            parent: vec![NodeId::MAX; node_count],
            frontier: Vec::new(),
            depth_frontier: Vec::new(),
        }
    }

    fn next_generation(&mut self) -> u32 {
        if self.generation == u32::MAX {
            self.visited.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::record_batch;

    use super::*;
    use crate::graph::repo::{EDGE_DEST_COL, EDGE_SRC_COL, ID_COL};

    #[test]
    fn batched_shortest_paths_reuse_topology_scratch() {
        let graph = small_graph();

        assert_eq!(
            graph
                .shortest_paths_u64([(0, 3), (2, 0), (4, 4), (99, 1)])
                .unwrap()
                .unwrap(),
            vec![Some(vec![0, 1, 2, 3]), None, Some(vec![4]), None]
        );
    }

    #[test]
    fn batched_bfs_reuses_topology_scratch() {
        let graph = small_graph();

        assert_eq!(
            graph.bfs_many_u64([0, 2, 99], Some(1)).unwrap().unwrap(),
            vec![Some(vec![0, 1]), Some(vec![2, 3]), None]
        );
    }

    fn small_graph() -> Graph {
        let nodes = record_batch!((ID_COL, UInt64, [0, 1, 2, 3, 4])).unwrap();
        let edges = record_batch!(
            (ID_COL, UInt64, [0, 1, 2]),
            (EDGE_SRC_COL, UInt64, [0, 1, 2]),
            (EDGE_DEST_COL, UInt64, [1, 2, 3])
        )
        .unwrap();
        Graph::new(nodes, edges).unwrap()
    }

    #[test]
    fn builds_topology_from_u64_edges() {
        let graph = Graph::from_u64_edges(5, [(0, 1), (0, 2), (1, 3), (2, 3)]).unwrap();

        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 4);
        assert_eq!(graph.bfs_u64(0, Some(1)).unwrap(), Some(vec![0, 1, 2]));
        assert_eq!(
            graph.shortest_path_u64(0, 3).unwrap(),
            Some(Some(vec![0, 1, 3]))
        );
        assert_eq!(graph.out_degrees(), vec![2, 1, 1, 0, 0]);
        assert_eq!(graph.in_degrees(), vec![0, 1, 1, 2, 0]);
        assert_eq!(graph.degrees(), vec![2, 2, 2, 2, 0]);
        assert_eq!(
            graph.weakly_connected_components_u64(),
            Some(vec![vec![0, 1, 2, 3], vec![4]])
        );
    }

    #[test]
    fn rejects_u64_edge_endpoint_outside_node_range() {
        let Err(err) = Graph::from_u64_edges(2, [(0, 2)]) else {
            panic!("expected endpoint validation to fail");
        };
        let err = err.to_string();

        assert!(err.contains("edge row 0 references missing dest 2"));
    }
}
