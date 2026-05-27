//! Common graph algorithms built directly on the graph storage.

use std::collections::VecDeque;

use anyhow::Result;

use crate::{
    Graph,
    graph::{EdgeId, NodeId},
};

impl Graph {
    /// Returns nodes reachable from `start` in breadth-first order.
    pub fn bfs(&self, start: u64, max_depth: Option<usize>) -> Result<Vec<u64>> {
        let start = self.required_internal_node(start)?;
        Ok(self
            .walk_breadth_first(start, max_depth)
            .into_iter()
            .map(|node| self.external_node(node))
            .collect())
    }

    /// Returns nodes reachable from `start` in depth-first pre-order.
    pub fn dfs(&self, start: u64, max_depth: Option<usize>) -> Result<Vec<u64>> {
        let start = self.required_internal_node(start)?;
        Ok(self
            .walk_depth_first(start, max_depth)
            .into_iter()
            .map(|node| self.external_node(node))
            .collect())
    }

    /// Returns all nodes reachable from `start`.
    pub fn reachable_nodes(&self, start: u64) -> Result<Vec<u64>> {
        self.bfs(start, None)
    }

    /// Returns an unweighted directed shortest path, or `None` if unreachable.
    pub fn shortest_path(&self, source: u64, target: u64) -> Result<Option<Vec<u64>>> {
        let source = self.required_internal_node(source)?;
        let target = self.required_internal_node(target)?;

        if source == target {
            return Ok(Some(vec![self.external_node(source)]));
        }

        let mut visited = vec![false; self.node_count()];
        let mut parent = vec![None; self.node_count()];
        let mut frontier = VecDeque::from([source]);

        visited[source as usize] = true;

        while let Some(node) = frontier.pop_front() {
            for (_, dest) in self.outgoing(node) {
                let dest_idx = dest as usize;
                if visited[dest_idx] {
                    continue;
                }

                visited[dest_idx] = true;
                parent[dest_idx] = Some(node);

                if dest == target {
                    return Ok(Some(self.materialize_path(source, target, &parent)));
                }

                frontier.push_back(dest);
            }
        }

        Ok(None)
    }

    /// Returns out-degree for each node in node insertion order.
    pub fn out_degrees(&self) -> Vec<usize> {
        (0..self.node_count())
            .map(|node| self.out_degree(node as NodeId))
            .collect()
    }

    /// Returns in-degree for each node in node insertion order.
    pub fn in_degrees(&self) -> Vec<usize> {
        let mut degrees = vec![0; self.node_count()];

        for edge in 0..self.edge_count() {
            degrees[self.edge_dest(edge as EdgeId) as usize] += 1;
        }

        degrees
    }

    /// Returns total directed degree for each node in node insertion order.
    pub fn degrees(&self) -> Vec<usize> {
        let mut degrees = self.out_degrees();

        for edge in 0..self.edge_count() {
            degrees[self.edge_dest(edge as EdgeId) as usize] += 1;
        }

        degrees
    }

    /// Returns weakly connected components, ignoring edge direction.
    pub fn weakly_connected_components(&self) -> Vec<Vec<u64>> {
        let incoming = self.incoming_adjacency();
        let mut visited = vec![false; self.node_count()];
        let mut components = Vec::new();

        for start in 0..self.node_count() {
            if visited[start] {
                continue;
            }

            let mut component = Vec::new();
            let mut frontier = VecDeque::from([start as NodeId]);
            visited[start] = true;

            while let Some(node) = frontier.pop_front() {
                component.push(self.external_node(node));

                for (_, dest) in self.outgoing(node) {
                    if !visited[dest as usize] {
                        visited[dest as usize] = true;
                        frontier.push_back(dest);
                    }
                }

                for &src in &incoming[node as usize] {
                    if !visited[src as usize] {
                        visited[src as usize] = true;
                        frontier.push_back(src);
                    }
                }
            }

            components.push(component);
        }

        components
    }

    fn required_internal_node(&self, external: u64) -> Result<NodeId> {
        self.internal_node(external)
            .ok_or_else(|| anyhow::anyhow!("node id {external} is not present in the graph"))
    }

    fn walk_breadth_first(&self, start: NodeId, max_depth: Option<usize>) -> Vec<NodeId> {
        let mut visited = vec![false; self.node_count()];
        let mut order = Vec::new();
        let mut frontier = VecDeque::from([(start, 0usize)]);

        visited[start as usize] = true;

        while let Some((node, depth)) = frontier.pop_front() {
            order.push(node);

            if max_depth.is_some_and(|max_depth| depth >= max_depth) {
                continue;
            }

            for (_, dest) in self.outgoing(node) {
                if !visited[dest as usize] {
                    visited[dest as usize] = true;
                    frontier.push_back((dest, depth + 1));
                }
            }
        }

        order
    }

    fn walk_depth_first(&self, start: NodeId, max_depth: Option<usize>) -> Vec<NodeId> {
        let mut visited = vec![false; self.node_count()];
        let mut order = Vec::new();
        let mut stack = vec![(start, 0usize)];

        while let Some((node, depth)) = stack.pop() {
            if visited[node as usize] {
                continue;
            }

            visited[node as usize] = true;
            order.push(node);

            if max_depth.is_some_and(|max_depth| depth >= max_depth) {
                continue;
            }

            let outgoing = self.outgoing(node).collect::<Vec<_>>();
            for (_, dest) in outgoing.into_iter().rev() {
                if !visited[dest as usize] {
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
    ) -> Vec<u64> {
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

    fn incoming_adjacency(&self) -> Vec<Vec<NodeId>> {
        let mut incoming = vec![Vec::new(); self.node_count()];

        for edge in 0..self.edge_count() {
            let edge = edge as EdgeId;
            incoming[self.edge_dest(edge) as usize].push(self.edge_source(edge));
        }

        incoming
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphBuilder, test_utils};

    fn graph() -> Graph {
        GraphBuilder::new()
            .with_node_table(
                "n",
                test_utils::nodes(
                    &[10, 20, 30, 40, 50],
                    &["a", "b", "c", "d", "e"],
                    &[0, 0, 0, 0, 0],
                ),
            )
            .with_edge_table(
                "e",
                test_utils::edges(&[10, 10, 20, 30], &[20, 30, 40, 40], &["a", "b", "c", "d"]),
            )
            .build()
            .unwrap()
    }

    #[test]
    fn traverses_breadth_first() {
        assert_eq!(graph().bfs(10, None).unwrap(), vec![10, 20, 30, 40]);
        assert_eq!(graph().bfs(10, Some(1)).unwrap(), vec![10, 20, 30]);
    }

    #[test]
    fn traverses_depth_first() {
        assert_eq!(graph().dfs(10, None).unwrap(), vec![10, 20, 40, 30]);
    }

    #[test]
    fn finds_unweighted_shortest_path() {
        assert_eq!(
            graph().shortest_path(10, 40).unwrap(),
            Some(vec![10, 20, 40])
        );
        assert_eq!(graph().shortest_path(40, 10).unwrap(), None);
    }

    #[test]
    fn computes_degrees() {
        assert_eq!(graph().out_degrees(), vec![2, 1, 1, 0, 0]);
        assert_eq!(graph().in_degrees(), vec![0, 1, 1, 2, 0]);
        assert_eq!(graph().degrees(), vec![2, 2, 2, 2, 0]);
    }

    #[test]
    fn finds_weak_components() {
        assert_eq!(
            graph().weakly_connected_components(),
            vec![vec![10, 20, 30, 40], vec![50]]
        );
    }

    #[test]
    fn rejects_missing_start_node() {
        let err = graph().bfs(99, None).unwrap_err();
        assert!(err.to_string().contains("node id 99"));
    }
}
