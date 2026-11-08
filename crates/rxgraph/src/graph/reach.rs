//! Reusable reverse-reachability index for named sets of internal nodes.

use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, bail};

use super::{EdgeId, Graph, GraphRepo, NodeId};

/// Groups of possible ending nodes and an optional topology-only edge filter.
///
/// Node IDs and edge IDs are the dense internal IDs used by traversal contexts.
/// `skip_edge` is called once for each edge while building the index. The
/// filter must select edges that are safe to exclude from every search that
/// consults the index. Targets are reachable from themselves by a zero-edge path.
#[derive(Default)]
pub struct ReachSpec<'a> {
    pub sets: Vec<(u32, Vec<NodeId>)>,
    pub skip_edge: Option<&'a (dyn Fn(EdgeId) -> bool + Sync)>,
}

enum ReachStorage {
    /// One word per node for up to 64 groups.
    Bits(Vec<u64>),
    /// Sorted group IDs packed into a CSR-style array per node.
    Sparse {
        offsets: Vec<usize>,
        groups: Vec<u32>,
    },
}

/// Reverse reachability for multiple named sets on one graph topology.
///
/// Build once and reuse across searches. A topology identity is retained so an
/// index cannot silently be used with a different graph's internal node IDs.
/// Up to 64 groups use a bitset per node; larger indexes store sorted group IDs
/// per node. Membership is expected O(1) for bitsets and one binary search for
/// sparse membership, after an expected O(1) group lookup.
pub struct ReachIndex {
    identity: Arc<()>,
    node_count: usize,
    targets: Vec<(u32, Vec<NodeId>)>,
    positions: HashMap<u32, usize>,
    storage: ReachStorage,
}

impl Graph {
    /// Builds a reusable reachability index. Duplicate group IDs and unknown
    /// internal node IDs are errors. The graph's reverse adjacency is cached.
    pub fn reach_index(&self, mut spec: ReachSpec<'_>) -> Result<ReachIndex> {
        spec.sets.sort_unstable_by_key(|(group, _)| *group);
        if spec.sets.windows(2).any(|sets| sets[0].0 == sets[1].0) {
            bail!("duplicate reachability group ID");
        }
        for (group, nodes) in &mut spec.sets {
            if let Some(&node) = nodes.iter().find(|&&n| n as usize >= self.node_count()) {
                bail!("unknown internal target node {node} in reachability group {group}");
            }
            nodes.sort_unstable();
            nodes.dedup();
        }

        let skipped = spec.skip_edge.map(|skip| {
            (0..self.edge_count())
                .map(|edge| skip(edge as EdgeId))
                .collect::<Vec<_>>()
        });
        let mut visited = vec![0_usize; self.node_count()];
        let mut stack = Vec::new();
        let mut bits = (spec.sets.len() <= 64).then(|| vec![0_u64; self.node_count()]);
        let mut sparse = (bits.is_none()).then(|| vec![Vec::new(); self.node_count()]);

        for (position, (group, targets)) in spec.sets.iter().enumerate() {
            // Positions are less than an allocated Vec's maximum length, so +1
            // cannot overflow usize; zero remains the unvisited sentinel.
            let stamp = position + 1;
            for &target in targets {
                if visited[target as usize] != stamp {
                    visited[target as usize] = stamp;
                    stack.push(target);
                }
            }
            while let Some(node) = stack.pop() {
                if let Some(bits) = &mut bits {
                    bits[node as usize] |= 1_u64 << position;
                } else if let Some(sparse) = &mut sparse {
                    // Groups are processed in ascending ID order.
                    sparse[node as usize].push(*group);
                }
                if let Some(skipped) = &skipped {
                    for (edge, src) in self.repo.incoming_edges(node) {
                        if !skipped[edge as usize] && visited[src as usize] != stamp {
                            visited[src as usize] = stamp;
                            stack.push(src);
                        }
                    }
                } else {
                    for src in self.repo.incoming(node) {
                        if visited[src as usize] != stamp {
                            visited[src as usize] = stamp;
                            stack.push(src);
                        }
                    }
                }
            }
        }

        let storage = if let Some(bits) = bits {
            ReachStorage::Bits(bits)
        } else {
            let sparse = sparse.expect("sparse storage for more than 64 groups");
            let mut offsets = Vec::with_capacity(self.node_count() + 1);
            let mut groups = Vec::new();
            offsets.push(0);
            for node_groups in sparse {
                groups.extend(node_groups);
                offsets.push(groups.len());
            }
            ReachStorage::Sparse { offsets, groups }
        };
        let positions = spec
            .sets
            .iter()
            .enumerate()
            .map(|(position, (group, _))| (*group, position))
            .collect();
        Ok(ReachIndex {
            identity: Arc::clone(&self.repo.topology_identity),
            node_count: self.node_count(),
            targets: spec.sets,
            positions,
            storage,
        })
    }
}

impl ReachIndex {
    /// Returns the distinct, sorted internal targets in a named group.
    /// Errors if the index belongs to another graph or the group is unknown.
    pub fn targets(&self, graph: &Graph, group: u32) -> Result<&[NodeId]> {
        self.validate_graph(graph)?;
        let position = self
            .positions
            .get(&group)
            .ok_or_else(|| anyhow::anyhow!("unknown reachability group {group}"))?;
        Ok(self.targets[*position].1.as_slice())
    }

    /// Whether the given internal node can reach at least one target in the
    /// group. Returns an error for a different graph, node, or unknown group.
    pub fn reaches(&self, graph: &Graph, group: u32, node: NodeId) -> Result<bool> {
        self.validate_graph(graph)?;
        if node as usize >= self.node_count {
            bail!("unknown internal node {node}");
        }
        let position = *self
            .positions
            .get(&group)
            .ok_or_else(|| anyhow::anyhow!("unknown reachability group {group}"))?;
        Ok(match &self.storage {
            ReachStorage::Bits(bits) => bits[node as usize] & (1_u64 << position) != 0,
            ReachStorage::Sparse { offsets, groups } => {
                let node = node as usize;
                groups[offsets[node]..offsets[node + 1]]
                    .binary_search(&group)
                    .is_ok()
            }
        })
    }

    fn validate_graph(&self, graph: &Graph) -> Result<()> {
        if !Arc::ptr_eq(&self.identity, &graph.repo.topology_identity) {
            bail!("reachability index belongs to a different graph");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Arc,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use arrow::{
        array::UInt64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };

    use super::*;

    fn graph() -> Graph {
        let node_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)]));
        let nodes = RecordBatch::try_new(
            node_schema,
            vec![Arc::new(UInt64Array::from(vec![10, 11, 12, 13, 14, 15]))],
        )
        .unwrap();
        let edge_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("src", DataType::UInt64, false),
            Field::new("dest", DataType::UInt64, false),
        ]));
        // The edge rows are not grouped by source, so internal edge IDs must
        // follow the forward CSR's edge_ids when building reverse adjacency.
        let edges = RecordBatch::try_new(
            edge_schema,
            vec![
                Arc::new(UInt64Array::from(vec![100, 101, 102, 103, 104])),
                Arc::new(UInt64Array::from(vec![11, 10, 13, 11, 12])),
                Arc::new(UInt64Array::from(vec![12, 11, 14, 13, 11])),
            ],
        )
        .unwrap();
        Graph::new(nodes, edges).unwrap()
    }

    #[test]
    fn bits_are_per_group_and_validate_ids() {
        let graph = graph();
        let index = graph
            .reach_index(ReachSpec {
                sets: vec![(42, vec![4, 4]), (9, vec![2]), (0, vec![])],
                ..Default::default()
            })
            .unwrap();
        assert_eq!(graph.internal_node_id(14_u64), Some(4));
        assert_eq!(graph.internal_node_id(99_u64), None);
        assert_eq!(index.targets(&graph, 42).unwrap(), [4].as_slice());
        assert!(index.targets(&graph, 8).is_err());
        assert!(index.targets(&self::graph(), 42).is_err());
        assert!(index.reaches(&graph, 9, 0).unwrap());
        assert!(index.reaches(&graph, 9, 2).unwrap());
        assert!(!index.reaches(&graph, 9, 3).unwrap());
        assert!(index.reaches(&graph, 42, 0).unwrap());
        assert!(!index.reaches(&graph, 42, 4 + 1).unwrap());
        assert!(!index.reaches(&graph, 0, 0).unwrap());
        assert!(index.reaches(&graph, 8, 0).is_err());
        assert!(index.reaches(&graph, 9, 6).is_err());
        assert!(index.reaches(&self::graph(), 9, 0).is_err());
        assert!(
            graph
                .reach_index(ReachSpec {
                    sets: vec![(1, vec![6])],
                    ..Default::default()
                })
                .is_err()
        );
        assert!(
            graph
                .reach_index(ReachSpec {
                    sets: vec![(1, vec![1]), (1, vec![2])],
                    ..Default::default()
                })
                .is_err()
        );
    }

    #[test]
    fn skipped_edges_are_applied_to_the_correct_internal_edge() {
        let graph = graph();
        let calls = AtomicUsize::new(0);
        let skip = |edge| {
            calls.fetch_add(1, Ordering::Relaxed);
            edge == 2
        };
        let index = graph
            .reach_index(ReachSpec {
                sets: vec![(0, vec![4]), (1, vec![2])],
                skip_edge: Some(&skip),
            })
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), graph.edge_count());
        assert!(!index.reaches(&graph, 0, 0).unwrap());
        assert!(index.reaches(&graph, 0, 4).unwrap());
        assert!(index.reaches(&graph, 1, 0).unwrap());
    }

    #[test]
    fn more_than_64_groups_keep_individual_reachability() {
        let graph = graph();
        let sets = (0..80)
            .rev()
            .map(|id| (id * 17, vec![if id % 2 == 0 { 4 } else { 2 }]))
            .collect();
        let index = graph
            .reach_index(ReachSpec {
                sets,
                ..Default::default()
            })
            .unwrap();
        assert!(matches!(index.storage, ReachStorage::Sparse { .. }));
        for id in 0..80 {
            assert_eq!(index.reaches(&graph, id * 17, 3).unwrap(), id % 2 == 0);
            assert!(index.reaches(&graph, id * 17, 2).unwrap());
            assert!(!index.reaches(&graph, id * 17, 5).unwrap());
        }
    }
}
