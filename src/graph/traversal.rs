use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

pub type NodeId = u32;

#[derive(Debug)]
pub struct CsrGraph {
    pub offsets: Vec<usize>,
    pub targets: Vec<NodeId>,
    pub edge_weights: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct NodePayload {
    pub value: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NodeState {
    pub depth: u32,
    pub score: u32,
    pub parent: Option<NodeId>,
}

#[derive(Debug)]
pub struct TraversalStats {
    pub visited_nodes: usize,
    pub traversed_edges: usize,
    pub max_depth: u32,
}

pub trait TraversalKernel: Sync {
    fn should_visit(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_weight: u32,
        src_state: NodeState,
        dst_payload: NodePayload,
    ) -> bool;

    fn update_state(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_weight: u32,
        src_state: NodeState,
        dst_payload: NodePayload,
    ) -> NodeState;
}

pub fn traverse_parallel<K>(
    graph: &CsrGraph,
    node_payloads: &[NodePayload],
    starts: &[NodeId],
    kernel: K,
    max_depth: u32,
) -> TraversalStats
where
    K: TraversalKernel,
{
    let node_count = graph.offsets.len() - 1;

    let visited: Vec<AtomicBool> = (0..node_count).map(|_| AtomicBool::new(false)).collect();

    let mut states: Vec<Option<NodeState>> = vec![None; node_count];
    let mut frontier = Vec::from(starts);

    for &start in starts {
        let i = start as usize;

        visited[i].store(true, Ordering::Relaxed);

        states[i] = Some(NodeState {
            depth: 0,
            score: node_payloads[i].value,
            parent: None,
        });
    }

    let mut total_visited = frontier.len();
    let mut total_edges = 0usize;
    let mut depth = 0u32;

    while !frontier.is_empty() && depth < max_depth {
        let discovered: Vec<(NodeId, NodeState, usize)> = frontier
            .par_iter()
            .flat_map_iter(|&src| {
                let src_idx = src as usize;
                let src_state = states[src_idx].unwrap();

                let start = graph.offsets[src_idx];
                let end = graph.offsets[src_idx + 1];

                let mut local = Vec::new();

                for edge_idx in start..end {
                    let dst = graph.targets[edge_idx];
                    let dst_idx = dst as usize;
                    let edge_weight = graph.edge_weights[edge_idx];
                    let dst_payload = node_payloads[dst_idx];

                    if !kernel.should_visit(src, dst, edge_weight, src_state, dst_payload) {
                        continue;
                    }

                    let claimed = visited[dst_idx]
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok();

                    if claimed {
                        let new_state =
                            kernel.update_state(src, dst, edge_weight, src_state, dst_payload);

                        local.push((dst, new_state, end - start));
                    }
                }

                local.into_iter()
            })
            .collect();

        total_edges += frontier
            .iter()
            .map(|&src| {
                let i = src as usize;
                graph.offsets[i + 1] - graph.offsets[i]
            })
            .sum::<usize>();

        frontier.clear();

        for (dst, state, _) in discovered {
            states[dst as usize] = Some(state);
            frontier.push(dst);
        }

        total_visited += frontier.len();
        depth += 1;
    }

    TraversalStats {
        visited_nodes: total_visited,
        traversed_edges: total_edges,
        max_depth: depth,
    }
}

// TODO: Only used for benchmarking
pub fn graph_memory_bytes(graph: &CsrGraph, payloads: &[NodePayload]) -> usize {
    graph.offsets.len() * std::mem::size_of::<usize>()
        + graph.targets.len() * std::mem::size_of::<NodeId>()
        + graph.edge_weights.len() * std::mem::size_of::<u32>()
        + std::mem::size_of_val(payloads)
}
