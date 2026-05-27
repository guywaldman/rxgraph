//! Traversal configuration and execution.
//!
//! DFS is optimized for early path discovery and remains serial. BFS runs
//! layer-by-layer and can evaluate large layers in parallel with Rayon while
//! merging results in deterministic parent/edge order.

use std::collections::VecDeque;

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde_json::Value as JsonValue;

use crate::{
    Graph,
    dsl::{BoundKernel, DslKernel, EvalCtx, StateRow, state_to_value},
    graph::{EdgeId, NodeId},
};

/// Search order used by a traversal.
#[derive(Debug, Clone, Copy, Default)]
pub enum TraversalStrategy {
    /// Expand all paths at depth `n` before depth `n + 1`.
    #[default]
    BreadthFirst,
    /// Follow the newest accepted path first.
    DepthFirst,
}

const DEFAULT_PARALLEL_MIN_FRONTIER: usize = 512;
const DEFAULT_PARALLEL_MIN_EDGES: usize = 8_192;
const DEFAULT_PARALLEL_MIN_REMAINING_PATHS: usize = 64;

/// Controls whether BFS layers may be evaluated in parallel.
///
/// Parallelism applies only to breadth-first traversal. Depth-first traversal is
/// intentionally serial because its early-stop behavior and stack order are the
/// main reason it is fast for many path-finding workloads.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Parallelism {
    /// Let the engine decide from frontier size, outgoing edge count, and
    /// remaining path budget.
    #[default]
    Auto,
    /// Always use the serial BFS layer evaluator.
    Disabled,
    /// Use the parallel BFS layer evaluator once either threshold is met.
    Enabled {
        /// Minimum number of frontier path entries in a layer.
        min_frontier: usize,
        /// Minimum number of outgoing edges in a layer.
        min_edges: usize,
    },
}

impl Parallelism {
    fn should_parallelize(
        self,
        frontier_len: usize,
        edge_count: usize,
        remaining_paths: Option<usize>,
    ) -> bool {
        match self {
            Self::Disabled => false,
            Self::Auto => {
                remaining_paths
                    .is_none_or(|remaining| remaining >= DEFAULT_PARALLEL_MIN_REMAINING_PATHS)
                    && (frontier_len >= DEFAULT_PARALLEL_MIN_FRONTIER
                        || edge_count >= DEFAULT_PARALLEL_MIN_EDGES)
            }
            Self::Enabled {
                min_frontier,
                min_edges,
            } => frontier_len >= min_frontier || edge_count >= min_edges,
        }
    }
}

/// One materialized path returned by a traversal.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchPath {
    /// External node IDs in path order, including the start and final node.
    pub nodes: Vec<u64>,
    /// Edge IDs in path order.
    pub edges: Vec<EdgeId>,
    /// Final traversal state after the path's stopping edge.
    pub state: JsonValue,
}

/// Counters collected during traversal.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Number of path entries allocated in the traversal arena.
    pub visited_path_entries: usize,
    /// Number of outgoing edges evaluated by the traversal.
    pub evaluated_edges: usize,
    /// Number of edges accepted by the visit predicate.
    pub accepted_edges: usize,
    /// Number of accepted edges that satisfied the stop predicate.
    pub stopped_paths: usize,
    /// Reserved for future error-skipping modes. Current searches fail fast, so
    /// this remains zero.
    pub skipped_errors: usize,
    /// Deepest accepted path depth.
    pub max_depth: usize,
}

/// Result of a graph traversal.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Materialized stopped paths, in deterministic traversal order.
    pub paths: Vec<SearchPath>,
    /// Traversal counters.
    pub stats: SearchStats,
}

/// Fully configured DSL traversal.
#[derive(Debug, Clone)]
pub struct DslTraversal {
    pub(crate) kernel: DslKernel,
    pub(crate) start_nodes: Vec<u64>,
    pub(crate) max_depth: Option<usize>,
    pub(crate) max_paths: Option<usize>,
    pub(crate) strategy: TraversalStrategy,
    pub(crate) max_revisits_per_node: usize,
    pub(crate) parallelism: Parallelism,
}

/// Builder for a [`DslTraversal`].
///
/// The builder defaults to depth-first traversal, no depth/path limits, no node
/// revisits inside a path, and automatic BFS parallelism. Automatic parallelism
/// has no effect unless the strategy is breadth-first.
#[derive(Debug)]
pub struct DslTraversalBuilder {
    kernel: DslKernel,
    start_nodes: Vec<u64>,
    max_depth: Option<usize>,
    max_paths: Option<usize>,
    strategy: TraversalStrategy,
    max_revisits_per_node: usize,
    parallelism: Parallelism,
}

impl DslTraversalBuilder {
    /// Starts a traversal builder for `kernel`.
    pub fn new(kernel: DslKernel) -> Self {
        Self {
            kernel,
            start_nodes: Vec::new(),
            max_depth: None,
            max_paths: None,
            strategy: TraversalStrategy::DepthFirst,
            max_revisits_per_node: 0,
            parallelism: Parallelism::Auto,
        }
    }

    /// Sets external node IDs where traversal begins.
    pub fn with_start_nodes(mut self, nodes: impl IntoIterator<Item = u64>) -> Self {
        self.start_nodes = nodes.into_iter().collect();
        self
    }

    /// Limits the number of accepted edges in any path.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Limits the number of materialized stopped paths.
    pub fn with_max_paths(mut self, paths: usize) -> Self {
        self.max_paths = Some(paths);
        self
    }

    /// Chooses depth-first or breadth-first traversal.
    pub fn with_strategy(mut self, strategy: TraversalStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Configures BFS parallelism.
    pub fn with_parallelism(mut self, parallelism: Parallelism) -> Self {
        self.parallelism = parallelism;
        self
    }

    /// Forces parallel BFS mode with explicit layer thresholds.
    pub fn with_parallel_thresholds(mut self, min_frontier: usize, min_edges: usize) -> Self {
        self.parallelism = Parallelism::Enabled {
            min_frontier,
            min_edges,
        };
        self
    }

    /// Builds the immutable traversal configuration.
    pub fn build(self) -> DslTraversal {
        DslTraversal {
            kernel: self.kernel,
            start_nodes: self.start_nodes,
            max_depth: self.max_depth,
            max_paths: self.max_paths,
            strategy: self.strategy,
            max_revisits_per_node: self.max_revisits_per_node,
            parallelism: self.parallelism,
        }
    }
}

#[derive(Debug, Clone)]
struct PathEntry {
    node: NodeId,
    incoming_edge: Option<EdgeId>,
    parent: Option<usize>,
    depth: usize,
    state: StateRow,
}

#[derive(Debug)]
struct ParentResult {
    decisions: Vec<Result<EdgeDecision>>,
}

#[derive(Debug, Clone, Copy)]
struct EdgeTask {
    parent_index: usize,
    path_id: usize,
    edge: EdgeId,
    dest: NodeId,
}

#[derive(Debug)]
enum EdgeDecision {
    Rejected,
    Accepted {
        edge: EdgeId,
        dest: NodeId,
        state: StateRow,
        stop: bool,
    },
}

pub(crate) fn search(graph: &Graph, traversal: DslTraversal) -> Result<SearchResult> {
    let kernel = traversal.kernel.clone().bind(graph)?;
    match traversal.strategy {
        TraversalStrategy::BreadthFirst => search_breadth_first(graph, traversal, kernel),
        TraversalStrategy::DepthFirst => search_depth_first(graph, traversal, kernel),
    }
}

fn search_depth_first(
    graph: &Graph,
    traversal: DslTraversal,
    kernel: BoundKernel,
) -> Result<SearchResult> {
    let mut arena = Vec::new();
    let mut frontier = VecDeque::new();

    for &external in &traversal.start_nodes {
        let node = graph
            .internal_node(external)
            .with_context(|| format!("start node {external} does not exist"))?;
        let path = arena.len();

        arena.push(PathEntry {
            node,
            incoming_edge: None,
            parent: None,
            depth: 0,
            state: kernel.initial_state().clone(),
        });
        frontier.push_back(path);
    }

    let max_depth = traversal.max_depth.unwrap_or(usize::MAX);
    let mut paths = Vec::new();
    let mut stats = SearchStats {
        visited_path_entries: arena.len(),
        ..SearchStats::default()
    };

    while let Some(path_id) = pop_frontier(&mut frontier, traversal.strategy) {
        let src = arena[path_id].node;
        let depth = arena[path_id].depth;

        if depth >= max_depth {
            continue;
        }

        for (edge, dest) in graph.outgoing(src) {
            stats.evaluated_edges += 1;

            if !can_visit_node(&arena, path_id, dest, traversal.max_revisits_per_node) {
                continue;
            }

            let ctx = EvalCtx {
                graph,
                src,
                dest,
                edge,
                state: &arena[path_id].state,
            };

            if !kernel.visit(&ctx)? {
                continue;
            }

            let next_state = kernel.next_state(&arena[path_id].state, &ctx)?;
            let stop_ctx = EvalCtx {
                graph,
                src,
                dest,
                edge,
                state: &next_state,
            };
            let stop = kernel.stop(&stop_ctx)?;

            let child = arena.len();
            let depth = depth + 1;
            arena.push(PathEntry {
                node: dest,
                incoming_edge: Some(edge),
                parent: Some(path_id),
                depth,
                state: next_state,
            });
            stats.accepted_edges += 1;
            stats.visited_path_entries += 1;
            stats.max_depth = stats.max_depth.max(depth);

            if stop {
                stats.stopped_paths += 1;
                paths.push(materialize_path(graph, &arena, child));

                if traversal
                    .max_paths
                    .is_some_and(|max_paths| paths.len() >= max_paths)
                {
                    return Ok(SearchResult { paths, stats });
                }
            } else {
                frontier.push_back(child);
            }
        }
    }

    Ok(SearchResult { paths, stats })
}

fn search_breadth_first(
    graph: &Graph,
    traversal: DslTraversal,
    kernel: BoundKernel,
) -> Result<SearchResult> {
    let mut arena = Vec::new();
    let mut frontier = Vec::new();

    for &external in &traversal.start_nodes {
        let node = graph
            .internal_node(external)
            .with_context(|| format!("start node {external} does not exist"))?;
        let path = arena.len();

        arena.push(PathEntry {
            node,
            incoming_edge: None,
            parent: None,
            depth: 0,
            state: kernel.initial_state().clone(),
        });
        frontier.push(path);
    }

    let max_depth = traversal.max_depth.unwrap_or(usize::MAX);
    let mut paths = Vec::new();
    let mut stats = SearchStats {
        visited_path_entries: arena.len(),
        ..SearchStats::default()
    };

    while !frontier.is_empty() {
        let edge_count = frontier
            .iter()
            .map(|&path_id| graph.out_degree(arena[path_id].node))
            .sum();
        let use_parallel = traversal.parallelism.should_parallelize(
            frontier.len(),
            edge_count,
            traversal
                .max_paths
                .map(|max_paths| max_paths.saturating_sub(paths.len())),
        );
        let layer = if use_parallel {
            evaluate_layer_parallel(graph, &arena, &frontier, max_depth, &kernel, &traversal)
        } else {
            evaluate_layer_serial(graph, &arena, &frontier, max_depth, &kernel, &traversal)
        };

        frontier.clear();

        for (parent_id, parent) in layer.into_iter() {
            let depth = arena[parent_id].depth + 1;

            for decision in parent.decisions {
                stats.evaluated_edges += 1;

                let EdgeDecision::Accepted {
                    edge,
                    dest,
                    state,
                    stop,
                } = decision?
                else {
                    continue;
                };

                let child = arena.len();
                arena.push(PathEntry {
                    node: dest,
                    incoming_edge: Some(edge),
                    parent: Some(parent_id),
                    depth,
                    state,
                });
                stats.accepted_edges += 1;
                stats.visited_path_entries += 1;
                stats.max_depth = stats.max_depth.max(depth);

                if stop {
                    stats.stopped_paths += 1;
                    paths.push(materialize_path(graph, &arena, child));

                    if traversal
                        .max_paths
                        .is_some_and(|max_paths| paths.len() >= max_paths)
                    {
                        return Ok(SearchResult { paths, stats });
                    }
                } else {
                    frontier.push(child);
                }
            }
        }
    }

    Ok(SearchResult { paths, stats })
}

fn evaluate_layer_serial(
    graph: &Graph,
    arena: &[PathEntry],
    frontier: &[usize],
    max_depth: usize,
    kernel: &BoundKernel,
    traversal: &DslTraversal,
) -> Vec<(usize, ParentResult)> {
    frontier
        .iter()
        .map(|&path_id| {
            (
                path_id,
                evaluate_parent(graph, arena, path_id, max_depth, kernel, traversal),
            )
        })
        .collect()
}

fn evaluate_layer_parallel(
    graph: &Graph,
    arena: &[PathEntry],
    frontier: &[usize],
    max_depth: usize,
    kernel: &BoundKernel,
    traversal: &DslTraversal,
) -> Vec<(usize, ParentResult)> {
    // Flatten the layer into edge-sized tasks so a single high-fanout parent
    // can still saturate the Rayon pool. Results are merged back by parent
    // index, preserving frontier order and each parent's outgoing edge order.
    let parents = frontier
        .iter()
        .map(|&path_id| {
            let degree = if arena[path_id].depth >= max_depth {
                0
            } else {
                graph.out_degree(arena[path_id].node)
            };
            (path_id, degree)
        })
        .collect::<Vec<_>>();
    let edge_count = parents.iter().map(|(_, degree)| degree).sum();
    let mut tasks = Vec::with_capacity(edge_count);

    for (parent_index, &(path_id, degree)) in parents.iter().enumerate() {
        if degree == 0 {
            continue;
        }

        let src = arena[path_id].node;
        tasks.extend(graph.outgoing(src).map(|(edge, dest)| EdgeTask {
            parent_index,
            path_id,
            edge,
            dest,
        }));
    }

    let decisions = tasks
        .par_iter()
        .map(|task| {
            (
                task.parent_index,
                evaluate_edge(
                    graph,
                    arena,
                    task.path_id,
                    task.edge,
                    task.dest,
                    kernel,
                    traversal,
                ),
            )
        })
        .collect::<Vec<_>>();
    let mut layer = parents
        .into_iter()
        .map(|(path_id, degree)| {
            (
                path_id,
                ParentResult {
                    decisions: Vec::with_capacity(degree),
                },
            )
        })
        .collect::<Vec<_>>();

    for (parent_index, decision) in decisions {
        layer[parent_index].1.decisions.push(decision);
    }

    layer
}

fn evaluate_parent(
    graph: &Graph,
    arena: &[PathEntry],
    path_id: usize,
    max_depth: usize,
    kernel: &BoundKernel,
    traversal: &DslTraversal,
) -> ParentResult {
    let src = arena[path_id].node;
    let depth = arena[path_id].depth;

    if depth >= max_depth {
        return ParentResult {
            decisions: Vec::new(),
        };
    }

    let mut decisions = Vec::with_capacity(graph.out_degree(src));

    for (edge, dest) in graph.outgoing(src) {
        let decision = evaluate_edge(graph, arena, path_id, edge, dest, kernel, traversal);
        let stop = decision.is_err();
        decisions.push(decision);

        if stop {
            break;
        }
    }

    ParentResult { decisions }
}

fn evaluate_edge(
    graph: &Graph,
    arena: &[PathEntry],
    path_id: usize,
    edge: EdgeId,
    dest: NodeId,
    kernel: &BoundKernel,
    traversal: &DslTraversal,
) -> Result<EdgeDecision> {
    if !can_visit_node(arena, path_id, dest, traversal.max_revisits_per_node) {
        return Ok(EdgeDecision::Rejected);
    }

    let src = arena[path_id].node;
    let ctx = EvalCtx {
        graph,
        src,
        dest,
        edge,
        state: &arena[path_id].state,
    };

    if !kernel.visit(&ctx)? {
        return Ok(EdgeDecision::Rejected);
    }

    let state = kernel.next_state(&arena[path_id].state, &ctx)?;
    let stop_ctx = EvalCtx {
        graph,
        src,
        dest,
        edge,
        state: &state,
    };
    let stop = kernel.stop(&stop_ctx)?;

    Ok(EdgeDecision::Accepted {
        edge,
        dest,
        state,
        stop,
    })
}

fn pop_frontier(frontier: &mut VecDeque<usize>, strategy: TraversalStrategy) -> Option<usize> {
    match strategy {
        TraversalStrategy::DepthFirst => frontier.pop_back(),
        TraversalStrategy::BreadthFirst => frontier.pop_front(),
    }
}

fn can_visit_node(
    arena: &[PathEntry],
    mut path_id: usize,
    node: NodeId,
    max_revisits: usize,
) -> bool {
    let mut visits = 0usize;

    loop {
        let entry = &arena[path_id];

        if entry.node == node {
            visits += 1;
            if visits > max_revisits {
                return false;
            }
        }

        if let Some(parent) = entry.parent {
            path_id = parent;
        } else {
            return true;
        }
    }
}

fn materialize_path(graph: &Graph, arena: &[PathEntry], path_id: usize) -> SearchPath {
    // Path entries form a parent-linked arena. Materialization walks backward
    // from the stopped child and reverses once, avoiding per-step front inserts.
    let depth = arena[path_id].depth;
    let mut nodes = Vec::with_capacity(depth + 1);
    let mut edges = Vec::with_capacity(depth);
    let mut current = Some(path_id);
    let state = state_to_value(&arena[path_id].state);

    while let Some(path_id) = current {
        let entry = &arena[path_id];
        nodes.push(graph.external_node(entry.node));
        if let Some(edge) = entry.incoming_edge {
            edges.push(edge);
        }
        current = entry.parent;
    }

    nodes.reverse();
    edges.reverse();

    SearchPath {
        nodes,
        edges,
        state,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphBuilder, Scalar, test_utils::edges, test_utils::nodes};

    #[test]
    fn parallel_breadth_first_preserves_max_paths_order_and_stats() {
        let graph = GraphBuilder::new()
            .with_node_table(
                "n",
                nodes(&[0, 1, 2, 3], &["n", "n", "n", "n"], &[0, 0, 0, 0]),
            )
            .with_edge_table(
                "e",
                edges(&[0, 0, 1, 2], &[1, 2, 3, 3], &["e", "e", "e", "e"]),
            )
            .build()
            .unwrap();
        let kernel = DslKernel::new(
            r#"{"Literal":{"Scalar":{"Boolean":true}}}"#,
            [],
            r#"{"BinaryExpr":{"left":{"Column":"dest.id"},"op":"Eq","right":{"Literal":{"Dyn":{"UInt":3}}}}}"#,
            [("seen".to_string(), Scalar::U64(0))],
        )
        .unwrap();
        let serial = DslTraversalBuilder::new(kernel.clone())
            .with_start_nodes([0])
            .with_max_depth(2)
            .with_max_paths(1)
            .with_strategy(TraversalStrategy::BreadthFirst)
            .with_parallelism(Parallelism::Disabled)
            .build();
        let parallel = DslTraversalBuilder::new(kernel)
            .with_start_nodes([0])
            .with_max_depth(2)
            .with_max_paths(1)
            .with_strategy(TraversalStrategy::BreadthFirst)
            .with_parallel_thresholds(0, 0)
            .build();

        assert_eq!(
            graph.search(serial).unwrap(),
            graph.search(parallel).unwrap()
        );
    }

    #[test]
    fn rejects_field_missing_from_all_tables_before_traversal() {
        let graph = GraphBuilder::new()
            .with_node_table("n", nodes(&[0, 1], &["n", "n"], &[0, 0]))
            .with_edge_table("e", edges(&[0], &[1], &["e"]))
            .build()
            .unwrap();
        let kernel = DslKernel::new(
            r#"{"Column":"edge.price"}"#,
            [],
            r#"{"Literal":{"Scalar":{"Boolean":false}}}"#,
            [],
        )
        .unwrap();
        let traversal = DslTraversalBuilder::new(kernel)
            .with_start_nodes([0])
            .with_max_depth(1)
            .build();

        let err = graph.search(traversal).unwrap_err();
        assert!(
            err.to_string()
                .contains("column \"price\" is not present in any edge table")
        );
    }
}
