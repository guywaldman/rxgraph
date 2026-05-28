use std::{
    collections::VecDeque,
    sync::atomic::{AtomicUsize, Ordering},
};

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::{
    dsl::{BoundKernel, EvalCtx, StateValues},
    graph::{EdgeId, Graph, GraphRepo, NodeId, OwnedGraphId},
    traversal::{
        GraphPath, SearchResult, SearchStats,
        config::{TraversalConfig, TraversalStrategy},
    },
};

const MIN_PAR_FRONTIER: usize = 512;
const MIN_PAR_EDGES: usize = 8_192;
const DFS_SEEDS_PER_THREAD: usize = 8;
const MIN_PAR_DFS_PATHS: usize = 64;

impl Graph {
    /// Runs a configured DSL traversal and materializes matching paths.
    ///
    /// Start nodes are resolved from external IDs to compact internal IDs before
    /// search. Returned paths are materialized back to external IDs after `stop`
    /// accepts a path.
    pub fn search(&self, config: TraversalConfig) -> Result<SearchResult<'_>> {
        let TraversalConfig {
            kernel,
            start_nodes,
            max_depth,
            max_paths,
            strategy,
            max_revisits_per_node,
            parallel,
        } = config;
        let kernel = kernel.bind(self)?;
        let cfg = RunConfig {
            start_nodes,
            max_depth: max_depth.unwrap_or(usize::MAX),
            max_paths,
            strategy,
            max_revisits_per_node,
        };

        match (parallel, strategy) {
            (false, _) => search_serial(self, &cfg, &kernel),
            (true, TraversalStrategy::BreadthFirst) => search_bfs_parallel(self, &cfg, &kernel),
            (true, TraversalStrategy::DepthFirst) if should_parallelize_dfs(&cfg) => {
                search_dfs_parallel(self, &cfg, &kernel)
            }
            (true, TraversalStrategy::DepthFirst) => search_serial(self, &cfg, &kernel),
        }
    }
}

struct RunConfig {
    start_nodes: Vec<OwnedGraphId>,
    max_depth: usize,
    max_paths: Option<usize>,
    strategy: TraversalStrategy,
    max_revisits_per_node: usize,
}

#[derive(Debug, Clone)]
struct PathEntry {
    node: NodeId,
    incoming_edge: Option<EdgeId>,
    parent: Option<usize>,
    depth: usize,
    state: StateValues,
}

#[derive(Debug, Clone)]
struct PathTask {
    node: NodeId,
    incoming_edge: Option<EdgeId>,
    parent: Option<usize>,
    depth: usize,
    state: StateValues,
}

struct EdgeEval {
    edge: EdgeId,
    dest: NodeId,
    state: StateValues,
    stop: bool,
}

struct TaskResult<'a> {
    paths: Vec<GraphPath<'a>>,
    stats: SearchStats,
}

fn search_serial<'a>(
    graph: &'a Graph,
    cfg: &RunConfig,
    kernel: &BoundKernel,
) -> Result<SearchResult<'a>> {
    let (mut arena, mut frontier, mut stats) = initial_arena(graph, cfg, kernel)?;
    let mut paths = Vec::new();

    while let Some(parent) = pop(&mut frontier, cfg.strategy) {
        if arena[parent].depth >= cfg.max_depth {
            continue;
        }

        let (edges, dests) = graph.repo.outgoing_slice(arena[parent].node);
        for (&edge, &dest) in edges.iter().zip(dests) {
            let Some(edge) =
                eval_arena_edge(graph, &arena, parent, edge, dest, cfg, kernel, &mut stats)?
            else {
                continue;
            };
            let stop = edge.stop;
            let child = push_entry(&mut arena, parent, edge);

            stats.accepted_edges += 1;
            stats.path_entries += 1;
            stats.max_depth = stats.max_depth.max(arena[child].depth);

            if stop {
                paths.push(materialize(graph, &arena, child)?);
                stats.stopped_paths += 1;
                if cfg.max_paths.is_some_and(|max| paths.len() >= max) {
                    return Ok(SearchResult { paths, stats });
                }
            } else {
                frontier.push_back(child);
            }
        }
    }

    Ok(SearchResult { paths, stats })
}

fn search_bfs_parallel<'a>(
    graph: &'a Graph,
    cfg: &RunConfig,
    kernel: &BoundKernel,
) -> Result<SearchResult<'a>> {
    let (mut arena, frontier, mut stats) = initial_arena(graph, cfg, kernel)?;
    let mut frontier = frontier.into_iter().collect::<Vec<_>>();
    let mut paths = Vec::new();

    while !frontier.is_empty() {
        let edge_count = frontier
            .iter()
            .map(|&p| graph.repo.out_degree(arena[p].node))
            .sum::<usize>();
        let parents = if frontier.len() >= MIN_PAR_FRONTIER || edge_count >= MIN_PAR_EDGES {
            frontier
                .par_iter()
                .map(|&parent| eval_parent(graph, &arena, parent, cfg, kernel))
                .collect::<Result<Vec<_>>>()?
        } else {
            frontier
                .iter()
                .map(|&parent| eval_parent(graph, &arena, parent, cfg, kernel))
                .collect::<Result<Vec<_>>>()?
        };

        let mut next = Vec::new();
        for (parent, edges, local) in parents {
            merge_stats(&mut stats, local);
            for edge in edges {
                let stop = edge.stop;
                let child = push_entry(&mut arena, parent, edge);
                stats.accepted_edges += 1;
                stats.path_entries += 1;
                stats.max_depth = stats.max_depth.max(arena[child].depth);
                if stop {
                    paths.push(materialize(graph, &arena, child)?);
                    stats.stopped_paths += 1;
                } else {
                    next.push(child);
                }
            }
        }

        if let Some(max) = cfg.max_paths
            && paths.len() >= max
        {
            paths.truncate(max);
            return Ok(SearchResult { paths, stats });
        }
        frontier = next;
    }

    Ok(SearchResult { paths, stats })
}

fn search_dfs_parallel<'a>(
    graph: &'a Graph,
    cfg: &RunConfig,
    kernel: &BoundKernel,
) -> Result<SearchResult<'a>> {
    let (queue, mut stats) = initial_tasks(graph, cfg, kernel)?;
    let mut seed_paths = Vec::new();
    let seeds = build_dfs_seeds(graph, cfg, kernel, queue, &mut seed_paths, &mut stats)?;

    if let Some(max) = cfg.max_paths
        && seed_paths.len() >= max
    {
        seed_paths.truncate(max);
        return Ok(SearchResult {
            paths: seed_paths,
            stats,
        });
    }

    let found = AtomicUsize::new(seed_paths.len());
    let results = if seeds.len() < rayon::current_num_threads() {
        seeds
            .into_iter()
            .map(|seed| dfs_seed(graph, cfg, kernel, seed, &found))
            .collect::<Result<Vec<_>>>()?
    } else {
        seeds
            .into_par_iter()
            .map(|seed| dfs_seed(graph, cfg, kernel, seed, &found))
            .collect::<Result<Vec<_>>>()?
    };

    let mut paths = seed_paths;
    for result in results {
        merge_stats(&mut stats, result.stats);
        paths.extend(result.paths);
    }
    if let Some(max) = cfg.max_paths {
        paths.truncate(max);
    }
    Ok(SearchResult { paths, stats })
}

fn initial_arena(
    graph: &Graph,
    cfg: &RunConfig,
    kernel: &BoundKernel,
) -> Result<(Vec<PathEntry>, VecDeque<usize>, SearchStats)> {
    let mut arena = Vec::with_capacity(cfg.start_nodes.len());
    let mut frontier = VecDeque::with_capacity(cfg.start_nodes.len());
    let mut stats = SearchStats::default();

    for external in &cfg.start_nodes {
        let node = graph
            .repo
            .internal_node(external.as_ref())
            .with_context(|| format!("unknown start node {external}"))?;
        frontier.push_back(arena.len());
        arena.push(PathEntry {
            node,
            incoming_edge: None,
            parent: None,
            depth: 0,
            state: kernel.initial_state().clone(),
        });
        stats.start_nodes += 1;
        stats.path_entries += 1;
    }

    Ok((arena, frontier, stats))
}

fn initial_tasks(
    graph: &Graph,
    cfg: &RunConfig,
    kernel: &BoundKernel,
) -> Result<(VecDeque<PathTask>, SearchStats)> {
    let mut queue = VecDeque::with_capacity(cfg.start_nodes.len());
    let mut stats = SearchStats::default();

    for external in &cfg.start_nodes {
        let node = graph
            .repo
            .internal_node(external.as_ref())
            .with_context(|| format!("unknown start node {external}"))?;
        queue.push_back(PathTask {
            node,
            incoming_edge: None,
            parent: None,
            depth: 0,
            state: kernel.initial_state().clone(),
        });
        stats.start_nodes += 1;
        stats.path_entries += 1;
    }

    Ok((queue, stats))
}

fn eval_parent(
    graph: &Graph,
    arena: &[PathEntry],
    parent: usize,
    cfg: &RunConfig,
    kernel: &BoundKernel,
) -> Result<(usize, Vec<EdgeEval>, SearchStats)> {
    let mut stats = SearchStats::default();
    let mut edges = Vec::new();

    if arena[parent].depth < cfg.max_depth {
        let (edge_ids, dests) = graph.repo.outgoing_slice(arena[parent].node);
        for (&edge, &dest) in edge_ids.iter().zip(dests) {
            if let Some(edge) =
                eval_arena_edge(graph, arena, parent, edge, dest, cfg, kernel, &mut stats)?
            {
                edges.push(edge);
            }
        }
    }

    Ok((parent, edges, stats))
}

fn eval_arena_edge(
    graph: &Graph,
    arena: &[PathEntry],
    parent: usize,
    edge: EdgeId,
    dest: NodeId,
    cfg: &RunConfig,
    kernel: &BoundKernel,
    stats: &mut SearchStats,
) -> Result<Option<EdgeEval>> {
    if !can_visit_arena(arena, parent, dest, cfg.max_revisits_per_node) {
        stats.skipped_revisits += 1;
        return Ok(None);
    }

    stats.evaluated_edges += 1;
    let ctx = EvalCtx {
        graph,
        src: arena[parent].node,
        dest,
        edge,
        state: &arena[parent].state,
    };
    if !kernel.visit(&ctx)? {
        stats.rejected_edges += 1;
        return Ok(None);
    }

    let state = kernel.next_state(&arena[parent].state, &ctx)?;
    let stop = kernel.stop(&EvalCtx {
        state: &state,
        ..ctx
    })?;
    Ok(Some(EdgeEval {
        edge,
        dest,
        state,
        stop,
    }))
}

fn push_entry(arena: &mut Vec<PathEntry>, parent: usize, edge: EdgeEval) -> usize {
    let child = arena.len();
    arena.push(PathEntry {
        node: edge.dest,
        incoming_edge: Some(edge.edge),
        parent: Some(parent),
        depth: arena[parent].depth + 1,
        state: edge.state,
    });
    child
}

fn build_dfs_seeds<'a>(
    graph: &'a Graph,
    cfg: &RunConfig,
    kernel: &BoundKernel,
    mut queue: VecDeque<PathTask>,
    paths: &mut Vec<GraphPath<'a>>,
    stats: &mut SearchStats,
) -> Result<Vec<PathTask>> {
    let target = rayon::current_num_threads() * DFS_SEEDS_PER_THREAD;

    while queue.len() < target {
        if cfg.max_paths.is_some_and(|max| paths.len() >= max) {
            break;
        }
        let Some(task) = queue.pop_front() else {
            break;
        };
        if task.depth >= cfg.max_depth {
            continue;
        }

        let mut arena = vec![task];
        let children = expand_task(graph, cfg, kernel, &arena, 0, stats)?;
        for (child, stop) in children {
            let child = push_task(&mut arena, 0, child);
            if stop {
                paths.push(materialize_task(graph, &arena, child)?);
                stats.stopped_paths += 1;
                if cfg.max_paths.is_some_and(|max| paths.len() >= max) {
                    break;
                }
            } else {
                queue.push_back(arena[child].clone());
            }
        }
    }

    Ok(queue.into())
}

fn dfs_seed<'a>(
    graph: &'a Graph,
    cfg: &RunConfig,
    kernel: &BoundKernel,
    seed: PathTask,
    found: &AtomicUsize,
) -> Result<TaskResult<'a>> {
    let mut arena = vec![seed];
    let mut stack = vec![0usize];
    let mut paths = Vec::new();
    let mut stats = SearchStats::default();

    while let Some(task) = stack.pop() {
        if cfg
            .max_paths
            .is_some_and(|max| found.load(Ordering::Relaxed) >= max)
            || arena[task].depth >= cfg.max_depth
        {
            continue;
        }

        let children = expand_task(graph, cfg, kernel, &arena, task, &mut stats)?;
        for (child, stop) in children {
            let child = push_task(&mut arena, task, child);
            if stop {
                let previous = found.fetch_add(1, Ordering::Relaxed);
                stats.stopped_paths += 1;
                if cfg.max_paths.is_none_or(|max| previous < max) {
                    paths.push(materialize_task(graph, &arena, child)?);
                }
            } else {
                stack.push(child);
            }
        }
    }

    Ok(TaskResult { paths, stats })
}

fn expand_task(
    graph: &Graph,
    cfg: &RunConfig,
    kernel: &BoundKernel,
    arena: &[PathTask],
    task: usize,
    stats: &mut SearchStats,
) -> Result<Vec<(EdgeEval, bool)>> {
    let mut children = Vec::new();
    let (edge_ids, dests) = graph.repo.outgoing_slice(arena[task].node);
    for (&edge, &dest) in edge_ids.iter().zip(dests) {
        if !can_visit_task(arena, task, dest, cfg.max_revisits_per_node) {
            stats.skipped_revisits += 1;
            continue;
        }

        stats.evaluated_edges += 1;
        let ctx = EvalCtx {
            graph,
            src: arena[task].node,
            dest,
            edge,
            state: &arena[task].state,
        };
        if !kernel.visit(&ctx)? {
            stats.rejected_edges += 1;
            continue;
        }

        let state = kernel.next_state(&arena[task].state, &ctx)?;
        let stop = kernel.stop(&EvalCtx {
            state: &state,
            ..ctx
        })?;
        stats.accepted_edges += 1;
        stats.path_entries += 1;
        stats.max_depth = stats.max_depth.max(arena[task].depth + 1);
        children.push((
            EdgeEval {
                edge,
                dest,
                state,
                stop,
            },
            stop,
        ));
    }
    Ok(children)
}

fn push_task(arena: &mut Vec<PathTask>, parent: usize, edge: EdgeEval) -> usize {
    let child = arena.len();
    arena.push(PathTask {
        node: edge.dest,
        incoming_edge: Some(edge.edge),
        parent: Some(parent),
        depth: arena[parent].depth + 1,
        state: edge.state,
    });
    child
}

fn should_parallelize_dfs(cfg: &RunConfig) -> bool {
    cfg.max_paths.is_none_or(|max| max >= MIN_PAR_DFS_PATHS)
        && cfg.start_nodes.len() >= rayon::current_num_threads()
}

fn can_visit_arena(
    arena: &[PathEntry],
    mut path: usize,
    node: NodeId,
    max_revisits: usize,
) -> bool {
    let mut visits = 0usize;
    loop {
        if arena[path].node == node {
            visits += 1;
            if visits > max_revisits {
                return false;
            }
        }
        match arena[path].parent {
            Some(parent) => path = parent,
            None => return true,
        }
    }
}

fn can_visit_task(arena: &[PathTask], mut path: usize, node: NodeId, max_revisits: usize) -> bool {
    let mut visits = 0usize;
    loop {
        if arena[path].node == node {
            visits += 1;
            if visits > max_revisits {
                return false;
            }
        }
        match arena[path].parent {
            Some(parent) => path = parent,
            None => return true,
        }
    }
}

fn pop(frontier: &mut VecDeque<usize>, strategy: TraversalStrategy) -> Option<usize> {
    match strategy {
        TraversalStrategy::BreadthFirst => frontier.pop_front(),
        TraversalStrategy::DepthFirst => frontier.pop_back(),
    }
}

fn materialize<'a>(
    graph: &'a Graph,
    arena: &[PathEntry],
    mut path: usize,
) -> Result<GraphPath<'a>> {
    let mut nodes = Vec::with_capacity(arena[path].depth + 1);
    let mut edges = Vec::with_capacity(arena[path].depth);

    loop {
        nodes.push(
            graph
                .repo
                .external_node(arena[path].node)
                .context("path references missing node")?,
        );
        if let Some(edge) = arena[path].incoming_edge {
            edges.push(
                graph
                    .repo
                    .external_edge(edge)
                    .context("path references missing edge")?,
            );
        }
        match arena[path].parent {
            Some(parent) => path = parent,
            None => break,
        }
    }

    nodes.reverse();
    edges.reverse();
    Ok(GraphPath { nodes, edges })
}

fn materialize_task<'a>(
    graph: &'a Graph,
    arena: &[PathTask],
    mut path: usize,
) -> Result<GraphPath<'a>> {
    let mut nodes = Vec::with_capacity(arena[path].depth + 1);
    let mut edges = Vec::with_capacity(arena[path].depth);

    loop {
        nodes.push(
            graph
                .repo
                .external_node(arena[path].node)
                .context("path references missing node")?,
        );
        if let Some(edge) = arena[path].incoming_edge {
            edges.push(
                graph
                    .repo
                    .external_edge(edge)
                    .context("path references missing edge")?,
            );
        }
        match arena[path].parent {
            Some(parent) => path = parent,
            None => break,
        }
    }

    nodes.reverse();
    edges.reverse();
    Ok(GraphPath { nodes, edges })
}

fn merge_stats(into: &mut SearchStats, from: SearchStats) {
    into.start_nodes += from.start_nodes;
    into.path_entries += from.path_entries;
    into.evaluated_edges += from.evaluated_edges;
    into.accepted_edges += from.accepted_edges;
    into.rejected_edges += from.rejected_edges;
    into.skipped_revisits += from.skipped_revisits;
    into.stopped_paths += from.stopped_paths;
    into.max_depth = into.max_depth.max(from.max_depth);
}

#[cfg(test)]
mod tests {
    use arrow::array::record_batch;

    use super::*;
    use crate::{
        dsl::{DslExpr as e, DslKernel, Scalar},
        graph::{EDGE_DEST_COL, EDGE_SRC_COL, GraphId, ID_COL, Repo},
        traversal::config::{TraversalConfigBuilder, TraversalStrategy},
    };

    fn graph() -> Graph {
        Graph {
            repo: Repo::from_tables(
                record_batch!(
                    (ID_COL, Utf8, ["a", "b", "c", "d"]),
                    ("kind", Utf8, ["start", "mid", "mid", "end"])
                )
                .unwrap(),
                record_batch!(
                    (ID_COL, Utf8, ["ab", "ac", "bd", "cd", "ba"]),
                    (EDGE_SRC_COL, Utf8, ["a", "a", "b", "c", "b"]),
                    (EDGE_DEST_COL, Utf8, ["b", "c", "d", "d", "a"]),
                    ("ok", Boolean, [true, false, true, true, true])
                )
                .unwrap(),
            )
            .unwrap(),
        }
    }

    fn branching_graph() -> Graph {
        Graph {
            repo: Repo::from_tables(
                record_batch!(
                    (ID_COL, Utf8, ["s", "a", "b", "c", "d", "e", "f", "z"]),
                    (
                        "kind",
                        Utf8,
                        ["start", "mid", "mid", "mid", "mid", "mid", "mid", "end"]
                    )
                )
                .unwrap(),
                record_batch!(
                    (
                        ID_COL,
                        Utf8,
                        ["sa", "sb", "sc", "ad", "ae", "bf", "cz", "dz", "ez", "fz"]
                    ),
                    (
                        EDGE_SRC_COL,
                        Utf8,
                        ["s", "s", "s", "a", "a", "b", "c", "d", "e", "f"]
                    ),
                    (
                        EDGE_DEST_COL,
                        Utf8,
                        ["a", "b", "c", "d", "e", "f", "z", "z", "z", "z"]
                    ),
                    (
                        "ok",
                        Boolean,
                        [true, true, true, true, true, true, true, true, true, true]
                    )
                )
                .unwrap(),
            )
            .unwrap(),
        }
    }

    fn integer_graph() -> Graph {
        Graph {
            repo: Repo::from_tables(
                record_batch!((ID_COL, UInt64, [1, 2, 3])).unwrap(),
                record_batch!(
                    (ID_COL, UInt64, [10, 20]),
                    (EDGE_SRC_COL, UInt64, [1, 2]),
                    (EDGE_DEST_COL, UInt64, [2, 3])
                )
                .unwrap(),
            )
            .unwrap(),
        }
    }

    fn traversal(visit: e, stop: e) -> TraversalConfig {
        TraversalConfigBuilder::new(DslKernel::new(visit, [], stop, []))
            .with_start_nodes(["a".to_string()])
            .with_strategy(TraversalStrategy::BreadthFirst)
            .with_parallelism(false)
            .build()
    }

    fn path_set(result: &SearchResult<'_>) -> Vec<Vec<String>> {
        let mut paths = result
            .paths
            .iter()
            .map(|p| p.nodes.iter().map(|&n| id_label(n)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        paths.sort();
        paths
    }

    fn id_label(id: GraphId<'_>) -> String {
        match id {
            GraphId::U64(value) => value.to_string(),
            GraphId::Str(value) => value.to_owned(),
        }
    }

    #[test]
    fn returns_stopped_paths() {
        let graph = graph();
        let result = graph
            .search(traversal(
                e::bool(true),
                e::dest("kind").eq(e::string("end")),
            ))
            .unwrap();
        assert_eq!(result.paths.len(), 2);
        assert_eq!(
            result.paths[0].nodes,
            vec![GraphId::Str("a"), GraphId::Str("b"), GraphId::Str("d")]
        );
        assert_eq!(
            result.paths[1].nodes,
            vec![GraphId::Str("a"), GraphId::Str("c"), GraphId::Str("d")]
        );
        assert_eq!(
            result.stats,
            SearchStats {
                start_nodes: 1,
                path_entries: 5,
                evaluated_edges: 4,
                accepted_edges: 4,
                rejected_edges: 0,
                skipped_revisits: 1,
                stopped_paths: 2,
                max_depth: 2,
            }
        );
    }

    #[test]
    fn filters_edges_and_limits_depth() {
        let config =
            TraversalConfigBuilder::new(DslKernel::new(e::edge("ok"), [], e::bool(true), []))
                .with_start_nodes(["a".to_string()])
                .with_max_depth(1)
                .with_parallelism(false)
                .build();

        let graph = graph();
        let result = graph.search(config).unwrap();
        assert_eq!(
            result
                .paths
                .iter()
                .map(|p| p.nodes.clone())
                .collect::<Vec<_>>(),
            vec![vec![GraphId::Str("a"), GraphId::Str("b")]]
        );
        assert_eq!(result.stats.evaluated_edges, 2);
        assert_eq!(result.stats.accepted_edges, 1);
        assert_eq!(result.stats.rejected_edges, 1);
        assert_eq!(result.stats.max_depth, 1);
    }

    #[test]
    fn rejects_revisits_by_default() {
        let graph = graph();
        let result = graph
            .search(traversal(e::bool(true), e::bool(false)))
            .unwrap();
        assert!(result.paths.is_empty());
        assert_eq!(result.stats.skipped_revisits, 1);
        assert_eq!(result.stats.stopped_paths, 0);
    }

    #[test]
    fn reports_unknown_start_node() {
        let config = TraversalConfigBuilder::new(DslKernel::new(
            e::bool(true),
            [],
            e::bool(true),
            [("x".to_string(), Scalar::U64(0))],
        ))
        .with_start_nodes(["missing".to_string()])
        .build();

        assert!(
            graph()
                .search(config)
                .unwrap_err()
                .to_string()
                .contains("unknown start node")
        );
    }

    #[test]
    fn integer_ids_materialize_and_dsl_ids_work() {
        let graph = integer_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(DslKernel::new(
                    e::bool(true),
                    [],
                    e::dest_id().eq(e::uint(3)),
                    [],
                ))
                .with_start_nodes([1u64])
                .with_strategy(TraversalStrategy::BreadthFirst)
                .with_parallelism(false)
                .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 1);
        assert_eq!(
            result.paths[0].nodes,
            vec![GraphId::U64(1), GraphId::U64(2), GraphId::U64(3)]
        );
        assert_eq!(
            result.paths[0].edges,
            vec![GraphId::U64(10), GraphId::U64(20)]
        );
    }

    #[test]
    fn string_special_id_columns_work() {
        let graph = graph();
        let result = graph
            .search(traversal(
                e::edge_id().ne(e::string("ac")),
                e::dest_id().eq(e::string("d")),
            ))
            .unwrap();

        assert_eq!(path_set(&result), vec![vec!["a", "b", "d"]]);
    }

    #[test]
    fn parallel_bfs_matches_serial_path_set() {
        let graph = branching_graph();
        let base = DslKernel::new(e::bool(true), [], e::dest("kind").eq(e::string("end")), []);
        let serial = graph
            .search(
                TraversalConfigBuilder::new(base.clone())
                    .with_start_nodes(["s".to_string()])
                    .with_strategy(TraversalStrategy::BreadthFirst)
                    .with_parallelism(false)
                    .build(),
            )
            .unwrap();
        let parallel = graph
            .search(
                TraversalConfigBuilder::new(base)
                    .with_start_nodes(["s".to_string()])
                    .with_strategy(TraversalStrategy::BreadthFirst)
                    .with_parallelism(true)
                    .build(),
            )
            .unwrap();

        assert_eq!(path_set(&parallel), path_set(&serial));
        assert_eq!(
            parallel.stats.accepted_edges + parallel.stats.rejected_edges,
            parallel.stats.evaluated_edges
        );
        assert_eq!(parallel.stats.stopped_paths, parallel.paths.len());
    }

    #[test]
    fn parallel_bfs_truncates_max_paths() {
        let graph = branching_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(DslKernel::new(
                    e::bool(true),
                    [],
                    e::dest("kind").eq(e::string("end")),
                    [],
                ))
                .with_start_nodes(["s".to_string()])
                .with_strategy(TraversalStrategy::BreadthFirst)
                .with_max_paths(2)
                .with_parallelism(true)
                .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 2);
    }

    #[test]
    fn parallel_dfs_returns_valid_paths() {
        let graph = branching_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(DslKernel::new(
                    e::bool(true),
                    [],
                    e::dest("kind").eq(e::string("end")),
                    [],
                ))
                .with_start_nodes(["s".to_string()])
                .with_strategy(TraversalStrategy::DepthFirst)
                .with_parallelism(true)
                .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 4);
        for path in &result.paths {
            assert_eq!(path.nodes.first(), Some(&GraphId::Str("s")));
            assert_eq!(path.nodes.last(), Some(&GraphId::Str("z")));
        }
        assert_eq!(path_set(&result).len(), 4);
    }

    #[test]
    fn parallel_dfs_soft_stop_returns_exact_max_paths() {
        let graph = branching_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(DslKernel::new(
                    e::bool(true),
                    [],
                    e::dest("kind").eq(e::string("end")),
                    [],
                ))
                .with_start_nodes(["s".to_string()])
                .with_strategy(TraversalStrategy::DepthFirst)
                .with_max_paths(2)
                .with_parallelism(true)
                .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 2);
    }

    #[test]
    fn builder_parallelism_defaults_on_and_can_be_disabled() {
        let kernel = DslKernel::new(e::bool(true), [], e::bool(true), []);
        assert!(TraversalConfigBuilder::new(kernel.clone()).build().parallel);
        assert!(
            !TraversalConfigBuilder::new(kernel)
                .with_parallelism(false)
                .build()
                .parallel
        );
    }
}
