use std::{
    collections::{HashMap, VecDeque},
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
        progress::Progress,
    },
};

const MIN_PAR_FRONTIER: usize = 512;
const MIN_PAR_EDGES: usize = 8_192;
const DFS_SEEDS_PER_THREAD: usize = 8;
const MIN_PAR_DFS_PATHS: usize = 64;

type VisitCounts = HashMap<NodeId, usize>;

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
            intermediate_states,
            progress,
        } = config;
        let kernel = kernel.bind(self)?;
        let cfg = RunConfig {
            start_nodes,
            max_depth: max_depth.unwrap_or(usize::MAX),
            max_paths,
            strategy,
            max_revisits_per_node,
            intermediate_states,
            progress,
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
    intermediate_states: bool,
    progress: bool,
}

#[derive(Debug, Clone)]
struct PathEntry {
    node: NodeId,
    incoming_edge: Option<EdgeId>,
    parent: Option<usize>,
    depth: usize,
    state: StateValues,
}

/// One node in a local DFS path arena.
#[derive(Debug, Clone)]
struct PathTask {
    node: NodeId,
    incoming_edge: Option<EdgeId>,
    /// Index into the arena of the parent task, if there is such.
    parent: Option<usize>,
    depth: usize,
    state: StateValues,
}

/// Independent DFS subtree work item.
/// Carries its own arena so parent links stay valid.
/// Important specifically for parallel DFS to avoid synchronization on
/// a shared arena and frontier.
struct DfsSeed {
    arena: Vec<PathTask>,
    task: usize,
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
    let mut progress = Progress::new(cfg.progress);

    while let Some(parent) = pop(&mut frontier, cfg.strategy) {
        progress.tick(&stats);
        if arena[parent].depth >= cfg.max_depth {
            continue;
        }

        let (edges, dests) = graph.repo.outgoing_slice(arena[parent].node);
        let visit_counts = visit_counts_arena(&arena, parent, edges.len());
        for (&edge, &dest) in edges.iter().zip(dests) {
            let Some(edge) = eval_arena_edge(
                graph,
                &arena,
                parent,
                edge,
                dest,
                cfg,
                kernel,
                &mut stats,
                visit_counts.as_ref(),
            )?
            else {
                continue;
            };
            let stop = edge.stop;
            let child = push_entry(&mut arena, parent, edge);

            stats.accepted_edges += 1;
            stats.path_entries += 1;
            stats.max_depth = stats.max_depth.max(arena[child].depth);

            if stop {
                paths.push(materialize(graph, &arena, child, cfg, kernel)?);
                stats.stopped_paths += 1;
                if cfg.max_paths.is_some_and(|max| paths.len() >= max) {
                    progress.finish(&stats);
                    return Ok(SearchResult { paths, stats });
                }
            } else {
                frontier.push_back(child);
            }
        }
    }

    progress.finish(&stats);
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
    let mut progress = Progress::new(cfg.progress);

    while !frontier.is_empty() {
        progress.tick(&stats);
        let edge_count = frontier
            .iter()
            .map(|&p| graph.repo.out_degree(arena[p].node))
            .sum::<usize>();
        let (edges, local) = if frontier.len() >= MIN_PAR_FRONTIER || edge_count >= MIN_PAR_EDGES {
            eval_frontier_parallel(graph, &arena, &frontier, cfg, kernel)?
        } else {
            eval_frontier_serial(graph, &arena, &frontier, edge_count, cfg, kernel)?
        };
        merge_stats(&mut stats, local);

        let mut next = Vec::with_capacity(edges.len());
        for (parent, edge) in edges {
            let stop = edge.stop;
            let child = push_entry(&mut arena, parent, edge);
            stats.accepted_edges += 1;
            stats.path_entries += 1;
            stats.max_depth = stats.max_depth.max(arena[child].depth);
            if stop {
                paths.push(materialize(graph, &arena, child, cfg, kernel)?);
                stats.stopped_paths += 1;
            } else {
                next.push(child);
            }
        }

        if let Some(max) = cfg.max_paths
            && paths.len() >= max
        {
            paths.truncate(max);
            progress.finish(&stats);
            return Ok(SearchResult { paths, stats });
        }
        frontier = next;
    }

    progress.finish(&stats);
    Ok(SearchResult { paths, stats })
}

fn search_dfs_parallel<'a>(
    graph: &'a Graph,
    cfg: &RunConfig,
    kernel: &BoundKernel,
) -> Result<SearchResult<'a>> {
    let (queue, mut stats) = initial_tasks(graph, cfg, kernel)?;
    let mut seed_paths = Vec::new();
    let mut progress = Progress::new(cfg.progress);
    progress.tick(&stats);
    let seeds = build_dfs_seeds(graph, cfg, kernel, queue, &mut seed_paths, &mut stats)?;
    progress.tick(&stats);

    if let Some(max) = cfg.max_paths
        && seed_paths.len() >= max
    {
        seed_paths.truncate(max);
        progress.finish(&stats);
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
    progress.finish(&stats);
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
) -> Result<(VecDeque<DfsSeed>, SearchStats)> {
    let mut queue = VecDeque::with_capacity(cfg.start_nodes.len());
    let mut stats = SearchStats::default();

    for external in &cfg.start_nodes {
        let node = graph
            .repo
            .internal_node(external.as_ref())
            .with_context(|| format!("unknown start node {external}"))?;
        queue.push_back(DfsSeed {
            arena: vec![PathTask {
                node,
                incoming_edge: None,
                parent: None,
                depth: 0,
                state: kernel.initial_state().clone(),
            }],
            task: 0,
        });
        stats.start_nodes += 1;
        stats.path_entries += 1;
    }

    Ok((queue, stats))
}

fn eval_frontier_serial(
    graph: &Graph,
    arena: &[PathEntry],
    frontier: &[usize],
    edge_count: usize,
    cfg: &RunConfig,
    kernel: &BoundKernel,
) -> Result<(Vec<(usize, EdgeEval)>, SearchStats)> {
    let mut edges = Vec::with_capacity(edge_count);
    let mut stats = SearchStats::default();

    for &parent in frontier {
        let local = eval_parent_into(graph, arena, parent, cfg, kernel, &mut edges)?;
        merge_stats(&mut stats, local);
    }

    Ok((edges, stats))
}

fn eval_frontier_parallel(
    graph: &Graph,
    arena: &[PathEntry],
    frontier: &[usize],
    cfg: &RunConfig,
    kernel: &BoundKernel,
) -> Result<(Vec<(usize, EdgeEval)>, SearchStats)> {
    frontier
        .par_iter()
        .try_fold(
            || (Vec::new(), SearchStats::default()),
            |(mut edges, mut stats), &parent| {
                let local = eval_parent_into(graph, arena, parent, cfg, kernel, &mut edges)?;
                merge_stats(&mut stats, local);
                Ok((edges, stats))
            },
        )
        .try_reduce(
            || (Vec::new(), SearchStats::default()),
            |(mut left_edges, mut left_stats), (right_edges, right_stats)| {
                left_edges.extend(right_edges);
                merge_stats(&mut left_stats, right_stats);
                Ok((left_edges, left_stats))
            },
        )
}

fn eval_parent_into(
    graph: &Graph,
    arena: &[PathEntry],
    parent: usize,
    cfg: &RunConfig,
    kernel: &BoundKernel,
    out: &mut Vec<(usize, EdgeEval)>,
) -> Result<SearchStats> {
    let mut stats = SearchStats::default();

    if arena[parent].depth < cfg.max_depth {
        let (edge_ids, dests) = graph.repo.outgoing_slice(arena[parent].node);
        out.reserve(edge_ids.len());
        let visit_counts = visit_counts_arena(arena, parent, edge_ids.len());
        for (&edge, &dest) in edge_ids.iter().zip(dests) {
            if let Some(edge) = eval_arena_edge(
                graph,
                arena,
                parent,
                edge,
                dest,
                cfg,
                kernel,
                &mut stats,
                visit_counts.as_ref(),
            )? {
                out.push((parent, edge));
            }
        }
    }

    Ok(stats)
}

#[allow(clippy::too_many_arguments)]
fn eval_arena_edge(
    graph: &Graph,
    arena: &[PathEntry],
    parent: usize,
    edge: EdgeId,
    dest: NodeId,
    cfg: &RunConfig,
    kernel: &BoundKernel,
    stats: &mut SearchStats,
    visit_counts: Option<&VisitCounts>,
) -> Result<Option<EdgeEval>> {
    if !can_visit_arena(arena, parent, dest, cfg.max_revisits_per_node, visit_counts) {
        stats.skipped_revisits += 1;
        return Ok(None);
    }

    stats.evaluated_edges += 1;
    let ctx = EvalCtx::new(graph, arena[parent].node, dest, edge, &arena[parent].state);
    if !kernel.visit(&ctx)? {
        stats.rejected_edges += 1;
        return Ok(None);
    }

    let state = kernel.next_state(&arena[parent].state, &ctx)?;
    let stop = kernel.stop(&ctx.with_state(&state))?;
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
    mut queue: VecDeque<DfsSeed>,
    paths: &mut Vec<GraphPath<'a>>,
    stats: &mut SearchStats,
) -> Result<Vec<DfsSeed>> {
    let target = rayon::current_num_threads() * DFS_SEEDS_PER_THREAD;

    while queue.len() < target {
        if cfg.max_paths.is_some_and(|max| paths.len() >= max) {
            break;
        }
        let Some(seed) = queue.pop_front() else {
            break;
        };
        let mut arena = seed.arena;
        let task = seed.task;
        if arena[task].depth >= cfg.max_depth {
            continue;
        }

        let children = expand_task(graph, cfg, kernel, &arena, task, stats)?;
        for (child, stop) in children {
            let child = push_task(&mut arena, task, child);
            if stop {
                paths.push(materialize_task(graph, &arena, child, cfg, kernel)?);
                stats.stopped_paths += 1;
                if cfg.max_paths.is_some_and(|max| paths.len() >= max) {
                    break;
                }
            } else {
                queue.push_back(standalone_seed(&arena, child));
            }
        }
    }

    Ok(queue.into())
}

fn dfs_seed<'a>(
    graph: &'a Graph,
    cfg: &RunConfig,
    kernel: &BoundKernel,
    seed: DfsSeed,
    found: &AtomicUsize,
) -> Result<TaskResult<'a>> {
    let mut arena = seed.arena;
    let mut stack = vec![seed.task];
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
                    paths.push(materialize_task(graph, &arena, child, cfg, kernel)?);
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
    let (edge_ids, dests) = graph.repo.outgoing_slice(arena[task].node);
    let mut children = Vec::with_capacity(edge_ids.len());
    let visit_counts = visit_counts_task(arena, task, edge_ids.len());
    for (&edge, &dest) in edge_ids.iter().zip(dests) {
        if !can_visit_task(
            arena,
            task,
            dest,
            cfg.max_revisits_per_node,
            visit_counts.as_ref(),
        ) {
            stats.skipped_revisits += 1;
            continue;
        }

        stats.evaluated_edges += 1;
        let ctx = EvalCtx::new(graph, arena[task].node, dest, edge, &arena[task].state);
        if !kernel.visit(&ctx)? {
            stats.rejected_edges += 1;
            continue;
        }

        let state = kernel.next_state(&arena[task].state, &ctx)?;
        let stop = kernel.stop(&ctx.with_state(&state))?;
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

/// Copies one path chain into a new arena so the resulting seed has valid parent indexes.
fn standalone_seed(arena: &[PathTask], mut path: usize) -> DfsSeed {
    let mut chain = Vec::with_capacity(arena[path].depth + 1);
    loop {
        chain.push(path);
        match arena[path].parent {
            Some(parent) => path = parent,
            None => break,
        }
    }
    chain.reverse();

    let mut seed_arena = Vec::with_capacity(chain.len());
    for index in chain {
        let parent = if seed_arena.is_empty() {
            None
        } else {
            Some(seed_arena.len() - 1)
        };
        let task = &arena[index];
        seed_arena.push(PathTask {
            node: task.node,
            incoming_edge: task.incoming_edge,
            parent,
            depth: task.depth,
            state: task.state.clone(),
        });
    }

    DfsSeed {
        task: seed_arena.len() - 1,
        arena: seed_arena,
    }
}

fn should_parallelize_dfs(cfg: &RunConfig) -> bool {
    cfg.max_paths.is_none_or(|max| max >= MIN_PAR_DFS_PATHS)
        && cfg.start_nodes.len() >= rayon::current_num_threads()
}

fn visit_counts_arena(
    arena: &[PathEntry],
    mut path: usize,
    edge_count: usize,
) -> Option<VisitCounts> {
    if edge_count <= 1 {
        return None;
    }

    let mut counts = HashMap::with_capacity(arena[path].depth + 1);
    loop {
        *counts.entry(arena[path].node).or_insert(0) += 1;
        match arena[path].parent {
            Some(parent) => path = parent,
            None => return Some(counts),
        }
    }
}

fn visit_counts_task(
    arena: &[PathTask],
    mut path: usize,
    edge_count: usize,
) -> Option<VisitCounts> {
    if edge_count <= 1 {
        return None;
    }

    let mut counts = HashMap::with_capacity(arena[path].depth + 1);
    loop {
        *counts.entry(arena[path].node).or_insert(0) += 1;
        match arena[path].parent {
            Some(parent) => path = parent,
            None => return Some(counts),
        }
    }
}

fn can_visit_arena(
    arena: &[PathEntry],
    mut path: usize,
    node: NodeId,
    max_revisits: usize,
    visit_counts: Option<&VisitCounts>,
) -> bool {
    if let Some(visit_counts) = visit_counts {
        return visit_counts.get(&node).copied().unwrap_or(0) <= max_revisits;
    }

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

fn can_visit_task(
    arena: &[PathTask],
    mut path: usize,
    node: NodeId,
    max_revisits: usize,
    visit_counts: Option<&VisitCounts>,
) -> bool {
    if let Some(visit_counts) = visit_counts {
        return visit_counts.get(&node).copied().unwrap_or(0) <= max_revisits;
    }

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
    cfg: &RunConfig,
    kernel: &BoundKernel,
) -> Result<GraphPath<'a>> {
    let mut nodes = Vec::with_capacity(arena[path].depth + 1);
    let mut edges = Vec::with_capacity(arena[path].depth);
    let state = kernel.state_row(&arena[path].state);
    let mut states = cfg
        .intermediate_states
        .then(|| Vec::with_capacity(nodes.capacity()));

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
        if let Some(states) = &mut states {
            states.push(kernel.state_row(&arena[path].state));
        }
        match arena[path].parent {
            Some(parent) => path = parent,
            None => break,
        }
    }

    nodes.reverse();
    edges.reverse();
    if let Some(states) = &mut states {
        states.reverse();
    }
    Ok(GraphPath {
        nodes,
        edges,
        state,
        intermediate_states: states,
    })
}

fn materialize_task<'a>(
    graph: &'a Graph,
    arena: &[PathTask],
    mut path: usize,
    cfg: &RunConfig,
    kernel: &BoundKernel,
) -> Result<GraphPath<'a>> {
    let mut nodes = Vec::with_capacity(arena[path].depth + 1);
    let mut edges = Vec::with_capacity(arena[path].depth);
    let state = kernel.state_row(&arena[path].state);
    let mut states = cfg
        .intermediate_states
        .then(|| Vec::with_capacity(nodes.capacity()));

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
        if let Some(states) = &mut states {
            states.push(kernel.state_row(&arena[path].state));
        }
        match arena[path].parent {
            Some(parent) => path = parent,
            None => break,
        }
    }

    nodes.reverse();
    edges.reverse();
    if let Some(states) = &mut states {
        states.reverse();
    }
    Ok(GraphPath {
        nodes,
        edges,
        state,
        intermediate_states: states,
    })
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
        dsl::{DslExpr as e, DslKernel, Value},
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

    fn state_u64(state: &[(String, Value)], name: &str) -> u64 {
        match state
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
        {
            Some(Value::U64(value)) => *value,
            other => panic!("expected {name} to be U64, got {other:?}"),
        }
    }

    #[test]
    fn returns_stopped_paths() {
        let graph = graph();
        let result = graph
            .search(traversal(
                e::bool_lit(true),
                e::dest("kind").eq(e::string_lit("end")),
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
            TraversalConfigBuilder::new(DslKernel::new(e::edge("ok"), [], e::bool_lit(true), []))
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
    fn returns_final_state_and_optional_intermediate_states() {
        let kernel = DslKernel::new(
            e::bool_lit(true),
            [("hops".into(), e::state("hops").plus(e::uint_lit(1)))],
            e::dest("kind").eq(e::string_lit("end")),
            [("hops".into(), Value::U64(0))],
        );

        let graph = graph();
        let without_history = graph
            .search(
                TraversalConfigBuilder::new(kernel.clone())
                    .with_start_nodes(["a".to_string()])
                    .with_strategy(TraversalStrategy::BreadthFirst)
                    .with_parallelism(false)
                    .build(),
            )
            .unwrap();

        assert_eq!(state_u64(&without_history.paths[0].state, "hops"), 2);
        assert!(without_history.paths[0].intermediate_states.is_none());

        let with_history = graph
            .search(
                TraversalConfigBuilder::new(kernel)
                    .with_start_nodes(["a".to_string()])
                    .with_strategy(TraversalStrategy::BreadthFirst)
                    .with_parallelism(false)
                    .with_intermediate_states(true)
                    .build(),
            )
            .unwrap();

        let states = with_history.paths[0].intermediate_states.as_ref().unwrap();
        assert_eq!(states.len(), with_history.paths[0].nodes.len());
        assert_eq!(
            states
                .iter()
                .map(|state| state_u64(state, "hops"))
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(states.last().unwrap(), &with_history.paths[0].state);
    }

    #[test]
    fn rejects_revisits_by_default() {
        let graph = graph();
        let result = graph
            .search(traversal(e::bool_lit(true), e::bool_lit(false)))
            .unwrap();
        assert!(result.paths.is_empty());
        assert_eq!(result.stats.skipped_revisits, 1);
        assert_eq!(result.stats.stopped_paths, 0);
    }

    #[test]
    fn reports_unknown_start_node() {
        let config = TraversalConfigBuilder::new(DslKernel::new(
            e::bool_lit(true),
            [],
            e::bool_lit(true),
            [("x".to_string(), Value::U64(0))],
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
                    e::bool_lit(true),
                    [],
                    e::dest_id().eq(e::uint_lit(3)),
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
                e::edge_id().ne(e::string_lit("ac")),
                e::dest_id().eq(e::string_lit("d")),
            ))
            .unwrap();

        assert_eq!(path_set(&result), vec![vec!["a", "b", "d"]]);
    }

    #[test]
    fn parallel_bfs_matches_serial_path_set() {
        let graph = branching_graph();
        let base = DslKernel::new(
            e::bool_lit(true),
            [],
            e::dest("kind").eq(e::string_lit("end")),
            [],
        );
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
                    e::bool_lit(true),
                    [],
                    e::dest("kind").eq(e::string_lit("end")),
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
                    e::bool_lit(true),
                    [],
                    e::dest("kind").eq(e::string_lit("end")),
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

    /// This test case is needed because parallel DFS was highly unoptimized for 1 thread previously,
    /// and remediation required adding [`DfsSeed`] so parent links remain valid after splitting work out of a temporary arena.
    #[test]
    fn parallel_dfs_single_thread_branch_returns_valid_paths() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            let graph = branching_graph();
            let parallel = graph
                .search(
                    TraversalConfigBuilder::new(DslKernel::new(
                        e::bool_lit(true),
                        [],
                        e::dest("kind").eq(e::string_lit("end")),
                        [],
                    ))
                    .with_start_nodes(["s".to_string()])
                    .with_strategy(TraversalStrategy::DepthFirst)
                    .with_parallelism(true)
                    .build(),
                )
                .unwrap();
            let serial = graph
                .search(
                    TraversalConfigBuilder::new(DslKernel::new(
                        e::bool_lit(true),
                        [],
                        e::dest("kind").eq(e::string_lit("end")),
                        [],
                    ))
                    .with_start_nodes(["s".to_string()])
                    .with_strategy(TraversalStrategy::DepthFirst)
                    .with_parallelism(false)
                    .build(),
                )
                .unwrap();

            assert_eq!(parallel.paths.len(), 4);
            assert_eq!(path_set(&parallel), path_set(&serial));
        });
    }

    #[test]
    fn parallel_dfs_soft_stop_returns_exact_max_paths() {
        let graph = branching_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(DslKernel::new(
                    e::bool_lit(true),
                    [],
                    e::dest("kind").eq(e::string_lit("end")),
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
        let kernel = DslKernel::new(e::bool_lit(true), [], e::bool_lit(true), []);
        assert!(TraversalConfigBuilder::new(kernel.clone()).build().parallel);
        assert!(
            !TraversalConfigBuilder::new(kernel)
                .with_parallelism(false)
                .build()
                .parallel
        );
    }
}
