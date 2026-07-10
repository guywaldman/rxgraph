use std::{
    collections::{HashMap, VecDeque},
    sync::atomic::{AtomicUsize, Ordering},
};

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::{
    graph::{EdgeId, GraphId, NodeId},
    traversal::{RunOptions, SearchStats, config::TraversalStrategy, progress::Progress},
};

const MIN_PAR_FRONTIER: usize = 512;
const MIN_PAR_EDGES: usize = 8_192;
const DFS_SEEDS_PER_THREAD: usize = 8;
const MIN_PAR_DFS_PATHS: usize = 64;

type VisitCounts = HashMap<NodeId, usize>;
type InitArena<S> = (Vec<PathEntry<S>>, VecDeque<usize>, SearchStats);
type FrontierEdges<S> = (Vec<(usize, EdgeEval<S>)>, SearchStats);

pub(crate) struct SearchOutput<P> {
    pub(crate) paths: Vec<P>,
    pub(crate) stats: SearchStats,
}

pub(crate) trait SearchAdapter {
    type State: Clone;
    type Path;
    type Cache;

    fn resolve_node(&self, external: GraphId<'_>) -> Result<Option<NodeId>>;
    fn initial_state(&self, node: NodeId) -> Result<Self::State>;
    fn dense_node_count(&self) -> Option<usize> {
        None
    }
    fn prefetch_outgoing(&self, _nodes: &[NodeId]) -> Result<()> {
        Ok(())
    }
    fn out_degree(&self, node: NodeId) -> Result<usize>;
    fn for_each_outgoing<F>(&self, node: NodeId, visit: F) -> Result<()>
    where
        F: FnMut(EdgeId, NodeId) -> Result<bool>;
    fn outgoing_at(&self, node: NodeId, index: usize) -> Result<(EdgeId, NodeId)>;
    fn make_cache(&self) -> Self::Cache;
    fn eval_edge(
        &self,
        src: NodeId,
        edge: EdgeId,
        dest: NodeId,
        state: &Self::State,
        cache: &Self::Cache,
    ) -> Result<Option<(Self::State, bool)>>;
    fn materialize(
        &self,
        arena: &[PathEntry<Self::State>],
        path: usize,
        intermediate_states: bool,
    ) -> Result<Self::Path>;
}

#[derive(Debug, Clone)]
pub(crate) struct PathEntry<S> {
    pub(crate) node: NodeId,
    pub(crate) incoming_edge: Option<EdgeId>,
    pub(crate) parent: Option<usize>,
    pub(crate) depth: usize,
    pub(crate) state: S,
    visits: VisitFingerprint,
}

#[derive(Debug, Clone, Copy)]
struct VisitFingerprint([u64; 4]);

impl VisitFingerprint {
    fn for_node(node: NodeId) -> Self {
        let mut fingerprint = Self([0; 4]);
        fingerprint.insert(node);
        fingerprint
    }

    #[inline]
    fn might_contain(&self, node: NodeId) -> bool {
        let [first, second] = visit_hashes(node);
        self.contains_bit(first) && self.contains_bit(second)
    }

    #[inline]
    fn insert(&mut self, node: NodeId) {
        for bit in visit_hashes(node) {
            self.0[bit / 64] |= 1 << (bit % 64);
        }
    }

    #[inline]
    fn contains_bit(&self, bit: usize) -> bool {
        self.0[bit / 64] & (1 << (bit % 64)) != 0
    }
}

#[inline]
fn visit_hashes(node: NodeId) -> [usize; 2] {
    [
        (node & 0xff) as usize,
        (node.wrapping_mul(0x9e37_79b1) >> 24) as usize,
    ]
}

#[derive(Debug, Clone)]
struct RunConfig {
    start_nodes: Vec<crate::OwnedGraphId>,
    max_depth: usize,
    max_paths: Option<usize>,
    strategy: TraversalStrategy,
    max_visits_per_node: usize,
    intermediate_states: bool,
    progress: bool,
}

impl RunConfig {
    fn from_run(run: RunOptions) -> Self {
        Self {
            start_nodes: run.start_nodes,
            max_depth: run.max_depth.unwrap_or(usize::MAX),
            max_paths: run.max_paths,
            strategy: run.strategy,
            max_visits_per_node: run.max_visits_per_node,
            intermediate_states: run.intermediate_states,
            progress: run.progress,
        }
    }
}

struct DfsSeed<S> {
    arena: Vec<PathEntry<S>>,
    task: usize,
}

struct DfsFrame {
    next_edge: usize,
    edge_count: usize,
}

trait ActiveVisits {
    fn count(&self, node: NodeId) -> usize;
    fn increment(&mut self, node: NodeId);
    fn decrement(&mut self, node: NodeId);
}

struct DenseVisits(Vec<usize>);

impl ActiveVisits for DenseVisits {
    #[inline]
    fn count(&self, node: NodeId) -> usize {
        self.0[node as usize]
    }

    #[inline]
    fn increment(&mut self, node: NodeId) {
        self.0[node as usize] += 1;
    }

    #[inline]
    fn decrement(&mut self, node: NodeId) {
        self.0[node as usize] -= 1;
    }
}

struct SparseVisits(VisitCounts);

impl ActiveVisits for SparseVisits {
    #[inline]
    fn count(&self, node: NodeId) -> usize {
        self.0.get(&node).copied().unwrap_or(0)
    }

    #[inline]
    fn increment(&mut self, node: NodeId) {
        *self.0.entry(node).or_insert(0) += 1;
    }

    #[inline]
    fn decrement(&mut self, node: NodeId) {
        let count = self.0.get_mut(&node).expect("active path node is counted");
        *count -= 1;
        if *count == 0 {
            self.0.remove(&node);
        }
    }
}

struct EdgeEval<S> {
    edge: EdgeId,
    dest: NodeId,
    state: S,
    stop: bool,
}

struct TaskResult<P> {
    paths: Vec<P>,
    stats: SearchStats,
}

pub(crate) fn search<A>(adapter: &A, run: RunOptions) -> Result<SearchOutput<A::Path>>
where
    A: SearchAdapter + Sync,
    A::State: Send + Sync + Clone,
    A::Path: Send,
    A::Cache: Send,
{
    if run.max_visits_per_node == 0 {
        bail!("max_visits_per_node must be at least 1");
    }
    let strategy = run.strategy;
    let parallel = run.parallel;
    let cfg = RunConfig::from_run(run);

    match (parallel, strategy) {
        (false, _) => search_serial_cfg(adapter, &cfg),
        (true, TraversalStrategy::BreadthFirst) => search_bfs_parallel(adapter, &cfg),
        (true, TraversalStrategy::DepthFirst) if should_parallelize_dfs(&cfg) => {
            search_dfs_parallel(adapter, &cfg)
        }
        (true, TraversalStrategy::DepthFirst) => search_serial_cfg(adapter, &cfg),
    }
}

pub(crate) fn search_serial<A>(adapter: &A, run: RunOptions) -> Result<SearchOutput<A::Path>>
where
    A: SearchAdapter,
{
    if run.max_visits_per_node == 0 {
        bail!("max_visits_per_node must be at least 1");
    }
    search_serial_cfg(adapter, &RunConfig::from_run(run))
}

fn search_serial_cfg<A>(adapter: &A, cfg: &RunConfig) -> Result<SearchOutput<A::Path>>
where
    A: SearchAdapter,
{
    if matches!(cfg.strategy, TraversalStrategy::BreadthFirst) {
        return search_bfs_serial(adapter, cfg);
    }

    search_dfs_serial(adapter, cfg)
}

fn search_dfs_serial<A>(adapter: &A, cfg: &RunConfig) -> Result<SearchOutput<A::Path>>
where
    A: SearchAdapter,
{
    if let Some(node_count) = adapter.dense_node_count() {
        search_dfs_serial_with_visits(adapter, cfg, DenseVisits(vec![0; node_count]))
    } else {
        search_dfs_serial_with_visits(adapter, cfg, SparseVisits(HashMap::new()))
    }
}

fn search_dfs_serial_with_visits<A, V>(
    adapter: &A,
    cfg: &RunConfig,
    mut visits: V,
) -> Result<SearchOutput<A::Path>>
where
    A: SearchAdapter,
    V: ActiveVisits,
{
    let mut stats = SearchStats::default();
    let mut paths = Vec::new();
    let mut progress = Progress::new(cfg.progress);
    let cache = adapter.make_cache();

    for external in &cfg.start_nodes {
        let node = adapter
            .resolve_node(external.as_ref())?
            .with_context(|| format!("unknown start node {external}"))?;
        let mut arena = vec![PathEntry {
            node,
            incoming_edge: None,
            parent: None,
            depth: 0,
            state: adapter.initial_state(node)?,
            visits: VisitFingerprint::for_node(node),
        }];
        let mut frames = vec![DfsFrame {
            next_edge: 0,
            edge_count: if cfg.max_depth == 0 {
                0
            } else {
                adapter.out_degree(node)?
            },
        }];
        visits.increment(node);
        stats.start_nodes += 1;
        stats.path_entries += 1;

        while !frames.is_empty() {
            progress.tick(&stats);
            let path = arena.len() - 1;
            let frame = frames.last_mut().expect("frame exists");
            if frame.next_edge >= frame.edge_count {
                frames.pop();
                let entry = arena.pop().expect("frame has path entry");
                visits.decrement(entry.node);
                continue;
            }

            let (edge, dest) = adapter.outgoing_at(arena[path].node, frame.next_edge)?;
            frame.next_edge += 1;
            if visits.count(dest) >= cfg.max_visits_per_node {
                stats.skipped_revisits += 1;
                continue;
            }

            stats.evaluated_edges += 1;
            let Some((state, stop)) =
                adapter.eval_edge(arena[path].node, edge, dest, &arena[path].state, &cache)?
            else {
                stats.rejected_edges += 1;
                continue;
            };

            let child = push_entry(
                &mut arena,
                path,
                EdgeEval {
                    edge,
                    dest,
                    state,
                    stop,
                },
            );
            stats.accepted_edges += 1;
            stats.path_entries += 1;
            stats.max_depth = stats.max_depth.max(arena[child].depth);

            if stop {
                paths.push(adapter.materialize(&arena, child, cfg.intermediate_states)?);
                stats.stopped_paths += 1;
                arena.pop();
                if cfg.max_paths.is_some_and(|max| paths.len() >= max) {
                    progress.finish(&stats);
                    return Ok(SearchOutput { paths, stats });
                }
                continue;
            }

            visits.increment(dest);
            frames.push(DfsFrame {
                next_edge: 0,
                edge_count: if arena[child].depth >= cfg.max_depth {
                    0
                } else {
                    adapter.out_degree(dest)?
                },
            });
        }
    }

    progress.finish(&stats);
    Ok(SearchOutput { paths, stats })
}

fn search_bfs_serial<A>(adapter: &A, cfg: &RunConfig) -> Result<SearchOutput<A::Path>>
where
    A: SearchAdapter,
{
    if cfg.max_paths == Some(1) {
        return search_bfs_first(adapter, cfg);
    }

    let (mut arena, frontier, mut stats) = initial_arena(adapter, cfg)?;
    let mut frontier = frontier.into_iter().collect::<Vec<_>>();
    let mut paths = Vec::new();
    let mut progress = Progress::new(cfg.progress);

    while !frontier.is_empty() {
        progress.tick(&stats);
        let frontier_nodes = frontier
            .iter()
            .map(|&path| arena[path].node)
            .collect::<Vec<_>>();
        adapter.prefetch_outgoing(&frontier_nodes)?;
        let edge_count = frontier.iter().try_fold(0usize, |sum, &path| {
            Ok::<_, anyhow::Error>(sum + adapter.out_degree(arena[path].node)?)
        })?;
        let (edges, local) = eval_frontier_serial(adapter, &arena, &frontier, edge_count, cfg)?;
        merge_stats(&mut stats, local);

        let mut next = Vec::with_capacity(edges.len());
        for (parent, edge) in edges {
            let stop = edge.stop;
            let child = push_entry(&mut arena, parent, edge);
            stats.accepted_edges += 1;
            stats.path_entries += 1;
            stats.max_depth = stats.max_depth.max(arena[child].depth);
            if stop {
                paths.push(adapter.materialize(&arena, child, cfg.intermediate_states)?);
                stats.stopped_paths += 1;
                if let Some(max) = cfg.max_paths
                    && paths.len() >= max
                {
                    paths.truncate(max);
                    progress.finish(&stats);
                    return Ok(SearchOutput { paths, stats });
                }
            } else {
                next.push(child);
            }
        }
        frontier = next;
    }

    progress.finish(&stats);
    Ok(SearchOutput { paths, stats })
}

fn search_bfs_first<A>(adapter: &A, cfg: &RunConfig) -> Result<SearchOutput<A::Path>>
where
    A: SearchAdapter,
{
    let (mut arena, frontier, mut stats) = initial_arena(adapter, cfg)?;
    let mut frontier = frontier.into_iter().collect::<Vec<_>>();
    let mut progress = Progress::new(cfg.progress);
    let cache = adapter.make_cache();

    while !frontier.is_empty() {
        progress.tick(&stats);
        let frontier_nodes = frontier
            .iter()
            .map(|&path| arena[path].node)
            .collect::<Vec<_>>();
        adapter.prefetch_outgoing(&frontier_nodes)?;
        let mut next = Vec::new();

        for &parent in &frontier {
            if arena[parent].depth >= cfg.max_depth {
                continue;
            }
            let node = arena[parent].node;
            let edge_count = adapter.out_degree(node)?;
            next.reserve(edge_count);
            let mut found = None;
            adapter.for_each_outgoing(node, |edge, dest| {
                let Some(edge) =
                    eval_arena_edge(adapter, &arena, parent, edge, dest, cfg, &mut stats, &cache)?
                else {
                    return Ok(true);
                };
                let stop = edge.stop;
                let child = push_entry(&mut arena, parent, edge);
                stats.accepted_edges += 1;
                stats.path_entries += 1;
                stats.max_depth = stats.max_depth.max(arena[child].depth);
                if stop {
                    stats.stopped_paths += 1;
                    found = Some(adapter.materialize(&arena, child, cfg.intermediate_states)?);
                    Ok(false)
                } else {
                    next.push(child);
                    Ok(true)
                }
            })?;
            if let Some(path) = found {
                progress.finish(&stats);
                return Ok(SearchOutput {
                    paths: vec![path],
                    stats,
                });
            }
        }
        frontier = next;
    }

    progress.finish(&stats);
    Ok(SearchOutput {
        paths: Vec::new(),
        stats,
    })
}

fn search_bfs_parallel<A>(adapter: &A, cfg: &RunConfig) -> Result<SearchOutput<A::Path>>
where
    A: SearchAdapter + Sync,
    A::State: Send + Sync + Clone,
    A::Path: Send,
    A::Cache: Send,
{
    let (mut arena, frontier, mut stats) = initial_arena(adapter, cfg)?;
    let mut frontier = frontier.into_iter().collect::<Vec<_>>();
    let mut paths = Vec::new();
    let mut progress = Progress::new(cfg.progress);

    while !frontier.is_empty() {
        progress.tick(&stats);
        let frontier_nodes = frontier
            .iter()
            .map(|&path| arena[path].node)
            .collect::<Vec<_>>();
        adapter.prefetch_outgoing(&frontier_nodes)?;
        let edge_count = frontier.iter().try_fold(0usize, |sum, &path| {
            Ok::<_, anyhow::Error>(sum + adapter.out_degree(arena[path].node)?)
        })?;
        let (edges, local) = if frontier.len() >= MIN_PAR_FRONTIER || edge_count >= MIN_PAR_EDGES {
            eval_frontier_parallel(adapter, &arena, &frontier, cfg)?
        } else {
            eval_frontier_serial(adapter, &arena, &frontier, edge_count, cfg)?
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
                paths.push(adapter.materialize(&arena, child, cfg.intermediate_states)?);
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
            return Ok(SearchOutput { paths, stats });
        }
        frontier = next;
    }

    progress.finish(&stats);
    Ok(SearchOutput { paths, stats })
}

fn search_dfs_parallel<A>(adapter: &A, cfg: &RunConfig) -> Result<SearchOutput<A::Path>>
where
    A: SearchAdapter + Sync,
    A::State: Send + Sync + Clone,
    A::Path: Send,
    A::Cache: Send,
{
    let (queue, mut stats) = initial_tasks(adapter, cfg)?;
    let mut seed_paths = Vec::new();
    let mut progress = Progress::new(cfg.progress);
    progress.tick(&stats);
    let seeds = build_dfs_seeds(adapter, cfg, queue, &mut seed_paths, &mut stats)?;
    progress.tick(&stats);

    if let Some(max) = cfg.max_paths
        && seed_paths.len() >= max
    {
        seed_paths.truncate(max);
        progress.finish(&stats);
        return Ok(SearchOutput {
            paths: seed_paths,
            stats,
        });
    }

    let found = AtomicUsize::new(seed_paths.len());
    let parallel = seeds.len() >= rayon::current_num_threads();
    let results = if !parallel {
        seeds
            .into_iter()
            .map(|seed| dfs_seed(adapter, cfg, seed, &found))
            .collect::<Result<Vec<_>>>()?
    } else {
        seeds
            .into_par_iter()
            .map(|seed| dfs_seed(adapter, cfg, seed, &found))
            .collect::<Result<Vec<_>>>()?
    };

    let mut paths = seed_paths;
    for mut result in results {
        if parallel {
            result.stats.parallel_edges = result.stats.evaluated_edges;
        }
        merge_stats(&mut stats, result.stats);
        paths.extend(result.paths);
    }
    if let Some(max) = cfg.max_paths {
        paths.truncate(max);
    }
    progress.finish(&stats);
    Ok(SearchOutput { paths, stats })
}

fn initial_arena<A>(adapter: &A, cfg: &RunConfig) -> Result<InitArena<A::State>>
where
    A: SearchAdapter,
{
    let mut arena = Vec::with_capacity(cfg.start_nodes.len());
    let mut frontier = VecDeque::with_capacity(cfg.start_nodes.len());
    let mut stats = SearchStats::default();

    for external in &cfg.start_nodes {
        let node = adapter
            .resolve_node(external.as_ref())?
            .with_context(|| format!("unknown start node {external}"))?;
        frontier.push_back(arena.len());
        arena.push(PathEntry {
            node,
            incoming_edge: None,
            parent: None,
            depth: 0,
            state: adapter.initial_state(node)?,
            visits: VisitFingerprint::for_node(node),
        });
        stats.start_nodes += 1;
        stats.path_entries += 1;
    }

    Ok((arena, frontier, stats))
}

fn initial_tasks<A>(
    adapter: &A,
    cfg: &RunConfig,
) -> Result<(VecDeque<DfsSeed<A::State>>, SearchStats)>
where
    A: SearchAdapter,
{
    let mut queue = VecDeque::with_capacity(cfg.start_nodes.len());
    let mut stats = SearchStats::default();

    for external in &cfg.start_nodes {
        let node = adapter
            .resolve_node(external.as_ref())?
            .with_context(|| format!("unknown start node {external}"))?;
        queue.push_back(DfsSeed {
            arena: vec![PathEntry {
                node,
                incoming_edge: None,
                parent: None,
                depth: 0,
                state: adapter.initial_state(node)?,
                visits: VisitFingerprint::for_node(node),
            }],
            task: 0,
        });
        stats.start_nodes += 1;
        stats.path_entries += 1;
    }

    Ok((queue, stats))
}

fn eval_frontier_serial<A>(
    adapter: &A,
    arena: &[PathEntry<A::State>],
    frontier: &[usize],
    edge_count: usize,
    cfg: &RunConfig,
) -> Result<FrontierEdges<A::State>>
where
    A: SearchAdapter,
{
    let mut edges = Vec::with_capacity(edge_count);
    let mut stats = SearchStats::default();
    let cache = adapter.make_cache();

    for &parent in frontier {
        let local = eval_parent_into(adapter, arena, parent, cfg, &mut edges, &cache)?;
        merge_stats(&mut stats, local);
    }

    Ok((edges, stats))
}

fn eval_frontier_parallel<A>(
    adapter: &A,
    arena: &[PathEntry<A::State>],
    frontier: &[usize],
    cfg: &RunConfig,
) -> Result<FrontierEdges<A::State>>
where
    A: SearchAdapter + Sync,
    A::State: Send + Sync + Clone,
    A::Cache: Send,
{
    let (edges, mut stats) = frontier
        .par_iter()
        .try_fold(
            || (Vec::new(), SearchStats::default(), adapter.make_cache()),
            |(mut edges, mut stats, cache), &parent| {
                let local = eval_parent_into(adapter, arena, parent, cfg, &mut edges, &cache)?;
                merge_stats(&mut stats, local);
                Ok::<_, anyhow::Error>((edges, stats, cache))
            },
        )
        .map(|fold| fold.map(|(edges, stats, _cache)| (edges, stats)))
        .try_reduce(
            || (Vec::new(), SearchStats::default()),
            |(mut left_edges, mut left_stats), (right_edges, right_stats)| {
                left_edges.extend(right_edges);
                merge_stats(&mut left_stats, right_stats);
                Ok::<_, anyhow::Error>((left_edges, left_stats))
            },
        )?;
    stats.parallel_edges = stats.evaluated_edges;
    Ok((edges, stats))
}

fn eval_parent_into<A>(
    adapter: &A,
    arena: &[PathEntry<A::State>],
    parent: usize,
    cfg: &RunConfig,
    out: &mut Vec<(usize, EdgeEval<A::State>)>,
    cache: &A::Cache,
) -> Result<SearchStats>
where
    A: SearchAdapter,
{
    let mut stats = SearchStats::default();

    if arena[parent].depth < cfg.max_depth {
        let node = arena[parent].node;
        let edge_count = adapter.out_degree(node)?;
        out.reserve(edge_count);
        adapter.for_each_outgoing(node, |edge, dest| {
            if let Some(edge) =
                eval_arena_edge(adapter, arena, parent, edge, dest, cfg, &mut stats, cache)?
            {
                out.push((parent, edge));
            }
            Ok(true)
        })?;
    }

    Ok(stats)
}

#[allow(clippy::too_many_arguments)]
fn eval_arena_edge<A>(
    adapter: &A,
    arena: &[PathEntry<A::State>],
    parent: usize,
    edge: EdgeId,
    dest: NodeId,
    cfg: &RunConfig,
    stats: &mut SearchStats,
    cache: &A::Cache,
) -> Result<Option<EdgeEval<A::State>>>
where
    A: SearchAdapter,
{
    if !can_visit_arena(arena, parent, dest, cfg.max_visits_per_node) {
        stats.skipped_revisits += 1;
        return Ok(None);
    }

    stats.evaluated_edges += 1;
    let src = arena[parent].node;
    let Some((state, stop)) = adapter.eval_edge(src, edge, dest, &arena[parent].state, cache)?
    else {
        stats.rejected_edges += 1;
        return Ok(None);
    };

    Ok(Some(EdgeEval {
        edge,
        dest,
        state,
        stop,
    }))
}

fn push_entry<S>(arena: &mut Vec<PathEntry<S>>, parent: usize, edge: EdgeEval<S>) -> usize {
    let child = arena.len();
    let mut visits = arena[parent].visits;
    visits.insert(edge.dest);
    arena.push(PathEntry {
        node: edge.dest,
        incoming_edge: Some(edge.edge),
        parent: Some(parent),
        depth: arena[parent].depth + 1,
        state: edge.state,
        visits,
    });
    child
}

fn build_dfs_seeds<A>(
    adapter: &A,
    cfg: &RunConfig,
    mut queue: VecDeque<DfsSeed<A::State>>,
    paths: &mut Vec<A::Path>,
    stats: &mut SearchStats,
) -> Result<Vec<DfsSeed<A::State>>>
where
    A: SearchAdapter,
{
    let target = rayon::current_num_threads() * DFS_SEEDS_PER_THREAD;
    let cache = adapter.make_cache();

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

        let children = expand_task(adapter, cfg, &arena, task, stats, &cache)?;
        for (child, stop) in children {
            let child = push_entry(&mut arena, task, child);
            if stop {
                paths.push(adapter.materialize(&arena, child, cfg.intermediate_states)?);
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

fn dfs_seed<A>(
    adapter: &A,
    cfg: &RunConfig,
    seed: DfsSeed<A::State>,
    found: &AtomicUsize,
) -> Result<TaskResult<A::Path>>
where
    A: SearchAdapter + Sync,
    A::State: Send + Sync + Clone,
{
    let mut arena = seed.arena;
    let mut stack = vec![seed.task];
    let mut paths = Vec::new();
    let mut stats = SearchStats::default();
    let cache = adapter.make_cache();

    while let Some(task) = stack.pop() {
        if cfg
            .max_paths
            .is_some_and(|max| found.load(Ordering::Relaxed) >= max)
            || arena[task].depth >= cfg.max_depth
        {
            continue;
        }

        let children = expand_task(adapter, cfg, &arena, task, &mut stats, &cache)?;
        for (child, stop) in children {
            let child = push_entry(&mut arena, task, child);
            if stop {
                let previous = found.fetch_add(1, Ordering::Relaxed);
                stats.stopped_paths += 1;
                if cfg.max_paths.is_none_or(|max| previous < max) {
                    paths.push(adapter.materialize(&arena, child, cfg.intermediate_states)?);
                }
            } else {
                stack.push(child);
            }
        }
    }

    Ok(TaskResult { paths, stats })
}

fn expand_task<A>(
    adapter: &A,
    cfg: &RunConfig,
    arena: &[PathEntry<A::State>],
    task: usize,
    stats: &mut SearchStats,
    cache: &A::Cache,
) -> Result<Vec<(EdgeEval<A::State>, bool)>>
where
    A: SearchAdapter,
{
    let node = arena[task].node;
    let edge_count = adapter.out_degree(node)?;
    let mut children = Vec::with_capacity(edge_count);
    adapter.for_each_outgoing(node, |edge, dest| {
        if !can_visit_arena(arena, task, dest, cfg.max_visits_per_node) {
            stats.skipped_revisits += 1;
            return Ok(true);
        }

        stats.evaluated_edges += 1;
        let Some((state, stop)) = adapter.eval_edge(node, edge, dest, &arena[task].state, cache)?
        else {
            stats.rejected_edges += 1;
            return Ok(true);
        };
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
        Ok(true)
    })?;
    Ok(children)
}

fn standalone_seed<S: Clone>(arena: &[PathEntry<S>], mut path: usize) -> DfsSeed<S> {
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
        seed_arena.push(PathEntry {
            node: task.node,
            incoming_edge: task.incoming_edge,
            parent,
            depth: task.depth,
            state: task.state.clone(),
            visits: task.visits,
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

fn can_visit_arena<S>(
    arena: &[PathEntry<S>],
    mut path: usize,
    node: NodeId,
    max_visits: usize,
) -> bool {
    if !arena[path].visits.might_contain(node) {
        return true;
    }

    let mut visits = 0usize;
    loop {
        if arena[path].node == node {
            visits += 1;
            if visits >= max_visits {
                return false;
            }
        }
        match arena[path].parent {
            Some(parent) => path = parent,
            None => return true,
        }
    }
}

fn merge_stats(into: &mut SearchStats, from: SearchStats) {
    into.start_nodes += from.start_nodes;
    into.path_entries += from.path_entries;
    into.evaluated_edges += from.evaluated_edges;
    into.parallel_edges += from.parallel_edges;
    into.accepted_edges += from.accepted_edges;
    into.rejected_edges += from.rejected_edges;
    into.skipped_revisits += from.skipped_revisits;
    into.stopped_paths += from.stopped_paths;
    into.max_depth = into.max_depth.max(from.max_depth);
    into.materialized_node_payloads += from.materialized_node_payloads;
    into.materialized_edge_payloads += from.materialized_edge_payloads;
    into.lazy_payload_read_calls += from.lazy_payload_read_calls;
    into.lazy_payload_requested_rows += from.lazy_payload_requested_rows;
    into.lazy_payload_selected_rows += from.lazy_payload_selected_rows;
    into.lazy_payload_row_groups += from.lazy_payload_row_groups;
}
