use std::collections::{HashMap, VecDeque};

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::{
    graph::{EdgeId, GraphId, NodeId},
    traversal::{RunOptions, SearchStats, config::TraversalStrategy, progress::Progress},
};

const MIN_PAR_FRONTIER: usize = 512;
const MIN_PAR_EDGES: usize = 8_192;
const MIN_PAR_DFS_PATHS: usize = 64;

type VisitCounts = HashMap<NodeId, usize>;
type InitArena<S> = (Vec<PathEntry<S>>, VecDeque<usize>, SearchStats);

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
    fn for_each_range<F>(&self, node: NodeId, start: usize, end: usize, visit: F) -> Result<()>
    where
        F: FnMut(EdgeId, NodeId) -> Result<bool>;
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
        }
    }
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

pub(crate) mod cursor;

pub(crate) fn search<A>(adapter: &A, run: RunOptions) -> Result<SearchOutput<A::Path>>
where
    A: SearchAdapter + Sync,
    A::State: Send + Sync + Clone,
    A::Path: Send,
    A::Cache: Send,
{
    let mut cursor = cursor::GraphCursor::new(run)?;
    let mut paths = Vec::new();
    while let Some(batch) = cursor.next_batch(adapter, 1024)? {
        paths.extend(batch);
    }
    Ok(SearchOutput {
        paths,
        stats: cursor.stats(),
    })
}
pub(crate) fn search_serial<A: SearchAdapter>(
    adapter: &A,
    run: RunOptions,
) -> Result<SearchOutput<A::Path>> {
    let mut cursor = cursor::Cursor::new(run)?;
    let mut paths = Vec::new();
    while let Some(batch) = cursor.next_batch(adapter, 1024)? {
        paths.extend(batch);
    }
    Ok(SearchOutput {
        paths,
        stats: cursor.stats,
    })
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

fn eval_arena_edge<A>(
    adapter: &A,
    arena: &[PathEntry<A::State>],
    parent: usize,
    candidate: (EdgeId, NodeId),
    cfg: &RunConfig,
    stats: &mut SearchStats,
    cache: &A::Cache,
) -> Result<Option<EdgeEval<A::State>>>
where
    A: SearchAdapter,
{
    let (edge, dest) = candidate;
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

fn should_parallelize_dfs(cfg: &RunOptions) -> bool {
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
