//! Row-lazy native Rust traversal.
//!
//! This module is separate from [`Graph::search`](crate::Graph::search). It is
//! for callers that own a backend capable of resolving topology and native
//! payload rows lazily, then want traversal kernels to operate on Rust structs
//! instead of Arrow-backed [`Value`](crate::Value) rows.

use std::{
    collections::{HashMap, VecDeque},
    sync::OnceLock,
};

use anyhow::{Context, Result};

use crate::{
    graph::{EdgeId, Graph, GraphId, GraphRepo, NodeId},
    traversal::{RunOptions, SearchStats, config::TraversalStrategy, progress::Progress},
};

type VisitCounts = HashMap<NodeId, usize>;
type StoreRef<'a, N, E> = &'a dyn GraphStore<Node = N, Edge = E>;

/// A row-lazy graph storage backend.
///
/// Implementations own the topology/payload loading policy. The traversal
/// engine calls [`GraphStore::outgoing`] only for expanded nodes, and calls
/// [`GraphStore::node`] / [`GraphStore::edge`] only when a kernel accessor or a
/// returned path requests that payload.
pub trait GraphStore {
    /// Native node payload.
    type Node;
    /// Native edge payload.
    type Edge;

    /// Resolve an external start ID to an internal row ID.
    fn resolve_node(&self, external: GraphId<'_>) -> Result<Option<NodeId>>;

    /// External node ID for materialized paths, if the backend exposes one.
    fn external_node(&self, internal: NodeId) -> Result<Option<GraphId<'_>>>;

    /// External edge ID for materialized paths, if the backend exposes one.
    fn external_edge(&self, internal: EdgeId) -> Result<Option<GraphId<'_>>>;

    /// Outgoing topology for `src`.
    fn outgoing(&self, src: NodeId) -> Result<&[OutgoingEdge]>;

    /// Optional hook for stores that can batch-load outgoing topology.
    fn prefetch_outgoing(&self, _nodes: &[NodeId]) -> Result<()> {
        Ok(())
    }

    /// Native node payload for `id`.
    fn node(&self, id: NodeId) -> Result<&Self::Node>;

    /// Native edge payload for `id`.
    fn edge(&self, id: EdgeId) -> Result<&Self::Edge>;
}

/// One outgoing topology entry from a source node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutgoingEdge {
    /// Internal edge row ID.
    pub edge: EdgeId,
    /// Internal destination node row ID.
    pub dest: NodeId,
}

/// Context passed to [`Kernel::initial_state`].
pub struct StartCtx<'a, N, E> {
    store: StoreRef<'a, N, E>,
    node: NodeId,
}

impl<'a, N, E> StartCtx<'a, N, E> {
    fn new(store: StoreRef<'a, N, E>, node: NodeId) -> Self {
        Self { store, node }
    }

    /// Internal node row ID.
    pub fn id(&self) -> NodeId {
        self.node
    }

    /// External node ID, if available.
    pub fn external_id(&self) -> Result<Option<GraphId<'a>>> {
        self.store.external_node(self.node)
    }

    /// Native node payload.
    pub fn node(&self) -> Result<&'a N> {
        self.store.node(self.node)
    }
}

/// Per-edge context passed to a native [`Kernel`].
pub struct EdgeCtx<'a, N, E, S> {
    store: StoreRef<'a, N, E>,
    src: NodeId,
    dest: NodeId,
    edge: EdgeId,
    state: &'a S,
}

impl<'a, N, E, S> EdgeCtx<'a, N, E, S> {
    fn new(
        store: StoreRef<'a, N, E>,
        src: NodeId,
        dest: NodeId,
        edge: EdgeId,
        state: &'a S,
    ) -> Self {
        Self {
            store,
            src,
            dest,
            edge,
            state,
        }
    }

    fn with_state<'b>(&'b self, state: &'b S) -> EdgeCtx<'b, N, E, S> {
        EdgeCtx {
            store: self.store,
            src: self.src,
            dest: self.dest,
            edge: self.edge,
            state,
        }
    }

    /// Internal source node row ID.
    pub fn src_id(&self) -> NodeId {
        self.src
    }

    /// Internal destination node row ID.
    pub fn dest_id(&self) -> NodeId {
        self.dest
    }

    /// Internal edge row ID.
    pub fn edge_id(&self) -> EdgeId {
        self.edge
    }

    /// External source node ID, if available.
    pub fn src_external_id(&self) -> Result<Option<GraphId<'a>>> {
        self.store.external_node(self.src)
    }

    /// External destination node ID, if available.
    pub fn dest_external_id(&self) -> Result<Option<GraphId<'a>>> {
        self.store.external_node(self.dest)
    }

    /// External edge ID, if available.
    pub fn edge_external_id(&self) -> Result<Option<GraphId<'a>>> {
        self.store.external_edge(self.edge)
    }

    /// Native source node payload.
    pub fn src(&self) -> Result<&'a N> {
        self.store.node(self.src)
    }

    /// Native destination node payload.
    pub fn dest(&self) -> Result<&'a N> {
        self.store.node(self.dest)
    }

    /// Native edge payload.
    pub fn edge(&self) -> Result<&'a E> {
        self.store.edge(self.edge)
    }

    /// Per-path parent or child state, depending on the callback.
    pub fn state(&self) -> &S {
        self.state
    }
}

/// Native Rust traversal predicate/state machine.
pub trait Kernel {
    /// Native node payload type.
    type Node;
    /// Native edge payload type.
    type Edge;
    /// Path-local traversal state.
    type State: Clone;

    /// Initial state for a path that begins at `cx.id()`.
    fn initial_state(&self, cx: &StartCtx<'_, Self::Node, Self::Edge>) -> Result<Self::State>;

    /// Whether the candidate edge in `cx` may be accepted.
    fn visit(&self, cx: &EdgeCtx<'_, Self::Node, Self::Edge, Self::State>) -> Result<bool>;

    /// State for the child path after accepting the edge in `cx`.
    fn next_state(
        &self,
        cx: &EdgeCtx<'_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<Self::State>;

    /// Whether the accepted path should be emitted.
    ///
    /// `cx` carries the child state produced by [`Kernel::next_state`].
    fn stop(&self, cx: &EdgeCtx<'_, Self::Node, Self::Edge, Self::State>) -> Result<bool>;
}

/// One node in a returned native path.
#[derive(Debug, Clone, PartialEq)]
pub struct PathNode<'a, N, S> {
    /// Internal node row ID.
    pub id: NodeId,
    /// External node ID, if available.
    pub external_id: Option<GraphId<'a>>,
    /// Native node payload.
    pub payload: &'a N,
    /// Optional path-local state at this node.
    pub state: Option<S>,
}

/// One edge in a returned native path.
#[derive(Debug, Clone, PartialEq)]
pub struct PathEdge<'a, E> {
    /// Internal edge row ID.
    pub id: EdgeId,
    /// External edge ID, if available.
    pub external_id: Option<GraphId<'a>>,
    /// Native edge payload.
    pub payload: &'a E,
}

/// One returned native path.
#[derive(Debug, Clone, PartialEq)]
pub struct Path<'a, N, E, S> {
    /// Nodes in path order, including the start and final node.
    pub nodes: Vec<PathNode<'a, N, S>>,
    /// Edges in path order.
    pub edges: Vec<PathEdge<'a, E>>,
    /// Final path-local state.
    pub state: S,
}

/// Native traversal result.
#[derive(Debug)]
pub struct SearchResult<'a, N, E, S> {
    /// Materialized stopped paths.
    pub paths: Vec<Path<'a, N, E, S>>,
    /// Counters for completed work.
    pub stats: SearchStats,
}

/// Eager adapter over the existing [`Graph`] topology and row-aligned native
/// payload slices.
///
/// This is a compatibility/testing adapter. The underlying `Graph` already owns
/// full topology, and the supplied payload slices are already materialized.
pub struct EagerGraphStore<'a, N, E> {
    graph: &'a Graph,
    nodes: &'a [N],
    edges: &'a [E],
    outgoing: Vec<OnceLock<Vec<OutgoingEdge>>>,
}

impl<'a, N, E> EagerGraphStore<'a, N, E> {
    /// Builds an adapter. `nodes` and `edges` must be row-aligned with `graph`.
    pub fn new(graph: &'a Graph, nodes: &'a [N], edges: &'a [E]) -> Result<Self> {
        if nodes.len() != graph.node_count() {
            anyhow::bail!(
                "native node payload length {} does not match graph node count {}",
                nodes.len(),
                graph.node_count()
            );
        }
        if edges.len() != graph.edge_count() {
            anyhow::bail!(
                "native edge payload length {} does not match graph edge count {}",
                edges.len(),
                graph.edge_count()
            );
        }

        Ok(Self {
            graph,
            nodes,
            edges,
            outgoing: (0..graph.node_count()).map(|_| OnceLock::new()).collect(),
        })
    }
}

impl<N, E> GraphStore for EagerGraphStore<'_, N, E> {
    type Node = N;
    type Edge = E;

    fn resolve_node(&self, external: GraphId<'_>) -> Result<Option<NodeId>> {
        Ok(self.graph.repo.internal_node(external))
    }

    fn external_node(&self, internal: NodeId) -> Result<Option<GraphId<'_>>> {
        Ok(self.graph.repo.external_node(internal))
    }

    fn external_edge(&self, internal: EdgeId) -> Result<Option<GraphId<'_>>> {
        Ok(self.graph.repo.external_edge(internal))
    }

    fn outgoing(&self, src: NodeId) -> Result<&[OutgoingEdge]> {
        let slot = self
            .outgoing
            .get(src as usize)
            .with_context(|| format!("node row {src} is out of range"))?;
        Ok(slot
            .get_or_init(|| {
                let (edges, dests) = self.graph.repo.outgoing_slice(src);
                edges
                    .iter()
                    .zip(dests)
                    .map(|(&edge, &dest)| OutgoingEdge { edge, dest })
                    .collect()
            })
            .as_slice())
    }

    fn node(&self, id: NodeId) -> Result<&Self::Node> {
        self.nodes
            .get(id as usize)
            .with_context(|| format!("node row {id} is out of range"))
    }

    fn edge(&self, id: EdgeId) -> Result<&Self::Edge> {
        self.edges
            .get(id as usize)
            .with_context(|| format!("edge row {id} is out of range"))
    }
}

#[derive(Debug, Clone)]
struct RunConfig {
    start_nodes: Vec<crate::OwnedGraphId>,
    max_depth: usize,
    max_paths: Option<usize>,
    strategy: TraversalStrategy,
    max_revisits_per_node: usize,
    intermediate_states: bool,
    progress: bool,
}

#[derive(Debug, Clone)]
struct PathEntry<S> {
    node: NodeId,
    incoming_edge: Option<EdgeId>,
    parent: Option<usize>,
    depth: usize,
    state: S,
}

type InitArena<S> = (Vec<PathEntry<S>>, VecDeque<usize>, SearchStats);

struct EdgeEval<S> {
    edge: EdgeId,
    dest: NodeId,
    state: S,
    stop: bool,
}

/// Runs a serial native row-lazy traversal over `store`.
pub fn search_native<'a, K, G>(
    store: &'a G,
    kernel: K,
    run: RunOptions,
) -> Result<SearchResult<'a, K::Node, K::Edge, K::State>>
where
    K: Kernel,
    G: GraphStore<Node = K::Node, Edge = K::Edge>,
{
    let cfg = RunConfig {
        start_nodes: run.start_nodes,
        max_depth: run.max_depth.unwrap_or(usize::MAX),
        max_paths: run.max_paths,
        strategy: run.strategy,
        max_revisits_per_node: run.max_revisits_per_node,
        intermediate_states: run.intermediate_states,
        progress: run.progress,
    };
    let store: StoreRef<'a, K::Node, K::Edge> = store;
    search_serial(store, &cfg, &kernel)
}

fn search_serial<'a, K>(
    store: StoreRef<'a, K::Node, K::Edge>,
    cfg: &RunConfig,
    kernel: &K,
) -> Result<SearchResult<'a, K::Node, K::Edge, K::State>>
where
    K: Kernel,
{
    let (mut arena, mut frontier, mut stats) = initial_arena(store, cfg, kernel)?;
    let mut paths = Vec::new();
    let mut progress = Progress::new(cfg.progress);

    while let Some(parent) = pop(&mut frontier, cfg.strategy) {
        progress.tick(&stats);
        if arena[parent].depth >= cfg.max_depth {
            continue;
        }

        let parent_node = arena[parent].node;
        store.prefetch_outgoing(&[parent_node])?;
        let outgoing = store.outgoing(parent_node)?;
        let visit_counts = visit_counts_arena(&arena, parent, outgoing.len());
        for &OutgoingEdge { edge, dest } in outgoing {
            let Some(edge) = eval_edge(
                store,
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
                paths.push(materialize(store, &arena, child, cfg)?);
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

fn initial_arena<'a, K>(
    store: StoreRef<'a, K::Node, K::Edge>,
    cfg: &RunConfig,
    kernel: &K,
) -> Result<InitArena<K::State>>
where
    K: Kernel,
{
    let mut arena = Vec::with_capacity(cfg.start_nodes.len());
    let mut frontier = VecDeque::with_capacity(cfg.start_nodes.len());
    let mut stats = SearchStats::default();

    for external in &cfg.start_nodes {
        let node = store
            .resolve_node(external.as_ref())?
            .with_context(|| format!("unknown start node {external}"))?;
        frontier.push_back(arena.len());
        let cx = StartCtx::new(store, node);
        arena.push(PathEntry {
            node,
            incoming_edge: None,
            parent: None,
            depth: 0,
            state: kernel.initial_state(&cx)?,
        });
        stats.start_nodes += 1;
        stats.path_entries += 1;
    }

    Ok((arena, frontier, stats))
}

#[allow(clippy::too_many_arguments)]
fn eval_edge<'a, K>(
    store: StoreRef<'a, K::Node, K::Edge>,
    arena: &[PathEntry<K::State>],
    parent: usize,
    edge: EdgeId,
    dest: NodeId,
    cfg: &RunConfig,
    kernel: &K,
    stats: &mut SearchStats,
    visit_counts: Option<&VisitCounts>,
) -> Result<Option<EdgeEval<K::State>>>
where
    K: Kernel,
{
    if !can_visit_arena(arena, parent, dest, cfg.max_revisits_per_node, visit_counts) {
        stats.skipped_revisits += 1;
        return Ok(None);
    }

    stats.evaluated_edges += 1;
    let cx = EdgeCtx::new(store, arena[parent].node, dest, edge, &arena[parent].state);
    if !kernel.visit(&cx)? {
        stats.rejected_edges += 1;
        return Ok(None);
    }

    let state = kernel.next_state(&cx)?;
    let stop = kernel.stop(&cx.with_state(&state))?;
    Ok(Some(EdgeEval {
        edge,
        dest,
        state,
        stop,
    }))
}

fn push_entry<S>(arena: &mut Vec<PathEntry<S>>, parent: usize, edge: EdgeEval<S>) -> usize {
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

fn materialize<'a, N, E, S: Clone>(
    store: StoreRef<'a, N, E>,
    arena: &[PathEntry<S>],
    mut path: usize,
    cfg: &RunConfig,
) -> Result<Path<'a, N, E, S>> {
    let final_state = arena[path].state.clone();
    let mut node_ids = Vec::with_capacity(arena[path].depth + 1);
    let mut edge_ids = Vec::with_capacity(arena[path].depth);
    let mut states = cfg
        .intermediate_states
        .then(|| Vec::with_capacity(node_ids.capacity()));

    loop {
        node_ids.push(arena[path].node);
        if let Some(edge) = arena[path].incoming_edge {
            edge_ids.push(edge);
        }
        if let Some(states) = &mut states {
            states.push(arena[path].state.clone());
        }
        match arena[path].parent {
            Some(parent) => path = parent,
            None => break,
        }
    }

    node_ids.reverse();
    edge_ids.reverse();
    if let Some(states) = &mut states {
        states.reverse();
    }

    let mut state_iter = states.map(Vec::into_iter);
    let nodes = node_ids
        .into_iter()
        .map(|id| {
            Ok(PathNode {
                id,
                external_id: store.external_node(id)?,
                payload: store.node(id)?,
                state: state_iter.as_mut().map(|states| {
                    states
                        .next()
                        .expect("intermediate state count must match node count")
                }),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let edges = edge_ids
        .into_iter()
        .map(|id| {
            Ok(PathEdge {
                id,
                external_id: store.external_edge(id)?,
                payload: store.edge(id)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Path {
        nodes,
        edges,
        state: final_state,
    })
}

fn pop(frontier: &mut VecDeque<usize>, strategy: TraversalStrategy) -> Option<usize> {
    match strategy {
        TraversalStrategy::BreadthFirst => frontier.pop_front(),
        TraversalStrategy::DepthFirst => frontier.pop_back(),
    }
}

fn visit_counts_arena<S>(
    arena: &[PathEntry<S>],
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

fn can_visit_arena<S>(
    arena: &[PathEntry<S>],
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

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        collections::{BTreeMap, BTreeSet},
        sync::OnceLock,
    };

    use arrow::array::record_batch;
    use pretty_assertions::assert_eq;

    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Account {
        label: &'static str,
        target: bool,
        blocked: bool,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Transfer {
        risk: u64,
        allowed: bool,
    }

    #[derive(Clone, Debug)]
    struct EdgeRow {
        src: NodeId,
        dest: NodeId,
        payload: Transfer,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct RiskState {
        total_risk: u64,
        seen: BTreeSet<NodeId>,
        labels: BTreeMap<NodeId, &'static str>,
    }

    #[derive(Clone, Debug)]
    struct RiskKernel {
        max_risk: u64,
    }

    impl Kernel for RiskKernel {
        type Node = Account;
        type Edge = Transfer;
        type State = RiskState;

        fn initial_state(&self, cx: &StartCtx<'_, Self::Node, Self::Edge>) -> Result<Self::State> {
            let mut seen = BTreeSet::new();
            seen.insert(cx.id());
            let mut labels = BTreeMap::new();
            labels.insert(cx.id(), cx.node()?.label);
            Ok(RiskState {
                total_risk: 0,
                seen,
                labels,
            })
        }

        fn visit(&self, cx: &EdgeCtx<'_, Self::Node, Self::Edge, Self::State>) -> Result<bool> {
            let edge = cx.edge()?;
            let dest = cx.dest()?;
            Ok(edge.allowed
                && !dest.blocked
                && cx.state().total_risk.saturating_add(edge.risk) <= self.max_risk)
        }

        fn next_state(
            &self,
            cx: &EdgeCtx<'_, Self::Node, Self::Edge, Self::State>,
        ) -> Result<Self::State> {
            let edge = cx.edge()?;
            let dest = cx.dest()?;
            let mut next = cx.state().clone();
            next.total_risk += edge.risk;
            next.seen.insert(cx.dest_id());
            next.labels.insert(cx.dest_id(), dest.label);
            Ok(next)
        }

        fn stop(&self, cx: &EdgeCtx<'_, Self::Node, Self::Edge, Self::State>) -> Result<bool> {
            Ok(cx.dest()?.target)
        }
    }

    struct CountingStore {
        nodes: Vec<Account>,
        edges: Vec<EdgeRow>,
        outgoing_index: Vec<Vec<EdgeId>>,
        node_cache: Vec<OnceLock<Account>>,
        edge_cache: Vec<OnceLock<Transfer>>,
        outgoing_cache: Vec<OnceLock<Vec<OutgoingEdge>>>,
        node_loads: RefCell<BTreeSet<NodeId>>,
        edge_loads: RefCell<BTreeSet<EdgeId>>,
        outgoing_loads: RefCell<BTreeSet<NodeId>>,
    }

    impl CountingStore {
        fn new(nodes: Vec<Account>, edges: Vec<EdgeRow>) -> Self {
            let mut outgoing_index = vec![Vec::new(); nodes.len()];
            for (edge, row) in edges.iter().enumerate() {
                outgoing_index[row.src as usize].push(edge as EdgeId);
            }
            let node_cache = (0..nodes.len()).map(|_| OnceLock::new()).collect();
            let edge_cache = (0..edges.len()).map(|_| OnceLock::new()).collect();
            let outgoing_cache = (0..nodes.len()).map(|_| OnceLock::new()).collect();
            Self {
                nodes,
                edges,
                outgoing_index,
                node_cache,
                edge_cache,
                outgoing_cache,
                node_loads: RefCell::new(BTreeSet::new()),
                edge_loads: RefCell::new(BTreeSet::new()),
                outgoing_loads: RefCell::new(BTreeSet::new()),
            }
        }

        fn line() -> Self {
            Self::new(
                vec![
                    Account {
                        label: "start",
                        target: false,
                        blocked: false,
                    },
                    Account {
                        label: "middle",
                        target: false,
                        blocked: false,
                    },
                    Account {
                        label: "target",
                        target: true,
                        blocked: false,
                    },
                    Account {
                        label: "unrelated",
                        target: false,
                        blocked: false,
                    },
                ],
                vec![
                    EdgeRow {
                        src: 0,
                        dest: 1,
                        payload: Transfer {
                            risk: 2,
                            allowed: true,
                        },
                    },
                    EdgeRow {
                        src: 1,
                        dest: 2,
                        payload: Transfer {
                            risk: 3,
                            allowed: true,
                        },
                    },
                    EdgeRow {
                        src: 2,
                        dest: 3,
                        payload: Transfer {
                            risk: 1,
                            allowed: true,
                        },
                    },
                    EdgeRow {
                        src: 3,
                        dest: 2,
                        payload: Transfer {
                            risk: 1,
                            allowed: true,
                        },
                    },
                ],
            )
        }

        fn cycle() -> Self {
            Self::new(
                vec![
                    Account {
                        label: "start",
                        target: false,
                        blocked: false,
                    },
                    Account {
                        label: "middle",
                        target: false,
                        blocked: false,
                    },
                    Account {
                        label: "target",
                        target: true,
                        blocked: false,
                    },
                ],
                vec![
                    EdgeRow {
                        src: 0,
                        dest: 1,
                        payload: Transfer {
                            risk: 1,
                            allowed: true,
                        },
                    },
                    EdgeRow {
                        src: 1,
                        dest: 0,
                        payload: Transfer {
                            risk: 1,
                            allowed: true,
                        },
                    },
                    EdgeRow {
                        src: 1,
                        dest: 2,
                        payload: Transfer {
                            risk: 1,
                            allowed: true,
                        },
                    },
                ],
            )
        }

        fn loaded_nodes(&self) -> BTreeSet<NodeId> {
            self.node_loads.borrow().clone()
        }

        fn loaded_edges(&self) -> BTreeSet<EdgeId> {
            self.edge_loads.borrow().clone()
        }

        fn loaded_outgoing(&self) -> BTreeSet<NodeId> {
            self.outgoing_loads.borrow().clone()
        }
    }

    impl GraphStore for CountingStore {
        type Node = Account;
        type Edge = Transfer;

        fn resolve_node(&self, external: GraphId<'_>) -> Result<Option<NodeId>> {
            Ok(match external {
                GraphId::U64(value) if (value as usize) < self.nodes.len() => Some(value as NodeId),
                _ => None,
            })
        }

        fn external_node(&self, internal: NodeId) -> Result<Option<GraphId<'_>>> {
            Ok(((internal as usize) < self.nodes.len()).then_some(GraphId::U64(internal as u64)))
        }

        fn external_edge(&self, internal: EdgeId) -> Result<Option<GraphId<'_>>> {
            Ok(((internal as usize) < self.edges.len()).then_some(GraphId::U64(internal as u64)))
        }

        fn outgoing(&self, src: NodeId) -> Result<&[OutgoingEdge]> {
            let src_index = src as usize;
            let source = self
                .outgoing_index
                .get(src_index)
                .with_context(|| format!("node row {src} is out of range"))?;
            let cached = self
                .outgoing_cache
                .get(src_index)
                .context("outgoing cache row is missing")?
                .get_or_init(|| {
                    self.outgoing_loads.borrow_mut().insert(src);
                    source
                        .iter()
                        .map(|&edge| OutgoingEdge {
                            edge,
                            dest: self.edges[edge as usize].dest,
                        })
                        .collect()
                });
            Ok(cached)
        }

        fn node(&self, id: NodeId) -> Result<&Self::Node> {
            let id_index = id as usize;
            let source = self
                .nodes
                .get(id_index)
                .with_context(|| format!("node row {id} is out of range"))?;
            Ok(self
                .node_cache
                .get(id_index)
                .context("node cache row is missing")?
                .get_or_init(|| {
                    self.node_loads.borrow_mut().insert(id);
                    source.clone()
                }))
        }

        fn edge(&self, id: EdgeId) -> Result<&Self::Edge> {
            let id_index = id as usize;
            let source = self
                .edges
                .get(id_index)
                .with_context(|| format!("edge row {id} is out of range"))?;
            Ok(self
                .edge_cache
                .get(id_index)
                .context("edge cache row is missing")?
                .get_or_init(|| {
                    self.edge_loads.borrow_mut().insert(id);
                    source.payload.clone()
                }))
        }
    }

    fn run_opts() -> RunOptions {
        RunOptions {
            start_nodes: vec![0_u64.into()],
            strategy: TraversalStrategy::BreadthFirst,
            parallel: true,
            ..RunOptions::default()
        }
    }

    #[test]
    fn native_search_returns_rich_state_without_materializing_unreached_rows() {
        let store = CountingStore::line();
        let result = search_native(&store, RiskKernel { max_risk: 10 }, run_opts()).unwrap();

        assert_eq!(result.paths.len(), 1);
        let path = &result.paths[0];
        assert_eq!(
            path.nodes.iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            path.edges.iter().map(|edge| edge.id).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(path.state.total_risk, 5);
        assert_eq!(path.state.seen, BTreeSet::from([0, 1, 2]));
        assert_eq!(
            path.state.labels,
            BTreeMap::from([(0, "start"), (1, "middle"), (2, "target")])
        );
        assert!(path.nodes.iter().all(|node| node.state.is_none()));

        assert_eq!(store.loaded_nodes(), BTreeSet::from([0, 1, 2]));
        assert_eq!(store.loaded_edges(), BTreeSet::from([0, 1]));
        assert_eq!(store.loaded_outgoing(), BTreeSet::from([0, 1]));
        assert_eq!(result.stats.stopped_paths, 1);
    }

    #[test]
    fn native_intermediate_states_are_attached_to_path_nodes() {
        let store = CountingStore::line();
        let result = search_native(
            &store,
            RiskKernel { max_risk: 10 },
            RunOptions {
                intermediate_states: true,
                ..run_opts()
            },
        )
        .unwrap();

        let path = &result.paths[0];
        assert_eq!(
            path.nodes
                .iter()
                .map(|node| node.state.as_ref().unwrap().total_risk)
                .collect::<Vec<_>>(),
            vec![0, 2, 5]
        );
        assert_eq!(path.nodes.last().unwrap().state.as_ref(), Some(&path.state));
    }

    #[test]
    fn native_search_preserves_revisit_rules_before_payload_loads() {
        let store = CountingStore::cycle();
        let result = search_native(&store, RiskKernel { max_risk: 10 }, run_opts()).unwrap();

        assert_eq!(
            result.paths[0]
                .nodes
                .iter()
                .map(|node| node.id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(result.stats.skipped_revisits, 1);
        assert_eq!(store.loaded_edges(), BTreeSet::from([0, 2]));
    }

    #[test]
    fn eager_graph_store_adapts_existing_graph_topology() {
        let graph = Graph::new(
            record_batch!(
                ("id", Utf8, ["a", "b", "c"]),
                ("kind", Utf8, ["start", "middle", "target"])
            )
            .unwrap(),
            record_batch!(
                ("id", Utf8, ["ab", "bc"]),
                ("src", Utf8, ["a", "b"]),
                ("dest", Utf8, ["b", "c"])
            )
            .unwrap(),
        )
        .unwrap();
        let nodes = vec![
            Account {
                label: "start",
                target: false,
                blocked: false,
            },
            Account {
                label: "middle",
                target: false,
                blocked: false,
            },
            Account {
                label: "target",
                target: true,
                blocked: false,
            },
        ];
        let edges = vec![
            Transfer {
                risk: 2,
                allowed: true,
            },
            Transfer {
                risk: 3,
                allowed: true,
            },
        ];
        let store = EagerGraphStore::new(&graph, &nodes, &edges).unwrap();

        let result = search_native(
            &store,
            RiskKernel { max_risk: 10 },
            RunOptions {
                start_nodes: vec!["a".into()],
                strategy: TraversalStrategy::BreadthFirst,
                ..RunOptions::default()
            },
        )
        .unwrap();

        assert_eq!(
            result.paths[0]
                .nodes
                .iter()
                .map(|node| node.external_id)
                .collect::<Vec<_>>(),
            vec![
                Some(GraphId::Str("a")),
                Some(GraphId::Str("b")),
                Some(GraphId::Str("c"))
            ]
        );
        assert_eq!(
            result.paths[0]
                .edges
                .iter()
                .map(|edge| edge.external_id)
                .collect::<Vec<_>>(),
            vec![Some(GraphId::Str("ab")), Some(GraphId::Str("bc"))]
        );
    }
}
