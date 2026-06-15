//! Row-lazy native Rust traversal.
//!
//! This module is separate from [`Graph::search`](crate::Graph::search). It is
//! for callers that own a backend capable of resolving topology and native
//! payload rows lazily, then want traversal kernels to operate on Rust structs
//! instead of Arrow-backed [`Value`](crate::Value) rows.

use std::sync::OnceLock;

use anyhow::{Context, Result};

use crate::{
    dsl::StateRow,
    graph::{EdgeId, Graph, GraphId, GraphRepo, NodeId},
    traversal::{
        RunOptions, SearchStats,
        engine::{self, PathEntry, SearchAdapter},
    },
};

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
pub struct EdgeCtx<'store, 'state, N, E, S> {
    store: StoreRef<'store, N, E>,
    src: NodeId,
    dest: NodeId,
    edge: EdgeId,
    state: &'state S,
}

impl<'store, 'state, N, E, S> EdgeCtx<'store, 'state, N, E, S> {
    fn new(
        store: StoreRef<'store, N, E>,
        src: NodeId,
        dest: NodeId,
        edge: EdgeId,
        state: &'state S,
    ) -> Self {
        Self {
            store,
            src,
            dest,
            edge,
            state,
        }
    }

    fn with_state<'b>(&'b self, state: &'b S) -> EdgeCtx<'store, 'b, N, E, S> {
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
    pub fn src_external_id(&self) -> Result<Option<GraphId<'store>>> {
        self.store.external_node(self.src)
    }

    /// External destination node ID, if available.
    pub fn dest_external_id(&self) -> Result<Option<GraphId<'store>>> {
        self.store.external_node(self.dest)
    }

    /// External edge ID, if available.
    pub fn edge_external_id(&self) -> Result<Option<GraphId<'store>>> {
        self.store.external_edge(self.edge)
    }

    /// Native source node payload.
    pub fn src(&self) -> Result<&'store N> {
        self.store.node(self.src)
    }

    /// Native destination node payload.
    pub fn dest(&self) -> Result<&'store N> {
        self.store.node(self.dest)
    }

    /// Native edge payload.
    pub fn edge(&self) -> Result<&'store E> {
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
    fn visit(&self, cx: &EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>) -> Result<bool>;

    /// State for the child path after accepting the edge in `cx`.
    fn next_state(
        &self,
        cx: &EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<Self::State>;

    /// Whether the accepted path should be emitted.
    ///
    /// `cx` carries the child state produced by [`Kernel::next_state`].
    fn stop(&self, cx: &EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>) -> Result<bool>;

    /// Materializes the path state into the public named-row representation.
    fn state_row(&self, state: &Self::State) -> StateRow;
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
    let adapter = NativeSearchAdapter {
        store,
        kernel: &kernel,
    };
    let result = engine::search_serial(&adapter, run)?;
    Ok(SearchResult {
        paths: result.paths,
        stats: result.stats,
    })
}

struct NativeSearchAdapter<'store, 'kernel, G, K> {
    store: &'store G,
    kernel: &'kernel K,
}

impl<'store, 'kernel, G, K> SearchAdapter for NativeSearchAdapter<'store, 'kernel, G, K>
where
    K: Kernel,
    G: GraphStore<Node = K::Node, Edge = K::Edge>,
    K::Node: 'store,
    K::Edge: 'store,
{
    type State = K::State;
    type Path = Path<'store, K::Node, K::Edge, K::State>;
    type Cache = ();

    fn resolve_node(&self, external: GraphId<'_>) -> Result<Option<NodeId>> {
        self.store.resolve_node(external)
    }

    fn initial_state(&self, node: NodeId) -> Result<Self::State> {
        let store: StoreRef<'store, K::Node, K::Edge> = self.store;
        let cx = StartCtx::new(store, node);
        self.kernel.initial_state(&cx)
    }

    fn out_degree(&self, node: NodeId) -> Result<usize> {
        self.store.prefetch_outgoing(&[node])?;
        Ok(self.store.outgoing(node)?.len())
    }

    fn for_each_outgoing<F>(&self, node: NodeId, mut visit: F) -> Result<()>
    where
        F: FnMut(EdgeId, NodeId) -> Result<bool>,
    {
        for &OutgoingEdge { edge, dest } in self.store.outgoing(node)? {
            if !visit(edge, dest)? {
                break;
            }
        }
        Ok(())
    }

    fn make_cache(&self) -> Self::Cache {}

    fn eval_edge(
        &self,
        src: NodeId,
        edge: EdgeId,
        dest: NodeId,
        state: &Self::State,
        _cache: &Self::Cache,
    ) -> Result<Option<(Self::State, bool)>> {
        let store: StoreRef<'store, K::Node, K::Edge> = self.store;
        let cx = EdgeCtx::new(store, src, dest, edge, state);
        if !self.kernel.visit(&cx)? {
            return Ok(None);
        }
        let state = self.kernel.next_state(&cx)?;
        let stop = self.kernel.stop(&cx.with_state(&state))?;
        Ok(Some((state, stop)))
    }

    fn materialize(
        &self,
        arena: &[PathEntry<Self::State>],
        mut path: usize,
        intermediate_states: bool,
    ) -> Result<Self::Path> {
        let final_state = arena[path].state.clone();
        let mut node_ids = Vec::with_capacity(arena[path].depth + 1);
        let mut edge_ids = Vec::with_capacity(arena[path].depth);
        let mut states = intermediate_states.then(|| Vec::with_capacity(node_ids.capacity()));

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
                    external_id: self.store.external_node(id)?,
                    payload: self.store.node(id)?,
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
                    external_id: self.store.external_edge(id)?,
                    payload: self.store.edge(id)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Path {
            nodes,
            edges,
            state: final_state,
        })
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

    use crate::{TraversalStrategy, dsl::Value};

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

        fn visit(&self, cx: &EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>) -> Result<bool> {
            let edge = cx.edge()?;
            let dest = cx.dest()?;
            Ok(edge.allowed
                && !dest.blocked
                && cx.state().total_risk.saturating_add(edge.risk) <= self.max_risk)
        }

        fn next_state(
            &self,
            cx: &EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
        ) -> Result<Self::State> {
            let edge = cx.edge()?;
            let dest = cx.dest()?;
            let mut next = cx.state().clone();
            next.total_risk += edge.risk;
            next.seen.insert(cx.dest_id());
            next.labels.insert(cx.dest_id(), dest.label);
            Ok(next)
        }

        fn stop(&self, cx: &EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>) -> Result<bool> {
            Ok(cx.dest()?.target)
        }

        fn state_row(&self, state: &Self::State) -> StateRow {
            vec![("total_risk".to_string(), Value::U64(state.total_risk))]
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
