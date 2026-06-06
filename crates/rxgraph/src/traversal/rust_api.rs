//! Rust-native traversal kernels.
//!
//! This API is for Rust callers that want handwritten, monomorphized traversal
//! logic instead of the portable DSL/Polars expression kernel.

use crate::{
    graph::{EdgeId, Graph, GraphId, NodeId, OwnedGraphId},
    traversal::{SearchStats, TraversalStrategy},
};

/// Candidate edge context passed to a Rust-native traversal kernel.
///
/// `src`, `dest`, and `edge` are compact internal IDs. For graphs built with
/// [`Graph::from_u64_edges`](crate::graph::Graph::from_u64_edges), those node
/// and edge IDs match the external contiguous `u64` IDs.
pub struct RustEdgeContext<'a, S> {
    pub graph: &'a Graph,
    pub src: NodeId,
    pub dest: NodeId,
    pub edge: EdgeId,
    pub state: &'a S,
}

impl<S> Copy for RustEdgeContext<'_, S> {}

impl<S> Clone for RustEdgeContext<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Rust-native predicate/state kernel.
///
/// Use [`RustKernel`] for closure-based kernels. Implement this trait directly
/// when a named type gives clearer ownership over captured data.
pub trait RustSearchKernel {
    type State: Clone;

    fn initial_state(&self) -> Self::State;
    fn visit(&self, ctx: RustEdgeContext<'_, Self::State>) -> bool;
    fn next_state(&self, ctx: RustEdgeContext<'_, Self::State>) -> Self::State;
    fn stop(&self, ctx: RustEdgeContext<'_, Self::State>) -> bool;
}

/// Closure-backed Rust-native traversal kernel.
pub struct RustKernel<S, V, N, T> {
    initial_state: S,
    visit: V,
    next_state: N,
    stop: T,
}

impl<S, V, N, T> RustKernel<S, V, N, T> {
    pub fn new(initial_state: S, visit: V, next_state: N, stop: T) -> Self {
        Self {
            initial_state,
            visit,
            next_state,
            stop,
        }
    }
}

impl<S, V, N, T> RustSearchKernel for RustKernel<S, V, N, T>
where
    S: Clone,
    V: for<'a> Fn(RustEdgeContext<'a, S>) -> bool,
    N: for<'a> Fn(RustEdgeContext<'a, S>) -> S,
    T: for<'a> Fn(RustEdgeContext<'a, S>) -> bool,
{
    type State = S;

    fn initial_state(&self) -> Self::State {
        self.initial_state.clone()
    }

    fn visit(&self, ctx: RustEdgeContext<'_, Self::State>) -> bool {
        (self.visit)(ctx)
    }

    fn next_state(&self, ctx: RustEdgeContext<'_, Self::State>) -> Self::State {
        (self.next_state)(ctx)
    }

    fn stop(&self, ctx: RustEdgeContext<'_, Self::State>) -> bool {
        (self.stop)(ctx)
    }
}

/// Fully configured Rust-native traversal.
pub struct RustTraversalConfig<K> {
    pub(crate) kernel: K,
    pub(crate) start_nodes: Vec<OwnedGraphId>,
    pub(crate) max_depth: Option<usize>,
    pub(crate) max_paths: Option<usize>,
    pub(crate) strategy: TraversalStrategy,
    pub(crate) max_revisits_per_node: usize,
    pub(crate) intermediate_states: bool,
    pub(crate) progress: bool,
}

/// Builder for [`RustTraversalConfig`].
///
/// This mirrors the DSL traversal builder but stays serial: Rust closures are
/// intended as the low-overhead apples-to-apples path for handwritten kernels.
#[derive(Debug)]
pub struct RustTraversalConfigBuilder<K> {
    kernel: K,
    start_nodes: Vec<OwnedGraphId>,
    max_depth: Option<usize>,
    max_paths: Option<usize>,
    strategy: TraversalStrategy,
    max_revisits_per_node: usize,
    intermediate_states: bool,
    progress: bool,
}

impl<K> RustTraversalConfigBuilder<K> {
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            start_nodes: Vec::new(),
            max_depth: None,
            max_paths: None,
            strategy: TraversalStrategy::DepthFirst,
            max_revisits_per_node: 0,
            intermediate_states: false,
            progress: false,
        }
    }

    /// Sets external node IDs where traversal begins.
    pub fn with_start_nodes<I, N>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = N>,
        N: Into<OwnedGraphId>,
    {
        self.start_nodes = nodes.into_iter().map(Into::into).collect();
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

    /// Enables or disables per-node state history on returned paths.
    pub fn with_intermediate_states(mut self, enabled: bool) -> Self {
        self.intermediate_states = enabled;
        self
    }

    /// Reports search progress on stderr.
    pub fn with_progress(mut self, enabled: bool) -> Self {
        self.progress = enabled;
        self
    }

    pub fn build(self) -> RustTraversalConfig<K> {
        RustTraversalConfig {
            kernel: self.kernel,
            start_nodes: self.start_nodes,
            max_depth: self.max_depth,
            max_paths: self.max_paths,
            strategy: self.strategy,
            max_revisits_per_node: self.max_revisits_per_node,
            intermediate_states: self.intermediate_states,
            progress: self.progress,
        }
    }
}

/// One typed path returned by [`Graph::search_rust`](crate::graph::Graph::search_rust).
#[derive(Debug, Clone, PartialEq)]
pub struct RustGraphPath<'a, S> {
    pub nodes: Vec<GraphId<'a>>,
    pub edges: Vec<GraphId<'a>>,
    pub state: S,
    pub intermediate_states: Option<Vec<S>>,
}

/// Typed result returned by [`Graph::search_rust`](crate::graph::Graph::search_rust).
#[derive(Debug)]
pub struct RustSearchResult<'a, S> {
    pub paths: Vec<RustGraphPath<'a, S>>,
    pub stats: SearchStats,
}
