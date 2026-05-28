//! Traversal configuration.
//!
//! [`TraversalConfigBuilder`] is the public construction API. It keeps the
//! execution knobs explicit while leaving the immutable [`TraversalConfig`] cheap
//! to pass into [`Graph::search`](crate::graph::Graph::search).

use crate::{dsl::DslKernel, graph::OwnedGraphId};

/// Search order used by a traversal.
#[derive(Debug, Clone, Copy, Default)]
pub enum TraversalStrategy {
    /// Expand all paths at depth `n` before depth `n + 1`.
    #[default]
    BreadthFirst,
    /// Follow the newest accepted path first.
    DepthFirst,
}

/// Fully configured DSL traversal.
///
/// Prefer [`TraversalConfigBuilder`] unless constructing configs mechanically.
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Predicate/state kernel evaluated for each candidate edge.
    pub kernel: DslKernel,
    /// External node IDs where traversal starts.
    pub start_nodes: Vec<OwnedGraphId>,
    /// Maximum accepted-edge depth.
    pub max_depth: Option<usize>,
    /// Maximum returned path count.
    ///
    /// Parallel traversal treats this as a soft early-stop limit internally and
    /// truncates the returned path vector exactly.
    pub max_paths: Option<usize>,
    /// Search order.
    pub strategy: TraversalStrategy,
    /// Maximum revisits allowed per node inside one path.
    pub max_revisits_per_node: usize,
    /// Whether Rayon-backed parallel traversal is enabled.
    pub parallel: bool,
}

/// Builder for a [`TraversalConfig`].
///
/// The builder defaults to depth-first traversal, no depth/path limits, no node
/// revisits inside a path, and parallel traversal enabled.
///
/// ```
/// use rxgraph::{DslExpr as e, DslKernel, Scalar, TraversalConfigBuilder};
///
/// let kernel = DslKernel::new(
///     e::edge("enabled"),
///     std::iter::empty::<(String, rxgraph::DslExpr)>(),
///     e::dest("is_target"),
///     [("seen".into(), Scalar::Bool(true))],
/// );
/// let config = TraversalConfigBuilder::new(kernel)
///     .with_start_nodes([0_u64])
///     .with_max_depth(4)
///     .with_max_paths(100)
///     .with_parallelism(true)
///     .build();
/// ```
#[derive(Debug)]
pub struct TraversalConfigBuilder {
    kernel: DslKernel,
    start_nodes: Vec<OwnedGraphId>,
    max_depth: Option<usize>,
    max_paths: Option<usize>,
    strategy: TraversalStrategy,
    max_revisits_per_node: usize,
    parallel: bool,
}

impl TraversalConfigBuilder {
    /// Starts a traversal builder for `kernel`.
    pub fn new(kernel: DslKernel) -> Self {
        Self {
            kernel,
            start_nodes: Vec::new(),
            max_depth: None,
            max_paths: None,
            strategy: TraversalStrategy::DepthFirst,
            max_revisits_per_node: 0,
            parallel: true,
        }
    }

    /// Sets external node IDs where traversal begins.
    ///
    /// IDs must match the graph's ID mode: all integer IDs or all string IDs.
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
    ///
    /// The returned result is always truncated to this limit. With parallel
    /// traversal, some in-flight work may finish after the limit is reached.
    pub fn with_max_paths(mut self, paths: usize) -> Self {
        self.max_paths = Some(paths);
        self
    }

    /// Chooses depth-first or breadth-first traversal.
    pub fn with_strategy(mut self, strategy: TraversalStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enables or disables Rayon-backed parallel traversal.
    ///
    /// Parallel traversal is enabled by default. Path ordering is unspecified
    /// when this is true.
    pub fn with_parallelism(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    /// Builds the immutable traversal configuration.
    pub fn build(self) -> TraversalConfig {
        TraversalConfig {
            kernel: self.kernel,
            start_nodes: self.start_nodes,
            max_depth: self.max_depth,
            max_paths: self.max_paths,
            strategy: self.strategy,
            max_revisits_per_node: self.max_revisits_per_node,
            parallel: self.parallel,
        }
    }
}
