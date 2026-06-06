//! Path traversal over a [`Graph`](crate::graph::Graph).
//!
//! Traversal is configured with [`TraversalConfigBuilder`] and a
//! [`DslKernel`](crate::dsl::DslKernel). The kernel decides which candidate
//! edges are accepted, how per-path state changes, and when a path is emitted.
//!
//! Returned paths contain external graph IDs, not compact internal indexes.
//! Ordering is stable for serial traversal and intentionally unspecified when
//! parallel traversal is enabled.
//!
//! The public flow is:
//!
//! 1. Build a [`DslKernel`](crate::dsl::DslKernel).
//! 2. Build a [`TraversalConfig`] with [`TraversalConfigBuilder`].
//! 3. Call [`Graph::search`](crate::graph::Graph::search).
//!
//! `max_paths` limits the returned vector exactly. When parallel traversal is
//! enabled, workers may complete a little extra in-flight work, so
//! [`SearchStats`] can report more stopped paths than the returned path count.

mod algo;
mod config;
mod kernel;
mod progress;
mod registry;

use crate::{dsl::StateRow, graph::GraphId};

pub use algo::RunOptions;
pub use config::{TraversalConfig, TraversalConfigBuilder, TraversalStrategy};
pub use kernel::{EdgeCtx, Kernel};
pub use registry::{
    BoxedRun, KernelEntry, RunKernel, boxed_run, build_kernel, inventory, register_kernel,
};

/// One materialized path returned by a traversal.
///
/// IDs are borrowed from the graph. For integer-ID graphs the variants are
/// [`GraphId::U64`](crate::graph::GraphId::U64); for string-ID graphs they are
/// [`GraphId::Str`](crate::graph::GraphId::Str).
#[derive(Debug, Clone, PartialEq)]
pub struct GraphPath<'a> {
    /// External node IDs in path order, including the start and final node.
    pub nodes: Vec<GraphId<'a>>,
    /// Edge IDs in path order.
    pub edges: Vec<GraphId<'a>>,
    /// Final named state after the last accepted edge.
    ///
    /// For a zero-edge path this is the kernel's initial state.
    pub state: StateRow,
    /// Optional per-node state history in path order.
    ///
    /// Present only when [`TraversalConfigBuilder::with_intermediate_states`]
    /// is enabled. The first entry is the initial state at the start node; the
    /// final entry equals [`GraphPath::state`].
    pub intermediate_states: Option<Vec<StateRow>>,
}

/// Result of a graph traversal.
///
/// `paths` contains only stopped paths, never intermediate frontier states.
/// Use `stats` to inspect how much work was evaluated.
#[derive(Debug)]
pub struct SearchResult<'a> {
    /// Materialized paths. Order is unspecified when parallel traversal is enabled.
    pub paths: Vec<GraphPath<'a>>,
    /// Counters for the completed work.
    pub stats: SearchStats,
}

/// Traversal counters.
///
/// With parallel traversal, `max_paths` is a soft early-stop limit for in-flight
/// work. Returned paths are truncated exactly, while stats describe completed
/// evaluated work.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SearchStats {
    /// Number of resolved start nodes.
    pub start_nodes: usize,
    /// Number of path states allocated or created.
    pub path_entries: usize,
    /// Candidate edges evaluated by the DSL kernel.
    pub evaluated_edges: usize,
    /// Evaluated edges accepted by `visit`.
    pub accepted_edges: usize,
    /// Evaluated edges rejected by `visit`.
    pub rejected_edges: usize,
    /// Candidate edges skipped before evaluation because of revisit limits.
    pub skipped_revisits: usize,
    /// Accepted paths where `stop` evaluated to true.
    pub stopped_paths: usize,
    /// Maximum accepted-edge depth reached by any completed path state.
    pub max_depth: usize,
}
