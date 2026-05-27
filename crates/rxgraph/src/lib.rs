//! Graph traversal over Arrow-backed node and edge tables.
//!
//! `rxgraph` stores graph topology in compact internal IDs while keeping all
//! node and edge attributes in Arrow arrays. Traversals are driven by a small
//! expression kernel that can be serialized from higher-level frontends.

mod dsl;
mod graph;
mod traversal;

#[cfg(test)]
mod test_utils;

pub use dsl::{DslKernel, Scalar};
pub use graph::{EdgeId, Graph, GraphBuilder, NodeId};
pub use traversal::{
    DslTraversal, DslTraversalBuilder, Parallelism, SearchPath, SearchResult, SearchStats,
    TraversalStrategy,
};
