//! Arrow-backed graph traversal.
//!
//! `rxgraph` stores node and edge tables as Arrow [`RecordBatch`](arrow::record_batch::RecordBatch)
//! values and builds compact CSR topology for traversal. The graph schema is deliberately
//! small:
//!
//! - nodes require `id`
//! - edges require `id`, `src`, and `dest`
//! - all identity columns must be either `UInt64` or string
//! - optional `type` columns must be string
//!
//! Use [`Graph::new`] to build a graph, [`TraversalConfigBuilder`] plus
//! [`DslKernel`] for stateful path search, and the convenience methods on
//! [`Graph`] for simple BFS/DFS, shortest path, degree, and component queries.
//!
//! Traversal evaluates a [`DslKernel`] against every candidate edge. The kernel
//! decides whether the edge is accepted, how named path state changes, and
//! whether the accepted path should be returned. Search uses compact internal
//! IDs and materializes external [`GraphId`] values only in returned results.

mod arrow;
pub mod dsl;
pub mod graph;
pub mod traversal;

pub use dsl::{DslExpr, DslKernel, Scalar, StateRow, Value};
pub use graph::{EdgeId, Graph, GraphId, GraphRepo, NodeId, OwnedGraphId};
pub use traversal::{
    GraphPath, SearchResult, SearchStats, TraversalConfig, TraversalConfigBuilder,
    TraversalStrategy,
};
