//! Native Rust traversal kernel abstraction.
//!
//! A [`Kernel`] supplies the same three decisions the DSL makes for every
//! candidate edge `(src)-[edge]->(dest)`:
//!
//! 1. [`Kernel::visit`]: whether the edge may be accepted.
//! 2. [`Kernel::next_state`]: how per-path state changes after accepting it.
//! 3. [`Kernel::stop`]: whether the newly accepted path should be emitted.
//!
//! The engine ([`Graph::search_with`](crate::Graph::search_with)) is generic
//! over `K: Kernel`, so a kernel's per-edge calls are statically dispatched -
//! there is no per-edge vtable. The DSL is implemented as just another kernel
//! (see `impl Kernel for BoundKernel`).
//!
//! [`EdgeCtx`] is the per-edge context handed to a kernel. Its typed payload
//! getters read Arrow columns through a [`PayloadCache`] that binds each
//! `(batch, column)` [`ColumnReader`] at most once and reuses it across all
//! edges of a search, so kernels in hot loops do not re-downcast/clone the
//! underlying Arrow array on every call.

use std::{cell::RefCell, collections::HashMap};

use anyhow::Result;
use arrow::record_batch::RecordBatch;

use crate::{
    dsl::{StateRow, Value, arrow_value::ColumnReader},
    graph::{EdgeId, Graph, GraphId, GraphRepo, NodeId},
};

/// Memoizes [`ColumnReader`]s bound per payload column for a single search.
///
/// Binding a reader downcasts and clones the typed Arrow array; doing it once
/// per `(batch, column)` and reusing the (cheap, Arc-backed) clone avoids
/// repeating that work on every edge in a hot loop.
///
/// Node and edge payloads live in distinct batches that never change during a
/// search (the search borrows `&self`), so caching by column name is safe.
///
/// Not `Sync`: the interior `RefCell`s mean one cache must never be shared
/// across threads. Parallel evaluators create a fresh cache per worker/fold so
/// each stays single-threaded.
pub(crate) struct PayloadCache {
    node: RefCell<HashMap<String, ColumnReader>>,
    edge: RefCell<HashMap<String, ColumnReader>>,
}

impl PayloadCache {
    /// An empty cache for one search run (or one parallel worker).
    pub(crate) fn new() -> Self {
        Self {
            node: RefCell::new(HashMap::new()),
            edge: RefCell::new(HashMap::new()),
        }
    }

    /// Returns a reader for node column `col`, binding and caching on first use.
    fn node_reader(&self, batch: &RecordBatch, col: &str) -> Result<ColumnReader> {
        Self::reader(&self.node, batch, col)
    }

    /// Returns a reader for edge column `col`, binding and caching on first use.
    fn edge_reader(&self, batch: &RecordBatch, col: &str) -> Result<ColumnReader> {
        Self::reader(&self.edge, batch, col)
    }

    /// Looks up `col` in `map`, binding and inserting it if absent.
    ///
    /// Borrow scopes are kept disjoint: an immutable borrow checks for a hit, a
    /// mutable borrow inserts a miss, and the returned clone is produced after
    /// every borrow has been dropped, so no borrow is ever held across a call
    /// that could re-enter the cache.
    fn reader(
        map: &RefCell<HashMap<String, ColumnReader>>,
        batch: &RecordBatch,
        col: &str,
    ) -> Result<ColumnReader> {
        if let Some(reader) = map.borrow().get(col) {
            return Ok(reader.clone());
        }
        let reader = ColumnReader::bind(batch, col)?;
        map.borrow_mut().insert(col.to_string(), reader.clone());
        Ok(reader)
    }
}

/// A native traversal predicate/state machine.
///
/// Implementors decide edge acceptance, state evolution, and stopping. The
/// associated [`State`](Kernel::State) is the per-path payload carried through
/// the search and materialized via [`state_row`](Kernel::state_row).
pub trait Kernel {
    /// Per-path state carried along each path.
    type State: Clone;

    /// Initial state for a path that begins at `start`.
    fn initial_state(&self, graph: &Graph, start: NodeId) -> Self::State;

    /// Whether the candidate edge in `cx` may be accepted.
    fn visit(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<bool>;

    /// State for the child path after accepting the edge in `cx`.
    ///
    /// The parent state is available as [`EdgeCtx::state`].
    fn next_state(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Self::State>;

    /// Whether the accepted path should be emitted.
    ///
    /// `cx` carries the *child* state produced by [`next_state`](Kernel::next_state).
    fn stop(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<bool>;

    /// Materializes the named state row for a returned path.
    fn state_row(&self, state: &Self::State) -> StateRow;
}

/// Per-edge context passed to a [`Kernel`].
///
/// Holds the current edge `(src)-[edge]->(dest)` and the relevant per-path
/// state. Typed getters read node/edge Arrow payload columns by internal row
/// id.
pub struct EdgeCtx<'a, S> {
    graph: &'a Graph,
    src: NodeId,
    dest: NodeId,
    edge: EdgeId,
    state: &'a S,
    cache: &'a PayloadCache,
}

impl<'a, S> EdgeCtx<'a, S> {
    /// Builds a context for the candidate edge `(src)-[edge]->(dest)`.
    ///
    /// `cache` is borrowed (not owned) so its bound readers persist across all
    /// edges of a search rather than being rebuilt per [`EdgeCtx`].
    pub(crate) fn new(
        graph: &'a Graph,
        src: NodeId,
        dest: NodeId,
        edge: EdgeId,
        state: &'a S,
        cache: &'a PayloadCache,
    ) -> Self {
        Self {
            graph,
            src,
            dest,
            edge,
            state,
            cache,
        }
    }

    /// Returns a context borrowing `state` in place of the current one.
    ///
    /// Mirrors the DSL's `with_state`; used to evaluate `stop` against the
    /// child state produced by `next_state`.
    pub fn with_state<'b>(&'b self, state: &'b S) -> EdgeCtx<'b, S> {
        EdgeCtx {
            graph: self.graph,
            src: self.src,
            dest: self.dest,
            edge: self.edge,
            state,
            cache: self.cache,
        }
    }

    /// The graph being traversed.
    pub fn graph(&self) -> &'a Graph {
        self.graph
    }

    /// Internal id of the edge's source node.
    pub fn src(&self) -> NodeId {
        self.src
    }

    /// Internal id of the edge's destination node.
    pub fn dest(&self) -> NodeId {
        self.dest
    }

    /// Internal id of the current edge.
    pub fn edge(&self) -> EdgeId {
        self.edge
    }

    /// The per-path state.
    pub fn state(&self) -> &S {
        self.state
    }

    /// External id of the source node, if present.
    pub fn src_id(&self) -> Option<GraphId<'a>> {
        self.graph.repo.external_node(self.src)
    }

    /// External id of the destination node, if present.
    pub fn dest_id(&self) -> Option<GraphId<'a>> {
        self.graph.repo.external_node(self.dest)
    }

    /// External id of the current edge, if present.
    pub fn edge_id(&self) -> Option<GraphId<'a>> {
        self.graph.repo.external_edge(self.edge)
    }

    /// Reads node payload column `col` for the source node.
    pub fn src_value(&self, col: &str) -> Result<Value> {
        self.cache
            .node_reader(self.graph.repo.node_batch(), col)?
            .value(self.src as usize)
    }

    /// Reads node payload column `col` for the destination node.
    pub fn dest_value(&self, col: &str) -> Result<Value> {
        self.cache
            .node_reader(self.graph.repo.node_batch(), col)?
            .value(self.dest as usize)
    }

    /// Reads edge payload column `col` for the current edge.
    pub fn edge_value(&self, col: &str) -> Result<Value> {
        self.cache
            .edge_reader(self.graph.repo.edge_batch(), col)?
            .value(self.edge as usize)
    }

    /// Source node payload `col` as `u64`.
    ///
    /// Coerces from integer/float when lossless; `Ok(None)` if null. Errors on
    /// negative or non-integral values.
    pub fn src_u64(&self, col: &str) -> Result<Option<u64>> {
        as_u64(self.src_value(col)?)
    }

    /// Destination node payload `col` as `u64`.
    ///
    /// Coerces from integer/float when lossless; `Ok(None)` if null. Errors on
    /// negative or non-integral values.
    pub fn dest_u64(&self, col: &str) -> Result<Option<u64>> {
        as_u64(self.dest_value(col)?)
    }

    /// Edge payload `col` as `u64`.
    ///
    /// Coerces from integer/float when lossless; `Ok(None)` if null. Errors on
    /// negative or non-integral values.
    pub fn edge_u64(&self, col: &str) -> Result<Option<u64>> {
        as_u64(self.edge_value(col)?)
    }

    /// Source node payload `col` as `i64`.
    ///
    /// Coerces from integer/float when lossless; `Ok(None)` if null. Errors on
    /// out-of-range or non-integral values.
    pub fn src_i64(&self, col: &str) -> Result<Option<i64>> {
        as_i64(self.src_value(col)?)
    }

    /// Destination node payload `col` as `i64`.
    ///
    /// Coerces from integer/float when lossless; `Ok(None)` if null. Errors on
    /// out-of-range or non-integral values.
    pub fn dest_i64(&self, col: &str) -> Result<Option<i64>> {
        as_i64(self.dest_value(col)?)
    }

    /// Edge payload `col` as `i64`.
    ///
    /// Coerces from integer/float when lossless; `Ok(None)` if null. Errors on
    /// out-of-range or non-integral values.
    pub fn edge_i64(&self, col: &str) -> Result<Option<i64>> {
        as_i64(self.edge_value(col)?)
    }

    /// Source node payload `col` as `f64`.
    ///
    /// Coerces from any numeric type; `Ok(None)` if null.
    pub fn src_f64(&self, col: &str) -> Result<Option<f64>> {
        as_f64(self.src_value(col)?)
    }

    /// Destination node payload `col` as `f64`.
    ///
    /// Coerces from any numeric type; `Ok(None)` if null.
    pub fn dest_f64(&self, col: &str) -> Result<Option<f64>> {
        as_f64(self.dest_value(col)?)
    }

    /// Edge payload `col` as `f64`.
    ///
    /// Coerces from any numeric type; `Ok(None)` if null.
    pub fn edge_f64(&self, col: &str) -> Result<Option<f64>> {
        as_f64(self.edge_value(col)?)
    }

    /// Source node payload `col` as `bool`. `Ok(None)` if null.
    pub fn src_bool(&self, col: &str) -> Result<Option<bool>> {
        as_bool(self.src_value(col)?)
    }

    /// Destination node payload `col` as `bool`. `Ok(None)` if null.
    pub fn dest_bool(&self, col: &str) -> Result<Option<bool>> {
        as_bool(self.dest_value(col)?)
    }

    /// Edge payload `col` as `bool`. `Ok(None)` if null.
    pub fn edge_bool(&self, col: &str) -> Result<Option<bool>> {
        as_bool(self.edge_value(col)?)
    }

    /// Source node payload `col` as `String`. `Ok(None)` if null.
    pub fn src_str(&self, col: &str) -> Result<Option<String>> {
        as_str(self.src_value(col)?)
    }

    /// Destination node payload `col` as `String`. `Ok(None)` if null.
    pub fn dest_str(&self, col: &str) -> Result<Option<String>> {
        as_str(self.dest_value(col)?)
    }

    /// Edge payload `col` as `String`. `Ok(None)` if null.
    pub fn edge_str(&self, col: &str) -> Result<Option<String>> {
        as_str(self.edge_value(col)?)
    }
}

/// Coerces a [`Value`] to `u64` with SAFE (error-on-loss) semantics.
///
/// Accepts `U64` directly, non-negative `I64`, and integral non-negative `F64`
/// within `u64` range. `Null` yields `Ok(None)`. The `fract() == 0.0` integral
/// check is an intentional exact float comparison (guarded by `is_finite`).
#[allow(clippy::float_cmp)]
fn as_u64(value: Value) -> Result<Option<u64>> {
    match value {
        Value::Null => Ok(None),
        Value::U64(v) => Ok(Some(v)),
        Value::I64(v) => {
            if v >= 0 {
                Ok(Some(v as u64))
            } else {
                anyhow::bail!("cannot read negative value {v} as u64")
            }
        }
        Value::F64(v) => {
            if v.is_finite() && v.fract() == 0.0 && v >= 0.0 && v <= u64::MAX as f64 {
                Ok(Some(v as u64))
            } else {
                anyhow::bail!("cannot read f64 {v} as u64 without loss")
            }
        }
        other => anyhow::bail!("expected an integer value, got {other:?}"),
    }
}

/// Coerces a [`Value`] to `i64` with SAFE (error-on-loss) semantics.
///
/// Accepts `I64` directly, `U64` within `i64` range, and integral `F64` within
/// `i64` range. `Null` yields `Ok(None)`. The `fract() == 0.0` integral check is
/// an intentional exact float comparison (guarded by `is_finite`).
#[allow(clippy::float_cmp)]
fn as_i64(value: Value) -> Result<Option<i64>> {
    match value {
        Value::Null => Ok(None),
        Value::I64(v) => Ok(Some(v)),
        Value::U64(v) => {
            if v <= i64::MAX as u64 {
                Ok(Some(v as i64))
            } else {
                anyhow::bail!("cannot read u64 {v} as i64 (out of range)")
            }
        }
        Value::F64(v) => {
            if v.is_finite() && v.fract() == 0.0 && v >= i64::MIN as f64 && v <= i64::MAX as f64 {
                Ok(Some(v as i64))
            } else {
                anyhow::bail!("cannot read f64 {v} as i64 without loss")
            }
        }
        other => anyhow::bail!("expected an integer value, got {other:?}"),
    }
}

/// Coerces a [`Value`] to `f64`, widening any numeric type.
///
/// `I64`/`U64` are widened to `f64`; very large integers may lose precision,
/// which is acceptable and matches the DSL (which widens everything to `f64`).
/// `Null` yields `Ok(None)`.
fn as_f64(value: Value) -> Result<Option<f64>> {
    match value {
        Value::Null => Ok(None),
        Value::F64(v) => Ok(Some(v)),
        Value::I64(v) => Ok(Some(v as f64)),
        Value::U64(v) => Ok(Some(v as f64)),
        other => anyhow::bail!("expected a numeric value, got {other:?}"),
    }
}

fn as_bool(value: Value) -> Result<Option<bool>> {
    match value {
        Value::Null => Ok(None),
        Value::Bool(v) => Ok(Some(v)),
        other => anyhow::bail!("expected bool value, got {other:?}"),
    }
}

fn as_str(value: Value) -> Result<Option<String>> {
    match value {
        Value::Null => Ok(None),
        Value::Str(v) => Ok(Some(v.to_string())),
        other => anyhow::bail!("expected string value, got {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{as_f64, as_i64, as_u64};
    use crate::dsl::Value;

    #[test]
    fn as_u64_coercion() {
        assert_eq!(as_u64(Value::U64(5)).unwrap(), Some(5));
        assert_eq!(as_u64(Value::I64(5)).unwrap(), Some(5));
        assert!(as_u64(Value::I64(-1)).is_err());
        assert_eq!(as_u64(Value::F64(3.0)).unwrap(), Some(3));
        assert!(as_u64(Value::F64(3.5)).is_err());
        assert!(as_u64(Value::F64(-1.0)).is_err());
        assert_eq!(as_u64(Value::Null).unwrap(), None);
        assert!(as_u64(Value::Bool(true)).is_err());
    }

    #[test]
    fn as_i64_coercion() {
        assert_eq!(as_i64(Value::I64(-3)).unwrap(), Some(-3));
        assert_eq!(as_i64(Value::U64(7)).unwrap(), Some(7));
        assert!(as_i64(Value::U64(u64::MAX)).is_err());
        assert_eq!(as_i64(Value::F64(4.0)).unwrap(), Some(4));
        assert!(as_i64(Value::F64(4.5)).is_err());
        assert_eq!(as_i64(Value::Null).unwrap(), None);
    }

    #[test]
    fn as_f64_coercion() {
        assert_eq!(as_f64(Value::F64(1.5)).unwrap(), Some(1.5));
        assert_eq!(as_f64(Value::I64(-2)).unwrap(), Some(-2.0));
        assert_eq!(as_f64(Value::U64(9)).unwrap(), Some(9.0));
        assert_eq!(as_f64(Value::Null).unwrap(), None);
        assert!(as_f64(Value::Bool(true)).is_err());
    }
}
