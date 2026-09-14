//! Pull-based native search. No work is scheduled between calls to `next`.
use super::{
    OwnedGraphPath, OwnedSearchResult,
    algo::GraphSearchAdapter,
    engine::cursor::{Cursor, GraphCursor},
    kernel::PayloadCache,
    native::{self, NativeSearchAdapter},
};
use crate::{Graph, GraphPath, Kernel, RunOptions, SearchStats, StateRow};
use anyhow::{Result, bail};
use std::{marker::PhantomData, sync::Arc};

/// An owned plugin session. Dispatch occurs once per batch, never per edge.
pub trait NativeBatchStream: Send {
    /// Returns up to `size` paths, or `None` after exhaustion. Zero is invalid.
    fn next_batch(&mut self, size: usize) -> Result<Option<OwnedSearchResult>>;
    fn stats(&self) -> SearchStats;
    fn close(&mut self);
}

/// A native graph cursor retaining typed state and borrowing graph identity.
pub struct GraphSearchStream<'g, K: Kernel> {
    graph: &'g Graph,
    kernel: K,
    cursor: GraphCursor<K::State, PayloadCache>,
    batch_size: usize,
}
impl Graph {
    /// Starts a native pull stream. `batch_size` is a positive maximum, not a minimum.
    pub fn search_batches_with<K: Kernel + Sync>(
        &self,
        kernel: K,
        run: RunOptions,
        batch_size: usize,
    ) -> Result<GraphSearchStream<'_, K>>
    where
        K::State: Send + Sync + Clone,
    {
        if batch_size == 0 {
            bail!("batch_size must be at least 1");
        }
        Ok(GraphSearchStream {
            graph: self,
            kernel,
            cursor: GraphCursor::new(run)?,
            batch_size,
        })
    }
}
impl<K: Kernel> GraphSearchStream<'_, K> {
    pub fn stats(&self) -> SearchStats {
        self.cursor.stats()
    }
    pub fn close(&mut self) {
        self.cursor.close();
    }
}
impl<'g, K: Kernel + Sync> Iterator for GraphSearchStream<'g, K>
where
    K::State: Send + Sync + Clone,
{
    type Item = Result<Vec<GraphPath<'g, K::State>>>;
    fn next(&mut self) -> Option<Self::Item> {
        let adapter = GraphSearchAdapter {
            graph: self.graph,
            kernel: &self.kernel,
            project: |state: &K::State| Ok(state.clone()),
            output: PhantomData,
        };
        self.cursor
            .next_batch(&adapter, self.batch_size)
            .transpose()
    }
}
impl<K: Kernel> Drop for GraphSearchStream<'_, K> {
    fn drop(&mut self) {
        self.close();
    }
}

/// A native store cursor. Returned paths borrow the caller's store, not the cursor.
pub struct NativeSearchStream<
    's,
    K: native::Kernel,
    G: native::GraphStore<Node = K::Node, Edge = K::Edge>,
> {
    store: &'s G,
    kernel: K,
    cursor: Cursor<K::State, ()>,
    batch_size: usize,
}
impl<'s, K: native::Kernel, G: native::GraphStore<Node = K::Node, Edge = K::Edge>>
    NativeSearchStream<'s, K, G>
{
    pub(crate) fn new(store: &'s G, kernel: K, run: RunOptions, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            bail!("batch_size must be at least 1");
        }
        Ok(Self {
            store,
            kernel,
            cursor: Cursor::new(run)?,
            batch_size,
        })
    }
    pub fn stats(&self) -> SearchStats {
        self.cursor.stats
    }
    pub fn close(&mut self) {
        self.cursor.close();
    }
}
impl<'s, K: native::Kernel, G: native::GraphStore<Node = K::Node, Edge = K::Edge>> Iterator
    for NativeSearchStream<'s, K, G>
where
    K::Node: 's,
    K::Edge: 's,
{
    type Item = Result<Vec<native::Path<'s, K::Node, K::Edge, K::State>>>;
    fn next(&mut self) -> Option<Self::Item> {
        let adapter = NativeSearchAdapter {
            store: self.store,
            kernel: &self.kernel,
            materialize: native::materialize_native::<G, K::State>,
        };
        self.cursor
            .next_batch(&adapter, self.batch_size)
            .transpose()
    }
}
impl<K: native::Kernel, G: native::GraphStore<Node = K::Node, Edge = K::Edge>> Drop
    for NativeSearchStream<'_, K, G>
{
    fn drop(&mut self) {
        self.close();
    }
}

pub(crate) fn owned_paths(paths: Vec<GraphPath<'_>>, stats: SearchStats) -> OwnedSearchResult {
    OwnedSearchResult {
        paths: paths
            .into_iter()
            .map(|path| OwnedGraphPath {
                nodes: path.nodes.into_iter().map(|id| id.into_owned()).collect(),
                edges: path.edges.into_iter().map(|id| id.into_owned()).collect(),
                state: path.state,
                intermediate_states: path.intermediate_states,
            })
            .collect(),
        stats,
    }
}

pub(crate) fn graph_stream<K, F>(
    graph: Arc<Graph>,
    kernel: K,
    encode: Arc<F>,
    run: RunOptions,
) -> Result<Box<dyn NativeBatchStream>>
where
    K: Kernel + Send + Sync + 'static,
    K::State: Send + Sync + Clone,
    F: Fn(&K::State) -> Result<StateRow> + Send + Sync + 'static,
{
    struct Stream<K: Kernel, F> {
        graph: Arc<Graph>,
        kernel: K,
        encode: Arc<F>,
        cursor: GraphCursor<K::State, PayloadCache>,
    }
    impl<K, F> NativeBatchStream for Stream<K, F>
    where
        K: Kernel + Send + Sync + 'static,
        K::State: Send + Sync + Clone,
        F: Fn(&K::State) -> Result<StateRow> + Send + Sync + 'static,
    {
        fn next_batch(&mut self, size: usize) -> Result<Option<OwnedSearchResult>> {
            let adapter = GraphSearchAdapter {
                graph: self.graph.as_ref(),
                kernel: &self.kernel,
                project: |state: &K::State| (self.encode)(state),
                output: PhantomData,
            };
            Ok(self
                .cursor
                .next_batch(&adapter, size)?
                .map(|paths| owned_paths(paths, self.cursor.stats())))
        }
        fn stats(&self) -> SearchStats {
            self.cursor.stats()
        }
        fn close(&mut self) {
            self.cursor.close();
        }
    }
    Ok(managed(Stream {
        graph,
        kernel,
        encode,
        cursor: GraphCursor::new(run)?,
    }))
}

// Dropping the concrete session releases graph ownership and lazy caches on close,
// exhaustion, or error, even if a Rust caller retains the boxed stream itself.
pub(crate) fn managed<S: NativeBatchStream + 'static>(stream: S) -> Box<dyn NativeBatchStream> {
    struct Managed<S: NativeBatchStream> {
        inner: Option<S>,
        stats: SearchStats,
    }
    impl<S: NativeBatchStream> NativeBatchStream for Managed<S> {
        fn next_batch(&mut self, size: usize) -> Result<Option<OwnedSearchResult>> {
            let Some(inner) = &mut self.inner else {
                return Ok(None);
            };
            let result = inner.next_batch(size);
            self.stats = inner.stats();
            if result.is_err() || matches!(result, Ok(None)) {
                self.close();
            }
            result
        }
        fn stats(&self) -> SearchStats {
            self.stats
        }
        fn close(&mut self) {
            if let Some(mut inner) = self.inner.take() {
                self.stats = inner.stats();
                inner.close();
            }
        }
    }
    impl<S: NativeBatchStream> Drop for Managed<S> {
        fn drop(&mut self) {
            self.close();
        }
    }
    Box::new(Managed {
        inner: Some(stream),
        stats: SearchStats::default(),
    })
}

/// Defer payload decoding and file setup until the first pull. Closing before
/// that pull drops the initializer and all of its retained graph ownership.
pub(crate) fn deferred<F>(initialize: F) -> Box<dyn NativeBatchStream>
where
    F: FnOnce() -> Result<Box<dyn NativeBatchStream>> + Send + 'static,
{
    struct Deferred<F> {
        initialize: Option<F>,
        inner: Option<Box<dyn NativeBatchStream>>,
    }
    impl<F> NativeBatchStream for Deferred<F>
    where
        F: FnOnce() -> Result<Box<dyn NativeBatchStream>> + Send,
    {
        fn next_batch(&mut self, size: usize) -> Result<Option<OwnedSearchResult>> {
            if size == 0 {
                bail!("batch_size must be at least 1");
            }
            if let Some(initialize) = self.initialize.take() {
                self.inner = Some(initialize()?);
            }
            match &mut self.inner {
                Some(inner) => inner.next_batch(size),
                None => Ok(None),
            }
        }
        fn stats(&self) -> SearchStats {
            self.inner
                .as_ref()
                .map_or_else(SearchStats::default, |s| s.stats())
        }
        fn close(&mut self) {
            self.initialize = None;
            if let Some(inner) = &mut self.inner {
                inner.close();
            }
            self.inner = None;
        }
    }
    managed(Deferred {
        initialize: Some(initialize),
        inner: None,
    })
}
