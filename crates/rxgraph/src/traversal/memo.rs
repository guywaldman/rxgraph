//! Opt-in, per-search transition caching for deterministic Rust kernels.

use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    hash::Hash,
    marker::PhantomData,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use anyhow::{Result, anyhow, bail};

use crate::{
    EdgeId, Graph, GraphId, GraphPath, Kernel, NodeId, RunOptions, SearchResult, SearchStats,
};

use super::{
    algo::{GraphSearchAdapter, materialize_path},
    engine::{self, PathEntry, SearchAdapter, cursor::GraphCursor},
    kernel::PayloadCache,
};

/// A kernel whose transitions can be reused within a search.
///
/// For a fixed graph, kernel and edge, equal keys must imply identical
/// transition behavior, including rejection, errors, completion, and the entire
/// child state. Transitions must not rely on side effects or other mutable
/// context. Keys must remain stable while their states are used. Hash collisions
/// are checked using equality.
///
/// A compact key can avoid hashing a large state, but it must satisfy this
/// contract; a fingerprint alone is not a safe key if it can collide.
pub trait MemoizableKernel: Kernel {
    type Key: Clone + Eq + Hash + Send;

    fn state_key(&self, state: &Self::State) -> Self::Key;

    /// Wrap this kernel in a bounded cache usable by the other native searches.
    /// Construct a fresh wrapper for each search.
    fn memoized(self, capacity: usize) -> MemoizedKernel<Self>
    where
        Self: Sized,
    {
        MemoizedKernel::new(self, capacity)
    }
}

/// A bounded cache around a deterministic kernel. The cache is shared across
/// workers, so a cache hit takes a lock. It is scoped to this wrapper's lifetime.
pub struct MemoizedKernel<K: MemoizableKernel> {
    kernel: K,
    capacity: usize,
    entries: Mutex<Entries<K::Key, K::State>>,
}

impl<K: MemoizableKernel> MemoizedKernel<K> {
    fn new(kernel: K, capacity: usize) -> Self {
        Self {
            kernel,
            capacity,
            entries: Mutex::new(Entries {
                values: HashMap::new(),
                order: VecDeque::new(),
            }),
        }
    }
}

impl<K: MemoizableKernel> Kernel for MemoizedKernel<K> {
    type State = K::State;

    fn initial_state(&self, graph: &Graph, start: NodeId) -> Result<Self::State> {
        self.kernel.initial_state(graph, start)
    }

    fn transition(
        &self,
        cx: &crate::EdgeCtx<'_, Self::State>,
    ) -> Result<crate::Transition<Self::State>> {
        if self.capacity == 0 {
            return self.kernel.transition(cx);
        }
        let key = (cx.edge(), self.kernel.state_key(cx.state()));
        {
            let entries = self
                .entries
                .lock()
                .map_err(|_| anyhow!("memo cache poisoned"))?;
            if let Some(value) = entries.values.get(&key) {
                return Ok(match value.clone() {
                    None => crate::Transition::Reject,
                    Some((state, false)) => crate::Transition::Continue(state),
                    Some((state, true)) => crate::Transition::Complete(state),
                });
            }
        }
        let result = self.kernel.transition(cx)?;
        let value = match &result {
            crate::Transition::Reject => None,
            crate::Transition::Continue(state) => Some((state.clone(), false)),
            crate::Transition::Complete(state) => Some((state.clone(), true)),
        };
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| anyhow!("memo cache poisoned"))?;
        // Another worker may have computed this key while the lock was released.
        if !entries.values.contains_key(&key) {
            if entries.order.len() == self.capacity {
                let oldest = entries.order.pop_front().expect("full cache has entries");
                entries.values.remove(&oldest);
            }
            entries.order.push_back(key.clone());
            entries.values.insert(key, value);
        }
        Ok(result)
    }

    fn possible_targets(
        &self,
        cx: &super::targets::TargetCtx<'_, Self::State>,
    ) -> Result<Option<Vec<crate::OwnedGraphId>>> {
        self.kernel.possible_targets(cx)
    }
}

/// Per-search cache capacity and expanded BFS state retention.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoOptions {
    /// Maximum cached entries shared among all workers. Zero disables caching.
    pub capacity: usize,
    /// Release expanded BFS path states when intermediate history is not requested.
    pub release_expanded: bool,
}

// Preserve the original prototype's capacity-only behavior.
impl From<usize> for MemoOptions {
    fn from(capacity: usize) -> Self {
        Self {
            capacity,
            release_expanded: true,
        }
    }
}

// A search-wide budget shared by the otherwise worker-local caches. A slot is
// reserved once; subsequent replacements are local and need no atomic update.
struct Budget {
    capacity: usize,
    reserved: AtomicUsize,
}

impl Budget {
    fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            reserved: AtomicUsize::new(0),
        })
    }

    fn reserve(&self) -> bool {
        let mut used = self.reserved.load(Ordering::Relaxed);
        loop {
            if used >= self.capacity {
                return false;
            }
            match self.reserved.compare_exchange_weak(
                used,
                used + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => used = actual,
            }
        }
    }
}

struct Entries<K, S> {
    values: HashMap<(EdgeId, K), Option<(S, bool)>>,
    order: VecDeque<(EdgeId, K)>,
}

pub(crate) struct MemoCache<C, K, S> {
    inner: C,
    entries: RefCell<Entries<K, S>>,
    budget: Arc<Budget>,
}

struct MemoAdapter<'k, A, K> {
    inner: A,
    kernel: &'k K,
    budget: Arc<Budget>,
    release_expanded: bool,
}

impl<'g, 'k, K, P> SearchAdapter for MemoAdapter<'k, GraphSearchAdapter<'g, 'k, K, P, K::State>, K>
where
    K: MemoizableKernel,
    P: Fn(&K::State) -> Result<K::State>,
{
    type State = Option<K::State>;
    type Path = GraphPath<'g, K::State>;
    type Cache = MemoCache<PayloadCache, K::Key, K::State>;

    fn resolve_node(&self, external: GraphId<'_>) -> Result<Option<NodeId>> {
        self.inner.resolve_node(external)
    }
    fn initial_state(&self, node: NodeId) -> Result<Self::State> {
        self.inner.initial_state(node).map(Some)
    }
    fn dense_node_count(&self) -> Option<usize> {
        self.inner.dense_node_count()
    }
    fn prefetch_outgoing(&self, nodes: &[NodeId]) -> Result<()> {
        self.inner.prefetch_outgoing(nodes)
    }
    fn out_degree(&self, node: NodeId) -> Result<usize> {
        self.inner.out_degree(node)
    }
    fn for_each_range<F>(&self, node: NodeId, start: usize, end: usize, visit: F) -> Result<()>
    where
        F: FnMut(EdgeId, NodeId) -> Result<bool>,
    {
        self.inner.for_each_range(node, start, end, visit)
    }
    fn make_cache(&self) -> Self::Cache {
        MemoCache {
            inner: self.inner.make_cache(),
            entries: RefCell::new(Entries {
                values: HashMap::new(),
                order: VecDeque::new(),
            }),
            budget: Arc::clone(&self.budget),
        }
    }
    fn can_release_expanded_states(&self) -> bool {
        self.release_expanded
    }
    fn release_expanded_state(&self, state: &mut Self::State) {
        *state = None;
    }
    fn eval_edge(
        &self,
        src: NodeId,
        edge: EdgeId,
        dest: NodeId,
        state: &Self::State,
        cache: &Self::Cache,
    ) -> Result<Option<(Self::State, bool)>> {
        let state = state.as_ref().expect("only unexpanded paths are evaluated");
        if cache.budget.capacity == 0 {
            return self
                .inner
                .eval_edge(src, edge, dest, state, &cache.inner)
                .map(|result| result.map(|(state, stop)| (Some(state), stop)));
        }
        let key = (edge, self.kernel.state_key(state));
        if let Some(result) = cache.entries.borrow().values.get(&key) {
            return Ok(result.clone().map(|(state, stop)| (Some(state), stop)));
        }
        let result = self.inner.eval_edge(src, edge, dest, state, &cache.inner)?;
        let mut entries = cache.entries.borrow_mut();
        if !cache.budget.reserve() {
            if let Some(old) = entries.order.pop_front() {
                entries.values.remove(&old);
            } else {
                // Other workers hold the entire budget.
                return Ok(result.map(|(state, stop)| (Some(state), stop)));
            }
        }
        entries.order.push_back(key.clone());
        entries.values.insert(key, result.clone());
        Ok(result.map(|(state, stop)| (Some(state), stop)))
    }
    fn materialize(
        &self,
        arena: &[PathEntry<Self::State>],
        path: usize,
        history: bool,
    ) -> Result<Self::Path> {
        materialize_path(self.inner.graph, arena, path, history, |state| {
            (self.inner.project)(state.as_ref().expect("requested path state is retained"))
        })
    }
}

impl Graph {
    /// Enumerates paths while reusing transitions of a deterministic kernel.
    ///
    /// The capacity bounds cached entries across the entire search, not bytes.
    /// Entries are worker-local and evicted in insertion order when full; a
    /// capacity of zero disables caching. Errors are not cached. `MemoOptions`
    /// controls BFS state release independently; a bare capacity retains the
    /// prototype's original behavior of releasing expanded states.
    pub fn search_paths_memoized_with<K: MemoizableKernel + Sync>(
        &self,
        kernel: K,
        run: RunOptions,
        options: impl Into<MemoOptions>,
    ) -> Result<SearchResult<'_, K::State>>
    where
        K::State: Send + Sync,
    {
        let options = options.into();
        let adapter = MemoAdapter {
            inner: GraphSearchAdapter {
                graph: self,
                kernel: &kernel,
                project: |state: &K::State| Ok(state.clone()),
                output: PhantomData,
            },
            kernel: &kernel,
            budget: Budget::new(options.capacity),
            release_expanded: options.release_expanded,
        };
        let result = engine::search(&adapter, run)?;
        Ok(SearchResult {
            paths: result.paths,
            stats: result.stats,
        })
    }

    /// Starts a pull stream with a bounded transition cache scoped to the stream.
    /// See [`Graph::search_paths_memoized_with`] for cache and state-release behavior.
    pub fn search_batches_memoized_with<K: MemoizableKernel + Sync>(
        &self,
        kernel: K,
        run: RunOptions,
        batch_size: usize,
        options: impl Into<MemoOptions>,
    ) -> Result<MemoizedGraphSearchStream<'_, K>>
    where
        K::State: Send + Sync,
    {
        if batch_size == 0 {
            bail!("batch_size must be at least 1");
        }
        let options = options.into();
        Ok(MemoizedGraphSearchStream {
            graph: self,
            kernel,
            cursor: GraphCursor::new(run)?,
            batch_size,
            budget: Budget::new(options.capacity),
            release_expanded: options.release_expanded,
        })
    }
}

type MemoCursor<K> = GraphCursor<
    Option<<K as Kernel>::State>,
    MemoCache<PayloadCache, <K as MemoizableKernel>::Key, <K as Kernel>::State>,
>;

/// A native graph stream with worker-local, search-scoped transition caches.
pub struct MemoizedGraphSearchStream<'g, K: MemoizableKernel> {
    graph: &'g Graph,
    kernel: K,
    cursor: MemoCursor<K>,
    batch_size: usize,
    budget: Arc<Budget>,
    release_expanded: bool,
}

impl<K: MemoizableKernel> MemoizedGraphSearchStream<'_, K> {
    pub fn stats(&self) -> SearchStats {
        self.cursor.stats()
    }
    pub fn close(&mut self) {
        self.cursor.close();
    }
}

impl<'g, K: MemoizableKernel + Sync> Iterator for MemoizedGraphSearchStream<'g, K>
where
    K::State: Send + Sync,
{
    type Item = Result<Vec<GraphPath<'g, K::State>>>;
    fn next(&mut self) -> Option<Self::Item> {
        let adapter = MemoAdapter {
            inner: GraphSearchAdapter {
                graph: self.graph,
                kernel: &self.kernel,
                project: |state: &K::State| Ok(state.clone()),
                output: PhantomData,
            },
            kernel: &self.kernel,
            budget: Arc::clone(&self.budget),
            release_expanded: self.release_expanded,
        };
        self.cursor
            .next_batch(&adapter, self.batch_size)
            .transpose()
    }
}

impl<K: MemoizableKernel> Drop for MemoizedGraphSearchStream<'_, K> {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::Budget;
    use rayon::prelude::*;

    #[test]
    fn shared_budget_is_bounded_under_contention() {
        let budget = Budget::new(73);
        let reserved = (0..10_000)
            .into_par_iter()
            .filter(|_| budget.reserve())
            .count();
        assert_eq!(reserved, 73);
        assert!(!budget.reserve());
        assert!(!Budget::new(0).reserve());
    }
}
