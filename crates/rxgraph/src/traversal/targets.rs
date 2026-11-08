//! Conservative per-path bounds on possible ending nodes.
use std::{borrow::Cow, marker::PhantomData};

use super::{
    algo::{GraphSearchAdapter, materialize_path},
    engine::{self, PathEntry, SearchAdapter, cursor::GraphCursor},
    kernel::PayloadCache,
};
use crate::{
    EdgeCtx, EdgeId, Graph, GraphId, GraphPath, GraphRepo, Kernel, NodeId, OwnedGraphId,
    RunOptions, SearchResult, SearchStats,
};
use anyhow::{Context, Result, bail};
use smallvec::SmallVec;

/// Context for an optional conservative bound on possible completion nodes.
/// Edge contexts use the original graph direction. A pre-transition call has
/// no child state; a post-transition call has both parent and child state.
/// Parent state is also absent when exact evaluation is deferred. A stateless
/// start bound can be used to prepare a reverse reachability index.
pub enum TargetCtx<'a, S> {
    /// A path beginning at `start`.
    Start {
        /// Graph used for this search.
        graph: &'a Graph,
        /// Internal starting node ID.
        start: NodeId,
        /// Initial state, if it has been computed.
        state: Option<&'a S>,
    },
    /// A candidate edge, before or after evaluating its transition.
    Edge {
        /// Topology and payloads for the edge.
        edge: EdgeCtx<'a, ()>,
        /// Parent state, if it has been computed.
        parent: Option<&'a S>,
        /// Child state, after an accepted transition.
        child: Option<&'a S>,
    },
}

/// Optional search optimizations for target-aware Rust kernels.
#[derive(Debug, Clone, Copy, Default)]
pub struct TargetOptions {
    /// Whether exact transitions run incrementally or at candidate endings.
    pub evaluation: Evaluation,
    /// Precompute reverse reachability from initial targets when available.
    pub reverse: bool,
    /// Drop expanded BFS states when intermediate states are not requested.
    pub release_expanded: bool,
}

/// When to evaluate exact kernel transitions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Evaluation {
    /// Evaluate the kernel on every accepted candidate edge.
    #[default]
    Incremental,
    /// Replay exact transitions when a possible ending is encountered.
    /// The kernel and bound hook must be deterministic and allow skipped calls.
    /// `SearchStats::evaluated_edges` counts candidate topology evaluations; it
    /// does not count exact transition invocations performed during replay.
    /// Shared prefixes can be evaluated repeatedly without memoization.
    Lazy,
}

/// `None` is unknown; `Some` is a sorted set of internal ending nodes.
pub(crate) type TargetBound = Option<SmallVec<[NodeId; 8]>>;

fn resolve(graph: &Graph, ids: Option<Vec<OwnedGraphId>>) -> Result<TargetBound> {
    ids.map(|ids| {
        let mut nodes = ids
            .into_iter()
            .map(|id| {
                graph
                    .repo
                    .internal_node(id.as_ref())
                    .with_context(|| format!("unknown target node {id}"))
            })
            .collect::<Result<SmallVec<[NodeId; 8]>>>()?;
        nodes.sort_unstable();
        nodes.dedup();
        Ok(nodes)
    })
    .transpose()
}

pub(crate) fn intersect(old: &TargetBound, new: TargetBound) -> TargetBound {
    match (old, new) {
        (None, new) => new,
        (Some(old), None) => Some(old.clone()),
        (Some(old), Some(new)) => Some(
            old.iter()
                .copied()
                .filter(|n| new.binary_search(n).is_ok())
                .collect(),
        ),
    }
}

pub(crate) fn initial_bound<K: Kernel>(
    graph: &Graph,
    kernel: &K,
    start: NodeId,
    state: Option<&K::State>,
) -> Result<TargetBound> {
    resolve(
        graph,
        kernel.possible_targets(&TargetCtx::Start {
            graph,
            start,
            state,
        })?,
    )
}

pub(crate) fn advance_bound<K: Kernel>(
    graph: &Graph,
    kernel: &K,
    candidate: (NodeId, EdgeId, NodeId),
    states: (Option<&K::State>, Option<&K::State>),
    old: &TargetBound,
    cache: &PayloadCache,
) -> Result<TargetBound> {
    let next = edge_update(graph, kernel, candidate, states, cache)?;
    Ok(intersect(old, next))
}

fn edge_update<K: Kernel>(
    graph: &Graph,
    kernel: &K,
    candidate: (NodeId, EdgeId, NodeId),
    states: (Option<&K::State>, Option<&K::State>),
    cache: &PayloadCache,
) -> Result<TargetBound> {
    let (src, edge, dest) = candidate;
    let (parent, child) = states;
    let unit = ();
    let edge = EdgeCtx::new(graph, src, dest, edge, &unit, cache);
    resolve(
        graph,
        kernel.possible_targets(&TargetCtx::Edge {
            edge,
            parent,
            child,
        })?,
    )
}

pub(crate) enum Reverse {
    PerTarget { ids: Vec<NodeId>, masks: Vec<u64> },
    Coarse(Vec<bool>),
}

impl Reverse {
    pub(crate) fn reachable(&self, node: NodeId, targets: &TargetBound) -> bool {
        match (self, targets) {
            (_, Some(ids)) if ids.is_empty() => false,
            (Self::Coarse(keep), _) => keep[node as usize],
            (Self::PerTarget { ids, masks }, bound) => {
                let Some(bound) = bound else {
                    return masks[node as usize] != 0;
                };
                let mut active = 0u64;
                for target in bound {
                    if let Ok(index) = ids.binary_search(target) {
                        active |= 1 << index;
                    }
                }
                masks[node as usize] & active != 0
            }
        }
    }
}

/// Build from stateless start bounds only. If any start is unknown, no global
/// reverse bound is safe. Fallback is deliberately conservative for many targets.
pub(crate) fn build_reverse<K: Kernel>(
    graph: &Graph,
    kernel: &K,
    run: &RunOptions,
    enabled: bool,
) -> Result<Option<Reverse>> {
    if !enabled || run.start_nodes.is_empty() || run.max_paths == Some(0) {
        return Ok(None);
    }
    let mut targets = Vec::new();
    for external in &run.start_nodes {
        let start = graph
            .repo
            .internal_node(external.as_ref())
            .with_context(|| format!("unknown start node {external}"))?;
        let Some(ids) = initial_bound(graph, kernel, start, None)? else {
            return Ok(None);
        };
        targets.extend(ids);
    }
    targets.sort_unstable();
    targets.dedup();
    // Bound per-target preparation work and mask allocation; the coarse mask
    // remains a safe overapproximation when the target set is larger.
    if targets.len() <= 8 && graph.node_count().saturating_mul(8) <= 128 * 1024 * 1024 {
        let mut masks = vec![0u64; graph.node_count()];
        for (index, &target) in targets.iter().enumerate() {
            let bit = 1u64 << index;
            let mut queue = vec![target];
            masks[target as usize] |= bit;
            let mut head = 0;
            while let Some(&node) = queue.get(head) {
                head += 1;
                for source in graph.repo.incoming(node) {
                    if masks[source as usize] & bit == 0 {
                        masks[source as usize] |= bit;
                        queue.push(source);
                    }
                }
            }
        }
        Ok(Some(Reverse::PerTarget {
            ids: targets,
            masks,
        }))
    } else {
        let mut keep = vec![false; graph.node_count()];
        let mut queue = Vec::new();
        for target in targets {
            if !keep[target as usize] {
                keep[target as usize] = true;
                queue.push(target);
            }
        }
        let mut head = 0;
        while let Some(&node) = queue.get(head) {
            head += 1;
            for source in graph.repo.incoming(node) {
                if !keep[source as usize] {
                    keep[source as usize] = true;
                    queue.push(source);
                }
            }
        }
        Ok(Some(Reverse::Coarse(keep)))
    }
}

#[derive(Clone)]
pub(crate) struct TargetState<S> {
    pub(crate) state: Option<S>,
    pub(crate) targets: TargetBound,
}

pub(crate) struct TargetAdapter<'g, 'k, K: Kernel, P> {
    pub(crate) inner: GraphSearchAdapter<'g, 'k, K, P, <K as Kernel>::State>,
    pub(crate) reverse: Option<&'k Reverse>,
    pub(crate) release: bool,
}

impl<'g, 'k, K: Kernel, P: Fn(&K::State) -> Result<K::State>> SearchAdapter
    for TargetAdapter<'g, 'k, K, P>
{
    type State = TargetState<K::State>;
    type Path = GraphPath<'g, K::State>;
    type Cache = PayloadCache;
    fn resolve_node(&self, id: GraphId<'_>) -> Result<Option<NodeId>> {
        self.inner.resolve_node(id)
    }
    fn initial_state(&self, node: NodeId) -> Result<Self::State> {
        let initial = initial_bound(self.inner.graph, self.inner.kernel, node, None)?;
        if initial.as_ref().is_some_and(|nodes| nodes.is_empty()) {
            return Ok(TargetState {
                state: None,
                targets: initial,
            });
        }
        let state = self.inner.initial_state(node)?;
        let next = initial_bound(self.inner.graph, self.inner.kernel, node, Some(&state))?;
        Ok(TargetState {
            state: Some(state),
            targets: intersect(&initial, next),
        })
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
        self.inner.make_cache()
    }
    fn can_release_expanded_states(&self) -> bool {
        self.release
    }
    fn release_expanded_state(&self, state: &mut Self::State) {
        state.state = None;
        state.targets = None;
    }
    fn eval_edge(
        &self,
        src: NodeId,
        edge: EdgeId,
        dest: NodeId,
        state: &Self::State,
        cache: &Self::Cache,
    ) -> Result<Option<(Self::State, bool)>> {
        if state.targets.as_ref().is_some_and(|nodes| nodes.is_empty()) {
            return Ok(None);
        }
        let parent = state.state.as_ref().expect("unexpanded path state");
        let update = edge_update(
            self.inner.graph,
            self.inner.kernel,
            (src, edge, dest),
            (Some(parent), None),
            cache,
        )?;
        let bound = match update {
            None => Cow::Borrowed(&state.targets),
            Some(update) => Cow::Owned(intersect(&state.targets, Some(update))),
        };
        if bound
            .as_ref()
            .as_ref()
            .is_some_and(|nodes| nodes.is_empty())
            || self
                .reverse
                .is_some_and(|r| !r.reachable(dest, bound.as_ref()))
        {
            return Ok(None);
        }
        let Some((child, stop)) = self.inner.eval_edge(src, edge, dest, parent, cache)? else {
            return Ok(None);
        };
        let update = edge_update(
            self.inner.graph,
            self.inner.kernel,
            (src, edge, dest),
            (Some(parent), Some(&child)),
            cache,
        )?;
        let bound = match update {
            None => bound.into_owned(),
            Some(update) => intersect(bound.as_ref(), Some(update)),
        };
        if stop
            && bound
                .as_ref()
                .is_some_and(|ids| ids.binary_search(&dest).is_err())
        {
            bail!("possible_targets excluded a completed path ending");
        }
        if bound.as_ref().is_some_and(|nodes| nodes.is_empty())
            || self.reverse.is_some_and(|r| !r.reachable(dest, &bound))
        {
            return Ok(None);
        }
        Ok(Some((
            TargetState {
                state: Some(child),
                targets: bound,
            },
            stop,
        )))
    }
    fn materialize(
        &self,
        arena: &[PathEntry<Self::State>],
        path: usize,
        history: bool,
    ) -> Result<Self::Path> {
        materialize_path(self.inner.graph, arena, path, history, |entry| {
            (self.inner.project)(entry.state.as_ref().expect("requested state retained"))
        })
    }
}

impl Graph {
    /// Starts a native stream using the selected evaluation and pruning options.
    pub(crate) fn search_batches_optimized_with<K: Kernel + Sync>(
        &self,
        kernel: K,
        run: RunOptions,
        batch_size: usize,
        options: TargetOptions,
    ) -> Result<OptimizedSearchStream<'_, K>>
    where
        K::State: Send + Sync,
    {
        let inner = match options.evaluation {
            Evaluation::Incremental => StreamKind::Incremental(
                self.search_batches_targets_with(kernel, run, batch_size, options)?,
            ),
            Evaluation::Lazy => {
                StreamKind::Lazy(self.search_batches_lazy_with(kernel, run, batch_size, options)?)
            }
        };
        Ok(OptimizedSearchStream { inner })
    }

    /// Materializes the paths produced by an optimized native search.
    pub(crate) fn search_paths_optimized_with<K: Kernel + Sync>(
        &self,
        kernel: K,
        run: RunOptions,
        options: TargetOptions,
    ) -> Result<SearchResult<'_, K::State>>
    where
        K::State: Send + Sync,
    {
        match options.evaluation {
            Evaluation::Incremental => self.search_paths_targets_with(kernel, run, options),
            Evaluation::Lazy => self.search_paths_lazy_with(kernel, run, options),
        }
    }

    /// Enumerates native paths using optional per-prefix completion bounds.
    pub(crate) fn search_paths_targets_with<K: Kernel + Sync>(
        &self,
        kernel: K,
        run: RunOptions,
        options: TargetOptions,
    ) -> Result<SearchResult<'_, K::State>>
    where
        K::State: Send + Sync,
    {
        let reverse = build_reverse(self, &kernel, &run, options.reverse)?;
        let adapter = TargetAdapter {
            inner: GraphSearchAdapter {
                graph: self,
                kernel: &kernel,
                project: |s: &K::State| Ok(s.clone()),
                output: PhantomData,
            },
            reverse: reverse.as_ref(),
            release: options.release_expanded,
        };
        let result = engine::search(&adapter, run)?;
        Ok(SearchResult {
            paths: result.paths,
            stats: result.stats,
        })
    }
    /// Starts a pull stream with per-prefix completion bounds.
    pub(crate) fn search_batches_targets_with<K: Kernel + Sync>(
        &self,
        kernel: K,
        run: RunOptions,
        batch_size: usize,
        options: TargetOptions,
    ) -> Result<TargetSearchStream<'_, K>>
    where
        K::State: Send + Sync,
    {
        if batch_size == 0 {
            bail!("batch_size must be at least 1");
        }
        let preparation = options.reverse.then(|| run.clone());
        Ok(TargetSearchStream {
            graph: self,
            kernel: Some(kernel),
            cursor: GraphCursor::new(run)?,
            batch_size,
            reverse: None,
            preparation,
            release: options.release_expanded,
        })
    }
}

/// Pull stream for a native target-aware kernel.
pub(crate) struct TargetSearchStream<'g, K: Kernel> {
    graph: &'g Graph,
    kernel: Option<K>,
    cursor: GraphCursor<TargetState<K::State>, PayloadCache>,
    batch_size: usize,
    reverse: Option<Reverse>,
    preparation: Option<RunOptions>,
    release: bool,
}
impl<K: Kernel> TargetSearchStream<'_, K> {
    pub fn stats(&self) -> SearchStats {
        self.cursor.stats()
    }
    pub fn close(&mut self) {
        self.cursor.close();
        self.reverse = None;
        self.preparation = None;
        self.kernel = None;
    }
}
impl<'g, K: Kernel + Sync> Iterator for TargetSearchStream<'g, K>
where
    K::State: Send + Sync,
{
    type Item = Result<Vec<GraphPath<'g, K::State>>>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(run) = self.preparation.take() {
            let kernel = self.kernel.as_ref()?;
            match build_reverse(self.graph, kernel, &run, true) {
                Ok(reverse) => self.reverse = reverse,
                Err(error) => {
                    self.close();
                    return Some(Err(error));
                }
            }
        }
        let adapter = TargetAdapter {
            inner: GraphSearchAdapter {
                graph: self.graph,
                kernel: self.kernel.as_ref()?,
                project: |s: &K::State| Ok(s.clone()),
                output: PhantomData,
            },
            reverse: self.reverse.as_ref(),
            release: self.release,
        };
        let result = self.cursor.next_batch(&adapter, self.batch_size);
        if result.is_err() || matches!(result, Ok(None)) {
            self.close();
        }
        result.transpose()
    }
}
impl<K: Kernel> Drop for TargetSearchStream<'_, K> {
    fn drop(&mut self) {
        self.close();
    }
}

/// Pull stream for either incremental or lazy native evaluation.
pub(crate) struct OptimizedSearchStream<'g, K: Kernel> {
    inner: StreamKind<'g, K>,
}
enum StreamKind<'g, K: Kernel> {
    Incremental(TargetSearchStream<'g, K>),
    Lazy(super::lazy::LazyGraphSearchStream<'g, K>),
}
impl<K: Kernel> OptimizedSearchStream<'_, K> {
    pub fn stats(&self) -> SearchStats {
        match &self.inner {
            StreamKind::Incremental(s) => s.stats(),
            StreamKind::Lazy(s) => s.stats(),
        }
    }
    pub fn close(&mut self) {
        match &mut self.inner {
            StreamKind::Incremental(s) => s.close(),
            StreamKind::Lazy(s) => s.close(),
        }
    }
}
impl<'g, K: Kernel + Sync> Iterator for OptimizedSearchStream<'g, K>
where
    K::State: Send + Sync,
{
    type Item = Result<Vec<GraphPath<'g, K::State>>>;
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            StreamKind::Incremental(s) => s.next(),
            StreamKind::Lazy(s) => s.next(),
        }
    }
}
