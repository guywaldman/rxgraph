//! Opt-in search that evaluates the exact kernel when a possible ending is reached.
use std::sync::Arc;

use crate::{
    EdgeId, Graph, GraphId, GraphPath, GraphRepo, Kernel, NodeId, RunOptions, SearchResult,
    SearchStats,
};
use anyhow::{Context, Result, bail};

use super::{
    TargetOptions,
    algo::GraphSearchAdapter,
    engine::{PathEntry, SearchAdapter, cursor::GraphCursor},
    kernel::PayloadCache,
    targets::{Reverse, TargetBackend, TargetBound, build_reverse, intersect},
};

#[derive(Clone)]
struct Frame {
    parent: Option<Arc<Frame>>,
    src: NodeId,
    edge: EdgeId,
    dest: NodeId,
}

impl Drop for Frame {
    fn drop(&mut self) {
        let mut parent = self.parent.take();
        while let Some(frame) = parent {
            match Arc::try_unwrap(frame) {
                Ok(mut frame) => parent = frame.parent.take(),
                Err(shared) => {
                    drop(shared);
                    break;
                }
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct LazyState<S> {
    start: NodeId,
    path: Option<Arc<Frame>>,
    targets: TargetBound,
    pub(crate) final_state: Option<S>,
    pub(crate) history: Option<Vec<S>>,
}

pub(crate) struct LazyAdapter<'k, A> {
    pub(crate) inner: A,
    pub(crate) reverse: Option<&'k Reverse>,
    pub(crate) retain_history: bool,
    pub(crate) release_expanded: bool,
}

impl<A: TargetBackend> SearchAdapter for LazyAdapter<'_, A> {
    type State = LazyState<A::State>;
    type Path = A::Path;
    type Cache = A::Cache;

    fn resolve_node(&self, external: GraphId<'_>) -> Result<Option<NodeId>> {
        self.inner.resolve_node(external)
    }
    fn initial_state(&self, start: NodeId) -> Result<Self::State> {
        let targets = self.inner.possible_start(start, None)?;
        Ok(LazyState {
            start,
            path: None,
            targets,
            final_state: None,
            history: None,
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
        self.release_expanded
    }
    fn release_expanded_state(&self, state: &mut Self::State) {
        state.path = None;
        state.targets = None;
    }
    fn eval_edge(
        &self,
        src: NodeId,
        edge: EdgeId,
        dest: NodeId,
        parent: &Self::State,
        cache: &Self::Cache,
    ) -> Result<Option<(Self::State, bool)>> {
        let targets = parent.targets.clone();
        if targets.as_ref().is_some_and(|t| t.is_empty()) {
            return Ok(None);
        }
        let update = self
            .inner
            .possible_edge((src, edge, dest), (None, None), cache, &targets)?;
        let targets = intersect(&targets, update);
        if targets.as_ref().is_some_and(|t| t.is_empty()) {
            return Ok(None);
        }
        if self
            .reverse
            .is_some_and(|reverse| !reverse.reachable(dest, &targets))
        {
            return Ok(None);
        }
        let frame = Arc::new(Frame {
            parent: parent.path.clone(),
            src,
            edge,
            dest,
        });
        let mut state = LazyState {
            start: parent.start,
            path: Some(frame),
            targets,
            final_state: None,
            history: None,
        };
        if state.targets.as_ref().is_some_and(|t| !t.contains(&dest)) {
            return Ok(Some((state, false)));
        }
        let mut edges = Vec::new();
        let mut cursor = state.path.as_deref();
        while let Some(frame) = cursor {
            edges.push(frame);
            cursor = frame.parent.as_deref();
        }
        edges.reverse();
        let mut exact = self.inner.initial_state(state.start)?;
        let initial = self.inner.possible_start(state.start, None)?;
        let mut replay_targets = intersect(
            &initial,
            self.inner.possible_start(state.start, Some(&exact))?,
        );
        let mut history = self.retain_history.then(|| vec![exact.clone()]);
        for (index, frame) in edges.iter().enumerate() {
            let update = self.inner.possible_edge(
                (frame.src, frame.edge, frame.dest),
                (Some(&exact), None),
                cache,
                &replay_targets,
            )?;
            replay_targets = intersect(&replay_targets, update);
            if replay_targets.as_ref().is_some_and(|t| t.is_empty()) {
                return Ok(None);
            }
            let Some((child, complete)) = self
                .inner
                .eval_edge(frame.src, frame.edge, frame.dest, &exact, cache)?
            else {
                return Ok(None);
            };
            if self.inner.target_needs_post() {
                let update = self.inner.possible_edge(
                    (frame.src, frame.edge, frame.dest),
                    (Some(&exact), Some(&child)),
                    cache,
                    &replay_targets,
                )?;
                replay_targets = intersect(&replay_targets, update);
            }
            if complete
                && replay_targets
                    .as_ref()
                    .is_some_and(|targets| !targets.contains(&frame.dest))
            {
                bail!("possible_targets excluded a completed path ending");
            }
            if replay_targets.as_ref().is_some_and(|t| t.is_empty()) {
                return Ok(None);
            }
            if let Some(history) = &mut history {
                history.push(child.clone());
            }
            exact = child;
            if complete {
                if index + 1 != edges.len() {
                    bail!("possible_targets omitted an earlier completed path");
                }
                state.final_state = Some(exact);
                state.history = history;
                return Ok(Some((state, true)));
            }
        }
        // The candidate may itself be an intermediate node. Preserve the exact
        // refinement while leaving its descendants eligible for traversal.
        if let Some(replay) = replay_targets {
            state.targets = intersect(&state.targets, Some(replay));
        }
        if state.targets.as_ref().is_some_and(|t| t.is_empty()) {
            return Ok(None);
        }
        Ok(Some((state, false)))
    }
    fn materialize(
        &self,
        arena: &[PathEntry<Self::State>],
        path: usize,
        history: bool,
    ) -> Result<Self::Path> {
        self.inner.materialize_lazy(arena, path, history)
    }
}

pub(crate) fn materialize_graph_lazy<'g, S: Clone>(
    graph: &'g Graph,
    arena: &[PathEntry<LazyState<S>>],
    mut path: usize,
    history: bool,
) -> Result<GraphPath<'g, S>> {
    let state = arena[path]
        .state
        .final_state
        .as_ref()
        .expect("completed path has exact state")
        .clone();
    let intermediate_states = if history {
        Some(
            arena[path]
                .state
                .history
                .as_ref()
                .expect("history was retained")
                .clone(),
        )
    } else {
        None
    };
    let mut nodes = Vec::with_capacity(arena[path].depth + 1);
    let mut edges = Vec::with_capacity(arena[path].depth);
    loop {
        nodes.push(
            graph
                .repo
                .external_node(arena[path].node)
                .context("path references missing node")?,
        );
        if let Some(edge) = arena[path].incoming_edge {
            edges.push(
                graph
                    .repo
                    .external_edge(edge)
                    .context("path references missing edge")?,
            );
        }
        match arena[path].parent {
            Some(parent) => path = parent,
            None => break,
        }
    }
    nodes.reverse();
    edges.reverse();
    Ok(GraphPath {
        nodes,
        edges,
        state,
        intermediate_states,
    })
}

/// A pull stream for opt-in deferred native kernel evaluation.
pub(crate) struct LazyGraphSearchStream<'g, K: Kernel> {
    graph: &'g Graph,
    kernel: Option<K>,
    cursor: GraphCursor<LazyState<K::State>, PayloadCache>,
    reverse_run: Option<RunOptions>,
    options: TargetOptions,
    reverse: Option<Reverse>,
    initialized: bool,
    empty: bool,
    retain_history: bool,
    batch_size: usize,
}

impl Graph {
    /// Starts a deferred search. The kernel and target hook must be deterministic;
    /// callbacks on paths eliminated by target bounds may never be evaluated.
    pub(crate) fn search_batches_lazy_with<K: Kernel + Sync>(
        &self,
        kernel: K,
        run: RunOptions,
        batch_size: usize,
        options: TargetOptions,
    ) -> Result<LazyGraphSearchStream<'_, K>>
    where
        K::State: Send + Sync,
    {
        if batch_size == 0 {
            bail!("batch_size must be at least 1");
        }
        let empty = run.max_paths == Some(0);
        let retain_history = run.intermediate_states;
        let reverse_run = (options.reverse && !empty).then(|| run.clone());
        Ok(LazyGraphSearchStream {
            graph: self,
            kernel: Some(kernel),
            cursor: GraphCursor::new(run)?,
            reverse_run,
            options,
            reverse: None,
            initialized: false,
            empty,
            retain_history,
            batch_size,
        })
    }
    /// Materializes all paths from a deferred search.
    pub(crate) fn search_paths_lazy_with<K: Kernel + Sync>(
        &self,
        kernel: K,
        run: RunOptions,
        options: TargetOptions,
    ) -> Result<SearchResult<'_, K::State>>
    where
        K::State: Send + Sync,
    {
        let mut stream = self.search_batches_lazy_with(kernel, run, 1024, options)?;
        let mut paths = Vec::new();
        for batch in stream.by_ref() {
            paths.extend(batch?);
        }
        Ok(SearchResult {
            paths,
            stats: stream.stats(),
        })
    }
}
impl<K: Kernel> LazyGraphSearchStream<'_, K> {
    pub fn stats(&self) -> SearchStats {
        self.cursor.stats()
    }
    pub fn close(&mut self) {
        self.cursor.close();
        self.reverse = None;
        self.reverse_run = None;
        self.kernel = None;
    }
    fn initialize_reverse(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }
        self.initialized = true;
        if self.empty || !self.options.reverse {
            return Ok(());
        }
        if let Some(run) = self.reverse_run.take() {
            let kernel = self.kernel.as_ref().expect("open stream has kernel");
            self.reverse = build_reverse(self.graph, kernel, &run, true)?;
        }
        Ok(())
    }
}
impl<'g, K: Kernel + Sync> Iterator for LazyGraphSearchStream<'g, K>
where
    K::State: Send + Sync,
{
    type Item = Result<Vec<GraphPath<'g, K::State>>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.kernel.as_ref()?;
        if let Err(error) = self.initialize_reverse() {
            self.close();
            return Some(Err(error));
        }
        let adapter = LazyAdapter {
            inner: GraphSearchAdapter {
                graph: self.graph,
                kernel: self.kernel.as_ref()?,
                project: |state: &K::State| Ok(state.clone()),
                output: std::marker::PhantomData,
            },
            reverse: self.reverse.as_ref(),
            retain_history: self.retain_history,
            release_expanded: self.options.release_expanded,
        };
        let result = self.cursor.next_batch(&adapter, self.batch_size);
        if result.is_err() || matches!(result, Ok(None)) {
            self.close();
        }
        result.transpose()
    }
}
impl<K: Kernel> Drop for LazyGraphSearchStream<'_, K> {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::{Arc, Frame};

    #[test]
    fn drops_deep_prefix_without_recursive_stack_growth() {
        let mut path = None;
        for i in 0..100_000 {
            path = Some(Arc::new(Frame {
                parent: path,
                src: i,
                edge: i,
                dest: i + 1,
            }));
        }
        drop(path);
    }
}
