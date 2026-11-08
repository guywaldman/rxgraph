use std::{
    hash::{Hash, Hasher},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use anyhow::{Result, bail};
use arrow::array::record_batch;
use rxgraph::{
    EdgeCtx, Graph, GraphId, GraphPath, Kernel, MemoizableKernel, NodeId, RunOptions, Transition,
    TraversalStrategy,
};

#[derive(Clone, Eq, PartialEq)]
struct CollidingKey(u64);
impl Hash for CollidingKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        0_u8.hash(state);
    }
}

#[derive(Clone)]
struct TestKernel {
    calls: Arc<AtomicUsize>,
    divergent: bool,
}
impl Kernel for TestKernel {
    type State = u64;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<u64> {
        Ok(0)
    }
    fn transition(&self, cx: &EdgeCtx<'_, u64>) -> Result<Transition<u64>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let state = cx.state() + 1 + u64::from(self.divergent && cx.edge() == 1);
        Ok(match cx.dest() {
            5 => Transition::Reject,
            4 => Transition::Complete(state),
            _ => Transition::Continue(state),
        })
    }
}
impl MemoizableKernel for TestKernel {
    type Key = CollidingKey;
    fn state_key(&self, state: &u64) -> Self::Key {
        CollidingKey(*state)
    }
}

fn graph() -> Result<Graph> {
    Graph::new(
        record_batch!(("id", UInt64, [0, 1, 2, 3, 4, 5]))?,
        record_batch!(
            ("id", UInt64, [10, 11, 12, 13, 14, 15, 16, 17]),
            ("src", UInt64, [0, 0, 1, 2, 3, 3, 3, 1]),
            ("dest", UInt64, [1, 2, 3, 3, 1, 4, 5, 4])
        )?,
    )
}

type PathRow = (Vec<u64>, Vec<u64>, u64, Option<Vec<u64>>);

fn rows(paths: Vec<GraphPath<'_, u64>>) -> Vec<PathRow> {
    let ids = |ids: Vec<GraphId<'_>>| {
        ids.into_iter()
            .map(|id| match id {
                GraphId::U64(n) => n,
                _ => unreachable!(),
            })
            .collect::<Vec<_>>()
    };
    let mut paths = paths
        .into_iter()
        .map(|p| (ids(p.nodes), ids(p.edges), p.state, p.intermediate_states))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

#[test]
fn cached_paths_match_with_cycles_collisions_and_eviction() -> Result<()> {
    let graph = graph()?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    pool.install(|| -> Result<()> {
        for strategy in [TraversalStrategy::DepthFirst, TraversalStrategy::BreadthFirst] {
            for parallel in [false, true] {
                for divergent in [false, true] {
                    for history in [false, true] {
                        let run = RunOptions {
                            start_nodes: vec![0_u64.into(); if parallel { 1024 } else { 1 }],
                            max_depth: Some(6), max_visits_per_node: 1, strategy, parallel,
                            intermediate_states: history, ..Default::default()
                        };
                        let calls = Arc::new(AtomicUsize::new(0));
                        let kernel = TestKernel { calls: Arc::clone(&calls), divergent };
                        let expected = graph.search_paths_with(kernel.clone(), run.clone())?;
                        let baseline_calls = calls.swap(0, Ordering::Relaxed);
                        let expected_rows = rows(expected.paths);
                        assert!(expected.stats.skipped_revisits > 0);
                        if parallel { assert!(expected.stats.parallel_edges > 0); }
                        for capacity in [0, 1, 128] {
                            let mut stream = graph.search_batches_memoized_with(kernel.clone(), run.clone(), 7, capacity)?;
                            let mut paths = Vec::new();
                            for batch in stream.by_ref() { paths.extend(batch?); }
                            let stats = stream.stats();
                            let actual_calls = calls.swap(0, Ordering::Relaxed);
                            assert_eq!(rows(paths), expected_rows, "{strategy:?} parallel={parallel}, divergent={divergent}, history={history}, capacity={capacity}");
                            assert_eq!(stats.evaluated_edges, expected.stats.evaluated_edges);
                            assert_eq!(stats.rejected_edges, expected.stats.rejected_edges);
                            assert_eq!(stats.skipped_revisits, expected.stats.skipped_revisits);
                            if capacity == 0 { assert_eq!(actual_calls, baseline_calls); }
                            if capacity == 128 && (!divergent || parallel) { assert!(actual_calls < baseline_calls); }
                        }
                    }
                }
            }
        }
        Ok(())
    })
}

#[test]
fn cache_is_fresh_per_search_and_fifo_eviction_recomputes() -> Result<()> {
    let graph = graph()?;
    let calls = Arc::new(AtomicUsize::new(0));
    let kernel = TestKernel {
        calls: Arc::clone(&calls),
        divergent: false,
    };
    let run = RunOptions {
        start_nodes: vec![0_u64.into()],
        parallel: false,
        strategy: TraversalStrategy::DepthFirst,
        ..Default::default()
    };
    let expected = rows(
        graph
            .search_paths_memoized_with(kernel.clone(), run.clone(), 128)?
            .paths,
    );
    let first = calls.swap(0, Ordering::Relaxed);
    let again = rows(
        graph
            .search_paths_memoized_with(kernel.clone(), run.clone(), 128)?
            .paths,
    );
    let second = calls.swap(0, Ordering::Relaxed);
    assert_eq!(expected, again);
    assert_eq!(first, second);
    let evicted = rows(graph.search_paths_memoized_with(kernel, run, 1)?.paths);
    assert_eq!(expected, evicted);
    assert!(calls.load(Ordering::Relaxed) > first);
    Ok(())
}

#[test]
fn serial_limits_and_multiple_visits_match_the_baseline() -> Result<()> {
    let graph = graph()?;
    let calls = Arc::new(AtomicUsize::new(0));
    let kernel = TestKernel {
        calls,
        divergent: false,
    };
    for strategy in [
        TraversalStrategy::DepthFirst,
        TraversalStrategy::BreadthFirst,
    ] {
        for limit in [0, 1, 2, 7] {
            let run = RunOptions {
                start_nodes: vec![0_u64.into()],
                max_paths: Some(limit),
                max_depth: Some(6),
                max_visits_per_node: 2,
                strategy,
                parallel: false,
                intermediate_states: true,
                ..Default::default()
            };
            let expected = graph.search_paths_with(kernel.clone(), run.clone())?.paths;
            let actual = graph
                .search_paths_memoized_with(kernel.clone(), run, 2)?
                .paths;
            assert_eq!(actual, expected);
        }
    }
    Ok(())
}

struct TrackedState {
    value: usize,
    live: Arc<AtomicUsize>,
}
impl TrackedState {
    fn new(value: usize, live: &Arc<AtomicUsize>) -> Self {
        live.fetch_add(1, Ordering::Relaxed);
        Self {
            value,
            live: Arc::clone(live),
        }
    }
}
impl Clone for TrackedState {
    fn clone(&self) -> Self {
        Self::new(self.value, &self.live)
    }
}
impl Drop for TrackedState {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::Relaxed);
    }
}
struct TrackedKernel(Arc<AtomicUsize>);
impl Kernel for TrackedKernel {
    type State = TrackedState;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<Self::State> {
        Ok(TrackedState::new(0, &self.0))
    }
    fn transition(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Transition<Self::State>> {
        let next = TrackedState::new(cx.state().value + 1, &self.0);
        Ok(if cx.dest() == 3 {
            Transition::Complete(next)
        } else {
            Transition::Continue(next)
        })
    }
}
impl MemoizableKernel for TrackedKernel {
    type Key = usize;
    fn state_key(&self, state: &Self::State) -> usize {
        state.value
    }
}

#[test]
fn bfs_drops_expanded_states_unless_history_is_requested() -> Result<()> {
    let graph = Graph::new(
        record_batch!(("id", UInt64, [0, 1, 2, 3]))?,
        record_batch!(
            ("id", UInt64, [0, 1, 2]),
            ("src", UInt64, [0, 1, 2]),
            ("dest", UInt64, [1, 2, 3])
        )?,
    )?;
    for history in [false, true] {
        let live = Arc::new(AtomicUsize::new(0));
        let run = RunOptions {
            start_nodes: vec![0_u64.into()],
            strategy: TraversalStrategy::BreadthFirst,
            parallel: false,
            intermediate_states: history,
            ..Default::default()
        };
        let mut stream =
            graph.search_batches_memoized_with(TrackedKernel(Arc::clone(&live)), run, 1, 0)?;
        let result = stream.next().unwrap()?;
        assert_eq!(result.len(), 1);
        assert_eq!(live.load(Ordering::Relaxed), if history { 8 } else { 2 });
        if history {
            assert_eq!(
                result[0]
                    .intermediate_states
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|s| s.value)
                    .collect::<Vec<_>>(),
                vec![0, 1, 2, 3]
            );
        }
        stream.close();
        assert_eq!(live.load(Ordering::Relaxed), if history { 5 } else { 1 });
        drop(result);
        assert_eq!(live.load(Ordering::Relaxed), 0);
    }
    Ok(())
}

#[test]
fn expanded_state_release_is_independent_of_cache_capacity() -> Result<()> {
    let graph = Graph::new(
        record_batch!(("id", UInt64, [0, 1, 2, 3]))?,
        record_batch!(
            ("id", UInt64, [0, 1, 2]),
            ("src", UInt64, [0, 1, 2]),
            ("dest", UInt64, [1, 2, 3])
        )?,
    )?;
    for release_expanded in [false, true] {
        let live = Arc::new(AtomicUsize::new(0));
        let run = RunOptions {
            start_nodes: vec![0_u64.into()],
            strategy: TraversalStrategy::BreadthFirst,
            parallel: false,
            ..Default::default()
        };
        let options = rxgraph::MemoOptions {
            capacity: 0,
            release_expanded,
        };
        let mut stream = graph.search_batches_memoized_with(
            TrackedKernel(Arc::clone(&live)),
            run,
            1,
            options,
        )?;
        let result = stream.next().unwrap()?;
        assert_eq!(
            live.load(Ordering::Relaxed),
            if release_expanded { 2 } else { 4 }
        );
        stream.close();
        assert_eq!(live.load(Ordering::Relaxed), 1);
        drop(result);
        assert_eq!(live.load(Ordering::Relaxed), 0);
    }
    Ok(())
}

struct ErrorKernel(TestKernel);
impl Kernel for ErrorKernel {
    type State = u64;
    fn initial_state(&self, graph: &Graph, start: NodeId) -> Result<u64> {
        self.0.initial_state(graph, start)
    }
    fn transition(&self, cx: &EdgeCtx<'_, u64>) -> Result<Transition<u64>> {
        if cx.dest() == 5 {
            bail!("intentional failure");
        }
        self.0.transition(cx)
    }
}
impl MemoizableKernel for ErrorKernel {
    type Key = CollidingKey;
    fn state_key(&self, state: &u64) -> Self::Key {
        CollidingKey(*state)
    }
}

#[test]
fn transition_errors_end_the_stream_after_delivered_results() -> Result<()> {
    let graph = graph()?;
    for strategy in [
        TraversalStrategy::DepthFirst,
        TraversalStrategy::BreadthFirst,
    ] {
        let run = RunOptions {
            start_nodes: vec![0_u64.into()],
            strategy,
            parallel: false,
            ..Default::default()
        };
        let kernel = ErrorKernel(TestKernel {
            calls: Arc::new(AtomicUsize::new(0)),
            divergent: false,
        });
        let mut stream = graph.search_batches_memoized_with(kernel, run, 1, 128)?;
        let mut delivered = 0;
        loop {
            match stream.next() {
                Some(Ok(paths)) => delivered += paths.len(),
                Some(Err(error)) => {
                    assert!(error.to_string().contains("intentional failure"));
                    break;
                }
                None => panic!("expected an error"),
            }
        }
        assert!(delivered > 0);
        assert!(stream.next().is_none());
    }
    Ok(())
}
