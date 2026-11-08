use anyhow::Result;
use arrow::array::record_batch;
use rxgraph::{
    EdgeCtx, Graph, Kernel, MemoizableKernel, NodeId, OwnedGraphId, RunOptions, TargetCtx,
    TargetOptions, Transition, TraversalStrategy,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn optimized(run: RunOptions, options: TargetOptions) -> rxgraph::NativeRunOptions {
    run.with_optimizations(options)
}

struct HookSpy(Arc<AtomicUsize>);
impl Kernel for HookSpy {
    type State = ();
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<()> {
        Ok(())
    }
    fn transition(&self, _: &EdgeCtx<'_, ()>) -> Result<Transition<()>> {
        Ok(Transition::Complete(()))
    }
    fn possible_targets(&self, cx: &TargetCtx<'_, ()>) -> Result<Option<Vec<OwnedGraphId>>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(match cx {
            TargetCtx::Start { .. } => Some(vec![1u64.into(), 2u64.into(), 4u64.into()]),
            TargetCtx::Edge { .. } => None,
        })
    }
}

#[test]
fn native_search_uses_target_hooks_only_when_options_are_requested() -> Result<()> {
    let graph = graph()?;
    let hooks = Arc::new(AtomicUsize::new(0));
    let run = RunOptions {
        start_nodes: vec![0u64.into()],
        parallel: false,
        ..Default::default()
    };
    let baseline = graph.search_paths_with(HookSpy(Arc::clone(&hooks)), run.clone())?;
    assert_eq!(baseline.paths.len(), 3);
    assert_eq!(hooks.load(Ordering::Relaxed), 0);
    let tuned = graph.search_paths_with(
        HookSpy(Arc::clone(&hooks)),
        optimized(run.clone(), TargetOptions::default()),
    )?;
    assert_eq!(tuned.paths, baseline.paths);
    assert!(hooks.swap(0, Ordering::Relaxed) > 0);
    let mut stream = graph.search_batches_with(HookSpy(Arc::clone(&hooks)), run, 1)?;
    let mut count = 0;
    for batch in stream.by_ref() {
        count += batch?.len();
    }
    assert_eq!(count, 3);
    assert_eq!(hooks.load(Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn first_search_honors_optimization_options_and_keeps_canonical_result() -> Result<()> {
    let graph = graph()?;
    let hooks = Arc::new(AtomicUsize::new(0));
    let run = RunOptions {
        start_nodes: vec![0u64.into()],
        parallel: true,
        ..Default::default()
    };
    let expected = graph.search_first_with(HookSpy(Arc::clone(&hooks)), run.clone())?;
    assert_eq!(hooks.load(Ordering::Relaxed), 0);
    let options = TargetOptions {
        evaluation: rxgraph::Evaluation::Lazy,
        reverse: true,
        ..Default::default()
    };
    let actual = graph.search_first_with(HookSpy(Arc::clone(&hooks)), optimized(run, options))?;
    assert_eq!(actual.path, expected.path);
    assert!(hooks.load(Ordering::Relaxed) > 0);
    Ok(())
}

#[derive(Clone, Copy)]
struct K;
impl Kernel for K {
    type State = u64;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<u64> {
        Ok(0)
    }
    fn transition(&self, cx: &EdgeCtx<'_, u64>) -> Result<Transition<u64>> {
        let n = cx.state() + cx.edge() as u64 + 1;
        Ok(if cx.dest() == 4 {
            Transition::Reject
        } else if cx.dest() == 3 || (cx.src() == 2 && cx.dest() == 1) {
            Transition::Complete(n)
        } else {
            Transition::Continue(n)
        })
    }
    fn possible_targets(&self, cx: &TargetCtx<'_, u64>) -> Result<Option<Vec<OwnedGraphId>>> {
        Ok(match cx {
            TargetCtx::Start { .. } => Some(vec![1u64.into(), 3u64.into()]),
            TargetCtx::Edge {
                edge, child: None, ..
            } if edge.dest() == 4 => Some(vec![]),
            TargetCtx::Edge {
                edge, child: None, ..
            } if edge.dest() == 3 => Some(vec![3u64.into()]),
            TargetCtx::Edge {
                edge, child: None, ..
            } if edge.src() == 2 && edge.dest() == 1 => Some(vec![1u64.into()]),
            _ => None,
        })
    }
}
fn graph() -> Result<Graph> {
    Graph::new(
        record_batch!(("id", UInt64, [0, 1, 2, 3, 4]))?,
        record_batch!(
            ("id", UInt64, [0, 1, 2, 3, 4, 5, 6]),
            ("src", UInt64, [0, 0, 0, 1, 1, 2, 2]),
            ("dest", UInt64, [1, 2, 4, 2, 3, 1, 3])
        )?,
    )
}
#[test]
fn targeted_eager_matches_ordinary_search_with_cycles_and_intermediate_targets() -> Result<()> {
    let graph = graph()?;
    for strategy in [
        TraversalStrategy::DepthFirst,
        TraversalStrategy::BreadthFirst,
    ] {
        for visits in [1, 2] {
            for history in [false, true] {
                for limit in [None, Some(0), Some(1), Some(2), Some(20)] {
                    let run = RunOptions {
                        start_nodes: vec![0u64.into()],
                        strategy,
                        parallel: false,
                        max_depth: Some(6),
                        max_visits_per_node: visits,
                        intermediate_states: history,
                        max_paths: limit,
                        ..Default::default()
                    };
                    let expected = graph.search_paths_with(K, run.clone())?.paths;
                    for reverse in [false, true] {
                        for release_expanded in [false, true] {
                            let options = TargetOptions {
                                reverse,
                                release_expanded,
                                ..Default::default()
                            };
                            let actual = graph
                                .search_paths_with(K, optimized(run.clone(), options))?
                                .paths;
                            assert_eq!(
                                actual, expected,
                                "{strategy:?} visits={visits} history={history} limit={limit:?} reverse={reverse} release={release_expanded}"
                            );
                            let mut stream =
                                graph.search_batches_with(K, optimized(run.clone(), options), 1)?;
                            let mut streamed = Vec::new();
                            for batch in stream.by_ref() {
                                streamed.extend(batch?);
                            }
                            assert_eq!(streamed, expected);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

struct Count(Arc<AtomicUsize>);
impl Kernel for Count {
    type State = ();
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<()> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn transition(&self, _: &EdgeCtx<'_, ()>) -> Result<Transition<()>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(Transition::Complete(()))
    }
    fn possible_targets(&self, _: &TargetCtx<'_, ()>) -> Result<Option<Vec<OwnedGraphId>>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(Some(vec![3u64.into()]))
    }
}
#[test]
fn reverse_stream_is_lazy_and_zero_limit_does_no_kernel_work() -> Result<()> {
    let graph = graph()?;
    let calls = Arc::new(AtomicUsize::new(0));
    let run = RunOptions {
        start_nodes: vec![0u64.into()],
        max_paths: Some(0),
        ..Default::default()
    };
    let options = TargetOptions {
        reverse: true,
        release_expanded: false,
        ..Default::default()
    };
    let mut stream = graph.search_batches_with(
        Count(Arc::clone(&calls)),
        optimized(run.clone(), options),
        1,
    )?;
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(stream.next().is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(
        graph
            .search_paths_with(Count(Arc::clone(&calls)), optimized(run, options))?
            .paths
            .is_empty()
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    Ok(())
}

struct Reconverging(Arc<AtomicUsize>);
impl Kernel for Reconverging {
    type State = u64;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<u64> {
        Ok(0)
    }
    fn transition(&self, cx: &EdgeCtx<'_, u64>) -> Result<Transition<u64>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        let state = if cx.src() == 0 {
            cx.dest() as u64
        } else {
            *cx.state()
        };
        Ok(if cx.dest() == 3 || cx.dest() == 4 {
            if cx.dest() as u64 == state + 2 {
                Transition::Complete(state)
            } else {
                Transition::Reject
            }
        } else {
            Transition::Continue(state)
        })
    }
    fn possible_targets(&self, cx: &TargetCtx<'_, u64>) -> Result<Option<Vec<OwnedGraphId>>> {
        Ok(match cx {
            TargetCtx::Start { .. } => Some(vec![3u64.into(), 4u64.into()]),
            TargetCtx::Edge {
                edge, child: None, ..
            } if edge.src() == 0 => Some(vec![(edge.dest() as u64 + 2).into()]),
            _ => None,
        })
    }
}
impl MemoizableKernel for Reconverging {
    type Key = u64;
    fn state_key(&self, state: &u64) -> u64 {
        *state
    }
}
#[test]
fn reverse_intersects_each_reconverging_paths_own_target_bound() -> Result<()> {
    let graph = Graph::new(
        record_batch!(("id", UInt64, [0, 1, 2, 3, 4, 5]))?,
        record_batch!(
            ("id", UInt64, [0, 1, 2, 3, 4, 5]),
            ("src", UInt64, [0, 0, 1, 2, 5, 5]),
            ("dest", UInt64, [1, 2, 5, 5, 3, 4])
        )?,
    )?;
    for strategy in [
        TraversalStrategy::DepthFirst,
        TraversalStrategy::BreadthFirst,
    ] {
        let run = RunOptions {
            start_nodes: vec![0u64.into()],
            strategy,
            parallel: false,
            ..Default::default()
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let expected = graph
            .search_paths_with(Reconverging(Arc::clone(&calls)), run.clone())?
            .paths;
        calls.store(0, Ordering::Relaxed);
        let paths = graph
            .search_paths_with(
                Reconverging(Arc::clone(&calls)),
                optimized(
                    run,
                    TargetOptions {
                        reverse: true,
                        release_expanded: false,
                        ..Default::default()
                    },
                ),
            )?
            .paths;
        assert_eq!(paths, expected);
        assert_eq!(paths.len(), 2);
        assert_eq!(calls.load(Ordering::Relaxed), 6);
    }
    Ok(())
}

#[test]
fn target_refinement_composes_with_memoization() -> Result<()> {
    let graph = Graph::new(
        record_batch!(("id", UInt64, [0, 1, 2, 3, 4, 5]))?,
        record_batch!(
            ("id", UInt64, [0, 1, 2, 3, 4, 5]),
            ("src", UInt64, [0, 0, 1, 2, 5, 5]),
            ("dest", UInt64, [1, 2, 5, 5, 3, 4])
        )?,
    )?;
    let options = TargetOptions {
        reverse: true,
        release_expanded: false,
        ..Default::default()
    };
    for strategy in [
        TraversalStrategy::DepthFirst,
        TraversalStrategy::BreadthFirst,
    ] {
        let run = RunOptions {
            start_nodes: vec![0u64.into(); 8],
            strategy,
            parallel: false,
            ..Default::default()
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let expected = graph
            .search_paths_with(
                Reconverging(Arc::clone(&calls)),
                optimized(run.clone(), options),
            )?
            .paths;
        let baseline = calls.swap(0, Ordering::Relaxed);
        let paths = graph
            .search_paths_with(
                Reconverging(Arc::clone(&calls)).memoized(128),
                optimized(run, options),
            )?
            .paths;
        assert_eq!(paths, expected);
        assert_eq!(paths.len(), 16);
        assert!(calls.load(Ordering::Relaxed) < baseline);
    }
    Ok(())
}

struct ManyTargets;
impl Kernel for ManyTargets {
    type State = ();
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<()> {
        Ok(())
    }
    fn transition(&self, _: &EdgeCtx<'_, ()>) -> Result<Transition<()>> {
        Ok(Transition::Complete(()))
    }
    fn possible_targets(&self, cx: &TargetCtx<'_, ()>) -> Result<Option<Vec<OwnedGraphId>>> {
        Ok(if matches!(cx, TargetCtx::Start { .. }) {
            Some((1..=10u64).map(Into::into).collect())
        } else {
            None
        })
    }
}
#[test]
fn many_targets_use_conservative_reverse_fallback() -> Result<()> {
    let graph = Graph::new(
        record_batch!(("id", UInt64, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))?,
        record_batch!(
            ("id", UInt64, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            ("src", UInt64, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            ("dest", UInt64, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        )?,
    )?;
    let run = RunOptions {
        start_nodes: vec![0u64.into()],
        parallel: false,
        ..Default::default()
    };
    let expected = graph.search_paths_with(ManyTargets, run.clone())?.paths;
    let actual = graph
        .search_paths_with(
            ManyTargets,
            optimized(
                run,
                TargetOptions {
                    reverse: true,
                    ..Default::default()
                },
            ),
        )?
        .paths;
    assert_eq!(actual, expected);
    assert_eq!(actual.len(), 10);
    Ok(())
}

struct EmptyTargets(Arc<AtomicUsize>);
impl Kernel for EmptyTargets {
    type State = ();
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<()> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn transition(&self, _: &EdgeCtx<'_, ()>) -> Result<Transition<()>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(Transition::Complete(()))
    }
    fn possible_targets(&self, cx: &TargetCtx<'_, ()>) -> Result<Option<Vec<OwnedGraphId>>> {
        Ok(if matches!(cx, TargetCtx::Start { .. }) {
            Some(vec![])
        } else {
            None
        })
    }
}
#[test]
fn empty_initial_targets_skip_exact_kernel_work() -> Result<()> {
    let graph = graph()?;
    for strategy in [
        TraversalStrategy::DepthFirst,
        TraversalStrategy::BreadthFirst,
    ] {
        for evaluation in [rxgraph::Evaluation::Incremental, rxgraph::Evaluation::Lazy] {
            for reverse in [false, true] {
                let calls = Arc::new(AtomicUsize::new(0));
                let run = RunOptions {
                    start_nodes: vec![0u64.into()],
                    strategy,
                    parallel: false,
                    ..Default::default()
                };
                let options = TargetOptions {
                    evaluation,
                    reverse,
                    ..Default::default()
                };
                assert!(
                    graph
                        .search_paths_with(
                            EmptyTargets(Arc::clone(&calls)),
                            optimized(run, options)
                        )?
                        .paths
                        .is_empty()
                );
                assert_eq!(calls.load(Ordering::Relaxed), 0);
            }
        }
    }
    Ok(())
}
