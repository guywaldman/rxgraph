use std::borrow::Cow;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::{Result, bail};
use arrow::array::record_batch;
use rxgraph::{
    EdgeCtx, Evaluation, Graph, GraphId, GraphPath, Kernel, NodeId, RunOptions, TargetCtx,
    TargetOptions, Transition, TraversalStrategy,
};

fn optimized(run: RunOptions, options: TargetOptions) -> rxgraph::NativeRunOptions {
    run.with_optimizations(options)
}

#[derive(Clone)]
struct Example {
    calls: Arc<AtomicUsize>,
    fail: bool,
}

impl Kernel for Example {
    type State = u64;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<u64> {
        Ok(0)
    }
    fn transition(&self, cx: &EdgeCtx<'_, u64>) -> Result<Transition<u64>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.fail && cx.edge() == 4 {
            bail!("intentional failure");
        }
        let next = *cx.state() + u64::from(cx.edge()) + 1;
        Ok(
            if cx.dest() == 3 || (cx.src() == 2 && cx.dest() == 1 && *cx.state() == 2) {
                Transition::Complete(next)
            } else {
                Transition::Continue(next)
            },
        )
    }
    fn possible_targets<'a>(
        &'a self,
        cx: &TargetCtx<'_, u64>,
    ) -> Result<Option<Cow<'a, [NodeId]>>> {
        let targets = match cx {
            TargetCtx::Start { .. } => Some(vec![1, 3]),
            TargetCtx::Edge { edge, .. } if edge.edge() == 2 => Some(vec![]),
            TargetCtx::Edge { edge, .. } if edge.dest() == 3 => Some(vec![3]),
            TargetCtx::Edge {
                edge,
                parent: Some(value),
                ..
            } if edge.edge() == 0 && **value == 0 => Some(vec![3]),
            TargetCtx::Edge {
                edge,
                parent: Some(value),
                ..
            } if edge.edge() == 5 && **value != 2 => Some(vec![3]),
            _ => None,
        };
        Ok(targets.map(Cow::Owned))
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

struct Unknown(Example);
impl Kernel for Unknown {
    type State = u64;
    fn initial_state(&self, graph: &Graph, start: NodeId) -> Result<u64> {
        self.0.initial_state(graph, start)
    }
    fn transition(&self, cx: &EdgeCtx<'_, u64>) -> Result<Transition<u64>> {
        self.0.transition(cx)
    }
}

#[test]
fn unknown_targets_consider_every_node_a_candidate() -> Result<()> {
    let graph = graph()?;
    for strategy in [
        TraversalStrategy::DepthFirst,
        TraversalStrategy::BreadthFirst,
    ] {
        let run = RunOptions {
            start_nodes: vec![0u64.into()],
            strategy,
            parallel: false,
            max_depth: Some(5),
            intermediate_states: true,
            ..Default::default()
        };
        let kernel = Example {
            calls: Arc::new(AtomicUsize::new(0)),
            fail: false,
        };
        let expected = graph.search_paths_with(Unknown(kernel.clone()), run.clone())?;
        let actual = graph.search_paths_with(
            Unknown(kernel),
            optimized(
                run,
                TargetOptions {
                    reverse: true,
                    release_expanded: false,
                    evaluation: Evaluation::Lazy,
                },
            ),
        )?;
        assert_eq!(rows(actual.paths), rows(expected.paths));
    }
    Ok(())
}
type Row = (Vec<u64>, Vec<u64>, u64, Option<Vec<u64>>);
fn rows(paths: Vec<GraphPath<'_, u64>>) -> Vec<Row> {
    let ids = |v: Vec<GraphId<'_>>| {
        v.into_iter()
            .map(|id| match id {
                GraphId::U64(n) => n,
                _ => unreachable!(),
            })
            .collect()
    };
    paths
        .into_iter()
        .map(|p| (ids(p.nodes), ids(p.edges), p.state, p.intermediate_states))
        .collect()
}

#[test]
fn deferred_matches_exact_paths_history_cycles_and_limits() -> Result<()> {
    let graph = graph()?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    pool.install(||->Result<()> {
        for strategy in [TraversalStrategy::DepthFirst,TraversalStrategy::BreadthFirst] {
            for parallel in [false,true] {
                for visits in [1,2] {
                    for history in [false,true] {
                        for reverse in [false,true] {
                            for release in [false,true] {
                                let run=RunOptions {start_nodes:vec![0u64.into();if parallel{512}else{1}],strategy,parallel,max_depth:Some(6),max_visits_per_node:visits,intermediate_states:history,..Default::default()};
                                let kernel=Example{calls:Arc::new(AtomicUsize::new(0)),fail:false};
                                let expected=graph.search_paths_with(kernel.clone(),run.clone())?;
                                let mut stream=graph.search_batches_with(kernel,optimized(run,TargetOptions{reverse,release_expanded:release,evaluation:Evaluation::Lazy}),7)?;
                                let mut actual=Vec::new();for batch in stream.by_ref(){actual.extend(batch?);}
                                let (mut expected,mut actual)=(rows(expected.paths),rows(actual));
                                if parallel {expected.sort();actual.sort();}
                                assert_eq!(actual,expected,"{strategy:?} parallel={parallel} visits={visits} history={history} reverse={reverse} release={release}");
                                assert!(actual.iter().any(|p|p.0==[0,1,3]),"possible ending 1 must be traversable as an intermediate");
                                assert!(actual.iter().any(|p|p.0==[0,2,1]),"1 can also be a completed path");
                            }
                        }
                    }
                }
            }
            for limit in [0,1,2,20] {
                let run=RunOptions{start_nodes:vec![0u64.into()],strategy,parallel:false,max_depth:Some(6),max_visits_per_node:2,max_paths:Some(limit),intermediate_states:true,..Default::default()};
                let kernel=Example{calls:Arc::new(AtomicUsize::new(0)),fail:false};
                let expected=graph.search_paths_with(kernel.clone(),run.clone())?;
                let actual=graph.search_paths_with(kernel,optimized(run,TargetOptions{reverse:true,release_expanded:true,evaluation:Evaluation::Lazy}))?;
                assert_eq!(rows(actual.paths),rows(expected.paths));
            }
        }
        Ok(())
    })
}

#[test]
fn deferred_propagates_errors_and_does_not_execute_after_close() -> Result<()> {
    let graph = graph()?;
    let calls = Arc::new(AtomicUsize::new(0));
    let kernel = Example {
        calls: Arc::clone(&calls),
        fail: true,
    };
    let run = RunOptions {
        start_nodes: vec![0u64.into()],
        strategy: TraversalStrategy::DepthFirst,
        parallel: false,
        ..Default::default()
    };
    let mut stream = graph.search_batches_with(
        kernel,
        optimized(
            run,
            TargetOptions {
                evaluation: Evaluation::Lazy,
                ..Default::default()
            },
        ),
        1,
    )?;
    let mut delivered = 0;
    loop {
        match stream.next() {
            Some(Ok(batch)) => delivered += batch.len(),
            Some(Err(error)) => {
                assert!(error.to_string().contains("intentional failure"));
                break;
            }
            None => panic!("expected error"),
        }
    }
    assert!(delivered > 0);
    let before = calls.load(Ordering::Relaxed);
    assert!(stream.next().is_none());
    assert_eq!(calls.load(Ordering::Relaxed), before);
    Ok(())
}

struct Spy {
    hooks: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
}
impl Kernel for Spy {
    type State = ();
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<()> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn transition(&self, _: &EdgeCtx<'_, ()>) -> Result<Transition<()>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(Transition::Reject)
    }
    fn possible_targets<'a>(&'a self, _: &TargetCtx<'_, ()>) -> Result<Option<Cow<'a, [NodeId]>>> {
        self.hooks.fetch_add(1, Ordering::Relaxed);
        Ok(Some(Cow::Owned(vec![])))
    }
}
#[test]
fn deferred_empty_and_zero_limit_skip_kernel_work() -> Result<()> {
    let graph = graph()?;
    for limit in [None, Some(0)] {
        let hooks = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        let run = RunOptions {
            start_nodes: vec![0u64.into()],
            parallel: false,
            max_paths: limit,
            ..Default::default()
        };
        let mut stream = graph.search_batches_with(
            Spy {
                hooks: Arc::clone(&hooks),
                calls: Arc::clone(&calls),
            },
            optimized(
                run,
                TargetOptions {
                    reverse: true,
                    release_expanded: false,
                    evaluation: Evaluation::Lazy,
                },
            ),
            1,
        )?;
        assert!(stream.next().is_none());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        if limit == Some(0) {
            assert_eq!(hooks.load(Ordering::Relaxed), 0);
        }
    }
    Ok(())
}

struct InvalidPostBound;
impl Kernel for InvalidPostBound {
    type State = u64;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<u64> {
        Ok(0)
    }
    fn transition(&self, cx: &EdgeCtx<'_, u64>) -> Result<Transition<u64>> {
        Ok(Transition::Complete(*cx.state() + 1))
    }
    fn possible_targets<'a>(
        &'a self,
        cx: &TargetCtx<'_, u64>,
    ) -> Result<Option<Cow<'a, [NodeId]>>> {
        Ok((match cx {
            TargetCtx::Start { .. } => Some(vec![1]),
            TargetCtx::Edge { child: Some(_), .. } => Some(vec![]),
            _ => None,
        })
        .map(Cow::Owned))
    }
}

#[test]
fn completed_path_excluded_by_post_bound_reports_contract_error() -> Result<()> {
    let graph = Graph::new(
        record_batch!(("id", UInt64, [0, 1]))?,
        record_batch!(
            ("id", UInt64, [0]),
            ("src", UInt64, [0]),
            ("dest", UInt64, [1])
        )?,
    )?;
    for evaluation in [Evaluation::Incremental, Evaluation::Lazy] {
        let run = RunOptions {
            start_nodes: vec![0u64.into()],
            parallel: false,
            ..Default::default()
        };
        let result = graph.search_paths_with(
            InvalidPostBound,
            optimized(
                run,
                TargetOptions {
                    evaluation,
                    ..Default::default()
                },
            ),
        );
        let error = match result {
            Ok(_) => panic!("expected invalid completion error for {evaluation:?}"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("excluded a completed"),
            "{evaluation:?}: {error}"
        );
    }
    Ok(())
}
