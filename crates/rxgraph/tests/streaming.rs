use anyhow::{Result, bail};
use arrow::array::record_batch;
use rxgraph::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone)]
struct Count {
    calls: Arc<AtomicUsize>,
    fail: bool,
}
impl Kernel for Count {
    type State = u64;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<u64> {
        Ok(0)
    }
    fn transition(&self, cx: &EdgeCtx<'_, u64>) -> Result<Transition<u64>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.fail && cx.dest() == 2 {
            bail!("intentional transition failure");
        }
        Ok(Transition::Complete(cx.state() + 1))
    }
}
fn graph() -> Graph {
    Graph::new(
        record_batch!(("id", UInt64, [0, 1, 2, 3])).unwrap(),
        record_batch!(
            ("id", UInt64, [0, 1, 2]),
            ("src", UInt64, [0, 0, 0]),
            ("dest", UInt64, [1, 2, 3])
        )
        .unwrap(),
    )
    .unwrap()
}
fn run(strategy: TraversalStrategy, parallel: bool) -> RunOptions {
    RunOptions {
        start_nodes: vec![0.into()],
        strategy,
        parallel,
        ..Default::default()
    }
}
#[test]
fn pulls_are_lazy_and_returned_paths_survive_close() -> Result<()> {
    let graph = graph();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut stream = graph.search_batches_with(
        Count {
            calls: Arc::clone(&calls),
            fail: false,
        },
        run(TraversalStrategy::DepthFirst, false),
        1,
    )?;
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let first = stream.next().unwrap()?;
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    stream.close();
    assert!(stream.next().is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(first[0].nodes, vec![GraphId::U64(0), GraphId::U64(1)]);
    assert_eq!(stream.stats().stopped_paths, 1);
    Ok(())
}
#[test]
fn errors_terminate_after_previously_delivered_results() -> Result<()> {
    let graph = graph();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut stream = graph.search_batches_with(
        Count { calls, fail: true },
        run(TraversalStrategy::BreadthFirst, false),
        1,
    )?;
    assert_eq!(stream.next().unwrap()?.len(), 1);
    assert!(
        stream
            .next()
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("intentional")
    );
    assert!(stream.next().is_none());
    Ok(())
}
#[test]
fn serial_order_global_limits_and_history() -> Result<()> {
    let graph = graph();
    for strategy in [
        TraversalStrategy::DepthFirst,
        TraversalStrategy::BreadthFirst,
    ] {
        for parallel in [false, true] {
            for limit in [0, 1, 2, 3, 4, 64] {
                for size in [1, 2, 1024] {
                    let kernel = Count {
                        calls: Arc::new(AtomicUsize::new(0)),
                        fail: false,
                    };
                    let mut run = run(strategy, parallel);
                    run.max_paths = Some(limit);
                    run.intermediate_states = true;
                    let expected = graph.search_paths_with(kernel.clone(), run.clone())?;
                    let mut stream = graph.search_batches_with(kernel, run, size)?;
                    let mut paths = vec![];
                    for batch in stream.by_ref() {
                        let batch = batch?;
                        assert!(!batch.is_empty() && batch.len() <= size);
                        paths.extend(batch);
                    }
                    assert_eq!(paths, expected.paths);
                    assert_eq!(paths.len(), limit.min(3));
                    assert_eq!(stream.stats().stopped_paths, paths.len());
                    for path in paths {
                        assert_eq!(path.intermediate_states, Some(vec![0, 1]));
                    }
                }
            }
        }
    }
    Ok(())
}
#[test]
fn parallel_workers_obey_one_global_limit() -> Result<()> {
    let graph = graph();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    pool.install(|| -> Result<()> {
        for strategy in [
            TraversalStrategy::DepthFirst,
            TraversalStrategy::BreadthFirst,
        ] {
            let mut run = run(strategy, true);
            run.start_nodes = vec![0.into(); 1024];
            run.max_paths = Some(73);
            let mut stream = graph.search_batches_with(
                Count {
                    calls: Arc::new(AtomicUsize::new(0)),
                    fail: false,
                },
                run,
                7,
            )?;
            let mut n = 0;
            for batch in stream.by_ref() {
                let batch = batch?;
                assert!(batch.len() <= 7);
                n += batch.len();
            }
            assert_eq!(n, 73);
            assert!(stream.stats().parallel_edges > 0);
        }
        Ok(())
    })
}
#[test]
fn custom_eager_only_runner_rejects_streaming() -> Result<()> {
    struct Eager;
    impl RunKernel for Eager {
        fn run<'a>(&self, _: &'a Graph, _: RunOptions) -> Result<SearchResult<'a>> {
            Ok(SearchResult {
                paths: vec![],
                stats: Default::default(),
            })
        }
    }
    assert!(
        Eager
            .stream(Arc::new(graph()), RunOptions::default())
            .is_err()
    );
    Ok(())
}
#[test]
fn native_store_cursor_retains_borrowed_payloads() -> Result<()> {
    use rxgraph::traversal::native;
    struct Native;
    impl native::Kernel for Native {
        type Node = ();
        type Edge = ();
        type State = usize;
        fn initial_state(&self, _: &native::StartCtx<'_, (), ()>) -> Result<usize> {
            Ok(0)
        }
        fn transition(
            &self,
            cx: &native::EdgeCtx<'_, '_, (), (), usize>,
        ) -> Result<Transition<usize>> {
            Ok(Transition::Complete(cx.state() + 1))
        }
    }
    let graph = graph();
    let nodes = vec![(); 4];
    let edges = vec![(); 3];
    let store = native::EagerGraphStore::new(&graph, &nodes, &edges)?;
    let mut stream =
        native::search_batches(&store, Native, run(TraversalStrategy::DepthFirst, false), 2)?;
    let first = stream.next().unwrap()?;
    assert_eq!(first.len(), 2);
    assert_eq!(stream.next().unwrap()?.len(), 1);
    assert!(stream.next().is_none());
    drop(stream);
    assert_eq!(first[0].nodes[1].external_id, Some(GraphId::U64(1)));
    Ok(())
}

#[test]
fn boxed_stream_close_releases_graph_ownership() -> Result<()> {
    let graph = Arc::new(graph());
    let calls = Arc::new(AtomicUsize::new(0));
    let runner = boxed_run(Count {
        calls: Arc::clone(&calls),
        fail: false,
    });
    let mut stream = runner.stream(
        Arc::clone(&graph),
        run(TraversalStrategy::DepthFirst, false),
    )?;
    assert_eq!(Arc::strong_count(&graph), 2);
    stream.close();
    assert_eq!(Arc::strong_count(&graph), 1);
    assert!(stream.next_batch(1)?.is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    // Scalar state cannot be encoded as an object: this failure must also release ownership.
    let mut stream = runner.stream(
        Arc::clone(&graph),
        run(TraversalStrategy::DepthFirst, false),
    )?;
    assert!(stream.next_batch(1).is_err());
    assert_eq!(Arc::strong_count(&graph), 1);
    assert!(stream.next_batch(1)?.is_none());
    Ok(())
}

#[test]
fn typed_plugin_encoding_does_not_clone_result_states() -> Result<()> {
    use rxgraph::traversal::native;
    #[derive(serde::Serialize)]
    struct State {
        value: u64,
        #[serde(skip)]
        clones: Arc<AtomicUsize>,
    }
    impl Clone for State {
        fn clone(&self) -> Self {
            self.clones.fetch_add(1, Ordering::Relaxed);
            Self {
                value: self.value,
                clones: Arc::clone(&self.clones),
            }
        }
    }
    #[derive(Clone)]
    struct Typed(Arc<AtomicUsize>);
    impl TypedKernel for Typed {
        type Node = ();
        type Edge = ();
        type State = State;
        fn initial_state(&self, _: &native::StartCtx<'_, (), ()>) -> Result<State> {
            Ok(State {
                value: 0,
                clones: Arc::clone(&self.0),
            })
        }
        fn transition(
            &self,
            cx: &native::EdgeCtx<'_, '_, (), (), State>,
        ) -> Result<Transition<State>> {
            Ok(Transition::Complete(State {
                value: cx.state().value + 1,
                clones: Arc::clone(&self.0),
            }))
        }
    }
    let graph = Arc::new(graph());
    let clones = Arc::new(AtomicUsize::new(0));
    let runner = boxed_typed_run(Typed(Arc::clone(&clones)));
    let mut run = run(TraversalStrategy::DepthFirst, false);
    run.intermediate_states = true;
    let eager = runner.run_eager(&graph, run.clone())?;
    let mut stream =
        runner.stream_eager_cached(Arc::clone(&graph), &TypedPayloadCache::default(), run)?;
    let mut index = 0;
    while let Some(batch) = stream.next_batch(1)? {
        assert_eq!(batch.paths[0].state, eager.paths[index].state);
        assert_eq!(
            batch.paths[0].intermediate_states,
            eager.paths[index].intermediate_states
        );
        index += 1;
    }
    assert_eq!(index, 3);
    assert_eq!(clones.load(Ordering::Relaxed), 0);
    Ok(())
}
