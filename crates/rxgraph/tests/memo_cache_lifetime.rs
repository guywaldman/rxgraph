//! Cached state must be released when a composable stream finishes or closes.
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::{Result, bail};
use arrow::array::record_batch;
use rxgraph::{
    EdgeCtx, Evaluation, Graph, Kernel, MemoizableKernel, NodeId, RunOptions, TargetOptions,
    Transition, TraversalStrategy,
};

fn optimized(run: RunOptions, options: TargetOptions) -> rxgraph::NativeRunOptions {
    run.with_optimizations(options)
}

struct State {
    value: u64,
    live: Arc<AtomicUsize>,
}
impl State {
    fn new(value: u64, live: &Arc<AtomicUsize>) -> Self {
        live.fetch_add(1, Ordering::Relaxed);
        Self {
            value,
            live: Arc::clone(live),
        }
    }
}
impl Clone for State {
    fn clone(&self) -> Self {
        Self::new(self.value, &self.live)
    }
}
impl Drop for State {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::Relaxed);
    }
}

struct TestKernel {
    live: Arc<AtomicUsize>,
    fail: bool,
}
impl Kernel for TestKernel {
    type State = State;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<State> {
        Ok(State::new(0, &self.live))
    }
    fn transition(&self, cx: &EdgeCtx<'_, State>) -> Result<Transition<State>> {
        if self.fail && cx.edge() == 1 {
            bail!("intentional failure");
        }
        Ok(Transition::Complete(State::new(
            cx.state().value + 1,
            &self.live,
        )))
    }
}
impl MemoizableKernel for TestKernel {
    type Key = u64;
    fn state_key(&self, state: &State) -> u64 {
        state.value
    }
}

#[derive(Clone, Copy, Debug)]
enum End {
    Close,
    Exhaust,
    Error,
}

fn graph() -> Result<Graph> {
    Graph::new(
        record_batch!(("id", UInt64, [0, 1, 2]))?,
        record_batch!(
            ("id", UInt64, [0, 1]),
            ("src", UInt64, [0, 0]),
            ("dest", UInt64, [1, 2])
        )?,
    )
}

#[test]
fn cached_states_drop_before_stream_object_after_close_exhaustion_or_error() -> Result<()> {
    let graph = graph()?;
    for evaluation in [Evaluation::Incremental, Evaluation::Lazy] {
        for end in [End::Close, End::Exhaust, End::Error] {
            let live = Arc::new(AtomicUsize::new(0));
            let kernel = TestKernel {
                live: Arc::clone(&live),
                fail: matches!(end, End::Error),
            }
            .memoized(16);
            let run = RunOptions {
                start_nodes: vec![0u64.into()],
                strategy: TraversalStrategy::DepthFirst,
                parallel: false,
                ..Default::default()
            };
            let options = TargetOptions {
                evaluation,
                ..Default::default()
            };
            let mut stream = graph.search_batches_with(kernel, optimized(run, options), 1)?;
            let first = stream.next().expect("first result")?;
            assert_eq!(first.len(), 1);
            drop(first);
            match end {
                End::Close => stream.close(),
                End::Exhaust => {
                    for batch in stream.by_ref() {
                        drop(batch?);
                    }
                }
                End::Error => {
                    let error = match stream.next().expect("error") {
                        Ok(_) => panic!("expected error"),
                        Err(error) => error,
                    };
                    assert!(error.to_string().contains("intentional failure"));
                }
            }
            let stats = stream.stats();
            assert!(stream.next().is_none());
            assert_eq!(
                stream.stats(),
                stats,
                "stats survive close: {evaluation:?}/{end:?}"
            );
            assert_eq!(
                live.load(Ordering::Relaxed),
                0,
                "search-owned cached states remain after {evaluation:?}/{end:?}"
            );
            // Keep `stream` alive through the assertion: dropping it would hide
            // a cache retained inside the kernel wrapper after close.
            drop(stream);
        }
    }
    Ok(())
}

#[test]
fn regular_stream_releases_composed_cache_on_close_exhaustion_and_error() -> Result<()> {
    let graph = graph()?;
    for end in [End::Close, End::Exhaust, End::Error] {
        let live = Arc::new(AtomicUsize::new(0));
        let kernel = TestKernel {
            live: Arc::clone(&live),
            fail: matches!(end, End::Error),
        }
        .memoized(16);
        let run = RunOptions {
            start_nodes: vec![0u64.into()],
            strategy: TraversalStrategy::DepthFirst,
            parallel: false,
            ..Default::default()
        };
        let mut stream = graph.search_batches_with(kernel, run, 1)?;
        drop(stream.next().expect("first result")?);
        match end {
            End::Close => stream.close(),
            End::Exhaust => {
                for batch in stream.by_ref() {
                    drop(batch?);
                }
            }
            End::Error => {
                let error = match stream.next().expect("error") {
                    Ok(_) => panic!("expected error"),
                    Err(error) => error,
                };
                assert!(error.to_string().contains("intentional failure"));
            }
        }
        let stats = stream.stats();
        assert!(stream.next().is_none());
        assert_eq!(stream.stats(), stats);
        assert_eq!(
            live.load(Ordering::Relaxed),
            0,
            "cached states remain after {end:?}"
        );
        drop(stream);
    }
    Ok(())
}

#[test]
fn returned_states_outlive_the_closed_stream() -> Result<()> {
    let graph = graph()?;
    for evaluation in [Evaluation::Incremental, Evaluation::Lazy] {
        let live = Arc::new(AtomicUsize::new(0));
        let kernel = TestKernel {
            live: Arc::clone(&live),
            fail: false,
        }
        .memoized(16);
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
                    evaluation,
                    ..Default::default()
                },
            ),
            1,
        )?;
        let first = stream.next().expect("first result")?;
        stream.close();
        assert_eq!(first[0].state.value, 1);
        assert_eq!(
            live.load(Ordering::Relaxed),
            1,
            "only the returned state should remain"
        );
        drop(first);
        assert_eq!(live.load(Ordering::Relaxed), 0);
    }
    Ok(())
}
