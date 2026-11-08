//! Target-aware search using the existing native Budget benchmark kernel.
#[allow(dead_code)]
#[path = "../benches/native_fixture.rs"]
mod native_fixture;

use anyhow::Result;
use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use native_fixture::{Budget, State};
use rxgraph::{
    EdgeCtx, Evaluation, Graph, GraphId, GraphPath, Kernel, NodeId, RunOptions, TargetCtx,
    TargetOptions, Transition, TraversalStrategy,
};
use serde_json::json;
use std::{
    borrow::Cow,
    hint::black_box,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

struct Fixture {
    graph: Graph,
    kernel: Budget,
    starts: usize,
    depth: usize,
}
impl Fixture {
    fn new(case: &str) -> Result<Self> {
        let (starts, depth, fanout, budget, large) = match case {
            "reject" => (64u64, 2u64, 128u64, 0u64, false),
            "large_state" => (32u64, 32u64, 1u64, 32u64, true),
            _ => anyhow::bail!("unknown case {case}"),
        };
        let terminal = starts * depth;
        let mut src = Vec::new();
        let mut dest = Vec::new();
        let mut cost = Vec::new();
        for start in 0..starts {
            for level in 0..depth {
                let node = start * depth + level;
                src.push(node);
                dest.push(if level == depth - 1 {
                    terminal
                } else {
                    node + 1
                });
                cost.push(1);
                for _ in 1..fanout {
                    src.push(node);
                    dest.push(terminal);
                    cost.push(depth + 1);
                }
            }
        }
        let col = |v| Arc::new(UInt64Array::from(v)) as ArrayRef;
        let nodes = RecordBatch::try_from_iter([("id", col((0..=terminal).collect::<Vec<_>>()))])?;
        let edges = RecordBatch::try_from_iter([
            ("id", col((0..src.len() as u64).collect::<Vec<_>>())),
            ("src", col(src)),
            ("dest", col(dest)),
            ("cost", col(cost)),
        ])?;
        let graph = Graph::new(nodes, edges)?;
        let kernel = Budget {
            budget,
            target: terminal,
            large,
            cost: graph.edge_field("cost")?,
        };
        Ok(Self {
            graph,
            kernel,
            starts: starts as usize,
            depth: depth as usize,
        })
    }
}

struct WithTargets<'a, const COUNT: bool> {
    inner: &'a Budget,
    target: u64,
    calls: &'a AtomicUsize,
}
impl<const COUNT: bool> Kernel for WithTargets<'_, COUNT> {
    type State = State;
    fn initial_state(&self, graph: &Graph, start: NodeId) -> Result<State> {
        self.inner.initial_state(graph, start)
    }
    fn transition(&self, cx: &EdgeCtx<'_, State>) -> Result<Transition<State>> {
        if COUNT {
            self.calls.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.transition(cx)
    }
    fn possible_targets<'a>(
        &'a self,
        cx: &TargetCtx<'_, State>,
    ) -> Result<Option<Cow<'a, [NodeId]>>> {
        Ok(if matches!(cx, TargetCtx::Start { .. }) {
            Some(Cow::Owned(vec![self.target as NodeId]))
        } else {
            None
        })
    }
}

type Row = (Vec<u64>, Vec<u64>, String);
fn row(path: GraphPath<'_, State>) -> Result<Row> {
    fn ids(items: Vec<GraphId<'_>>) -> Vec<u64> {
        items
            .into_iter()
            .map(|id| match id {
                GraphId::U64(value) => value,
                _ => unreachable!(),
            })
            .collect()
    }
    Ok((
        ids(path.nodes),
        ids(path.edges),
        serde_json::to_string(&path.state)?,
    ))
}
fn run<const COUNT: bool, const AUDIT: bool>(
    fixture: &Fixture,
    strategy: TraversalStrategy,
    arm: usize,
) -> Result<(Vec<Row>, usize, usize, usize)> {
    let calls = AtomicUsize::new(0);
    let run = RunOptions {
        start_nodes: (0..fixture.starts)
            .map(|i| ((i * fixture.depth) as u64).into())
            .collect(),
        max_depth: Some(fixture.depth),
        strategy,
        parallel: false,
        ..Default::default()
    };
    let kernel = WithTargets::<COUNT> {
        inner: &fixture.kernel,
        target: fixture.kernel.target,
        calls: &calls,
    };
    let mut rows = Vec::new();
    let mut count = 0;
    let evaluated = if arm == 0 {
        let mut stream = fixture.graph.search_batches_with(kernel, run, 1024)?;
        for paths in stream.by_ref() {
            for path in paths? {
                count += 1;
                if AUDIT {
                    rows.push(row(path)?);
                } else {
                    black_box(path);
                }
            }
        }
        stream.stats().evaluated_edges
    } else {
        let options = TargetOptions {
            evaluation: Evaluation::Incremental,
            reverse: arm == 2,
            release_expanded: false,
        };
        let mut stream =
            fixture
                .graph
                .search_batches_with(kernel, run.with_optimizations(options), 1024)?;
        for paths in stream.by_ref() {
            for path in paths? {
                count += 1;
                if AUDIT {
                    rows.push(row(path)?);
                } else {
                    black_box(path);
                }
            }
        }
        stream.stats().evaluated_edges
    };
    Ok((rows, count, evaluated, calls.load(Ordering::Relaxed)))
}

fn main() -> Result<()> {
    anyhow::ensure!(!cfg!(debug_assertions), "use --release");
    let case = std::env::var("RX_NATIVE_SIGNOFF_CASE").unwrap_or_else(|_| "reject".into());
    let strategy = if std::env::var("RX_NATIVE_SIGNOFF_STRATEGY").as_deref() == Ok("bfs") {
        TraversalStrategy::BreadthFirst
    } else {
        TraversalStrategy::DepthFirst
    };
    let mode = std::env::var("RX_NATIVE_SIGNOFF_MODE").unwrap_or_else(|_| "timing".into());
    let arm: usize = std::env::var("RX_NATIVE_SIGNOFF_ARM")
        .unwrap_or_else(|_| "0".into())
        .parse()?;
    anyhow::ensure!(arm < 3, "invalid arm");
    let fixture = Fixture::new(&case)?;
    let output = if mode == "validate" {
        let baseline = run::<true, true>(&fixture, strategy, 0)?;
        let local = run::<true, true>(&fixture, strategy, 1)?;
        let reverse = run::<true, true>(&fixture, strategy, 2)?;
        assert_eq!(baseline.0, local.0);
        assert_eq!(baseline.0, reverse.0);
        assert_eq!(baseline.1, if case == "reject" { 0 } else { 32 });
        json!({"case":case,"strategy":format!("{strategy:?}"),"nodes":fixture.graph.node_count(),"edges":fixture.graph.edge_count(),"paths":baseline.1,"ordered_paths_and_state_match":true,"evaluated_edges":[baseline.2,local.2,reverse.2],"transition_calls":[baseline.3,local.3,reverse.3]})
    } else if mode == "rss" {
        let result = run::<true, false>(&fixture, strategy, arm)?;
        json!({"arm":arm,"paths":result.1,"evaluated_edges":result.2,"transition_calls":result.3})
    } else {
        let warm = Instant::now();
        while warm.elapsed().as_secs_f64() < 0.5 {
            black_box(run::<false, false>(&fixture, strategy, arm)?);
        }
        let start = Instant::now();
        let mut count = 0;
        while start.elapsed().as_secs_f64() < 2.0 {
            black_box(run::<false, false>(&fixture, strategy, arm)?);
            count += 1;
        }
        let elapsed = start.elapsed().as_secs_f64();
        json!({"arm":arm,"window_seconds":elapsed,"searches":count,"seconds_per_search":elapsed / count as f64})
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
