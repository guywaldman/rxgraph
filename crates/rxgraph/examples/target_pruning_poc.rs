//! Compares target reachability pruning and deferred state replay on a branching graph.
use std::{
    hint::black_box,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

use anyhow::Result;
use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use rxgraph::{
    EdgeCtx, Graph, GraphId, GraphPath, Kernel, NodeId, RunOptions, Transition, TraversalStrategy,
};
use serde_json::json;

const WIDTH: u64 = 4;
const DEPTH: u32 = 10;
const WORK: usize = 512;

struct Fixture {
    graph: Graph,
    target: u64,
    gate: bool,
}

impl Fixture {
    fn new(gate: bool) -> Result<Self> {
        let leaves = WIDTH.pow(DEPTH);
        let first_leaf = (leaves - 1) / (WIDTH - 1);
        let target = first_leaf + leaves;
        let mut src = Vec::with_capacity((target + if gate { leaves } else { 8 }) as usize);
        let mut dest = Vec::with_capacity(src.capacity());
        for child in 1..target {
            src.push((child - 1) / WIDTH);
            dest.push(child);
        }
        for i in 0..leaves {
            if gate || i % (leaves / 8) == 0 {
                src.push(first_leaf + i);
                dest.push(target);
            }
        }
        let column = |values| Arc::new(UInt64Array::from(values)) as ArrayRef;
        let nodes = RecordBatch::try_from_iter([("id", column((0..=target).collect::<Vec<_>>()))])?;
        let edges = RecordBatch::try_from_iter([
            ("id", column((0..src.len() as u64).collect::<Vec<_>>())),
            ("src", column(src)),
            ("dest", column(dest)),
        ])?;
        Ok(Self {
            graph: Graph::new(nodes, edges)?,
            target,
            gate,
        })
    }

    fn reachable(&self) -> Result<Vec<bool>> {
        let mut keep = vec![false; self.graph.node_count()];
        keep[self.target as usize] = true;
        for id in self
            .graph
            .ancestors_many_u64(&[self.target])?
            .expect("u64 graph")
        {
            // The fixture's external IDs and row positions are identical.
            keep[id as usize] = true;
        }
        Ok(keep)
    }
}

#[inline(never)]
fn heavy(mut state: u64) -> u64 {
    for i in 0..WORK {
        state = state
            .rotate_left(9)
            .wrapping_add(i as u64 + 19)
            .wrapping_mul(0x9e37_79b1_85eb_ca87);
    }
    state
}

fn apply(state: u64, edge: u32, dest: u32, target: u32, gate: bool) -> Transition<u64> {
    let folded = heavy(state);
    if gate && dest != target && (u64::from(dest) - 1) % WIDTH != folded % WIDTH {
        return Transition::Reject;
    }
    let next = folded.wrapping_add(u64::from(edge));
    if dest == target {
        Transition::Complete(next)
    } else {
        Transition::Continue(next)
    }
}

struct HeavyKernel<'a, const COUNT: bool> {
    data: &'a Fixture,
    reachable: Option<&'a [bool]>,
    calls: &'a AtomicUsize,
}
impl<const COUNT: bool> Kernel for HeavyKernel<'_, COUNT> {
    type State = u64;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<u64> {
        Ok(7)
    }
    fn transition(&self, cx: &EdgeCtx<'_, u64>) -> Result<Transition<u64>> {
        if self.reachable.is_some_and(|keep| !keep[cx.dest() as usize]) {
            return Ok(Transition::Reject);
        }
        if COUNT {
            self.calls.fetch_add(1, Ordering::Relaxed);
        }
        Ok(apply(
            *cx.state(),
            cx.edge(),
            cx.dest(),
            self.data.target as u32,
            self.data.gate,
        ))
    }
}

struct TopologyKernel {
    target: NodeId,
}
impl Kernel for TopologyKernel {
    type State = ();
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<()> {
        Ok(())
    }
    fn transition(&self, cx: &EdgeCtx<'_, ()>) -> Result<Transition<()>> {
        Ok(if cx.dest() == self.target {
            Transition::Complete(())
        } else {
            Transition::Continue(())
        })
    }
}

type PathRow = (Vec<u64>, Vec<u64>, u64);
fn ids(values: Vec<GraphId<'_>>) -> Vec<u64> {
    values
        .into_iter()
        .map(|id| match id {
            GraphId::U64(id) => id,
            _ => unreachable!(),
        })
        .collect()
}
fn row(path: GraphPath<'_, u64>) -> PathRow {
    (ids(path.nodes), ids(path.edges), path.state)
}

struct Outcome {
    paths: Vec<PathRow>,
    evaluated: usize,
    heavy_calls: usize,
    candidates: usize,
}

fn run<const COUNT: bool>(
    fixture: &Fixture,
    strategy: TraversalStrategy,
    arm: usize,
) -> Result<Outcome> {
    let run = RunOptions {
        start_nodes: vec![0_u64.into()],
        strategy,
        parallel: false,
        ..Default::default()
    };
    let calls = AtomicUsize::new(0);
    let mut rows = Vec::new();
    let mut candidates = 0;
    let evaluated;
    if arm <= 1 {
        let keep = if arm == 1 {
            Some(fixture.reachable()?)
        } else {
            None
        };
        let kernel = HeavyKernel::<COUNT> {
            data: fixture,
            reachable: keep.as_deref(),
            calls: &calls,
        };
        let mut stream = fixture.graph.search_batches_with(kernel, run, 256)?;
        for paths in stream.by_ref() {
            rows.extend(paths?.into_iter().map(row));
        }
        evaluated = stream.stats().evaluated_edges;
        candidates = rows.len();
    } else {
        let mut stream = fixture.graph.search_batches_with(
            TopologyKernel {
                target: fixture.target as u32,
            },
            run,
            256,
        )?;
        for batch in stream.by_ref() {
            for path in batch? {
                candidates += 1;
                let nodes = ids(path.nodes);
                let edges = ids(path.edges);
                let mut state = 7;
                let mut accepted = true;
                for (&edge, &dest) in edges.iter().zip(nodes.iter().skip(1)) {
                    if COUNT {
                        calls.fetch_add(1, Ordering::Relaxed);
                    }
                    match apply(
                        state,
                        edge as u32,
                        dest as u32,
                        fixture.target as u32,
                        fixture.gate,
                    ) {
                        Transition::Reject => {
                            accepted = false;
                            break;
                        }
                        Transition::Continue(next) | Transition::Complete(next) => state = next,
                    }
                }
                if accepted {
                    rows.push((nodes, edges, state));
                }
            }
        }
        evaluated = stream.stats().evaluated_edges;
    }
    rows.sort();
    Ok(Outcome {
        paths: rows,
        evaluated,
        heavy_calls: calls.load(Ordering::Relaxed),
        candidates,
    })
}

fn main() -> Result<()> {
    anyhow::ensure!(!cfg!(debug_assertions), "run with --release");
    let gate = std::env::var("RX_TARGET_CASE").as_deref() == Ok("state-rejection");
    let strategy = if std::env::var("RX_TARGET_STRATEGY").as_deref() == Ok("bfs") {
        TraversalStrategy::BreadthFirst
    } else {
        TraversalStrategy::DepthFirst
    };
    let mode = std::env::var("RX_TARGET_MODE").unwrap_or_else(|_| "timing".into());
    let arm: usize = std::env::var("RX_TARGET_ARM")
        .unwrap_or_else(|_| "0".into())
        .parse()?;
    anyhow::ensure!(arm <= 2, "invalid arm");
    let fixture = Fixture::new(gate)?;
    let output = if mode == "validate" {
        let init = Instant::now();
        let keep = fixture.reachable()?;
        let cold_setup = init.elapsed().as_secs_f64();
        let warm = Instant::now();
        let warm_keep = fixture.reachable()?;
        let warm_setup = warm.elapsed().as_secs_f64();
        assert_eq!(keep, warm_keep);
        drop((keep, warm_keep));
        let baseline = run::<true>(&fixture, strategy, 0)?;
        let reverse = run::<true>(&fixture, strategy, 1)?;
        let lazy = run::<true>(&fixture, strategy, 2)?;
        assert_eq!(baseline.paths, reverse.paths);
        assert_eq!(baseline.paths, lazy.paths);
        json!({"case":if gate {"state-rejection"} else {"sparse-target"}, "strategy":format!("{strategy:?}"), "nodes":fixture.graph.node_count(), "edges":fixture.graph.edge_count(), "paths":baseline.paths.len(), "candidate_paths": [baseline.candidates, reverse.candidates, lazy.candidates], "evaluated_edges": [baseline.evaluated, reverse.evaluated, lazy.evaluated], "heavy_calls": [baseline.heavy_calls, reverse.heavy_calls, lazy.heavy_calls], "cold_reverse_setup_seconds":cold_setup, "warm_reverse_setup_seconds":warm_setup, "exact_paths_and_states_match":true})
    } else if mode == "rss" {
        let result = run::<true>(&fixture, strategy, arm)?;
        json!({"arm":arm, "paths":result.paths.len(), "candidate_paths":result.candidates, "evaluated_edges":result.evaluated, "heavy_calls":result.heavy_calls})
    } else {
        if arm == 1 {
            black_box(fixture.reachable()?);
        }
        let warm_start = Instant::now();
        while warm_start.elapsed().as_secs_f64() < 0.5 {
            black_box(run::<false>(&fixture, strategy, arm)?);
        }
        let start = Instant::now();
        let mut count = 0;
        while start.elapsed().as_secs_f64() < 2.0 {
            black_box(run::<false>(&fixture, strategy, arm)?);
            count += 1;
        }
        let elapsed = start.elapsed().as_secs_f64();
        json!({"arm":arm, "window_seconds":elapsed, "searches":count, "seconds_per_search":elapsed / count as f64})
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
