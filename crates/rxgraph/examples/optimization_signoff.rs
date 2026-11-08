//! End-to-end measurements through native traversal APIs.
use anyhow::Result;
use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use rxgraph::{
    EdgeCtx, Evaluation, Graph, GraphId, GraphPath, Kernel, MemoizableKernel, NodeId, RunOptions,
    SearchStats, TargetCtx, TargetOptions, Transition, TraversalStrategy,
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

const WIDTH: u64 = 4;
const DEPTH: u32 = 10;
const ALL: u8 = 255;
const WORK: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct State {
    score: u64,
    mask: u8,
    depth: u8,
}
struct Fixture {
    graph: Graph,
    target_base: u64,
    gate: bool,
    unknown: bool,
}
impl Fixture {
    fn new(gate: bool, unknown: bool) -> Result<Self> {
        let leaves = WIDTH.pow(DEPTH);
        let first_leaf = (leaves - 1) / (WIDTH - 1);
        let base = first_leaf + leaves;
        let mut src = Vec::with_capacity((base - 1 + leaves) as usize);
        let mut dest = Vec::with_capacity(src.capacity());
        for child in 1..base {
            src.push((child - 1) / WIDTH);
            dest.push(child);
        }
        for leaf in 0..leaves {
            let group = leaf / WIDTH.pow(DEPTH - 2);
            src.push(first_leaf + leaf);
            dest.push(base + if group < 8 { group } else { 0 });
        }
        let col = |v| Arc::new(UInt64Array::from(v)) as ArrayRef;
        let nodes = RecordBatch::try_from_iter([("id", col((0..base + 8).collect::<Vec<_>>()))])?;
        let edges = RecordBatch::try_from_iter([
            ("id", col((0..src.len() as u64).collect::<Vec<_>>())),
            ("src", col(src)),
            ("dest", col(dest)),
        ])?;
        Ok(Self {
            graph: Graph::new(nodes, edges)?,
            target_base: base,
            gate,
            unknown,
        })
    }
    fn targets(&self, mask: u8) -> Vec<NodeId> {
        (0..8)
            .filter(|i| mask & (1 << i) != 0)
            .map(|i| (self.target_base + i) as NodeId)
            .collect()
    }
}
#[inline(never)]
fn heavy(mut x: u64) -> u64 {
    for i in 0..WORK {
        x = x
            .rotate_left(9)
            .wrapping_add(i as u64 + 19)
            .wrapping_mul(0x9e37_79b1_85eb_ca87);
    }
    x
}
fn node_depth(mut n: u64) -> u8 {
    let mut depth = 0;
    while n > 0 {
        n = (n - 1) / WIDTH;
        depth += 1;
    }
    depth
}
fn possible(f: &Fixture, depth: u8, dest: NodeId) -> Option<u8> {
    if f.gate {
        return None;
    }
    let dest = dest as u64;
    if dest >= f.target_base {
        return Some(1 << (dest - f.target_base));
    }
    let branch = (dest - 1) % WIDTH;
    Some(match depth {
        0 => {
            if branch == 0 {
                0x0f
            } else if branch == 1 {
                0xf0
            } else {
                0
            }
        }
        1 => (0..8)
            .filter(|t| (*t % 4) as u64 == branch)
            .fold(0u8, |m, t| m | (1 << t)),
        _ => {
            if branch == 0 {
                ALL
            } else {
                0
            }
        }
    })
}
struct WorkKernel<'a, const COUNT: bool> {
    data: &'a Fixture,
    calls: &'a AtomicUsize,
}
impl<const COUNT: bool> Kernel for WorkKernel<'_, COUNT> {
    type State = State;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<State> {
        Ok(State {
            score: 7,
            mask: ALL,
            depth: 0,
        })
    }
    fn transition(&self, cx: &EdgeCtx<'_, State>) -> Result<Transition<State>> {
        if COUNT {
            self.calls.fetch_add(1, Ordering::Relaxed);
        }
        let folded = heavy(cx.state().score);
        let dest = cx.dest() as u64;
        let terminal = dest >= self.data.target_base;
        if self.data.gate && !terminal && ((dest - 1) % WIDTH != folded % WIDTH) {
            return Ok(Transition::Reject);
        }
        let mask = possible(self.data, cx.state().depth, cx.dest())
            .map_or(cx.state().mask, |m| m & cx.state().mask);
        let next = State {
            score: folded.wrapping_add(cx.edge() as u64),
            mask,
            depth: cx.state().depth + 1,
        };
        Ok(if terminal {
            if mask & (1 << (dest - self.data.target_base)) != 0 {
                Transition::Complete(next)
            } else {
                Transition::Reject
            }
        } else {
            Transition::Continue(next)
        })
    }
    fn possible_targets<'a>(
        &'a self,
        cx: &TargetCtx<'_, State>,
    ) -> Result<Option<Cow<'a, [NodeId]>>> {
        if self.data.unknown {
            return Ok(None);
        }
        match cx {
            TargetCtx::Start { .. } => Ok(Some(Cow::Owned(self.data.targets(ALL)))),
            TargetCtx::Edge { edge, .. } => {
                Ok(
                    possible(self.data, node_depth(edge.src() as u64), edge.dest())
                        .map(|m| Cow::Owned(self.data.targets(m))),
                )
            }
        }
    }
}
impl<const COUNT: bool> MemoizableKernel for WorkKernel<'_, COUNT> {
    type Key = State;
    fn state_key(&self, state: &State) -> State {
        *state
    }
}
type Row = (Vec<u64>, Vec<u64>, State);
fn ids(v: Vec<GraphId<'_>>) -> Vec<u64> {
    v.into_iter()
        .map(|id| match id {
            GraphId::U64(n) => n,
            _ => unreachable!(),
        })
        .collect()
}
fn row(p: GraphPath<'_, State>) -> Row {
    (ids(p.nodes), ids(p.edges), p.state)
}
struct Outcome {
    rows: Vec<Row>,
    stats: SearchStats,
    calls: usize,
}
fn run<const COUNT: bool>(f: &Fixture, strategy: TraversalStrategy, arm: usize) -> Result<Outcome> {
    let run = RunOptions {
        start_nodes: vec![0u64.into()],
        strategy,
        parallel: false,
        ..Default::default()
    };
    let calls = AtomicUsize::new(0);
    let kernel = WorkKernel::<COUNT> {
        data: f,
        calls: &calls,
    };
    let mut rows = Vec::new();
    macro_rules! drain {
        ($stream:expr) => {{
            let mut stream = $stream?;
            for batch in stream.by_ref() {
                rows.extend(batch?.into_iter().map(row));
            }
            stream.stats()
        }};
    }
    let opts = TargetOptions {
        reverse: matches!(arm, 2 | 4 | 5 | 6),
        release_expanded: false,
        evaluation: if matches!(arm, 3 | 4 | 6) {
            Evaluation::Lazy
        } else {
            Evaluation::Incremental
        },
    };
    let stats = match arm {
        0 => drain!(f.graph.search_batches_with(kernel, run, 256)),
        1..=4 => drain!(
            f.graph
                .search_batches_with(kernel, run.with_optimizations(opts), 256)
        ),
        5 | 6 => {
            drain!(f.graph.search_batches_with(
                kernel.memoized(4096),
                run.with_optimizations(opts),
                256
            ))
        }
        _ => anyhow::bail!("invalid arm"),
    };
    Ok(Outcome {
        rows,
        stats,
        calls: calls.load(Ordering::Relaxed),
    })
}
fn main() -> Result<()> {
    anyhow::ensure!(!cfg!(debug_assertions), "use --release");
    let case = std::env::var("RX_SIGNOFF_CASE").unwrap_or_else(|_| "selective".into());
    let gate = case == "state_rejection" || case == "unknown";
    anyhow::ensure!(gate || case == "selective", "invalid case");
    let strategy = if std::env::var("RX_SIGNOFF_STRATEGY").as_deref() == Ok("bfs") {
        TraversalStrategy::BreadthFirst
    } else {
        TraversalStrategy::DepthFirst
    };
    let mode = std::env::var("RX_SIGNOFF_MODE").unwrap_or_else(|_| "timing".into());
    let arm: usize = std::env::var("RX_SIGNOFF_ARM")
        .unwrap_or_else(|_| "0".into())
        .parse()?;
    let f = Fixture::new(gate, case == "unknown")?;
    let output = if mode == "validate" {
        let cold = Instant::now();
        let _ = f
            .graph
            .ancestors_many_u64(&(f.target_base..f.target_base + 8).collect::<Vec<_>>())?;
        let cold = cold.elapsed().as_secs_f64();
        let runs = (0..7)
            .map(|a| run::<true>(&f, strategy, a))
            .collect::<Result<Vec<_>>>()?;
        for result in &runs[1..] {
            assert_eq!(
                result.rows, runs[0].rows,
                "ordered nodes, edges and states must match"
            );
        }
        assert_eq!(runs[0].rows.len(), if gate { 1 } else { 8 });
        json!({"case":case,"strategy":format!("{strategy:?}"),"nodes":f.graph.node_count(),"edges":f.graph.edge_count(),"targets":8,"paths":runs[0].rows.len(),"exact_match":true,"heavy_calls":runs.iter().map(|r|r.calls).collect::<Vec<_>>(),"evaluated_edges":runs.iter().map(|r|r.stats.evaluated_edges).collect::<Vec<_>>(),"cold_reverse_seconds":cold})
    } else if mode == "rss" {
        let r = run::<true>(&f, strategy, arm)?;
        json!({"arm":arm,"paths":r.rows.len(),"heavy_calls":r.calls,"evaluated_edges":r.stats.evaluated_edges})
    } else {
        if matches!(arm, 2 | 4 | 5 | 6) {
            black_box(
                f.graph
                    .ancestors_many_u64(&(f.target_base..f.target_base + 8).collect::<Vec<_>>())?,
            );
        }
        let warm = Instant::now();
        while warm.elapsed().as_secs_f64() < 0.5 {
            black_box(run::<false>(&f, strategy, arm)?);
        }
        let start = Instant::now();
        let mut n = 0;
        while start.elapsed().as_secs_f64() < 2.0 {
            black_box(run::<false>(&f, strategy, arm)?);
            n += 1;
        }
        let elapsed = start.elapsed().as_secs_f64();
        json!({"arm":arm,"window_seconds":elapsed,"searches":n,"seconds_per_search":elapsed/n as f64})
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
