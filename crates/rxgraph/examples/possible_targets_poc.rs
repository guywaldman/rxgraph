//! Experimental per-prefix possible ending sets, using existing native search.
use anyhow::Result;
use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use rxgraph::{
    EdgeCtx, Graph, GraphId, GraphPath, Kernel, NodeId, RunOptions, SearchStats, Transition,
    TraversalStrategy,
};
use serde_json::json;
use std::{
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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Exact {
    score: u64,
    mask: u8,
    depth: u8,
}
#[derive(Clone, Copy)]
struct Light {
    mask: u8,
    depth: u8,
}
struct Fixture {
    graph: Graph,
    target_base: u64,
    first_leaf: u64,
    gate: bool,
}
impl Fixture {
    fn new(gate: bool) -> Result<Self> {
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
            let target = if group < 8 { group } else { 0 };
            src.push(first_leaf + leaf);
            dest.push(base + target);
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
            first_leaf,
            gate,
        })
    }
    fn coarse(&self) -> Result<Vec<bool>> {
        let mut keep = vec![false; self.graph.node_count()];
        let targets = (self.target_base..self.target_base + 8).collect::<Vec<_>>();
        for &t in &targets {
            keep[t as usize] = true;
        }
        for n in self.graph.ancestors_many_u64(&targets)?.expect("u64 graph") {
            keep[n as usize] = true;
        }
        Ok(keep)
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
// Returns a conservative set of possible END nodes after this edge.
// The 8-bit representation is fixture-specific, not a general API limit.
fn possible_targets(f: &Fixture, depth: u8, mask: u8, dest: NodeId) -> Option<u8> {
    if f.gate {
        return None;
    }
    let dest = dest as u64;
    if dest >= f.target_base {
        return Some(mask & (1 << (dest - f.target_base)));
    }
    let branch = (dest - 1) % WIDTH;
    let allowed = match depth {
        0 => {
            if branch < 2 {
                if branch == 0 { 0x0f } else { 0xf0 }
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
    };
    Some(mask & allowed)
}
fn apply(f: &Fixture, s: Exact, edge: u32, dest: NodeId) -> Transition<Exact> {
    let folded = heavy(s.score);
    let terminal = dest as u64 >= f.target_base;
    if f.gate && !terminal && ((dest as u64 - 1) % WIDTH != folded % WIDTH) {
        return Transition::Reject;
    }
    let mask = possible_targets(f, s.depth, s.mask, dest).unwrap_or(s.mask);
    let next = Exact {
        score: folded.wrapping_add(edge as u64),
        mask,
        depth: s.depth + 1,
    };
    if terminal {
        let bit = 1u8 << (dest as u64 - f.target_base);
        if mask & bit != 0 {
            Transition::Complete(next)
        } else {
            Transition::Reject
        }
    } else {
        Transition::Continue(next)
    }
}
struct Eager<'a, const COUNT: bool> {
    f: &'a Fixture,
    keep: Option<&'a [bool]>,
    refine: bool,
    calls: &'a AtomicUsize,
}
impl<const COUNT: bool> Kernel for Eager<'_, COUNT> {
    type State = Exact;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<Exact> {
        Ok(Exact {
            score: 7,
            mask: ALL,
            depth: 0,
        })
    }
    fn transition(&self, cx: &EdgeCtx<'_, Exact>) -> Result<Transition<Exact>> {
        if self.keep.is_some_and(|k| !k[cx.dest() as usize]) {
            return Ok(Transition::Reject);
        }
        if self.refine
            && possible_targets(self.f, cx.state().depth, cx.state().mask, cx.dest()) == Some(0)
        {
            return Ok(Transition::Reject);
        }
        if COUNT {
            self.calls.fetch_add(1, Ordering::Relaxed);
        }
        Ok(apply(self.f, *cx.state(), cx.edge(), cx.dest()))
    }
}
struct Lazy<'a> {
    f: &'a Fixture,
    keep: &'a [bool],
}
impl Kernel for Lazy<'_> {
    type State = Light;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<Light> {
        Ok(Light {
            mask: ALL,
            depth: 0,
        })
    }
    fn transition(&self, cx: &EdgeCtx<'_, Light>) -> Result<Transition<Light>> {
        if !self.keep[cx.dest() as usize] {
            return Ok(Transition::Reject);
        }
        let mask = possible_targets(self.f, cx.state().depth, cx.state().mask, cx.dest())
            .unwrap_or(cx.state().mask);
        if mask == 0 {
            return Ok(Transition::Reject);
        }
        let next = Light {
            mask,
            depth: cx.state().depth + 1,
        };
        // This fixture's end nodes have no outgoing edges. A generic implementation
        // must keep exploring a candidate end when the exact kernel says Continue.
        Ok(if cx.dest() as u64 >= self.f.target_base {
            Transition::Complete(next)
        } else {
            Transition::Continue(next)
        })
    }
}
type Row = (Vec<u64>, Vec<u64>, Exact);
fn ids(v: Vec<GraphId<'_>>) -> Vec<u64> {
    v.into_iter()
        .map(|id| match id {
            GraphId::U64(v) => v,
            _ => unreachable!(),
        })
        .collect()
}
fn row(p: GraphPath<'_, Exact>) -> Row {
    (ids(p.nodes), ids(p.edges), p.state)
}
struct Outcome {
    rows: Vec<Row>,
    stats: SearchStats,
    heavy: usize,
    candidates: usize,
}
fn run<const COUNT: bool>(f: &Fixture, strategy: TraversalStrategy, arm: usize) -> Result<Outcome> {
    let cfg = RunOptions {
        start_nodes: vec![0u64.into()],
        strategy,
        parallel: false,
        ..Default::default()
    };
    let keep = if arm > 0 { Some(f.coarse()?) } else { None };
    let calls = AtomicUsize::new(0);
    let mut rows = Vec::new();
    let mut candidates = 0;
    let stats;
    if arm < 3 {
        let mut stream = f.graph.search_batches_with(
            Eager::<COUNT> {
                f,
                keep: keep.as_deref(),
                refine: arm == 2,
                calls: &calls,
            },
            cfg,
            256,
        )?;
        for batch in stream.by_ref() {
            rows.extend(batch?.into_iter().map(row));
        }
        stats = stream.stats();
        candidates = rows.len();
    } else {
        let mut stream = f.graph.search_batches_with(
            Lazy {
                f,
                keep: keep.as_deref().unwrap(),
            },
            cfg,
            256,
        )?;
        for batch in stream.by_ref() {
            for p in batch? {
                candidates += 1;
                let nodes = ids(p.nodes);
                let edges = ids(p.edges);
                let mut s = Exact {
                    score: 7,
                    mask: ALL,
                    depth: 0,
                };
                let mut valid = false;
                for (&edge, &dest) in edges.iter().zip(nodes.iter().skip(1)) {
                    if COUNT {
                        calls.fetch_add(1, Ordering::Relaxed);
                    }
                    match apply(f, s, edge as u32, dest as u32) {
                        Transition::Reject => break,
                        Transition::Continue(next) => s = next,
                        Transition::Complete(next) => {
                            s = next;
                            valid = true;
                            break;
                        }
                    }
                }
                if valid {
                    rows.push((nodes, edges, s));
                }
            }
        }
        stats = stream.stats();
    }
    Ok(Outcome {
        rows,
        stats,
        heavy: calls.load(Ordering::Relaxed),
        candidates,
    })
}
fn main() -> Result<()> {
    anyhow::ensure!(!cfg!(debug_assertions), "use --release");
    let case = std::env::var("RX_PT_CASE").unwrap_or_else(|_| "refinable".into());
    let gate = case == "state_rejection";
    anyhow::ensure!(gate || case == "refinable", "invalid case");
    let strategy = if std::env::var("RX_PT_STRATEGY").as_deref() == Ok("bfs") {
        TraversalStrategy::BreadthFirst
    } else {
        TraversalStrategy::DepthFirst
    };
    let mode = std::env::var("RX_PT_MODE").unwrap_or_else(|_| "timing".into());
    let arm: usize = std::env::var("RX_PT_ARM")
        .unwrap_or_else(|_| "0".into())
        .parse()?;
    anyhow::ensure!(arm < 4, "invalid arm");
    let f = Fixture::new(gate)?;
    let output = if mode == "validate" {
        let start = Instant::now();
        let cold = f.coarse()?;
        let cold_s = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let warm = f.coarse()?;
        let warm_s = start.elapsed().as_secs_f64();
        assert_eq!(cold, warm);
        drop((cold, warm));
        let runs = (0..4)
            .map(|a| run::<true>(&f, strategy, a))
            .collect::<Result<Vec<_>>>()?;
        for r in &runs[1..] {
            assert_eq!(
                r.rows, runs[0].rows,
                "serial ordering, nodes, edges and state must match"
            );
        }
        assert_eq!(runs[0].rows.len(), if gate { 1 } else { 8 });
        json!({"case":case,"strategy":format!("{strategy:?}"),"nodes":f.graph.node_count(),"edges":f.graph.edge_count(),"targets":8,"target_mask_bits":8,"paths":runs[0].rows.len(),"exact_match":true,"ordered_paths_match":true,"evaluated_edges":runs.iter().map(|r|r.stats.evaluated_edges).collect::<Vec<_>>(),"heavy_calls":runs.iter().map(|r|r.heavy).collect::<Vec<_>>(),"candidate_paths":runs.iter().map(|r|r.candidates).collect::<Vec<_>>(),"cold_reverse_setup_seconds":cold_s,"warm_reverse_setup_seconds":warm_s,"refinement_phase":"before transition when computable from topology; otherwise unchanged","lazy_replay":"synthetic shared apply helper, not generic Kernel::transition","reachability":"coarse reverse mask rebuilt per search; incoming CSR retained"})
    } else if mode == "rss" {
        let r = run::<true>(&f, strategy, arm)?;
        json!({"arm":arm,"paths":r.rows.len(),"evaluated_edges":r.stats.evaluated_edges,"heavy_calls":r.heavy,"candidate_paths":r.candidates})
    } else {
        if arm > 0 {
            black_box(f.coarse()?);
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
