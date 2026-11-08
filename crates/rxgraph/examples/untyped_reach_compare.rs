//! Compare reusable group reachability against coarse per-search reverse pruning.
use anyhow::Result;
use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use rxgraph::{
    EdgeCtx, Evaluation, Graph, GraphId, Kernel, NodeId, ReachIndex, ReachSpec, RunOptions,
    TargetCtx, TargetOptions, Transition, TraversalStrategy,
};
use serde_json::json;
use std::{
    borrow::Cow,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    hint::black_box,
    sync::Arc,
    time::Instant,
};

const ROUNDS: u64 = 64;
#[derive(Clone, Copy)]
struct State {
    group: u32,
    score: u64,
}
struct Fixture {
    graph: Graph,
    targets: Vec<NodeId>,
    hub: NodeId,
}
impl Fixture {
    fn new(groups: usize, corridor: usize) -> Result<Self> {
        let hub = (groups + 1) as NodeId;
        let base = groups + 2;
        let count = base + groups * corridor;
        let mut src = Vec::with_capacity(groups * (corridor + 2));
        let mut dst = Vec::with_capacity(src.capacity());
        for group in 0..groups {
            src.push(0u64);
            dst.push((group + 1) as u64);
        }
        for group in 0..groups {
            src.push((group + 1) as u64);
            dst.push(hub as u64);
        }
        for group in 0..groups {
            let first = base + group * corridor;
            src.push(hub as u64);
            dst.push(first as u64);
            for node in first..first + corridor - 1 {
                src.push(node as u64);
                dst.push((node + 1) as u64);
            }
        }
        let targets = (0..groups)
            .map(|g| (base + (g + 1) * corridor - 1) as NodeId)
            .collect();
        let a = |v| Arc::new(UInt64Array::from(v)) as ArrayRef;
        let nodes = RecordBatch::try_from_iter([("id", a((0..count as u64).collect::<Vec<_>>()))])?;
        let edges = RecordBatch::try_from_iter([
            ("id", a((0..src.len() as u64).collect::<Vec<_>>())),
            ("src", a(src)),
            ("dest", a(dst)),
        ])?;
        Ok(Self {
            graph: Graph::new(nodes, edges)?,
            targets,
            hub,
        })
    }
    fn index(&self) -> Result<ReachIndex> {
        self.graph.reach_index(ReachSpec {
            sets: self
                .targets
                .iter()
                .enumerate()
                .map(|(g, &n)| (g as u32, vec![n]))
                .collect(),
            skip_edge: None,
        })
    }
}
#[inline(never)]
fn mix(mut x: u64, edge: u32) -> u64 {
    for i in 0..ROUNDS {
        x = x
            .rotate_left(9)
            .wrapping_add(i + edge as u64)
            .wrapping_mul(0x9e37_79b1_85eb_ca87);
    }
    x
}
struct BenchKernel<'a> {
    f: &'a Fixture,
    index: Option<&'a ReachIndex>,
}
impl Kernel for BenchKernel<'_> {
    type State = State;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<State> {
        Ok(State {
            group: u32::MAX,
            score: 7,
        })
    }
    fn transition(&self, cx: &EdgeCtx<'_, State>) -> Result<Transition<State>> {
        let mut state = *cx.state();
        if cx.src() == 0 {
            state.group = cx.dest() - 1;
        }
        state.score = mix(state.score, cx.edge());
        if self.f.targets.binary_search(&cx.dest()).is_ok() {
            Ok(if self.f.targets[state.group as usize] == cx.dest() {
                Transition::Complete(state)
            } else {
                Transition::Reject
            })
        } else {
            Ok(Transition::Continue(state))
        }
    }
    fn possible_targets<'a>(
        &'a self,
        cx: &TargetCtx<'_, State>,
    ) -> Result<Option<Cow<'a, [NodeId]>>> {
        match cx {
            TargetCtx::Start { .. } => Ok(Some(Cow::Borrowed(&self.f.targets))),
            TargetCtx::Edge { edge, parent, .. } => {
                let group = if edge.src() == 0 {
                    Some((edge.dest() - 1) as usize)
                } else {
                    parent
                        .filter(|s| s.group != u32::MAX)
                        .map(|s| s.group as usize)
                };
                let Some(group) = group else { return Ok(None) };
                if let Some(index) = self.index {
                    if !edge.reaches(index, group as u32, edge.dest())? {
                        return Ok(Some(Cow::Borrowed(&[])));
                    }
                    Ok(Some(Cow::Borrowed(
                        index.targets(edge.graph(), group as u32)?,
                    )))
                } else {
                    Ok(Some(Cow::Borrowed(std::slice::from_ref(
                        &self.f.targets[group],
                    ))))
                }
            }
        }
    }
    fn targets_depend_on_child_state(&self) -> bool {
        false
    }
}
fn search(
    f: &Fixture,
    index: Option<&ReachIndex>,
    strategy: TraversalStrategy,
) -> Result<(usize, u64, u64)> {
    let run = RunOptions {
        start_nodes: vec![0u64.into()],
        strategy,
        parallel: false,
        ..Default::default()
    };
    let options = TargetOptions {
        reverse: index.is_none(),
        release_expanded: false,
        evaluation: Evaluation::Incremental,
    };
    let result = f
        .graph
        .search_paths_with(BenchKernel { f, index }, run.with_optimizations(options))?;
    let count = result.paths.len();
    let digest = result.paths.into_iter().fold(0u64, |acc, p| {
        let mut hash = DefaultHasher::new();
        for id in p.nodes.into_iter().chain(p.edges) {
            match id {
                GraphId::U64(id) => id.hash(&mut hash),
                _ => unreachable!(),
            }
        }
        p.state.score.hash(&mut hash);
        p.state.group.hash(&mut hash);
        acc.wrapping_add(hash.finish())
    });
    Ok((count, digest, result.stats.evaluated_edges as u64))
}
fn main() -> Result<()> {
    anyhow::ensure!(!cfg!(debug_assertions), "use --release");
    let groups: usize = std::env::var("REACH_GROUPS")
        .unwrap_or("16".into())
        .parse()?;
    let corridor: usize = std::env::var("REACH_CORRIDOR")
        .unwrap_or("4096".into())
        .parse()?;
    let indexed = std::env::var("REACH_ARM").unwrap_or("indexed".into()) == "indexed";
    let strategy = if std::env::var("REACH_STRATEGY").as_deref() == Ok("bfs") {
        TraversalStrategy::BreadthFirst
    } else {
        TraversalStrategy::DepthFirst
    };
    let mode = std::env::var("REACH_MODE").unwrap_or("timing".into());
    let now = Instant::now();
    let f = Fixture::new(groups, corridor)?;
    let graph_build_seconds = now.elapsed().as_secs_f64();
    let now = Instant::now();
    let index = if indexed { Some(f.index()?) } else { None };
    let index_build_seconds = now.elapsed().as_secs_f64();
    let now = Instant::now();
    let (count, digest, edges) = if mode == "build" {
        (0, 0, 0)
    } else {
        let result = search(&f, index.as_ref(), strategy)?;
        anyhow::ensure!(
            result.0 == groups,
            "expected {groups} paths, got {}",
            result.0
        );
        result
    };
    let first_search_seconds = now.elapsed().as_secs_f64();
    let mut seconds = 0.;
    let mut searches = 1u64;
    if mode == "timing" {
        let start = Instant::now();
        searches = 0;
        while start.elapsed().as_secs_f64() < 2.0 {
            let result = search(&f, index.as_ref(), strategy)?;
            anyhow::ensure!(
                result == (count, digest, edges),
                "different results across repeated searches"
            );
            black_box(result);
            searches += 1;
        }
        seconds = start.elapsed().as_secs_f64();
    }
    println!(
        "{}",
        json!({"groups": groups, "corridor":corridor, "node_count":f.graph.node_count(), "edge_count":f.graph.edge_count(), "hub":f.hub, "rounds":ROUNDS, "mode":mode, "indexed":indexed, "graph_build_seconds":graph_build_seconds, "index_build_seconds":index_build_seconds, "first_search_seconds":first_search_seconds, "paths":count, "digest":digest, "evaluated_edges":edges, "searches":searches, "window_seconds":seconds, "seconds_per_search": if mode=="timing" { seconds/searches as f64 } else {0.}})
    );
    Ok(())
}
