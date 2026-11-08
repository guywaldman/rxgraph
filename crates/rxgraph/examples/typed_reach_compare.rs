//! Typed-kernel comparison; compile the previous revision without `--cfg reach_new`.
#![allow(unexpected_cfgs)]
use anyhow::Result;
use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use rxgraph::{
    Graph, RunOptions, Transition, TraversalStrategy, TypedKernel, TypedPayloadCache,
    boxed_typed_run, traversal::native,
};
#[cfg(reach_new)]
use rxgraph::{NodeId, ReachIndex, ReachSpec, TargetOptions};
use serde_json::json;
#[cfg(reach_new)]
use std::borrow::Cow;
use std::{
    hint::black_box,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

#[derive(Clone, serde::Serialize)]
struct State {
    group: usize,
    value: u64,
}

#[derive(Clone)]
struct Machine {
    groups: usize,
    len: usize,
    count: bool,
    calls: Arc<AtomicUsize>,
    #[cfg(reach_new)]
    all: Arc<Vec<NodeId>>,
    #[cfg(reach_new)]
    index: Option<Arc<ReachIndex>>,
}

impl TypedKernel for Machine {
    type Node = ();
    type Edge = ();
    type State = State;
    fn initial_state(&self, _: &native::StartCtx<'_, (), ()>) -> Result<State> {
        Ok(State {
            group: self.groups,
            value: 7,
        })
    }
    fn transition(&self, cx: &native::EdgeCtx<'_, '_, (), (), State>) -> Result<Transition<State>> {
        if self.count {
            self.calls.fetch_add(1, Ordering::Relaxed);
        }
        let mut state = cx.state().clone();
        if cx.src_id() == 0 {
            state.group = cx.dest_id() as usize - 1;
        }
        for i in 0..64 {
            state.value = black_box(
                state
                    .value
                    .rotate_left(9)
                    .wrapping_add(i + u64::from(cx.edge_id()))
                    .wrapping_mul(0x9e37_79b1_85eb_ca87),
            );
        }
        let dest = cx.dest_id() as usize;
        let first = self.groups + 2;
        if dest >= first && (dest - first + 1) % self.len == 0 {
            return Ok(if (dest - first) / self.len == state.group {
                Transition::Complete(state)
            } else {
                Transition::Reject
            });
        }
        Ok(Transition::Continue(state))
    }

    #[cfg(reach_new)]
    fn possible_targets<'a>(
        &'a self,
        cx: &native::TargetCtx<'a, (), (), State>,
    ) -> Result<Option<Cow<'a, [NodeId]>>> {
        use native::TargetCtx;
        match cx {
            TargetCtx::Start { .. } => Ok(Some(Cow::Borrowed(self.all.as_slice()))),
            TargetCtx::Edge { edge, parent, .. } => {
                let group = if edge.src_id() == 0 {
                    edge.dest_id() as usize - 1
                } else if let Some(parent) = parent {
                    parent.group
                } else {
                    self.groups
                };
                if group >= self.groups {
                    return Ok(None);
                }
                if let Some(index) = &self.index {
                    if !edge.reaches(index, group as u32, edge.dest_id())? {
                        return Ok(Some(Cow::Borrowed(&[])));
                    }
                    Ok(Some(Cow::Borrowed(edge.targets(index, group as u32)?)))
                } else {
                    Ok(Some(Cow::Borrowed(&self.all[group..group + 1])))
                }
            }
        }
    }
    #[cfg(reach_new)]
    fn targets_depend_on_child_state(&self) -> bool {
        false
    }
}

struct Fixture {
    graph: Graph,
    #[cfg(reach_new)]
    all: Arc<Vec<NodeId>>,
}
impl Fixture {
    fn new(groups: usize, len: usize) -> Result<Self> {
        let hub = groups + 1;
        let first = groups + 2;
        let node_count = first + groups * len;
        let mut src = Vec::new();
        let mut dest = Vec::new();
        for group in 0..groups {
            src.push(0u64);
            dest.push((group + 1) as u64);
        }
        for group in 0..groups {
            src.push((group + 1) as u64);
            dest.push(hub as u64);
        }
        for group in 0..groups {
            let base = first + group * len;
            src.push(hub as u64);
            dest.push(base as u64);
            for offset in 1..len {
                src.push((base + offset - 1) as u64);
                dest.push((base + offset) as u64);
            }
        }
        let col = |v| Arc::new(UInt64Array::from(v)) as ArrayRef;
        let nodes =
            RecordBatch::try_from_iter([("id", col((0..node_count as u64).collect::<Vec<_>>()))])?;
        let edges = RecordBatch::try_from_iter([
            ("id", col((0..src.len() as u64).collect::<Vec<_>>())),
            ("src", col(src)),
            ("dest", col(dest)),
        ])?;
        #[cfg(reach_new)]
        let all = Arc::new(
            (0..groups)
                .map(|g| (first + (g + 1) * len - 1) as NodeId)
                .collect(),
        );
        Ok(Self {
            graph: Graph::new(nodes, edges)?,
            #[cfg(reach_new)]
            all,
        })
    }
    #[cfg(reach_new)]
    fn build_index(&self) -> Result<Arc<ReachIndex>> {
        let sets = self
            .all
            .iter()
            .enumerate()
            .map(|(g, &id)| (g as u32, vec![id]))
            .collect();
        Ok(Arc::new(self.graph.reach_index(ReachSpec {
            sets,
            skip_edge: None,
        })?))
    }
}

fn main() -> Result<()> {
    let groups: usize = std::env::var("REACH_GROUPS")
        .unwrap_or_else(|_| "16".into())
        .parse()?;
    let len: usize = std::env::var("REACH_LEN")
        .unwrap_or_else(|_| "4096".into())
        .parse()?;
    let mode = std::env::var("REACH_MODE").unwrap_or_else(|_| "verify".into());
    let arm = std::env::var("REACH_ARM").unwrap_or_else(|_| "baseline".into());
    let strategy = match std::env::var("REACH_STRATEGY").as_deref() {
        Ok("bfs") => TraversalStrategy::BreadthFirst,
        _ => TraversalStrategy::DepthFirst,
    };
    let fixture = Fixture::new(groups, len)?;
    let prep = Instant::now();
    #[cfg(reach_new)]
    let index = if arm == "indexed" {
        Some(fixture.build_index()?)
    } else {
        None
    };
    #[cfg(not(reach_new))]
    if arm != "baseline" {
        anyhow::bail!("the previous typed kernel has no optimized search");
    }
    let build_ms = prep.elapsed().as_secs_f64() * 1000.0;
    let counter = Arc::new(AtomicUsize::new(0));
    let cache = TypedPayloadCache::default();
    let run = RunOptions {
        start_nodes: vec![0u64.into()],
        strategy,
        parallel: false,
        ..Default::default()
    };
    let one = |audit: bool| -> Result<(usize, usize, u64)> {
        let machine = Machine {
            groups,
            len,
            count: audit,
            calls: Arc::clone(&counter),
            #[cfg(reach_new)]
            all: Arc::clone(&fixture.all),
            #[cfg(reach_new)]
            index: if arm == "cold" {
                Some(fixture.build_index()?)
            } else {
                index.clone()
            },
        };
        let runner = boxed_typed_run(machine);
        #[cfg(reach_new)]
        let result = if arm == "baseline" {
            runner.run_eager_cached(&fixture.graph, &cache, run.clone())?
        } else {
            runner.run_eager_cached_optimized(
                &fixture.graph,
                &cache,
                run.clone(),
                TargetOptions {
                    reverse: arm == "control",
                    ..Default::default()
                },
            )?
        };
        #[cfg(not(reach_new))]
        let result = runner.run_eager_cached(&fixture.graph, &cache, run.clone())?;
        let mut digest = 0xcbf29ce484222325u64;
        for path in &result.paths {
            if audit {
                for byte in format!("{path:?}").bytes() {
                    digest = (digest ^ u64::from(byte)).wrapping_mul(1099511628211);
                }
            } else {
                black_box(path);
            }
        }
        Ok((result.paths.len(), result.stats.evaluated_edges, digest))
    };
    if mode == "index" {
        println!(
            "{}",
            json!({"groups":groups,"nodes":fixture.graph.node_count(),"edges":fixture.graph.edge_count(),"index_build_ms":build_ms})
        );
        return Ok(());
    }
    if mode == "verify" || mode == "rss" {
        let (paths, edges, digest) = one(true)?;
        anyhow::ensure!(paths == groups, "expected {groups} paths, got {paths}");
        println!(
            "{}",
            json!({"groups":groups,"len":len,"arm":arm,"paths":paths,"edges":edges,"calls":counter.load(Ordering::Relaxed),"digest":format!("{digest:016x}"),"index_build_ms":build_ms})
        );
    } else {
        let _ = one(false)?;
        let start = Instant::now();
        let mut loops = 0usize;
        while start.elapsed().as_secs_f64() < 2.0 {
            black_box(one(false)?);
            loops += 1;
        }
        println!(
            "{}",
            json!({"groups":groups,"len":len,"arm":arm,"loops":loops,"total_seconds":start.elapsed().as_secs_f64(),"seconds_per_search":start.elapsed().as_secs_f64()/loops as f64,"index_build_ms":build_ms})
        );
    }
    Ok(())
}
