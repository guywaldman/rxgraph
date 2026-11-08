use std::{
    borrow::Cow,
    sync::atomic::{AtomicUsize, Ordering},
};

use anyhow::Result;
use arrow::array::record_batch;
use rxgraph::{
    EdgeCtx, Graph, Kernel, NodeId, ReachIndex, ReachSpec, RunOptions, TargetCtx, TargetOptions,
    Transition, TraversalStrategy,
};

fn graph() -> Result<Graph> {
    Graph::new(
        record_batch!(("id", UInt64, [0, 1, 2, 3, 4, 5, 6, 7]))?,
        record_batch!(
            ("id", UInt64, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
            ("src", UInt64, [0, 1, 2, 1, 3, 0, 5, 6, 7]),
            ("dest", UInt64, [1, 2, 4, 3, 4, 5, 6, 7, 6])
        )?,
    )
}

struct KernelForGroup<'a> {
    index: &'a ReachIndex,
    group: u32,
    target: NodeId,
    prune: bool,
    transitions: &'a AtomicUsize,
}

impl Kernel for KernelForGroup<'_> {
    type State = usize;

    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<usize> {
        Ok(0)
    }

    fn transition(&self, cx: &EdgeCtx<'_, usize>) -> Result<Transition<usize>> {
        self.transitions.fetch_add(1, Ordering::Relaxed);
        if cx.edge() == 2 {
            return Ok(Transition::Reject);
        }
        let state = cx.state() + 1;
        Ok(if cx.dest() == self.target {
            Transition::Complete(state)
        } else {
            Transition::Continue(state)
        })
    }

    fn possible_targets<'a>(
        &'a self,
        cx: &TargetCtx<'_, usize>,
    ) -> Result<Option<Cow<'a, [NodeId]>>> {
        if !self.prune {
            return Ok(None);
        }
        match cx {
            TargetCtx::Start { graph, .. } => {
                Ok(Some(Cow::Borrowed(self.index.targets(graph, self.group)?)))
            }
            TargetCtx::Edge { edge, .. } => {
                if edge.reaches(self.index, self.group, edge.dest())? {
                    Ok(None)
                } else {
                    Ok(Some(Cow::Borrowed(&[])))
                }
            }
        }
    }

    fn targets_depend_on_child_state(&self) -> bool {
        false
    }
}

#[test]
fn filtered_index_reused_across_searches_and_rejects_another_graph() -> Result<()> {
    let graph = graph()?;
    let skip = |edge| edge == 2;
    let groups = (0..12)
        .map(|group| (group, vec![if group % 2 == 0 { 4 } else { 7 }]))
        .collect();
    let index = graph.reach_index(ReachSpec {
        sets: groups,
        skip_edge: Some(&skip),
    })?;
    let counter = AtomicUsize::new(0);

    for strategy in [
        TraversalStrategy::DepthFirst,
        TraversalStrategy::BreadthFirst,
    ] {
        for (group, target) in [(0, 4), (1, 7), (10, 4), (11, 7)] {
            let run = RunOptions {
                start_nodes: vec![0u64.into()],
                strategy,
                parallel: false,
                ..Default::default()
            };
            let make_kernel = |prune| KernelForGroup {
                index: &index,
                group,
                target,
                prune,
                transitions: &counter,
            };
            counter.store(0, Ordering::Relaxed);
            let baseline = graph.search_paths_with(make_kernel(false), run.clone())?;
            let baseline_calls = counter.swap(0, Ordering::Relaxed);
            let options = TargetOptions::default();
            let optimized = graph
                .search_paths_with(make_kernel(true), run.clone().with_optimizations(options))?;
            assert_eq!(optimized.paths, baseline.paths);
            assert_eq!(optimized.paths.len(), 1);
            assert!(counter.load(Ordering::Relaxed) < baseline_calls);

            let mut stream =
                graph.search_batches_with(make_kernel(true), run.with_optimizations(options), 1)?;
            let streamed = stream.by_ref().collect::<Result<Vec<_>>>()?.concat();
            assert_eq!(streamed, baseline.paths);
        }
    }

    let other = self::graph()?;
    let error = other.search_paths_with(
        KernelForGroup {
            index: &index,
            group: 0,
            target: 4,
            prune: true,
            transitions: &counter,
        },
        RunOptions {
            start_nodes: vec![0u64.into()],
            parallel: false,
            ..Default::default()
        }
        .with_optimizations(TargetOptions::default()),
    );
    assert!(error.is_err());
    assert!(format!("{:#}", error.err().unwrap()).contains("different graph"));
    Ok(())
}
