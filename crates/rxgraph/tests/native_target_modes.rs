use std::{
    borrow::Cow,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use anyhow::Result;
use rxgraph::{
    EdgeId, Evaluation, GraphId, NodeId, RunOptions, TargetOptions, Transition, TraversalStrategy,
    traversal::native::{self, GraphStore, OutgoingEdge},
};

struct Store {
    edges: Vec<Vec<OutgoingEdge>>,
    dense: bool,
}
impl Store {
    fn new(dense: bool) -> Self {
        Self {
            dense,
            edges: vec![
                vec![
                    OutgoingEdge { edge: 0, dest: 1 },
                    OutgoingEdge { edge: 1, dest: 2 },
                ],
                vec![OutgoingEdge { edge: 2, dest: 3 }],
                vec![OutgoingEdge { edge: 3, dest: 4 }],
                vec![],
                vec![],
            ],
        }
    }
}
impl GraphStore for Store {
    type Node = ();
    type Edge = ();
    fn resolve_node(&self, id: GraphId<'_>) -> Result<Option<NodeId>> {
        Ok(match id {
            GraphId::U64(id) if id < self.edges.len() as u64 => Some(id as NodeId),
            _ => None,
        })
    }
    fn external_node(&self, id: NodeId) -> Result<Option<GraphId<'_>>> {
        Ok(Some(GraphId::U64(id as u64)))
    }
    fn external_edge(&self, id: EdgeId) -> Result<Option<GraphId<'_>>> {
        Ok(Some(GraphId::U64(id as u64)))
    }
    fn dense_node_count(&self) -> Option<usize> {
        self.dense.then_some(self.edges.len())
    }
    fn outgoing(&self, id: NodeId) -> Result<&[OutgoingEdge]> {
        Ok(&self.edges[id as usize])
    }
    fn node(&self, _: NodeId) -> Result<&()> {
        Ok(&())
    }
    fn edge(&self, _: EdgeId) -> Result<&()> {
        Ok(&())
    }
}

#[derive(Clone)]
struct Machine {
    targets: Vec<NodeId>,
    stateful: bool,
    post: Arc<AtomicUsize>,
}
impl native::Kernel for Machine {
    type Node = ();
    type Edge = ();
    type State = u32;
    fn initial_state(&self, _: &native::StartCtx<'_, (), ()>) -> Result<u32> {
        Ok(0)
    }
    fn transition(&self, cx: &native::EdgeCtx<'_, '_, (), (), u32>) -> Result<Transition<u32>> {
        Ok(if cx.dest_id() == 3 {
            Transition::Complete(cx.state() + 1)
        } else {
            Transition::Continue(cx.state() + 1)
        })
    }
    fn possible_targets<'a>(
        &'a self,
        cx: &native::TargetCtx<'a, (), (), u32>,
    ) -> Result<Option<Cow<'a, [NodeId]>>> {
        if let native::TargetCtx::Edge { child: Some(_), .. } = cx {
            self.post.fetch_add(1, Ordering::Relaxed);
        }
        Ok(Some(Cow::Borrowed(&self.targets)))
    }
    fn targets_depend_on_child_state(&self) -> bool {
        self.stateful
    }
}

struct Unbounded(Machine);
impl native::Kernel for Unbounded {
    type Node = ();
    type Edge = ();
    type State = u32;
    fn initial_state(&self, cx: &native::StartCtx<'_, (), ()>) -> Result<u32> {
        self.0.initial_state(cx)
    }
    fn transition(&self, cx: &native::EdgeCtx<'_, '_, (), (), u32>) -> Result<Transition<u32>> {
        self.0.transition(cx)
    }
}

struct StateRefines;
impl native::Kernel for StateRefines {
    type Node = ();
    type Edge = ();
    type State = u32;
    fn initial_state(&self, _: &native::StartCtx<'_, (), ()>) -> Result<u32> {
        Ok(0)
    }
    fn transition(&self, cx: &native::EdgeCtx<'_, '_, (), (), u32>) -> Result<Transition<u32>> {
        Ok(if cx.dest_id() == 3 {
            Transition::Complete(cx.state() + 1)
        } else {
            Transition::Continue(cx.state() + 1)
        })
    }
    fn possible_targets<'a>(
        &'a self,
        cx: &native::TargetCtx<'a, (), (), u32>,
    ) -> Result<Option<Cow<'a, [NodeId]>>> {
        match cx {
            native::TargetCtx::Edge {
                edge,
                child: Some(1),
                ..
            } if edge.dest_id() == 2 => Ok(Some(Cow::Borrowed(&[]))),
            _ => Ok(Some(Cow::Borrowed(&[3]))),
        }
    }
}

fn run() -> RunOptions {
    RunOptions {
        start_nodes: vec![0u64.into()],
        strategy: TraversalStrategy::BreadthFirst,
        parallel: false,
        intermediate_states: true,
        ..Default::default()
    }
}
fn opts(evaluation: Evaluation, reverse: bool) -> TargetOptions {
    TargetOptions {
        evaluation,
        reverse,
        release_expanded: true,
    }
}

#[test]
fn native_target_modes_match_search_and_stream() -> Result<()> {
    let store = Store::new(true);
    let post = Arc::new(AtomicUsize::new(0));
    for evaluation in [Evaluation::Incremental, Evaluation::Lazy] {
        for reverse in [false, true] {
            let kernel = Machine {
                targets: vec![3],
                stateful: true,
                post: post.clone(),
            };
            let baseline = native::search_paths(&store, kernel.clone(), run())?;
            let optimized = native::search_paths(
                &store,
                kernel.clone(),
                run().with_optimizations(opts(evaluation, reverse)),
            )?;
            assert_eq!(baseline.paths, optimized.paths);
            let stream = native::search_batches(
                &store,
                kernel,
                run().with_optimizations(opts(evaluation, reverse)),
                1,
            )?;
            let paths = stream
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            assert_eq!(baseline.paths, paths);
        }
    }
    assert!(post.load(Ordering::Relaxed) > 0);
    Ok(())
}

#[test]
fn native_post_transition_hook_is_optional() -> Result<()> {
    let store = Store::new(true);
    for stateful in [false, true] {
        let post = Arc::new(AtomicUsize::new(0));
        let kernel = Machine {
            targets: vec![3],
            stateful,
            post: post.clone(),
        };
        let result = native::search_paths(
            &store,
            kernel,
            run().with_optimizations(opts(Evaluation::Incremental, false)),
        )?;
        assert_eq!(result.paths.len(), 1);
        assert_eq!(post.load(Ordering::Relaxed) > 0, stateful);
    }
    Ok(())
}

#[test]
fn native_validates_target_ids_and_dense_reverse_requirements() -> Result<()> {
    let post = Arc::new(AtomicUsize::new(0));
    let unknown = Machine {
        targets: vec![99],
        stateful: false,
        post: post.clone(),
    };
    let err = native::search_paths(
        &Store::new(true),
        unknown,
        run().with_optimizations(opts(Evaluation::Incremental, false)),
    )
    .unwrap_err();
    assert!(err.to_string().contains("unknown target node"));
    let kernel = Machine {
        targets: vec![3],
        stateful: false,
        post,
    };
    assert!(
        native::search_paths(
            &Store::new(false),
            kernel.clone(),
            run().with_optimizations(opts(Evaluation::Incremental, false))
        )
        .is_ok()
    );
    let err = native::search_paths(
        &Store::new(false),
        kernel,
        run().with_optimizations(opts(Evaluation::Incremental, true)),
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("dense, enumerable topology"));
    Ok(())
}

#[test]
fn native_unknown_bounds_preserve_paths() -> Result<()> {
    let store = Store::new(true);
    let kernel = Machine {
        targets: vec![3],
        stateful: false,
        post: Arc::new(AtomicUsize::new(0)),
    };
    let expected = native::search_paths(&store, Unbounded(kernel.clone()), run())?;
    for evaluation in [Evaluation::Incremental, Evaluation::Lazy] {
        let actual = native::search_paths(
            &store,
            Unbounded(kernel.clone()),
            run().with_optimizations(opts(evaluation, true)),
        )?;
        assert_eq!(expected.paths, actual.paths);
    }
    Ok(())
}

#[test]
fn child_state_refinement_preserves_native_results() -> Result<()> {
    let store = Store::new(true);
    let expected = native::search_paths(&store, StateRefines, run())?;
    for evaluation in [Evaluation::Incremental, Evaluation::Lazy] {
        let actual = native::search_paths(
            &store,
            StateRefines,
            run().with_optimizations(opts(evaluation, false)),
        )?;
        assert_eq!(expected.paths, actual.paths);
    }
    Ok(())
}
