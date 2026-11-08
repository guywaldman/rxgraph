use std::borrow::Cow;

use anyhow::{Context, Result, ensure};
use arrow::array::record_batch;
use rxgraph::{
    EdgeCtx, EdgeId, Evaluation, Graph, GraphId, Kernel, NodeId, RunOptions, TargetCtx,
    TargetOptions, Transition, traversal::native,
};

fn graph() -> Result<Graph> {
    Graph::new(
        record_batch!(
            ("id", UInt64, [100, 101, 102]),
            ("value", UInt64, [10, 20, 30])
        )?,
        record_batch!(
            ("id", UInt64, [200, 201]),
            ("src", UInt64, [100, 101]),
            ("dest", UInt64, [101, 102]),
            ("cost", UInt64, [4, 7])
        )?,
    )
}

#[derive(Clone)]
struct TypedAccess {
    graph_backed: bool,
}

impl TypedAccess {
    fn check_start(&self, cx: &native::StartCtx<'_, u64, u64>) -> Result<()> {
        ensure!(cx.graph().is_some() == self.graph_backed);
        ensure!(*cx.node_at(2)? == 30);
        ensure!(*cx.edge_at(1)? == 7);
        ensure!(cx.external_node_at(2)? == Some(GraphId::U64(102)));
        ensure!(cx.node_at(99).is_err());
        ensure!(cx.edge_at(99).is_err());
        Ok(())
    }

    fn check_edge<S>(&self, cx: &native::EdgeCtx<'_, '_, u64, u64, S>) -> Result<()> {
        ensure!(cx.graph().is_some() == self.graph_backed);
        ensure!(*cx.node_at(2)? == 30);
        ensure!(*cx.edge_at(1)? == 7);
        ensure!(cx.external_node_at(2)? == Some(GraphId::U64(102)));
        ensure!(cx.node_at(99).is_err());
        ensure!(cx.edge_at(99).is_err());
        Ok(())
    }
}

impl native::Kernel for TypedAccess {
    type Node = u64;
    type Edge = u64;
    type State = u64;

    fn initial_state(&self, cx: &native::StartCtx<'_, u64, u64>) -> Result<u64> {
        self.check_start(cx)?;
        Ok(0)
    }

    fn transition(&self, cx: &native::EdgeCtx<'_, '_, u64, u64, u64>) -> Result<Transition<u64>> {
        self.check_edge(cx)?;
        let value = cx.state() + cx.edge()?;
        Ok(if cx.dest_id() == 2 {
            Transition::Complete(value)
        } else {
            Transition::Continue(value)
        })
    }

    fn possible_targets<'a>(
        &'a self,
        cx: &native::TargetCtx<'a, u64, u64, u64>,
    ) -> Result<Option<Cow<'a, [NodeId]>>> {
        match cx {
            native::TargetCtx::Start { start, .. } => self.check_start(start)?,
            native::TargetCtx::Edge { edge, .. } => self.check_edge(edge)?,
        }
        Ok(Some(Cow::Borrowed(&[2])))
    }
}

struct Standalone {
    nodes: [u64; 3],
    edges: [u64; 2],
    outgoing: [Vec<native::OutgoingEdge>; 3],
}

impl native::GraphStore for Standalone {
    type Node = u64;
    type Edge = u64;

    fn resolve_node(&self, id: GraphId<'_>) -> Result<Option<NodeId>> {
        Ok(match id {
            GraphId::U64(id @ 100..=102) => Some((id - 100) as NodeId),
            _ => None,
        })
    }

    fn external_node(&self, id: NodeId) -> Result<Option<GraphId<'_>>> {
        Ok(((id as usize) < self.nodes.len()).then_some(GraphId::U64(id as u64 + 100)))
    }

    fn external_edge(&self, id: EdgeId) -> Result<Option<GraphId<'_>>> {
        Ok(((id as usize) < self.edges.len()).then_some(GraphId::U64(id as u64 + 200)))
    }

    fn dense_node_count(&self) -> Option<usize> {
        Some(self.nodes.len())
    }

    fn outgoing(&self, id: NodeId) -> Result<&[native::OutgoingEdge]> {
        Ok(self.outgoing.get(id as usize).context("invalid node")?)
    }

    fn node(&self, id: NodeId) -> Result<&u64> {
        self.nodes.get(id as usize).context("invalid node")
    }

    fn edge(&self, id: EdgeId) -> Result<&u64> {
        self.edges.get(id as usize).context("invalid edge")
    }
}

fn check_native<G: native::GraphStore<Node = u64, Edge = u64>>(
    store: &G,
    backed: bool,
) -> Result<()> {
    let kernel = TypedAccess {
        graph_backed: backed,
    };
    let run = RunOptions {
        start_nodes: vec![100u64.into()],
        parallel: false,
        ..Default::default()
    };
    let baseline = native::search_paths(store, kernel.clone(), run.clone())?;
    assert_eq!(baseline.paths.len(), 1);
    assert_eq!(baseline.paths[0].state, 11);
    let streamed = native::search_batches(store, kernel.clone(), run.clone(), 1)?
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(streamed, baseline.paths);
    for evaluation in [Evaluation::Incremental, Evaluation::Lazy] {
        let options = TargetOptions {
            evaluation,
            reverse: true,
            release_expanded: true,
        };
        let actual = native::search_paths(
            store,
            kernel.clone(),
            run.clone().with_optimizations(options),
        )?;
        assert_eq!(actual.paths, baseline.paths);
        let paths = native::search_batches(
            store,
            kernel.clone(),
            run.clone().with_optimizations(options),
            1,
        )?
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        assert_eq!(paths, baseline.paths);
    }
    Ok(())
}

#[test]
fn native_contexts_access_arbitrary_rows_in_graph_and_standalone_stores() -> Result<()> {
    let graph = graph()?;
    let nodes = [10, 20, 30];
    let edges = [4, 7];
    let store = native::EagerGraphStore::new(&graph, &nodes, &edges)?;
    check_native(&store, true)?;
    let store = Standalone {
        nodes,
        edges,
        outgoing: [
            vec![native::OutgoingEdge { edge: 0, dest: 1 }],
            vec![native::OutgoingEdge { edge: 1, dest: 2 }],
            vec![],
        ],
    };
    check_native(&store, false)
}

struct UntypedAccess;

fn check_graph_rows(graph: &Graph) -> Result<()> {
    ensure!(graph.node_at(2)?.u64("value")? == Some(30));
    ensure!(graph.edge_at(1)?.u64("cost")? == Some(7));
    ensure!(graph.external_node_at(2) == Some(GraphId::U64(102)));
    ensure!(graph.external_node_at(99).is_none());
    ensure!(graph.node_at(99).is_err());
    ensure!(graph.edge_at(99).is_err());
    Ok(())
}

impl Kernel for UntypedAccess {
    type State = u64;
    fn initial_state(&self, graph: &Graph, _: NodeId) -> Result<u64> {
        check_graph_rows(graph)?;
        Ok(0)
    }
    fn transition(&self, cx: &EdgeCtx<'_, u64>) -> Result<Transition<u64>> {
        check_graph_rows(cx.graph())?;
        let state = cx.state() + cx.edge_u64("cost")?.unwrap();
        Ok(if cx.dest() == 2 {
            Transition::Complete(state)
        } else {
            Transition::Continue(state)
        })
    }
    fn possible_targets<'a>(
        &'a self,
        cx: &TargetCtx<'_, u64>,
    ) -> Result<Option<Cow<'a, [NodeId]>>> {
        match cx {
            TargetCtx::Start { graph, .. } => check_graph_rows(graph)?,
            TargetCtx::Edge { edge, .. } => check_graph_rows(edge.graph())?,
        }
        Ok(Some(Cow::Borrowed(&[2])))
    }
}

#[test]
fn untyped_kernels_access_arbitrary_rows_in_search_and_stream() -> Result<()> {
    let graph = graph()?;
    let run = RunOptions {
        start_nodes: vec![100u64.into()],
        parallel: false,
        ..Default::default()
    };
    let baseline = graph.search_paths_with(UntypedAccess, run.clone())?;
    assert_eq!(baseline.paths[0].state, 11);
    let streamed = graph
        .search_batches_with(UntypedAccess, run.clone(), 1)?
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(streamed, baseline.paths);
    for evaluation in [Evaluation::Incremental, Evaluation::Lazy] {
        let options = TargetOptions {
            evaluation,
            reverse: true,
            release_expanded: true,
        };
        let result =
            graph.search_paths_with(UntypedAccess, run.clone().with_optimizations(options))?;
        assert_eq!(result.paths, baseline.paths);
        let paths = graph
            .search_batches_with(UntypedAccess, run.clone().with_optimizations(options), 1)?
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        assert_eq!(paths, baseline.paths);
    }
    Ok(())
}
