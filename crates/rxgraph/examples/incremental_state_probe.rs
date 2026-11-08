//! Count actual kernel calls without changing the traversal engine.
//! Run: cargo run -p rxgraph --example incremental_state_probe --offline

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::Result;
use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use rxgraph::{
    DslExpr as e, DslKernel, EdgeCtx, Graph, GraphId, Kernel, NodeId, RunOptions, Transition,
    TraversalConfigBuilder, TraversalStrategy, Value,
};

struct Trace {
    starts: AtomicUsize,
    // Parent states observed at each edge; indexed by external edge ID.
    inputs: Vec<Mutex<Vec<u64>>>,
}

impl Trace {
    fn new(edges: usize) -> Arc<Self> {
        Arc::new(Self {
            starts: AtomicUsize::new(0),
            inputs: (0..edges).map(|_| Mutex::new(Vec::new())).collect(),
        })
    }

    fn counts(&self) -> Vec<usize> {
        self.inputs
            .iter()
            .map(|v| v.lock().unwrap().len())
            .collect()
    }

    fn inputs(&self, edge: usize) -> Vec<u64> {
        let mut inputs = self.inputs[edge].lock().unwrap().clone();
        inputs.sort_unstable();
        inputs
    }
}

struct ProbeKernel {
    trace: Arc<Trace>,
    terminal_from: u64,
    weighted: bool,
}

fn id(id: GraphId<'_>) -> u64 {
    match id {
        GraphId::U64(id) => id,
        GraphId::Str(_) => unreachable!("probe graphs use integer IDs"),
    }
}

impl Kernel for ProbeKernel {
    type State = u64;

    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<u64> {
        self.trace.starts.fetch_add(1, Ordering::Relaxed);
        Ok(0)
    }

    fn transition(&self, cx: &EdgeCtx<'_, u64>) -> Result<Transition<u64>> {
        let edge = id(cx.edge_id().unwrap());
        self.trace.inputs[edge as usize]
            .lock()
            .unwrap()
            .push(*cx.state());
        let state = cx.state() + if self.weighted { edge + 1 } else { 1 };
        Ok(if id(cx.dest_id().unwrap()) >= self.terminal_from {
            Transition::Complete(state)
        } else {
            Transition::Continue(state)
        })
    }
}

fn graph(nodes: u64, edges: &[(u64, u64)]) -> Result<Graph> {
    let column = |values: Vec<u64>| Arc::new(UInt64Array::from(values)) as ArrayRef;
    Graph::new(
        RecordBatch::try_from_iter([("id", column((0..nodes).collect()))])?,
        RecordBatch::try_from_iter([
            ("id", column((0..edges.len() as u64).collect())),
            ("src", column(edges.iter().map(|&(src, _)| src).collect())),
            (
                "dest",
                column(edges.iter().map(|&(_, dest)| dest).collect()),
            ),
        ])?,
    )
}

fn run(
    graph: &Graph,
    trace: &Arc<Trace>,
    terminal_from: u64,
    weighted: bool,
    options: RunOptions,
    streaming: bool,
) -> Result<Vec<(Vec<u64>, u64)>> {
    let before: usize = trace.counts().iter().sum();
    let kernel = ProbeKernel {
        trace: trace.clone(),
        terminal_from,
        weighted,
    };
    let (paths, stats) = if streaming {
        // A single path per pull forces suspension between sibling branches.
        let mut stream = graph.search_batches_with(kernel, options, 1)?;
        let mut paths = Vec::new();
        for batch in stream.by_ref() {
            paths.extend(batch?);
        }
        (paths, stream.stats())
    } else {
        let result = graph.search_with(kernel, options)?;
        (result.paths, result.stats)
    };
    assert_eq!(
        trace.counts().iter().sum::<usize>() - before,
        stats.evaluated_edges
    );
    let mut paths = paths
        .into_iter()
        .map(|path| (path.nodes.into_iter().map(id).collect(), path.state))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn options(strategy: TraversalStrategy, parallel: bool) -> RunOptions {
    RunOptions {
        start_nodes: vec![0_u64.into()],
        strategy,
        parallel,
        ..Default::default()
    }
}

fn main() -> Result<()> {
    use TraversalStrategy::{BreadthFirst, DepthFirst};

    // A=0, B=1, C=2, D1=3, D2=4, E1=5, E2=6.
    let shared = graph(7, &[(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 6)])?;
    let expected = vec![(vec![0, 1, 2, 3, 5], 4), (vec![0, 1, 2, 4, 6], 4)];
    for strategy in [DepthFirst, BreadthFirst] {
        for parallel in [false, true] {
            for streaming in [false, true] {
                let trace = Trace::new(6);
                assert_eq!(
                    run(
                        &shared,
                        &trace,
                        5,
                        false,
                        options(strategy, parallel),
                        streaming
                    )?,
                    expected
                );
                assert_eq!(trace.counts(), vec![1; 6]);
                assert_eq!(trace.starts.load(Ordering::Relaxed), 1);
            }
        }
    }
    println!("shared prefix: 2 paths, 6 transitions (replaying each path: 8)");
    println!("  checked DFS/BFS, parallel disabled/enabled, eager/batch-size-1 streaming");

    // The Polars-expression entry point binds to the same engine and is incremental too.
    let dsl = DslKernel::new(
        e::bool_lit(true),
        [("hops".into(), e::state("hops").plus(e::uint_lit(1)))],
        e::dest_id().ge(e::uint_lit(5)),
        [("hops".into(), Value::U64(0))],
    );
    let result = shared.search(
        TraversalConfigBuilder::new(dsl)
            .with_start_nodes([0_u64])
            .with_parallelism(false)
            .build(),
    )?;
    assert_eq!(result.stats.evaluated_edges, 6);
    let mut dsl_paths = result
        .paths
        .into_iter()
        .map(|path| {
            assert_eq!(path.state, vec![("hops".into(), Value::U64(4))]);
            (path.nodes.into_iter().map(id).collect(), 4)
        })
        .collect::<Vec<_>>();
    dsl_paths.sort();
    assert_eq!(dsl_paths, expected);
    println!("DSL: identical paths and final states, 6 transitions");

    // 98 shared edges followed by 1,000 distinct two-edge tails.
    let mut edges = (0..98).map(|i| (i, i + 1)).collect::<Vec<_>>();
    for i in 0..1_000 {
        edges.extend([(98, 99 + i), (99 + i, 1_099 + i)]);
    }
    let fanout = graph(2_099, &edges)?;
    for strategy in [DepthFirst, BreadthFirst] {
        let trace = Trace::new(edges.len());
        let paths = run(
            &fanout,
            &trace,
            1_099,
            false,
            options(strategy, true),
            false,
        )?;
        assert_eq!(paths.len(), 1_000);
        assert!(
            paths
                .iter()
                .all(|(nodes, state)| nodes.len() == 101 && *state == 100)
        );
        assert_eq!(trace.counts(), vec![1; 2_098]);
    }
    println!("long prefix: 1,000 paths, 2,098 transitions (replaying each path: 100,000)");

    let trace = Trace::new(6);
    for _ in 0..2 {
        assert_eq!(
            run(&shared, &trace, 5, false, options(DepthFirst, false), false)?,
            expected
        );
    }
    assert_eq!(trace.counts(), vec![2; 6]);
    assert_eq!(trace.starts.load(Ordering::Relaxed), 2);
    println!("two independent searches: 12 transitions, no cross-search state reuse");

    // Two different prefixes reconverge at 3, then share edge 3->4.
    let diamond = graph(5, &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])?;
    for (weighted, inputs, states) in [
        (false, vec![2, 2], vec![3, 3]),
        (true, vec![4, 6], vec![9, 11]),
    ] {
        let trace = Trace::new(5);
        let paths = run(
            &diamond,
            &trace,
            4,
            weighted,
            options(DepthFirst, false),
            false,
        )?;
        assert_eq!(trace.counts(), vec![1, 1, 1, 1, 2]);
        assert_eq!(trace.inputs(4), inputs);
        assert_eq!(
            paths.iter().map(|(_, state)| *state).collect::<Vec<_>>(),
            states
        );
        println!(
            "reconverging prefixes (weighted={weighted}): edge 3->4 evaluated with states {inputs:?}"
        );
    }

    // Equal (node, state) at 3 can still have different allowed continuations:
    // 0->1->3 cannot revisit 1, whereas 0->2->3 can.
    let cyclic = graph(5, &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 1), (1, 4)])?;
    let trace = Trace::new(6);
    let paths = run(&cyclic, &trace, 4, false, options(DepthFirst, false), false)?;
    assert_eq!(paths, vec![(vec![0, 1, 4], 2), (vec![0, 2, 3, 1, 4], 4)]);
    assert_eq!(trace.inputs(4), vec![2]);
    println!(
        "equal (node, state) can have different legal continuations under per-path revisit limits"
    );
    Ok(())
}
