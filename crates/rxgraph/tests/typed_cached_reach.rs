//! A registered typed kernel prepares graph-wide data once and reuses it.
use std::{
    borrow::Cow,
    fs::File,
    path::Path,
    sync::{
        Arc, Barrier, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use arrow::array::{RecordBatch, record_batch};
use parquet::arrow::ArrowWriter;
use rxgraph::{
    ArrowRow, Evaluation, Graph, GraphId, NativeBatchStream, NodeId, OwnedGraphId, OwnedGraphPath,
    ParquetPaths, PayloadField, ReachIndex, ReachSpec, RunOptions, SearchStats, TargetOptions,
    Transition, TraversalStrategy, TypedKernel, TypedPayloadCache, boxed_typed_run,
    build_typed_kernel, traversal::native,
};

static BUILDS: AtomicUsize = AtomicUsize::new(0);
static SLOW_BUILD: AtomicBool = AtomicBool::new(false);
static TARGET_START_HOOKS: AtomicUsize = AtomicUsize::new(0);
static TARGET_EDGE_HOOKS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
struct Node {
    target: bool,
}
impl TryFrom<ArrowRow<'_>> for Node {
    type Error = anyhow::Error;
    fn try_from(row: ArrowRow<'_>) -> Result<Self> {
        Ok(Self {
            target: row.bool("target")?.context("target")?,
        })
    }
}

#[derive(Clone)]
struct Edge {
    cost: u64,
}
impl TryFrom<ArrowRow<'_>> for Edge {
    type Error = anyhow::Error;
    fn try_from(row: ArrowRow<'_>) -> Result<Self> {
        Ok(Self {
            cost: row.u64("cost")?.context("cost")?,
        })
    }
}

struct Prepared {
    index: ReachIndex,
}

#[derive(Clone, serde::Serialize)]
struct State {
    cost: u64,
}

#[derive(Clone)]
struct Kernel {
    prune: bool,
    prepared: Arc<OnceLock<Arc<Prepared>>>,
}

impl Kernel {
    fn new(prune: bool) -> Self {
        Self {
            prune,
            prepared: Arc::new(OnceLock::new()),
        }
    }
}

impl TypedKernel for Kernel {
    type Node = Node;
    type Edge = Edge;
    type State = State;

    fn node_fields(&self) -> Vec<PayloadField> {
        vec![PayloadField::new("target")]
    }
    fn edge_fields(&self) -> Vec<PayloadField> {
        vec![PayloadField::new("cost")]
    }

    fn initial_state(&self, cx: &native::StartCtx<'_, Node, Edge>) -> Result<State> {
        let graph = cx.graph().context("graph-backed store required")?;
        let prepared = graph.cached("typed-cached-reach-test", || {
            BUILDS.fetch_add(1, Ordering::SeqCst);
            if SLOW_BUILD.load(Ordering::SeqCst) {
                // Hold initialization long enough for other searches to arrive.
                std::thread::sleep(Duration::from_millis(20));
            }
            let mut targets = Vec::new();
            for node in 0..graph.node_count() as NodeId {
                if cx.node_at(node)?.target {
                    targets.push(node);
                }
            }
            // A non-current edge can be read while preparing graph-wide data.
            ensure!(cx.edge_at(4)?.cost == 5, "unexpected edge payload");
            ensure!(matches!(cx.external_node_at(5)?, Some(GraphId::U64(150))));
            Ok(Prepared {
                index: graph.reach_index(ReachSpec {
                    sets: vec![(7, targets)],
                    skip_edge: None,
                })?,
            })
        })?;
        let _ = self.prepared.set(prepared);
        Ok(State { cost: 0 })
    }

    fn transition(
        &self,
        cx: &native::EdgeCtx<'_, '_, Node, Edge, State>,
    ) -> Result<Transition<State>> {
        ensure!(cx.graph().is_some());
        ensure!(!cx.node_at(5)?.target);
        ensure!(cx.edge_at(4)?.cost == 5);
        ensure!(matches!(cx.external_node_at(5)?, Some(GraphId::U64(150))));
        if self.prune
            && !cx.reaches(
                &self.prepared.get().context("kernel not prepared")?.index,
                7,
                cx.dest_id(),
            )?
        {
            return Ok(Transition::Reject);
        }
        let state = State {
            cost: cx.state().cost + cx.edge()?.cost,
        };
        Ok(if cx.dest()?.target {
            Transition::Complete(state)
        } else {
            Transition::Continue(state)
        })
    }

    fn possible_targets<'a>(
        &'a self,
        cx: &native::TargetCtx<'a, Node, Edge, State>,
    ) -> Result<Option<Cow<'a, [NodeId]>>> {
        match cx {
            native::TargetCtx::Start { start, .. } => {
                TARGET_START_HOOKS.fetch_add(1, Ordering::Relaxed);
                ensure!(start.graph().is_some());
                ensure!(!start.node_at(5)?.target);
                ensure!(start.edge_at(4)?.cost == 5);
                ensure!(matches!(
                    start.external_node_at(5)?,
                    Some(GraphId::U64(150))
                ));
            }
            native::TargetCtx::Edge { edge, .. } => {
                TARGET_EDGE_HOOKS.fetch_add(1, Ordering::Relaxed);
                ensure!(edge.graph().is_some());
                ensure!(!edge.node_at(5)?.target);
                ensure!(edge.edge_at(4)?.cost == 5);
                ensure!(matches!(edge.external_node_at(5)?, Some(GraphId::U64(150))));
            }
        }
        Ok(None)
    }
}

rxgraph::inventory::submit! {
    rxgraph::TypedKernelEntry {
        name: "typed_cached_reach_test",
        make: |params| Ok(boxed_typed_run(Kernel::new(params["prune"].as_bool().unwrap_or(false)))),
    }
}

fn batches() -> Result<(RecordBatch, RecordBatch)> {
    Ok((
        record_batch!(
            ("id", UInt64, [100, 110, 120, 130, 140, 150]),
            ("target", Boolean, [false, false, false, true, false, false])
        )?,
        record_batch!(
            ("id", UInt64, [200, 201, 202, 203, 204]),
            ("src", UInt64, [100, 110, 100, 120, 140]),
            ("dest", UInt64, [110, 130, 120, 140, 150]),
            ("cost", UInt64, [1, 2, 3, 4, 5])
        )?,
    ))
}

fn run() -> RunOptions {
    RunOptions {
        start_nodes: vec![100_u64.into()],
        strategy: TraversalStrategy::BreadthFirst,
        parallel: false,
        ..Default::default()
    }
}

fn snapshot(paths: &[OwnedGraphPath]) -> Vec<(Vec<OwnedGraphId>, Vec<OwnedGraphId>, String)> {
    paths
        .iter()
        .map(|p| (p.nodes.clone(), p.edges.clone(), format!("{:?}", p.state)))
        .collect()
}

fn collect_stream(
    mut stream: Box<dyn NativeBatchStream>,
) -> Result<(Vec<OwnedGraphPath>, SearchStats)> {
    let mut paths = Vec::new();
    while let Some(batch) = stream.next_batch(1)? {
        paths.extend(batch.paths);
    }
    let stats = stream.stats();
    stream.close();
    Ok((paths, stats))
}

fn check_graph(graph: Arc<Graph>, parquet: Option<ParquetPaths>) -> Result<()> {
    let cache = TypedPayloadCache::default();
    let make = |prune| {
        build_typed_kernel(
            "typed_cached_reach_test",
            &serde_json::json!({"prune": prune}),
        )
    };
    let direct = |prune, options: Option<TargetOptions>| -> Result<_> {
        let kernel = make(prune)?;
        match (&parquet, options) {
            (None, None) => kernel.run_eager_cached(&graph, &cache, run()),
            (None, Some(o)) => kernel.run_eager_cached_optimized(&graph, &cache, run(), o),
            (Some(paths), None) => kernel.run_parquet_lazy(&graph, paths.clone(), run()),
            (Some(paths), Some(o)) => {
                kernel.run_parquet_lazy_optimized(&graph, paths.clone(), run(), o)
            }
        }
    };
    let stream = |prune, options: Option<TargetOptions>| -> Result<_> {
        let kernel = make(prune)?;
        match (&parquet, options) {
            (None, None) => kernel.stream_eager_cached(Arc::clone(&graph), &cache, run()),
            (None, Some(o)) => {
                kernel.stream_eager_cached_optimized(Arc::clone(&graph), &cache, run(), o)
            }
            (Some(paths), None) => {
                kernel.stream_parquet_lazy(Arc::clone(&graph), paths.clone(), run())
            }
            (Some(paths), Some(o)) => {
                kernel.stream_parquet_lazy_optimized(Arc::clone(&graph), paths.clone(), run(), o)
            }
        }
    };
    let before_builds = BUILDS.load(Ordering::SeqCst);
    let baseline = direct(false, None)?;
    assert_eq!(BUILDS.load(Ordering::SeqCst), before_builds + 1);
    assert_eq!(baseline.paths.len(), 1);
    assert_eq!(
        baseline.paths[0].nodes,
        vec![100_u64.into(), 110_u64.into(), 130_u64.into()]
    );
    assert_eq!(
        baseline.paths[0].edges,
        vec![200_u64.into(), 201_u64.into()]
    );
    let pruned = direct(true, None)?;
    assert_eq!(snapshot(&baseline.paths), snapshot(&pruned.paths));
    assert!(pruned.stats.evaluated_edges < baseline.stats.evaluated_edges);
    if parquet.is_some() {
        // Building derived data scans every node in the first lazy search;
        // a later search reuses the cache instead of repeating that scan.
        assert_eq!(baseline.stats.materialized_node_payloads, 6);
        assert!(pruned.stats.materialized_node_payloads < 6);
    }
    for options in [
        None,
        Some(TargetOptions {
            evaluation: Evaluation::Incremental,
            reverse: true,
            release_expanded: true,
        }),
        Some(TargetOptions {
            evaluation: Evaluation::Lazy,
            reverse: true,
            release_expanded: true,
        }),
    ] {
        let result = direct(true, options)?;
        assert_eq!(snapshot(&baseline.paths), snapshot(&result.paths));
        let (streamed, _) = collect_stream(stream(true, options)?)?;
        assert_eq!(snapshot(&baseline.paths), snapshot(&streamed));
    }
    assert_eq!(
        BUILDS.load(Ordering::SeqCst),
        before_builds + 1,
        "fresh kernels must share graph cache"
    );
    Ok(())
}

fn check_parallel_searches(graph: Arc<Graph>) -> Result<()> {
    const THREADS: usize = 8;
    let barrier = Arc::new(Barrier::new(THREADS));
    let before_builds = BUILDS.load(Ordering::SeqCst);
    SLOW_BUILD.store(true, Ordering::SeqCst);
    let results = std::thread::scope(|scope| -> Result<Vec<_>> {
        let tasks = (0..THREADS)
            .map(|_| {
                let graph = Arc::clone(&graph);
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || -> Result<_> {
                    let kernel = build_typed_kernel(
                        "typed_cached_reach_test",
                        &serde_json::json!({"prune": true}),
                    )?;
                    let cache = TypedPayloadCache::default();
                    barrier.wait();
                    Ok(snapshot(
                        &kernel.run_eager_cached(&graph, &cache, run())?.paths,
                    ))
                })
            })
            .collect::<Vec<_>>();
        tasks
            .into_iter()
            .map(|t| t.join().expect("search thread panicked"))
            .collect()
    });
    SLOW_BUILD.store(false, Ordering::SeqCst);
    let results = results?;
    assert!(results.iter().all(|result| result == &results[0]));
    assert_eq!(results[0].len(), 1);
    assert_eq!(BUILDS.load(Ordering::SeqCst), before_builds + 1);
    Ok(())
}

fn write_parquet(path: &Path, batch: &RecordBatch) -> Result<()> {
    let mut writer = ArrowWriter::try_new(File::create(path)?, batch.schema(), None)?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}

#[test]
fn registered_typed_kernel_reuses_index_for_eager_and_file_backed_searches() -> Result<()> {
    let (nodes, edges) = batches()?;
    let concurrent = Arc::new(Graph::new(nodes.clone(), edges.clone())?);
    check_parallel_searches(concurrent)?;
    let eager = Arc::new(Graph::new(nodes.clone(), edges.clone())?);
    check_graph(eager, None)?;
    let id = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let dir = std::env::temp_dir();
    let paths = ParquetPaths {
        nodes: dir.join(format!(
            "rxgraph-cached-nodes-{}-{id}.parquet",
            std::process::id()
        )),
        edges: dir.join(format!(
            "rxgraph-cached-edges-{}-{id}.parquet",
            std::process::id()
        )),
    };
    write_parquet(&paths.nodes, &nodes)?;
    write_parquet(&paths.edges, &edges)?;
    let file_eager = Arc::new(Graph::from_parquet(
        paths.nodes.clone(),
        paths.edges.clone(),
    )?);
    check_graph(file_eager, None)?;
    let file_graph = Arc::new(Graph::from_parquet_topology(
        paths.nodes.clone(),
        paths.edges.clone(),
    )?);
    let result = check_graph(file_graph, Some(paths.clone()));
    std::fs::remove_file(&paths.nodes)?;
    std::fs::remove_file(&paths.edges)?;
    result?;
    assert!(TARGET_START_HOOKS.load(Ordering::Relaxed) > 0);
    assert!(TARGET_EDGE_HOOKS.load(Ordering::Relaxed) > 0);
    Ok(())
}
