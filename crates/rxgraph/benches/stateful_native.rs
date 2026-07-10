//! Macro benchmark for the Arrow-backed stateful search engine.
//!
//! This is deliberately a small executable rather than a Criterion benchmark:
//! the Python benchmark runner owns profile selection, cached data preparation,
//! result reporting, and raw-sample persistence for every benchmark family.

use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use rxgraph::{
    DslExpr as e, DslKernel, EdgeCtx, EdgeField, Graph, GraphId, Kernel, NodeId, RunOptions,
    Transition, TraversalConfigBuilder, TraversalStrategy, Value,
};
use serde::Serialize;
use serde_json::Value as JsonValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SearchState {
    spent: u64,
}

#[derive(Clone)]
struct BoundKernel {
    allowed: EdgeField<bool>,
    cost: EdgeField<u64>,
    target: u64,
    budget: u64,
}

impl Kernel for BoundKernel {
    type State = SearchState;

    fn initial_state(&self, _graph: &Graph, _start: NodeId) -> Result<Self::State> {
        Ok(SearchState { spent: 0 })
    }

    fn transition(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Transition<Self::State>> {
        if !cx.edge_field(&self.allowed).unwrap_or(false) {
            return Ok(Transition::Reject);
        }
        transition(
            cx,
            cx.edge_field(&self.cost).unwrap_or(0),
            self.target,
            self.budget,
        )
    }
}

fn transition(
    cx: &EdgeCtx<'_, SearchState>,
    cost: u64,
    target: u64,
    budget: u64,
) -> Result<Transition<SearchState>> {
    let state = SearchState {
        spent: cx.state().spent + cost,
    };
    Ok(if state.spent > budget {
        Transition::Reject
    } else if cx.dest_id() == Some(GraphId::U64(target)) {
        Transition::Complete(state)
    } else {
        Transition::Continue(state)
    })
}

struct Workload {
    graph: Graph,
    starts: Vec<rxgraph::OwnedGraphId>,
    bound: BoundKernel,
    depth: usize,
}

impl Workload {
    fn load(manifest: &Path) -> Result<Self> {
        let metadata: JsonValue = serde_json::from_slice(
            &fs::read(manifest).with_context(|| format!("read {}", manifest.display()))?,
        )?;
        let spec = metadata
            .get("spec")
            .context("benchmark manifest has no spec")?;
        let starts = spec_usize(spec, "starts")?;
        let depth = spec_usize(spec, "depth")?;
        let graph = Graph::from_parquet(
            manifest.with_file_name("nodes.parquet"),
            manifest.with_file_name("edges.parquet"),
        )?;
        let target = graph.node_count() as u64 - 1;
        let bound = BoundKernel {
            allowed: graph.edge_field("allowed")?,
            cost: graph.edge_field("cost")?,
            target,
            budget: depth as u64,
        };
        Ok(Self {
            graph,
            starts: (0..starts)
                .map(|case| ((case * depth) as u64).into())
                .collect(),
            bound,
            depth,
        })
    }

    fn run(&self, strategy: TraversalStrategy, parallel: bool, first: bool) -> RunOptions {
        RunOptions {
            start_nodes: self.starts.clone(),
            max_depth: Some(self.depth),
            max_paths: Some(if first { 1 } else { self.starts.len() }),
            strategy,
            max_visits_per_node: 1,
            parallel,
            intermediate_states: false,
            progress: false,
        }
    }

    fn dsl_config(
        &self,
        strategy: TraversalStrategy,
        parallel: bool,
        first: bool,
    ) -> rxgraph::TraversalConfig {
        let kernel = DslKernel::new(
            e::edge("allowed").and(
                e::state("spent")
                    .plus(e::edge("cost"))
                    .le(e::uint_lit(self.depth as u64)),
            ),
            [("spent".to_owned(), e::state("spent").plus(e::edge("cost")))],
            e::dest_id().eq(e::uint_lit(self.graph.node_count() as u64 - 1)),
            [("spent".to_owned(), Value::U64(0))],
        );
        TraversalConfigBuilder::new(kernel)
            .with_start_nodes(self.starts.clone())
            .with_max_depth(self.depth)
            .with_max_paths(if first { 1 } else { self.starts.len() })
            .with_max_visits_per_node(1)
            .with_strategy(strategy)
            .with_parallelism(parallel)
            .build()
    }
}

fn spec_usize(spec: &JsonValue, field: &str) -> Result<usize> {
    spec.get(field)
        .and_then(JsonValue::as_u64)
        .map(|value| value as usize)
        .with_context(|| format!("benchmark manifest spec.{field} is missing"))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Outcome {
    paths: Vec<Vec<u64>>,
    spent: Vec<u64>,
    evaluated_edges: usize,
    parallel_edges: usize,
    accepted_edges: usize,
    rejected_edges: usize,
    stopped_paths: usize,
}

fn ids(nodes: &[GraphId<'_>]) -> Result<Vec<u64>> {
    nodes
        .iter()
        .map(|id| match id {
            GraphId::U64(id) => Ok(*id),
            GraphId::Str(_) => bail!("benchmark requires UInt64 node ids"),
        })
        .collect()
}

fn native_outcome(result: rxgraph::SearchResult<'_, SearchState>) -> Result<Outcome> {
    let mut paths = Vec::with_capacity(result.paths.len());
    let mut spent = Vec::with_capacity(result.paths.len());
    for path in result.paths {
        paths.push(ids(&path.nodes)?);
        spent.push(path.state.spent);
    }
    paths.sort_unstable();
    spent.sort_unstable();
    Ok(Outcome {
        paths,
        spent,
        evaluated_edges: result.stats.evaluated_edges,
        parallel_edges: result.stats.parallel_edges,
        accepted_edges: result.stats.accepted_edges,
        rejected_edges: result.stats.rejected_edges,
        stopped_paths: result.stats.stopped_paths,
    })
}

fn dsl_outcome(result: rxgraph::SearchResult<'_>) -> Result<Outcome> {
    let mut paths = Vec::with_capacity(result.paths.len());
    let mut spent = Vec::with_capacity(result.paths.len());
    for path in result.paths {
        paths.push(ids(&path.nodes)?);
        let value = path
            .state
            .iter()
            .find_map(|(name, value)| (name == "spent").then_some(value))
            .context("DSL result is missing spent state")?;
        match value {
            Value::U64(value) => spent.push(*value),
            value => bail!("DSL spent state has unexpected value {value:?}"),
        }
    }
    paths.sort_unstable();
    spent.sort_unstable();
    Ok(Outcome {
        paths,
        spent,
        evaluated_edges: result.stats.evaluated_edges,
        parallel_edges: result.stats.parallel_edges,
        accepted_edges: result.stats.accepted_edges,
        rejected_edges: result.stats.rejected_edges,
        stopped_paths: result.stats.stopped_paths,
    })
}

fn native_first(result: rxgraph::FirstResult<'_, SearchState>) -> Result<Outcome> {
    let mut paths = Vec::new();
    let mut spent = Vec::new();
    if let Some(path) = result.path {
        paths.push(ids(&path.nodes)?);
        spent.push(path.state.spent);
    }
    Ok(Outcome {
        paths,
        spent,
        evaluated_edges: result.stats.evaluated_edges,
        parallel_edges: result.stats.parallel_edges,
        accepted_edges: result.stats.accepted_edges,
        rejected_edges: result.stats.rejected_edges,
        stopped_paths: result.stats.stopped_paths,
    })
}

fn dsl_first(result: rxgraph::FirstResult<'_>) -> Result<Outcome> {
    let mut paths = Vec::new();
    let mut spent = Vec::new();
    if let Some(path) = result.path {
        paths.push(ids(&path.nodes)?);
        let value = path
            .state
            .iter()
            .find_map(|(name, value)| (name == "spent").then_some(value))
            .context("DSL first result is missing spent state")?;
        match value {
            Value::U64(value) => spent.push(*value),
            value => bail!("DSL spent state has unexpected value {value:?}"),
        }
    }
    Ok(Outcome {
        paths,
        spent,
        evaluated_edges: result.stats.evaluated_edges,
        parallel_edges: result.stats.parallel_edges,
        accepted_edges: result.stats.accepted_edges,
        rejected_edges: result.stats.rejected_edges,
        stopped_paths: result.stats.stopped_paths,
    })
}

#[derive(Serialize)]
struct CaseReport {
    name: String,
    engine: String,
    operation: String,
    execution: String,
    samples_seconds: Vec<f64>,
    result_paths: usize,
    evaluated_edges: usize,
    parallel_edges: usize,
}

#[derive(Serialize)]
struct CoreReport {
    schema_version: u32,
    graph: GraphReport,
    rayon_threads: usize,
    cases: Vec<CaseReport>,
}

#[derive(Serialize)]
struct GraphReport {
    nodes: usize,
    edges: usize,
}

struct Case<'a> {
    name: &'a str,
    engine: &'a str,
    operation: &'a str,
    parallel: bool,
    run: Box<dyn Fn() -> Result<Outcome> + 'a>,
}

fn checked_case(
    case: Case<'_>,
    expected: &Outcome,
    warmups: usize,
    runs: usize,
) -> Result<CaseReport> {
    let preflight = (case.run)()?;
    validate_outcome(case.name, &preflight)?;
    if preflight.paths != expected.paths || preflight.spent != expected.spent {
        bail!("{} returned different paths or final states", case.name);
    }
    if case.parallel && preflight.parallel_edges == 0 {
        bail!("{} silently fell back to serial execution", case.name);
    }
    if !case.parallel && preflight.parallel_edges != 0 {
        bail!("{} reported parallel work in serial execution", case.name);
    }
    for _ in 0..warmups {
        let outcome = (case.run)()?;
        validate_outcome(case.name, &outcome)?;
        if outcome.paths != expected.paths || outcome.spent != expected.spent {
            bail!("{} changed result during warmup", case.name);
        }
    }
    let mut samples_seconds = Vec::with_capacity(runs);
    let mut last = preflight;
    for _ in 0..runs {
        let started = Instant::now();
        let outcome = (case.run)()?;
        samples_seconds.push(started.elapsed().as_secs_f64());
        validate_outcome(case.name, &outcome)?;
        if outcome.paths != expected.paths || outcome.spent != expected.spent {
            bail!("{} changed result while measuring", case.name);
        }
        last = outcome;
    }
    Ok(CaseReport {
        name: case.name.to_owned(),
        engine: case.engine.to_owned(),
        operation: case.operation.to_owned(),
        execution: if case.parallel { "parallel" } else { "serial" }.to_owned(),
        samples_seconds,
        result_paths: last.paths.len(),
        evaluated_edges: last.evaluated_edges,
        parallel_edges: last.parallel_edges,
    })
}

fn validate_outcome(name: &str, outcome: &Outcome) -> Result<()> {
    if outcome.accepted_edges + outcome.rejected_edges != outcome.evaluated_edges {
        bail!("{name} reported inconsistent evaluated-edge counters");
    }
    if outcome.parallel_edges > outcome.evaluated_edges {
        bail!("{name} reported more parallel edges than evaluated edges");
    }
    if outcome.stopped_paths != outcome.paths.len() {
        bail!("{name} reported inconsistent stopped-path counters");
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    if rayon::current_num_threads() < 2 {
        bail!("core benchmark requires at least two Rayon workers");
    }
    let workload = Workload::load(&args.manifest)?;
    let metadata: JsonValue = serde_json::from_slice(&fs::read(&args.manifest)?)?;
    let spec = metadata
        .get("spec")
        .context("benchmark manifest has no spec")?;
    let warmups = spec_usize(spec, "warmups")?;
    let runs = spec_usize(spec, "runs")?;
    // Quick is a setup/correctness smoke profile. Its 128-start frontier is
    // deliberately below the engine's parallel threshold, so do not publish
    // misleading rows that requested Rayon but ran serially.
    let run_parallel = spec.get("profile").and_then(JsonValue::as_str) != Some("quick");
    let expected = native_outcome(workload.graph.search_paths_with(
        workload.bound.clone(),
        workload.run(TraversalStrategy::DepthFirst, false, false),
    )?)?;
    if expected.paths.len() != workload.starts.len() {
        bail!(
            "preflight returned {} paths, expected {}",
            expected.paths.len(),
            workload.starts.len()
        );
    }
    let expected_first = native_first(workload.graph.search_first_with(
        workload.bound.clone(),
        workload.run(TraversalStrategy::BreadthFirst, false, true),
    )?)?;

    let mut cases = Vec::new();
    macro_rules! add {
        ($name:literal, $engine:literal, $operation:literal, $parallel:expr, $expected:expr, $run:expr) => {
            if args.matches($name) {
                cases.push(checked_case(
                    Case {
                        name: $name,
                        engine: $engine,
                        operation: $operation,
                        parallel: $parallel,
                        run: Box::new($run),
                    },
                    $expected,
                    warmups,
                    runs,
                )?);
            }
        };
    }

    for (label, strategy) in [
        ("dfs", TraversalStrategy::DepthFirst),
        ("bfs", TraversalStrategy::BreadthFirst),
    ] {
        for (execution, parallel) in [("serial", false), ("parallel", true)] {
            if parallel && !run_parallel {
                continue;
            }
            let dsl_name = format!("dsl/{label}/{execution}");
            let bound_name = format!("native-bound/{label}/{execution}");
            if args.matches(&dsl_name) {
                cases.push(checked_case(
                    Case {
                        name: &dsl_name,
                        engine: "dsl",
                        operation: label,
                        parallel,
                        run: Box::new(|| {
                            dsl_outcome(
                                workload
                                    .graph
                                    .search(workload.dsl_config(strategy, parallel, false))?,
                            )
                        }),
                    },
                    &expected,
                    warmups,
                    runs,
                )?);
            }
            if args.matches(&bound_name) {
                cases.push(checked_case(
                    Case {
                        name: &bound_name,
                        engine: "native-bound",
                        operation: label,
                        parallel,
                        run: Box::new(|| {
                            native_outcome(workload.graph.search_paths_with(
                                workload.bound.clone(),
                                workload.run(strategy, parallel, false),
                            )?)
                        }),
                    },
                    &expected,
                    warmups,
                    runs,
                )?);
            }
        }
    }
    add!(
        "dsl/search_first/serial",
        "dsl",
        "search_first",
        false,
        &expected_first,
        || dsl_first(workload.graph.search_first(workload.dsl_config(
            TraversalStrategy::BreadthFirst,
            false,
            true
        ))?)
    );
    add!(
        "native-bound/search_first/serial",
        "native-bound",
        "search_first",
        false,
        &expected_first,
        || native_first(workload.graph.search_first_with(
            workload.bound.clone(),
            workload.run(TraversalStrategy::BreadthFirst, false, true)
        )?)
    );
    if cases.is_empty() {
        bail!("filter did not select any core benchmark case");
    }
    let report = CoreReport {
        schema_version: 1,
        graph: GraphReport {
            nodes: workload.graph.node_count(),
            edges: workload.graph.edge_count(),
        },
        rayon_threads: rayon::current_num_threads(),
        cases,
    };
    fs::write(&args.output, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", args.output.display()))?;
    Ok(())
}

struct Args {
    manifest: PathBuf,
    output: PathBuf,
    filter: Option<String>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut values = env::args().skip(1);
        let mut manifest = None;
        let mut output = None;
        let mut filter = None;
        while let Some(arg) = values.next() {
            match arg.as_str() {
                "--manifest" => manifest = values.next().map(PathBuf::from),
                "--output" => output = values.next().map(PathBuf::from),
                "--filter" => filter = values.next(),
                "--bench" => {}
                _ => bail!("unknown argument {arg}"),
            }
        }
        Ok(Self {
            manifest: manifest.context("missing --manifest")?,
            output: output.context("missing --output")?,
            filter,
        })
    }

    fn matches(&self, name: &str) -> bool {
        self.filter
            .as_ref()
            .is_none_or(|filter| name.contains(filter))
    }
}
