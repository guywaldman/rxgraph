//! Benchmark-only kernels: compare Arrow access, typed access and compound state.
use anyhow::{Context, Result};
use pyo3::prelude::*;
use rxgraph::{traversal::native, *};
use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};

static EXACT_TRANSITIONS: AtomicUsize = AtomicUsize::new(0);

#[pyfunction]
fn reset_exact_transitions() {
    EXACT_TRANSITIONS.store(0, Ordering::Relaxed);
}
#[pyfunction]
fn exact_transitions() -> usize {
    EXACT_TRANSITIONS.load(Ordering::Relaxed)
}

#[derive(Clone, serde::Serialize)]
struct Details {
    label: String,
    values: Vec<u64>,
}
#[derive(Clone, serde::Serialize)]
struct State {
    spent: u64,
    details: Details,
}
#[derive(Clone)]
struct Budget {
    budget: u64,
    target: u64,
    large: bool,
}
impl Budget {
    fn from_params(p: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            budget: p["budget"].as_u64().context("budget")?,
            target: p["target"].as_u64().context("target")?,
            large: p["large"].as_bool().unwrap_or(false),
        })
    }
    fn initial(&self) -> State {
        State {
            spent: 0,
            details: Details {
                label: "native".into(),
                values: if self.large { vec![7; 256] } else { vec![] },
            },
        }
    }
    fn step(&self, old: &State, cost: u64, target: Option<GraphId<'_>>) -> Transition<State> {
        let spent = old.spent + cost;
        if spent > self.budget {
            return Transition::Reject;
        }
        let mut state = old.clone();
        state.spent = spent;
        if target == Some(GraphId::U64(self.target)) {
            Transition::Complete(state)
        } else {
            Transition::Continue(state)
        }
    }
}
impl Kernel for Budget {
    type State = State;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<State> {
        Ok(self.initial())
    }
    fn transition(&self, cx: &EdgeCtx<'_, State>) -> Result<Transition<State>> {
        EXACT_TRANSITIONS.fetch_add(1, Ordering::Relaxed);
        Ok(self.step(cx.state(), cx.edge_u64("cost")?.unwrap(), cx.dest_id()))
    }
    fn possible_targets<'a>(
        &'a self,
        cx: &TargetCtx<'_, State>,
    ) -> Result<Option<Cow<'a, [NodeId]>>> {
        let graph = match cx {
            TargetCtx::Start { graph, .. } => graph,
            TargetCtx::Edge { edge, .. } => edge.graph(),
        };
        let target = graph.internal_node_id(self.target).context("target node")?;
        Ok(Some(Cow::Owned(vec![target])))
    }
}
impl MemoizableKernel for Budget {
    type Key = u64;
    fn state_key(&self, state: &State) -> u64 {
        state.spent
    }
}
#[derive(Clone)]
struct Cost(u64);
impl TryFrom<ArrowRow<'_>> for Cost {
    type Error = anyhow::Error;
    fn try_from(row: ArrowRow<'_>) -> Result<Self> {
        Ok(Self(row.u64("cost")?.context("cost")?))
    }
}
impl TypedKernel for Budget {
    type Node = ();
    type Edge = Cost;
    type State = State;
    fn edge_fields(&self) -> Vec<PayloadField> {
        vec![PayloadField::new("cost")]
    }
    fn initial_state(&self, _: &native::StartCtx<'_, (), Cost>) -> Result<State> {
        Ok(self.initial())
    }
    fn transition(
        &self,
        cx: &native::EdgeCtx<'_, '_, (), Cost, State>,
    ) -> Result<Transition<State>> {
        Ok(self.step(cx.state(), cx.edge()?.0, cx.dest_external_id()?))
    }
    fn possible_targets<'a>(
        &'a self,
        cx: &native::TargetCtx<'a, (), Cost, State>,
    ) -> Result<Option<Cow<'a, [NodeId]>>> {
        let target = match cx {
            native::TargetCtx::Start { start, .. } => {
                start.resolve_node(GraphId::U64(self.target))?
            }
            native::TargetCtx::Edge { edge, .. } => edge.resolve_node(GraphId::U64(self.target))?,
        }
        .context("target node")?;
        Ok(Some(Cow::Owned(vec![target])))
    }
}
rxgraph::inventory::submit! {rxgraph::KernelEntry {name:"bench_arrow",make:|p|Ok(rxgraph::boxed_run(Budget::from_params(p)?))}}
rxgraph::inventory::submit! {rxgraph::KernelEntry {name:"bench_memo",make:|p|Ok(rxgraph::boxed_memoizable_run(Budget::from_params(p)?))}}
rxgraph::inventory::submit! {rxgraph::TypedKernelEntry {name:"bench_typed",make:|p|Ok(rxgraph::boxed_typed_run(Budget::from_params(p)?))}}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    rxgraph::register(m)?;
    m.add_function(wrap_pyfunction!(reset_exact_transitions, m)?)?;
    m.add_function(wrap_pyfunction!(exact_transitions, m)?)?;
    Ok(())
}

struct LegacyRunner(BoxedRun);
impl RunKernel for LegacyRunner {
    fn run<'g>(&self, graph: &'g Graph, run: RunOptions) -> Result<SearchResult<'g>> {
        self.0.run(graph, run)
    }
}
rxgraph::inventory::submit! {rxgraph::KernelEntry {name:"bench_legacy",make:|p|Ok(Box::new(LegacyRunner(rxgraph::boxed_run(Budget::from_params(p)?))))}}
