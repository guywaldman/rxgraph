//! Benchmark-only kernels: compare Arrow access, typed access and compound state.
use anyhow::{Context, Result};
use pyo3::prelude::*;
use rxgraph::{traversal::native, *};

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
        Ok(self.step(cx.state(), cx.edge_u64("cost")?.unwrap(), cx.dest_id()))
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
}
rxgraph::inventory::submit! {rxgraph::KernelEntry {name:"bench_arrow",make:|p|Ok(rxgraph::boxed_run(Budget::from_params(p)?))}}
rxgraph::inventory::submit! {rxgraph::TypedKernelEntry {name:"bench_typed",make:|p|Ok(rxgraph::boxed_typed_run(Budget::from_params(p)?))}}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    rxgraph::register(m)
}
