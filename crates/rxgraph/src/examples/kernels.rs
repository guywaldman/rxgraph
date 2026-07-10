//! Example traversal kernel using a Rust kernel as a plugin.
//!
//! [`WeightedBudget`] accumulates an edge weight along a path and accepts an
//! edge only while the running total stays within a budget, stopping once a
//! target node is reached. It is registered under the name `"weighted_budget"`
//! so it can also be built by name through [`build_kernel`](crate::build_kernel).

use anyhow::{Result, anyhow};

use crate::{
    graph::{Graph, NodeId, OwnedGraphId},
    traversal::{ArrowRow, EdgeCtx, Kernel, PayloadField, Transition, TypedKernel, native},
};

/// Per-path state for [`WeightedBudget`]: the weight accumulated so far.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct BudgetState {
    /// Total edge weight spent along the current path.
    pub spent: u64,
}

/// A budgeted-weight kernel.
///
/// Traversal accumulates `weight_col` along each path. An edge is accepted only
/// while `spent + weight <= budget`, and a path stops once it reaches `target`.
///
/// A null/missing weight rejects the edge: a missing cost is treated as
/// "cannot price this hop", which is safer for a budget than silently charging
/// zero.
#[derive(Clone, Debug)]
pub struct WeightedBudget {
    /// Edge payload column holding the `u64` weight.
    pub weight_col: String,
    /// Maximum total weight a returned path may spend.
    pub budget: u64,
    /// Destination node that ends a path.
    pub target: OwnedGraphId,
}

impl Kernel for WeightedBudget {
    type State = BudgetState;

    fn initial_state(&self, _graph: &Graph, _start: NodeId) -> Result<Self::State> {
        Ok(BudgetState { spent: 0 })
    }

    fn transition(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Transition<Self::State>> {
        // Null/missing weight rejects the edge (see type docs).
        let Some(weight) = cx.edge_u64(&self.weight_col)? else {
            return Ok(Transition::Reject);
        };
        let state = BudgetState {
            spent: cx.state().spent.saturating_add(weight),
        };
        Ok(if state.spent > self.budget {
            Transition::Reject
        } else if cx.dest_id() == Some(self.target.as_ref()) {
            Transition::Complete(state)
        } else {
            Transition::Continue(state)
        })
    }
}

impl WeightedBudget {
    /// Builds a [`WeightedBudget`] from JSON params.
    ///
    /// Accepts something like:
    /// ```json
    /// { "weight_col": "cost", "budget": 10, "target": 3 }
    /// ```
    fn from_params(params: &serde_json::Value) -> Result<Self> {
        let weight_col = params
            .get("weight_col")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("weighted_budget: missing string param 'weight_col'"))?
            .to_string();

        let budget = params
            .get("budget")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| anyhow!("weighted_budget: missing u64 param 'budget'"))?;

        let target = match params.get("target") {
            Some(serde_json::Value::Number(n)) => n
                .as_u64()
                .map(OwnedGraphId::U64)
                .ok_or_else(|| anyhow!("weighted_budget: 'target' number must be a u64"))?,
            Some(serde_json::Value::String(s)) => OwnedGraphId::Str(s.clone()),
            _ => {
                return Err(anyhow!(
                    "weighted_budget: 'target' must be a u64 or string node id"
                ));
            }
        };

        Ok(Self {
            weight_col,
            budget,
            target,
        })
    }
}

// Link-time registration: makes `build_kernel("weighted_budget", ..)` work.
crate::inventory::submit! {
    crate::KernelEntry {
        name: "weighted_budget",
        make: |params| Ok(crate::boxed_run(WeightedBudget::from_params(params)?)),
    }
}

#[derive(Clone, Debug)]
pub struct WeightedBudgetTyped {
    pub weight_col: String,
    pub budget: u64,
    pub target: OwnedGraphId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetEdge {
    pub weight: u64,
}

impl TryFrom<ArrowRow<'_>> for BudgetEdge {
    type Error = anyhow::Error;

    fn try_from(row: ArrowRow<'_>) -> Result<Self> {
        Ok(Self {
            weight: row.u64("weight")?.unwrap_or(0),
        })
    }
}

impl TypedKernel for WeightedBudgetTyped {
    type Node = ();
    type Edge = BudgetEdge;
    type State = BudgetState;

    fn edge_fields(&self) -> Vec<PayloadField> {
        vec![PayloadField::aliased(self.weight_col.clone(), "weight")]
    }

    fn initial_state(
        &self,
        _cx: &native::StartCtx<'_, Self::Node, Self::Edge>,
    ) -> Result<Self::State> {
        Ok(BudgetState { spent: 0 })
    }

    fn transition(
        &self,
        cx: &native::EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<Transition<Self::State>> {
        let state = BudgetState {
            spent: cx.state().spent.saturating_add(cx.edge()?.weight),
        };
        Ok(if state.spent > self.budget {
            Transition::Reject
        } else if cx.dest_external_id()? == Some(self.target.as_ref()) {
            Transition::Complete(state)
        } else {
            Transition::Continue(state)
        })
    }
}

impl WeightedBudgetTyped {
    fn from_params(params: &serde_json::Value) -> Result<Self> {
        let base = WeightedBudget::from_params(params)?;
        Ok(Self {
            weight_col: base.weight_col,
            budget: base.budget,
            target: base.target,
        })
    }
}

crate::inventory::submit! {
    crate::TypedKernelEntry {
        name: "weighted_budget",
        make: |params| Ok(crate::boxed_typed_run(WeightedBudgetTyped::from_params(params)?)),
    }
}
