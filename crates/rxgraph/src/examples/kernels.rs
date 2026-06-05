//! Example traversal kernels.
//!
//! [`WeightedBudget`] accumulates an edge weight along a path and accepts an
//! edge only while the running total stays within a budget, stopping once a
//! target node is reached. It is registered under the name `"weighted_budget"`
//! so it can also be built by name through [`build_kernel`](crate::build_kernel).

use anyhow::{Result, anyhow};

use crate::{
    dsl::{StateRow, Value},
    graph::{Graph, NodeId, OwnedGraphId},
    traversal::{EdgeCtx, Kernel},
};

/// Per-path state for [`WeightedBudget`]: the weight accumulated so far.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetState {
    /// Total edge weight spent along the current path.
    pub spent: u64,
}

/// A budgeted-weight kernel.
///
/// Traversal accumulates `weight_col` along each path. An edge is accepted only
/// while `spent + weight <= budget`, and a path stops once it reaches `target`.
///
/// A null/missing weight rejects the edge (returns `Ok(false)` from
/// [`visit`](Kernel::visit)): a missing cost is treated as "cannot price this
/// hop", which is safer for a budget than silently charging zero.
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

    fn initial_state(&self, _graph: &Graph, _start: NodeId) -> Self::State {
        BudgetState { spent: 0 }
    }

    fn visit(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<bool> {
        // Null/missing weight rejects the edge (see type docs).
        let Some(weight) = cx.edge_u64(&self.weight_col)? else {
            return Ok(false);
        };
        Ok(cx.state().spent.saturating_add(weight) <= self.budget)
    }

    fn next_state(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Self::State> {
        // `next_state` runs only after `visit` accepted the edge, so the weight
        // is present; default to 0 defensively rather than re-erroring.
        let weight = cx.edge_u64(&self.weight_col)?.unwrap_or(0);
        Ok(BudgetState {
            spent: cx.state().spent.saturating_add(weight),
        })
    }

    fn stop(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<bool> {
        Ok(cx.dest_id() == Some(self.target.as_ref()))
    }

    fn state_row(&self, state: &Self::State) -> StateRow {
        vec![("spent".to_string(), Value::U64(state.spent))]
    }
}

impl WeightedBudget {
    /// Builds a [`WeightedBudget`] from JSON params.
    ///
    /// Expected shape:
    ///
    /// ```json
    /// { "weight_col": "cost", "budget": 10, "target": 3 }
    /// ```
    ///
    /// `target` accepts a JSON number (integer node ID) or a JSON string
    /// (string node ID), matching the graph's ID mode.
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
