//! Example native traversal kernel for `rxgraph`.
//!
//! This crate implements [`rxgraph::TypedKernel`] and exposes it through a
//! generated Python extension module. The kernel is selectable with
//! `graph.search(kernel="hop_budget", params={...})`.
//!
//! The kernel implemented here is [`HopBudget`]: starting from a node, walk the
//! graph over native Rust payload structs. Node payloads contain a `profile`
//! struct (`target`); edge payloads contain a `policy` struct (`enabled`,
//! `hop_costs`). The kernel emits a path once it reaches a target node, or once
//! the hop budget is used.

use anyhow::{Context, Result};
use rxgraph::{
    ArrowRow, ArrowStruct, PayloadField, StateRow, TypedKernel, Value, traversal::native,
};

/// Per-path state carried by [`HopBudget`].
///
/// Kept deliberately tiny and `Copy`; the engine clones state for every child
/// path, so cheap state keeps search fast.
#[derive(Clone, Copy, Debug, Default)]
pub struct HopState {
    /// Number of edges accepted so far on this path.
    hops: u64,
}

/// Stop after `max_hops` weighted hops, or earlier at a target node.
///
/// `profile_col` names a node struct column with fields:
/// - `target: bool`
///
/// `policy_col` names an edge struct column with fields:
/// - `enabled: bool`
/// - `hop_costs: list[int]`
#[derive(Clone, Debug)]
pub struct HopBudget {
    /// Maximum number of edges any emitted path may contain.
    pub max_hops: u64,
    /// Name of the node struct column that marks targets.
    pub profile_col: String,
    /// Name of the edge struct column that controls edge acceptance/cost.
    pub policy_col: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HopNode {
    target: bool,
}

impl HopNode {
    fn is_goal(&self) -> bool {
        self.target
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeProfile {
    target: bool,
}

impl TryFrom<ArrowStruct<'_>> for NodeProfile {
    type Error = anyhow::Error;

    fn try_from(row: ArrowStruct<'_>) -> Result<Self> {
        Ok(Self {
            target: row.bool("target")?.unwrap_or(false),
        })
    }
}

impl TryFrom<ArrowRow<'_>> for HopNode {
    type Error = anyhow::Error;

    fn try_from(row: ArrowRow<'_>) -> Result<Self> {
        let profile = row.struct_as::<NodeProfile>("profile")?.unwrap_or_default();
        Ok(Self {
            target: profile.target,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HopEdge {
    enabled: bool,
    hop_costs: Vec<u64>,
}

impl HopEdge {
    fn cost(&self) -> u64 {
        self.hop_costs.iter().sum()
    }
}

impl TryFrom<ArrowRow<'_>> for HopEdge {
    type Error = anyhow::Error;

    fn try_from(row: ArrowRow<'_>) -> Result<Self> {
        let policy = row
            .struct_("policy")?
            .context("edge policy struct is required")?;
        Ok(Self {
            enabled: policy.bool("enabled")?.unwrap_or(false),
            hop_costs: read_u64_list(policy.list_items("hop_costs")?)?,
        })
    }
}

impl HopBudget {
    /// Builds a [`HopBudget`] from a JSON params object.
    ///
    /// Expected shape:
    /// `{ "max_hops": <u64>, "profile_col": "<column>", "policy_col": "<column>" }`.
    pub fn from_params(params: &serde_json::Value) -> Result<Self> {
        let max_hops = params
            .get("max_hops")
            .and_then(serde_json::Value::as_u64)
            .context("param `max_hops` (u64) is required")?;
        let profile_col = params
            .get("profile_col")
            .and_then(serde_json::Value::as_str)
            .context("param `profile_col` (string) is required")?
            .to_string();
        let policy_col = params
            .get("policy_col")
            .and_then(serde_json::Value::as_str)
            .context("param `policy_col` (string) is required")?
            .to_string();
        Ok(Self {
            max_hops,
            profile_col,
            policy_col,
        })
    }
}

impl TypedKernel for HopBudget {
    type Node = HopNode;
    type Edge = HopEdge;
    type State = HopState;

    fn node_fields(&self) -> Vec<PayloadField> {
        vec![PayloadField::aliased(self.profile_col.clone(), "profile")]
    }

    fn edge_fields(&self) -> Vec<PayloadField> {
        vec![PayloadField::aliased(self.policy_col.clone(), "policy")]
    }

    fn initial_state(
        &self,
        _cx: &native::StartCtx<'_, Self::Node, Self::Edge>,
    ) -> Result<Self::State> {
        Ok(HopState::default())
    }

    fn visit(
        &self,
        cx: &native::EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<bool> {
        // `cx.state()` is the parent path's state. Edge policy is a native
        // struct decoded from Arrow before traversal calls into this kernel.
        let edge = cx.edge()?;
        let next_hops = cx.state().hops.saturating_add(edge.cost());
        Ok(edge.enabled && next_hops <= self.max_hops)
    }

    fn next_state(
        &self,
        cx: &native::EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<Self::State> {
        Ok(HopState {
            hops: cx.state().hops.saturating_add(cx.edge()?.cost()),
        })
    }

    fn stop(
        &self,
        cx: &native::EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<bool> {
        // `cx` here carries the *child* state produced by `next_state`.
        // Emit the path if we reached the hop budget...
        if cx.state().hops >= self.max_hops {
            return Ok(true);
        }
        // ...or if the destination node's `profile` marks it as a goal.
        Ok(cx.dest()?.is_goal())
    }

    fn state_row(&self, state: &Self::State) -> StateRow {
        vec![("hops".to_string(), Value::U64(state.hops))]
    }
}

fn read_u64_list(items: Option<rxgraph::ArrowList>) -> Result<Vec<u64>> {
    let Some(items) = items else {
        return Ok(Vec::new());
    };
    (0..items.len())
        .map(|index| {
            items
                .u64(index)?
                .with_context(|| format!("hop_costs[{index}] cannot be null"))
        })
        .collect()
}

rxgraph::typed_plugin! {
    module = _native;
    "hop_budget" => HopBudget::from_params,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{
            Array, ArrayRef, BooleanArray, ListArray, RecordBatch, StringArray, StructArray,
        },
        datatypes::{DataType, Field, Int64Type, Schema},
    };
    use rxgraph::{Graph, RunOptions};

    // Required graph identity columns: nodes need `id`; edges need `id`, `src`,
    // `dest`. (These names are part of the public data model.)
    /// a -> b -> c -> d, with `profile.target` true only at node `c`.
    fn line_graph() -> Graph {
        let profile = StructArray::from(vec![(
            Arc::new(Field::new("target", DataType::Boolean, true)),
            Arc::new(BooleanArray::from(vec![
                Some(false),
                Some(false),
                Some(true),
                Some(false),
            ])) as ArrayRef,
        )]);
        let nodes = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("profile", profile.data_type().clone(), true),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
                Arc::new(profile),
            ],
        )
        .unwrap();

        let hop_costs = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(1)]),
            Some(vec![Some(1), Some(1)]),
            Some(vec![Some(1)]),
        ]);
        let policy = StructArray::from(vec![
            (
                Arc::new(Field::new("enabled", DataType::Boolean, true)),
                Arc::new(BooleanArray::from(vec![Some(true), Some(true), Some(true)]))
                    as ArrayRef,
            ),
            (
                Arc::new(Field::new("hop_costs", hop_costs.data_type().clone(), true)),
                Arc::new(hop_costs) as ArrayRef,
            ),
        ]);
        let edges = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("src", DataType::Utf8, false),
                Field::new("dest", DataType::Utf8, false),
                Field::new("policy", policy.data_type().clone(), true),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["ab", "bc", "cd"])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(StringArray::from(vec!["b", "c", "d"])),
                Arc::new(policy),
            ],
        )
        .unwrap();

        Graph::new(nodes, edges).unwrap()
    }

    fn run(graph: &Graph, params: serde_json::Value) -> rxgraph::OwnedSearchResult {
        // Resolve the kernel by name exactly like the Python selector does.
        let kernel = rxgraph::build_typed_kernel("hop_budget", &params).unwrap();
        kernel
            .run_eager(
                graph,
                RunOptions {
                    start_nodes: vec!["a".into()],
                    parallel: false,
                    ..RunOptions::default()
                },
            )
            .unwrap()
    }

    #[test]
    fn registered_under_its_name() {
        // build_typed_kernel succeeds => the macro registered the kernel by name.
        assert!(rxgraph::build_typed_kernel("hop_budget", &serde_json::json!({})).is_err());
        assert!(
            rxgraph::build_typed_kernel(
                "hop_budget",
                &serde_json::json!({
                    "max_hops": 3,
                    "profile_col": "profile",
                    "policy_col": "policy",
                })
            )
            .is_ok()
        );
    }

    #[test]
    fn stops_at_target_node_before_hop_budget() {
        let graph = line_graph();
        let result = run(
            &graph,
            serde_json::json!({
                "max_hops": 10,
                "profile_col": "profile",
                "policy_col": "policy",
            }),
        );

        // First emitted path ends at target node `c` after weighted hop cost 3.
        assert_eq!(result.paths.len(), 1);
        let path = &result.paths[0];
        assert_eq!(path.nodes, vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(path.edges, vec!["ab".into(), "bc".into()]);
        assert_eq!(
            path.state,
            vec![("hops".to_string(), rxgraph::Value::U64(3))]
        );
    }

    #[test]
    fn stops_at_hop_budget_when_no_target_reached() {
        let graph = line_graph();
        // Budget of 1 hop is exhausted at node `b`, before reaching target `c`.
        let result = run(
            &graph,
            serde_json::json!({
                "max_hops": 1,
                "profile_col": "profile",
                "policy_col": "policy",
            }),
        );

        assert_eq!(result.paths.len(), 1);
        let path = &result.paths[0];
        assert_eq!(path.nodes, vec!["a".into(), "b".into()]);
        assert_eq!(path.edges, vec!["ab".into()]);
        assert_eq!(
            path.state,
            vec![("hops".to_string(), rxgraph::Value::U64(1))]
        );
    }
}
