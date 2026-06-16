# Rust kernel plugin example

## What this is

A standalone Python extension package that exposes a native Rust search kernel
through the normal `rxgraph` Python API.

The example implements `hop_budget`:

- carry one state field: `hops`
- decode node `profile` structs into Rust node payloads
- decode edge `policy` structs with `hop_costs` lists into Rust edge payloads
- stop when the destination node has `profile.target == true`, or when the hop budget is used

## Rust shape

Implement `rxgraph::TypedKernel`, declare the payload columns it needs, decode
those projected Arrow rows into native structs with `TryFrom<ArrowRow<'_>>`, then call
`rxgraph::typed_plugin!` once:

```rust
use anyhow::{Context, Result};
use rxgraph::{ArrowRow, ArrowStruct, PayloadField, StateRow, TypedKernel, Value, traversal::native};

#[derive(Clone, Copy, Debug, Default)]
pub struct HopState {
    hops: u64,
}

#[derive(Clone, Debug)]
pub struct HopBudget {
    pub max_hops: u64,
    pub profile_col: String,
    pub policy_col: String,
}

#[derive(Clone, Debug)]
pub struct HopNode {
    target: bool,
}

#[derive(Clone, Debug, Default)]
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

#[derive(Clone, Debug)]
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
        let policy = row.struct_("policy")?.context("edge policy struct is required")?;
        let hop_costs = policy.list_items("hop_costs")?.unwrap();
        Ok(Self {
            enabled: policy.bool("enabled")?.unwrap_or(false),
            hop_costs: (0..hop_costs.len())
                .map(|i| hop_costs.u64(i)?.context("hop cost cannot be null"))
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl HopBudget {
    pub fn from_params(params: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            max_hops: params
                .get("max_hops")
                .and_then(serde_json::Value::as_u64)
                .context("param `max_hops` (u64) is required")?,
            profile_col: params
                .get("profile_col")
                .and_then(serde_json::Value::as_str)
                .context("param `profile_col` (string) is required")?
                .to_string(),
            policy_col: params
                .get("policy_col")
                .and_then(serde_json::Value::as_str)
                .context("param `policy_col` (string) is required")?
                .to_string(),
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
        let edge = cx.edge()?;
        Ok(edge.enabled && cx.state().hops.saturating_add(edge.cost()) <= self.max_hops)
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
        Ok(cx.state().hops >= self.max_hops || cx.dest()?.target)
    }

    fn state_row(&self, state: &Self::State) -> StateRow {
        vec![("hops".to_string(), Value::U64(state.hops))]
    }
}

rxgraph::typed_plugin! {
    module = _native;
    "hop_budget" => HopBudget::from_params,
}
```

The macro registers the named kernel and emits the PyO3 `_native` module. The
search hot path runs in Rust over native payload structs; the named lookup adds
one boxed call per search, not per edge.

## Python shape

The Python package binds the reusable `rxgraph` wrapper to this extension's
native backend:

```python
from . import _native
from rxgraph.plugin import export_api

export_api(globals(), _native)
```

Users import the plugin package as their graph API:

```python
from pathlib import Path

import polars as pl
import rxgraph_hop_budget as rxg

nodes = Path("nodes.parquet")
edges = Path("edges.parquet")
pl.DataFrame(
    {
        "id": ["a", "b", "c"],
        "profile": [{"target": False}, {"target": False}, {"target": True}],
    }
).write_parquet(nodes)
pl.DataFrame(
    {
        "id": ["ab", "bc"],
        "src": ["a", "b"],
        "dest": ["b", "c"],
        "policy": [
            {"enabled": True, "hop_costs": [1]},
            {"enabled": True, "hop_costs": [1, 1]},
        ],
    }
).write_parquet(edges)

graph = rxg.Graph.from_parquet(nodes, edges, payloads="lazy")

result = graph.search(
    start_nodes=["a"],
    kernel="hop_budget",
    params={"max_hops": 10, "profile_col": "profile", "policy_col": "policy"},
    max_paths=10,
    parallel=False,
)
```

NOTE: the plugin wheel links its own copy of the `rxgraph` engine, so the
plugin's `rxgraph` and Python wrapper versions should match.

## Layout

- `src/lib.rs` - typed kernel implementation, tests, and `rxgraph::typed_plugin!`.
- `python/rxgraph_hop_budget/` - tiny Python package that calls `export_api`.
- `pyproject.toml` - maturin package config.
- `example.py` - runnable end-to-end demo.

## Checks

From the repo root:

```bash
cargo test --manifest-path examples/rust-kernel-plugin/Cargo.toml --offline
just test-kernel-plugin-example
```

The `just` recipe builds `rxgraph`, builds the plugin wheel for the repo venv's
Python interpreter, installs that exact wheel, runs `example.py`, and runs the
plugin pytest coverage.

## Native context

A typed kernel's `visit`/`next_state`/`stop` receive a
`native::EdgeCtx<'_, '_, Node, Edge, State>` for the candidate edge
`(src)-[edge]->(dest)`.

| Accessor | Returns | Notes |
| --- | --- | --- |
| `state()` | `&State` | Parent state in `visit`/`next_state`; child state in `stop`. |
| `src_id()` / `dest_id()` / `edge_id()` | `NodeId` / `NodeId` / `EdgeId` | Internal row ids. |
| `src_external_id()` / `dest_external_id()` / `edge_external_id()` | `Result<Option<GraphId>>` | External ids, if present. |
| `src()` / `dest()` / `edge()` | `Result<&Node>` / `Result<&Node>` / `Result<&Edge>` | Native structs decoded via `TryFrom<ArrowRow<'_>>`. |
