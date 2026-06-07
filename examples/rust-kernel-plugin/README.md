# Rust kernel plugin example

## What this is

A standalone Python extension package that exposes a native Rust search kernel
through the normal `rxgraph` Python API.

The example implements `hop_budget`:

- carry one state field: `hops`
- accept edges while `hops < max_hops`
- stop when the destination node has `target == true`, or when the hop budget is used

## Rust shape

Implement `rxgraph::Kernel`, parse runtime params, then call
`rxgraph::plugin!` once:

```rust
use anyhow::{Context, Result};
use rxgraph::{EdgeCtx, Graph, Kernel, NodeId, StateRow, Value};

#[derive(Clone, Copy, Debug, Default)]
pub struct HopState {
    hops: u64,
}

#[derive(Clone, Debug)]
pub struct HopBudget {
    pub max_hops: u64,
    pub target_col: String,
}

impl HopBudget {
    pub fn from_params(params: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            max_hops: params
                .get("max_hops")
                .and_then(serde_json::Value::as_u64)
                .context("param `max_hops` (u64) is required")?,
            target_col: params
                .get("target_col")
                .and_then(serde_json::Value::as_str)
                .context("param `target_col` (string) is required")?
                .to_string(),
        })
    }
}

impl Kernel for HopBudget {
    type State = HopState;

    fn initial_state(&self, _graph: &Graph, _start: NodeId) -> Self::State {
        HopState::default()
    }

    fn visit(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<bool> {
        Ok(cx.state().hops < self.max_hops)
    }

    fn next_state(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Self::State> {
        Ok(HopState {
            hops: cx.state().hops + 1,
        })
    }

    fn stop(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<bool> {
        Ok(cx.state().hops >= self.max_hops
            || cx.dest_bool(&self.target_col)?.unwrap_or(false))
    }

    fn state_row(&self, state: &Self::State) -> StateRow {
        vec![("hops".to_string(), Value::U64(state.hops))]
    }
}

rxgraph::plugin! {
    module = _native;
    "hop_budget" => HopBudget::from_params,
}
```

The macro registers the named kernel and emits the PyO3 `_native` module. The
search hot path still runs through `Graph::search_with` monomorphized over the
concrete kernel; the named lookup adds one boxed call per search, not per edge.

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
import rxgraph_hop_budget as rxg

graph = rxg.Graph.from_edges(
    [("a", "b"), ("b", "c"), ("c", "d")],
    nodes=[
        ("a", {"target": False}),
        ("b", {"target": False}),
        ("c", {"target": True}),
        ("d", {"target": False}),
    ],
)

result = graph.search(
    start_nodes=["a"],
    kernel="hop_budget",
    params={"max_hops": 3, "target_col": "target"},
    max_paths=10,
    parallel=False,
)
```

NOTE: the plugin wheel links its own copy of the `rxgraph` engine, so the
plugin's `rxgraph` and Python wrapper versions should match.

## Layout

- `src/lib.rs` - kernel implementation, tests, and `rxgraph::plugin!`.
- `python/rxgraph_hop_budget/` - tiny Python package that calls `export_api`.
- `pyproject.toml` - maturin package config.
- `example.py` - runnable end-to-end demo.

## Checks

From the repo root:

```bash
cargo test --manifest-path examples/rust-kernel-plugin/Cargo.toml --locked
just test-kernel-plugin-example
```

The `just` recipe builds `rxgraph`, builds the plugin wheel for the repo venv's
Python interpreter, installs that exact wheel, runs `example.py`, and runs the
plugin pytest coverage.

## `EdgeCtx` accessor reference

A kernel's `visit`/`next_state`/`stop` receive an `EdgeCtx<'_, State>` for the
candidate edge `(src)-[edge]->(dest)`. Source of truth:
`crates/rxgraph/src/traversal/kernel.rs`.

| Accessor | Returns | Notes |
| --- | --- | --- |
| `state()` | `&State` | Per-path state. Parent state in `visit`/`next_state`; child state in `stop`. |
| `graph()` | `&Graph` | The graph being traversed. |
| `src()` / `dest()` / `edge()` | `NodeId` / `NodeId` / `EdgeId` | Internal row ids. |
| `src_id()` / `dest_id()` / `edge_id()` | `Option<GraphId>` | External ids, if present. |
| `src_value(col)` / `dest_value(col)` / `edge_value(col)` | `Result<Value>` | Raw payload value. |
| `src_u64(col)` / `dest_u64(col)` / `edge_u64(col)` | `Result<Option<u64>>` | Lossless numeric coercion; `None` if null. |
| `src_i64(col)` / `dest_i64(col)` / `edge_i64(col)` | `Result<Option<i64>>` | Lossless numeric coercion; `None` if null. |
| `src_f64(col)` / `dest_f64(col)` / `edge_f64(col)` | `Result<Option<f64>>` | Widening numeric coercion; `None` if null. |
| `src_bool(col)` / `dest_bool(col)` / `edge_bool(col)` | `Result<Option<bool>>` | `None` if null; errors on non-bool. |
| `src_str(col)` / `dest_str(col)` / `edge_str(col)` | `Result<Option<String>>` | `None` if null; errors on non-string. |

Typed getters read through a per-search payload cache, so repeated reads of the
same column do not re-downcast the underlying Arrow array on every edge.
