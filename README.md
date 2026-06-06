# rxgraph

High-performance graph traversal and graph algorithms for Python, implemented
in Rust with an ergonomic object API and Polars expression support.

`rxgraph` supports:

1. Efficient graph construction leveraging Arrow-backed DataFrames as inputs
1. Optimized stateful search, where the traversal predicates are expressed with Poalrs expressions
1. Common graph algorithms like BFS, DFS, shortest path, weakly connected components, etc.

From initial benchmarks, `rxgraph` is comparable in performance and CPU/memory consumption (and very possibly better in some cases) with `igraph` and `networkx`.

The place where `rxgraph` really shines is **stateful search** - you can use `rxgraph` for stateful blind search across a very large graph.
See the traversal example in [Quickstart](#quick-start).

The main focus of this library is Python and its Python bindings, but its Rust core is also published as a crate to crates.io.

> [!IMPORTANT]
>
> `rxgraph` is under heavy active development - the core implementation (Python bindings & Rust crate) are usable with decent test coverage, but it is not ready for production and you should use it at your own risk. The public API is likely to change as well.
>
> Having said that, the library is usable and should work for most scenarios it supports, and I would love for some initial feedback.

## Installation

```bash
# uv
uv add rxgraph polars

# pip
pip install rxgraph polars
```

> [!NOTE]
>
> Requires Python 3.11+ and currently depends on Polars for expression input.

## Quick Start

```python
import rxgraph as rxg

graph = rxg.Graph.from_edges(
    [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
    nodes=["a", "b", "c", "d", "isolated"],
)

assert graph.node_count == 5
assert graph.edge_count == 4
assert graph.bfs("a") == ["a", "b", "c", "d"]
assert graph.shortest_path("a", "d") == ["a", "b", "d"]
assert graph.reachable_nodes("isolated") == ["isolated"]
```

For the most powerful capability of `rxgraph`, see the [Stateful Search](#stateful-search) example.

## Data Model

For small or Python-native graphs, use `Graph.from_edges` with hashable node
labels:

```python
routes = rxg.Graph.from_edges(
    [
        ("a", "b", {"price": 5, "kind": "route"}),
        ("b", "c", {"price": 6, "kind": "route"}),
        ("a", "c", {"price": 100, "kind": "skip"}),
    ],
    nodes=[
        ("a", {"closed": False}),
        ("b", {"closed": False}),
        ("c", {"closed": False}),
    ],
)
```

For table-backed graphs, pass Polars `DataFrame`s directly:

```python
import polars as pl
import rxgraph as rxg

nodes = pl.DataFrame(
    {"id": [10, 20, 30]},
    schema={"id": pl.UInt64},
)
edges = pl.DataFrame(
    {"id": [1, 2], "src": [10, 20], "dest": [20, 30]},
    schema={"id": pl.UInt64, "src": pl.UInt64, "dest": pl.UInt64},
)

table_graph = rxg.Graph(nodes, edges)
assert table_graph.shortest_path(10, 30) == [10, 20, 30]
```

Node tables require an `id` column. Edge tables require `id`, `src`, and `dest`.
All identity columns must be either unsigned integers or strings. Extra columns
remain available to traversal expressions.

## Algorithms

The high-level object API includes:

- `bfs(start, max_depth=None)`
- `dfs(start, max_depth=None)`
- `reachable_nodes(start)`
- `shortest_path(source, target)`
- `out_degrees()`, `in_degrees()`, and `degrees()`
- `weakly_connected_components()`

These methods return the same labels or IDs used to build the graph.

## Stateful Search

`Graph.search` evaluates Polars expressions against candidate edges. Expressions
can read source-node fields (`src.*`), destination-node fields (`dest.*`), edge
fields (`edge.*`), and path state (`state.*`).

Using the `routes` graph from the data model example:

```python
s = lambda name: rxg.col(f"state.{name}")
d = lambda name: rxg.col(f"dest.{name}")
e = lambda name: rxg.col(f"edge.{name}")

result = routes.search(
    start_nodes=["a"],
    visit=(~d("closed")) & (e("kind") != "skip") & ((s("spent") + e("price")) < 20),
    next_state={"spent": s("spent") + e("price")},
    stop=rxg.col("dest.id") == rxg.lit(routes.node_id("c")),
    initial_state={"spent": 0},
    max_depth=3,
    max_paths=10,
)

path = result.paths[0]
assert path.nodes == ["a", "b", "c"]
assert path.edges == [0, 1]
assert path.state == {"spent": 11}
```

Search supports DFS or BFS ordering, optional Rayon-backed parallel traversal,
depth/path limits, and optional intermediate state materialization.

Search kernels evaluate supported Polars scalar, list, and struct expressions
natively in Rust. List and struct columns/state can be read and updated inside
`visit`, `next_state`, and `stop`. Polars JSON list literals are intentionally
not decoded yet; use list columns or list state for list-valued operands.

See [`examples/nyc_taxi_zone_search.py`](examples/nyc_taxi_zone_search.py)
for a public NYC TLC trip-record example that uses list and struct state over
millions of raw trip edges.

## Native Rust kernels

The Polars expression DSL covers most stateful searches with no compilation
step. When you need arbitrary state types, logic the DSL cannot express, or the
last bit of performance, you can supply a **native Rust kernel** instead.

A kernel implements the `rxgraph::Kernel` trait - the same three per-edge
decisions the DSL makes (`visit`, `next_state`, `stop`) plus its own per-path
`State` type. The engine is monomorphized over your kernel, so per-edge calls
are statically dispatched; there is no per-edge virtual dispatch. You register a
kernel under a name with `inventory::submit!` and then select it by that name.

```rust
use rxgraph::{EdgeCtx, Graph, Kernel, NodeId, StateRow, Value};

#[derive(Clone)]
struct HopBudget { max_hops: u64, target_col: String }

impl Kernel for HopBudget {
    type State = u64; // hops taken so far
    fn initial_state(&self, _g: &Graph, _start: NodeId) -> u64 { 0 }
    fn visit(&self, cx: &EdgeCtx<'_, u64>) -> anyhow::Result<bool> {
        Ok(*cx.state() < self.max_hops)
    }
    fn next_state(&self, cx: &EdgeCtx<'_, u64>) -> anyhow::Result<u64> {
        Ok(cx.state() + 1)
    }
    fn stop(&self, cx: &EdgeCtx<'_, u64>) -> anyhow::Result<bool> {
        Ok(*cx.state() >= self.max_hops
            || cx.dest_bool(&self.target_col)?.unwrap_or(false))
    }
    fn state_row(&self, state: &u64) -> StateRow {
        vec![("hops".into(), Value::U64(*state))]
    }
}

// Register the kernel by name (inventory is re-exported by rxgraph).
rxgraph::inventory::submit! {
    rxgraph::KernelEntry {
        name: "hop_budget",
        make: |p| Ok(rxgraph::boxed_run(HopBudget {
            max_hops: p["max_hops"].as_u64().unwrap(),
            target_col: p["target_col"].as_str().unwrap().to_string(),
        })),
    }
}
```

You compile your kernel into your own Python extension (statically linked with
`rxgraph`); importing it registers the kernel, and Python can then select it by
name:

```python
graph.search(start_nodes=["a"], kernel="hop_budget",
             params={"max_hops": 3, "target_col": "target"}, max_paths=10)
```

See [`examples/rust-kernel-plugin/`](examples/rust-kernel-plugin/README.md) for
the full guide, including the `EdgeCtx` accessor reference and the maturin build.

## Architecture

The Python package is backed by a Rust core. Internally, `rxgraph` stores node
and edge tables as Arrow `RecordBatch` values, validates graph identity columns
once, and builds compact CSR topology for traversal. User columns stay in
columnar form and remain available to stateful search expressions.

## Rust Crate

The Rust engine is published as the `rxgraph` crate and exposes the same
traversal kernel model used by the Python bindings.

See `crates/rxgraph/README.md` for crate-specific usage.
