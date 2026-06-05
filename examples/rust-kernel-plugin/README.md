# Native Rust kernel plugin example

This crate shows how to supply a **native Rust traversal kernel** to `rxgraph`
and select it from Python by name. The kernel here is `HopBudget`: starting from
a node, it walks the graph and emits a path as soon as it reaches a node flagged
as a target, or once it has taken `max_hops` edges.

> [!NOTE]
>
> This is documentation-grade example code. It is a **standalone** crate (it has
> its own `[workspace]` table) so it does not disturb the main `rxgraph`
> workspace build. Build and test it with `--manifest-path`.

## When to use a Rust kernel vs the Polars DSL

`Graph.search(...)` with Polars expressions (the DSL) covers most stateful
searches and needs no compilation step. Reach for a native kernel when you want:

- **Full Rust control** - arbitrary state types and logic that the expression
  DSL cannot express, including calling into your own crates.
- **Performance** - the engine is monomorphized over your `Kernel`, so per-edge
  `visit` / `next_state` / `stop` calls are **statically dispatched** (no
  per-edge vtable). State is whatever Rust type you choose, not a name-keyed row.

The trade-off is that a native kernel must be compiled into an extension module
(see [Model A](#model-a-static-recommended-now)).

## The `Kernel` trait

A kernel makes the same three per-edge decisions the DSL makes for every
candidate edge `(src)-[edge]->(dest)`:

```rust
pub trait Kernel {
    type State: Clone;                                              // per-path payload
    fn initial_state(&self, graph: &Graph, start: NodeId) -> Self::State;
    fn visit(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<bool>;       // accept edge?
    fn next_state(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Self::State>; // child state
    fn stop(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<bool>;        // emit path?
    fn state_row(&self, state: &Self::State) -> StateRow;          // Vec<(String, Value)>
}
```

`visit` and `next_state` see the **parent** state via `cx.state()`. `stop` sees
the **child** state produced by `next_state`. `state_row` materializes named
state for returned paths.

### `EdgeCtx` accessors (brief reference)

`EdgeCtx<'_, S>` is the per-edge context. Handles:

- `graph()`, `src()`, `dest()`, `edge()` - internal ids and the graph.
- `state()` - the relevant per-path state `&S`.
- `with_state(&S)` - re-borrow with a different state (engine-internal pattern).
- `src_id()` / `dest_id()` / `edge_id()` -> `Option<GraphId>` (external ids).

Payload column reads (by column name), for `src` / `dest` / `edge`:

- `*_value(col) -> Result<Value>` - the dynamic value.
- typed getters returning `Result<Option<T>>` (`Ok(None)` on null):
  `*_u64`, `*_i64`, `*_f64`, `*_bool`, `*_str`.

Typed getters bind an Arrow column reader per call (cheap, but not cached), so in
hot loops read only the columns you need.

## Registering a kernel by name

Submit a `KernelEntry` with `inventory::submit!`. `rxgraph` re-exports
`inventory`, so your crate does **not** need its own `inventory` dependency:

```rust
rxgraph::inventory::submit! {
    rxgraph::KernelEntry {
        name: "hop_budget",
        make: |params| Ok(rxgraph::boxed_run(HopBudget::from_params(params)?)),
    }
}
```

`make` receives the JSON `params` blob and returns a `BoxedRun`. `boxed_run`
wraps your concrete `Kernel` so the inner search stays monomorphized (one boxed
call per *whole* search, not per edge). The kernel is then reachable by name:

```rust
let params = serde_json::json!({ "max_hops": 3, "target_col": "target" });
let runnable = rxgraph::build_kernel("hop_budget", &params)?;
let result = runnable.run(&graph, rxgraph::RunOptions {
    start_nodes: vec!["a".into()],
    parallel: false,
    ..Default::default()
})?;
```

See [`src/lib.rs`](src/lib.rs) for the full `HopBudget` implementation and its
tests, which build a tiny graph and run it through `build_kernel`.

## Build and test the kernel

From the repo root:

```bash
cargo test --manifest-path examples/rust-kernel-plugin/Cargo.toml
```

The default build is a plain library crate, so this requires no Python toolchain.

## Model A (static, recommended now)

In the static model you build your **own** Python extension that statically
links your kernel together with `rxgraph`. Importing the resulting module runs
the `inventory::submit!` registration, after which the kernel is selectable by
name.

This crate already exposes the optional wiring behind a cargo feature `python`
(OFF by default). With the feature on, `[lib] crate-type` includes `cdylib` and a
`#[pymodule] rxgraph_kernel_example` is compiled:

```toml
[features]
python = ["dep:pyo3"]

[lib]
crate-type = ["rlib", "cdylib"]
```

> [!NOTE]
>
> `cargo build --features python` may fail at the final link step with undefined
> `_Py*` symbols when there is no Python interpreter to link against. That is the
> usual extension-module link path and is expected - build through **maturin**,
> which configures the link correctly.

Add a `pyproject.toml` pointing maturin at this crate and turning on the feature:

```toml
[build-system]
requires = ["maturin>=1.13,<1.14"]
build-backend = "maturin"

[project]
name = "rxgraph-kernel-example"
requires-python = ">=3.11"
dynamic = ["version"]
dependencies = ["rxgraph", "polars"]

[tool.maturin]
manifest-path = "examples/rust-kernel-plugin/Cargo.toml"
module-name = "rxgraph_kernel_example"
features = ["python"]
```

Then build and install it into your environment:

```bash
maturin develop --manifest-path examples/rust-kernel-plugin/Cargo.toml --features python
```

From Python, import your module (this is what registers the kernel) and then
select it by name on a graph:

```python
import rxgraph as rxg
import rxgraph_kernel_example  # noqa: F401  -- import registers "hop_budget"

graph = rxg.Graph.from_edges(
    [("a", "b"), ("b", "c"), ("c", "d")],
    nodes=[("a", {"target": False}),
           ("b", {"target": False}),
           ("c", {"target": True}),
           ("d", {"target": False})],
)

result = graph.search(
    start_nodes=["a"],
    kernel="hop_budget",
    params={"max_hops": 3, "target_col": "target"},
    max_paths=10,
)
```

> [!IMPORTANT]
>
> The Python `kernel=` / `params=` selector is being added in parallel with this
> example. The signature above
> (`graph.search(start_nodes=[...], kernel="hop_budget", params={...}, max_paths=10)`)
> is the **intended** call site; confirm against the shipped Python API, as the
> exact keyword names may change.

## Model B (dlopen, future)

> [!NOTE]
>
> Not yet implemented. Described here so you can see where the runtime hook fits.

`rxgraph` also exposes `register_kernel(name, make)` - a **runtime** registry
that `build_kernel` consults before the link-time `inventory` entries. The
intended future use is `dlopen`: a prebuilt `rxgraph` wheel could load a
separately compiled `.so` at runtime, the `.so` would expose a stable C-ABI
entrypoint, and that entrypoint would call `register_kernel(...)` to add its
kernels - no recompilation of `rxgraph` required.

This is deferred because it requires a **stable ABI** across the boundary
(parameter encoding, error propagation, `Graph`/result layout) and the
entrypoint is necessarily **unsafe FFI**. Until that surface is designed and
frozen, Model A (compile your kernel together with the engine) is the supported
path and gives you full static dispatch for free.
