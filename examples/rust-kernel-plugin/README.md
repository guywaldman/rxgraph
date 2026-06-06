# Rust search kernel plugin example

This is a small static plugin crate for `rxgraph`.

It implements a Rust search kernel named `hop_budget`:

- carry one state field: `hops`
- accept edges while `hops < max_hops`
- stop when the destination node has `target == true`, or when the hop budget is used

The kernel is registered with `rxgraph::inventory::submit!` in
[`src/lib.rs`](src/lib.rs).

## Rust-only check

From the repo root:

```bash
cargo test --manifest-path examples/rust-kernel-plugin/Cargo.toml --locked
```

That compiles this crate and runs a tiny Rust graph through
`rxgraph::build_kernel("hop_budget", ...)`.

## Python E2E check

The plugin must be linked into the Python extension at build time:

```bash
.venv/bin/maturin develop \
  --manifest-path crates/rxgraph-python/Cargo.toml \
  --release \
  --features kernel-plugin-example
```

Then Python can select the Rust kernel by name:

```python
import rxgraph as rxg

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
)

assert result.paths[0].nodes == ["a", "b", "c"]
assert result.paths[0].state == {"hops": 2}
```

CI runs both checks through:

```bash
just test-kernel-plugin-example
```
