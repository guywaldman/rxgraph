# rxgraph

High-performance graph traversal engine for Rust, built for usage in the Python library [rxgraph](https://github.com/guywaldman/rxgraph).

This crate focuses on fast traversal, explicit stateful search, and a small API
surface for common graph algorithms.

> [!IMPORTANT]
>
> `rxgraph` is under heavy active development - the core implementation (Python bindings & Rust crate) are usable with decent test coverage, but it is not ready for production and you should use it at your own risk. The public API is likely to change as well.
>
> Having said that, the library is usable and should work for most scenarios it supports, and I would love for some initial feedback.

## Architecture

Internally, `rxgraph` stores node and edge tables as Arrow `RecordBatch` values,
validates graph identity columns once, builds compact CSR topology, and
evaluates stateful traversal kernels without copying user columns out of Arrow.

## Install

```toml
[dependencies]
rxgraph = "0.0.5"
```

## Data Model

`Graph::new(nodes, edges)` expects:

- nodes: `id`
- edges: `id`, `src`, `dest`
- all identity columns uniformly `UInt64` or uniformly string
- optional `type` columns as strings

Additional node and edge columns can be read by traversal DSL expressions.

## Topology Queries

Use the graph methods for common directed graph algorithms:

- `bfs` / `dfs`
- `reachable_nodes`
- `shortest_path`
- `out_degrees`, `in_degrees`, `degrees`
- `weakly_connected_components`

Integer-ID graphs also have `_u64` variants that avoid materializing borrowed
`GraphId` values.

```rust
use std::sync::Arc;

use arrow::{
    array::{ArrayRef, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use rxgraph::Graph;

fn main() -> anyhow::Result<()> {
    let nodes = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)])),
        vec![Arc::new(UInt64Array::from(vec![0, 1, 2])) as ArrayRef],
    )?;
    let edges = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("src", DataType::UInt64, false),
            Field::new("dest", DataType::UInt64, false),
            Field::new("price", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(UInt64Array::from(vec![10, 11])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![0, 1])),
            Arc::new(UInt64Array::from(vec![1, 2])),
            Arc::new(UInt64Array::from(vec![5, 6])),
        ],
    )?;

    let graph = Graph::new(nodes, edges)?;
    assert_eq!(graph.shortest_path_u64(0, 2)?, Some(Some(vec![0, 1, 2])));
    Ok(())
}
```

## Stateful Search

Traversal uses a `DslKernel` with three parts:

- `visit`: whether a candidate edge is accepted
- `next_state`: how named path state changes after accepting the edge
- `stop`: whether the accepted path should be returned

```rust
use rxgraph::{DslExpr as e, DslKernel, TraversalConfigBuilder, Value};

# fn example(graph: &rxgraph::Graph) -> anyhow::Result<()> {
let kernel = DslKernel::new(
    e::edge("price").le(e::uint(20)),
    [("spent".into(), e::state("spent").plus(e::edge("price")))],
    e::dest_id().eq(e::uint(2)),
    [("spent".into(), Value::U64(0))],
);

let config = TraversalConfigBuilder::new(kernel)
    .with_start_nodes([0_u64])
    .with_max_depth(3)
    .with_max_paths(10)
    .with_parallelism(true)
    .build();

let result = graph.search(config)?;
println!("paths={} evaluated={}", result.paths.len(), result.stats.evaluated_edges);
# Ok(())
# }
```

Search can run depth-first or breadth-first, with optional Rayon-backed
parallelism and optional per-node intermediate state materialization.

## Python Bindings

Python bindings are published separately as the `rxgraph` package on PyPI from
the same repository.
