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
rxgraph = "0.6"
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

### Native Rust fast path

Native kernels should bind primitive Arrow fields once instead of performing a
name lookup for every candidate edge:

```rust
use rxgraph::{EdgeCtx, EdgeField, Graph, Kernel, NodeId, Transition};

#[derive(Clone)]
struct BudgetKernel {
    price: EdgeField<u64>,
    budget: u64,
}

impl Kernel for BudgetKernel {
    type State = u64;

    fn initial_state(&self, _graph: &Graph, _start: NodeId) -> anyhow::Result<u64> {
        Ok(0)
    }

    fn transition(&self, cx: &EdgeCtx<'_, u64>) -> anyhow::Result<Transition<u64>> {
        let spent = cx.state() + cx.edge_field(&self.price).unwrap_or(0);
        Ok(if spent > self.budget {
            Transition::Reject
        } else {
            Transition::Continue(spent)
        })
    }
}

# fn example(graph: &Graph) -> anyhow::Result<()> {
let kernel = BudgetKernel {
    price: graph.edge_field("price")?,
    budget: 100,
};
# let _ = kernel;
# Ok(())
# }
```

The name-based `EdgeCtx` getters remain available for convenience and coercion.
Pre-bound fields use exact Arrow types and are the intended maximum-throughput
path.

### Reusable reachability index

Build a reachability index once for named sets of internal ending nodes and
reuse it across searches:

```rust
use rxgraph::{Graph, ReachIndex, ReachSpec};

fn index(graph: &Graph) -> anyhow::Result<ReachIndex> {
    let target = graph.internal_node_id(42_u64).expect("target exists");
    graph.reach_index(ReachSpec {
        sets: vec![(3, vec![target])],
        skip_edge: None,
    })
}
```

Call `cx.reaches(&index, 3, cx.dest())?` from a graph kernel transition, or
return `Some(Cow::Borrowed(index.targets(graph, 3)?))` from `possible_targets`
to reuse a group's endpoints without allocating in the hook. A `skip_edge`
predicate can exclude edges only when they cannot occur in any valid
continuation. An index belongs to its source topology and must not be used with
a different graph.

### Target-aware native search

For a `Graph` search using `rxgraph::Kernel`, implement `possible_targets` to
give the engine a conservative bound on where a path can **end**. This hook is
optional and the normal native search path does not call it unless optimizations
are requested. `None` means no additional information, an empty slice means the
prefix cannot complete, and a nonempty result lists possible ending nodes by
internal `NodeId`. Each refinement is intersected with the path's previous bound.

The following kernel uses an `allowed: bool` edge column and completes at node
`42`. The graph must contain start node `1` and target node `42`.

```rust
use rxgraph::{
    EdgeCtx, EdgeField, Evaluation, Graph, Kernel, NodeId,
    RunOptions, TargetCtx, TargetOptions, Transition,
};
use std::borrow::Cow;

struct TargetKernel {
    allowed: EdgeField<bool>,
    target: NodeId,
}

impl Kernel for TargetKernel {
    type State = u64;

    fn initial_state(&self, _: &Graph, _: NodeId) -> anyhow::Result<u64> {
        Ok(0)
    }

    fn transition(&self, cx: &EdgeCtx<'_, u64>) -> anyhow::Result<Transition<u64>> {
        if !cx.edge_field(&self.allowed).unwrap_or(false) {
            return Ok(Transition::Reject);
        }
        let next = cx.state() + 1;
        Ok(if cx.dest() == self.target {
            Transition::Complete(next)
        } else {
            Transition::Continue(next)
        })
    }

    fn possible_targets<'a>(
        &'a self,
        cx: &TargetCtx<'_, u64>,
    ) -> anyhow::Result<Option<Cow<'a, [NodeId]>>> {
        Ok(match cx {
            TargetCtx::Start { .. } => Some(Cow::Borrowed(std::slice::from_ref(&self.target))),
            TargetCtx::Edge { edge, .. }
                if !edge.edge_field(&self.allowed).unwrap_or(false) => Some(Cow::Borrowed(&[])),
            _ => None,
        })
    }

    fn targets_depend_on_child_state(&self) -> bool {
        false
    }
}

fn search(graph: &Graph) -> anyhow::Result<()> {
    let kernel = TargetKernel {
        allowed: graph.edge_field("allowed")?,
        target: graph.internal_node_id(42_u64).expect("target exists"),
    };
    let run = RunOptions {
        start_nodes: vec![1_u64.into()],
        ..Default::default()
    }
    .with_optimizations(TargetOptions {
        evaluation: Evaluation::Incremental,
        reverse: true,
        release_expanded: false,
    });
    let mut stream = graph.search_batches_with(kernel, run, 1024)?;
    for batch in stream.by_ref() {
        for path in batch? {
            println!("{:?}: {}", path.nodes, path.state);
        }
    }
    println!("{:?}", stream.stats());
    Ok(())
}
```

Every bound must include every node at which the current prefix or any of its
valid continuations could complete. A possible ending can still be traversed as
an intermediate node; `transition` decides whether a path completes. A missing
ending can silently remove valid results. The hook must be deterministic and
must not depend on call count or side effects: the engine may skip transitions
and their errors on pruned branches. Edge contexts expose the source, edge and
destination in graph direction. Parent and child state may be absent, so return
`None` when a state-dependent restriction cannot be established safely.
Return `false` from `targets_depend_on_child_state` when the bound does not
depend on the child state to skip the post-transition callback.

`TargetOptions` selects independent, opt-in execution settings:

- `evaluation`: `Incremental` evaluates transitions during traversal; `Lazy`
  enumerates candidates and replays the exact kernel at possible endings.
  Lazy evaluation requires deterministic transitions, can repeat work on shared
  prefixes, and can be much slower when state rejects branches early. Its
  `stats.evaluated_edges` counts topology candidates, not replayed calls.
- `reverse`: builds reverse reachability from stateless start bounds. If a start
  has no such bound, reverse preprocessing is skipped. For streams, preparation
  begins on the first pull. This can save work on selective targets but adds
  preparation time and memory.
- `release_expanded`: releases expanded BFS states when intermediate history is
  not requested. This can reduce peak memory at a latency cost.

Implement `MemoizableKernel` and call `kernel.memoized(capacity)` to reuse
transitions with an optimized search. Equal keys for the same edge must imply
the same transition behavior, including rejection, completion, errors and the
entire child state. Errors are not cached. Capacity limits entries, not bytes,
and the wrapper uses a lock shared by workers. Create a fresh wrapper for each
search. Memoization and lazy replay are workload-dependent and can increase
latency or memory; measure them for the kernel and graph in use.

Target options also apply to `traversal::native::GraphStore` and `TypedKernel`
searches. A custom store without a backing `Graph` must expose a dense node
count to use reverse preprocessing with known targets. Transition memoization
remains available for `Graph` kernels.

## Python Bindings

Python bindings are published as the `rxgraph` package on PyPI. The same PyO3
binding layer is available from this crate via the `python` feature; downstream
native-kernel plugins enable `features = ["python"]` and use the
`rxgraph::plugin!` macro to register kernels and build a custom extension; see
[`examples/rust-kernel-plugin/`](../../examples/rust-kernel-plugin/README.md).

## Native result streams

`graph.search_batches_with(kernel, run, batch_size)` returns an iterator of
`Result<Vec<GraphPath<K::State>>>`. For custom stores, use
`rxgraph::traversal::native::search_batches(&store, kernel, run, batch_size)`.
Both retain traversal state between calls and expose `stats()` and `close()`.
Native-store paths borrow the caller's store, so they remain usable after the
cursor closes. Store traversal remains serial and accepts non-`Send` kernels.
Graph kernels retain Rayon support; parallel result ordering is unspecified.

The batch size must be positive and is a maximum; batches may be smaller.
No empty batches are emitted. Errors terminate the cursor, and `max_paths`
applies across the whole stream. Dropping the cursor releases its scratch state.

Named runners expose optional `RunKernel::stream`,
`RunTypedKernel::stream_eager_cached`, and `stream_parquet_lazy` methods returning
owned `NativeBatchStream` sessions. Existing custom eager-only implementations
remain valid and report an unsupported-capability error for streaming.
The standard `boxed_run` and `boxed_typed_run` wrappers implement streaming.
