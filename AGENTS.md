# Repository guidance

- Support and test kernel changes in `Kernel`, `TypedKernel`, and `native::Kernel`, including streams.
- Treat `NodeId` and `EdgeId` as graph-local; preserve external IDs and Python labels at API boundaries.
- Keep target bounds conservative and exclude edges only when no valid continuation uses them.
- Preserve stream pull laziness, cleanup, and returned-path lifetimes.
- Preserve serial traversal order, global limits, and cycle rules; parallel order is unspecified.
- `Graph` is directed; `DiGraph` adds reverse edges.
- Keep Python bindings, `plugin.py`, type stubs, and downstream kernel plugins in sync.
- Require plugin integration tests explicitly when validating plugins; they otherwise may skip.
- Preserve compatibility with Polars 1.40.1+ and 2.x.
- Run relevant tests; `just ci` runs the full CI workflow.
