# Plan: Native Go port of rxgraph (`go/`)

## Goal

A pure-Go reimplementation of the rxgraph engine in a new top-level `go/`
directory. Go consumers write search kernels in **native Go** (a normal generic
interface, zero FFI, goroutine-parallel). No DSL.

## Locked decisions

- **Engine:** full port to Go, not FFI/cgo.
- **Search kernels:** native Go only (no Polars/expression DSL).
- **Payload store:** `apache/arrow-go` as the core internal store (mirrors Rust's
  Arrow `RecordBatch`).
- **Execution:** serial **and** goroutine-parallel BFS/DFS in v1 (parity with Rust).
- **Conformance:** a Rust binary generates golden outputs from the real engine;
  a Go test asserts equality in CI.
- **Module path:** `github.com/guywaldman/rxgraph/go` (subdir module).
- **State clone semantics:** value semantics. `NextState` returns a fresh state
  (mirrors Rust `next_state`). Documented rule: state must be a value type, or
  the kernel must deep-copy reference fields in `NextState`. No `Clone()` method
  required.
- **`ID` type:** API edge accepts `any` constrained to `uint64 | string`,
  validated against the graph's ID mode. No enum wrapper.
- **`Value::List`/`Struct` in StateRow:** supported in v1 (full parity).

## Guiding principles

1. Simplicity first, but never at the cost of performance/memory. No scope creep.
2. Mirror the (proven-fast) Rust logic where it helps; avoid redundant code.
   Aim for minimal code.

## In scope

- Graph construction from Arrow node/edge tables; ID identity validation
  (uniform `UInt64` or uniform string), external->internal `u32` mapping,
  optional `type` columns.
- CSR out-adjacency + reverse adjacency (in-edges).
- Arrow payload typed column readers + per-search column cache, with the exact
  coercion rules from Rust (`as_u64/as_i64/as_f64/as_bool/as_str`).
- `Value`/`StateRow` types incl. `List`/`Struct`.
- Topology: `bfs`, `dfs` (depth-limited), `reachable_nodes`, `shortest_path`
  (unweighted BFS), `out/in/degrees`, `weakly_connected_components`.
- Stateful search: serial + parallel BFS/DFS, with `max_depth`, `max_paths`,
  `max_revisits_per_node`, `intermediate_states`, `SearchStats`, path
  materialization.
- `EdgeCtx` typed accessor surface and the `Kernel` contract.

## Out of scope (explicitly NOT ported)

- Polars/expression DSL: `dsl/eval.rs`, `dsl/expr.rs`, `dsl/bind.rs`,
  `dsl/ops/`, `dsl/polars_json.rs`.
- PyO3/Python bindings; lazy/`from_lazy` payload projection; Rust
  plugin/registry/inventory machinery; the progress spinner (port as a no-op /
  trivial counter; not output-relevant).

## Public Go API (`go/`, package `rxgraph`)

```go
// Construction
func NewGraph(nodes, edges arrow.Record) (*Graph, error)      // Arrow core
func FromEdges(edges []Edge, opts ...Option) (*Graph, error)  // ergonomic, builds Arrow internally

// Topology
func (g *Graph) BFS(start ID, maxDepth *int) ([]ID, error)
func (g *Graph) DFS(start ID, maxDepth *int) ([]ID, error)
func (g *Graph) ShortestPath(src, dst ID) ([]ID, error)
func (g *Graph) ReachableNodes(start ID) ([]ID, error)
func (g *Graph) OutDegrees() []int
func (g *Graph) InDegrees() []int
func (g *Graph) Degrees() []int
func (g *Graph) WeaklyConnectedComponents() [][]ID

// Native Go kernel
type Kernel[S any] interface {
    InitialState(g *Graph, start NodeID) S
    Visit(cx *EdgeCtx[S]) (bool, error)
    NextState(cx *EdgeCtx[S]) (S, error)
    Stop(cx *EdgeCtx[S]) (bool, error)
    StateRow(s S) StateRow
}

func Search[S any](g *Graph, k Kernel[S], opts SearchOptions) (*SearchResult, error)
```

- `EdgeCtx[S]`: `Src()/Dest()/Edge()` (internal `NodeId`/`EdgeId`),
  `SrcID()/DestID()/EdgeID()` (external `ID`), `State()`, and typed payload
  getters mirroring Rust exactly: `Dest{Bool,U64,I64,F64,Str}`, plus `Src*` and
  `Edge*` variants. Return `(value, ok, error)` to mirror Rust
  `Result<Option<T>>` (`ok=false` == null).
- `SearchOptions`: `StartNodes`, `MaxDepth`, `MaxPaths`, `Strategy` (DFS/BFS),
  `MaxRevisitsPerNode`, `Parallel`, `IntermediateStates`.
- `SearchResult`: `Paths []GraphPath`, `Stats SearchStats`.
- `GraphPath`: `Nodes []ID`, `Edges []ID`, `State StateRow`,
  `IntermediateStates []StateRow` (nil unless requested).
- `ID` = `any` constrained to `uint64 | string`, validated against ID mode.

## Internal port mapping (Rust -> Go)

| Rust source | Go file | Notes |
|---|---|---|
| `graph/csr.rs` | `internal/csr.go` | Direct port; `u32` -> `uint32`. |
| `graph/repo.rs` | `internal/repo.go` | ID validation, external<->internal maps, in/out adjacency, payload batches (arrow-go). |
| `dsl/arrow_value.rs` (`ColumnReader`) | `internal/column.go` | Typed Arrow column reader + per-search column cache (`PayloadCache`, kernel.rs:43). |
| `dsl/value.rs` | `value.go` | `Value` sum type incl. List/Struct; equality only where StateRow output needs it. |
| `kernel.rs` coercions + `EdgeCtx` | `edgectx.go` | Exact coercion rules (lossless int/float, error cases) for conformance. |
| `traversal/algo.rs` | `internal/search.go` | Serial + parallel BFS/DFS, arena/seed model, revisit counts, materialization, stats. |
| `traversal/config.rs`, `traversal/mod.rs` | `options.go`, `result.go` | `SearchOptions`, `SearchResult`, `SearchStats`, `GraphPath`. |
| `graph/graph.rs` topology | `topology.go` | BFS/DFS/shortest_path/WCC/degrees. |

### Parallelism

Replace Rayon with goroutines + a bounded worker pool sized to `GOMAXPROCS`:

- **Parallel BFS:** parallel frontier expansion, chunked across goroutines,
  results merged (mirrors `eval_frontier_parallel`, algo.rs:436).
- **Parallel DFS:** seed model (`build_dfs_seeds` -> `DFS_SEEDS_PER_THREAD`,
  algo.rs:562) with goroutines and an atomic found-counter (`sync/atomic`).
- The column cache is **per-worker** (Rust `PayloadCache` is `!Sync` for the same
  reason, kernel.rs:38-42). Each goroutine/fold gets its own cache.
- Parallelization thresholds ported as-is: `MIN_PAR_FRONTIER=512`,
  `MIN_PAR_EDGES=8192`, `DFS_SEEDS_PER_THREAD=8`, `MIN_PAR_DFS_PATHS=64`,
  `should_parallelize_dfs` (algo.rs:750).

### Semantics that must match bit-for-bit

- Revisit accounting: `visit_counts_*` / `can_visit_*` (algo.rs:755-843),
  including the `edge_count <= 1` fast path that skips count maps.
- `eval_arena_edge` order: `visit` -> `next_state` -> `stop(with child state)`
  (algo.rs:509-548).
- Stats fields and `merge_stats` (algo.rs:954).
- `max_paths` truncation: exact for serial; soft early-stop + exact truncation
  for parallel (`stopped_paths` may exceed returned count).
- Coercion edge cases: `as_u64` rejects non-integral / negative floats; `compare`
  on NaN; `is_finite` + exact `fract()==0.0` checks (kernel.rs:329-414).
- `shortest_path` is unweighted BFS; `source==target` returns single-node path.

## Conformance suite (Rust generates goldens)

1. **Fixtures** (`tests/conformance/fixtures/*.json`, shared by both languages):
   each case = graph tables (nodes/edges + payload columns) + kernel id + kernel
   params + search options.
2. **Golden generator** (new Rust bin, e.g. `crates/rxgraph/examples/conformance_gen.rs`):
   loads each fixture, runs the real Rust engine with the matching Rust kernel,
   writes golden `paths` + `stats` JSON to `tests/conformance/golden/`.
3. **Go conformance test** (`go/conformance_test.go`): loads each fixture, runs
   the Go engine with the same-named Go kernel, asserts equality vs. golden.
4. **Shared kernels** implemented in both languages, referenced by name via a
   tiny conformance-only registry on each side. Start with `hop_budget` (the
   existing Rust example); add 1-2 more covering list/struct state and revisits.
   Consumers still write their own Go kernels directly; the registry is for tests.
5. **Ordering caveat:** Rust parallel path order is unspecified (mod.rs:8-9).
   - Generate goldens from the Rust **serial** path (deterministic); assert exact
     equality for Go **serial**.
   - For **parallel**, assert **set-equality** of returned paths (sorted by a
     canonical key) + equality of order-independent stats. Document which stats
     are compared under parallel (`stopped_paths` excluded from exact compare).

## CI / build wiring (`justfile`)

- `gen-conformance`: build+run the Rust golden generator, write goldens.
- `test-go`: `cd go && go test ./...` (requires goldens present).
- `conformance-check`: regenerate goldens and `git diff --exit-code` to ensure
  they are committed and current.
- Add `test-go` + `conformance-check` to the `ci` recipe. Document the new Go
  toolchain dependency in `CONTRIBUTING.md`.

## Execution order

1. `internal/csr.go`, `internal/repo.go`, Arrow ingestion + ID validation (+ unit tests).
2. `internal/column.go` + `value.go` + `edgectx.go` coercions (+ unit tests ported from Rust cases).
3. `topology.go` (BFS/DFS/shortest/WCC/degrees) (+ tests).
4. `internal/search.go` serial, then parallel; `Kernel`/`Search` public API.
5. Conformance: fixtures, Rust golden generator, `hop_budget` (+ list/struct, revisit kernels) in both, Go conformance test.
6. `FromEdges` ergonomic constructor, `go/README.md`, examples, `justfile`/CI wiring.

## Docs

- `go/README.md` quickstart mirroring the Python README structure.
- A "Native Go kernels" section in the top-level `README.md` paralleling
  "Native Rust kernels".
