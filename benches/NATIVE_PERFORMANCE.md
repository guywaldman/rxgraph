# Native streaming performance

The acceptance target is **2× in every measured workload**, independently at the
Rust and Python boundaries. Draining a default-size stream must take no more
than **1.05×** the optimized eager call. Aggregate improvements do not satisfy
these gates. Results are measured against `3df81bd`.

## Measurement

- `native_fixture.rs` defines the original 80 Rust cases: early match, rejection,
  exhaustive no-match, cycles, wide output, many paths, history, and large state;
  DFS/BFS; Arrow kernels and typed cold/warm/lazy runners.
- `native_acceptance.rs` and `native_streaming.rs` use the normal system allocator.
  The separate `native_allocations.rs` executable measures allocation counts and
  bytes. Allocation-instrumented exploratory timings are not acceptance results.
- `native_acceptance.py` retains the original 42 Python built-in-kernel cases.
  The benchmark-only plugin adds Arrow-native, typed-native, compound-state, and
  supported parallel coverage without removing any original cases.
- `compare_native.py` alternates complete Rust baseline/candidate/stream suites
  for three rounds, collecting 27 timed samples per case. Python baseline,
  candidate, and four batch sizes are interleaved within each case, with 15 timed
  samples after preflight and warmup. Python output values are compared against
  the original implementation, including edges, complete state, and history.
- Rust first-use timings are retained separately, with one sample per round
  (three samples). Their median ratios are reported separately and are noisy
  for sub-microsecond cases; they cannot establish a robust speedup alone.
- Graph construction and fixture writes are excluded from timings. Search-local
  setup, payload decoding/reads, result conversion, and stream creation are
  included. Python first-use decoding is measured on fresh graphs separately.
  These are not cold operating-system page-cache measurements.
- Rust uses four Rayon workers. Python's existing pool initialization chooses the
  available CPU count and ignores `RAYON_NUM_THREADS`; the comparison checks that
  baseline and candidate have the same count and records it in `gates.json`.
  Requested parallel cases retain actual `parallel_edges`, including fallbacks.
- Batches of 1, 64, 1024, and 4096 are measured. Only 1024 is subject to the 5%
  gate. First-batch latency is recorded separately; no-result searches record
  `null` for that metric.
- RSS is a whole-suite process high-water mark, including fixture preparation.
  Rust runs have separate process measurements; the Python measurement includes
  both loaded backends and is not an isolated per-backend memory comparison.

## Implemented changes

- Graph and native-store cursors share execution with eager search. DFS retains
  active frames and dense/sparse visits; parallel workers reuse their scratch.
  BFS processes bounded frontier chunks and retains pending transitions across
  pulls. Parallelism runs only inside a pull.
- Python consumes result vectors once. Named typed runners encode directly from
  traversal state, without constructing intermediate typed result states.
  The Serde encoder constructs traversal values without a JSON object tree.
- Unlabeled native streams construct the public Python `SearchPath` directly at
  the binding boundary. They do not allocate an internal path wrapper and tuple
  before constructing the public object. Labeled graphs retain the mapping path.
- Native stores can expose a dense node count. Built-in stores use dense DFS
  visit tracking; custom sparse stores keep their fallback.
- Typed eager decoding and lazy file setup are deferred to the first pull.
  Lazy caches survive subsequent pulls, loaded-row counts are incremental, and
  redundant metadata reads and prefetch/deduplication work are removed.
- Sessions retain graph ownership through `Arc`; close/error/exhaustion releases
  session ownership. Shared eager decode caches remain reusable. Python payload
  replacement is blocked while a session retains the graph.

Functional verification: 70 Rust tests plus one doctest passed (two existing
doctest examples remain ignored); 185 Python tests passed with both plugin
requirements enabled. Clippy with warnings denied, Ruff, and formatting checks
passed. Coverage includes DSL regressions, serial ordering, global limits,
partial batches, close/lifetimes, deferred payload failures, errors after earlier
batches, and rebuilt downstream plugin behavior.

## Reproduction

Build release extensions and benchmarks before timing. Preserve the baseline
source, binaries, and wheels independently of candidate artifacts. **Use separate
Cargo target directories for different checkouts**: identical package names and
versions can otherwise reuse incompatible artifacts.

To regenerate the baseline, export `3df81bd` to a temporary directory. Copy
`native_fixture.rs`, `native_acceptance.rs`, and `native_allocations.rs` into its
crate's `benches/` directory and add harness-free bench entries for the latter
two. Copy `benches/native-plugin/` to the corresponding location in the baseline
checkout. Build both baseline wheels and unpack them without installing them
over the candidate packages. The benchmark plugin source works against both
versions; it has no production registration or dependencies in the main package.

Build the candidate with:

```sh
just build-maturin
.venv/bin/maturin develop --manifest-path benches/native-plugin/Cargo.toml --release --offline
cargo bench -p rxgraph --bench native_acceptance --bench native_streaming --bench native_allocations --no-run
```

Pass the executable paths printed by Cargo and the unpacked extension paths:

```sh
.venv/bin/python -m benches.compare_native \
  --baseline-binary BASELINE_EXECUTABLE \
  --candidate-binary CANDIDATE_EXECUTABLE \
  --stream-binary STREAM_EXECUTABLE \
  --baseline-package BASELINE_PYTHON_PACKAGE_DIRECTORY \
  --baseline-plugin BASELINE_BENCH_PLUGIN_EXTENSION \
  --candidate-plugin CANDIDATE_BENCH_PLUGIN_EXTENSION \
  --output dist/native-acceptance/final --check
```

`--check` exits unsuccessfully when either performance gate misses. Run the two
`native_allocations` binaries separately and save their JSON next to the timing
reports. Functional checks are `cargo test --workspace --locked`, the Python test
suite, and rebuilding/testing the downstream plugin example. Set
`RXGRAPH_REQUIRE_KERNEL_PLUGIN_EXAMPLE=1` and
`RXGRAPH_REQUIRE_NATIVE_BENCH_PLUGIN=1` to prohibit skipped plugin coverage.

## Results

Measured on 2026-09-14 on the same arm64 macOS machine, using four Rust workers
and 16 Python workers. **Both acceptance gates remain unmet.** The comparison
command with `--check` exited with status 1. No failing workload was removed.

| Boundary | Cases reaching 2× | Worst speedup | Median across cases | Best speedup |
| --- | ---: | ---: | ---: | ---: |
| Rust repeated calls | 6 / 80 | 0.81× | 1.16× | 49.17× |
| Rust first calls | 4 / 80 | 0.72× | 1.19× | 27.34× |
| Python repeated calls | 20 / 106 | 0.32× | 1.49× | 10.25× |
| Python first calls | 13 / 106 | 0.31× | 1.31× | 11.43× |

These are baseline/candidate ratios. Values below 1 mean regressions; the median
across cases is descriptive and does not replace the per-case acceptance gate.

In the retained full-matrix artifacts, before the direct Python stream-boundary
change above, batch size 1024 has **74/80 Rust cases and 88/106 Python cases**
within the 5% full-drain gate: **162/186 total**, with 24 failures. The largest
measured overhead was 18.8% in Rust (`first/BreadthFirst/typed-cold`) and 17.3%
in Python (`plugin/wide/eager/dfs`).

Latency improves even in some full-drain failures. For
`plugin/wide/arrow/bfs/parallel`, the first 1024 results arrive in a median
**1.20 ms**, versus **14.05 ms** for optimized eager completion. Full streaming
drain takes **16.45 ms**, so this case still fails the overhead gate.

### Custom Rust kernel follow-up

The former worst Python output case was rerun after removing the intermediate
stream representation. The focused run used 8 warmups and 51 interleaved timed
samples and retained the same 12,160-path output-equivalence check:

| `plugin/wide/eager/dfs` | Median | Ratio to eager | p90 |
| --- | ---: | ---: | ---: |
| Optimized eager | 12.386 ms | 1.000 | 12.858 ms |
| Stream, batch 64 | 12.139 ms | 0.980 | 12.646 ms |
| Stream, batch 1024 | 11.760 ms | 0.950 | 12.176 ms |
| Stream, batch 4096 | 11.747 ms | 0.948 | 12.181 ms |

This custom typed Rust-kernel case now clears the 5% full-drain gate at the
default size. The retained full-matrix totals above have not been regenerated,
so the overall acceptance result remains unmet until that complete run passes.

`LazyFrame` is not used for the batch boundary. The search has already produced
materialized paths, so wrapping them in a lazy query plan does not avoid result
conversion. Arrow/Polars output would need a separate columnar result contract
and a stable schema for arbitrary serialized kernel state.

### Remaining costs and regressions

- Python `plugin/reject/arrow/dfs/parallel` regresses from **0.332 ms to 1.043 ms**.
  Both evaluate 8192 edges; baseline reports zero parallel edges and candidate
  reports 8192. This points to parallel admission/worker overhead on rejection-heavy
  work as the next issue to profile and fix.
- Small first-match calls still pay cursor and stream setup overhead. The worst
  repeated Rust ratio is `first/DepthFirst/bound-serial` at 0.81×.
- Lazy `many/DepthFirst/typed-lazy` still issues **4096 payload reads** in both
  versions. Caching across pulls and cheaper bookkeeping have not eliminated
  repeated Parquet decoding within a search.
- The remaining full-matrix stream failures still require a complete rerun after
  the Python boundary change; the focused former worst case now passes.

Separate Rust allocation profiles show where storage/conversion changes help:

| Case | Allocations, baseline → candidate | Bytes allocated, baseline → candidate |
| --- | ---: | ---: |
| Cyclic DFS, Arrow parallel | 6,163 → 66 | 28,833,664 → 190,208 |
| History DFS, typed warm | 11,277 → 10,954 | 3,006,020 → 1,898,848 |
| Rejection BFS, Arrow parallel | 59 → 15 | 396,544 → 13,552 |
| Many results DFS, typed lazy | 291,062 → 282,758 | 207,345,030 → 205,424,212 |

Maximum Rust suite RSS across the three runs was **45.1 MiB baseline**, **25.4 MiB
candidate eager**, and **26.0 MiB streaming**. The combined Python process peaked
at **279.7 MiB**. These are process high-water marks, not per-search or isolated
Python-backend memory measurements. Allocation totals describe Rust calls;
Python object allocation counts are not separately instrumented.

Raw samples, p90 timings, first-batch timings at all four sizes, per-case gates,
allocation profiles and I/O counters are retained locally in
[`dist/native-acceptance/verified/`](../dist/native-acceptance/verified/), including
[`gates.json`](../dist/native-acceptance/verified/gates.json) and
[`python.json`](../dist/native-acceptance/verified/python.json). The directory is
ignored by Git; use the reproduction commands above to regenerate reports.
