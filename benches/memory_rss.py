"""RSS cap verification.

This test builds a graph whose payload columns are wide enough that the eager API holds an
order of magnitude more working/resident memory than the lazy API, which is column-projected.

Some notes:
- Each mode runs in its own subprocess so resident memory is measured cleanly
after construction settles and the source frames are dropped.
- The parent process compares the two and exits non-zero unless lazy stays at least `--min-ratio`
times below eager.
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import psutil

import rxgraph as rxg


@dataclass(frozen=True)
class Scale:
    nodes: int
    pad_cols: int
    pad_width: int


SCALES = {
    # Sized for CI: eager holds ~0.5 GB, lazy a few tens of MB (>10x apart),
    # and the whole run finishes in a few seconds.
    "small": Scale(nodes=80_000, pad_cols=56, pad_width=96),
    # Manual stress fixture: eager is multiple GB, lazy stays flat.
    "large": Scale(nodes=1_000_000, pad_cols=60, pad_width=128),
}


def _write_fixture(scale: Scale, node_path: Path, edge_path: Path) -> None:
    ids = pl.Series("id", range(scale.nodes), dtype=pl.UInt64)
    # Distinct values per row so the payload genuinely consumes memory. Identical
    # strings would be dictionary/COW-compressed and the payload would vanish,
    # making the eager/lazy comparison meaningless.
    suffix = "x" * max(scale.pad_width - 12, 0)
    pad_cols = {
        f"pad{i}": (
            pl.int_range(scale.nodes, dtype=pl.UInt64)
            .cast(pl.String)
            .str.pad_start(12, "0")
            + f"_{i}_{suffix}"
        )
        for i in range(scale.pad_cols)
    }
    n = pl.int_range(scale.nodes, dtype=pl.Int64)
    nodes = pl.select(
        id=ids,
        risk_score=n % 100,
        country=pl.Series(["US", "GB", "DE", "FR"] * (scale.nodes // 4 + 1))[
            : scale.nodes
        ],
        tags=pl.concat_list(("tag_" + (n % 7).cast(pl.String)), pl.lit("seen")),
        **pad_cols,
    )  # ty:ignore[no-matching-overload]
    nodes.write_parquet(node_path)

    # A shallow fanout tree keeps searches fast: the RSS difference comes from
    # held payload columns, not from traversal work.
    fanout = 4
    src = []
    dest = []
    nxt = 1
    frontier = [0]
    while nxt < scale.nodes and frontier:
        parent = frontier.pop(0)
        for _ in range(fanout):
            if nxt >= scale.nodes:
                break
            src.append(parent)
            dest.append(nxt)
            frontier.append(nxt)
            nxt += 1
    edges = pl.DataFrame(
        {
            "id": pl.Series("id", range(len(src)), dtype=pl.UInt64),
            "src": pl.Series("src", src, dtype=pl.UInt64),
            "dest": pl.Series("dest", dest, dtype=pl.UInt64),
            "amount": pl.Series(
                "amount", [float(i % 500) for i in range(len(src))], dtype=pl.Float64
            ),
        }
    )
    edges.write_parquet(edge_path)


def _run_searches(graph: rxg.Graph) -> None:
    # A risk-graph style traversal:
    # Follow non-trivial transactions into allowed countries, accumulating spend, peak risk,
    # and the tags seen along the path.
    # It reads several node/edge columns (incl. the compound tags list), so the
    # projection pulls exactly those and skips every pad column.
    for _ in range(3):
        graph.search(
            start_nodes=[0],
            visit=(pl.col("edge.amount") > 0) & (pl.col("dest.country") != "FR"),
            next_state={
                "spend": pl.col("state.spend") + pl.col("edge.amount"),
                "peak_risk": pl.when(
                    pl.col("dest.risk_score") > pl.col("state.peak_risk")
                )
                .then(pl.col("dest.risk_score"))
                .otherwise(pl.col("state.peak_risk")),
                "tags": pl.col("state.tags").list.set_union(pl.col("dest.tags")),
            },
            initial_state={"spend": 0.0, "peak_risk": 0, "tags": []},
            stop=pl.col("dest.id") == 0,
            strategy="bfs",
            max_depth=6,
            max_paths=1,
        )


def _stable_rss_mb() -> float:
    gc.collect()
    return psutil.Process().memory_info().rss / 1e6


def _worker(mode: str, node_path: Path, edge_path: Path) -> None:
    # Baseline RSS before any graph data is loaded. Subtracting it isolates the
    # graph's own resident footprint from the interpreter/library overhead.
    baseline = _stable_rss_mb()

    if mode == "eager":
        nodes = pl.read_parquet(node_path)
        edges = pl.read_parquet(edge_path)
        graph = rxg.Graph(nodes, edges)
        del nodes, edges
    elif mode == "lazy":
        graph = rxg.Graph.from_lazy(
            pl.scan_parquet(node_path), pl.scan_parquet(edge_path)
        )
    else:
        raise SystemExit(f"unknown mode: {mode}")

    _run_searches(graph)
    delta = max(_stable_rss_mb() - baseline, 0.0)
    # Print a single parseable line; the parent reads the last token.
    print(f"RSS_MB {delta:.1f}")


def _measure(mode: str, node_path: Path, edge_path: Path) -> float:
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "benches.memory_rss",
            "--worker",
            mode,
            "--nodes-path",
            str(node_path),
            "--edges-path",
            str(edge_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in out.stdout.splitlines():
        if line.startswith("RSS_MB "):
            return float(line.split()[1])
    raise RuntimeError(f"worker did not report RSS:\n{out.stdout}\n{out.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=SCALES, default="small")
    parser.add_argument("--min-ratio", type=float, default=5.0)
    parser.add_argument("--nodes", type=int, help="override fixture node count")
    parser.add_argument("--pad-cols", type=int, help="override payload column count")
    parser.add_argument("--pad-width", type=int, help="override payload string width")
    parser.add_argument("--worker", choices=["eager", "lazy"])
    parser.add_argument("--nodes-path")
    parser.add_argument("--edges-path")
    args = parser.parse_args()

    if args.worker is not None:
        _worker(args.worker, Path(args.nodes_path), Path(args.edges_path))
        return 0

    base = SCALES[args.scale]
    scale = Scale(
        nodes=args.nodes or base.nodes,
        pad_cols=args.pad_cols or base.pad_cols,
        pad_width=args.pad_width or base.pad_width,
    )
    with tempfile.TemporaryDirectory() as tmp:
        node_path = Path(tmp) / "nodes.parquet"
        edge_path = Path(tmp) / "edges.parquet"
        _write_fixture(scale, node_path, edge_path)

        eager_rss = _measure("eager", node_path, edge_path)
        lazy_rss = _measure("lazy", node_path, edge_path)

    ratio = eager_rss / lazy_rss if lazy_rss > 0 else float("inf")
    status = "PASS" if ratio >= args.min_ratio else "FAIL"
    print(
        f"scale={args.scale} nodes={scale.nodes} pad_cols={scale.pad_cols}\n"
        f"eager RSS = {eager_rss:8.1f} MB\n"
        f"lazy  RSS = {lazy_rss:8.1f} MB\n"
        f"ratio     = {ratio:8.1f}x  (min {args.min_ratio:.0f}x)  {status}"
    )
    return 0 if ratio >= args.min_ratio else 1


if __name__ == "__main__":
    raise SystemExit(main())
