"""Frozen native-search workloads; run against baseline and candidate wheels.

python -m benches.native_acceptance --output dist/native.json [--stream]
Graph construction is excluded. Search setup, payload reads and conversion are timed.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import tempfile
import time
from pathlib import Path

import polars as pl
import rxgraph as rxg

# Freeze before optimization. No data-dependent pruning or benchmark filtering.
CASES = (
    ("first", 1, 2, 1, 1, False),
    ("reject", 64, 2, 128, 0, False),
    ("no_match", 32, 64, 1, None, False),
    ("cycle", 16, 128, 2, None, False),
    ("wide", 64, 3, 64, None, False),
    ("many", 2048, 2, 1, None, False),
    ("history", 64, 32, 1, None, True),
)


def tables(case):
    name, starts, depth, fanout, limit, history = case
    # Independent chains, duplicate rejected edges, common terminal node.
    terminal = starts * depth
    src, dest, cost = [], [], []
    for start in range(starts):
        for level in range(depth):
            node = start * depth + level
            src.append(node)
            dest.append(terminal if level == depth - 1 else node + 1)
            cost.append(1)
            for extra in range(1, fanout):
                src.append(node)
                dest.append(start * depth if name == "cycle" else terminal)
                cost.append(1 if name in ("cycle", "wide") else depth + 1)
    nodes = pl.DataFrame({"id": range(terminal + 1)}, schema={"id": pl.UInt64})
    edges = pl.DataFrame(
        {"id": range(len(src)), "src": src, "dest": dest, "cost": cost},
        schema={k: pl.UInt64 for k in ("id", "src", "dest", "cost")},
    )
    kwargs = dict(
        start_nodes=[i * depth for i in range(starts)],
        kernel="weighted_budget",
        params=dict(
            weight_col="cost",
            budget=0 if name == "reject" else depth,
            target=terminal + 1 if name == "no_match" else terminal,
        ),
        max_depth=depth,
        max_paths=limit or None,
        intermediate_states=history,
    )
    return nodes, edges, kwargs


def normalize(paths):
    return sorted(
        (
            tuple(p.nodes),
            tuple(p.edges),
            json.dumps(p.state, sort_keys=True),
            repr(p.intermediate_states),
        )
        for p in paths
    )


def measure(graph, kwargs, stream, runs):
    expected = normalize(graph.search(**kwargs).paths)
    samples, first = [], []
    for repeat in range(runs + 2):
        started = time.perf_counter()
        if stream:
            with graph.search_batches(**kwargs, batch_size=stream) as cursor:
                paths = []
                first_seconds = None
                for batch in cursor:
                    if first_seconds is None:
                        first_seconds = time.perf_counter() - started
                    paths.extend(batch)
                stats = cursor.stats
        else:
            result = graph.search(**kwargs)
            paths, stats = result.paths, result.stats
            first_seconds = None
        elapsed = time.perf_counter() - started
        assert normalize(paths) == expected
        if repeat >= 2:
            samples.append(elapsed)
            first.append(first_seconds)
    return dict(
        samples_seconds=samples,
        median_seconds=statistics.median(samples),
        p90_seconds=sorted(samples)[max(0, (len(samples) * 9 + 9) // 10 - 1)],
        first_batch_seconds=first,
        result_size=len(expected),
        evaluated_edges=stats.evaluated_edges,
        parallel_edges=stats.parallel_edges,
        lazy_payload_read_calls=stats.lazy_payload_read_calls,
        lazy_payload_selected_rows=stats.lazy_payload_selected_rows,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--runs", type=int, default=9)
    args = parser.parse_args()
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        for case in CASES:
            nodes, edges, kwargs = tables(case)
            node_path, edge_path = (
                Path(directory) / "nodes.parquet",
                Path(directory) / "edges.parquet",
            )
            nodes.write_parquet(node_path)
            edges.write_parquet(edge_path)
            for mode in ("arrow", "eager", "lazy"):
                for strategy in ("dfs", "bfs"):
                    graph = (
                        rxg.Graph(nodes, edges)
                        if mode == "arrow"
                        else rxg.Graph.from_parquet(node_path, edge_path, payloads=mode)
                    )
                    run = dict(kwargs, strategy=strategy, parallel=False)
                    start = time.perf_counter()
                    initial = graph.search(**run)
                    cold = time.perf_counter() - start
                    key = f"{case[0]}/{mode}/{strategy}"
                    results[key] = measure(graph, run, 0, args.runs)
                    results[key]["first_search_seconds"] = cold
                    results[key]["first_search_result_size"] = len(initial.paths)
                    if args.stream:
                        for size in (1, 64, 1024, 4096):
                            results[f"{key}/batch-{size}"] = measure(
                                graph, run, size, args.runs
                            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                schema_version=1,
                baseline="3df81bd",
                cases=results,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
