"""File-backed typed-kernel benchmarks, kept separate from in-memory search."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import rxgraph as rxg

from benches.data import Profile, prepare_search, selected_profiles
from benches.main import normalize_search, scale_factor, start_nodes, typed_kwargs
from benches.measure import as_json, measure_cases, measure_one, TimedCase

DIST = Path("dist")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rxgraph file-backed benchmarks.")
    parser.add_argument(
        "profile", nargs="?", choices=("quick", "standard", "large"), default="standard"
    )
    parser.add_argument("scale", nargs="?", type=scale_factor, default=1.0)
    args = parser.parse_args()
    DIST.mkdir(exist_ok=True)
    profile = selected_profiles(args.profile, args.scale)[0]
    report = run(profile)
    output = DIST / f"bench-storage-{profile.name}-x{profile.scale:g}.json"
    output.write_text(json.dumps(report, indent=2))
    print(f"wrote {output}")


def run(profile: Profile) -> dict[str, Any]:
    data = prepare_search(profile, codec="zstd")
    kwargs = typed_kwargs(profile)
    build = {
        mode: measure_one(
            lambda mode=mode: rxg.Graph.from_parquet(
                data.node_path, data.edge_path, payloads=mode
            ),
            warmups=profile.warmups,
            runs=profile.runs,
        )
        for mode in ("eager", "lazy")
    }
    graphs = {
        mode: rxg.Graph.from_parquet(data.node_path, data.edge_path, payloads=mode)
        for mode in ("eager", "lazy")
    }

    first = {}
    repeated = {}
    for mode, graph in graphs.items():
        started = time.perf_counter()
        first_result = graph.search(**kwargs, parallel=False)
        first[mode] = {
            "seconds": time.perf_counter() - started,
            "lazy_io": lazy_io(first_result),
        }
        measurements = measure_cases(
            [
                TimedCase(
                    f"typed-row/{mode}/serial",
                    lambda graph=graph: graph.search(**kwargs, parallel=False),
                    normalize_search,
                    stats=lambda result: (
                        result.stats.evaluated_edges,
                        result.stats.parallel_edges,
                    ),
                )
            ],
            warmups=profile.warmups,
            runs=profile.runs,
        )
        repeated[mode] = as_json(measurements[0])
    return {
        "schema_version": 1,
        "profile": profile.name,
        "scale": profile.scale,
        "cache": {
            "hit": data.cache_hit,
            "path": str(data.manifest_path.parent),
            "size_bytes": data.size_bytes,
            "setup_seconds": data.setup_seconds,
        },
        "graph": {
            "nodes": profile.search.node_count,
            "edges": profile.search.edge_count,
            "starts": len(start_nodes(profile)),
        },
        "construction": {
            mode: {
                "samples_seconds": samples,
                "median_seconds": statistics.median(samples),
            }
            for mode, samples in build.items()
        },
        "first_search": first,
        "repeated_search": repeated,
    }


def lazy_io(result: rxg.SearchResult) -> dict[str, int]:
    stats = result.stats
    return {
        "read_calls": stats.lazy_payload_read_calls,
        "requested_rows": stats.lazy_payload_requested_rows,
        "selected_rows": stats.lazy_payload_selected_rows,
        "row_groups": stats.lazy_payload_row_groups,
    }


if __name__ == "__main__":
    main()
