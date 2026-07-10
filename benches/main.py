"""Trustworthy macro benchmarks for rxgraph's in-memory execution paths."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

import polars as pl
import rxgraph as rxg

from benches.data import (
    PreparedData,
    Profile,
    prepare_search,
    prepare_topology,
    selected_profiles,
)
from benches.measure import TimedCase, as_json, measure_cases, measure_one

REPORT_VERSION = 1
DIST = Path("dist")


def main() -> None:
    args = parser().parse_args()
    if rxg.rayon_thread_count() < 2:
        raise RuntimeError("benchmarks require at least two Rayon workers")
    DIST.mkdir(exist_ok=True)
    reports = [
        run_profile(profile, args.filter) for profile in selected_profiles(args.profile)
    ]
    output = DIST / "benchmarks.json"
    output.write_text(json.dumps(report_payload(reports), indent=2))
    print(f"wrote {output}")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rxgraph macro benchmarks.")
    parser.add_argument(
        "--profile", choices=("quick", "standard", "large", "all"), default="all"
    )
    parser.add_argument("--filter", help=argparse.SUPPRESS)
    return parser


def report_payload(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema_version": REPORT_VERSION, "profiles": profiles}


def run_profile(profile: Profile, filter_text: str | None) -> dict[str, Any]:
    search_data = prepare_search(profile)
    topology_data = prepare_topology(profile)
    # The large graph is intentionally sizeable. Run the Rust core executable
    # before constructing Python's in-memory graph so the two CSR copies never
    # coexist in this process tree.
    core = run_core(search_data, profile, filter_text)
    nodes, edges, read_seconds = read_tables(search_data)
    started = now()
    graph = rxg.Graph(nodes, edges)
    graph_seconds = now() - started
    setup = setup_report(search_data, read_seconds, graph_seconds)

    python_api = run_python_api(graph, profile)
    topology = run_topology(topology_data, profile) if topology_data else None
    report = {
        "profile": profile.name,
        "setup": setup,
        "core_search": core,
        "python_api": python_api,
        "topology": topology,
    }
    print_profile(report)
    return report


def now() -> float:
    from time import perf_counter

    return perf_counter()


def read_tables(data: PreparedData) -> tuple[pl.DataFrame, pl.DataFrame, float]:
    started = now()
    nodes = pl.read_parquet(data.node_path)
    edges = pl.read_parquet(data.edge_path)
    return nodes, edges, now() - started


def setup_report(
    data: PreparedData, read_seconds: float, graph_seconds: float
) -> dict[str, Any]:
    return {
        "cache": "hit" if data.cache_hit else "miss",
        "cache_path": str(data.manifest_path.parent),
        "cache_size_bytes": data.size_bytes,
        "generation_or_validation_seconds": data.setup_seconds,
        "parquet_read_seconds": read_seconds,
        "graph_construction_seconds": graph_seconds,
        "rows": json.loads(data.manifest_path.read_text())["rows"],
    }


def run_core(
    data: PreparedData, profile: Profile, filter_text: str | None
) -> dict[str, Any]:
    output = DIST / f"bench-core-{profile.name}.json"
    command = [
        "cargo",
        "bench",
        "-p",
        "rxgraph",
        "--bench",
        "stateful_native",
        "--",
        "--manifest",
        str(data.manifest_path.resolve()),
        "--output",
        str(output.resolve()),
    ]
    if filter_text:
        command.extend(("--filter", filter_text))
    subprocess.run(command, check=True)
    raw = json.loads(output.read_text())
    cases = raw["cases"]
    by_key = {
        (case["engine"], case["operation"], case["execution"]): case for case in cases
    }
    for case in cases:
        case["median_seconds"] = statistics.median(case["samples_seconds"])
        case["p90_seconds"] = percentile90(case["samples_seconds"])
        dsl = by_key.get(("dsl", case["operation"], case["execution"]))
        if dsl:
            case["dsl_speedup"] = dsl["median_seconds"] / case["median_seconds"]
        serial = by_key.get((case["engine"], case["operation"], "serial"))
        if case["execution"] == "parallel" and serial:
            case["same_engine_parallel_speedup"] = (
                serial["median_seconds"] / case["median_seconds"]
            )
    return raw


def run_python_api(graph: rxg.Graph, profile: Profile) -> dict[str, Any]:
    kwargs = dsl_kwargs(profile)
    typed = typed_kwargs(profile)
    cases = [
        TimedCase(
            "dsl/serial",
            lambda: graph.search(**kwargs, parallel=False),
            normalize_search,
            stats=search_stats,
        ),
        TimedCase(
            "typed-row/eager-cache/serial",
            lambda: graph.search(**typed, parallel=False),
            normalize_search,
            stats=search_stats,
        ),
    ]
    if profile.name != "quick":
        cases.insert(
            1,
            TimedCase(
                "dsl/parallel",
                lambda: graph.search(**kwargs, parallel=True),
                normalize_search,
                execution="parallel",
                stats=search_stats,
            ),
        )
    measurements = measure_cases(cases, warmups=profile.warmups, runs=profile.runs)
    baseline = measurements[0].median_seconds
    return {
        "graph": {"nodes": graph.node_count, "edges": graph.edge_count},
        "rayon_threads": rxg.rayon_thread_count(),
        "cases": [as_json(measurement, baseline) for measurement in measurements],
    }


def dsl_kwargs(profile: Profile) -> dict[str, Any]:
    spent = rxg.col("state.spent")
    cost = rxg.col("edge.cost")
    return {
        "start_nodes": start_nodes(profile),
        "visit": rxg.col("edge.allowed") & ((spent + cost) <= profile.search.depth),
        "next_state": {"spent": spent + cost},
        "stop": rxg.col("dest.id") == profile.search.node_count - 1,
        "initial_state": {"spent": 0},
        "max_depth": profile.search.depth,
        "max_paths": profile.search.starts,
        "strategy": "bfs",
        "max_visits_per_node": 1,
    }


def typed_kwargs(profile: Profile) -> dict[str, Any]:
    return {
        "start_nodes": start_nodes(profile),
        "kernel": "weighted_budget",
        "params": {
            "weight_col": "cost",
            "budget": profile.search.depth,
            "target": profile.search.node_count - 1,
        },
        "max_depth": profile.search.depth,
        "max_paths": profile.search.starts,
        "strategy": "bfs",
        "max_visits_per_node": 1,
    }


def start_nodes(profile: Profile) -> list[int]:
    return [case * profile.search.depth for case in range(profile.search.starts)]


def normalize_search(
    result: rxg.SearchResult,
) -> tuple[tuple[tuple[int, ...], int], ...]:
    return tuple(
        sorted((tuple(path.nodes), int(path.state["spent"])) for path in result.paths)
    )


def search_stats(result: rxg.SearchResult) -> tuple[int, int]:
    return result.stats.evaluated_edges, result.stats.parallel_edges


def run_topology(data: PreparedData, profile: Profile) -> dict[str, Any]:
    import igraph
    import networkx as nx

    nodes, edges, read_seconds = read_tables(data)
    pairs = list(zip(edges["src"].to_list(), edges["dest"].to_list(), strict=True))
    node_count = nodes.height

    def build_rx() -> rxg.Graph:
        return rxg.Graph(nodes, edges)

    def build_igraph() -> igraph.Graph:
        return igraph.Graph(n=node_count, edges=pairs, directed=True)

    def build_networkx() -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(range(node_count))
        graph.add_edges_from(pairs)
        return graph

    builders = {
        "rxgraph-arrow-u64": build_rx,
        "igraph": build_igraph,
        "networkx": build_networkx,
    }
    build_samples = {
        name: measure_one(build, warmups=profile.warmups, runs=profile.runs)
        for name, build in builders.items()
    }
    graphs = {name: build() for name, build in builders.items()}
    target = node_count - 1
    operations = {
        "bfs": {
            "rxgraph-arrow-u64": lambda: graphs["rxgraph-arrow-u64"].bfs(0),
            "igraph": lambda: graphs["igraph"].bfs(0, mode="OUT")[0],
            "networkx": lambda: list(nx.bfs_tree(graphs["networkx"], 0)),
        },
        "shortest_path": {
            "rxgraph-arrow-u64": lambda: graphs["rxgraph-arrow-u64"].shortest_path(
                0, target
            ),
            "igraph": lambda: graphs["igraph"].get_shortest_paths(
                0, to=target, mode="OUT"
            )[0],
            "networkx": lambda: nx.shortest_path(graphs["networkx"], 0, target),
        },
        "degrees": {
            "rxgraph-arrow-u64": lambda: graphs["rxgraph-arrow-u64"].degrees(),
            "igraph": lambda: graphs["igraph"].degree(mode="ALL"),
            "networkx": lambda: [
                graphs["networkx"].degree(node) for node in range(node_count)
            ],
        },
        "weak_components": {
            "rxgraph-arrow-u64": lambda: graphs[
                "rxgraph-arrow-u64"
            ].weakly_connected_components(),
            "igraph": lambda: graphs["igraph"].connected_components(mode="weak"),
            "networkx": lambda: list(
                nx.weakly_connected_components(graphs["networkx"])
            ),
        },
    }
    normalizers = {
        "bfs": lambda result: tuple(sorted(result)),
        "shortest_path": lambda result: (len(result), result[0], result[-1]),
        "degrees": tuple,
        "weak_components": lambda result: tuple(
            sorted(tuple(sorted(group)) for group in result)
        ),
    }
    measured = {}
    for operation, runners in operations.items():
        cases = [
            TimedCase(name, run, normalizers[operation])
            for name, run in runners.items()
        ]
        values = measure_cases(cases, warmups=profile.warmups, runs=profile.runs)
        baseline = values[0].median_seconds
        measured[operation] = [as_json(value, baseline) for value in values]
    return {
        "setup": setup_report(data, read_seconds, 0.0),
        "graph": {"nodes": node_count, "edges": len(pairs)},
        "build": {
            name: {
                "samples_seconds": samples,
                "median_seconds": statistics.median(samples),
                "p90_seconds": percentile90(samples),
            }
            for name, samples in build_samples.items()
        },
        "operations": measured,
    }


def percentile90(samples: list[float]) -> float:
    sorted_samples = sorted(samples)
    return sorted_samples[max(0, (len(samples) * 9 + 9) // 10 - 1)]


def print_profile(report: dict[str, Any]) -> None:
    setup = report["setup"]
    graph = report["core_search"]["graph"]
    print(
        f"\n{report['profile']}: {graph['nodes']:,} nodes, {graph['edges']:,} edges; "
        f"cache {setup['cache']} ({setup['cache_size_bytes'] / 2**20:.1f} MiB); "
        f"prepare/read/graph {setup['generation_or_validation_seconds']:.3f}/"
        f"{setup['parquet_read_seconds']:.3f}/{setup['graph_construction_seconds']:.3f}s"
    )
    print("core search")
    for case in report["core_search"]["cases"]:
        fraction = case["parallel_edges"] / max(1, case["evaluated_edges"])
        dsl_speedup = case.get("dsl_speedup")
        parallel_speedup = case.get("same_engine_parallel_speedup")
        speedups = " ".join(
            value
            for value in (
                f"vs DSL {dsl_speedup:.2f}x" if dsl_speedup else "",
                f"same-engine {parallel_speedup:.2f}x" if parallel_speedup else "",
            )
            if value
        )
        print(
            f"  {case['name']:<42} {case['median_seconds'] * 1e3:9.3f} ms "
            f"p90 {case['p90_seconds'] * 1e3:9.3f} ms  {case['execution']:<8} "
            f"parallel-edge fraction {fraction:.0%} {speedups}"
        )
    print("python api")
    for case in report["python_api"]["cases"]:
        print(
            f"  {case['name']:<42} {case['median_seconds'] * 1e3:9.3f} ms "
            f"p90 {case['p90_seconds'] * 1e3:9.3f} ms  {case['execution']} "
            f"vs DSL {case['baseline_speedup']:.2f}x"
        )
    if topology := report["topology"]:
        print("topology")
        for operation, cases in topology["operations"].items():
            values = ", ".join(
                f"{case['name']} {case['median_seconds'] * 1e3:.3f} ms "
                f"p90 {case['p90_seconds'] * 1e3:.3f} ms "
                f"vs rxgraph {case['baseline_speedup']:.2f}x"
                for case in cases
            )
            print(f"  {operation:<18} {values}")


if __name__ == "__main__":
    main()
