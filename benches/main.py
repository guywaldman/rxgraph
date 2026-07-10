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
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

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
DEFAULT_BASELINE = "native-bound/serial"


def main() -> None:
    args = parser().parse_args()
    if rxg.rayon_thread_count() < 2:
        raise RuntimeError("benchmarks require at least two Rayon workers")
    DIST.mkdir(exist_ok=True)
    console = Console(width=max(Console().width, 200))
    reports = [
        run_profile(profile, args.filter, args.baseline, console)
        for profile in selected_profiles(args.profile, args.scale)
    ]
    output = DIST / "benchmarks.json"
    output.write_text(json.dumps(report_payload(reports), indent=2))
    print(f"wrote {output}")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rxgraph macro benchmarks.")
    parser.add_argument(
        "profile",
        nargs="?",
        choices=("quick", "standard", "large", "all"),
        default="all",
    )
    parser.add_argument(
        "scale",
        nargs="?",
        type=scale_factor,
        default=1.0,
        help="Positive workload multiplier; scales starts and topology nodes.",
    )
    parser.add_argument(
        "--baseline",
        type=baseline_key,
        default=DEFAULT_BASELINE,
        metavar="ENGINE/MODE",
        help=f"Core-search baseline (default: {DEFAULT_BASELINE}).",
    )
    parser.add_argument("--filter", help=argparse.SUPPRESS)
    return parser


def scale_factor(value: str) -> float:
    try:
        scale = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("scale must be a positive number") from error
    if scale <= 0:
        raise argparse.ArgumentTypeError("scale must be a positive number")
    return scale


def baseline_key(value: str) -> str:
    try:
        engine, execution = value.split("/")
    except ValueError as error:
        raise argparse.ArgumentTypeError("baseline must be ENGINE/MODE") from error
    engine = {"native": "native-bound"}.get(engine, engine)
    if engine not in {"dsl", "native-bound"}:
        raise argparse.ArgumentTypeError(f"unknown baseline engine {engine!r}")
    if execution not in {"serial", "parallel"}:
        raise argparse.ArgumentTypeError("baseline mode must be serial or parallel")
    return f"{engine}/{execution}"


def report_payload(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema_version": REPORT_VERSION, "profiles": profiles}


def run_profile(
    profile: Profile, filter_text: str | None, baseline: str, console: Console
) -> dict[str, Any]:
    search_data = prepare_search(profile)
    topology_data = prepare_topology(profile)
    # The large graph is intentionally sizeable. Run the Rust core executable
    # before constructing Python's in-memory graph so the two CSR copies never
    # coexist in this process tree.
    core = run_core(search_data, profile, filter_text, baseline)
    nodes, edges, read_seconds = read_tables(search_data)
    started = now()
    graph = rxg.Graph(nodes, edges)
    graph_seconds = now() - started
    setup = setup_report(search_data, read_seconds, graph_seconds)

    python_api = run_python_api(graph, profile)
    topology = run_topology(topology_data, profile) if topology_data else None
    report = {
        "profile": profile.name,
        "scale": profile.scale,
        "baseline": baseline,
        "setup": setup,
        "core_search": core,
        "python_api": python_api,
        "topology": topology,
    }
    print_profile(report, console)
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
    data: PreparedData, profile: Profile, filter_text: str | None, baseline: str
) -> dict[str, Any]:
    output = DIST / f"bench-core-{profile.name}-{scale_tag(profile.scale)}.json"
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
    if not any(case_key(case) == baseline for case in cases):
        raise ValueError(f"baseline {baseline!r} is not present in this benchmark run")
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
    annotate_core_categories(cases, baseline)
    raw["baseline"] = baseline
    return raw


CORE_CATEGORY_LABELS = {
    "dfs": "dfs",
    "bfs": "bfs",
    "search_first": "first",
    "build": "filter-build",
    "one-shot-bfs": "filter+bfs",
}


def case_key(case: dict[str, Any]) -> str:
    return f"{case['engine']}/{case['execution']}"


def annotate_core_categories(cases: list[dict[str, Any]], baseline: str) -> None:
    by_operation: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_operation.setdefault(case["operation"], []).append(case)
    for operation, category in by_operation.items():
        winner = min(category, key=lambda case: case["median_seconds"])
        baseline_case = next(
            (case for case in category if case_key(case) == baseline), None
        )
        for case in category:
            case["category"] = CORE_CATEGORY_LABELS[operation]
            case["is_best"] = case is winner
            case["is_baseline"] = case is baseline_case
            case["baseline_speedup"] = (
                baseline_case["median_seconds"] / case["median_seconds"]
                if baseline_case
                else None
            )


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


def scale_tag(scale: float) -> str:
    return f"x{scale:g}".replace(".", "_")


def print_profile(report: dict[str, Any], console: Console | None = None) -> None:
    console = console or Console(width=max(Console().width, 200))
    setup = report["setup"]
    graph = report["core_search"]["graph"]
    console.print()
    baseline = report.get(
        "baseline", report["core_search"].get("baseline", DEFAULT_BASELINE)
    )
    table = Table(
        title=f"rxgraph benchmarks: {report['profile']} × {report.get('scale', 1):g}",
        box=box.ROUNDED,
    )
    for name, justify in [
        ("Bench", "left"),
        ("Implementation", "left"),
        ("Test setup", "right"),
        ("Median", "right"),
        ("P90", "right"),
        ("Best", "right"),
        ("Speedup", "right"),
        ("Graph", "right"),
        ("Result size", "right"),
    ]:
        table.add_column(name, justify=justify, no_wrap=True)
    rows = display_rows(report, baseline)
    for index, row in enumerate(rows):
        table.add_row(
            row["bench"],
            best_label(row["implementation"], row["is_best"]),
            Text(format_seconds(row["setup_seconds"]), style="dim"),
            format_seconds(row["median_seconds"]),
            format_seconds(row["p90_seconds"]),
            format_seconds(row["best_seconds"]),
            comparison_text(row["baseline_speedup"], row["is_baseline"]),
            row["graph"],
            fmt_count(row["result_size"]),
            end_section=index + 1 < len(rows)
            and rows[index + 1]["bench"] != row["bench"],
        )
    console.print(
        f"[bold]Workload[/bold]: {fmt_count(graph['nodes'])} nodes / "
        f"{fmt_count(graph['edges'])} edges; cache {setup['cache']} "
        f"({setup['cache_size_bytes'] / 2**20:.1f} MiB); core baseline {baseline}"
    )
    console.print(table)


def format_seconds(seconds: float) -> str:
    return f"{seconds * 1e3:.3f} ms" if seconds < 1 else f"{seconds:.3f} s"


def display_rows(report: dict[str, Any], baseline: str) -> list[dict[str, Any]]:
    rows = core_display_rows(report, baseline)
    rows.extend(python_display_rows(report))
    rows.extend(topology_display_rows(report))
    return rows


def core_display_rows(report: dict[str, Any], baseline: str) -> list[dict[str, Any]]:
    setup_seconds = report["setup"]["graph_construction_seconds"]
    graph = report["core_search"]["graph"]
    rows = []
    for category, cases in core_categories(report["core_search"]["cases"], baseline):
        for case in cases:
            rows.append(
                {
                    "bench": f"core/{category}",
                    "implementation": f"{case['engine']}/{case['execution']}",
                    "setup_seconds": setup_seconds,
                    "median_seconds": case["median_seconds"],
                    "p90_seconds": case["p90_seconds"],
                    "best_seconds": best_sample_seconds(case),
                    "baseline_speedup": case["baseline_speedup"],
                    "is_baseline": case["is_baseline"],
                    "is_best": case["is_best"],
                    "graph": graph_label(graph),
                    "result_size": case["result_paths"],
                }
            )
    return rows


def python_display_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    cases = report["python_api"]["cases"]
    baseline = next(case for case in cases if case["name"] == "dsl/serial")
    winner = min(cases, key=lambda case: case["median_seconds"])
    setup_seconds = report["setup"]["graph_construction_seconds"]
    graph = report["python_api"]["graph"]
    return [
        {
            "bench": "python/api",
            "implementation": case["name"],
            "setup_seconds": setup_seconds,
            "median_seconds": case["median_seconds"],
            "p90_seconds": case["p90_seconds"],
            "best_seconds": best_sample_seconds(case),
            "baseline_speedup": baseline["median_seconds"] / case["median_seconds"],
            "is_baseline": case is baseline,
            "is_best": case is winner,
            "graph": graph_label(graph),
            "result_size": case["result_size"],
        }
        for case in cases
    ]


def topology_display_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    if not (topology := report["topology"]):
        return []
    rows = []
    for operation, cases in topology["operations"].items():
        baseline = next(case for case in cases if case["name"] == "rxgraph-arrow-u64")
        winner = min(cases, key=lambda case: case["median_seconds"])
        for case in cases:
            rows.append(
                {
                    "bench": f"topology/{operation}",
                    "implementation": case["name"],
                    "setup_seconds": topology["build"][case["name"]]["median_seconds"],
                    "median_seconds": case["median_seconds"],
                    "p90_seconds": case["p90_seconds"],
                    "best_seconds": best_sample_seconds(case),
                    "baseline_speedup": baseline["median_seconds"]
                    / case["median_seconds"],
                    "is_baseline": case is baseline,
                    "is_best": case is winner,
                    "graph": graph_label(topology["graph"]),
                    "result_size": case["result_size"],
                }
            )
    return rows


def graph_label(graph: dict[str, int]) -> str:
    return f"{fmt_count(graph['nodes'])}/{fmt_count(graph['edges'])}"


def best_sample_seconds(case: dict[str, Any]) -> float:
    return min(case["samples_seconds"])


def fmt_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}".rstrip("0").rstrip(".") + "K"
    return str(value)


def core_categories(
    cases: list[dict[str, Any]], baseline: str
) -> list[tuple[str, list[dict[str, Any]]]]:
    annotate_core_categories(cases, baseline)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(case["category"], []).append(case)
    return [
        (category, sorted(rows, key=core_sort_key))
        for category, rows in grouped.items()
    ]


def core_sort_key(case: dict[str, Any]) -> tuple[int, int, float]:
    engine_order = {"dsl": 0, "native-bound": 1}
    execution_order = {"serial": 0, "parallel": 1}
    return (
        engine_order[case["engine"]],
        execution_order[case["execution"]],
        case["median_seconds"],
    )


def best_label(label: str, is_best: bool) -> Text:
    return Text(
        f"{label} (best)" if is_best else label, style="bold green" if is_best else ""
    )


def comparison_text(speedup: float | None, is_baseline: bool) -> Text:
    if speedup is None:
        return Text("—", style="dim")
    if is_baseline:
        return Text("baseline", style="bold")
    if 1 / 1.05 < speedup < 1.05:
        return Text("same", style="dim")
    if speedup > 1:
        return Text(f"{speedup:.1f}× faster", style="green")
    return Text(f"{1 / speedup:.1f}× slower", style="red")


if __name__ == "__main__":
    main()
