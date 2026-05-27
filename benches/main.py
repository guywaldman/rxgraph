"""
Runs benchmarks against rxgraph, igraph and networkx.
Compares various algorithms/traversals wih low/high/mid scale, which is configurable.

Admittedly, this is mostly AI-generated, so these benchmarks can lie more than usual.
They should not be taken seriously at this stage :)
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

import rxgraph as rxg

from benches.scenarios import (
    GraphData,
    algorithm_cases,
    build_graph_data,
    build_library_graphs,
)


@dataclass
class TraversalData:
    nodes: pl.DataFrame
    edges: pl.DataFrame
    node_count: int
    edge_count: int
    destination: int


@dataclass
class TraversalGraphs:
    rxgraph: rxg.Graph
    networkx: Any | None
    igraph: Any | None
    node_rows: list[dict[str, Any]]
    edge_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class Scale:
    name: str
    node_count: int
    extra_edges: int


@dataclass
class BenchResult:
    bench: str
    algorithm: str
    scale: str
    library: str
    node_count: int
    edge_count: int
    values: list[float]
    result_size: int

    @property
    def median(self) -> float:
        return statistics.median(self.values)

    @property
    def best(self) -> float:
        return min(self.values)

    @property
    def p90(self) -> float:
        values = sorted(self.values)
        index = math.ceil(len(values) * 0.9) - 1
        return values[max(index, 0)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark rxgraph algorithms against NetworkX and igraph."
    )
    parser.add_argument(
        "--low-nodes",
        type=int,
        default=10_000,
        help="Node count for the low scale.",
    )
    parser.add_argument(
        "--mid-nodes",
        type=int,
        default=100_000,
        help="Node count for the mid scale.",
    )
    parser.add_argument(
        "--high-nodes",
        type=int,
        default=1_000_000,
        help="Node count for the high scale.",
    )
    parser.add_argument(
        "--extra-edges",
        type=int,
        default=4,
        help="Extra outgoing edge fanout for the high scale.",
    )
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--json", type=Path, default=Path("dist/algorithm-benchmarks.json")
    )
    parser.add_argument("--max-paths", type=int, default=50)
    parser.add_argument(
        "--traversal-fanout",
        type=int,
        default=256,
        help="Additional noisy outgoing edges added at each traversal stress point.",
    )
    args = parser.parse_args()

    scales = build_scales(
        args.low_nodes, args.mid_nodes, args.high_nodes, args.extra_edges
    )
    results = []

    for scale in scales:
        data = build_graph_data(scale.node_count, scale.extra_edges)
        graphs = build_library_graphs(data)
        for case in algorithm_cases(data, graphs):
            results.append(
                run_benchmark(
                    case.name,
                    scale.name,
                    case.library,
                    data,
                    case.run,
                    args.warmups,
                    args.runs,
                )
            )
        traversal_data = build_traversal_data(scale.node_count, args.traversal_fanout)
        traversal_graphs = build_traversal_graphs(traversal_data)
        for algorithm, library, func in build_traversal_benches(
            traversal_data, traversal_graphs, args.max_paths
        ):
            results.append(
                run_benchmark(
                    algorithm,
                    scale.name,
                    library,
                    GraphData(
                        nodes=traversal_data.nodes,
                        edges=traversal_data.edges,
                        node_count=traversal_data.node_count,
                        edge_count=traversal_data.edge_count,
                        source=0,
                        target=traversal_data.destination,
                    ),
                    func,
                    args.warmups,
                    args.runs,
                )
            )

    write_pyperf_json(args.json, results)
    print_report(results, scales, args.json)


def build_scales(
    low_nodes: int, mid_nodes: int, high_nodes: int, extra_edges: int
) -> list[Scale]:
    low_nodes = max(low_nodes, 2)
    mid_nodes = max(mid_nodes, low_nodes + 1)
    high_nodes = max(high_nodes, mid_nodes + 1)
    return [
        Scale("low", low_nodes, max(1, extra_edges // 2)),
        Scale("mid", mid_nodes, max(1, (extra_edges * 3) // 4)),
        Scale("high", high_nodes, max(1, extra_edges)),
    ]


def build_traversal_data(node_count: int, traversal_fanout: int) -> TraversalData:
    node_count = max(node_count, 2)
    traversal_fanout = max(traversal_fanout, 0)
    step = max(node_count // 18, 1)
    destination = node_count - 1

    nodes = pl.DataFrame(
        {
            "id": range(node_count),
            "risk": [(node * 7) % 9 for node in range(node_count)],
            "min_connection": [35 + ((node * 11) % 50) for node in range(node_count)],
            "closed": [
                node != 0
                and node + 1 != node_count
                and node % 23 == 0
                and node % step != 0
                for node in range(node_count)
            ],
        },
        schema={
            "id": pl.UInt64,
            "risk": pl.Int32,
            "min_connection": pl.UInt64,
            "closed": pl.Boolean,
        },
    )

    src: list[int] = []
    dest: list[int] = []
    price: list[int] = []
    departure: list[int] = []
    arrival: list[int] = []
    reliability: list[int] = []
    route_kind: list[str] = []
    detour_cost: list[int] = []

    strides = sorted(
        {
            1,
            2,
            3,
            5,
            8,
            13,
            21,
            max(step - 1, 1),
            step,
            step + 1,
            max(node_count // 7, 1),
            max(node_count // 5, 1),
        }
    )

    for from_id in range(node_count - 1):
        for stride in strides:
            to = min(from_id + stride, node_count - 1)
            if to == from_id:
                continue
            push_traversal_edge(
                src,
                dest,
                price,
                departure,
                arrival,
                reliability,
                route_kind,
                detour_cost,
                from_id,
                to,
                25 + ((stride * 3 + from_id) % 110),
                92 if is_main_stride(stride, node_count, step) else 45,
                "route",
                0,
            )

        if is_stress_node(from_id, node_count, step):
            for extra in range(traversal_fanout):
                to = 1 + ((from_id + extra * 37 + 17) % (node_count - 1))
                if to != from_id:
                    is_valid = extra % 5 == 0
                    push_traversal_edge(
                        src,
                        dest,
                        price,
                        departure,
                        arrival,
                        reliability,
                        route_kind,
                        detour_cost,
                        from_id,
                        to,
                        20 + (extra % 50),
                        95 if is_valid else 35 + (extra % 30),
                        "route" if is_valid else "skip",
                        1 if is_valid else 0,
                    )

    edges = pl.DataFrame(
        {
            "src": src,
            "dest": dest,
            "price": price,
            "departure": departure,
            "arrival": arrival,
            "reliability": reliability,
            "route_kind": route_kind,
            "detour_cost": detour_cost,
        },
        schema={
            "src": pl.UInt64,
            "dest": pl.UInt64,
            "price": pl.UInt64,
            "departure": pl.UInt64,
            "arrival": pl.UInt64,
            "reliability": pl.Int32,
            "route_kind": pl.String,
            "detour_cost": pl.UInt64,
        },
    )

    return TraversalData(nodes, edges, node_count, len(src), destination)


def build_traversal_graphs(data: TraversalData) -> TraversalGraphs:
    rx_graph = rxg.Graph([("airport", data.nodes)], [("flight", data.edges)])
    node_rows = data.nodes.to_dicts()
    edge_rows = data.edges.to_dicts()

    nx_graph = None
    try:
        import networkx as nx

        nx_graph = nx.MultiDiGraph()
        for node in node_rows:
            nx_graph.add_node(node["id"], **node)
        for edge_id, edge in enumerate(edge_rows):
            nx_graph.add_edge(edge["src"], edge["dest"], edge_id=edge_id, **edge)
    except ImportError:
        pass

    ig_graph = None
    try:
        import igraph as ig

        ig_graph = ig.Graph(
            n=data.node_count,
            edges=[(edge["src"], edge["dest"]) for edge in edge_rows],
            directed=True,
        )
        for key in node_rows[0]:
            ig_graph.vs[key] = [node[key] for node in node_rows]
        for key in edge_rows[0]:
            ig_graph.es[key] = [edge[key] for edge in edge_rows]
    except ImportError:
        pass

    return TraversalGraphs(rx_graph, nx_graph, ig_graph, node_rows, edge_rows)


def build_traversal_benches(
    data: TraversalData, graphs: TraversalGraphs, max_paths: int
) -> list[tuple[str, str, Callable[[], Any]]]:
    s = lambda name: pl.col(f"state.{name}")
    d = lambda name: pl.col(f"dest.{name}")
    e = lambda name: pl.col(f"edge.{name}")
    kernel = rxg.Kernel(
        visit=(
            (s("detours") == 0)
            & (~d("closed"))
            & (e("reliability") >= 70)
            & (e("route_kind") != "skip")
            & (s("hops") < 18)
            & ((s("spent") + e("price")) <= 950)
            & (e("departure") >= s("ready_at"))
            & ((s("risk") + d("risk")) <= 90)
        ),
        next_state={
            "spent": s("spent") + e("price"),
            "hops": s("hops") + 1,
            "ready_at": e("arrival") + d("min_connection"),
            "risk": s("risk") + d("risk"),
            "detours": s("detours") + e("detour_cost"),
        },
        stop=pl.col("dest.id") == data.destination,
        initial_state={"spent": 0, "hops": 0, "ready_at": 0, "risk": 0, "detours": 0},
    )
    traversal = rxg.Traversal(kernel, [0], 18, max_paths, "dfs")
    benches: list[tuple[str, str, Callable[[], Any]]] = [
        ("traversal", "rxgraph", lambda: graphs.rxgraph.search(traversal).paths)
    ]

    if graphs.networkx is not None:
        nx_graph = graphs.networkx
        benches.append(
            (
                "traversal",
                "networkx",
                lambda: traversal_python_dfs(
                    lambda node: (
                        (dest, edge)
                        for _, dest, edge in nx_graph.out_edges(node, data=True)
                    ),
                    lambda node: nx_graph.nodes[node],
                    data.destination,
                    max_paths,
                ),
            )
        )

    if graphs.igraph is not None:
        ig_graph = graphs.igraph
        benches.append(
            (
                "traversal",
                "igraph",
                lambda: traversal_python_dfs(
                    lambda node: (
                        (edge.target, edge.attributes())
                        for edge in ig_graph.es.select(_source=node)
                    ),
                    lambda node: ig_graph.vs[node].attributes(),
                    data.destination,
                    max_paths,
                ),
            )
        )

    return benches


def push_traversal_edge(
    src: list[int],
    dest: list[int],
    price: list[int],
    departure: list[int],
    arrival: list[int],
    reliability: list[int],
    route_kind: list[str],
    detour_cost: list[int],
    from_id: int,
    to: int,
    fare: int,
    reliability_score: int,
    kind: str,
    detour: int,
) -> None:
    depart = from_id * 120 + ((to % 9) * 7)
    flight_time = 45 + ((to * 13 + from_id) % 240)
    src.append(from_id)
    dest.append(to)
    price.append(fare)
    departure.append(depart)
    arrival.append(depart + flight_time)
    reliability.append(reliability_score)
    route_kind.append(kind)
    detour_cost.append(detour)


def is_main_stride(stride: int, count: int, step: int) -> bool:
    return stride in {step, step + 1, max(count // 5, 1), max(count // 7, 1)}


def is_stress_node(node: int, count: int, step: int) -> bool:
    return (
        node % step == 0
        or node % max(count // 5, 1) == 0
        or node % max(count // 7, 1) == 0
    )


def traversal_python_dfs(
    out_edges: Any, node_data: Any, destination: int, max_paths: int
) -> list[int]:
    frontier = [
        (0, (0,), {"spent": 0, "hops": 0, "ready_at": 0, "risk": 0, "detours": 0})
    ]
    paths: list[int] = []

    while frontier and len(paths) < max_paths:
        node, path, state = frontier.pop()
        for dest, edge in out_edges(node):
            if dest in path:
                continue
            dest_data = node_data(dest)
            if not traversal_visit(dest_data, edge, state):
                continue
            next_state = {
                "spent": state["spent"] + edge["price"],
                "hops": state["hops"] + 1,
                "ready_at": edge["arrival"] + dest_data["min_connection"],
                "risk": state["risk"] + dest_data["risk"],
                "detours": state["detours"] + edge["detour_cost"],
            }
            if dest == destination:
                paths.append(dest)
                if len(paths) >= max_paths:
                    break
            else:
                frontier.append((dest, (*path, dest), next_state))

    return paths


def traversal_visit(
    dest: dict[str, Any], edge: dict[str, Any], state: dict[str, int]
) -> bool:
    return (
        state["detours"] == 0
        and not dest["closed"]
        and edge["reliability"] >= 70
        and edge["route_kind"] != "skip"
        and state["hops"] < 18
        and state["spent"] + edge["price"] <= 950
        and edge["departure"] >= state["ready_at"]
        and state["risk"] + dest["risk"] <= 90
    )


def run_benchmark(
    algorithm: str,
    scale: str,
    library: str,
    data: GraphData,
    func: Callable[[], Any],
    warmups: int,
    runs: int,
) -> BenchResult:
    result = None
    for _ in range(warmups):
        result = func()

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        values = []
        for _ in range(runs):
            gc.collect()
            started = time.perf_counter()
            result = func()
            elapsed = time.perf_counter() - started
            values.append(elapsed)
    finally:
        if gc_was_enabled:
            gc.enable()

    return BenchResult(
        bench=f"{algorithm}/{scale}",
        algorithm=algorithm,
        scale=scale,
        library=library,
        node_count=data.node_count,
        edge_count=data.edge_count,
        values=values,
        result_size=result_size(result),
    )


def result_size(result: Any) -> int:
    if result is None:
        return 0
    if isinstance(result, list):
        return len(result)
    return 1


def write_pyperf_json(path: Path, results: list[BenchResult]) -> None:
    import pyperf

    path.parent.mkdir(parents=True, exist_ok=True)
    benchmarks = []

    for result in results:
        run = pyperf.Run(
            result.values,
            metadata={
                "name": f"{result.bench}:{result.library}",
                "unit": "second",
                "loops": 1,
                "algorithm": result.algorithm,
                "scale": result.scale,
                "node_count": result.node_count,
                "edge_count": result.edge_count,
                "result_size": result.result_size,
            },
            collect_metadata=False,
        )
        benchmarks.append(pyperf.Benchmark([run]))

    pyperf.BenchmarkSuite(benchmarks).dump(str(path), replace=True)


def print_report(
    results: list[BenchResult], scales: list[Scale], json_path: Path
) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print_plain_report(results, scales, json_path)
        return

    console = Console(width=max(Console().width, 160))
    table = Table(title="rxgraph algorithm benchmarks")
    table.add_column("Bench", no_wrap=True)
    table.add_column("Library", no_wrap=True)
    table.add_column("Median", justify="right", no_wrap=True)
    table.add_column("P90", justify="right", no_wrap=True)
    table.add_column("Best", justify="right", no_wrap=True)
    table.add_column("rxgraph speedup", justify="right", no_wrap=True)
    table.add_column("Graph", justify="right", no_wrap=True)
    table.add_column("Result size", justify="right", no_wrap=True)

    baselines = {
        result.bench: result.median for result in results if result.library == "rxgraph"
    }
    best_p90s = best_p90_by_bench(results)

    ordered_results = sorted(results, key=report_sort_key)
    for index, result in enumerate(ordered_results):
        baseline = baselines.get(result.bench)
        speedup = ""
        speedup_cell: str | Text = ""
        if baseline is not None:
            ratio = result.median / baseline
            speedup = f"{ratio:.1f}x"
            speedup_cell = Text(speedup, style=speedup_style(result.library, ratio))
        library = result.library
        if result.p90 == best_p90s[result.bench]:
            library = f"{library} (best)"
        library_cell: str | Text = library
        if result.library == "rxgraph":
            library_cell = Text(library, style="bold cyan")
        next_result = (
            ordered_results[index + 1] if index + 1 < len(ordered_results) else None
        )
        table.add_row(
            result.bench,
            library_cell,
            format_seconds(result.median),
            format_seconds(result.p90),
            format_seconds(result.best),
            speedup_cell,
            f"{result.node_count:,}n/{result.edge_count:,}e",
            str(result.result_size),
            end_section=next_result is not None
            and next_result.algorithm != result.algorithm,
            style="bold" if result.library == "rxgraph" else None,
        )

    console.print(
        "[bold]Workloads[/bold]: "
        + ", ".join(
            f"{scale.name}={scale.node_count:,} nodes/fanout {scale.extra_edges}"
            for scale in scales
        )
        + ". Synthetic data generation and graph construction are excluded from timings."
    )
    console.print(table)
    console.print(f"[dim]pyperf JSON written to {json_path}[/dim]")


def print_plain_report(
    results: list[BenchResult], scales: list[Scale], json_path: Path
) -> None:
    scale_summary = ", ".join(
        f"{scale.name}={scale.node_count:,} nodes/fanout {scale.extra_edges}"
        for scale in scales
    )
    print(f"Workloads: {scale_summary}")
    baselines = {
        result.bench: result.median for result in results if result.library == "rxgraph"
    }
    best_p90s = best_p90_by_bench(results)
    previous_algorithm = None
    for result in sorted(results, key=report_sort_key):
        if previous_algorithm is not None and result.algorithm != previous_algorithm:
            print()
        previous_algorithm = result.algorithm
        baseline = baselines.get(result.bench)
        speedup = "n/a" if baseline is None else f"{result.median / baseline:.1f}x"
        library = result.library
        if result.p90 == best_p90s[result.bench]:
            library = f"{library} (best)"
        print(
            f"{result.bench:22} {library:17} "
            f"median={format_seconds(result.median):>10} "
            f"p90={format_seconds(result.p90):>10} "
            f"best={format_seconds(result.best):>10} "
            f"rxgraph_speedup={speedup:>7} "
            f"graph={result.node_count:,}n/{result.edge_count:,}e "
            f"size={result.result_size}"
        )
    print(f"pyperf JSON written to {json_path}")


def report_sort_key(result: BenchResult) -> tuple[str, int, int, float]:
    scale_order = {"low": 0, "mid": 1, "high": 2}
    return (
        result.algorithm,
        scale_order.get(result.scale, 99),
        0 if result.library == "rxgraph" else 1,
        result.median,
    )


def best_p90_by_bench(results: list[BenchResult]) -> dict[str, float]:
    best: dict[str, float] = {}
    for result in results:
        current = best.get(result.bench)
        if current is None or result.p90 < current:
            best[result.bench] = result.p90
    return best


def speedup_style(library: str, ratio: float) -> str:
    if library == "rxgraph":
        return "bold"
    if ratio >= 10:
        return "bold bright_green"
    if ratio >= 2:
        return "green"
    if ratio > 1:
        return "dim green"
    if ratio <= 0.5:
        return "bold bright_red"
    if ratio < 1:
        return "red"
    return ""


def format_seconds(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if value < 1e-6:
        return f"{value * 1e9:.1f} ns"
    if value < 1e-3:
        return f"{value * 1e6:.1f} us"
    if value < 1:
        return f"{value * 1e3:.1f} ms"
    return f"{value:.2f} s"


if __name__ == "__main__":
    main()
