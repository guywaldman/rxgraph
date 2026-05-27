from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Any

import polars as pl

import rxgraph as rxg


@dataclass
class Workload:
    airports: pl.DataFrame
    flights: pl.DataFrame
    destination: int


@dataclass
class BenchResult:
    name: str
    build: float
    searches: list[float]
    paths: int
    evaluated: int
    accepted: int

    @property
    def median_search(self) -> float:
        return statistics.median(self.searches)

    @property
    def min_search(self) -> float:
        return min(self.searches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("airports", type=int, nargs="?", default=10_000)
    parser.add_argument("--max-paths", type=int, default=50)
    parser.add_argument("--hub-decoys", type=int, default=2048)
    parser.add_argument("--hub-branches", type=int, default=512)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--strategy",
        choices=["dfs", "bfs"],
        default="dfs",
    )
    parser.add_argument(
        "--parallel",
        choices=["auto", "off", "on"],
        default="auto",
    )
    parser.add_argument("--parallel-min-frontier", type=int, default=512)
    parser.add_argument("--parallel-min-edges", type=int, default=8192)
    parser.add_argument(
        "--bench-bfs-parallel",
        action="store_true",
        help="also run rxgraph BFS with parallel off and forced on",
    )
    args = parser.parse_args()

    workload = build_workload(args.airports, args.hub_decoys, args.hub_branches)

    results = [
        bench_rxgraph(
            "rxgraph",
            workload,
            args.max_paths,
            args.runs,
            args.strategy,
            args.parallel,
            args.parallel_min_frontier,
            args.parallel_min_edges,
        )
    ]
    if args.bench_bfs_parallel:
        results.extend(
            [
                bench_rxgraph(
                    "rxgraph-bfs",
                    workload,
                    args.max_paths,
                    args.runs,
                    "bfs",
                    "off",
                    args.parallel_min_frontier,
                    args.parallel_min_edges,
                ),
                bench_rxgraph(
                    "rxgraph-bfs-parallel",
                    workload,
                    args.max_paths,
                    args.runs,
                    "bfs",
                    "on",
                    0,
                    0,
                ),
            ]
        )
    results.extend(
        [
            bench_networkx(workload, args.max_paths, args.runs),
            bench_igraph(workload, args.max_paths, args.runs),
        ]
    )
    print_results(results)


def bench_rxgraph(
    name: str,
    workload: Workload,
    max_paths: int,
    runs: int,
    strategy: str,
    parallel: str,
    parallel_min_frontier: int,
    parallel_min_edges: int,
) -> BenchResult:
    started = time.perf_counter()
    graph = rxg.Graph(
        [("airport", workload.airports)],
        [("flight", workload.flights)],
    )
    build = time.perf_counter() - started

    s = lambda name: pl.col(f"state.{name}")
    d = lambda name: pl.col(f"dest.{name}")
    e = lambda name: pl.col(f"edge.{name}")
    kernel = rxg.Kernel(
        visit=(
            (s("detours") == 0)
            & (~d("closed"))
            & (e("reliability") >= 70)
            & (e("route_kind") != "decoy")
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
        stop=pl.col("dest.id") == workload.destination,
        initial_state={"spent": 0, "hops": 0, "ready_at": 0, "risk": 0, "detours": 0},
    )
    traversal = rxg.Traversal(
        kernel,
        [0],
        18,
        max_paths,
        strategy,
        parallel,
        parallel_min_frontier,
        parallel_min_edges,
    )

    searches: list[float] = []
    result = None
    for _ in range(runs):
        started = time.perf_counter()
        result = graph.search(traversal)
        searches.append(time.perf_counter() - started)

    assert result is not None
    return BenchResult(
        name,
        build,
        searches,
        len(result.paths),
        result.stats.evaluated_edges,
        result.stats.accepted_edges,
    )


def bench_networkx(workload: Workload, max_paths: int, runs: int) -> BenchResult:
    try:
        import networkx as nx
    except ImportError:
        return BenchResult("networkx", float("nan"), [float("nan")], 0, 0, 0)

    airports, flights = to_python_rows(workload)

    started = time.perf_counter()
    graph = nx.MultiDiGraph()
    for airport in airports:
        graph.add_node(airport["id"], **airport)
    for edge_id, flight in enumerate(flights):
        graph.add_edge(flight["src"], flight["dest"], edge_id=edge_id, **flight)
    build = time.perf_counter() - started

    searches: list[float] = []
    paths = evaluated = accepted = 0
    for _ in range(runs):
        started = time.perf_counter()
        paths, evaluated, accepted = python_dfs(
            lambda node: (
                (dest, data) for _, dest, data in graph.out_edges(node, data=True)
            ),
            lambda node: graph.nodes[node],
            workload.destination,
            max_paths,
        )
        searches.append(time.perf_counter() - started)

    return BenchResult(
        "networkx",
        build,
        searches,
        paths,
        evaluated,
        accepted,
    )


def bench_igraph(workload: Workload, max_paths: int, runs: int) -> BenchResult:
    try:
        import igraph as ig
    except ImportError:
        return BenchResult("igraph", float("nan"), [float("nan")], 0, 0, 0)

    airports, flights = to_python_rows(workload)

    started = time.perf_counter()
    graph = ig.Graph(
        n=len(airports),
        edges=[(flight["src"], flight["dest"]) for flight in flights],
        directed=True,
    )
    for key in airports[0]:
        graph.vs[key] = [airport[key] for airport in airports]
    for key in flights[0]:
        graph.es[key] = [flight[key] for flight in flights]
    build = time.perf_counter() - started

    searches: list[float] = []
    paths = evaluated = accepted = 0
    for _ in range(runs):
        started = time.perf_counter()
        paths, evaluated, accepted = python_dfs(
            lambda node: (
                (edge.target, edge.attributes())
                for edge in graph.es.select(_source=node)
            ),
            lambda node: graph.vs[node].attributes(),
            workload.destination,
            max_paths,
        )
        searches.append(time.perf_counter() - started)

    return BenchResult(
        "igraph",
        build,
        searches,
        paths,
        evaluated,
        accepted,
    )


def print_results(results: list[BenchResult]) -> None:
    rxgraph = next(result for result in results if result.name == "rxgraph")

    for result in results:
        if result.name == rxgraph.name:
            relative = "1.00x"
        elif result.median_search == result.median_search:
            relative = f"{result.median_search / rxgraph.median_search:.2f}x"
        else:
            relative = "n/a"

        print(
            result.name,
            f"build={result.build:.3f}s",
            f"search_median={result.median_search:.6f}s",
            f"search_min={result.min_search:.6f}s",
            f"relative_to_rxgraph={relative}",
            f"paths={result.paths}",
            f"evaluated={result.evaluated}",
            f"accepted={result.accepted}",
        )


def python_dfs(
    out_edges: Any, node_data: Any, destination: int, max_paths: int
) -> tuple[int, int, int]:
    frontier = [
        (0, (0,), {"spent": 0, "hops": 0, "ready_at": 0, "risk": 0, "detours": 0})
    ]
    paths = 0
    evaluated = 0
    accepted = 0

    while frontier and paths < max_paths:
        node, path, state = frontier.pop()

        for dest, edge in out_edges(node):
            evaluated += 1

            if dest in path:
                continue

            dest_data = node_data(dest)

            if not visit(dest_data, edge, state):
                continue

            accepted += 1
            next_state = {
                "spent": state["spent"] + edge["price"],
                "hops": state["hops"] + 1,
                "ready_at": edge["arrival"] + dest_data["min_connection"],
                "risk": state["risk"] + dest_data["risk"],
                "detours": state["detours"] + edge["detour_cost"],
            }

            if dest == destination:
                paths += 1
                if paths >= max_paths:
                    break
            else:
                frontier.append((dest, (*path, dest), next_state))

    return paths, evaluated, accepted


def visit(dest: dict[str, Any], edge: dict[str, Any], state: dict[str, int]) -> bool:
    return (
        state["detours"] == 0
        and not dest["closed"]
        and edge["reliability"] >= 70
        and edge["route_kind"] != "decoy"
        and state["hops"] < 18
        and state["spent"] + edge["price"] <= 950
        and edge["departure"] >= state["ready_at"]
        and state["risk"] + dest["risk"] <= 90
    )


def build_workload(count: int, hub_decoys: int, hub_branches: int) -> Workload:
    count = max(count, 2)
    step = max(count // 18, 1)
    destination = count - 1

    airports = pl.DataFrame(
        {
            "id": range(count),
            "code": [f"AP{i:06}" for i in range(count)],
            "risk": [(i * 7) % 9 for i in range(count)],
            "min_connection": [35 + ((i * 11) % 50) for i in range(count)],
            "closed": [
                i != 0 and i + 1 != count and i % 23 == 0 and i % step != 0
                for i in range(count)
            ],
        },
        schema={
            "id": pl.UInt64,
            "code": pl.String,
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
        set(
            [
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
                max(count // 7, 1),
                max(count // 5, 1),
            ]
        )
    )

    for from_id in range(count - 1):
        for stride in strides:
            to = min(from_id + stride, count - 1)
            if to == from_id:
                continue
            push_flight(
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
                92
                if is_main_stride(stride, count, step)
                else 45 + ((from_id * 5 + stride * 3) % 20),
                "route",
                0,
            )

        if is_stress_hub(from_id, count, step):
            for decoy in range(hub_decoys):
                to = 1 + ((from_id + decoy * 37 + 17) % (count - 1))
                if to != from_id:
                    push_flight(
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
                        15 + (decoy % 50),
                        35 + (decoy % 30),
                        "decoy",
                        0,
                    )
            for branch in range(hub_branches):
                to = 1 + ((from_id + branch * 53 + 29) % (count - 1))
                if to != from_id and to + 1 != count:
                    push_flight(
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
                        20 + (branch % 35),
                        95,
                        "branch",
                        1,
                    )

    flights = pl.DataFrame(
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

    return Workload(airports, flights, destination)


def push_flight(
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


def is_stress_hub(airport: int, count: int, step: int) -> bool:
    return (
        airport % step == 0
        or airport % max(count // 5, 1) == 0
        or airport % max(count // 7, 1) == 0
    )


def to_python_rows(
    workload: Workload,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    airports = workload.airports.to_dicts()
    flights = workload.flights.to_dicts()
    return airports, flights


if __name__ == "__main__":
    main()
