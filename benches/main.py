"""Scale-streaming benchmark harness for rxgraph.

The timed functions exclude graph construction. Each scale is built, measured,
printed, and then the next scale starts. ``rxgraph-df`` uses the DataFrame API;
``rxgraph-python`` uses ``Graph.from_edges``.
"""

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
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

SAME = 1.05
INIT = {"spent": 0, "hops": 0, "ready_at": 0, "risk": 0, "detours": 0}


@dataclass(frozen=True, slots=True)
class Scale:
    name: str
    nodes: int
    fanout: int


@dataclass(frozen=True, slots=True)
class Data:
    nodes: pl.DataFrame
    edges: pl.DataFrame
    target_node: int | None = None

    @property
    def target(self) -> int:
        return self.target_node if self.target_node is not None else self.nodes.height - 1

    @property
    def pairs(self) -> list[tuple[int, int]]:
        return list(zip(self.edges["src"].to_list(), self.edges["dest"].to_list()))


@dataclass(frozen=True, slots=True)
class Case:
    alg: str
    lib: str
    run: Callable[[], Any]
    norm: Callable[[Any], Any] = lambda value: value


@dataclass(frozen=True, slots=True)
class Result:
    case: Case
    scale: Scale
    data: Data
    times: list[float]
    size: int

    @property
    def bench(self) -> str:
        return f"{self.case.alg}/{self.scale.name}"

    @property
    def median(self) -> float:
        return statistics.median(self.times)

    @property
    def best(self) -> float:
        return min(self.times)

    @property
    def p90(self) -> float:
        return sorted(self.times)[max(math.ceil(len(self.times) * 0.9) - 1, 0)]


def main() -> None:
    args = args_parser().parse_args()
    console = Console(width=max(Console().width, 160))
    results: list[Result] = []
    for scale in make_scales(args):
        scale_results = run_scale(scale, args)
        results += scale_results
        print_table(console, scale, scale_results)
    write_json(args.json, results)
    console.print(f"[dim]pyperf JSON written to {args.json}[/dim]")


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark rxgraph algorithms.")
    for name, default in [
        ("low_nodes", 10_000),
        ("mid_nodes", 100_000),
        ("high_nodes", 1_000_000),
        ("extra_edges", 4),
        ("runs", 10),
        ("warmups", 3),
        ("max_paths", 50),
        ("traversal_fanout", 256),
    ]:
        p.add_argument(f"--{name.replace('_', '-')}", type=int, default=default)
    p.add_argument("--json", type=Path, default=Path("dist/algorithm-benchmarks.json"))
    return p


def make_scales(args: argparse.Namespace) -> list[Scale]:
    low, mid = max(args.low_nodes, 2), max(args.mid_nodes, args.low_nodes + 1)
    high = max(args.high_nodes, mid + 1)
    return [
        Scale("low", low, max(1, args.extra_edges // 2)),
        Scale("mid", mid, max(1, args.extra_edges * 3 // 4)),
        Scale("high", high, max(1, args.extra_edges)),
    ]


def run_scale(scale: Scale, args: argparse.Namespace) -> list[Result]:
    simple, travel = (
        simple_data(scale.nodes, scale.fanout),
        travel_data(scale.nodes, args.traversal_fanout),
    )
    cases = simple_cases(simple) + travel_cases(travel, args.max_paths)
    return [
        measure(
            case,
            scale,
            travel if case.alg == "traversal" else simple,
            args.warmups,
            args.runs,
        )
        for case in cases
    ]


def simple_data(n: int, fanout: int) -> Data:
    main = max(2, n - max(1, n // 20))
    edges = [
        (src, dst)
        for src in range(main - 1)
        for step in range(1, fanout + 2)
        if (dst := src + step) < main and (step == 1 or dst % step == 0)
    ]
    return Data(
        df({"id": range(n)}, {"id": pl.UInt64}),
        df(
            {
                "id": range(len(edges)),
                "src": [s for s, _ in edges],
                "dest": [d for _, d in edges],
            },
            {"id": pl.UInt64, "src": pl.UInt64, "dest": pl.UInt64},
        ),
        main - 1,
    )


def travel_data(n: int, noise: int) -> Data:
    step = max(n // 18, 1)
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
            max(n // 7, 1),
            max(n // 5, 1),
        }
    )
    rows = []
    for src in range(n - 1):
        rows += [
            flight(
                src,
                min(src + st, n - 1),
                25 + ((st * 3 + src) % 110),
                92 if st in strides[-4:] else 45,
                "route",
                0,
            )
            for st in strides
            if min(src + st, n - 1) != src
        ]
        if src % step == 0 or src % max(n // 5, 1) == 0 or src % max(n // 7, 1) == 0:
            rows += [
                flight(
                    src,
                    dst,
                    20 + i % 50,
                    95 if i % 5 == 0 else 35 + i % 30,
                    "route" if i % 5 == 0 else "skip",
                    int(i % 5 == 0),
                )
                for i in range(max(noise, 0))
                if (dst := 1 + ((src + i * 37 + 17) % (n - 1))) != src
            ]
    for i, row in enumerate(rows):
        row["id"] = i

    return Data(
        df(
            {
                "id": range(n),
                "risk": [(i * 7) % 9 for i in range(n)],
                "min_connection": [35 + ((i * 11) % 50) for i in range(n)],
                "closed": [i not in {0, n - 1} and i % 23 == 0 and i % step != 0 for i in range(n)],
            },
            {
                "id": pl.UInt64,
                "risk": pl.Int32,
                "min_connection": pl.UInt64,
                "closed": pl.Boolean,
            },
        ),
        df(
            rows,
            {
                "id": pl.UInt64,
                "src": pl.UInt64,
                "dest": pl.UInt64,
                "price": pl.UInt64,
                "departure": pl.UInt64,
                "arrival": pl.UInt64,
                "reliability": pl.Int32,
                "route_kind": pl.String,
                "detour_cost": pl.UInt64,
            },
        ),
    )


def df(data: Any, schema: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(data, schema=schema)


def flight(src: int, dst: int, price: int, reliability: int, kind: str, detour: int) -> dict[str, int | str]:
    depart = src * 120 + (dst % 9) * 7
    return {
        "src": src,
        "dest": dst,
        "price": price,
        "departure": depart,
        "arrival": depart + 45 + ((dst * 13 + src) % 240),
        "reliability": reliability,
        "route_kind": kind,
        "detour_cost": detour,
    }


def simple_cases(data: Data) -> list[Case]:
    return [case for lib, graph in simple_graphs(data).items() for case in alg_cases(lib, graph, data)]


def simple_graphs(data: Data) -> dict[str, Any]:
    string_data = string_ids(data)
    graphs = {
        "rxgraph-df": rxg.Graph(data.nodes, data.edges),
        "rxgraph-df-string-ids": rxg.Graph(string_data.nodes, string_data.edges),
        "rxgraph-python": rxg.Graph.from_edges(data.pairs, nodes=range(data.nodes.height)),
    }
    if nx := opt("networkx"):
        graphs["networkx"] = nx.MultiDiGraph()
        graphs["networkx"].add_nodes_from(range(data.nodes.height))
        graphs["networkx"].add_edges_from(data.pairs)
    if ig := opt("igraph"):
        graphs["igraph"] = ig.Graph(n=data.nodes.height, edges=data.pairs, directed=True)
    return graphs


def alg_cases(lib: str, graph: Any, data: Data) -> list[Case]:
    if is_rx(lib):
        start = "0" if lib == "rxgraph-df-string-ids" else 0
        target = str(data.target) if lib == "rxgraph-df-string-ids" else data.target
        bfs_norm = numeric_set if lib == "rxgraph-df-string-ids" else set
        path_norm = numeric_path_sig if lib == "rxgraph-df-string-ids" else path_sig
        comp_norm = numeric_comp_sig if lib == "rxgraph-df-string-ids" else comp_sig
        return [
            Case("bfs", lib, lambda: graph.bfs(start), bfs_norm),
            Case(
                "shortest_path",
                lib,
                lambda: graph.shortest_path(start, target),
                path_norm,
            ),
            Case("degrees", lib, graph.degrees),
            Case("weak_components", lib, graph.weakly_connected_components, comp_norm),
        ]
    if lib == "networkx":
        import networkx as nx

        return [
            Case("bfs", lib, lambda: list(nx.bfs_tree(graph, 0).nodes()), set),
            Case(
                "shortest_path",
                lib,
                lambda: nx.shortest_path(graph, 0, data.target),
                path_sig,
            ),
            Case("degrees", lib, lambda: [d for _, d in graph.degree()]),
            Case(
                "weak_components",
                lib,
                lambda: [list(c) for c in nx.weakly_connected_components(graph)],
                comp_sig,
            ),
        ]
    return [
        Case("bfs", lib, lambda: graph.bfs(0, mode="out")[0], set),
        Case(
            "shortest_path",
            lib,
            lambda: graph.get_shortest_paths(0, to=data.target, mode="out", output="vpath")[0],
            path_sig,
        ),
        Case("degrees", lib, lambda: graph.degree(mode="all")),
        Case(
            "weak_components",
            lib,
            lambda: [list(c) for c in graph.connected_components(mode="weak")],
            comp_sig,
        ),
    ]


def travel_cases(data: Data, max_paths: int) -> list[Case]:
    nodes, edges = data.nodes.to_dicts(), data.edges.to_dicts()
    graphs = travel_graphs(data, nodes, edges)
    traversal = rxg.Traversal(travel_kernel(data.target), [0], 18, max_paths, "dfs")
    string_traversal = rxg.Traversal(travel_kernel(str(data.target)), ["0"], 18, max_paths, "dfs")
    cases = [
        Case(
            "traversal",
            "rxgraph-df",
            lambda: graphs["rxgraph-df"].search(traversal).paths,
        ),
        Case(
            "traversal",
            "rxgraph-df-string-ids",
            lambda: graphs["rxgraph-df-string-ids"].search(string_traversal).paths,
        ),
        Case(
            "traversal",
            "rxgraph-python",
            lambda: graphs["rxgraph-python"].search(traversal).paths,
        ),
    ]
    if nxg := graphs.get("networkx"):
        cases.append(
            Case(
                "traversal",
                "networkx",
                lambda: py_travel(
                    lambda n: ((d, e) for _, d, e in nxg.out_edges(n, data=True)),
                    lambda n: nxg.nodes[n],
                    data.target,
                    max_paths,
                ),
            )
        )
    if igg := graphs.get("igraph"):
        cases.append(
            Case(
                "traversal",
                "igraph",
                lambda: py_travel(
                    lambda n: ((e.target, e.attributes()) for e in igg.es.select(_source=n)),
                    lambda n: igg.vs[n].attributes(),
                    data.target,
                    max_paths,
                ),
            )
        )
    return cases


def travel_graphs(data: Data, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    string_data = string_ids(data)
    graphs = {
        "rxgraph-df": rxg.Graph(data.nodes, data.edges),
        "rxgraph-df-string-ids": rxg.Graph(string_data.nodes, string_data.edges),
        "rxgraph-python": rxg.Graph.from_edges(
            [(e["src"], e["dest"], omit(e, "id", "src", "dest")) for e in edges],
            nodes=[(n["id"], omit(n, "id")) for n in nodes],
        ),
    }
    if nx := opt("networkx"):
        graphs["networkx"] = nx.MultiDiGraph()
        graphs["networkx"].add_nodes_from((n["id"], n) for n in nodes)
        for i, e in enumerate(edges):
            graphs["networkx"].add_edge(e["src"], e["dest"], edge_id=i, **e)
    if ig := opt("igraph"):
        graphs["igraph"] = ig.Graph(
            n=data.nodes.height,
            edges=[(e["src"], e["dest"]) for e in edges],
            directed=True,
        )
        for key in nodes[0]:
            graphs["igraph"].vs[key] = [n[key] for n in nodes]
        for key in edges[0]:
            graphs["igraph"].es[key] = [e[key] for e in edges]
    return graphs


def travel_kernel(target: int | str) -> rxg.Kernel:
    s, d, e = (
        (lambda n: rxg.col(f"state.{n}")),
        (lambda n: rxg.col(f"dest.{n}")),
        (lambda n: rxg.col(f"edge.{n}")),
    )
    return rxg.Kernel(
        visit=(s("detours") == 0)
        & ~d("closed")
        & (e("reliability") >= 70)
        & (e("route_kind") != "skip")
        & (s("hops") < 18)
        & ((s("spent") + e("price")) <= 950)
        & (e("departure") >= s("ready_at"))
        & ((s("risk") + d("risk")) <= 90),
        next_state={
            "spent": s("spent") + e("price"),
            "hops": s("hops") + 1,
            "ready_at": e("arrival") + d("min_connection"),
            "risk": s("risk") + d("risk"),
            "detours": s("detours") + e("detour_cost"),
        },
        stop=rxg.col("dest.id") == target,
        initial_state=INIT,
    )


def string_ids(data: Data) -> Data:
    return Data(
        data.nodes.with_columns(pl.col("id").cast(pl.String)),
        data.edges.with_columns(
            pl.col("id").cast(pl.String),
            pl.col("src").cast(pl.String),
            pl.col("dest").cast(pl.String),
        ),
        data.target,
    )


def py_travel(
    out_edges: Callable[[int], Any],
    node_data: Callable[[int], dict[str, Any]],
    target: int,
    max_paths: int,
) -> list[tuple[int, ...]]:
    frontier, paths = [(0, (0,), INIT)], []
    while frontier and len(paths) < max_paths:
        node, path, state = frontier.pop()
        for dst, edge in out_edges(node):
            dest = node_data(dst)
            if dst in path or not visit(dest, edge, state):
                continue
            next_state = {
                "spent": state["spent"] + edge["price"],
                "hops": state["hops"] + 1,
                "ready_at": edge["arrival"] + dest["min_connection"],
                "risk": state["risk"] + dest["risk"],
                "detours": state["detours"] + edge["detour_cost"],
            }
            next_path = (*path, dst)
            paths.append(next_path) if dst == target else frontier.append((dst, next_path, next_state))
            if len(paths) >= max_paths:
                break
    return paths


def visit(dest: dict[str, Any], edge: dict[str, Any], state: dict[str, int]) -> bool:
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


def measure(case: Case, scale: Scale, data: Data, warmups: int, runs: int) -> Result:
    result = None
    for _ in range(warmups):
        result = case.run()
    gc_enabled, values = gc.isenabled(), []
    gc.disable()
    try:
        for _ in range(runs):
            gc.collect()
            start = time.perf_counter()
            result = case.run()
            values.append(time.perf_counter() - start)
    finally:
        gc.enable() if gc_enabled else None
    return Result(
        case,
        scale,
        data,
        values,
        len(result) if isinstance(result, list) else int(result is not None),
    )


def print_table(console: Console, scale: Scale, results: list[Result]) -> None:
    table = Table(title=f"rxgraph benchmarks: {scale.name}", box=box.ROUNDED)
    for name, justify in [
        ("Bench", "left"),
        ("Library", "left"),
        ("Median", "right"),
        ("P90", "right"),
        ("Best", "right"),
        ("rxgraph speedup", "right"),
        ("Graph", "right"),
        ("Result size", "right"),
    ]:
        table.add_column(name, justify=justify, no_wrap=True)
    best, baselines = best_by_bench(results), rx_baselines(results)
    ordered = sorted(results, key=sort_key)
    for i, r in enumerate(ordered):
        is_best = r.case.lib == best[r.bench]
        lib = Text(
            f"{r.case.lib} (best)" if is_best else r.case.lib,
            style="bold green" if is_best else ("bold cyan" if is_rx(r.case.lib) else ""),
        )
        table.add_row(
            r.bench,
            lib,
            fmt_time(r.median),
            fmt_time(r.p90),
            fmt_time(r.best),
            speed(r, baselines[r.bench]),
            f"{fmt_count(r.data.nodes.height)}n/{fmt_count(r.data.edges.height)}e",
            fmt_count(r.size),
            end_section=i + 1 < len(ordered) and ordered[i + 1].case.alg != r.case.alg,
            style="bold" if is_rx(r.case.lib) else None,
        )
    console.print(f"[bold]Workload[/bold]: {scale.name}={fmt_count(scale.nodes)} nodes/fanout {scale.fanout}")
    console.print(table)


def write_json(path: Path, results: list[Result]) -> None:
    import pyperf

    path.parent.mkdir(parents=True, exist_ok=True)
    pyperf.BenchmarkSuite(
        [
            pyperf.Benchmark(
                [
                    pyperf.Run(
                        r.times,
                        metadata={
                            "name": f"{r.bench}:{r.case.lib}",
                            "unit": "second",
                            "loops": 1,
                            "algorithm": r.case.alg,
                            "scale": r.scale.name,
                            "node_count": r.data.nodes.height,
                            "edge_count": r.data.edges.height,
                            "result_size": r.size,
                        },
                        collect_metadata=False,
                    )
                ]
            )
            for r in results
        ]
    ).dump(str(path), replace=True)


def sort_key(r: Result) -> tuple[str, int, float]:
    return r.case.alg, lib_order(r.case.lib), r.median


def best_by_bench(results: list[Result]) -> dict[str, str]:
    by_bench = {r.bench: [] for r in results}
    for r in results:
        by_bench[r.bench].append(r)
    return {bench: min(rows, key=lambda r: (r.median, lib_order(r.case.lib))).case.lib for bench, rows in by_bench.items()}


def lib_order(library: str) -> int:
    return {"rxgraph-df": 0, "rxgraph-python": 1, "igraph": 2, "networkx": 3}.get(library, 99)


def rx_baselines(results: list[Result]) -> dict[str, float]:
    return {b: next(r.median for r in results if r.bench == b and r.case.lib == "rxgraph-df") for b in {r.bench for r in results}}


def plain_speedup(r: Result, baseline: float) -> str:
    ratio = r.median / baseline
    if 1 / SAME < ratio < SAME:
        return "baseline" if r.case.lib == "rxgraph-df" else "same"
    return f"{(1 / ratio if ratio < 1 else ratio):.1f}x {'faster' if ratio < 1 else 'slower'}"


def speed(r: Result, baseline: float) -> Text:
    label, ratio = plain_speedup(r, baseline), r.median / baseline
    fast, ratio = ratio < 1, (1 / ratio if ratio < 1 else ratio)
    style = (
        "bold"
        if label == "baseline"
        else ""
        if label == "same"
        else (
            "bold bright_green"
            if fast and ratio >= 10
            else "green"
            if fast and ratio >= 2
            else "dim green"
            if fast
            else "bold bright_red"
            if ratio >= 10
            else "red"
            if ratio >= 2
            else "dim red"
        )
    )
    return Text(label, style=style)


def fmt_count(value: int) -> str:
    for suffix, unit in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if abs(value) >= unit:
            scaled = value / unit
            return f"{scaled:.0f}{suffix}" if scaled >= 10 or scaled.is_integer() else f"{scaled:.1f}{suffix}"
    return str(value)


format_count = fmt_count


def fmt_time(value: float) -> str:
    return (
        f"{value * 1e9:.1f} ns"
        if value < 1e-6
        else f"{value * 1e6:.1f} us"
        if value < 1e-3
        else f"{value * 1e3:.1f} ms"
        if value < 1
        else f"{value:.2f} s"
    )


def path_sig(path: list[int] | None) -> tuple[int, int, int] | None:
    return None if not path else (path[0], path[-1], len(path))


def numeric_path_sig(path: list[int | str] | None) -> tuple[int, int, int] | None:
    return None if not path else (int(path[0]), int(path[-1]), len(path))


def comp_sig(components: list[list[int]]) -> list[list[int]]:
    return sorted(sorted(c) for c in components)


def numeric_comp_sig(components: list[list[int | str]]) -> list[list[int]]:
    return sorted(sorted(int(v) for v in c) for c in components)


def numeric_set(values: list[int | str]) -> set[int]:
    return {int(value) for value in values}


def omit(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in keys}


def opt(name: str) -> Any | None:
    try:
        return __import__(name)
    except ImportError:
        return None


def is_rx(library: str) -> bool:
    return library.startswith("rxgraph")


if __name__ == "__main__":
    main()
