"""Scale benchmark harness for rxgraph.

The timed functions exclude graph construction. Each scale is built, measured,
printed, and then the next scale starts. ``rxgraph-df`` uses the DataFrame API;
``rxgraph-python`` uses ``Graph.from_edges``.

NOTE:
For this script, it's mostly AI generated and is very messy, will be revisited.
Benchmarks here should not be trusted at this point.
"""

import argparse
import gc
import math
import statistics
import tempfile
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl
import rxgraph as rxg
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

SAME = 1.05
CACHE_VERSION = 2
CACHE_ROOT = Path(tempfile.gettempdir()) / "rxgraph-bench-cache"


class Lib(StrEnum):
    RX_DF = "rxgraph-df"
    RX_NATIVE_INMEMORY = "rxgraph-native-inmemory"
    RX_NATIVE_PARQUET_EAGER = "rxgraph-native-parquet-eager"
    RX_NATIVE_PARQUET_LAZY = "rxgraph-native-parquet-lazy"
    RX_DF_STRING_IDS = "rxgraph-df-string-ids"
    RX_PYTHON = "rxgraph-python"
    NETWORKX = "networkx"
    IGRAPH = "igraph"


LIB_ORDER = (
    Lib.RX_DF,
    Lib.RX_NATIVE_INMEMORY,
    Lib.RX_NATIVE_PARQUET_EAGER,
    Lib.RX_NATIVE_PARQUET_LAZY,
    Lib.RX_PYTHON,
    Lib.RX_DF_STRING_IDS,
    Lib.IGRAPH,
    Lib.NETWORKX,
)


class Alg(StrEnum):
    BFS = "bfs"
    SHORTEST_PATH = "shortest_path"
    DEGREES = "degrees"
    WEAK_COMPONENTS = "weak_components"
    TRAVERSAL_BFS = "traversal_bfs"
    TRAVERSAL_DFS = "traversal_dfs"


TRAVERSAL_STRATEGIES = (
    (Alg.TRAVERSAL_BFS, "bfs"),
    (Alg.TRAVERSAL_DFS, "dfs"),
)


class ScaleName(StrEnum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class Field(StrEnum):
    ID = "id"
    SRC = "src"
    DEST = "dest"
    PRICE = "price"
    DEPARTURE = "departure"
    ARRIVAL = "arrival"
    RELIABILITY = "reliability"
    ROUTE_KIND = "route_kind"
    DETOUR_COST = "detour_cost"
    RISK = "risk"
    MIN_CONNECTION = "min_connection"
    CLOSED = "closed"
    SPENT = "spent"
    HOPS = "hops"
    READY_AT = "ready_at"
    DETOURS = "detours"


class Scope(StrEnum):
    STATE = "state"
    DEST = "dest"
    EDGE = "edge"


@dataclass(frozen=True, slots=True)
class Scale:
    name: ScaleName
    nodes: int
    fanout: int


@dataclass(frozen=True, slots=True)
class Data:
    nodes: pl.DataFrame
    edges: pl.DataFrame
    target_node: int | None = None
    node_path: Path | None = None
    edge_path: Path | None = None

    @property
    def target(self) -> int:
        return (
            self.target_node if self.target_node is not None else self.nodes.height - 1
        )

    @property
    def pairs(self) -> list[tuple[int, int]]:
        return list(
            zip(self.edges[Field.SRC].to_list(), self.edges[Field.DEST].to_list())
        )


@dataclass(frozen=True, slots=True)
class Case:
    alg: Alg
    lib: Lib
    run: Callable[[], Any]
    norm: Callable[[Any], Any] = lambda value: value
    build_times: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class BuiltGraph:
    graph: Any
    build_times: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class LibraryFilter:
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    def matches(self, library: Lib | str) -> bool:
        name = library.lower()
        included = not self.include or any(term in name for term in self.include)
        excluded = any(term in name for term in self.exclude)
        return included and not excluded


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

    @property
    def graph_build_median(self) -> float:
        return statistics.median(self.case.build_times or (0.0,))


def main() -> None:
    args = args_parser().parse_args()
    console = Console(width=max(Console().width, 160))
    results: list[Result] = []
    for scale in make_scales(args):
        with console.status(
            f"Running {scale.name} benchmark suite...", spinner="boxBounce2"
        ):
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
        ("high_nodes", 10_000_000),
        ("extra_edges", 4),
        ("runs", 10),
        ("warmups", 3),
    ]:
        p.add_argument(f"--{name.replace('_', '-')}", type=int, default=default)
    p.add_argument(
        "--max-paths",
        type=int,
        default=50,
        help="Traversal result cap at mid scale; low/high scales are proportional.",
    )
    p.add_argument(
        "--traversal-fanout",
        type=int,
        default=1,
        help="Unreachable filler edge stride families per node for traversal data.",
    )
    p.add_argument(
        "--include",
        default="",
        help="Comma-separated library name substrings to include, e.g. rxgraph,igraph.",
    )
    p.add_argument(
        "--exclude",
        default="",
        help="Comma-separated library name substrings to exclude, e.g. networkx,igraph.",
    )
    p.add_argument(
        "--cache",
        action="store_true",
        help=f"Cache generated benchmark data as Parquet under {CACHE_ROOT}.",
    )
    p.add_argument("--json", type=Path, default=Path("dist/algorithm-benchmarks.json"))
    return p


def make_scales(args: argparse.Namespace) -> list[Scale]:
    low, mid = max(args.low_nodes, 2), max(args.mid_nodes, args.low_nodes + 1)
    high = max(args.high_nodes, mid + 1)
    return [
        Scale(ScaleName.LOW, low, max(1, args.extra_edges // 2)),
        Scale(ScaleName.MID, mid, max(1, args.extra_edges * 3 // 4)),
        Scale(ScaleName.HIGH, high, max(1, args.extra_edges)),
    ]


def run_scale(scale: Scale, args: argparse.Namespace) -> list[Result]:
    libraries = library_filter(args)
    cache_root = benchmark_cache_root(args)
    case_data: list[tuple[Case, Data]] = []
    if wants_simple(libraries):
        simple = simple_data(scale.nodes, scale.fanout)
        case_data += [(case, simple) for case in simple_cases(simple, libraries)]

    if wants_travel(libraries):
        max_paths = traversal_max_paths(scale, args)
        travel_cache_root = cache_root if wants_file_backed_travel(libraries) else None
        travel = cached_data(
            travel_cache_root,
            "travel",
            scale.nodes,
            f"paths={max_paths}-filler={args.traversal_fanout}",
            scale.nodes - 1,
            lambda: travel_data(scale.nodes, max_paths, args.traversal_fanout),
        )
        travel_graph_cache = travel_graphs(travel, libraries)
        case_data += [
            (case, travel)
            for case in travel_cases(travel, max_paths, libraries, travel_graph_cache)
        ]

    return [
        measure(
            case,
            scale,
            data,
            args.warmups,
            args.runs,
        )
        for case, data in case_data
    ]


def wants_simple(libraries: LibraryFilter) -> bool:
    return any(
        libraries.matches(lib)
        for lib in (
            Lib.RX_DF,
            Lib.RX_DF_STRING_IDS,
            Lib.RX_PYTHON,
            Lib.NETWORKX,
            Lib.IGRAPH,
        )
    )


def wants_travel(libraries: LibraryFilter) -> bool:
    return any(libraries.matches(lib) for lib in LIB_ORDER)


def wants_file_backed_travel(libraries: LibraryFilter) -> bool:
    return libraries.matches(Lib.RX_NATIVE_PARQUET_EAGER) or libraries.matches(
        Lib.RX_NATIVE_PARQUET_LAZY
    )


def traversal_max_paths(scale: Scale, args: argparse.Namespace) -> int:
    return max(1, round(args.max_paths * scale.nodes / max(args.mid_nodes, 1)))


def library_terms(value: str) -> tuple[str, ...]:
    return tuple(term.strip().lower() for term in value.split(",") if term.strip())


def library_filter(args: argparse.Namespace) -> LibraryFilter:
    return LibraryFilter(library_terms(args.include), library_terms(args.exclude))


def benchmark_cache_root(args: argparse.Namespace) -> Path | None:
    return CACHE_ROOT if args.cache else None


def cached_data(
    cache_root: Path | None,
    kind: str,
    nodes: int,
    variant: str,
    target_node: int | None,
    build: Callable[[], Data],
) -> Data:
    if cache_root is None:
        return build()

    path = data_cache_path(cache_root, kind, nodes, variant)
    node_path, edge_path = path / "nodes.parquet", path / "edges.parquet"
    if node_path.exists() and edge_path.exists():
        return Data(
            pl.read_parquet(node_path),
            pl.read_parquet(edge_path),
            target_node,
            node_path,
            edge_path,
        )

    data = build()
    path.mkdir(parents=True, exist_ok=True)
    write_parquet(data.nodes, node_path)
    write_parquet(data.edges, edge_path)
    return Data(data.nodes, data.edges, data.target_node, node_path, edge_path)


def data_cache_path(cache_root: Path, kind: str, nodes: int, variant: str) -> Path:
    return cache_root / f"v{CACHE_VERSION}" / f"{kind}-nodes={nodes}-{variant}"


def write_parquet(frame: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def simple_main(n: int) -> int:
    return max(2, n - max(1, n // 20))


def simple_target_node(n: int) -> int:
    return simple_main(n) - 1


def simple_data(n: int, fanout: int) -> Data:
    main = simple_main(n)
    edge_frames = []
    for step in range(1, fanout + 2):
        count = max(main - step, 0)
        if count == 0:
            continue
        frame = pl.DataFrame(
            {
                Field.SRC: pl.arange(0, count, eager=True, dtype=pl.UInt64),
                Field.DEST: pl.arange(step, main, eager=True, dtype=pl.UInt64),
            }
        )
        if step != 1:
            frame = frame.filter(pl.col(Field.DEST) % step == 0)
        edge_frames.append(frame)
    edges = pl.concat(edge_frames) if edge_frames else edge_frame()
    return Data(
        id_frame(n),
        with_edge_ids(edges),
        main - 1,
    )


def travel_data(n: int, max_paths: int, filler_fanout: int = 1) -> Data:
    target = n - 1
    lanes = min(max(max_paths, 0), max(n - 2, 0))
    frames = []
    if lanes:
        lane_nodes = pl.arange(1, lanes + 1, eager=True, dtype=pl.UInt64)
        frames += [
            edge_frame(
                pl.repeat(0, lanes, eager=True, dtype=pl.UInt64),
                lane_nodes,
                1,
            ),
            edge_frame(
                lane_nodes,
                pl.repeat(target, lanes, eager=True, dtype=pl.UInt64),
                1,
            ),
        ]

    filler_start = lanes + 1
    filler_count = max(target - filler_start, 0)
    if filler_count > 1:
        for step in range(1, max(filler_fanout, 0) + 1):
            src = pl.arange(filler_start, target, eager=True, dtype=pl.UInt64)
            dest = ((src - filler_start + step) % filler_count) + filler_start
            frames.append(edge_frame(src, dest, 1_000))

    edges = pl.concat(frames) if frames else edge_frame()
    return Data(
        id_frame(n),
        with_edge_ids(edges),
        target,
    )


def df(data: Any, schema: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(data, schema=schema)


def id_frame(n: int) -> pl.DataFrame:
    return pl.DataFrame({Field.ID: pl.arange(0, n, eager=True, dtype=pl.UInt64)})


def edge_frame(
    src: pl.Series | None = None,
    dest: pl.Series | None = None,
    price: int | None = None,
) -> pl.DataFrame:
    if src is None or dest is None:
        return pl.DataFrame(
            {
                Field.SRC: pl.Series([], dtype=pl.UInt64),
                Field.DEST: pl.Series([], dtype=pl.UInt64),
                Field.PRICE: pl.Series([], dtype=pl.UInt64),
            }
        )
    frame = pl.DataFrame({Field.SRC: src, Field.DEST: dest})
    if price is not None:
        frame = frame.with_columns(pl.lit(price, dtype=pl.UInt64).alias(Field.PRICE))
    return frame


def with_edge_ids(edges: pl.DataFrame) -> pl.DataFrame:
    return (
        edges.with_row_index(Field.ID)
        .with_columns(pl.col(Field.ID).cast(pl.UInt64))
        .select(Field.ID, Field.SRC, Field.DEST, *edge_payload_fields(edges))
    )


def edge_payload_fields(edges: pl.DataFrame) -> list[Field]:
    return [field for field in [Field.PRICE] if field in edges.columns]


def build_graph(build: Callable[[], Any]) -> BuiltGraph:
    start = time.perf_counter()
    graph = build()
    return BuiltGraph(graph, (time.perf_counter() - start,))


def simple_cases(data: Data, libraries: LibraryFilter = LibraryFilter()) -> list[Case]:
    return [
        case
        for lib, built in simple_graphs(data, libraries).items()
        for case in alg_cases(lib, built.graph, data, built.build_times)
    ]


def simple_graphs(
    data: Data, libraries: LibraryFilter = LibraryFilter()
) -> dict[Lib, BuiltGraph]:
    graphs = {}
    pairs = None

    def edge_pairs() -> list[tuple[int, int]]:
        nonlocal pairs
        if pairs is None:
            pairs = data.pairs
        return pairs

    if libraries.matches(Lib.RX_DF):
        graphs[Lib.RX_DF] = build_graph(lambda: rxg.Graph(data.nodes, data.edges))
    if libraries.matches(Lib.RX_DF_STRING_IDS):
        graphs[Lib.RX_DF_STRING_IDS] = build_graph(lambda: string_id_graph(data))
    if libraries.matches(Lib.RX_PYTHON):
        graphs[Lib.RX_PYTHON] = build_graph(
            lambda: rxg.Graph.from_edges(edge_pairs(), nodes=range(data.nodes.height))
        )
    if libraries.matches(Lib.NETWORKX) and (nx := opt(Lib.NETWORKX)):
        graphs[Lib.NETWORKX] = build_graph(
            lambda: simple_networkx_graph(nx, data.nodes.height, edge_pairs())
        )
    if libraries.matches(Lib.IGRAPH) and (ig := opt(Lib.IGRAPH)):
        graphs[Lib.IGRAPH] = build_graph(
            lambda: ig.Graph(n=data.nodes.height, edges=edge_pairs(), directed=True)
        )
    return graphs


def simple_networkx_graph(
    nx: Any, node_count: int, pairs: list[tuple[int, int]]
) -> Any:
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(pairs)
    return graph


def string_id_graph(data: Data) -> Any:
    string_data = string_ids(data)
    return rxg.Graph(string_data.nodes, string_data.edges)


def alg_cases(
    lib: Lib, graph: Any, data: Data, build_times: tuple[float, ...]
) -> list[Case]:
    if is_rx(lib):
        start = "0" if lib == Lib.RX_DF_STRING_IDS else 0
        target = str(data.target) if lib == Lib.RX_DF_STRING_IDS else data.target
        bfs_norm = numeric_set if lib == Lib.RX_DF_STRING_IDS else set
        path_norm = numeric_path_sig if lib == Lib.RX_DF_STRING_IDS else path_sig
        comp_norm = numeric_comp_sig if lib == Lib.RX_DF_STRING_IDS else comp_sig
        return [
            Case(Alg.BFS, lib, lambda: graph.bfs(start), bfs_norm, build_times),
            Case(
                Alg.SHORTEST_PATH,
                lib,
                lambda: graph.shortest_path(start, target),
                path_norm,
                build_times,
            ),
            Case(Alg.DEGREES, lib, graph.degrees, build_times=build_times),
            Case(
                Alg.WEAK_COMPONENTS,
                lib,
                graph.weakly_connected_components,
                comp_norm,
                build_times,
            ),
        ]
    if lib == Lib.NETWORKX:
        import networkx as nx

        return [
            Case(
                Alg.BFS,
                lib,
                lambda: list(nx.bfs_tree(graph, 0).nodes()),
                set,
                build_times,
            ),
            Case(
                Alg.SHORTEST_PATH,
                lib,
                lambda: nx.shortest_path(graph, 0, data.target),
                path_sig,
                build_times,
            ),
            Case(
                Alg.DEGREES,
                lib,
                lambda: [d for _, d in graph.degree()],
                build_times=build_times,
            ),
            Case(
                Alg.WEAK_COMPONENTS,
                lib,
                lambda: [list(c) for c in nx.weakly_connected_components(graph)],
                comp_sig,
                build_times,
            ),
        ]
    return [
        Case(Alg.BFS, lib, lambda: graph.bfs(0, mode="out")[0], set, build_times),
        Case(
            Alg.SHORTEST_PATH,
            lib,
            lambda: graph.get_shortest_paths(
                0, to=data.target, mode="out", output="vpath"
            )[0],
            path_sig,
            build_times,
        ),
        Case(
            Alg.DEGREES,
            lib,
            lambda: graph.degree(mode="all"),
            build_times=build_times,
        ),
        Case(
            Alg.WEAK_COMPONENTS,
            lib,
            lambda: [list(c) for c in graph.connected_components(mode="weak")],
            comp_sig,
            build_times,
        ),
    ]


def travel_cases(
    data: Data,
    max_paths: int,
    libraries: LibraryFilter = LibraryFilter(),
    graphs: dict[Lib, BuiltGraph] | None = None,
) -> list[Case]:
    graphs = graphs if graphs is not None else travel_graphs(data, libraries)
    cases = []
    for alg, strategy in TRAVERSAL_STRATEGIES:
        traversal = weighted_budget_search_kwargs(data.target, [0], max_paths, strategy)
        string_traversal = weighted_budget_search_kwargs(
            str(data.target), ["0"], max_paths, strategy
        )
        if Lib.RX_DF in graphs:
            built = graphs[Lib.RX_DF]
            cases.append(
                Case(
                    alg,
                    Lib.RX_DF,
                    lambda graph=built.graph, traversal=traversal: (
                        graph.search(**traversal).paths
                    ),
                    build_times=built.build_times,
                )
            )
        if Lib.RX_NATIVE_INMEMORY in graphs:
            built = graphs[Lib.RX_NATIVE_INMEMORY]
            native = weighted_budget_kernel_kwargs(
                data.target, [0], max_paths, strategy
            )
            cases.append(
                Case(
                    alg,
                    Lib.RX_NATIVE_INMEMORY,
                    lambda graph=built.graph, native=native: (
                        graph.search(**native).paths
                    ),
                    build_times=built.build_times,
                )
            )
        if Lib.RX_NATIVE_PARQUET_EAGER in graphs:
            built = graphs[Lib.RX_NATIVE_PARQUET_EAGER]
            native = weighted_budget_kernel_kwargs(
                data.target, [0], max_paths, strategy
            )
            cases.append(
                Case(
                    alg,
                    Lib.RX_NATIVE_PARQUET_EAGER,
                    lambda graph=built.graph, native=native: (
                        graph.search(**native).paths
                    ),
                    build_times=built.build_times,
                )
            )
        if Lib.RX_NATIVE_PARQUET_LAZY in graphs:
            built = graphs[Lib.RX_NATIVE_PARQUET_LAZY]
            native = weighted_budget_kernel_kwargs(
                data.target, [0], max_paths, strategy
            )
            cases.append(
                Case(
                    alg,
                    Lib.RX_NATIVE_PARQUET_LAZY,
                    lambda graph=built.graph, native=native: (
                        graph.search(**native).paths
                    ),
                    build_times=built.build_times,
                )
            )
        if Lib.RX_DF_STRING_IDS in graphs:
            built = graphs[Lib.RX_DF_STRING_IDS]
            cases.append(
                Case(
                    alg,
                    Lib.RX_DF_STRING_IDS,
                    lambda graph=built.graph, string_traversal=string_traversal: (
                        graph.search(**string_traversal).paths
                    ),
                    build_times=built.build_times,
                )
            )
        if Lib.RX_PYTHON in graphs:
            built = graphs[Lib.RX_PYTHON]
            cases.append(
                Case(
                    alg,
                    Lib.RX_PYTHON,
                    lambda graph=built.graph, traversal=traversal: (
                        graph.search(**traversal).paths
                    ),
                    build_times=built.build_times,
                )
            )
        if built := graphs.get(Lib.NETWORKX):
            nxg = built.graph
            cases.append(
                Case(
                    alg,
                    Lib.NETWORKX,
                    lambda strategy=strategy: py_weighted_budget(
                        lambda n: ((d, e) for _, d, e in nxg.out_edges(n, data=True)),
                        data.target,
                        max_paths,
                        strategy=strategy,
                    ),
                    build_times=built.build_times,
                )
            )
        if built := graphs.get(Lib.IGRAPH):
            igg = built.graph
            cases.append(
                Case(
                    alg,
                    Lib.IGRAPH,
                    lambda strategy=strategy: py_weighted_budget(
                        lambda n: (
                            (e.target, e.attributes()) for e in igg.es.select(_source=n)
                        ),
                        data.target,
                        max_paths,
                        strategy=strategy,
                    ),
                    build_times=built.build_times,
                )
            )
    return cases


def travel_graphs(
    data: Data, libraries: LibraryFilter = LibraryFilter()
) -> dict[Lib, BuiltGraph]:
    graphs = {}
    nodes = None
    edges = None

    def rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nonlocal nodes, edges
        if nodes is None or edges is None:
            nodes, edges = data.nodes.to_dicts(), data.edges.to_dicts()
        return nodes, edges

    if libraries.matches(Lib.RX_DF) or libraries.matches(Lib.RX_NATIVE_INMEMORY):
        built = build_graph(lambda: rxg.Graph(data.nodes, data.edges))
        if libraries.matches(Lib.RX_DF):
            graphs[Lib.RX_DF] = built
        if libraries.matches(Lib.RX_NATIVE_INMEMORY):
            graphs[Lib.RX_NATIVE_INMEMORY] = built
    if libraries.matches(Lib.RX_NATIVE_PARQUET_EAGER) or libraries.matches(
        Lib.RX_NATIVE_PARQUET_LAZY
    ):
        node_path, edge_path = parquet_paths(data, "travel")
        if libraries.matches(Lib.RX_NATIVE_PARQUET_EAGER):
            graphs[Lib.RX_NATIVE_PARQUET_EAGER] = build_graph(
                lambda: rxg.Graph.from_parquet(node_path, edge_path, payloads="eager")
            )
        if libraries.matches(Lib.RX_NATIVE_PARQUET_LAZY):
            graphs[Lib.RX_NATIVE_PARQUET_LAZY] = build_graph(
                lambda: rxg.Graph.from_parquet(node_path, edge_path, payloads="lazy")
            )
    if libraries.matches(Lib.RX_DF_STRING_IDS):
        graphs[Lib.RX_DF_STRING_IDS] = build_graph(lambda: string_id_graph(data))
    if libraries.matches(Lib.RX_PYTHON):
        graphs[Lib.RX_PYTHON] = build_graph(lambda: travel_rxgraph_python_graph(rows()))
    if libraries.matches(Lib.NETWORKX) and (nx := opt(Lib.NETWORKX)):
        graphs[Lib.NETWORKX] = build_graph(lambda: travel_networkx_graph(nx, rows()))
    if libraries.matches(Lib.IGRAPH) and (ig := opt(Lib.IGRAPH)):
        graphs[Lib.IGRAPH] = build_graph(
            lambda: travel_igraph_graph(ig, data.nodes.height, rows())
        )
    return graphs


def travel_networkx_graph(
    nx: Any, rows: tuple[list[dict[str, Any]], list[dict[str, Any]]]
) -> Any:
    node_rows, edge_rows = rows
    graph = nx.MultiDiGraph()
    graph.add_nodes_from((n[Field.ID], n) for n in node_rows)
    for i, e in enumerate(edge_rows):
        graph.add_edge(e[Field.SRC], e[Field.DEST], edge_id=i, **e)
    return graph


def travel_rxgraph_python_graph(
    rows: tuple[list[dict[str, Any]], list[dict[str, Any]]],
) -> Any:
    node_rows, edge_rows = rows
    return rxg.Graph.from_edges(
        (
            (
                e[Field.SRC],
                e[Field.DEST],
                omit(e, Field.ID, Field.SRC, Field.DEST),
            )
            for e in edge_rows
        ),
        nodes=((n[Field.ID], omit(n, Field.ID)) for n in node_rows),
    )


def travel_igraph_graph(
    ig: Any, node_count: int, rows: tuple[list[dict[str, Any]], list[dict[str, Any]]]
) -> Any:
    node_rows, edge_rows = rows
    graph = ig.Graph(
        n=node_count,
        edges=[(e[Field.SRC], e[Field.DEST]) for e in edge_rows],
        directed=True,
    )
    for key in node_rows[0]:
        graph.vs[key] = [n[key] for n in node_rows]
    for key in edge_rows[0]:
        graph.es[key] = [e[key] for e in edge_rows]
    return graph


def weighted_budget_search_kwargs(
    target: int | str,
    start_nodes: list[int | str],
    max_paths: int,
    strategy: str = "bfs",
) -> dict[str, Any]:
    s, d, e = (
        (lambda n: rxg.col(f"{Scope.STATE}.{n}")),
        (lambda n: rxg.col(f"{Scope.DEST}.{n}")),
        (lambda n: rxg.col(f"{Scope.EDGE}.{n}")),
    )
    return {
        "start_nodes": start_nodes,
        "visit": (s(Field.SPENT) + e(Field.PRICE)) <= 950,
        "next_state": {
            Field.SPENT: s(Field.SPENT) + e(Field.PRICE),
        },
        "stop": d(Field.ID) == target,
        "initial_state": {Field.SPENT: 0},
        "max_depth": 18,
        "max_paths": max_paths,
        "strategy": strategy,
    }


def weighted_budget_kernel_kwargs(
    target: int | str,
    start_nodes: list[int | str],
    max_paths: int,
    strategy: str = "bfs",
) -> dict[str, Any]:
    return {
        "start_nodes": start_nodes,
        "kernel": "weighted_budget",
        "params": {
            "weight_col": Field.PRICE.value,
            "budget": 950,
            "target": target,
        },
        "columns": [Field.PRICE.value],
        "max_depth": 18,
        "max_paths": max_paths,
        "strategy": strategy,
    }


def string_ids(data: Data) -> Data:
    return Data(
        data.nodes.with_columns(pl.col(Field.ID).cast(pl.String)),
        data.edges.with_columns(
            pl.col(Field.ID).cast(pl.String),
            pl.col(Field.SRC).cast(pl.String),
            pl.col(Field.DEST).cast(pl.String),
        ),
        data.target,
    )


def parquet_paths(data: Data, kind: str) -> tuple[Path, Path]:
    if data.node_path is not None and data.edge_path is not None:
        return data.node_path, data.edge_path

    path = (
        CACHE_ROOT
        / f"v{CACHE_VERSION}"
        / f"adhoc-{kind}-nodes={data.nodes.height}-edges={data.edges.height}"
    )
    node_path, edge_path = path / "nodes.parquet", path / "edges.parquet"
    if not node_path.exists() or not edge_path.exists():
        path.mkdir(parents=True, exist_ok=True)
        write_parquet(data.nodes, node_path)
        write_parquet(data.edges, edge_path)
    return node_path, edge_path


def py_weighted_budget(
    out_edges: Callable[[int], Any],
    target: int,
    max_paths: int,
    budget: int = 950,
    strategy: str = "bfs",
) -> list[tuple[int, ...]]:
    frontier, paths = deque([(0, (0,), 0)]), []
    while frontier and len(paths) < max_paths:
        node, path, spent = frontier.popleft() if strategy == "bfs" else frontier.pop()
        for dst, edge in out_edges(node):
            next_spent = spent + edge[Field.PRICE]
            if dst in path or next_spent > budget:
                continue
            next_path = (*path, dst)
            if dst == target:
                paths.append(next_path)
            else:
                frontier.append((dst, next_path, next_spent))
            if len(paths) >= max_paths:
                break
    return paths


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
        ("Test setup", "right"),
        ("Median", "right"),
        ("P90", "right"),
        ("Best", "right"),
        ("rxgraph speedup", "right"),
        ("Graph", "right"),
        ("Result size", "right"),
    ]:
        table.add_column(name, justify=justify, no_wrap=True)
    best, baselines = best_by_bench(results), speedup_baselines(results)
    ordered = sorted(results, key=sort_key)
    for i, r in enumerate(ordered):
        is_best = r.case.lib == best[r.bench]
        lib = Text(
            f"{r.case.lib} (best)" if is_best else r.case.lib,
            style="bold green"
            if is_best
            else ("bold cyan" if is_rx(r.case.lib) else ""),
        )
        table.add_row(
            r.bench,
            lib,
            Text(fmt_time(r.graph_build_median), style="dim"),
            fmt_time(r.median),
            fmt_time(r.p90),
            fmt_time(r.best),
            speed(r, baselines[r.bench]),
            f"{fmt_count(r.data.nodes.height)}n/{fmt_count(r.data.edges.height)}e",
            fmt_count(r.size),
            end_section=i + 1 < len(ordered) and ordered[i + 1].case.alg != r.case.alg,
            style="bold" if is_rx(r.case.lib) else None,
        )
    console.print(
        f"[bold]Workload[/bold]: {scale.name}={fmt_count(scale.nodes)} nodes/fanout {scale.fanout}"
    )
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
                            "graph_build_median": r.graph_build_median,
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
    return {
        bench: min(rows, key=lambda r: (r.median, lib_order(r.case.lib))).case.lib
        for bench, rows in by_bench.items()
    }


def lib_order(library: Lib | str) -> int:
    return LIB_ORDER.index(library)


def speedup_baselines(results: list[Result]) -> dict[str, Result]:
    by_bench = {r.bench: [] for r in results}
    for r in results:
        by_bench[r.bench].append(r)
    return {
        bench: min(
            [r for r in rows if is_rx(r.case.lib)] or rows,
            key=lambda r: (r.median, lib_order(r.case.lib)),
        )
        for bench, rows in by_bench.items()
    }


def plain_speedup(r: Result, baseline: Result | float) -> str:
    baseline_median = baseline.median if isinstance(baseline, Result) else baseline
    baseline_lib = baseline.case.lib if isinstance(baseline, Result) else Lib.RX_DF
    ratio = r.median / baseline_median
    if 1 / SAME < ratio < SAME:
        return "baseline" if r.case.lib == baseline_lib else "same"
    return f"{(1 / ratio if ratio < 1 else ratio):.1f}x {'faster' if ratio < 1 else 'slower'}"


def speed(r: Result, baseline: Result) -> Text:
    label, ratio = plain_speedup(r, baseline), r.median / baseline.median
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
            return (
                f"{scaled:.0f}{suffix}"
                if scaled >= 10 or scaled.is_integer()
                else f"{scaled:.1f}{suffix}"
            )
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


def opt(name: Lib | str) -> Any | None:
    try:
        return __import__(name)
    except ImportError:
        return None


def is_rx(library: Lib | str) -> bool:
    return library in (
        Lib.RX_DF,
        Lib.RX_NATIVE_INMEMORY,
        Lib.RX_NATIVE_PARQUET_EAGER,
        Lib.RX_NATIVE_PARQUET_LAZY,
        Lib.RX_DF_STRING_IDS,
        Lib.RX_PYTHON,
    )


if __name__ == "__main__":
    main()
