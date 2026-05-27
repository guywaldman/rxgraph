from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl
import rxgraph as rxg


@dataclass
class GraphData:
    nodes: pl.DataFrame
    edges: pl.DataFrame
    node_count: int
    edge_count: int
    source: int
    target: int


@dataclass
class LibraryGraphs:
    rxgraph: rxg.Graph
    networkx: Any | None
    igraph: Any | None


@dataclass(frozen=True)
class Case:
    name: str
    library: str
    run: Callable[[], Any]
    normalize: Callable[[Any], Any] = lambda value: value


def build_graph_data(node_count: int, extra_edges: int) -> GraphData:
    node_count = max(node_count, 2)
    ids = list(range(node_count))
    src = []
    dest = []

    for node in range(node_count - 1):
        src.append(node)
        dest.append(node + 1)

        for step in range(2, extra_edges + 2):
            target = node + step
            if target < node_count and (node + step) % step == 0:
                src.append(node)
                dest.append(target)

    nodes = pl.DataFrame({"id": ids}, schema={"id": pl.UInt64})
    edges = pl.DataFrame(
        {"src": src, "dest": dest}, schema={"src": pl.UInt64, "dest": pl.UInt64}
    )

    return GraphData(
        nodes=nodes,
        edges=edges,
        node_count=node_count,
        edge_count=len(src),
        source=0,
        target=node_count - 1,
    )


def build_library_graphs(data: GraphData) -> LibraryGraphs:
    rx_graph = rxg.Graph([("n", data.nodes)], [("e", data.edges)])

    nx_graph = None
    try:
        import networkx as nx

        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from(range(data.node_count))
        nx_graph.add_edges_from(
            zip(data.edges["src"].to_list(), data.edges["dest"].to_list())
        )
    except ImportError:
        pass

    ig_graph = None
    try:
        import igraph as ig

        ig_graph = ig.Graph(
            n=data.node_count,
            edges=list(zip(data.edges["src"].to_list(), data.edges["dest"].to_list())),
            directed=True,
        )
    except ImportError:
        pass

    return LibraryGraphs(rx_graph, nx_graph, ig_graph)


def algorithm_cases(data: GraphData, graphs: LibraryGraphs) -> list[Case]:
    cases = [
        Case(
            "bfs", "rxgraph", lambda: graphs.rxgraph.bfs(data.source), normalize_nodes
        ),
        Case(
            "shortest_path",
            "rxgraph",
            lambda: graphs.rxgraph.shortest_path(data.source, data.target),
            normalize_path,
        ),
        Case("degrees", "rxgraph", graphs.rxgraph.degrees),
        Case(
            "weak_components",
            "rxgraph",
            graphs.rxgraph.weakly_connected_components,
            normalize_components,
        ),
    ]

    if graphs.networkx is not None:
        cases.extend(networkx_cases(data, graphs.networkx))
    if graphs.igraph is not None:
        cases.extend(igraph_cases(data, graphs.igraph))
    return cases


def networkx_cases(data: GraphData, graph: Any) -> list[Case]:
    import networkx as nx

    return [
        Case(
            "bfs",
            "networkx",
            lambda: list(nx.bfs_tree(graph, data.source).nodes()),
            normalize_nodes,
        ),
        Case(
            "shortest_path",
            "networkx",
            lambda: nx.shortest_path(graph, data.source, data.target),
            normalize_path,
        ),
        Case("degrees", "networkx", lambda: [degree for _, degree in graph.degree()]),
        Case(
            "weak_components",
            "networkx",
            lambda: [
                list(component) for component in nx.weakly_connected_components(graph)
            ],
            normalize_components,
        ),
    ]


def igraph_cases(data: GraphData, graph: Any) -> list[Case]:
    return [
        Case(
            "bfs",
            "igraph",
            lambda: graph.bfs(data.source, mode="out")[0],
            normalize_nodes,
        ),
        Case(
            "shortest_path",
            "igraph",
            lambda: graph.get_shortest_paths(
                data.source, to=data.target, mode="out", output="vpath"
            )[0],
            normalize_path,
        ),
        Case("degrees", "igraph", lambda: graph.degree(mode="all")),
        Case(
            "weak_components",
            "igraph",
            lambda: [
                list(component) for component in graph.connected_components(mode="weak")
            ],
            normalize_components,
        ),
    ]


def normalize_nodes(nodes: list[int]) -> set[int]:
    return set(nodes)


def normalize_path(path: list[int] | None) -> tuple[int, int, int] | None:
    if not path:
        return None
    return (path[0], path[-1], len(path))


def normalize_components(components: list[list[int]]) -> list[list[int]]:
    return sorted(sorted(component) for component in components)
