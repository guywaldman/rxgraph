from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Self

import polars as pl

from . import _rxgraph

Kernel = _rxgraph.Kernel
SearchStats = _rxgraph.SearchStats
col = pl.col
lit = pl.lit
rayon_thread_count = _rxgraph.rayon_thread_count


class Graph:
    """Arrow-backed directed graph."""

    def __init__(
        self,
        nodes: Any,
        edges: Any,
        *,
        _label_to_id: dict[Hashable, int] | None = None,
        _id_to_label: list[Hashable] | None = None,
    ) -> None:
        self._inner = _rxgraph.Graph(_table(nodes), _table(edges))
        self._label_to_id = _label_to_id
        self._id_to_label = _id_to_label

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[Hashable, Hashable] | tuple[Hashable, Hashable, Mapping[str, Any]]],
        *,
        nodes: Iterable[Hashable | tuple[Hashable, Mapping[str, Any]]] | None = None,
    ) -> Self:
        label_to_id: dict[Hashable, int] = {}
        id_to_label: list[Hashable] = []
        node_attrs: list[dict[str, Any]] = []

        def add_node(label: Hashable, attrs: Mapping[str, Any] | None = None) -> int:
            if label in label_to_id:
                if attrs:
                    node_attrs[label_to_id[label]].update(_attrs(attrs, "node"))
                return label_to_id[label]

            node_id = len(id_to_label)
            label_to_id[label] = node_id
            id_to_label.append(label)
            node_attrs.append(_attrs(attrs or {}, "node"))
            return node_id

        if nodes is not None:
            for node in nodes:
                label, attrs = _parse_node(node)
                add_node(label, attrs)

        edge_srcs: list[int] = []
        edge_dests: list[int] = []
        edge_attrs: list[dict[str, Any]] = []
        for edge in edges:
            src, dest, attrs = _parse_edge(edge)
            edge_srcs.append(add_node(src))
            edge_dests.append(add_node(dest))
            edge_attrs.append(_attrs(attrs or {}, "edge"))

        node_data = _rows_to_columns(node_attrs)
        node_data["id"] = list(range(len(id_to_label)))
        edge_data = _rows_to_columns(edge_attrs)
        edge_data["id"] = list(range(len(edge_srcs)))
        edge_data["src"] = edge_srcs
        edge_data["dest"] = edge_dests

        node_table = pl.DataFrame(node_data, schema_overrides={"id": pl.UInt64})
        edge_table = pl.DataFrame(
            edge_data,
            schema_overrides={"id": pl.UInt64, "src": pl.UInt64, "dest": pl.UInt64},
        )
        return cls(
            node_table,
            edge_table,
            _label_to_id=label_to_id,
            _id_to_label=id_to_label,
        )

    @property
    def node_count(self) -> int:
        return self._inner.node_count

    @property
    def edge_count(self) -> int:
        return self._inner.edge_count

    def node_id(self, label: Hashable) -> int:
        """Return the graph ID used by the engine for a label."""
        if self._label_to_id is None:
            if not isinstance(label, int | str):
                raise ValueError("table-backed graph ids must be integers or strings")
            return label
        try:
            return self._label_to_id[label]
        except KeyError as exc:
            raise ValueError(f"node label {label!r} is not present in the graph") from exc

    def search(self, traversal: "Traversal") -> "SearchResult":
        inner = self._inner.search(traversal._to_inner(self))
        return SearchResult(inner, self._id_to_label)

    def bfs(self, start: Hashable, max_depth: int | None = None) -> list[Any]:
        return self._map_nodes(self._inner.bfs(self.node_id(start), max_depth))

    def dfs(self, start: Hashable, max_depth: int | None = None) -> list[Any]:
        return self._map_nodes(self._inner.dfs(self.node_id(start), max_depth))

    def reachable_nodes(self, start: Hashable) -> list[Any]:
        return self._map_nodes(self._inner.reachable_nodes(self.node_id(start)))

    def shortest_path(self, source: Hashable, target: Hashable) -> list[Any] | None:
        path = self._inner.shortest_path(self.node_id(source), self.node_id(target))
        if path is None:
            return None
        return self._map_nodes(path)

    def out_degrees(self) -> list[int]:
        return self._inner.out_degrees()

    def in_degrees(self) -> list[int]:
        return self._inner.in_degrees()

    def degrees(self) -> list[int]:
        return self._inner.degrees()

    def weakly_connected_components(self) -> list[list[Any]]:
        return [self._map_nodes(component) for component in self._inner.weakly_connected_components()]

    def _map_nodes(self, nodes: list[int]) -> list[Any]:
        if self._id_to_label is None:
            return nodes
        return [self._id_to_label[node] for node in nodes]


class Traversal:
    """Traversal configuration used by :meth:`Graph.search`."""

    def __init__(
        self,
        kernel: Kernel,
        start_nodes: list[Hashable],
        max_depth: int,
        max_paths: int,
        strategy: str = "dfs",
        parallel: bool | str = True,
        intermediate_states: bool = False,
    ) -> None:
        self.kernel = kernel
        self.start_nodes = list(start_nodes)
        self.max_depth = max_depth
        self.max_paths = max_paths
        self.strategy = strategy
        self.parallel = _parallel_bool(parallel)
        self.intermediate_states = intermediate_states

    def _to_inner(self, graph: Graph) -> _rxgraph.Traversal:
        return _rxgraph.Traversal(
            self.kernel,
            [graph.node_id(node) for node in self.start_nodes],
            self.max_depth,
            self.max_paths,
            self.strategy,
            self.parallel,
            self.intermediate_states,
        )


class SearchPath:
    """One stopped path returned by a traversal."""

    def __init__(self, inner: _rxgraph.SearchPath, id_to_label: list[Hashable] | None) -> None:
        self._inner = inner
        self._id_to_label = id_to_label

    @property
    def nodes(self) -> list[Any]:
        if self._id_to_label is None:
            return self._inner.nodes
        return [self._id_to_label[node] for node in self._inner.nodes]

    @property
    def edges(self) -> list[int]:
        return self._inner.edges

    @property
    def state(self) -> dict[str, Any]:
        return self._inner.state

    @property
    def intermediate_states(self) -> list[dict[str, Any]] | None:
        return self._inner.intermediate_states


class SearchResult:
    """Paths and stats returned by :meth:`Graph.search`."""

    def __init__(
        self,
        inner: _rxgraph.SearchResult,
        id_to_label: list[Hashable] | None,
    ) -> None:
        self._inner = inner
        self.paths = [SearchPath(path, id_to_label) for path in inner.paths]
        self.stats = inner.stats


def _parse_node(
    node: Hashable | tuple[Hashable, Mapping[str, Any]],
) -> tuple[Hashable, Mapping[str, Any] | None]:
    if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], Mapping):
        return node[0], node[1]
    return node, None


def _parse_edge(
    edge: tuple[Hashable, Hashable] | tuple[Hashable, Hashable, Mapping[str, Any]],
) -> tuple[Hashable, Hashable, Mapping[str, Any] | None]:
    if len(edge) == 2:
        return edge[0], edge[1], None
    if len(edge) == 3 and isinstance(edge[2], Mapping):
        return edge[0], edge[1], edge[2]
    raise ValueError("edges must be (src, dest) or (src, dest, attrs) tuples")


def _attrs(attrs: Mapping[str, Any], kind: str) -> dict[str, Any]:
    reserved = {"id"} if kind == "node" else {"id", "src", "dest"}
    overlap = reserved.intersection(attrs)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"{kind} attributes cannot use reserved keys: {names}")
    return dict(attrs)


def _rows_to_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    keys = sorted({key for row in rows for key in row if any(r.get(key) is not None for r in rows)})
    return {key: [row.get(key) for row in rows] for key in keys}


def _table(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("rxgraph expects one node DataFrame and one edge DataFrame")
        return value[0][1]
    return value


def _parallel_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    if value in {"on", "auto"}:
        return True
    if value == "off":
        return False
    raise ValueError("parallel must be a bool, or one of 'on', 'off', 'auto'")


__all__ = [
    "Graph",
    "Kernel",
    "rayon_thread_count",
    "SearchPath",
    "SearchResult",
    "SearchStats",
    "Traversal",
    "col",
    "lit",
]
