from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Self

import polars as pl

from . import _rxgraph
from ._graph_tables import (
    EdgeInput,
    GraphTables,
    NodeInput,
    build_bidirectional_edges,
    build_labeled_tables,
    normalize_table,
)

SearchStats = _rxgraph.SearchStats
col = pl.col
lit = pl.lit
rayon_thread_count = _rxgraph.rayon_thread_count
DEFAULT_KERNEL_VISIT = pl.lit(True)
DEFAULT_KERNEL_STOP = pl.lit(False)
DEFAULT_SEARCH_STOP = pl.lit(True)


class Kernel:
    """Traversal kernel built from Polars expressions."""

    __slots__ = ("visit", "next_state", "stop", "initial_state")

    def __init__(
        self,
        visit: Any | None = None,
        next_state: Mapping[str, Any] | None = None,
        stop: Any | None = None,
        initial_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a traversal kernel.

        ``visit`` defaults to accepting every edge. ``stop`` defaults to never
        emitting paths; :meth:`Graph.search` overrides this when ``max_paths`` is
        supplied without an explicit ``stop``.
        """
        self.visit = DEFAULT_KERNEL_VISIT if visit is None else visit
        self.next_state = dict(next_state or {})
        self.stop = DEFAULT_KERNEL_STOP if stop is None else stop
        self.initial_state = dict(initial_state or {})

    def _to_inner(self) -> _rxgraph.Kernel:
        return _rxgraph.Kernel(
            self.visit,
            self.next_state,
            self.stop,
            self.initial_state,
        )


class Graph:
    """Arrow-backed directed graph."""

    def __init__(
        self,
        nodes: Any,
        edges: Any,
        *,
        _label_to_id: dict[Hashable, int] | None = None,
        _id_to_label: list[Hashable] | None = None,
        _edge_id_to_label: dict[Hashable, Hashable] | None = None,
    ) -> None:
        self._inner = _rxgraph.Graph(normalize_table(nodes), normalize_table(edges))
        self._label_to_id = _label_to_id
        self._id_to_label = _id_to_label
        self._edge_id_to_label = _edge_id_to_label

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[EdgeInput],
        *,
        nodes: Iterable[NodeInput] | None = None,
    ) -> Self:
        tables = build_labeled_tables(edges, nodes)
        return cls._from_tables(tables)

    @classmethod
    def _from_tables(cls, tables: GraphTables) -> Self:
        graph = cls.__new__(cls)
        Graph.__init__(
            graph,
            tables.nodes,
            tables.edges,
            _label_to_id=tables.label_to_id,
            _id_to_label=tables.id_to_label,
            _edge_id_to_label=tables.edge_id_to_label,
        )
        return graph

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
            raise ValueError(
                f"node label {label!r} is not present in the graph"
            ) from exc

    def search(
        self,
        *,
        start_nodes: Iterable[Hashable],
        visit: Any | None = None,
        next_state: Mapping[str, Any] | None = None,
        stop: Any | None = None,
        initial_state: Mapping[str, Any] | None = None,
        max_depth: int | None = None,
        max_paths: int | None = None,
        strategy: str = "dfs",
        parallel: bool | str = True,
        intermediate_states: bool = False,
    ) -> "SearchResult":
        """Run a stateful traversal.

        ``visit`` defaults to accepting all candidate edges. ``stop`` decides
        which accepted paths are returned; if omitted, ``max_paths`` is required
        and every accepted edge is returned. ``next_state`` maps state names to
        Polars expressions evaluated after each accepted edge. ``initial_state``
        may contain scalars, Python lists, or dict-like struct values. Search
        kernels support native scalar, list, and struct Polars expressions.
        ``strategy`` is ``"dfs"`` or ``"bfs"``.

        >>> import rxgraph as rxg
        >>> graph = rxg.Graph.from_edges([("a", "b"), ("b", "c")])
        >>> result = graph.search(start_nodes=["a"], max_paths=1)
        >>> result.paths[0].nodes
        ['a', 'b']
        """
        stop = _default_stop(stop, max_paths)
        traversal = Traversal(
            Kernel(
                visit,
                next_state,
                stop,
                initial_state,
            ),
            list(start_nodes),
            max_depth,
            max_paths,
            strategy,
            parallel,
            intermediate_states,
        )

        inner = self._inner.search(traversal._to_inner(self))
        return SearchResult._from_inner(
            inner, self._id_to_label, self._edge_id_to_label
        )

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
        return [
            self._map_nodes(component)
            for component in self._inner.weakly_connected_components()
        ]

    def _map_nodes(self, nodes: list[int]) -> list[Any]:
        if self._id_to_label is None:
            return nodes
        return [self._id_to_label[node] for node in nodes]


class DiGraph(Graph):
    """Arrow-backed bidirectional graph."""

    def __init__(
        self,
        nodes: Any,
        edges: Any,
    ) -> None:
        edge_table, edge_id_to_label = build_bidirectional_edges(edges)
        super().__init__(
            nodes,
            edge_table,
            _edge_id_to_label=edge_id_to_label,
        )

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[EdgeInput],
        *,
        nodes: Iterable[NodeInput] | None = None,
    ) -> Self:
        tables = build_labeled_tables(
            edges,
            nodes,
            bidirectional=True,
        )
        return cls._from_tables(tables)


class Traversal:
    """Low-level traversal configuration for the native engine."""

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
        """Create a reusable traversal configuration.

        ``kernel`` contains visit/state/stop expressions. ``start_nodes`` are
        external graph node IDs or labels. ``max_depth`` limits accepted-edge
        depth, and ``max_paths`` limits returned paths. ``strategy`` is
        ``"dfs"`` or ``"bfs"``. ``parallel`` accepts ``True``/``False`` or
        ``"on"``/``"off"``/``"auto"``. ``intermediate_states`` stores per-node
        state history on returned paths and is disabled by default.
        """
        self.kernel = kernel
        self.start_nodes = list(start_nodes)
        self.max_depth = max_depth
        self.max_paths = max_paths
        self.strategy = strategy
        self.parallel = _parallel_bool(parallel)
        self.intermediate_states = intermediate_states

    def _to_inner(self, graph: Graph) -> _rxgraph.Traversal:
        return _rxgraph.Traversal(
            _inner_kernel(self.kernel),
            [graph.node_id(node) for node in self.start_nodes],
            self.max_depth,
            self.max_paths,
            self.strategy,
            self.parallel,
            self.intermediate_states,
        )


@dataclass(slots=True)
class SearchPath:
    """One stopped path returned by a traversal."""

    nodes: list[Any]
    edges: list[Any]
    state: dict[str, Any]
    intermediate_states: list[dict[str, Any]] | None = None

    @classmethod
    def _from_inner(
        cls,
        inner: _rxgraph.SearchPath,
        id_to_label: list[Hashable] | None,
        edge_id_to_label: dict[Hashable, Hashable] | None,
    ) -> Self:
        return cls(
            nodes=_map_search_nodes(inner.nodes, id_to_label),
            edges=_map_search_edges(inner.edges, edge_id_to_label),
            state=dict(inner.state),
            intermediate_states=(
                None
                if inner.intermediate_states is None
                else [dict(state) for state in inner.intermediate_states]
            ),
        )


@dataclass(slots=True)
class SearchResult:
    """Paths and stats returned by :meth:`Graph.search`."""

    paths: list[SearchPath]
    stats: SearchStats

    @classmethod
    def _from_inner(
        cls,
        inner: _rxgraph.SearchResult,
        id_to_label: list[Hashable] | None,
        edge_id_to_label: dict[Hashable, Hashable] | None,
    ) -> Self:
        return cls(
            paths=[
                SearchPath._from_inner(path, id_to_label, edge_id_to_label)
                for path in inner.paths
            ],
            stats=inner.stats,
        )


def _map_search_nodes(
    nodes: list[Any], id_to_label: list[Hashable] | None
) -> list[Any]:
    if id_to_label is None:
        return list(nodes)
    return [id_to_label[node] for node in nodes]


def _map_search_edges(
    edges: list[Any],
    edge_id_to_label: dict[Hashable, Hashable] | None,
) -> list[Any]:
    if edge_id_to_label is None:
        return list(edges)
    return [edge_id_to_label[edge] for edge in edges]


def _parallel_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    if value in {"on", "auto"}:
        return True
    if value == "off":
        return False
    raise ValueError("parallel must be a bool, or one of 'on', 'off', 'auto'")


def _default_stop(stop: Any | None, max_paths: int | None) -> Any:
    if stop is not None:
        return stop
    if max_paths is None:
        raise TypeError("search() requires 'stop' or 'max_paths'")
    return DEFAULT_SEARCH_STOP


def _inner_kernel(kernel: Kernel | _rxgraph.Kernel) -> _rxgraph.Kernel:
    if isinstance(kernel, Kernel):
        return kernel._to_inner()
    return kernel


__all__ = [
    "DiGraph",
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
