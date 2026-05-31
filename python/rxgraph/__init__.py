from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Generic, Literal, Self, TypeAlias, TypeVar, cast

import polars as pl

from . import _rxgraph
from ._graph_tables import (
    EdgeInput,
    GraphId,
    GraphTables,
    NodeInput,
    TableInput,
    build_bidirectional_edges,
    build_labeled_tables,
    normalize_table,
)

NodeT = TypeVar("NodeT", bound=Hashable)
ExpressionInput: TypeAlias = pl.Expr | str
StateValue: TypeAlias = None | bool | int | float | str | list["StateValue"]
StateMap: TypeAlias = Mapping[str, StateValue]
StateDict: TypeAlias = dict[str, StateValue]
SearchStrategy: TypeAlias = Literal["dfs", "bfs"]
ParallelMode: TypeAlias = bool | Literal["auto", "off", "on"]

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
        visit: ExpressionInput | None = None,
        next_state: Mapping[str, ExpressionInput] | None = None,
        stop: ExpressionInput | None = None,
        initial_state: StateMap | None = None,
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


class Graph(Generic[NodeT]):
    """Arrow-backed directed graph."""

    def __init__(
        self,
        nodes: TableInput,
        edges: TableInput,
        *,
        _label_to_id: dict[NodeT, int] | None = None,
        _id_to_label: list[NodeT] | None = None,
        _edge_id_to_label: dict[GraphId, GraphId] | None = None,
    ) -> None:
        self._inner = _rxgraph.Graph(normalize_table(nodes), normalize_table(edges))
        self._label_to_id = _label_to_id
        self._id_to_label = _id_to_label
        self._edge_id_to_label = _edge_id_to_label

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[EdgeInput[NodeT]],
        *,
        nodes: Iterable[NodeInput[NodeT]] | None = None,
    ) -> Self:
        tables = build_labeled_tables(edges, nodes)
        return cls._from_tables(tables)

    @classmethod
    def _from_tables(cls, tables: GraphTables[NodeT]) -> Self:
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

    def node_id(self, label: NodeT) -> GraphId:
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
        start_nodes: Iterable[NodeT],
        visit: ExpressionInput | None = None,
        next_state: Mapping[str, ExpressionInput] | None = None,
        stop: ExpressionInput | None = None,
        initial_state: StateMap | None = None,
        max_depth: int | None = None,
        max_paths: int | None = None,
        strategy: SearchStrategy = "dfs",
        parallel: ParallelMode = True,
        intermediate_states: bool = False,
    ) -> SearchResult[NodeT]:
        """Run a stateful traversal.

        ``visit`` defaults to accepting all candidate edges. ``stop`` decides
        which accepted paths are returned; if omitted, ``max_paths`` is required
        and every accepted edge is returned. ``next_state`` maps state names to
        Polars expressions evaluated after each accepted edge. ``initial_state``
        may contain scalars or Python lists; list state can be updated with
        Polars list expressions such as ``pl.concat_list``. ``strategy`` is
        ``"dfs"`` or ``"bfs"``.

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
        return SearchResult(inner, self._id_to_label, self._edge_id_to_label)

    def bfs(self, start: NodeT, max_depth: int | None = None) -> list[NodeT]:
        return self._map_nodes(self._inner.bfs(self.node_id(start), max_depth))

    def dfs(self, start: NodeT, max_depth: int | None = None) -> list[NodeT]:
        return self._map_nodes(self._inner.dfs(self.node_id(start), max_depth))

    def reachable_nodes(self, start: NodeT) -> list[NodeT]:
        return self._map_nodes(self._inner.reachable_nodes(self.node_id(start)))

    def shortest_path(self, source: NodeT, target: NodeT) -> list[NodeT] | None:
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

    def weakly_connected_components(self) -> list[list[NodeT]]:
        return [
            self._map_nodes(component)
            for component in self._inner.weakly_connected_components()
        ]

    def _map_nodes(self, nodes: list[GraphId]) -> list[NodeT]:
        if self._id_to_label is None:
            return cast(list[NodeT], nodes)
        return [self._id_to_label[cast(int, node)] for node in nodes]


class DiGraph(Graph[NodeT]):
    """Arrow-backed bidirectional graph."""

    def __init__(
        self,
        nodes: TableInput,
        edges: TableInput,
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
        edges: Iterable[EdgeInput[NodeT]],
        *,
        nodes: Iterable[NodeInput[NodeT]] | None = None,
    ) -> Self:
        tables = build_labeled_tables(
            edges,
            nodes,
            bidirectional=True,
        )
        return cls._from_tables(tables)


class Traversal(Generic[NodeT]):
    """Low-level traversal configuration for the native engine."""

    def __init__(
        self,
        kernel: Kernel,
        start_nodes: list[NodeT],
        max_depth: int | None,
        max_paths: int | None,
        strategy: SearchStrategy = "dfs",
        parallel: ParallelMode = True,
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

    def _to_inner(self, graph: Graph[NodeT]) -> _rxgraph.Traversal:
        return _rxgraph.Traversal(
            _inner_kernel(self.kernel),
            [graph.node_id(node) for node in self.start_nodes],
            self.max_depth,
            self.max_paths,
            self.strategy,
            self.parallel,
            self.intermediate_states,
        )


class SearchPath(Generic[NodeT]):
    """One stopped path returned by a traversal."""

    nodes: list[NodeT]
    edges: list[GraphId]
    state: StateDict
    intermediate_states: list[StateDict] | None

    __slots__ = ("nodes", "edges", "state", "intermediate_states")

    def __init__(
        self,
        inner: _rxgraph.SearchPath,
        id_to_label: list[NodeT] | None,
        edge_id_to_label: dict[GraphId, GraphId] | None,
    ) -> None:
        self.nodes = _map_search_nodes(inner.nodes, id_to_label)
        self.edges = _map_search_edges(inner.edges, edge_id_to_label)
        self.state = cast(StateDict, dict(inner.state))
        self.intermediate_states = (
            None
            if inner.intermediate_states is None
            else [cast(StateDict, dict(state)) for state in inner.intermediate_states]
        )


class SearchResult(Generic[NodeT]):
    """Paths and stats returned by :meth:`Graph.search`."""

    paths: list[SearchPath[NodeT]]
    stats: SearchStats

    __slots__ = ("paths", "stats")

    def __init__(
        self,
        inner: _rxgraph.SearchResult,
        id_to_label: list[NodeT] | None,
        edge_id_to_label: dict[GraphId, GraphId] | None,
    ) -> None:
        self.paths = [
            SearchPath(path, id_to_label, edge_id_to_label) for path in inner.paths
        ]
        self.stats = inner.stats


def _map_search_nodes(
    nodes: list[GraphId], id_to_label: list[NodeT] | None
) -> list[NodeT]:
    if id_to_label is None:
        return cast(list[NodeT], list(nodes))
    return [id_to_label[cast(int, node)] for node in nodes]


def _map_search_edges(
    edges: list[GraphId],
    edge_id_to_label: dict[GraphId, GraphId] | None,
) -> list[GraphId]:
    if edge_id_to_label is None:
        return list(edges)
    return [edge_id_to_label[edge] for edge in edges]


def _parallel_bool(value: ParallelMode) -> bool:
    if isinstance(value, bool):
        return value
    if value in {"on", "auto"}:
        return True
    if value == "off":
        return False
    raise ValueError("parallel must be a bool, or one of 'on', 'off', 'auto'")


def _default_stop(
    stop: ExpressionInput | None, max_paths: int | None
) -> ExpressionInput:
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
