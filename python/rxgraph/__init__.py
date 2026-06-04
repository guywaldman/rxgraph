from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Self

import polars as pl

from . import _rxgraph
from ._graph_tables import (
    DEST_COL,
    ID_COL,
    SRC_COL,
    EdgeInput,
    GraphTables,
    NodeInput,
    build_bidirectional_edges,
    build_labeled_tables,
    normalize_table,
)

TYPE_COL = "type"

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
        """
        Create a traversal kernel (defines the logic for visiting and aggregating when traversing the graph).

        :param visit: Predicate per candidate edge. Defaults to true (accepting every edge).
        :param next_state: Returns the next state, with each field in the dict being a Polars expression. Evaluated after each accepted edge.
        :param stop: Polars expression which is evaluated per node, and if it returns true, means to stop traversing and "bail out" on that node.
        :param initial_state: Dict with each field value (scalar/list/struct) being a Polars expression, representing the starting state values
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
        # Lazy payload sources. Set only by ``from_lazy``. ``None`` for eager graphs.
        self._lazy_nodes: pl.LazyFrame | None = None
        self._lazy_edges: pl.LazyFrame | None = None
        # Payload columns currently installed in the native graph (avoids re-projecting
        # the same columns across repeated searches).
        self._loaded_node_cols: frozenset[str] | None = None
        self._loaded_edge_cols: frozenset[str] | None = None

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
    def from_lazy(
        cls,
        nodes: "pl.LazyFrame",
        edges: "pl.LazyFrame",
    ) -> Self:
        """
        Build a graph from Polars ``LazyFrame``s, deferring payload columns.

        Similar to ``Graph(nodes=..., edges=...)`` but for LazyFrames.

        Use this with lazy I/O sources (e.g. ``pl.scan_parquet(...)``) to reduce I/O strain
        and also to cap memory.
        Ideal for scenarios of traversing over a large and complex graph, where the working set/frontier may be
        a lot smaller than the actual node count.

        IMPORTANT: The LazyFrames *must* keep a stable row order across collects (one row per
        node/edge), since payload reads index by row position.
        Sort the nodes by the ``id`` column, and edges at least by ``src`` and ``dest``.

        :param nodes: Lazy DataFrame for the nodes, with at least an ``id`` column.
        :param edges: Lazy DataFrame for the edges, with at least ``id``, ``src``, and ``dest`` columns.
        """
        node_schema = nodes.collect_schema()
        edge_schema = edges.collect_schema()

        node_topo = [ID_COL] + ([TYPE_COL] if TYPE_COL in node_schema else [])
        edge_topo = [ID_COL, SRC_COL, DEST_COL] + (
            [TYPE_COL] if TYPE_COL in edge_schema else []
        )

        graph = cls.__new__(cls)
        Graph.__init__(
            graph,
            nodes.select(node_topo).collect(),
            edges.select(edge_topo).collect(),
        )
        graph._lazy_nodes = nodes
        graph._lazy_edges = edges
        graph._loaded_node_cols = frozenset()
        graph._loaded_edge_cols = frozenset()
        return graph

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
        """Return the engine graph ID for a node label.

        :param label: Node label, or the raw integer/string ID for table-backed graphs.
        """
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
        progress: bool = False,
    ) -> "SearchResult":
        """Run a stateful traversal.

        API is similar to ``Graph.search``.



        :param visit: Predicate per candidate edge. Defaults to true (accepting every edge).
        :param next_state: Returns the next state, with each field in the dict being a Polars expression. Evaluated after each accepted edge.
        :param stop: Polars expression which is evaluated per node, and if it returns true, means to stop traversing and "bail out" on that node.
        :param initial_state: Dict with each field value (scalar/list/struct) being a Polars expression, representing the starting state values
        :param max_depth: Maximum accepted-edge depth per path.
        :param max_paths: Stop the search once this many paths have been returned.
        :param strategy: ``"dfs"`` or ``"bfs"``.
        :param parallel: ``True``/``False`` or one of ``"on"``/``"off"``/``"auto"``.
        :param intermediate_states: Whether to include per-node state history on returned paths (false by default).
        :param progress: Whether to report running counters on stderr (live spinner on a terminal/TTY, plain log lines otherwise).

        >>> import rxgraph as rxg
        >>> graph = rxg.Graph.from_edges([("a", "b"), ("b", "c")])
        >>> result = graph.search(start_nodes=["a"], max_paths=1)
        >>> result.paths[0].nodes
        ['a', 'b']
        """
        stop = _default_stop(stop, max_paths)
        kernel = Kernel(
            visit,
            next_state,
            stop,
            initial_state,
        )
        self._ensure_payloads(kernel)
        traversal = Traversal(
            kernel,
            list(start_nodes),
            max_depth,
            max_paths,
            strategy,
            parallel,
            intermediate_states,
            progress,
        )

        inner = self._inner.search(traversal._to_inner(self))
        return SearchResult._from_inner(
            inner, self._id_to_label, self._edge_id_to_label
        )

    def _ensure_payloads(self, kernel: Kernel) -> None:
        """Project and "install" the payload columns a lazy graph's kernel references.

        For lazy graphs, collects only the ``src``/``dest``
        (node) and ``edge`` columns the kernel reads, and swaps them into the native
        graph. Skips re-projection when the required columns are already loaded.
        No-op for eager graphs.
        """
        if self._lazy_nodes is None or self._lazy_edges is None:
            return

        node_cols, edge_cols = _referenced_payload_columns(kernel)
        if node_cols == self._loaded_node_cols and edge_cols == self._loaded_edge_cols:
            return

        nodes = self._lazy_nodes.select(_payload_projection(node_cols)).collect()
        edges = self._lazy_edges.select(_payload_projection(edge_cols)).collect()
        self._inner.set_payloads(nodes, edges)
        self._loaded_node_cols = node_cols
        self._loaded_edge_cols = edge_cols

    def bfs(self, start: Hashable, max_depth: int | None = None) -> list[Any]:
        """Breadth-first node order from ``start``.

        :param start: Node to start from.
        :param max_depth: Optional depth limit.
        """
        return self._map_nodes(self._inner.bfs(self.node_id(start), max_depth))

    def dfs(self, start: Hashable, max_depth: int | None = None) -> list[Any]:
        """Depth-first node order from ``start``.

        :param start: Node to start from.
        :param max_depth: Optional depth limit.
        """
        return self._map_nodes(self._inner.dfs(self.node_id(start), max_depth))

    def reachable_nodes(self, start: Hashable) -> list[Any]:
        """All nodes reachable from ``start``.

        :param start: Node to start from.
        """
        return self._map_nodes(self._inner.reachable_nodes(self.node_id(start)))

    def shortest_path(self, source: Hashable, target: Hashable) -> list[Any] | None:
        """Shortest path between two nodes, or ``None`` if unreachable.

        :param source: Start node.
        :param target: End node.
        """
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
        progress: bool = False,
    ) -> None:
        """Create a reusable traversal configuration.

        :param kernel: Visit/state/stop expressions.
        :param start_nodes: Node IDs or labels where traversal begins.
        :param max_depth: Maximum accepted-edge depth per path.
        :param max_paths: Maximum number of returned paths.
        :param strategy: ``"dfs"`` or ``"bfs"``.
        :param parallel: ``True``/``False`` or one of ``"on"``/``"off"``/``"auto"``.
        :param intermediate_states: Include per-node state history on returned paths.
        :param progress: Report progress on stderr (spinner on a terminal, log lines
            otherwise).
        """
        self.kernel = kernel
        self.start_nodes = list(start_nodes)
        self.max_depth = max_depth
        self.max_paths = max_paths
        self.strategy = strategy
        self.parallel = _parallel_bool(parallel)
        self.intermediate_states = intermediate_states
        self.progress = progress

    def _to_inner(self, graph: Graph) -> _rxgraph.Traversal:
        return _rxgraph.Traversal(
            _inner_kernel(self.kernel),
            [graph.node_id(node) for node in self.start_nodes],
            self.max_depth,
            self.max_paths,
            self.strategy,
            self.parallel,
            self.intermediate_states,
            self.progress,
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


def _referenced_payload_columns(
    kernel: Kernel,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return the (node_columns, edge_columns) a kernel reads.

    DSL columns are scoped: ``src.<field>``/``dest.<field>`` resolve to node payload
    columns and ``edge.<field>`` to edge payload columns. ``state.*`` and ``*.id``
    columns need no payload projection and are ignored.
    """
    node_cols: set[str] = set()
    edge_cols: set[str] = set()

    exprs: list[Any] = [kernel.visit, kernel.stop]
    exprs.extend(kernel.next_state.values())
    exprs.extend(kernel.initial_state.values())

    for expr in exprs:
        if not isinstance(expr, pl.Expr):
            continue
        for name in expr.meta.root_names():
            scope, _, field = name.partition(".")
            if not field or field == ID_COL:
                continue
            if scope in ("src", "dest"):
                node_cols.add(field)
            elif scope == "edge" and field not in {SRC_COL, DEST_COL}:
                edge_cols.add(field)

    return frozenset(node_cols), frozenset(edge_cols)


def _payload_projection(cols: frozenset[str]) -> list[str]:
    return sorted(cols) or [ID_COL]


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
