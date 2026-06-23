from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any, Literal, Self, TypeVar

from polars import DataFrame, Expr, LazyFrame, col as col, lit as lit

Node = TypeVar("Node", bound=Hashable)
GraphId = int | str

def rayon_thread_count() -> int:
    """Return the Rayon worker thread count used by rxgraph."""
    ...

class Graph:
    """Arrow-backed directed graph."""

    def __init__(
        self,
        nodes: DataFrame | list[tuple[str, DataFrame]],
        edges: DataFrame | list[tuple[str, DataFrame]],
    ) -> None:
        """Build a graph from node and edge tables.

        Nodes require ``id`` and edges require ``id``, ``src``, and ``dest``.
        All ID columns must be either UInt64 or string. Optional ``type``
        columns must be string when present.
        """
        ...

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[Node, Node] | tuple[Node, Node, Mapping[str, Any]]],
        *,
        nodes: Iterable[Node | tuple[Node, Mapping[str, Any]]] | None = None,
    ) -> Self:
        """Build a directed graph from Python node labels and edge tuples."""
        ...

    @classmethod
    def from_lazy(cls, nodes: LazyFrame, edges: LazyFrame) -> Self:
        """Build a graph from Polars ``LazyFrame``s, deferring payload columns.

        Only identity/topology columns (``id`` for nodes; ``id``/``src``/``dest``
        for edges, plus optional ``type``) are collected at construction. Payload
        columns are pulled lazily at :meth:`search` time, projected to only the
        columns the kernel references, bounding resident payload memory.
        """
        ...

    @classmethod
    def from_parquet(
        cls,
        nodes: str | PathLike[str],
        edges: str | PathLike[str],
        *,
        payloads: Literal["eager", "lazy"] = "eager",
    ) -> Self:
        """Build a file-backed graph from Parquet node and edge tables."""
        ...

    @property
    def node_count(self) -> int: ...
    @property
    def edge_count(self) -> int: ...
    def node_id(self, label: Hashable) -> GraphId:
        """Return the graph ID used by the engine for a node label."""
        ...
    def search(
        self,
        *,
        start_nodes: Iterable[Hashable],
        visit: Expr | str | None = None,
        next_state: Mapping[str, Expr | str] | None = None,
        stop: Expr | str | None = None,
        initial_state: Mapping[str, Any] | None = None,
        kernel: str | None = None,
        params: Mapping[str, Any] | None = None,
        columns: Iterable[str] | None = None,
        max_depth: int | None = None,
        max_paths: int | None = None,
        strategy: Literal["dfs", "bfs"] = "dfs",
        parallel: bool | Literal["auto", "off", "on"] = True,
        intermediate_states: bool = False,
        progress: bool = False,
        max_revisits_per_node: int = 0,
    ) -> SearchResult:
        """Run a stateful traversal.

        Two mutually exclusive modes:

        * DSL: ``visit``/``next_state``/``stop``/``initial_state`` Polars expressions.
        * Named native kernel: ``kernel`` (registered Rust kernel name) and ``params``.
          The DSL args must not be set in this mode.

        ``params`` node-label values under the conventional keys
        ``target``/``source``/``start``/``node`` are translated to engine IDs when
        the value is a known node label; other values pass through unchanged.

        ``columns`` lists payload columns to load for ``from_lazy`` before a
        legacy named kernel runs. File-backed ``from_parquet`` graphs use typed
        native kernels and do not need ``columns``.
        """
        ...
    def bfs(self, start: Hashable, max_depth: int | None = None) -> list[Any]:
        """Return nodes reachable from ``start`` in breadth-first order."""
        ...
    def dfs(self, start: Hashable, max_depth: int | None = None) -> list[Any]:
        """Return nodes reachable from ``start`` in depth-first pre-order."""
        ...
    def reachable_nodes(self, start: Hashable) -> list[Any]:
        """Return all nodes reachable from ``start``."""
        ...
    def shortest_path(self, source: Hashable, target: Hashable) -> list[Any] | None:
        """Return an unweighted directed shortest path, if one exists."""
        ...
    def out_degrees(self) -> list[int]:
        """Return out-degree for each node in node insertion order."""
        ...
    def in_degrees(self) -> list[int]:
        """Return in-degree for each node in node insertion order."""
        ...
    def degrees(self) -> list[int]:
        """Return total directed degree for each node in node insertion order."""
        ...
    def weakly_connected_components(self) -> list[list[Any]]:
        """Return weakly connected components, ignoring edge direction."""
        ...

class DiGraph(Graph):
    """Arrow-backed bidirectional graph."""

    def __init__(
        self,
        nodes: DataFrame | list[tuple[str, DataFrame]],
        edges: DataFrame | list[tuple[str, DataFrame]],
    ) -> None:
        """Build a bidirectional graph from node and edge tables.

        Each input edge is available in both directions. Reverse rows receive
        synthetic engine IDs, but traversal path edges are mapped back to the
        original input edge IDs.
        """
        ...

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[Node, Node] | tuple[Node, Node, Mapping[str, Any]]],
        *,
        nodes: Iterable[Node | tuple[Node, Mapping[str, Any]]] | None = None,
    ) -> Self:
        """Build a bidirectional graph from Python node labels and edge tuples."""
        ...

class Kernel:
    """Traversal kernel built from Polars expressions."""

    visit: Expr | str
    next_state: dict[str, Expr | str]
    stop: Expr | str
    initial_state: dict[str, Any]

    def __init__(
        self,
        visit: Expr | str | None = None,
        next_state: Mapping[str, Expr | str] | None = None,
        stop: Expr | str | None = None,
        initial_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a kernel.

        ``visit`` decides whether an edge is accepted, ``next_state`` computes
        state updates for accepted edges, and ``stop`` decides whether an
        accepted edge materializes a path. Defaults accept every edge, leave
        state unchanged, never stop, and start with empty state.
        """
        ...

class Traversal:
    """Low-level traversal configuration for the native engine."""

    def __init__(
        self,
        kernel: Kernel,
        start_nodes: list[Hashable],
        max_depth: int,
        max_paths: int,
        strategy: Literal["dfs", "bfs"] = "dfs",
        parallel: bool | Literal["auto", "off", "on"] = True,
        intermediate_states: bool = False,
        progress: bool = False,
    ) -> None:
        """Create a traversal.

        ``strategy`` is ``"dfs"`` or ``"bfs"``. ``parallel`` enables or disables
        rxgraph parallel traversal. ``intermediate_states`` stores per-node
        state history on returned paths.
        """
        ...

class SearchStats:
    """Counters collected while searching."""

    @property
    def start_nodes(self) -> int: ...
    @property
    def path_entries(self) -> int: ...
    @property
    def evaluated_edges(self) -> int: ...
    @property
    def accepted_edges(self) -> int: ...
    @property
    def stopped_paths(self) -> int: ...
    @property
    def rejected_edges(self) -> int: ...
    @property
    def skipped_revisits(self) -> int: ...
    @property
    def max_depth(self) -> int: ...
    @property
    def materialized_node_payloads(self) -> int: ...
    @property
    def materialized_edge_payloads(self) -> int: ...

@dataclass(slots=True)
class SearchPath:
    """One stopped path returned by a traversal."""

    nodes: list[Any]
    edges: list[Any]
    state: dict[str, Any]
    intermediate_states: list[dict[str, Any]] | None = None

@dataclass(slots=True)
class SearchResult:
    """Paths and stats returned by :meth:`Graph.search`."""

    paths: list[SearchPath]
    stats: SearchStats
