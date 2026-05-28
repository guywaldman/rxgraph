from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Literal, Self, TypeVar, overload

from polars import DataFrame, Expr, col as col, lit as lit

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

    @property
    def node_count(self) -> int: ...
    @property
    def edge_count(self) -> int: ...
    def node_id(self, label: Hashable) -> GraphId:
        """Return the graph ID used by the engine for a node label."""
        ...
    @overload
    def search(self, traversal: Traversal) -> SearchResult: ...
    @overload
    def search(
        self,
        *,
        start_nodes: Iterable[Hashable],
        stop: Expr | str,
        visit: Expr | str | None = None,
        next_state: Mapping[str, Expr | str] | None = None,
        initial_state: Mapping[str, Any] | None = None,
        max_depth: int | None = None,
        max_paths: int | None = None,
        strategy: Literal["dfs", "bfs"] = "dfs",
        parallel: bool | Literal["auto", "off", "on"] = True,
        intermediate_states: bool = False,
    ) -> SearchResult: ...
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

class Kernel:
    """Traversal kernel built from Polars expressions."""

    def __init__(
        self,
        visit: Expr | str,
        next_state: dict[str, Expr | str],
        stop: Expr | str,
        initial_state: Mapping[str, Any],
    ) -> None:
        """Create a kernel.

        ``visit`` decides whether an edge is accepted, ``next_state`` computes
        state updates for accepted edges, and ``stop`` decides whether an
        accepted edge materializes a path.
        """
        ...

class Traversal:
    """Traversal configuration used by :meth:`Graph.search`."""

    def __init__(
        self,
        kernel: Kernel,
        start_nodes: list[Hashable],
        max_depth: int,
        max_paths: int,
        strategy: Literal["dfs", "bfs"] = "dfs",
        parallel: bool | Literal["auto", "off", "on"] = True,
        intermediate_states: bool = False,
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

class SearchPath:
    """One stopped path returned by a traversal."""

    @property
    def nodes(self) -> list[Any]: ...
    @property
    def edges(self) -> list[Any]: ...
    @property
    def state(self) -> dict[str, Any]: ...
    @property
    def intermediate_states(self) -> list[dict[str, Any]] | None: ...

class SearchResult:
    """Paths and stats returned by :meth:`Graph.search`."""

    @property
    def paths(self) -> list[SearchPath]: ...
    @property
    def stats(self) -> SearchStats: ...
