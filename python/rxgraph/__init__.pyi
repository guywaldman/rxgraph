from typing import Any, Literal

class Graph:
    """Arrow-backed directed graph."""

    def __init__(
        self,
        node_tables: list[tuple[str, Any]],
        edge_tables: list[tuple[str, Any]],
    ) -> None:
        """Build a graph from node and edge tables.

        Node tables require an ``id`` UInt64 column. Edge tables require
        ``src`` and ``dest`` UInt64 columns.
        """
        ...

    @property
    def node_count(self) -> int: ...
    @property
    def edge_count(self) -> int: ...
    def search(self, traversal: Traversal) -> SearchResult:
        """Run a traversal and return stopped paths plus traversal stats."""
        ...

class Kernel:
    """Traversal kernel built from Polars expressions."""

    def __init__(
        self,
        visit: Any,
        next_state: dict[str, Any],
        stop: Any,
        initial_state: dict[str, bool | int | float | str | None],
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
        start_nodes: list[int],
        max_depth: int,
        max_paths: int,
        strategy: Literal["dfs", "bfs"] = "dfs",
        parallel: Literal["auto", "off", "on"] = "auto",
        parallel_min_frontier: int = 512,
        parallel_min_edges: int = 8192,
    ) -> None:
        """Create a traversal.

        ``strategy`` is ``"dfs"`` or ``"bfs"``. ``parallel`` affects only BFS;
        DFS remains serial because its early-stop order is usually the point.
        """
        ...

class SearchStats:
    """Counters collected while searching."""

    @property
    def visited_path_entries(self) -> int: ...
    @property
    def evaluated_edges(self) -> int: ...
    @property
    def accepted_edges(self) -> int: ...
    @property
    def stopped_paths(self) -> int: ...
    @property
    def skipped_errors(self) -> int: ...
    @property
    def max_depth(self) -> int: ...

class SearchPath:
    """One stopped path returned by a traversal."""

    @property
    def nodes(self) -> list[int]: ...
    @property
    def edges(self) -> list[int]: ...
    @property
    def state(self) -> str: ...

class SearchResult:
    """Paths and stats returned by :meth:`Graph.search`."""

    @property
    def paths(self) -> list[SearchPath]: ...
    @property
    def stats(self) -> SearchStats: ...
