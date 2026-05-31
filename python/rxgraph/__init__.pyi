from collections.abc import Hashable, Iterable, Mapping
from typing import Generic, Literal, TypeAlias, TypeVar

from polars import DataFrame, Expr, col as col, lit as lit

Node = TypeVar("Node", bound=Hashable)

GraphId: TypeAlias = int | str
TableInput: TypeAlias = DataFrame | list[tuple[str, DataFrame]]
_AttributeMap: TypeAlias = Mapping[str, object]
NodeInput: TypeAlias = Node | tuple[Node, _AttributeMap]
EdgeInput: TypeAlias = tuple[Node, Node] | tuple[Node, Node, _AttributeMap]
ExpressionInput: TypeAlias = Expr | str
StateValue: TypeAlias = None | bool | int | float | str | list["StateValue"]
StateMap: TypeAlias = Mapping[str, StateValue]
SearchStrategy: TypeAlias = Literal["dfs", "bfs"]
ParallelMode: TypeAlias = bool | Literal["auto", "off", "on"]

def rayon_thread_count() -> int:
    """Return the Rayon worker thread count used by rxgraph."""
    ...

class Graph(Generic[Node]):
    """Arrow-backed directed graph."""

    def __init__(
        self,
        nodes: TableInput,
        edges: TableInput,
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
        edges: Iterable[EdgeInput[Node]],
        *,
        nodes: Iterable[NodeInput[Node]] | None = None,
    ) -> Graph[Node]:
        """Build a directed graph from Python node labels and edge tuples."""
        ...

    @property
    def node_count(self) -> int: ...
    @property
    def edge_count(self) -> int: ...
    def node_id(self, label: Node) -> GraphId:
        """Return the graph ID used by the engine for a node label."""
        ...
    def search(
        self,
        *,
        start_nodes: Iterable[Node],
        visit: ExpressionInput | None = None,
        next_state: Mapping[str, ExpressionInput] | None = None,
        stop: ExpressionInput | None = None,
        initial_state: StateMap | None = None,
        max_depth: int | None = None,
        max_paths: int | None = None,
        strategy: SearchStrategy = "dfs",
        parallel: ParallelMode = True,
        intermediate_states: bool = False,
    ) -> SearchResult[Node]: ...
    def bfs(self, start: Node, max_depth: int | None = None) -> list[Node]:
        """Return nodes reachable from ``start`` in breadth-first order."""
        ...
    def dfs(self, start: Node, max_depth: int | None = None) -> list[Node]:
        """Return nodes reachable from ``start`` in depth-first pre-order."""
        ...
    def reachable_nodes(self, start: Node) -> list[Node]:
        """Return all nodes reachable from ``start``."""
        ...
    def shortest_path(self, source: Node, target: Node) -> list[Node] | None:
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
    def weakly_connected_components(self) -> list[list[Node]]:
        """Return weakly connected components, ignoring edge direction."""
        ...

class DiGraph(Graph[Node]):
    """Arrow-backed bidirectional graph."""

    def __init__(
        self,
        nodes: TableInput,
        edges: TableInput,
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
        edges: Iterable[EdgeInput[Node]],
        *,
        nodes: Iterable[NodeInput[Node]] | None = None,
    ) -> DiGraph[Node]:
        """Build a bidirectional graph from Python node labels and edge tuples."""
        ...

class Kernel:
    """Traversal kernel built from Polars expressions."""

    visit: ExpressionInput
    next_state: dict[str, ExpressionInput]
    stop: ExpressionInput
    initial_state: dict[str, StateValue]

    def __init__(
        self,
        visit: ExpressionInput | None = None,
        next_state: Mapping[str, ExpressionInput] | None = None,
        stop: ExpressionInput | None = None,
        initial_state: StateMap | None = None,
    ) -> None:
        """Create a kernel.

        ``visit`` decides whether an edge is accepted, ``next_state`` computes
        state updates for accepted edges, and ``stop`` decides whether an
        accepted edge materializes a path. Defaults accept every edge, leave
        state unchanged, never stop, and start with empty state.
        """
        ...

class Traversal(Generic[Node]):
    """Low-level traversal configuration for the native engine."""

    def __init__(
        self,
        kernel: Kernel,
        start_nodes: list[Node],
        max_depth: int | None,
        max_paths: int | None,
        strategy: SearchStrategy = "dfs",
        parallel: ParallelMode = True,
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

class SearchPath(Generic[Node]):
    """One stopped path returned by a traversal."""

    nodes: list[Node]
    edges: list[GraphId]
    state: dict[str, StateValue]
    intermediate_states: list[dict[str, StateValue]] | None

class SearchResult(Generic[Node]):
    """Paths and stats returned by :meth:`Graph.search`."""

    paths: list[SearchPath[Node]]
    stats: SearchStats
