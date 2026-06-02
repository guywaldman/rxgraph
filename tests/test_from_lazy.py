import polars as pl
import pytest
import rxgraph as rxg


def _tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    nodes = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "score": [10, 20, 30, 40],
            "name": ["a", "b", "c", "d"],
            "unused": ["x" * 32] * 4,
        },
        schema={
            "id": pl.UInt64,
            "score": pl.Int64,
            "name": pl.String,
            "unused": pl.String,
        },
    )
    edges = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "src": [0, 1, 2],
            "dest": [1, 2, 3],
            "weight": [1.0, 2.0, 3.0],
        },
        schema={
            "id": pl.UInt64,
            "src": pl.UInt64,
            "dest": pl.UInt64,
            "weight": pl.Float64,
        },
    )
    return nodes, edges


def _search_kwargs(strategy: str) -> dict:
    return dict(
        start_nodes=[0],
        visit=pl.col("edge.weight") < 10.0,
        next_state={"sum": pl.col("state.sum") + pl.col("dest.score")},
        initial_state={"sum": 0},
        stop=pl.col("dest.id") == 3,
        strategy=strategy,
        max_paths=10,
    )


def _paths(result) -> list[tuple[list, dict]]:
    return [(p.nodes, p.state) for p in result.paths]


@pytest.mark.parametrize("strategy", ["bfs", "dfs"])
def test_from_lazy_matches_eager(strategy: str) -> None:
    nodes, edges = _tables()
    eager = rxg.Graph(nodes, edges)
    lazy = rxg.Graph.from_lazy(nodes.lazy(), edges.lazy())

    assert lazy.node_count == eager.node_count
    assert lazy.edge_count == eager.edge_count

    kw = _search_kwargs(strategy)
    assert _paths(lazy.search(**kw)) == _paths(eager.search(**kw))


def test_from_lazy_projects_only_referenced_columns() -> None:
    nodes, edges = _tables()
    lazy = rxg.Graph.from_lazy(nodes.lazy(), edges.lazy())

    # No payload loaded until a search runs.
    assert lazy._loaded_node_cols == frozenset()
    assert lazy._loaded_edge_cols == frozenset()

    lazy.search(**_search_kwargs("bfs"))

    # Only the columns the kernel references are materialized.
    assert lazy._loaded_node_cols == frozenset({"score"})
    assert lazy._loaded_edge_cols == frozenset({"weight"})


def test_from_lazy_reprojects_when_columns_change() -> None:
    nodes, edges = _tables()
    lazy = rxg.Graph.from_lazy(nodes.lazy(), edges.lazy())

    lazy.search(
        start_nodes=[0],
        visit=pl.col("dest.score") > 0,
        stop=pl.col("dest.id") == 3,
        strategy="bfs",
        max_paths=10,
    )
    assert lazy._loaded_node_cols == frozenset({"score"})

    # A different referenced column triggers a re-projection.
    lazy.search(
        start_nodes=[0],
        visit=pl.col("dest.name") != "",
        stop=pl.col("dest.id") == 3,
        strategy="bfs",
        max_paths=10,
    )
    assert lazy._loaded_node_cols == frozenset({"name"})


def test_from_lazy_search_without_payload_columns() -> None:
    nodes, edges = _tables()
    lazy = rxg.Graph.from_lazy(nodes.lazy(), edges.lazy())

    result = lazy.search(
        start_nodes=[0],
        stop=pl.col("dest.id") == 2,
        strategy="bfs",
        max_paths=10,
    )
    assert [p.nodes for p in result.paths] == [[0, 1, 2]]
    assert lazy._loaded_node_cols == frozenset()
    assert lazy._loaded_edge_cols == frozenset()


def test_from_lazy_supports_topology_only_queries() -> None:
    nodes, edges = _tables()
    lazy = rxg.Graph.from_lazy(nodes.lazy(), edges.lazy())

    assert lazy.bfs(0) == [0, 1, 2, 3]
    assert lazy.reachable_nodes(0) == [0, 1, 2, 3]
    assert lazy.out_degrees() == [1, 1, 1, 0]


def test_from_lazy_without_optional_type_column() -> None:
    nodes, edges = _tables()
    # Add a string ``type`` column to exercise the optional-topology-column path.
    nodes = nodes.with_columns(pl.lit("node").alias("type"))
    edges = edges.with_columns(pl.lit("edge").alias("type"))

    lazy = rxg.Graph.from_lazy(nodes.lazy(), edges.lazy())
    eager = rxg.Graph(nodes, edges)
    kw = _search_kwargs("bfs")
    assert _paths(lazy.search(**kw)) == _paths(eager.search(**kw))
