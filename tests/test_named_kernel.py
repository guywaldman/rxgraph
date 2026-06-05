import polars as pl
import pytest
import rxgraph as rxg


def _weighted_graph() -> rxg.Graph:
    # a -> b (1), b -> c (1), a -> c (5). Cheapest path to "c" is a->b->c (cost 2).
    # ``from_edges`` infers Python ints as Arrow Int64; the "weighted_budget"
    # kernel reads ``cost`` via ``edge_u64`` which now coerces Int64 -> u64,
    # so no dtype workaround is needed.
    return rxg.Graph.from_edges(
        [
            ("a", "b", {"cost": 1}),
            ("b", "c", {"cost": 1}),
            ("a", "c", {"cost": 5}),
        ]
    )


def test_named_kernel_weighted_budget_returns_expected_paths() -> None:
    graph = _weighted_graph()

    result = graph.search(
        start_nodes=["a"],
        kernel="weighted_budget",
        params={"weight_col": "cost", "budget": 100, "target": "c"},
        columns=["cost"],
        max_paths=5,
    )

    paths = [p.nodes for p in result.paths]
    # Both routes to "c" are within budget.
    assert ["a", "b", "c"] in paths
    assert ["a", "c"] in paths
    assert all(p[-1] == "c" for p in paths)


def test_named_kernel_budget_rejects_over_budget_edges() -> None:
    graph = _weighted_graph()

    # Budget 2 only admits the cheap two-hop route; the direct a->c (cost 5) is rejected.
    result = graph.search(
        start_nodes=["a"],
        kernel="weighted_budget",
        params={"weight_col": "cost", "budget": 2, "target": "c"},
        columns=["cost"],
        max_paths=5,
    )

    assert [p.nodes for p in result.paths] == [["a", "b", "c"]]
    assert result.paths[0].state == {"spent": 2}


def test_named_kernel_target_accepts_raw_engine_id() -> None:
    graph = _weighted_graph()
    target_id = graph.node_id("c")

    result = graph.search(
        start_nodes=["a"],
        kernel="weighted_budget",
        params={"weight_col": "cost", "budget": 2, "target": target_id},
        columns=["cost"],
        max_paths=5,
    )

    assert [p.nodes for p in result.paths] == [["a", "b", "c"]]


def test_named_kernel_forbids_mixing_with_dsl() -> None:
    graph = _weighted_graph()

    with pytest.raises(ValueError, match="not both"):
        graph.search(
            start_nodes=["a"],
            kernel="weighted_budget",
            params={"weight_col": "cost", "budget": 2, "target": "c"},
            visit=pl.col("edge.cost") > 0,
            columns=["cost"],
            max_paths=5,
        )


def test_named_kernel_unknown_name_errors() -> None:
    graph = _weighted_graph()

    with pytest.raises(ValueError):
        graph.search(
            start_nodes=["a"],
            kernel="does_not_exist",
            params={},
            columns=["cost"],
            max_paths=5,
        )


def test_named_kernel_lazy_requires_columns() -> None:
    nodes = pl.DataFrame({"id": [0, 1, 2]}, schema={"id": pl.UInt64})
    edges = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "src": [0, 1, 0],
            "dest": [1, 2, 2],
            "cost": [1, 1, 5],
        },
        schema={
            "id": pl.UInt64,
            "src": pl.UInt64,
            "dest": pl.UInt64,
            "cost": pl.UInt64,
        },
    )
    lazy = rxg.Graph.from_lazy(nodes.lazy(), edges.lazy())

    with pytest.raises(ValueError, match="requires 'columns"):
        lazy.search(
            start_nodes=[0],
            kernel="weighted_budget",
            params={"weight_col": "cost", "budget": 100, "target": 2},
            max_paths=5,
        )


def test_named_kernel_lazy_loads_explicit_columns() -> None:
    nodes = pl.DataFrame({"id": [0, 1, 2]}, schema={"id": pl.UInt64})
    edges = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "src": [0, 1, 0],
            "dest": [1, 2, 2],
            "cost": [1, 1, 5],
        },
        schema={
            "id": pl.UInt64,
            "src": pl.UInt64,
            "dest": pl.UInt64,
            "cost": pl.UInt64,
        },
    )
    lazy = rxg.Graph.from_lazy(nodes.lazy(), edges.lazy())

    result = lazy.search(
        start_nodes=[0],
        kernel="weighted_budget",
        params={"weight_col": "cost", "budget": 2, "target": 2},
        columns=["cost"],
        max_paths=5,
    )

    assert [p.nodes for p in result.paths] == [[0, 1, 2]]
    assert lazy._loaded_edge_cols == frozenset({"cost"})
