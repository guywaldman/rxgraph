import polars as pl
import pytest
import rxgraph as rxg


def test_polars_kernel_traversal() -> None:
    nodes = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "closed": [False, False, False],
            "risk": [0, 1, 2],
        },
        schema={"id": pl.UInt64, "closed": pl.Boolean, "risk": pl.Int32},
    )
    edges = pl.DataFrame(
        {
            "id": [10, 11, 12],
            "src": [0, 1, 0],
            "dest": [1, 2, 2],
            "price": [5, 6, 100],
            "kind": ["route", "route", "decoy"],
        },
        schema={
            "id": pl.UInt64,
            "src": pl.UInt64,
            "dest": pl.UInt64,
            "price": pl.UInt64,
            "kind": pl.String,
        },
    )
    graph = rxg.Graph([("n", nodes)], [("e", edges)])

    s = lambda name: pl.col(f"state.{name}")
    d = lambda name: pl.col(f"dest.{name}")
    e = lambda name: pl.col(f"edge.{name}")
    kernel = rxg.Kernel(
        visit=(~d("closed")) & (e("kind") != "decoy") & ((s("spent") + e("price")) <= 12),
        next_state={"spent": s("spent") + e("price")},
        stop=pl.col("dest.id") == 2,
        initial_state={"spent": 0},
    )
    traversal = rxg.Traversal(kernel, [0], 3, 10, "dfs", intermediate_states=True)

    result = graph.search(traversal)

    assert len(result.paths) == 1
    assert result.paths[0].nodes == [0, 1, 2]
    assert result.paths[0].state == {"spent": 11}
    assert result.paths[0].intermediate_states == [
        {"spent": 0},
        {"spent": 5},
        {"spent": 11},
    ]
    assert result.stats.evaluated_edges == 3
    assert result.stats.accepted_edges == 2


def test_search_accepts_kernel_params_and_list_state() -> None:
    nodes = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "tags": [[10], [20, 21], [30]],
        },
        schema={"id": pl.UInt64, "tags": pl.List(pl.UInt64)},
    )
    edges = pl.DataFrame(
        {"id": [10, 11], "src": [0, 1], "dest": [1, 2]},
        schema={"id": pl.UInt64, "src": pl.UInt64, "dest": pl.UInt64},
    )
    graph = rxg.Graph(nodes, edges)

    result = graph.search(
        start_nodes=[0],
        stop=pl.col("dest.id") == 2,
        next_state={
            "tags": pl.concat_list([pl.col("state.tags"), pl.col("dest.tags")]),
        },
        initial_state={"tags": []},
        max_depth=2,
        max_paths=1,
        intermediate_states=True,
    )

    assert result.paths[0].nodes == [0, 1, 2]
    assert result.paths[0].state == {"tags": [20, 21, 30]}
    assert result.paths[0].intermediate_states == [
        {"tags": []},
        {"tags": [20, 21]},
        {"tags": [20, 21, 30]},
    ]


def test_search_docstrings_include_high_level_params() -> None:
    assert rxg.Graph.search.__doc__
    assert "visit" in rxg.Graph.search.__doc__
    assert "next_state" in rxg.Graph.search.__doc__
    assert "initial_state" in rxg.Graph.search.__doc__
    assert rxg.Traversal.__init__.__doc__
    assert "intermediate_states" in rxg.Traversal.__init__.__doc__


def test_parallel_bfs_matches_serial_bfs() -> None:
    nodes = pl.DataFrame(
        {"id": [0, 1, 2, 3], "closed": [False, False, False, False]},
        schema={"id": pl.UInt64, "closed": pl.Boolean},
    )
    edges = pl.DataFrame(
        {"id": [10, 11, 12, 13], "src": [0, 0, 1, 2], "dest": [1, 2, 3, 3]},
        schema={"id": pl.UInt64, "src": pl.UInt64, "dest": pl.UInt64},
    )
    graph = rxg.Graph([("n", nodes)], [("e", edges)])
    kernel = rxg.Kernel(
        visit=~pl.col("dest.closed"),
        next_state={},
        stop=pl.col("dest.id") == 3,
        initial_state={},
    )

    serial = graph.search(rxg.Traversal(kernel, [0], 2, 10, "bfs", "off"))
    parallel = graph.search(rxg.Traversal(kernel, [0], 2, 10, "bfs", "on"))

    assert parallel.paths[0].nodes == serial.paths[0].nodes
    assert parallel.stats.evaluated_edges == serial.stats.evaluated_edges
    assert parallel.stats.accepted_edges == serial.stats.accepted_edges


def test_graph_schema_errors_are_informative() -> None:
    nodes = pl.DataFrame(
        {"id": ["0"]},
        schema={"id": pl.String},
    )
    edges = pl.DataFrame(
        {"id": ["e"], "src": [0], "dest": [0]},
        schema={"id": pl.String, "src": pl.UInt64, "dest": pl.UInt64},
    )

    with pytest.raises(
        ValueError,
        match="same ID type",
    ):
        rxg.Graph([("n", nodes)], [("e", edges)])


def test_kernel_schema_errors_are_informative() -> None:
    nodes = pl.DataFrame(
        {"id": [0, 1]},
        schema={"id": pl.UInt64},
    )
    edges = pl.DataFrame(
        {"id": [10], "src": [0], "dest": [1]},
        schema={"id": pl.UInt64, "src": pl.UInt64, "dest": pl.UInt64},
    )
    graph = rxg.Graph([("n", nodes)], [("e", edges)])
    kernel = rxg.Kernel(
        visit=pl.col("edge.price") > 0,
        next_state={},
        stop=pl.col("dest.id") == 1,
        initial_state={},
    )

    with pytest.raises(
        RuntimeError,
        match='column "price" is missing',
    ):
        graph.search(rxg.Traversal(kernel, [0], 1, 1, "dfs"))
