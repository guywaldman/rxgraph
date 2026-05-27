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
            "src": [0, 1, 0],
            "dest": [1, 2, 2],
            "price": [5, 6, 100],
            "kind": ["route", "route", "decoy"],
        },
        schema={
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
        visit=(~d("closed"))
        & (e("kind") != "decoy")
        & ((s("spent") + e("price")) <= 12),
        next_state={"spent": s("spent") + e("price")},
        stop=pl.col("dest.id") == 2,
        initial_state={"spent": 0},
    )
    traversal = rxg.Traversal(kernel, [0], 3, 10, "dfs")

    result = graph.search(traversal)

    assert len(result.paths) == 1
    assert result.paths[0].nodes == [0, 1, 2]
    assert result.stats.evaluated_edges == 3
    assert result.stats.accepted_edges == 2


def test_parallel_bfs_matches_serial_bfs() -> None:
    nodes = pl.DataFrame(
        {"id": [0, 1, 2, 3], "closed": [False, False, False, False]},
        schema={"id": pl.UInt64, "closed": pl.Boolean},
    )
    edges = pl.DataFrame(
        {"src": [0, 0, 1, 2], "dest": [1, 2, 3, 3]},
        schema={"src": pl.UInt64, "dest": pl.UInt64},
    )
    graph = rxg.Graph([("n", nodes)], [("e", edges)])
    kernel = rxg.Kernel(
        visit=~pl.col("dest.closed"),
        next_state={},
        stop=pl.col("dest.id") == 3,
        initial_state={},
    )

    serial = graph.search(rxg.Traversal(kernel, [0], 2, 1, "bfs", "off"))
    parallel = graph.search(rxg.Traversal(kernel, [0], 2, 1, "bfs", "on", 0, 0))

    assert parallel.paths[0].nodes == serial.paths[0].nodes
    assert parallel.stats.evaluated_edges == serial.stats.evaluated_edges
    assert parallel.stats.accepted_edges == serial.stats.accepted_edges


def test_graph_schema_errors_are_informative() -> None:
    nodes = pl.DataFrame(
        {"id": ["0"]},
        schema={"id": pl.String},
    )
    edges = pl.DataFrame(
        {"src": [0], "dest": [0]},
        schema={"src": pl.UInt64, "dest": pl.UInt64},
    )

    with pytest.raises(
        ValueError,
        match='node table "n" column "id" must have Arrow type UInt64',
    ):
        rxg.Graph([("n", nodes)], [("e", edges)])


def test_kernel_schema_errors_are_informative() -> None:
    nodes = pl.DataFrame(
        {"id": [0, 1]},
        schema={"id": pl.UInt64},
    )
    edges = pl.DataFrame(
        {"src": [0], "dest": [1]},
        schema={"src": pl.UInt64, "dest": pl.UInt64},
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
        match='column "price" is not present in any edge table',
    ):
        graph.search(rxg.Traversal(kernel, [0], 1, 1, "dfs"))
