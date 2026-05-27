import polars as pl
import pytest
import rxgraph as rxg


def graph() -> rxg.Graph:
    nodes = pl.DataFrame(
        {"id": [10, 20, 30, 40, 50]},
        schema={"id": pl.UInt64},
    )
    edges = pl.DataFrame(
        {"src": [10, 10, 20, 30], "dest": [20, 30, 40, 40]},
        schema={"src": pl.UInt64, "dest": pl.UInt64},
    )
    return rxg.Graph([("n", nodes)], [("e", edges)])


def test_bfs_dfs_and_reachable_nodes() -> None:
    g = graph()

    assert g.bfs(10) == [10, 20, 30, 40]
    assert g.bfs(10, max_depth=1) == [10, 20, 30]
    assert g.dfs(10) == [10, 20, 40, 30]
    assert g.reachable_nodes(50) == [50]


def test_shortest_path() -> None:
    g = graph()

    assert g.shortest_path(10, 40) == [10, 20, 40]
    assert g.shortest_path(40, 10) is None
    assert g.shortest_path(10, 10) == [10]


def test_degrees_and_components() -> None:
    g = graph()

    assert g.out_degrees() == [2, 1, 1, 0, 0]
    assert g.in_degrees() == [0, 1, 1, 2, 0]
    assert g.degrees() == [2, 2, 2, 2, 0]
    assert g.weakly_connected_components() == [[10, 20, 30, 40], [50]]


def test_missing_node_errors_are_informative() -> None:
    with pytest.raises(ValueError, match="node id 99"):
        graph().bfs(99)
