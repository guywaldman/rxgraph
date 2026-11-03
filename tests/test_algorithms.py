import polars as pl
import pytest
import rxgraph as rxg


def graph() -> rxg.Graph:
    nodes = pl.DataFrame(
        {"id": [10, 20, 30, 40, 50]},
        schema={"id": pl.UInt64},
    )
    edges = pl.DataFrame(
        {"id": [1, 2, 3, 4], "src": [10, 10, 20, 30], "dest": [20, 30, 40, 40]},
        schema={"id": pl.UInt64, "src": pl.UInt64, "dest": pl.UInt64},
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


def test_multi_source_reachability() -> None:
    g = graph()
    assert g.ancestors_many([40, 30, 40]) == [20, 10]
    assert g.descendants_many(iter([20, 10, 20])) == [40, 30]
    assert g.ancestors_many([50]) == []
    assert g.descendants_many([50]) == []
    assert g.ancestors_many([]) == []
    assert g.descendants_many([]) == []
    for method in (g.ancestors_many, g.descendants_many):
        with pytest.raises(ValueError, match="node id 99"):
            method([10, 99])


@pytest.mark.parametrize("labeled", [False, True])
def test_multi_source_reachability_with_cycles_and_labels(labeled: bool) -> None:
    edges = [
        ("a", "b"),
        ("a", "b"),
        ("a", "c"),
        ("b", "d"),
        ("c", "d"),
        ("d", "b"),
        ("d", "d"),
    ]
    if labeled:
        g = rxg.Graph.from_edges(edges, nodes=["isolated"])
    else:
        nodes = pl.DataFrame({"id": ["a", "b", "c", "d", "isolated"]})
        edge_table = pl.DataFrame(
            {
                "id": [str(i) for i in range(len(edges))],
                "src": [src for src, _ in edges],
                "dest": [dest for _, dest in edges],
            }
        )
        g = rxg.Graph(nodes, edge_table)
    assert g.ancestors_many(["d", "c", "d"]) == ["b", "a"]
    assert g.descendants_many(["a", "c", "a"]) == ["b", "d"]
    assert g.ancestors_many(["b"]) == ["a", "d", "c"]
    assert g.descendants_many(["b"]) == ["d"]
    assert g.ancestors_many(["isolated"]) == []
    assert g.descendants_many(["isolated"]) == []
    assert g.ancestors_many([]) == []
    assert g.descendants_many([]) == []


def test_degrees_and_components() -> None:
    g = graph()

    assert g.out_degrees() == [2, 1, 1, 0, 0]
    assert g.in_degrees() == [0, 1, 1, 2, 0]
    assert g.degrees() == [2, 2, 2, 2, 0]
    assert g.weakly_connected_components() == [[10, 20, 30, 40], [50]]


def test_string_id_dataframe_graph() -> None:
    nodes = pl.DataFrame(
        {"id": ["a", "b", "c"]},
        schema={"id": pl.String},
    )
    edges = pl.DataFrame(
        {"id": ["ab", "bc"], "src": ["a", "b"], "dest": ["b", "c"]},
        schema={"id": pl.String, "src": pl.String, "dest": pl.String},
    )
    g = rxg.Graph(nodes, edges)

    assert g.bfs("a") == ["a", "b", "c"]
    assert g.shortest_path("a", "c") == ["a", "b", "c"]


def test_missing_node_errors_are_informative() -> None:
    with pytest.raises(ValueError, match="node id 99"):
        graph().bfs(99)
