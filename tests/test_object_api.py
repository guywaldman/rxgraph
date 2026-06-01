from dataclasses import asdict, is_dataclass

import pytest
import rxgraph as rxg


def test_from_edges_algorithms_return_labels() -> None:
    graph = rxg.Graph.from_edges(
        [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        nodes=["a", "b", "c", "d", "isolated"],
    )

    assert graph.node_count == 5
    assert graph.edge_count == 4
    assert graph.bfs("a") == ["a", "b", "c", "d"]
    assert graph.dfs("a") == ["a", "b", "d", "c"]
    assert graph.shortest_path("a", "d") == ["a", "b", "d"]
    assert graph.shortest_path("d", "a") is None
    assert graph.reachable_nodes("isolated") == ["isolated"]
    assert graph.weakly_connected_components() == [["a", "b", "c", "d"], ["isolated"]]


def test_digraph_from_edges_traverses_both_directions() -> None:
    graph = rxg.DiGraph.from_edges(
        [("a", "b"), ("b", "c")],
        nodes=["a", "b", "c", "isolated"],
    )

    assert graph.node_count == 4
    assert graph.edge_count == 4
    assert graph.shortest_path("c", "a") == ["c", "b", "a"]
    assert graph.shortest_path("a", "c") == ["a", "b", "c"]
    assert graph.reachable_nodes("isolated") == ["isolated"]


def test_from_edges_accepts_integer_labels_that_are_not_internal_ids() -> None:
    graph = rxg.Graph.from_edges([(10, 20), (20, 30)])

    assert graph.node_id(10) == 0
    assert graph.node_id(30) == 2
    assert graph.shortest_path(10, 30) == [10, 20, 30]


def test_from_edges_traversal_uses_polars_reexports_and_returns_labels() -> None:
    graph = rxg.Graph.from_edges(
        [
            ("a", "b", {"price": 5, "kind": "route"}),
            ("b", "c", {"price": 6, "kind": "route"}),
            ("a", "c", {"price": 100, "kind": "skip"}),
        ],
        nodes=[
            ("a", {"closed": False}),
            ("b", {"closed": False}),
            ("c", {"closed": False}),
        ],
    )
    s = lambda name: rxg.col(f"state.{name}")
    d = lambda name: rxg.col(f"dest.{name}")
    e = lambda name: rxg.col(f"edge.{name}")
    result = graph.search(
        start_nodes=["a"],
        visit=(~d("closed")) & (e("kind") != "skip") & ((s("spent") + e("price")) < 20),
        next_state={"spent": s("spent") + e("price")},
        stop=rxg.col("dest.id") == rxg.lit(graph.node_id("c")),
        initial_state={"spent": 0},
        max_depth=3,
        max_paths=10,
    )

    assert len(result.paths) == 1
    assert is_dataclass(result)
    assert is_dataclass(result.paths[0])
    assert asdict(result.paths[0]) == {
        "nodes": ["a", "b", "c"],
        "edges": [0, 1],
        "state": {"spent": 11},
        "intermediate_states": None,
    }
    assert result.paths[0].nodes == ["a", "b", "c"]
    assert result.paths[0].edges == [0, 1]


def test_from_edges_errors_for_unknown_label_and_reserved_attrs() -> None:
    graph = rxg.Graph.from_edges([("a", "b")])

    with pytest.raises(ValueError, match="node label 'missing'"):
        graph.node_id("missing")
    with pytest.raises(ValueError, match="reserved keys: id"):
        rxg.Graph.from_edges([], nodes=[("a", {"id": 1})])
    with pytest.raises(ValueError, match="reserved keys: src"):
        rxg.Graph.from_edges([("a", "b", {"src": 1})])
    with pytest.raises(ValueError, match="reserved keys: id"):
        rxg.Graph.from_edges([("a", "b", {"id": 1})])
