import json

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
    result = graph.search(
        start_nodes=[0],
        visit=(~d("closed"))
        & (e("kind") != "decoy")
        & ((s("spent") + e("price")) <= 12),
        next_state={"spent": s("spent") + e("price")},
        stop=pl.col("dest.id") == 2,
        initial_state={"spent": 0},
        max_depth=3,
        max_paths=10,
        intermediate_states=True,
    )

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


def test_search_defaults_stop_when_max_paths_is_set() -> None:
    graph = rxg.Graph.from_edges([("a", "b"), ("b", "c")])

    result = graph.search(start_nodes=["a"], max_paths=1)

    assert result.paths[0].nodes == ["a", "b"]


def test_search_requires_stop_or_max_paths() -> None:
    graph = rxg.Graph.from_edges([("a", "b")])

    with pytest.raises(TypeError, match="requires 'stop' or 'max_paths'"):
        graph.search(start_nodes=["a"])


def test_kernel_defaults_accept_and_never_stop() -> None:
    assert json.loads(rxg.Kernel().visit.meta.serialize(format="json")) == {
        "Literal": {"Scalar": {"Boolean": True}}
    }
    assert json.loads(rxg.Kernel().stop.meta.serialize(format="json")) == {
        "Literal": {"Scalar": {"Boolean": False}}
    }


def test_search_defaults_stop_when_max_paths_is_set_even_if_kernel_does_not() -> None:
    graph = rxg.Graph.from_edges([("a", "b")])
    result = graph.search(start_nodes=["a"], max_depth=1, max_paths=1)

    assert result.paths[0].nodes == ["a", "b"]


def test_search_result_objects_do_not_expose_internals() -> None:
    result = rxg.Graph.from_edges([("a", "b")]).search(
        start_nodes=["a"],
        max_paths=1,
    )
    path = result.paths[0]

    assert not hasattr(result, "__dict__")
    assert not hasattr(result, "_inner")
    assert not hasattr(path, "__dict__")
    assert not hasattr(path, "_inner")
    assert not hasattr(path, "_id_to_label")
    assert not hasattr(path, "_edge_id_to_label")
    assert path.nodes == ["a", "b"]
    assert path.edges == [0]


def test_digraph_search_maps_reverse_edges_to_original_ids() -> None:
    nodes = pl.DataFrame(
        {"id": [1, 2, 3]},
        schema={"id": pl.UInt64},
    )
    edges = pl.DataFrame(
        {"id": [10, 11], "src": [1, 2], "dest": [2, 3]},
        schema={"id": pl.UInt64, "src": pl.UInt64, "dest": pl.UInt64},
    )
    graph = rxg.DiGraph(nodes, edges)

    result = graph.search(
        start_nodes=[3],
        stop=pl.col("dest.id") == 1,
        max_depth=2,
        max_paths=1,
    )

    assert result.paths[0].nodes == [3, 2, 1]
    assert result.paths[0].edges == [11, 10]


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
    serial = graph.search(
        start_nodes=[0],
        visit=~pl.col("dest.closed"),
        stop=pl.col("dest.id") == 3,
        max_depth=2,
        max_paths=10,
        strategy="bfs",
        parallel="off",
    )
    parallel = graph.search(
        start_nodes=[0],
        visit=~pl.col("dest.closed"),
        stop=pl.col("dest.id") == 3,
        max_depth=2,
        max_paths=10,
        strategy="bfs",
        parallel="on",
    )

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
    with pytest.raises(
        RuntimeError,
        match='column "price" is missing',
    ):
        graph.search(
            start_nodes=[0],
            visit=pl.col("edge.price") > 0,
            stop=pl.col("dest.id") == 1,
            max_depth=1,
            max_paths=1,
        )
