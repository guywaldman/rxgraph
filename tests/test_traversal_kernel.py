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


def test_search_supports_native_list_operations() -> None:
    nodes = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "tags": [[10], [20, 21], [30, 31, None]],
            "dupes": [[1, 1, 2], [2, 2, 3], [3, 3, None]],
            "flags": [[True], [True, True], [True, False, None]],
            "names": [["s"], ["a", "b"], ["x", "y"]],
        },
        schema={
            "id": pl.UInt64,
            "tags": pl.List(pl.Int64),
            "dupes": pl.List(pl.Int64),
            "flags": pl.List(pl.Boolean),
            "names": pl.List(pl.String),
        },
    )
    edges = pl.DataFrame(
        {"id": [10, 11], "src": [0, 1], "dest": [1, 2]},
        schema={"id": pl.UInt64, "src": pl.UInt64, "dest": pl.UInt64},
    )
    graph = rxg.Graph(nodes, edges)

    tags = pl.col("dest.tags")
    result = graph.search(
        start_nodes=[0],
        visit=tags.list.len() > 0,
        next_state={
            "tags": pl.concat_list([pl.col("state.tags"), tags]),
            "contains": tags.list.contains(31),
            "first": tags.list.first(),
            "last": tags.list.last(),
            "slice": tags.list.slice(0, 2),
            "reverse": tags.list.reverse(),
            "sort": tags.list.sort(),
            "unique": pl.col("dest.dupes").list.unique(),
            "drop_nulls": tags.list.drop_nulls(),
            "sum": tags.list.sum(),
            "min": tags.list.min(),
            "max": tags.list.max(),
            "mean": tags.list.mean(),
            "median": tags.list.median(),
            "any": pl.col("dest.flags").list.any(),
            "all": pl.col("dest.flags").list.all(),
            "count": tags.list.count_matches(31),
            "n_unique": tags.list.n_unique(),
            "join": pl.col("dest.names").list.join(","),
            "shift": tags.list.shift(1),
            "every": tags.list.gather_every(2),
            "union": pl.col("state.tags").list.set_union(tags),
            "intersection": pl.col("state.tags").list.set_intersection(tags),
            "difference": pl.col("state.tags").list.set_difference(tags),
            "symmetric": pl.col("state.tags").list.set_symmetric_difference(tags),
            "inc": tags.list.eval(pl.element() + 1),
            "filtered": tags.list.filter(pl.element() > 30),
        },
        stop=(pl.col("dest.id") == 2) & pl.col("state.contains"),
        initial_state={"tags": [10]},
        max_depth=2,
        max_paths=1,
    )

    state = result.paths[0].state
    assert result.paths[0].nodes == [0, 1, 2]
    assert state["tags"] == [10, 20, 21, 30, 31, None]
    assert state["first"] == 30
    assert state["last"] is None
    assert state["slice"] == [30, 31]
    assert state["reverse"] == [None, 31, 30]
    assert state["sort"] == [None, 30, 31]
    assert state["unique"] == [3, None]
    assert state["drop_nulls"] == [30, 31]
    assert state["sum"] == 61
    assert state["min"] == 30
    assert state["max"] == 31
    assert state["mean"] == 30.5
    assert state["median"] == 30.5
    assert state["any"] is True
    assert state["all"] is False
    assert state["count"] == 1
    assert state["n_unique"] == 3
    assert state["join"] == "x,y"
    assert state["shift"] == [None, 30, 31]
    assert state["every"] == [30, None]
    assert state["union"] == [10, 20, 21, 30, 31, None]
    assert state["intersection"] == []
    assert state["difference"] == [10, 20, 21]
    assert state["symmetric"] == [10, 20, 21, 30, 31, None]
    assert state["inc"] == [31, 32, None]
    assert state["filtered"] == [31]


def test_search_supports_native_struct_operations() -> None:
    nodes = (
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "score": [0, 5, 9],
                "label": ["s", "a", "b"],
            },
            schema={"id": pl.UInt64, "score": pl.Int64, "label": pl.String},
        )
        .with_columns(pl.struct(["score", "label"]).alias("meta"))
        .select("id", "score", "meta")
    )
    edges = pl.DataFrame(
        {"id": [10, 11], "src": [0, 1], "dest": [1, 2]},
        schema={"id": pl.UInt64, "src": pl.UInt64, "dest": pl.UInt64},
    )
    graph = rxg.Graph(nodes, edges)

    meta = pl.col("dest.meta")
    result = graph.search(
        start_nodes=[0],
        next_state={
            "score": meta.struct.field("score"),
            "renamed": meta.struct.rename_fields(["points", "name"]),
            "extended": meta.struct.with_fields(
                (pl.col("dest.score") + 1).alias("next_score")
            ),
            "json": meta.struct.json_encode(),
        },
        stop=pl.col("state.score") == 9,
        initial_state={"score": 0},
        max_depth=2,
        max_paths=1,
    )

    state = result.paths[0].state
    assert state["score"] == 9
    assert state["renamed"] == {"points": 9, "name": "b"}
    assert state["extended"] == {"score": 9, "label": "b", "next_score": 10}
    assert json.loads(state["json"]) == {"score": 9, "label": "b"}


def test_search_supports_conditionals_struct_filters_and_edge_state() -> None:
    rule_dtype = pl.List(
        pl.Struct(
            {
                "protocol": pl.String,
                "from_port": pl.Int64,
                "to_port": pl.Int64,
            }
        )
    )
    nodes = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "label": ["start", "transparent", "gate"],
            "has_rules": [False, False, True],
            "allow_rules": [
                [],
                [],
                [
                    {"protocol": "tcp", "from_port": 80, "to_port": 80},
                    {"protocol": "udp", "from_port": 53, "to_port": 53},
                ],
            ],
        },
        schema={
            "id": pl.UInt64,
            "label": pl.String,
            "has_rules": pl.Boolean,
            "allow_rules": rule_dtype,
        },
    )
    edges = pl.DataFrame(
        {
            "id": [10, 11],
            "src": [0, 1],
            "dest": [1, 2],
            "map": [100, 3000],
        },
        schema={
            "id": pl.UInt64,
            "src": pl.UInt64,
            "dest": pl.UInt64,
            "map": pl.Int64,
        },
    )
    graph = rxg.Graph(nodes, edges)

    rules = pl.col("dest.allow_rules")
    matching_rules = rules.list.filter(
        pl.col("state.allowed_protocols").list.contains(
            pl.element().struct.field("protocol")
        )
    )
    result = graph.search(
        start_nodes=[0],
        visit=pl.when(pl.col("dest.has_rules"))
        .then(matching_rules.list.len() > 0)
        .otherwise(pl.lit(True)),
        next_state={
            "allowed_rules": pl.when(pl.col("dest.has_rules"))
            .then(pl.col("state.allowed_rules").list.set_intersection(rules))
            .otherwise(pl.col("state.allowed_rules")),
            "last_edge_map": pl.col("edge.map"),
            "lazy_guard": pl.when(pl.lit(True))
            .then(pl.lit(1))
            .otherwise(pl.col("dest.label") + 1),
        },
        stop=pl.col("dest.id") == 2,
        initial_state={
            "allowed_protocols": ["tcp"],
            "allowed_rules": [{"to_port": 80, "protocol": "tcp", "from_port": 80}],
            "last_edge_map": 0,
            "lazy_guard": 0,
        },
        max_depth=2,
        max_paths=1,
    )

    assert result.paths[0].nodes == [0, 1, 2]
    assert result.paths[0].state["allowed_rules"] == [
        {"to_port": 80, "protocol": "tcp", "from_port": 80}
    ]
    assert result.paths[0].state["last_edge_map"] == 3000
    assert result.paths[0].state["lazy_guard"] == 1


def test_search_supports_list_eval_when_literals_and_explode() -> None:
    nodes = pl.DataFrame(
        {
            "id": [0, 1],
            "protocols": [[], ["-1", "icmp"]],
        },
        schema={"id": pl.UInt64, "protocols": pl.List(pl.String)},
    )
    edges = pl.DataFrame(
        {"id": [10], "src": [0], "dest": [1]},
        schema={"id": pl.UInt64, "src": pl.UInt64, "dest": pl.UInt64},
    )
    graph = rxg.Graph(nodes, edges)

    expanded = (
        pl.col("dest.protocols")
        .list.eval(
            pl.when(pl.element() == "-1")
            .then(pl.lit(["tcp", "udp"]))
            .otherwise(pl.element())
        )
        .list.explode()
    )
    result = graph.search(
        start_nodes=[0],
        next_state={"expanded": expanded},
        stop=pl.col("dest.id") == 1,
        initial_state={"expanded": []},
        max_paths=1,
    )

    assert result.paths[0].state["expanded"] == ["tcp", "udp", "icmp"]


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
