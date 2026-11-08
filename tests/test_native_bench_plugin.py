"""Supplemental native-plugin tests; build benches/native-plugin to enable."""

import os

import polars as pl
import pytest


def plugin():
    try:
        import rxgraph_native_bench as rxg
    except ImportError:
        if os.environ.get("RXGRAPH_REQUIRE_NATIVE_BENCH_PLUGIN"):
            raise
        pytest.skip("benchmark plugin not built")
    return rxg


@pytest.mark.parametrize("mode", ["arrow", "eager", "lazy"])
@pytest.mark.parametrize("parallel", [False, True])
def test_compound_native_state_streams(mode, parallel, tmp_path):
    rxg = plugin()
    nodes = pl.DataFrame({"id": [0, 1, 2]}, schema={"id": pl.UInt64})
    edges = pl.DataFrame(
        {"id": [0, 1, 2], "src": [0, 0, 1], "dest": [1, 2, 2], "cost": [1, 1, 1]},
        schema={key: pl.UInt64 for key in ("id", "src", "dest", "cost")},
    )
    if mode == "arrow":
        graph = rxg.Graph(nodes, edges)
    else:
        nodes.write_parquet(tmp_path / "n.parquet")
        edges.write_parquet(tmp_path / "e.parquet")
        graph = rxg.Graph.from_parquet(
            tmp_path / "n.parquet", tmp_path / "e.parquet", payloads=mode
        )
    kwargs = dict(
        start_nodes=[0],
        kernel="bench_arrow" if mode == "arrow" else "bench_typed",
        params={"budget": 10, "target": 2, "large": True},
        parallel=parallel,
        intermediate_states=True,
    )
    expected = graph.search(**kwargs)
    with graph.search_batches(**kwargs, batch_size=1) as stream:
        actual = [path for batch in stream for path in batch]
    assert actual == expected.paths
    assert len(actual) == 2
    actual[0].state["details"]["values"][0] = 99
    assert actual[1].state["details"]["values"][0] == 7
    assert actual[0].intermediate_states[-1]["details"]["values"][0] == 7


def target_graph(rxg):
    return rxg.Graph.from_edges(
        [
            ("a", "b", {"cost": 1}),
            ("b", "target", {"cost": 1}),
            ("a", "dead", {"cost": 1}),
            ("dead", "dead2", {"cost": 1}),
            ("dead2", "dead3", {"cost": 1}),
        ]
    )


@pytest.mark.parametrize("strategy", ["dfs", "bfs"])
@pytest.mark.parametrize("evaluation", ["incremental", "lazy"])
def test_target_optimizations_eager_and_stream(strategy, evaluation):
    graph = target_graph(plugin())
    kwargs = dict(
        start_nodes=["a"],
        kernel="bench_arrow",
        params={"budget": 10, "target": "target"},
        strategy=strategy,
        parallel=False,
        intermediate_states=True,
    )
    baseline = graph.search(**kwargs)
    assert graph.search(**kwargs, optimizations={}).paths == baseline.paths
    opts = {"evaluation": evaluation, "reverse": True, "release_expanded": True}
    optimized = graph.search(**kwargs, optimizations=opts)
    assert optimized.paths == baseline.paths
    assert optimized.stats.evaluated_edges < baseline.stats.evaluated_edges
    with graph.search_batches(**kwargs, optimizations=opts, batch_size=1) as stream:
        assert [p for batch in stream for p in batch] == baseline.paths
    assert (
        graph.search_first(
            **{k: v for k, v in kwargs.items() if k not in ("strategy", "parallel")},
            optimizations=opts,
        )
        == baseline.paths[0]
    )
    assert graph.search(**kwargs, max_paths=0, optimizations=opts).paths == []
    with graph.search_batches(**kwargs, optimizations=opts, batch_size=1) as stream:
        assert stream.stats.evaluated_edges == 0
    assert list(stream) == []


def test_target_optimizations_validate_and_reject_unsupported():
    graph = target_graph(plugin())
    kwargs = dict(
        start_nodes=["a"], params={"budget": 10, "target": "target"}, parallel=False
    )
    with pytest.raises(ValueError, match="named graph-native"):
        graph.search(start_nodes=["a"], optimizations={})
    for method in (graph.search, graph.search_batches):
        with pytest.raises(ValueError, match="memoization.*typed kernels"):
            method(kernel="bench_typed", optimizations={"memo_capacity": 5}, **kwargs)
        with pytest.raises((TypeError, ValueError), match="evaluation"):
            method(
                kernel="bench_arrow", optimizations={"evaluation": "wrong"}, **kwargs
            )
        with pytest.raises(TypeError, match="reverse"):
            method(kernel="bench_arrow", optimizations={"reverse": 1}, **kwargs)
        with pytest.raises(ValueError, match="unknown optimization"):
            method(kernel="bench_arrow", optimizations={"unknown": True}, **kwargs)
        with pytest.raises((TypeError, ValueError), match="memo_capacity"):
            method(kernel="bench_memo", optimizations={"memo_capacity": -1}, **kwargs)
        with pytest.raises(RuntimeError, match="does not support memoization"):
            method(kernel="bench_arrow", optimizations={"memo_capacity": 5}, **kwargs)
    with pytest.raises(RuntimeError, match="does not support target optimizations"):
        graph.search(kernel="bench_legacy", optimizations={}, **kwargs)


@pytest.mark.parametrize("evaluation", ["incremental", "lazy"])
def test_memoizable_registered_kernel(evaluation):
    graph = target_graph(plugin())
    kwargs = dict(
        start_nodes=["a"],
        kernel="bench_memo",
        params={"budget": 10, "target": "target"},
        parallel=False,
    )
    expected = graph.search(**kwargs).paths
    opts = {"evaluation": evaluation, "reverse": True, "memo_capacity": 10}
    assert graph.search(**kwargs, optimizations=opts).paths == expected
    with graph.search_batches(**kwargs, optimizations=opts, batch_size=1) as stream:
        assert [p for batch in stream for p in batch] == expected


def test_memoization_reuses_transitions_and_is_scoped_to_each_search():
    rxg = plugin()
    graph = rxg.Graph.from_edges(
        [
            ("a", "b", {"cost": 1}),
            ("a", "c", {"cost": 1}),
            ("b", "join", {"cost": 1}),
            ("c", "join", {"cost": 1}),
            ("join", "target", {"cost": 1}),
        ]
    )
    kwargs = dict(
        start_nodes=["a"],
        kernel="bench_memo",
        params={"budget": 10, "target": "target"},
        parallel=False,
    )
    native = rxg._native
    native.reset_exact_transitions()
    baseline = graph.search(**kwargs).paths
    assert native.exact_transitions() == 6
    for stream in (False, True):
        native.reset_exact_transitions()
        options = {"memo_capacity": 10}
        if stream:
            with graph.search_batches(
                **kwargs, optimizations=options, batch_size=1
            ) as result:
                paths = [p for batch in result for p in batch]
        else:
            paths = graph.search(**kwargs, optimizations=options).paths
        assert paths == baseline
        assert native.exact_transitions() == 5


def test_dual_registration_retains_typed_dispatch():
    graph = target_graph(plugin())
    kwargs = dict(
        start_nodes=["a"],
        kernel="weighted_budget",
        params={"weight_col": "cost", "budget": 10, "target": "target"},
    )
    expected = graph.search(**kwargs).paths
    assert expected
    assert graph.search(**kwargs, optimizations={}).paths == expected
    with graph.search_batches(**kwargs, optimizations={}) as stream:
        assert [path for batch in stream for path in batch] == expected


@pytest.mark.parametrize("mode", ["eager", "lazy"])
@pytest.mark.parametrize("evaluation", ["incremental", "lazy"])
@pytest.mark.parametrize("strategy", ["dfs", "bfs"])
def test_file_backed_target_optimizations(mode, evaluation, strategy, tmp_path):
    rxg = plugin()
    nodes = pl.DataFrame({"id": [0, 1]}, schema={"id": pl.UInt64})
    edges = pl.DataFrame(
        {"id": [0], "src": [0], "dest": [1], "cost": [1]},
        schema={key: pl.UInt64 for key in ("id", "src", "dest", "cost")},
    )
    nodes.write_parquet(tmp_path / "nodes.parquet")
    edges.write_parquet(tmp_path / "edges.parquet")
    graph = rxg.Graph.from_parquet(
        tmp_path / "nodes.parquet", tmp_path / "edges.parquet", payloads=mode
    )
    kwargs = dict(
        start_nodes=[0],
        kernel="bench_typed",
        params={"budget": 10, "target": 1},
        strategy=strategy,
    )
    expected = graph.search(**kwargs).paths
    options = {"evaluation": evaluation, "reverse": True, "release_expanded": True}
    assert graph.search(**kwargs, optimizations=options).paths == expected
    with graph.search_batches(**kwargs, optimizations=options, batch_size=1) as stream:
        assert [path for batch in stream for path in batch] == expected


@pytest.mark.parametrize("evaluation", ["incremental", "lazy"])
@pytest.mark.parametrize("strategy", ["dfs", "bfs"])
def test_typed_target_optimizations_prune(evaluation, strategy):
    graph = target_graph(plugin())
    kwargs = dict(
        start_nodes=["a"],
        kernel="bench_typed",
        params={"budget": 10, "target": "target"},
        strategy=strategy,
        parallel=False,
    )
    baseline = graph.search(**kwargs)
    options = {"evaluation": evaluation, "reverse": True, "release_expanded": True}
    result = graph.search(**kwargs, optimizations=options)
    assert result.paths == baseline.paths
    assert result.stats.evaluated_edges < baseline.stats.evaluated_edges
    with graph.search_batches(**kwargs, optimizations=options, batch_size=1) as stream:
        assert [path for batch in stream for path in batch] == baseline.paths
