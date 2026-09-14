import gc

import polars as pl
import pytest
import rxgraph as rxg


@pytest.fixture(params=["arrow", "eager", "lazy"])
def graph(request, tmp_path):
    nodes = pl.DataFrame({"id": [0, 1, 2, 3]}, schema={"id": pl.UInt64})
    edges = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "src": [0, 0, 1, 2, 0],
            "dest": [1, 2, 3, 3, 3],
            "cost": [1, 1, 1, 1, 1],
        },
        schema={key: pl.UInt64 for key in ("id", "src", "dest", "cost")},
    )
    if request.param == "arrow":
        return rxg.Graph(nodes, edges)
    nodes.write_parquet(tmp_path / "nodes.parquet")
    edges.write_parquet(tmp_path / "edges.parquet")
    return rxg.Graph.from_parquet(
        tmp_path / "nodes.parquet", tmp_path / "edges.parquet", payloads=request.param
    )


def options(**kwargs):
    return dict(
        start_nodes=[0],
        kernel="weighted_budget",
        params={"weight_col": "cost", "budget": 10, "target": 3},
        parallel=False,
        **kwargs,
    )


@pytest.mark.parametrize("strategy", ["dfs", "bfs"])
@pytest.mark.parametrize("size", [1, 2, 1024])
@pytest.mark.parametrize("limit", [0, 1, 2, 10, None])
def test_batches_match_eager(graph, strategy, size, limit):
    kwargs = options(strategy=strategy, max_paths=limit, intermediate_states=True)
    expected = graph.search(**kwargs)
    with graph.search_batches(**kwargs, batch_size=size) as stream:
        assert isinstance(stream, rxg.SearchStream)
        assert rxg.SearchStream.__module__ == "rxgraph"
        assert stream.stats.evaluated_edges == 0
        batches = list(stream)
        assert all(0 < len(batch) <= size for batch in batches)
        actual = [path for batch in batches for path in batch]
        assert actual == expected.paths
        assert stream.stats.stopped_paths == len(actual)
    assert list(stream) == []


def test_close_keeps_delivered_results(graph):
    stream = graph.search_batches(**options(), batch_size=1)
    first = next(stream)
    count = stream.stats.evaluated_edges
    stream.close()
    stream.close()
    assert list(stream) == []
    assert stream.stats.evaluated_edges == count
    assert first[0].nodes == [0, 1, 3]
    assert first[0].state == {"spent": 2}


def test_stream_retains_graph_and_stats(graph):
    stream = graph.search_batches(**options(), batch_size=1)
    del graph
    gc.collect()
    assert sum(map(len, stream)) == 3
    assert stream.stats.stopped_paths == 3


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_batch_size(graph, size):
    with pytest.raises(ValueError, match="batch_size"):
        graph.search_batches(**options(), batch_size=size)


def test_labeled_results():
    graph = rxg.Graph.from_edges([("a", "b", {"cost": 1}), ("b", "c", {"cost": 2})])
    with graph.search_batches(
        start_nodes=["a"],
        kernel="weighted_budget",
        params={"weight_col": "cost", "budget": 5, "target": "c"},
    ) as stream:
        assert next(stream)[0].nodes == ["a", "b", "c"]


def test_payload_replacement_requires_closing_stream(graph):
    # Use the same projected tables that the legacy lazy loader passes to Rust.
    stream = graph.search_batches(**options(), batch_size=1)
    nodes = pl.DataFrame({"id": [0, 1, 2, 3]}, schema={"id": pl.UInt64})
    edges = pl.DataFrame({"id": [0, 1, 2, 3, 4]}, schema={"id": pl.UInt64})
    with pytest.raises(RuntimeError, match="stream is open"):
        graph._inner.set_payloads(nodes, edges)
    stream.close()
    graph._inner.set_payloads(nodes, edges)


def test_close_releases_label_mapping():
    import weakref

    class Label:
        pass

    label = Label()
    reference = weakref.ref(label)
    graph = rxg.Graph.from_edges([(label, "target", {"cost": 1})])
    stream = graph.search_batches(
        start_nodes=[label],
        kernel="weighted_budget",
        params={"weight_col": "cost", "budget": 2, "target": "target"},
    )
    del label, graph
    assert reference() is not None
    stream.close()
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_payload_errors_happen_on_pull_and_terminate(tmp_path, mode):
    nodes = pl.DataFrame({"id": [0, 1, 2]}, schema={"id": pl.UInt64})
    edges = pl.DataFrame(
        {"id": [0, 1], "src": [0, 1], "dest": [2, 2], "cost": [1, -1]},
        schema_overrides={name: pl.UInt64 for name in ("id", "src", "dest")},
    )
    nodes.write_parquet(tmp_path / "nodes.parquet")
    edges.write_parquet(tmp_path / "edges.parquet")
    graph = rxg.Graph.from_parquet(
        tmp_path / "nodes.parquet", tmp_path / "edges.parquet", payloads=mode
    )
    kwargs = dict(
        start_nodes=[0, 1],
        kernel="weighted_budget",
        params={"weight_col": "cost", "budget": 10, "target": 2},
        strategy="dfs",
        batch_size=1,
    )
    unopened = graph.search_batches(**kwargs)
    unopened.close()
    stream = graph.search_batches(**kwargs)
    first = next(stream) if mode == "lazy" else None
    with pytest.raises(RuntimeError, match="negative"):
        next(stream)
    assert list(stream) == []
    if first is not None:
        assert first[0].nodes == [0, 2]
        assert stream.stats.stopped_paths == 1


def test_lazy_files_are_opened_on_first_pull(tmp_path):
    nodes = tmp_path / "nodes.parquet"
    edges = tmp_path / "edges.parquet"
    pl.DataFrame({"id": [0, 1]}, schema={"id": pl.UInt64}).write_parquet(nodes)
    pl.DataFrame(
        {"id": [0], "src": [0], "dest": [1], "cost": [1]},
        schema={name: pl.UInt64 for name in ("id", "src", "dest", "cost")},
    ).write_parquet(edges)
    graph = rxg.Graph.from_parquet(nodes, edges, payloads="lazy")
    edges.unlink()
    stream = graph.search_batches(
        start_nodes=[0],
        kernel="weighted_budget",
        params={"weight_col": "cost", "budget": 10, "target": 1},
    )
    assert stream.stats.lazy_payload_read_calls == 0
    with pytest.raises(RuntimeError, match="failed to open"):
        next(stream)
    assert list(stream) == []
