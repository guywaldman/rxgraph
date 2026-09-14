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
