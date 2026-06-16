import os
import tempfile
from pathlib import Path

import polars as pl
import pytest

REQUIRE_PLUGIN = "RXGRAPH_REQUIRE_KERNEL_PLUGIN_EXAMPLE"


def _import_plugin():
    try:
        import rxgraph_hop_budget as rxg
    except (ModuleNotFoundError, ImportError) as exc:
        if os.environ.get(REQUIRE_PLUGIN):
            raise AssertionError(
                "example extension rxgraph_hop_budget was not built"
            ) from exc
        pytest.skip("example extension rxgraph_hop_budget not built")
    return rxg


def _assert_hop_budget_search(rxg) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nodes = tmp / "nodes.parquet"
        edges = tmp / "edges.parquet"

        pl.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "profile": [
                    {"target": False},
                    {"target": False},
                    {"target": True},
                    {"target": False},
                ],
            }
        ).write_parquet(nodes)
        pl.DataFrame(
            {
                "id": ["ab", "bc", "cd"],
                "src": ["a", "b", "c"],
                "dest": ["b", "c", "d"],
                "policy": [
                    {"enabled": True, "hop_costs": [1]},
                    {"enabled": True, "hop_costs": [1, 1]},
                    {"enabled": True, "hop_costs": [1]},
                ],
            }
        ).write_parquet(edges)

        graph = rxg.Graph.from_parquet(nodes, edges, payloads="lazy")
        result = graph.search(
            start_nodes=["a"],
            kernel="hop_budget",
            params={
                "max_hops": 10,
                "profile_col": "profile",
                "policy_col": "policy",
            },
            max_paths=10,
            parallel=False,
        )

    assert result.paths[0].nodes == ["a", "b", "c"]
    assert result.paths[0].edges == ["ab", "bc"]
    assert result.paths[0].state == {"hops": 3}


def test_rust_kernel_plugin_import_after_rxgraph_e2e() -> None:
    import rxgraph as base_rxg

    rxg = _import_plugin()

    assert rxg.Graph is not base_rxg.Graph
    _assert_hop_budget_search(rxg)


def test_rust_kernel_plugin_example_e2e() -> None:
    _assert_hop_budget_search(_import_plugin())
