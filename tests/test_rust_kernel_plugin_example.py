import os

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
    graph = rxg.Graph.from_edges(
        [("a", "b"), ("b", "c"), ("c", "d")],
        nodes=[
            ("a", {"target": False}),
            ("b", {"target": False}),
            ("c", {"target": True}),
            ("d", {"target": False}),
        ],
    )

    result = graph.search(
        start_nodes=["a"],
        kernel="hop_budget",
        params={"max_hops": 3, "target_col": "target"},
        max_paths=10,
        parallel=False,
    )

    assert result.paths[0].nodes == ["a", "b", "c"]
    assert result.paths[0].edges == [0, 1]
    assert result.paths[0].state == {"hops": 2}


def test_rust_kernel_plugin_import_after_rxgraph_e2e() -> None:
    import rxgraph as base_rxg

    rxg = _import_plugin()

    assert rxg.Graph is not base_rxg.Graph
    _assert_hop_budget_search(rxg)


def test_rust_kernel_plugin_example_e2e() -> None:
    _assert_hop_budget_search(_import_plugin())
