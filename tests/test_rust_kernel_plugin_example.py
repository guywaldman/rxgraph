import os

import pytest
import rxgraph as rxg


REQUIRE_PLUGIN = "RXGRAPH_REQUIRE_KERNEL_PLUGIN_EXAMPLE"


def test_rust_kernel_plugin_example_e2e() -> None:
    graph = rxg.Graph.from_edges(
        [("a", "b"), ("b", "c"), ("c", "d")],
        nodes=[
            ("a", {"target": False}),
            ("b", {"target": False}),
            ("c", {"target": True}),
            ("d", {"target": False}),
        ],
    )

    try:
        result = graph.search(
            start_nodes=["a"],
            kernel="hop_budget",
            params={"max_hops": 3, "target_col": "target"},
            max_paths=10,
            parallel=False,
        )
    except ValueError as exc:
        if os.environ.get(REQUIRE_PLUGIN):
            raise AssertionError("kernel plugin example was not linked") from exc
        pytest.skip("rxgraph was not built with the kernel plugin example")

    assert len(result.paths) == 1
    assert result.paths[0].nodes == ["a", "b", "c"]
    assert result.paths[0].edges == [0, 1]
    assert result.paths[0].state == {"hops": 2}
