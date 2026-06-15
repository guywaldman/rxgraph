import tempfile
from pathlib import Path

import polars as pl
import rxgraph_hop_budget as rxg  # our extension; rxg is the rxgraph wrapper bound to our native backend

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    nodes = tmp / "nodes.parquet"
    edges = tmp / "edges.parquet"

    pl.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "target": [False, False, True, False],
        }
    ).write_parquet(nodes)
    pl.DataFrame(
        {
            "id": ["ab", "bc", "cd"],
            "src": ["a", "b", "c"],
            "dest": ["b", "c", "d"],
        }
    ).write_parquet(edges)

    graph = rxg.Graph.from_parquet(nodes, edges, payloads="lazy")
    result = graph.search(
        start_nodes=["a"],
        kernel="hop_budget",
        params={"max_hops": 3, "target_col": "target"},
        max_paths=10,
        parallel=False,
    )

    assert result.paths[0].nodes == ["a", "b", "c"]
    assert result.paths[0].state == {"hops": 2}
    assert result.stats.materialized_node_payloads < graph.node_count
    print("OK:", result.paths[0].nodes, result.paths[0].state)
