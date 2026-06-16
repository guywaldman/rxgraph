import polars as pl
import rxgraph as rxg


def _write_tables(tmp_path):
    nodes = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "unused": ["x"] * 5,
        },
        schema={"id": pl.UInt64, "unused": pl.String},
    )
    edges = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "src_id": [0, 1, 0, 4],
            "dest_id": [1, 2, 2, 3],
            "cost": [1, 1, 5, 1],
        },
        schema={
            "id": pl.UInt64,
            "src_id": pl.UInt64,
            "dest_id": pl.UInt64,
            "cost": pl.UInt64,
        },
    )
    node_path = tmp_path / "nodes.parquet"
    edge_path = tmp_path / "edges.parquet"
    nodes.write_parquet(node_path)
    edges.write_parquet(edge_path)
    return node_path, edge_path


def _search(graph):
    return graph.search(
        start_nodes=[0],
        kernel="weighted_budget",
        params={"weight_col": "cost", "budget": 2, "target": 2},
        max_paths=5,
        strategy="bfs",
        parallel=False,
    )


def test_from_parquet_eager_is_default(tmp_path) -> None:
    nodes, edges = _write_tables(tmp_path)
    explicit = rxg.Graph.from_parquet(nodes, edges, payloads="eager")
    default = rxg.Graph.from_parquet(nodes, edges)

    assert [p.nodes for p in _search(default).paths] == [[0, 1, 2]]
    assert [p.nodes for p in _search(default).paths] == [
        p.nodes for p in _search(explicit).paths
    ]


def test_from_parquet_lazy_matches_eager_and_decodes_touched_rows(tmp_path) -> None:
    nodes, edges = _write_tables(tmp_path)
    eager = rxg.Graph.from_parquet(nodes, edges, payloads="eager")
    lazy = rxg.Graph.from_parquet(nodes, edges, payloads="lazy")

    eager_result = _search(eager)
    lazy_result = _search(lazy)

    assert [p.nodes for p in lazy_result.paths] == [p.nodes for p in eager_result.paths]
    assert lazy_result.paths[0].state == {"spent": 2}
    assert lazy_result.stats.materialized_node_payloads < lazy.node_count
    assert lazy_result.stats.materialized_edge_payloads < lazy.edge_count
    assert lazy_result.stats.lazy_payload_read_calls > 0
    assert (
        lazy_result.stats.lazy_payload_requested_rows
        >= lazy_result.stats.materialized_edge_payloads
    )
    assert (
        lazy_result.stats.lazy_payload_selected_rows
        >= lazy_result.stats.lazy_payload_requested_rows
    )
