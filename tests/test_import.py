def test_rxgraph_imports() -> None:
    import rxgraph as rxg

    g = rxg.Graph(node_schema=rxg.Schema(fields=[]))
    assert g is not None
