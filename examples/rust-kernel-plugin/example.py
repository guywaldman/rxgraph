import rxgraph_hop_budget as rxg  # our extension; rxg is the rxgraph wrapper bound to our native backend

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
assert result.paths[0].state == {"hops": 2}
print("OK:", result.paths[0].nodes, result.paths[0].state)
