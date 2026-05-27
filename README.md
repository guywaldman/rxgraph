# rxgraph

> [!IMPORTANT]
> 
> This project is under heavy development with a currently undetermined roadmap, so stay tuned for updates.

A WIP Python library for extremely high-performance graph traversal and graph algorithms.

```python
import rxgraph as rxg

g = rxg.Graph.from_edges(
    [
        ("a", "b", {"price": 5}),
        ("b", "c", {"price": 6}),
        ("a", "c", {"price": 100}),
    ],
    nodes=["a", "b", "c", "d"],
)

assert g.shortest_path("a", "c") == ["a", "c"]
assert g.reachable_nodes("d") == ["d"]

kernel = rxg.Kernel(
    visit=rxg.col("edge.price") < 50,
    next_state={"spent": rxg.col("state.spent") + rxg.col("edge.price")},
    stop=rxg.col("dest.id") == g.node_id("c"),
    initial_state={"spent": 0},
)
result = g.search(rxg.Traversal(kernel, ["a"], max_depth=3, max_paths=10))

assert result.paths[0].nodes == ["a", "b", "c"]
```
