from collections import defaultdict

from benches.main import (
    INIT,
    Result,
    Scale,
    best_by_bench,
    fmt_count,
    plain_speedup,
    simple_graphs,
    simple_cases,
    simple_data,
    travel_cases,
    travel_data,
    visit,
)


def test_algorithm_cases_match_networkx_and_igraph() -> None:
    data = simple_data(64, 4)
    cases = simple_cases(data)

    by_name = defaultdict(dict)
    for case in cases:
        by_name[case.alg][case.lib] = case.norm(case.run())

    for name, results in by_name.items():
        assert "rxgraph-df" in results
        assert "rxgraph-df-string-ids" in results
        assert "rxgraph-python" in results
        assert "networkx" in results
        assert "igraph" in results
        assert results["rxgraph-python"] == results["rxgraph-df"]
        for library, result in results.items():
            assert result == results["rxgraph-df"], f"{name} mismatch for {library}"
    assert len(by_name["weak_components"]["rxgraph-df"]) > 1


def test_traversal_matches_reference_libraries() -> None:
    data = travel_data(96, 8)
    cases = travel_cases(data, max_paths=8)
    by_lib = {case.lib: case.run() for case in cases}

    for library in [
        "rxgraph-df",
        "rxgraph-df-string-ids",
        "rxgraph-python",
        "networkx",
        "igraph",
    ]:
        assert library in by_lib

    reference = sorted(reference_travel_paths(data, max_paths=8))
    assert normalize_rx_paths(by_lib["rxgraph-df"]) == reference
    assert normalize_rx_paths(by_lib["rxgraph-df-string-ids"]) == reference
    assert normalize_rx_paths(by_lib["rxgraph-python"]) == reference
    assert normalize_reference_paths(by_lib["networkx"]) == reference
    assert normalize_reference_paths(by_lib["igraph"]) == reference


def test_benchmark_report_helpers_are_human_readable() -> None:
    assert set(simple_graphs(simple_data(4, 1))) >= {"rxgraph-df", "rxgraph-python"}
    assert fmt_count(5_000) == "5K"
    assert fmt_count(12_000) == "12K"
    assert plain_speedup(result("networkx", 1.04), 1.0) == "same"
    assert plain_speedup(result("rxgraph-python", 1.04), 1.0) == "same"
    assert plain_speedup(result("rxgraph-df", 1.04), 1.0) == "baseline"
    assert plain_speedup(result("networkx", 2.0), 1.0) == "2.0x slower"
    assert plain_speedup(result("networkx", 0.5), 1.0) == "2.0x faster"
    assert best_by_bench(
        [result("rxgraph-df", 1.0), result("rxgraph-python", 1.0)]
    ) == {"bfs/low": "rxgraph-df"}
    assert best_by_bench(
        [result("rxgraph-df", 1.0), result("rxgraph-python", 0.96)]
    ) == {"bfs/low": "rxgraph-python"}
    assert best_by_bench(
        [
            result("rxgraph-df", 1.0),
            result("igraph", 0.96),
            result("rxgraph-python", 0.98),
        ]
    ) == {"bfs/low": "igraph"}
    assert best_by_bench(
        [result("rxgraph-df", 1.0), result("rxgraph-python", 0.90)]
    ) == {"bfs/low": "rxgraph-python"}


def result(library: str, median: float) -> Result:
    case = next(c for c in simple_cases(simple_data(4, 1)) if c.lib == "rxgraph-df")
    return Result(
        case=case.__class__("bfs", library, case.run),
        scale=Scale("low", 4, 1),
        data=simple_data(4, 1),
        times=[median],
        size=1,
    )


def reference_travel_paths(data, max_paths: int) -> list[tuple[int, ...]]:
    nodes = {row["id"]: row for row in data.nodes.to_dicts()}
    edges = defaultdict(list)
    for row in data.edges.to_dicts():
        edges[row["src"]].append(row)

    frontier, paths = [(0, (0,), INIT)], []
    while frontier and len(paths) < max_paths:
        node, path, state = frontier.pop()
        for edge in edges[node]:
            dst = edge["dest"]
            if dst in path or not visit(nodes[dst], edge, state):
                continue
            next_state = {
                "spent": state["spent"] + edge["price"],
                "hops": state["hops"] + 1,
                "ready_at": edge["arrival"] + nodes[dst]["min_connection"],
                "risk": state["risk"] + nodes[dst]["risk"],
                "detours": state["detours"] + edge["detour_cost"],
            }
            next_path = (*path, dst)
            paths.append(next_path) if dst == data.target else frontier.append(
                (dst, next_path, next_state)
            )
            if len(paths) >= max_paths:
                break
    return paths


def normalize_rx_paths(paths) -> list[tuple[int, ...]]:
    return sorted(tuple(int(node) for node in path.nodes) for path in paths)


def normalize_reference_paths(paths) -> list[tuple[int, ...]]:
    return sorted(tuple(int(node) for node in path) for path in paths)
