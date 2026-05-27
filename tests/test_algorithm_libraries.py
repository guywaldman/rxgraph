from collections import defaultdict

from benches.scenarios import algorithm_cases, build_graph_data, build_library_graphs


def test_algorithm_cases_match_networkx_and_igraph() -> None:
    data = build_graph_data(64, 4)
    cases = algorithm_cases(data, build_library_graphs(data))

    by_name = defaultdict(dict)
    for case in cases:
        by_name[case.name][case.library] = case.normalize(case.run())

    for name, results in by_name.items():
        assert "rxgraph" in results
        for library, result in results.items():
            assert result == results["rxgraph"], f"{name} mismatch for {library}"
