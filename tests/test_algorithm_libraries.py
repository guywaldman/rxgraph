from collections import defaultdict

from benches.main import (
    Result,
    Scale,
    best_by_bench,
    fmt_count,
    plain_speedup,
    simple_graphs,
    simple_cases,
    simple_data,
)


def test_algorithm_cases_match_networkx_and_igraph() -> None:
    data = simple_data(64, 4)
    cases = simple_cases(data)

    by_name = defaultdict(dict)
    for case in cases:
        by_name[case.alg][case.lib] = case.norm(case.run())

    for name, results in by_name.items():
        assert "rxgraph-df" in results
        assert "rxgraph-python" in results
        assert results["rxgraph-python"] == results["rxgraph-df"]
        for library, result in results.items():
            assert result == results["rxgraph-df"], f"{name} mismatch for {library}"
    assert len(by_name["weak_components"]["rxgraph-df"]) > 1


def test_benchmark_report_helpers_are_human_readable() -> None:
    assert set(simple_graphs(simple_data(4, 1))) >= {"rxgraph-df", "rxgraph-python"}
    assert fmt_count(5_000) == "5K"
    assert fmt_count(12_000) == "12K"
    assert plain_speedup(result("networkx", 1.04), 1.0) == "same"
    assert plain_speedup(result("rxgraph-python", 1.04), 1.0) == "same"
    assert plain_speedup(result("rxgraph-df", 1.04), 1.0) == "baseline"
    assert plain_speedup(result("networkx", 2.0), 1.0) == "2.0x slower"
    assert plain_speedup(result("networkx", 0.5), 1.0) == "2.0x faster"
    assert best_by_bench([result("rxgraph-df", 1.0), result("rxgraph-python", 1.0)]) == {"bfs/low": "rxgraph-df"}
    assert best_by_bench([result("rxgraph-df", 1.0), result("rxgraph-python", 0.96)]) == {"bfs/low": "rxgraph-python"}
    assert best_by_bench([result("rxgraph-df", 1.0), result("igraph", 0.96), result("rxgraph-python", 0.98)]) == {"bfs/low": "igraph"}
    assert best_by_bench([result("rxgraph-df", 1.0), result("rxgraph-python", 0.90)]) == {"bfs/low": "rxgraph-python"}


def result(library: str, median: float) -> Result:
    case = next(c for c in simple_cases(simple_data(4, 1)) if c.lib == "rxgraph-df")
    return Result(
        case=case.__class__("bfs", library, case.run),
        scale=Scale("low", 4, 1),
        data=simple_data(4, 1),
        times=[median],
        size=1,
    )
