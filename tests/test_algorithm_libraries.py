import json

import polars as pl
import pytest
import rxgraph as rxg

from benches.data import Profile, SearchShape, cache_root, prepare_search, profiles
from benches.main import report_payload
from benches.measure import Measurement, TimedCase, as_json, measure_cases


def test_profile_dimensions_are_exact() -> None:
    available = profiles()
    quick = available["quick"]
    standard = available["standard"]
    large = available["large"]

    assert (quick.search.starts, quick.search.depth, quick.search.fanout) == (
        128,
        12,
        16,
    )
    assert quick.search.edge_count == 26_112
    assert standard.search.edge_count == 3_194_880
    assert large.search.edge_count == 33_816_576
    assert standard.topology_nodes == 100_000
    assert large.topology_nodes is None


def test_search_cache_hits_and_generated_rows_are_deterministic(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("RXGRAPH_BENCH_CACHE", str(tmp_path / "bench"))
    quick = profiles()["quick"]

    cold = prepare_search(quick)
    warm = prepare_search(quick)

    assert not cold.cache_hit
    assert warm.cache_hit
    manifest = json.loads(warm.manifest_path.read_text())
    assert manifest["rows"] == {
        "nodes": quick.search.node_count,
        "edges": quick.search.edge_count,
    }


def test_cache_invalidates_parameters_schema_and_partial_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("RXGRAPH_BENCH_CACHE", str(tmp_path / "bench"))
    quick = profiles()["quick"]
    first = prepare_search(quick)
    changed = Profile(
        name="quick-variant",
        search=SearchShape(
            starts=quick.search.starts, depth=quick.search.depth, fanout=17
        ),
        topology_nodes=quick.topology_nodes,
        warmups=quick.warmups,
        runs=quick.runs,
    )
    assert prepare_search(changed).manifest_path != first.manifest_path

    payload = json.loads(first.manifest_path.read_text())
    payload["spec"]["schema_version"] = 999
    first.manifest_path.write_text(json.dumps(payload))
    assert not prepare_search(quick).cache_hit

    recovered = prepare_search(quick)
    recovered.edge_path.unlink()
    repaired = prepare_search(quick)
    assert not repaired.cache_hit
    assert repaired.edge_path.is_file()


def test_default_cache_root_is_relative_to_the_current_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.delenv("RXGRAPH_BENCH_CACHE", raising=False)
    monkeypatch.chdir(tmp_path)

    assert cache_root() == tmp_path / ".cache" / "bench"


def test_cache_rejects_schema_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("RXGRAPH_BENCH_CACHE", str(tmp_path / "bench"))
    prepared = prepare_search(profiles()["quick"])
    pl.DataFrame({"id": [0], "bad": [1]}).write_parquet(prepared.edge_path)

    repaired = prepare_search(profiles()["quick"])

    assert not repaired.cache_hit
    assert pl.read_parquet_schema(repaired.edge_path)["allowed"] == pl.Boolean


def test_measurement_rejects_silent_parallel_fallback() -> None:
    case = TimedCase(
        "parallel",
        lambda: ("ok", 4, 0),
        lambda value: value[0],
        execution="parallel",
        stats=lambda value: (value[1], value[2]),
    )

    with pytest.raises(RuntimeError, match="silently fell back"):
        measure_cases([case], warmups=0, runs=1)


def test_measurement_json_schema_has_fixed_baseline_speedup() -> None:
    payload = as_json(
        Measurement("rxgraph-arrow-u64", [1.0, 2.0, 3.0], result_size=42),
        baseline_seconds=2.0,
    )

    assert set(payload) == {
        "name",
        "samples_seconds",
        "median_seconds",
        "p90_seconds",
        "result_size",
        "evaluated_edges",
        "parallel_edges",
        "execution",
        "baseline_speedup",
    }
    assert payload["baseline_speedup"] == 1.0


def test_benchmark_report_json_schema_is_versioned() -> None:
    assert report_payload([{"profile": "quick"}]) == {
        "schema_version": 1,
        "profiles": [{"profile": "quick"}],
    }


def test_python_parallel_edge_stats_reflect_actual_execution() -> None:
    if rxg.rayon_thread_count() < 2:
        pytest.skip("single Rayon worker")
    starts = 512
    target = starts
    graph = rxg.Graph(
        pl.DataFrame({"id": list(range(starts + 1))}, schema={"id": pl.UInt64}),
        pl.DataFrame(
            {
                "id": list(range(starts)),
                "src": list(range(starts)),
                "dest": [target] * starts,
                "allowed": [True] * starts,
            },
            schema={
                "id": pl.UInt64,
                "src": pl.UInt64,
                "dest": pl.UInt64,
                "allowed": pl.Boolean,
            },
        ),
    )
    kwargs = {
        "start_nodes": list(range(starts)),
        "visit": rxg.col("edge.allowed"),
        "stop": rxg.col("dest.id") == target,
        "max_paths": starts,
        "strategy": "bfs",
    }

    serial = graph.search(**kwargs, parallel=False)
    parallel = graph.search(**kwargs, parallel=True)

    assert serial.stats.parallel_edges == 0
    assert parallel.stats.parallel_edges == parallel.stats.evaluated_edges > 0
