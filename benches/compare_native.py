"""Interleave baseline/candidate suites and report the two explicit acceptance gates.

Requires a preserved baseline binary and baseline Python package, plus release
candidate builds. Outputs raw samples and a machine-readable gate report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import tempfile
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import rxgraph

from benches.measure import TimedCase, as_json, measure_cases
from benches.native_acceptance import CASES, normalize, tables


def load_baseline(path):
    spec = importlib.util.spec_from_file_location(
        "rxgraph_baseline", path / "__init__.py", submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_plugin(name, extension, api):
    module = ModuleType(name)
    sys.modules[name] = module
    spec = importlib.util.spec_from_file_location(name + "._native", extension)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    sys.modules[api.__name__ + ".plugin"].export_api(module.__dict__, native)
    return module


def merge(reports):
    merged = {}
    for key in reports[0]:
        samples = [
            sample for report in reports for sample in report[key]["samples_seconds"]
        ]
        merged[key] = dict(
            reports[-1][key],
            samples_seconds=samples,
            median_seconds=statistics.median(samples),
            p90_seconds=sorted(samples)[(len(samples) * 9 + 9) // 10 - 1],
        )
        if "first_search_seconds" in reports[0][key]:
            firsts = [report[key]["first_search_seconds"] for report in reports]
            merged[key]["first_search_samples_seconds"] = firsts
            merged[key]["first_search_median_seconds"] = statistics.median(firsts)
    return merged


def python_comparison(
    baseline, directory, runs, candidate=rxgraph, compound=False, parallel=False
):
    output = {}
    cases_to_run = (
        CASES + (("large_state", 32, 32, 1, None, False),) if compound else CASES
    )
    for case in cases_to_run:
        nodes, edges, kwargs = tables(case)
        nodes_path, edges_path = (
            directory / "nodes.parquet",
            directory / "edges.parquet",
        )
        nodes.write_parquet(nodes_path)
        edges.write_parquet(edges_path)
        for mode in ("arrow",) if parallel else ("arrow", "eager", "lazy"):
            for strategy in ("dfs", "bfs"):
                run = dict(kwargs, strategy=strategy, parallel=parallel)
                if compound:
                    run["kernel"] = "bench_arrow" if mode == "arrow" else "bench_typed"
                    run["params"] = dict(run["params"], large=case[0] == "large_state")

                def build(module):
                    if mode == "arrow":
                        return module.Graph(nodes, edges)
                    return module.Graph.from_parquet(
                        nodes_path, edges_path, payloads=mode
                    )

                graphs = {"baseline": build(baseline), "candidate": build(candidate)}
                first = {"baseline": [], "candidate": []}
                cold_expected = None
                # Fresh typed caches for every first-search sample. Building is untimed.
                for repeat in range(runs):
                    for name, module in (
                        ("baseline", baseline),
                        ("candidate", candidate),
                    )[:: 1 if repeat % 2 == 0 else -1]:
                        graph = build(module)
                        start = time.perf_counter()
                        result = graph.search(**run)
                        first[name].append(time.perf_counter() - start)
                        cold_result = normalize(result.paths)
                        if cold_expected is None:
                            cold_expected = cold_result
                        assert cold_result == cold_expected
                        del result

                cases = [
                    TimedCase(
                        name,
                        lambda graph=graph: graph.search(**run),
                        lambda result: normalize(result.paths),
                        execution="requested-parallel" if parallel else "serial",
                        stats=lambda result: (
                            result.stats.evaluated_edges,
                            result.stats.parallel_edges,
                        ),
                    )
                    for name, graph in graphs.items()
                ]
                first_batches = {size: [] for size in (1, 64, 1024, 4096)}

                def drain(size):
                    start = time.perf_counter()
                    first_seconds = None
                    paths = []
                    with graphs["candidate"].search_batches(
                        **run, batch_size=size
                    ) as stream:
                        for batch in stream:
                            if first_seconds is None:
                                first_seconds = time.perf_counter() - start
                            paths.extend(batch)
                        stats = stream.stats
                    first_batches[size].append(first_seconds)
                    return SimpleNamespace(paths=paths, stats=stats)

                for size in first_batches:
                    cases.append(
                        TimedCase(
                            f"batch-{size}",
                            lambda size=size: drain(size),
                            lambda result: normalize(result.paths),
                            execution="requested-parallel" if parallel else "serial",
                            stats=lambda result: (
                                result.stats.evaluated_edges,
                                result.stats.parallel_edges,
                            ),
                        )
                    )
                measured = measure_cases(cases, warmups=2, runs=runs)
                key = f"{case[0]}/{mode}/{strategy}" + ("/parallel" if parallel else "")
                output[key] = {row.name: as_json(row) for row in measured}
                for name in first:
                    output[key][name]["first_search_samples_seconds"] = first[name]
                    output[key][name]["first_search_median_seconds"] = (
                        statistics.median(first[name])
                    )
                for size, samples in first_batches.items():
                    output[key][f"batch-{size}"]["first_batch_seconds"] = samples[
                        -runs:
                    ]
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--stream-binary", type=Path, required=True)
    parser.add_argument("--baseline-package", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("dist/native-acceptance/final")
    )
    parser.add_argument("--baseline-plugin", type=Path, required=True)
    parser.add_argument("--candidate-plugin", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit unsuccessfully if either performance gate misses",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {"baseline": [], "candidate": [], "stream": []}
    memory = {}
    for repeat in range(args.rounds):
        order = list(reports)
        if repeat % 2:
            order.reverse()
        for name in order:
            executable = getattr(args, f"{name}_binary")
            with tempfile.TemporaryFile() as output:
                process = subprocess.Popen(
                    [str(executable.resolve())],
                    stdout=output,
                    env=dict(os.environ, RAYON_NUM_THREADS="4"),
                )
                _, status, usage = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(status)
                if process.returncode:
                    raise subprocess.CalledProcessError(
                        process.returncode, process.args
                    )
                output.seek(0)
                payload = json.load(output)
                rss = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
                memory[f"rust-{name}-{repeat}"] = rss
            reports[name].append(payload)
            (args.output / f"rust-{name}-{repeat}.json").write_text(
                json.dumps(payload, indent=2)
            )
        print(f"Rust comparison round {repeat + 1}/{args.rounds} complete", flush=True)
    rust = {name: merge(values) for name, values in reports.items()}
    baseline = load_baseline(args.baseline_package)
    assert baseline.rayon_thread_count() == rxgraph.rayon_thread_count()
    py = python_comparison(baseline, args.output, args.runs)
    print("Python built-in cases complete", flush=True)
    baseline_plugin = load_plugin(
        "rxgraph_plugin_baseline", args.baseline_plugin, baseline
    )
    candidate_plugin = load_plugin(
        "rxgraph_plugin_candidate", args.candidate_plugin, rxgraph
    )
    assert (
        baseline_plugin.rayon_thread_count()
        == candidate_plugin.rayon_thread_count()
        == rxgraph.rayon_thread_count()
    )
    supplemental = python_comparison(
        baseline_plugin, args.output, args.runs, candidate_plugin, compound=True
    )
    print("Python serial plugin cases complete", flush=True)
    supplemental.update(
        python_comparison(
            baseline_plugin,
            args.output,
            args.runs,
            candidate_plugin,
            compound=True,
            parallel=True,
        )
    )
    py.update({"plugin/" + key: value for key, value in supplemental.items()})
    (args.output / "python.json").write_text(json.dumps(py, indent=2))
    speedups = {}
    overhead = {}
    for key, old in rust["baseline"].items():
        new = rust["candidate"][key]
        assert old["result_size"] == new["result_size"]
        speedups[f"rust/{key}"] = old["median_seconds"] / new["median_seconds"]
        speedups[f"rust-first/{key}"] = (
            old["first_search_median_seconds"] / new["first_search_median_seconds"]
        )
        overhead[f"rust/{key}"] = (
            rust["stream"][key + "/batch-1024"]["median_seconds"]
            / new["median_seconds"]
        )
    for key, values in py.items():
        speedups[f"python/{key}"] = (
            values["baseline"]["median_seconds"] / values["candidate"]["median_seconds"]
        )
        speedups[f"python-first/{key}"] = (
            values["baseline"]["first_search_median_seconds"]
            / values["candidate"]["first_search_median_seconds"]
        )
        overhead[f"python/{key}"] = (
            values["batch-1024"]["median_seconds"]
            / values["candidate"]["median_seconds"]
        )
    gate = dict(
        baseline="3df81bd",
        rust_threads=4,
        python_threads=rxgraph.rayon_thread_count(),
        speedups=speedups,
        stream_time_ratio=overhead,
        speedup_passed=all(value >= 2 for value in speedups.values()),
        streaming_passed=all(value <= 1.05 for value in overhead.values()),
    )
    memory["python-combined-process"] = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    (args.output / "peak-rss.json").write_text(json.dumps(memory, indent=2))
    (args.output / "gates.json").write_text(json.dumps(gate, indent=2))
    for group in ("rust", "rust-first", "python", "python-first"):
        values = [
            value for key, value in speedups.items() if key.startswith(group + "/")
        ]
        print(
            f"{group}: {sum(value >= 2 for value in values)}/{len(values)} cases >=2x; min/median/max {min(values):.2f}/{statistics.median(values):.2f}/{max(values):.2f}"
        )
    print(
        f"Streaming: {sum(value <= 1.05 for value in overhead.values())}/{len(overhead)} cases within 5%"
    )
    if args.check and not (gate["speedup_passed"] and gate["streaming_passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
