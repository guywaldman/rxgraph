"""Measure Vec/Arc memory and latency using identical state/graph fixtures.

Build first:
  cargo build --release -p rxgraph --example reconvergence_memory --example reconvergence_arc_latency --offline
Run: python3 benches/reconvergence_arc.py
"""

import hashlib
import json
import os
import platform
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / "target/release/examples/reconvergence_memory"
LATENCY = ROOT / "target/release/examples/reconvergence_arc_latency"
OUTPUT = ROOT / "benches/results/reconvergence-arc-2026-11-08.json"
SOURCES = [
    "crates/rxgraph/examples/reconvergence_memory.rs",
    "crates/rxgraph/examples/reconvergence_arc_latency.rs",
    "crates/rxgraph/examples/reconvergence_bench.rs",
    "crates/rxgraph/examples/reconvergence_bench/scale.rs",
    "crates/rxgraph/examples/reconvergence_bench/heap.rs",
    "benches/reconvergence_arc.py",
]
CASES = [
    {
        "name": "eager",
        "suffix": "eager",
        "words": 512,
        "parallel": False,
        "clone_key": True,
    },
    {
        "name": "stream",
        "suffix": "stream",
        "words": 512,
        "parallel": False,
        "clone_key": True,
    },
    {
        "name": "bfs-stream",
        "suffix": "bfs-stream",
        "words": 512,
        "parallel": False,
        "clone_key": False,
    },
    {
        "name": "unique-eager",
        "suffix": "unique-eager",
        "words": 512,
        "parallel": False,
        "clone_key": False,
    },
    {
        "name": "small-stream",
        "suffix": "stream",
        "words": 8,
        "parallel": False,
        "clone_key": False,
    },
    {
        "name": "parallel-bfs-stream",
        "suffix": "bfs-stream",
        "words": 512,
        "parallel": True,
        "clone_key": False,
    },
]
MODES = {"none": "0", "borrowed-key": "2", "clone-key": "1"}


def child_memory(case, arm):
    storage, mode = arm.split("-", 1)
    env = os.environ.copy()
    env.update(
        RX_MEM_CASE=f"{storage}-{case['suffix']}",
        RX_MEM_CACHE=MODES[mode],
        RX_MEM_WORDS=str(case["words"]),
        RX_MEM_PARALLEL=str(int(case["parallel"])),
    )
    with (
        tempfile.TemporaryFile(mode="w+") as stdout,
        tempfile.TemporaryFile(mode="w+") as stderr,
    ):
        child = subprocess.Popen(
            [str(MEMORY)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr
        )
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        stdout.seek(0)
        stderr.seek(0)
        if child.returncode:
            raise RuntimeError(f"memory {case['name']}/{arm}: {stderr.read()}")
        row = json.loads(stdout.read())
    row.update(process_peak_rss_bytes=usage.ru_maxrss, process_swaps=usage.ru_nswap)
    assert row["exact_output_validation"] and row["net_live_after_teardown_bytes"] == 0
    assert row["cache_mode"] == mode and row["buffer_bytes"] == case["words"] * 8
    if case["parallel"]:
        assert row["parallel_edges"] > 0
    return row


def child_latency(case):
    env = os.environ.copy()
    env.update(
        RX_ARC_LAT_CASE=case["name"], RX_ARC_LAT_SAMPLES="5", RX_ARC_LAT_SECONDS="2"
    )
    with tempfile.TemporaryFile(mode="w+") as stdout:
        child = subprocess.Popen(
            [str(LATENCY)],
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=True,
        )
        lines = []
        for line in child.stderr:
            lines.append(line)
            print(line.rstrip(), flush=True)
        _, status, _ = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        if child.returncode:
            raise RuntimeError(f"latency {case['name']}: {''.join(lines)}")
        stdout.seek(0)
        result = json.loads(stdout.read())
    assert result["case"] == case["name"] and result["samples"] == 5
    for row in result["arms"]:
        assert len(row["samples_seconds"]) == 5
        assert min(row["measurement_window_seconds"]) >= 2.0
    return result


def main():
    assert MEMORY.exists() and LATENCY.exists(), "build both release examples first"
    current_source_sha256 = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES
    }
    current_binary_sha256 = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (MEMORY, LATENCY)
    }
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "platform": platform.platform(),
        "source_sha256": current_source_sha256,
        "binary_sha256": current_binary_sha256,
        "notes": [
            "Synthetic deterministic kernels; identical graph, state values, and output semantics for all arms within a case.",
            "Vec and Arc<Vec> use the same heap-state kernel. The cache uses full state equality and is scoped to one search.",
            "clone-key uses one map keyed by (edge, owned state); borrowed-key uses a per-edge map and borrows state on hits.",
            "Latency uses the default allocator, no kernel counters, and no output validation during timing; graph construction excluded.",
            "Latency windows are at least 2 seconds, five samples per arm, with rotating arm order and a fresh cache per search.",
            "Memory runs use an instrumented allocator and counters, three fresh processes per arm; all paths, edges, state and payload values are checked exactly.",
            "Peak Rust live heap excludes the prebuilt graph and audit bitset. Peak RSS is whole-process ru_maxrss on macOS.",
            "Parallel cases use four Rayon workers and per-physical-worker caches; instrumentation can affect scheduling in memory runs.",
        ],
        "cases": [],
    }
    source_revision = "initial"
    if os.environ.get("RX_ARC_RESUME") == "1":
        report = json.loads(OUTPUT.read_text())
        assert "finished_utc" not in report, "report is already complete"
        for old in report["cases"]:
            old["source_revision"] = "initial"
            assert all(len(rows) == 3 for rows in old["memory_runs"].values())
            assert all(
                len(row["samples_seconds"]) == 5
                and min(row["measurement_window_seconds"]) >= 2.0
                for row in old["latency"]["arms"]
            )
        report["resumed_source_sha256"] = current_source_sha256
        report["resumed_binary_sha256"] = current_binary_sha256
        report["notes"].append(
            "The remaining parallel case was resumed after priming worker-local cache initialization; earlier timing arms already included warmup."
        )
        source_revision = "resumed"
    completed = {old["spec"]["name"] for old in report["cases"]}
    for case in CASES:
        if case["name"] in completed:
            continue
        arms = [
            f"{storage}-{mode}"
            for storage in ("vec", "arc")
            for mode in ("none", "borrowed-key")
        ]
        if case["clone_key"]:
            arms.extend(["vec-clone-key", "arc-clone-key"])
        memory = {arm: [] for arm in arms}
        for repeat in range(3):
            order = arms[repeat:] + arms[:repeat]
            if repeat % 2:
                order.reverse()
            for arm in order:
                row = child_memory(case, arm)
                memory[arm].append(row)
                print(
                    f"{case['name']} memory {repeat + 1}/3 {arm}: peak RSS {row['process_peak_rss_bytes'] / 2**20:.1f} MiB",
                    flush=True,
                )
        memory_summary = {}
        for arm, rows in memory.items():
            metrics = (
                "peak_search_live_heap_bytes",
                "process_peak_rss_bytes",
                "bytes_allocated",
                "computed_transitions",
                "cache_entries",
            )
            memory_summary[arm] = {
                m: statistics.median(row[m] for row in rows) for m in metrics
            }
        latency = child_latency(case)
        assert {row["arm"] for row in latency["arms"]} == set(arms)
        for row in latency["arms"]:
            samples = memory[row["arm"]]
            assert all(
                m["paths"] == latency["paths"]
                and m["evaluated_edges"] == row["evaluated_edges"]
                for m in samples
            )
        report["cases"].append(
            {
                "spec": case,
                "memory_runs": memory,
                "memory_median": memory_summary,
                "latency": latency,
                "source_revision": source_revision,
            }
        )
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
        print(f"SAVED {case['name']}", flush=True)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"COMPLETE {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
