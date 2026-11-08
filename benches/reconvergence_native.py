"""Run isolated RSS samples and uninstrumented latency windows for the native cache."""

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
BINARY = ROOT / "target/release/examples/reconvergence_native"
OUTPUT = ROOT / "benches/results/reconvergence-native-2026-11-08.json"
CASES = [
    "vec-bfs",
    "vec-bfs-parallel",
    "vec-dfs",
    "vec-eager",
    "cpu-dfs",
]


def execute(case, mode, arm=0):
    env = os.environ.copy()
    env.update(RX_NATIVE_CASE=case, RX_NATIVE_MODE=mode, RX_NATIVE_ARM=str(arm))
    with (
        tempfile.TemporaryFile(mode="w+") as stdout,
        tempfile.TemporaryFile(mode="w+") as stderr,
    ):
        child = subprocess.Popen(
            [str(BINARY)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr
        )
        _, status, usage = os.wait4(child.pid, 0)
        stdout.seek(0)
        stderr.seek(0)
        if os.waitstatus_to_exitcode(status):
            raise RuntimeError(f"{case} {mode} arm={arm}: {stderr.read()}")
        result = json.loads(stdout.read())
        result["process_peak_rss_bytes"] = usage.ru_maxrss
        return result


def main():
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in [
                "crates/rxgraph/examples/reconvergence_native.rs",
                "crates/rxgraph/examples/reconvergence_bench.rs",
                "crates/rxgraph/examples/reconvergence_bench/scale.rs",
                "crates/rxgraph/examples/reconvergence_bench/heap.rs",
                "crates/rxgraph/src/traversal/memo.rs",
                "crates/rxgraph/src/traversal/engine.rs",
                "crates/rxgraph/src/traversal/engine/cursor.rs",
                "crates/rxgraph/src/traversal/engine/cursor/parallel.rs",
            ]
        },
        "notes": [
            "Three isolated memory processes per arm; process-wide peak RSS from Darwin wait4 includes runtime and graph.",
            "Memory runs count kernel calls and validate every node, edge, output state and buffer.",
            "Timing uses uninstrumented isolated processes: three samples per arm, at least two seconds per sample, rotating arm order.",
            "Each search has a cold, globally bounded 4096-entry cache; caches are worker-local.",
            "Arms: baseline, early BFS state release with no cache, early release plus cache; early release has no effect for DFS.",
            "The Vec fixture uses a compact exact key because its buffer is fully determined by the scalar state.",
        ],
        "cases": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        memory = [[], [], []]
        for repeat in range(3):
            for arm in [[0, 1, 2], [2, 1, 0], [1, 2, 0]][repeat]:
                memory[arm].append(execute(case, "rss", arm))
        timings = [[], [], []]
        for repeat in range(3):
            for arm in [[0, 1, 2], [2, 1, 0], [1, 2, 0]][repeat]:
                timings[arm].append(execute(case, "timing", arm))
        row = {
            "case": case,
            "memory_runs": memory,
            "median_peak_rss_bytes": [
                statistics.median(r["process_peak_rss_bytes"] for r in arm)
                for arm in memory
            ],
            "timing_runs": timings,
            "median_seconds": [
                statistics.median(r["seconds_per_search"] for r in arm)
                for arm in timings
            ],
        }
        report["cases"].append(row)
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{case}: RSS MiB {[round(v / 2**20, 1) for v in row['median_peak_rss_bytes']]}; seconds {row['median_seconds']}; hits {[arm[0]['hits'] for arm in memory]}",
            flush=True,
        )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
