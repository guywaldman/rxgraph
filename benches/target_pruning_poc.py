"""Measure the target-pruning prototype in isolated processes."""

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
BINARY = ROOT / "target/release/examples/target_pruning_poc"
OUTPUT = ROOT / "benches/results/target-pruning-poc-2026-11-08.json"
CASES = [("sparse-target", "dfs"), ("sparse-target", "bfs"), ("state-rejection", "dfs")]


def execute(case, strategy, mode, arm=0):
    env = os.environ.copy()
    env.update(
        RX_TARGET_CASE=case,
        RX_TARGET_STRATEGY=strategy,
        RX_TARGET_MODE=mode,
        RX_TARGET_ARM=str(arm),
    )
    with (
        tempfile.TemporaryFile(mode="w+") as stdout,
        tempfile.TemporaryFile(mode="w+") as stderr,
    ):
        child = subprocess.Popen(
            [str(BINARY)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr
        )
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        stdout.seek(0)
        stderr.seek(0)
        if child.returncode:
            raise RuntimeError(f"{case} {strategy} {mode} arm={arm}: {stderr.read()}")
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
                "crates/rxgraph/examples/target_pruning_poc.rs",
                "crates/rxgraph/src/traversal/engine/cursor.rs",
                "crates/rxgraph/src/graph/graph.rs",
                "crates/rxgraph/src/graph/repo.rs",
            ]
        },
        "arms": [
            "normal stateful search",
            "reverse-reachability pruning then stateful search",
            "topology search then lazy state replay",
        ],
        "notes": [
            "Each memory or timing sample runs in a fresh process; timing uses a 0.5s warmup and at least a 2s window, with no transition counters.",
            "RSS is peak process-wide Darwin wait4 RSS and includes graph construction; reverse RSS includes cold incoming-index construction.",
            "Reverse timing includes target-specific reachability computation but assumes the graph's incoming index is already built. Validation reports cold and warm setup time.",
            "The graph is a four-way branching tree, depth 10, with one sink; sparse case has eight incoming target paths, state-rejection case has 1048576 topological paths and one valid stateful path.",
            "All three strategies are compared for exact path edges, nodes, and states in a separate validation run.",
        ],
        "cases": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for case, strategy in CASES:
        validation = execute(case, strategy, "validate")
        rss = [[], [], []]
        timings = [[], [], []]
        for repeat in range(3):
            for arm in [[0, 1, 2], [2, 1, 0], [1, 2, 0]][repeat]:
                rss[arm].append(execute(case, strategy, "rss", arm))
        for repeat in range(3):
            for arm in [[0, 1, 2], [2, 1, 0], [1, 2, 0]][repeat]:
                timings[arm].append(execute(case, strategy, "timing", arm))
        row = {
            "case": case,
            "strategy": strategy,
            "validation": validation,
            "rss_runs": rss,
            "timing_runs": timings,
            "median_peak_rss_bytes": [
                statistics.median(r["process_peak_rss_bytes"] for r in arm)
                for arm in rss
            ],
            "median_seconds": [
                statistics.median(r["seconds_per_search"] for r in arm)
                for arm in timings
            ],
        }
        report["cases"].append(row)
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{case} {strategy}: seconds {row['median_seconds']}; RSS MiB {[round(v / 2**20, 1) for v in row['median_peak_rss_bytes']]}",
            flush=True,
        )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
