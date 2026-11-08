"""Measure path-local possible-target refinement in fresh processes.

Build first: cargo build --release --offline -p rxgraph --example possible_targets_poc
Run: python3 -u benches/possible_targets_poc.py
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
BINARY = ROOT / "target/release/examples/possible_targets_poc"
OUTPUT = ROOT / "benches/results/possible-targets-poc-2026-11-08.json"
CASES = [(case, strategy) for case in ("refinable", "state_rejection") for strategy in ("dfs", "bfs")]
ARMS = ["baseline", "coarse_targets", "refined_eager", "refined_lazy"]
ORDERS = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2]]


def execute(case, strategy, mode, arm=0):
    env = os.environ.copy()
    env.update(RX_PT_CASE=case, RX_PT_STRATEGY=strategy, RX_PT_MODE=mode, RX_PT_ARM=str(arm))
    with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(mode="w+") as stderr:
        child = subprocess.Popen([str(BINARY)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
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
    paths = [
        "crates/rxgraph/examples/possible_targets_poc.rs",
        "crates/rxgraph/src/traversal/engine/cursor.rs",
        "crates/rxgraph/src/traversal/engine/cursor/parallel.rs",
        "crates/rxgraph/src/graph/graph.rs",
        "crates/rxgraph/src/graph/repo.rs",
    ]
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
        "arms": ARMS,
        "notes": [
            "Each measurement runs in a fresh process. Three independent samples per arm; arm order rotates.",
            "Timing is a release build without transition counters, with at least two seconds in each measured window.",
            "RSS is process-wide Darwin wait4 peak in bytes and includes graph construction; it is a high-water mark, not incremental search heap.",
            "Validation is a separate run comparing exact path nodes, edges and final states among all arms.",
            "Per-transition target bounds are conservative possible ending nodes, maintained separately for each path.",
            "The refinable case uses state-free pre-transition edge/depth refinement; state-dependent rejection cannot be refined before the expensive transition.",
            "Deferred replay calls the same synthetic transition helper as the eager kernel; this is not a generic public Kernel replay API.",
            "The reverse index is reused warm in timing runs, while the coarse reachability mask is rebuilt per search. Validation records cold and warm preparation.",
        ],
        "cases": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for case, strategy in CASES:
        validation = execute(case, strategy, "validate")
        assert validation["exact_match"], validation
        rss = [[] for _ in ARMS]
        timing = [[] for _ in ARMS]
        for order in ORDERS:
            for arm in order:
                result = execute(case, strategy, "rss", arm)
                assert result["arm"] == arm
                rss[arm].append(result)
        for order in ORDERS:
            for arm in order:
                result = execute(case, strategy, "timing", arm)
                assert result["arm"] == arm
                assert result["window_seconds"] >= 2.0
                timing[arm].append(result)
        row = {
            "case": case,
            "strategy": strategy,
            "validation": validation,
            "rss_runs": rss,
            "timing_runs": timing,
            "median_peak_rss_bytes": [statistics.median(r["process_peak_rss_bytes"] for r in samples) for samples in rss],
            "median_seconds": [statistics.median(r["seconds_per_search"] for r in samples) for samples in timing],
        }
        report["cases"].append(row)
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{case} {strategy}: seconds {row['median_seconds']}; RSS MiB {[round(v/2**20, 1) for v in row['median_peak_rss_bytes']]}", flush=True)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
