"""Measure existing native Budget kernel with optimized target traversal."""
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
BINARY = ROOT / "target/release/examples/native_targets_signoff"
OUTPUT = ROOT / "benches/results/native-targets-signoff-2026-11-08.json"
CASES = [(case, strategy) for case in ("reject", "large_state") for strategy in ("dfs", "bfs")]
ARMS = ["baseline", "incremental_target_bounds", "incremental_bounds_reverse"]
ORDERS = [[0, 1, 2], [2, 1, 0], [1, 2, 0]]


def execute(case, strategy, mode, arm=0):
    env = os.environ.copy()
    env.update(RX_NATIVE_SIGNOFF_CASE=case, RX_NATIVE_SIGNOFF_STRATEGY=strategy,
               RX_NATIVE_SIGNOFF_MODE=mode, RX_NATIVE_SIGNOFF_ARM=str(arm))
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
    files = ["crates/rxgraph/examples/native_targets_signoff.rs", "crates/rxgraph/benches/native_fixture.rs", "crates/rxgraph/src/traversal/targets.rs", "crates/rxgraph/src/traversal/kernel.rs", "crates/rxgraph/src/traversal/engine/cursor.rs", "crates/rxgraph/src/traversal/engine/cursor/parallel.rs"]
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in files},
        "arms": ARMS,
        "notes": [
            "Existing native_fixture Budget kernel with representative generated graphs, not user data.",
            "reject: 64 starts, depth 2, fanout 128, budget 0; large_state: 32 starts, depth 32, fanout 1, 256-u64 Vec state.",
            "Every arm runs in a fresh process; timing has three samples, 0.5s warmup and at least 2s per window, without counters.",
            "RSS is process-wide Darwin wait4 peak and includes graph construction; memory runs include cold reverse index.",
            "Reverse-index allocation is warm during timing, while target reachability is recomputed on each search.",
            "Exact ordered node IDs, edge IDs and all serialized state fields compared in separate validation; output streamed in batches of 1024.",
        ],
        "cases": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for case, strategy in CASES:
        validation = execute(case, strategy, "validate")
        assert validation["ordered_paths_and_state_match"]
        rss = [[] for _ in ARMS]
        timings = [[] for _ in ARMS]
        for order in ORDERS:
            for arm in order:
                rss[arm].append(execute(case, strategy, "rss", arm))
        for order in ORDERS:
            for arm in order:
                result = execute(case, strategy, "timing", arm)
                assert result["window_seconds"] >= 2.0
                timings[arm].append(result)
        entry = {"case": case, "strategy": strategy, "validation": validation, "rss_runs": rss, "timing_runs": timings,
                 "median_peak_rss_bytes": [statistics.median(r["process_peak_rss_bytes"] for r in arm) for arm in rss],
                 "median_seconds": [statistics.median(r["seconds_per_search"] for r in arm) for arm in timings]}
        report["cases"].append(entry)
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{case} {strategy}: seconds {entry['median_seconds']}; RSS MiB {[round(n/2**20, 1) for n in entry['median_peak_rss_bytes']]}", flush=True)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
