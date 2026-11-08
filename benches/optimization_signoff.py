"""Isolated native optimization benchmark; build the release example first."""

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
BINARY = ROOT / "target/release/examples/optimization_signoff"
OUTPUT = ROOT / "benches/results/optimization-signoff.json"
CASES = [
    ("selective", "dfs", (0, 1, 2, 3, 4, 5)),
    ("selective", "bfs", (0, 1, 2, 3, 4)),
    ("state_rejection", "dfs", (0, 1, 2, 4, 6)),
    ("unknown", "dfs", (0, 1, 2)),
]
ARMS = {
    0: "baseline",
    1: "refined",
    2: "refined_reverse",
    3: "lazy",
    4: "lazy_reverse",
    5: "refined_reverse_shared_memo",
    6: "lazy_reverse_shared_memo",
}


def execute(case, strategy, mode, arm=0):
    env = os.environ.copy()
    env.update(
        RX_SIGNOFF_CASE=case,
        RX_SIGNOFF_STRATEGY=strategy,
        RX_SIGNOFF_MODE=mode,
        RX_SIGNOFF_ARM=str(arm),
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
        result["peak_rss_bytes"] = (
            usage.ru_maxrss if platform.system() == "Darwin" else usage.ru_maxrss * 1024
        )
        return result


def main():
    files = [
        "crates/rxgraph/examples/optimization_signoff.rs",
        "crates/rxgraph/src/traversal/kernel.rs",
        "crates/rxgraph/src/traversal/targets.rs",
        "crates/rxgraph/src/traversal/lazy.rs",
        "crates/rxgraph/src/traversal/memo.rs",
        "crates/rxgraph/src/traversal/engine.rs",
        "crates/rxgraph/src/traversal/engine/cursor.rs",
        "crates/rxgraph/src/traversal/engine/cursor/parallel.rs",
        "crates/rxgraph/src/graph/repo.rs",
    ]
    files = [f for f in files if (ROOT / f).exists()]
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files
        },
        "notes": [
            "Three independent fresh processes per arm, rotated ordering; timing uses >=2 seconds per sample and uninstrumented transition calls.",
            "RSS is process high-water including graph construction; memory and latency measured in separate processes.",
            "Each search has cold per-search state; reverse index is warm for timing. cold_reverse_seconds measures CSR construction plus a coarse multi-source ancestor traversal; actual per-target preprocessing is included in timed reverse searches.",
            "Exact node/edge/state and serial order compared across all modes in a separate validation run.",
            "The combined memo arm uses a per-wrapper shared lock, unlike worker-local search_paths_memoized_with.",
        ],
        "cases": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for case, strategy, arms in CASES:
        validate = execute(case, strategy, "validate")
        times = {a: [] for a in arms}
        memory = {a: [] for a in arms}
        for repeat in range(3):
            order = arms if repeat % 2 == 0 else tuple(reversed(arms))
            for arm in order:
                memory[arm].append(execute(case, strategy, "rss", arm))
        for repeat in range(3):
            order = arms if repeat % 2 == 0 else tuple(reversed(arms))
            for arm in order:
                sample = execute(case, strategy, "timing", arm)
                assert sample["window_seconds"] >= 2
                times[arm].append(sample)
        row = {
            "case": case,
            "strategy": strategy,
            "validation": validate,
            "arms": {
                ARMS[a]: {
                    "median_seconds": statistics.median(
                        x["seconds_per_search"] for x in times[a]
                    ),
                    "median_peak_rss_bytes": statistics.median(
                        x["peak_rss_bytes"] for x in memory[a]
                    ),
                    "timing_runs": times[a],
                    "rss_runs": memory[a],
                }
                for a in arms
            },
        }
        report["cases"].append(row)
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{case} {strategy}: "
            + ", ".join(
                f"{name}: {v['median_seconds']:.6f}s / {v['median_peak_rss_bytes'] / 1048576:.1f} MiB"
                for name, v in row["arms"].items()
            ),
            flush=True,
        )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
