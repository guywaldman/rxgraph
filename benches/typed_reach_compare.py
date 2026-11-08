"""Compare typed search before/after reusable reachability-index support.

Build the two examples into separate target dirs; copy binaries to the paths below.
"""

import json
import os
import platform
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benches/results/typed-reach-compare.json"
OLD = "/private/tmp/reach-typed-old"
NEW = "/private/tmp/reach-typed-new"
ARMS = (
    ("old", OLD, "baseline"),
    ("new_baseline", NEW, "baseline"),
    ("new_reverse", NEW, "control"),
    ("new_indexed", NEW, "indexed"),
    ("new_rebuild", NEW, "cold"),
)


def run(groups, length, strategy, name, binary, arm, mode):
    env = os.environ.copy()
    env.update(
        REACH_GROUPS=str(groups),
        REACH_LEN=str(length),
        REACH_STRATEGY=strategy,
        REACH_ARM=arm,
        REACH_MODE=mode,
    )
    with (
        tempfile.TemporaryFile(mode="w+") as stdout,
        tempfile.TemporaryFile(mode="w+") as stderr,
    ):
        child = subprocess.Popen(
            [binary], cwd=ROOT, env=env, stdout=stdout, stderr=stderr
        )
        _, status, usage = os.wait4(child.pid, 0)
        code = os.waitstatus_to_exitcode(status)
        stdout.seek(0)
        stderr.seek(0)
        if code:
            raise RuntimeError(
                f"{name} {strategy} {mode}: exit {code}; {stderr.read()}"
            )
        row = json.load(stdout)
        row["peak_rss_bytes"] = (
            usage.ru_maxrss if platform.system() == "Darwin" else usage.ru_maxrss * 1024
        )
        return row


def main():
    repeats = int(os.environ.get("REACH_REPEATS", "3"))
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "notes": [
            "TypedKernel through cached eager runner; previous revision lacks typed target optimizations.",
            "Same graph, transitions, and exact paths; indexed query excludes one-time index construction.",
            "Rebuild arm includes index creation per query after graph reverse adjacency is warm.",
            "RSS is process-wide high-water and includes graph construction and the index.",
        ],
        "cases": [],
    }
    if os.environ.get("REACH_RESUME") == "1" and OUTPUT.exists():
        report = json.loads(OUTPUT.read_text())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for groups, dfs_length, bfs_length in ((16, 4096, 256), (80, 512, 128)):
        for strategy, length in (("dfs", dfs_length), ("bfs", bfs_length)):
            if any(
                c["groups"] == groups
                and c["strategy"] == strategy
                and c["corridor_length"] == length
                for c in report["cases"]
            ):
                continue
            rows = {name: {"verify": [], "timing": []} for name, _, _ in ARMS}
            for repeat in range(repeats):
                sequence = ARMS[repeat % len(ARMS) :] + ARMS[: repeat % len(ARMS)]
                for name, binary, arm in sequence:
                    rows[name]["verify"].append(
                        run(groups, length, strategy, name, binary, arm, "verify")
                    )
            expected = rows["old"]["verify"][0]["digest"]
            assert all(
                row["digest"] == expected and row["paths"] == groups
                for data in rows.values()
                for row in data["verify"]
            )
            for repeat in range(repeats):
                sequence = ARMS[repeat % len(ARMS) :] + ARMS[: repeat % len(ARMS)]
                for name, binary, arm in sequence:
                    rows[name]["timing"].append(
                        run(groups, length, strategy, name, binary, arm, "timing")
                    )
            index_runs = [
                run(groups, length, strategy, "index", NEW, "indexed", "index")
                for _ in range(repeats)
            ]
            graph_runs = [
                run(groups, length, strategy, "graph", NEW, "baseline", "index")
                for _ in range(repeats)
            ]
            summary = {
                name: {
                    "median_ms": statistics.median(
                        r["seconds_per_search"] for r in data["timing"]
                    )
                    * 1000,
                    "median_peak_mib": statistics.median(
                        r["peak_rss_bytes"] for r in data["verify"]
                    )
                    / 1048576,
                    **data,
                }
                for name, data in rows.items()
            }
            report["cases"].append(
                {
                    "groups": groups,
                    "corridor_length": length,
                    "strategy": strategy,
                    "index_build_ms": statistics.median(
                        r["index_build_ms"] for r in index_runs
                    ),
                    "index_runs": index_runs,
                    "graph_runs": graph_runs,
                    "arms": summary,
                }
            )
            OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
            print(
                groups,
                strategy,
                {
                    name: (round(v["median_ms"], 2), round(v["median_peak_mib"], 1))
                    for name, v in summary.items()
                },
                flush=True,
            )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
