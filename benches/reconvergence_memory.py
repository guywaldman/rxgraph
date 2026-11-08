"""Compare each memory case in three fresh processes per arm.

Build first: cargo build --release -p rxgraph --example reconvergence_memory --offline
Run: python3 benches/reconvergence_memory.py
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
BINARY = ROOT / "target/release/examples/reconvergence_memory"
OUTPUT = ROOT / "benches/results/reconvergence-memory-2026-11-08.json"
SOURCES = [
    "crates/rxgraph/examples/reconvergence_memory.rs",
    "crates/rxgraph/examples/reconvergence_bench.rs",
    "crates/rxgraph/examples/reconvergence_bench/scale.rs",
    "crates/rxgraph/examples/reconvergence_bench/heap.rs",
]
CASES = [
    "inline-stream",
    "inline-eager",
    "inline-unique-stream",
    "vec-eager",
    "arc-eager",
    "arc-stream",
    "arc-one-stream",
    "arc-bfs-stream",
    "arc-unique-eager",
]


def main():
    assert BINARY.exists(), "build the release example first"
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "platform": platform.platform(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
        "notes": [
            "Three independent processes per case/arm, one cold search per process; arm order alternates.",
            "Identical state representation within each baseline/cache pair. Heap payloads are 4096 bytes.",
            "Peak live heap records requested Rust allocation sizes, excluding fixture graph and preallocated audit bitset.",
            "Allocation volume includes cache setup, traversal, output and teardown. All search allocations must be released.",
            "RSS from os.wait4 is process-wide (Darwin bytes), including graph construction and runtime startup.",
            "Each path, edge, scalar state, buffer element and unique path identity is checked exactly without audit copies.",
            "This is an allocation-instrumented memory test; its execution times are not CPU performance results.",
        ],
        "runs": [],
        "summary": [],
    }
    for case in CASES:
        for repeat in range(3):
            for cached in [False, True] if repeat % 2 == 0 else [True, False]:
                env = os.environ.copy()
                env.update(RX_MEM_CASE=case, RX_MEM_CACHE=str(int(cached)))
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
                        raise RuntimeError(f"{case} cached={cached}: {stderr.read()}")
                    row = json.loads(stdout.read())
                row.update(
                    repeat=repeat + 1,
                    process_peak_rss_bytes=usage.ru_maxrss,
                    process_swaps=usage.ru_nswap,
                )
                assert row["cached"] == cached
                assert row["net_live_after_teardown_bytes"] == 0
                report["runs"].append(row)
                print(
                    f"{case} {repeat + 1}/3 cached={cached}: "
                    f"peak heap {row['peak_search_live_heap_bytes'] / 2**20:.2f} MiB, "
                    f"RSS {row['process_peak_rss_bytes'] / 2**20:.2f} MiB",
                    flush=True,
                )
        summary = {"case": case}
        for cached in [False, True]:
            rows = [
                r for r in report["runs"] if r["case"] == case and r["cached"] == cached
            ]
            metrics = [
                "peak_search_live_heap_bytes",
                "process_peak_rss_bytes",
                "bytes_allocated",
                "allocations",
                "live_result_after_cache_drop_bytes",
            ]
            summary["cached" if cached else "baseline"] = {
                metric: statistics.median(r[metric] for r in rows) for metric in metrics
            }
            summary["cached" if cached else "baseline"]["peak_heap_range"] = [
                min(r["peak_search_live_heap_bytes"] for r in rows),
                max(r["peak_search_live_heap_bytes"] for r in rows),
            ]
        summary["peak_heap_saving_percent"] = 100 * (
            1
            - summary["cached"]["peak_search_live_heap_bytes"]
            / summary["baseline"]["peak_search_live_heap_bytes"]
        )
        report["summary"].append(summary)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"SAVED {case}: {summary['peak_heap_saving_percent']:.1f}% peak heap saving",
            flush=True,
        )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"COMPLETE {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
