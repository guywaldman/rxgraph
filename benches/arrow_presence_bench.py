"""Benchmark one-million-row decoding with five missing optional columns.

Build `cargo build --release -p rxgraph --example arrow_presence_bench` first.
Set RXGRAPH_PRESENCE_BENCH_BINARY if the binary is outside target/release.
"""

import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(
    os.environ.get(
        "RXGRAPH_PRESENCE_BENCH_BINARY",
        ROOT / "target/release/examples/arrow_presence_bench",
    )
)
ARMS = ("probe", "has", "optional")


def main():
    samples = {arm: [] for arm in ARMS}
    for repeat in range(3):
        for arm in ARMS[repeat:] + ARMS[:repeat]:
            row = json.loads(subprocess.check_output([str(BINARY), arm], text=True))
            assert row["rows"] == 1_000_000 and row["present"] == 0
            samples[arm].append(row)
    assert len({s["checksum"] for rows in samples.values() for s in rows}) == 1
    medians = {
        arm: statistics.median(row["seconds_per_pass"] for row in rows)
        for arm, rows in samples.items()
    }
    report = {
        "platform": platform.platform(),
        "rows": 1_000_000,
        "absent_columns": 5,
        "notes": "Three fresh processes per arm; each repeatedly decodes 1M rows for at least two seconds.",
        "medians_seconds": medians,
        "relative_to_probe": {
            arm: value / medians["probe"] for arm, value in medians.items()
        },
        "samples": samples,
    }
    output = ROOT / "benches/results/arrow-presence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
