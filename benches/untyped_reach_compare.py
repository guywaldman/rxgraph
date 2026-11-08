"""Compare old and current untyped target pruning in independent processes.

First build this checkout's `untyped_reach_compare` release example. To build
its previous-version counterpart, run this script with `--prepare-baseline`.
Set BASELINE_ROOT, NEW_BINARY and OLD_BINARY if using other checkout locations.
"""

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = Path(
    os.environ.get("BASELINE_ROOT", ROOT.parent / "reach-bench-baseline")
)
NEW_BINARY = Path(os.environ.get("NEW_BINARY", "/private/tmp/reach-untyped-new"))
OLD_BINARY = Path(os.environ.get("OLD_BINARY", "/private/tmp/reach-untyped-old"))
OUTPUT = ROOT / "benches/results/untyped-reach-compare.json"
SOURCE = ROOT / "crates/rxgraph/examples/untyped_reach_compare.rs"
CASES = [(16, 4096), (80, 512)]
ARMS = ("old", "current", "indexed")


def prepare_baseline():
    s = SOURCE.read_text()
    s = s.replace(
        "    EdgeCtx, Evaluation, Graph, GraphId, Kernel, NodeId, ReachIndex, ReachSpec, RunOptions,",
        "    EdgeCtx, Evaluation, Graph, GraphId, Kernel, NodeId, RunOptions,",
    )
    s = s.replace("use serde_json::json;", "use serde_json::json;\nstruct ReachIndex;")
    start = s.index("    fn index(&self) -> Result<ReachIndex> {")
    end = s.index("\n}\n#[inline(never)]", start)
    s = (
        s[:start]
        + '    fn index(&self) -> Result<ReachIndex> { anyhow::bail!("not supported in baseline") }'
        + s[end:]
    )
    start = s.index("    fn possible_targets<'a>(")
    end = s.index("\n}\nfn search(", start)
    s = (
        s[:start]
        + """    fn possible_targets(&self, cx: &TargetCtx<'_, State>) -> Result<Option<Vec<rxgraph::OwnedGraphId>>> {
        match cx {
            TargetCtx::Start { .. } => Ok(Some(self.f.targets.iter().map(|&n| (n as u64).into()).collect())),
            TargetCtx::Edge { edge, parent, .. } => {
                let group = if edge.src() == 0 { Some((edge.dest() - 1) as usize) } else { parent.filter(|s| s.group != u32::MAX).map(|s| s.group as usize) };
                Ok(group.map(|group| vec![(self.f.targets[group] as u64).into()]))
            }
        }
    }"""
        + s[end:]
    )
    path = BASELINE_ROOT / "crates/rxgraph/examples/untyped_reach_compare.rs"
    path.write_text(s)
    print(path)


def execute(arm, groups, corridor, strategy, mode):
    env = os.environ.copy()
    env.update(
        REACH_GROUPS=str(groups),
        REACH_CORRIDOR=str(corridor),
        REACH_STRATEGY=strategy,
        REACH_MODE=mode,
        REACH_ARM="indexed" if arm == "indexed" else "current",
    )
    binary = OLD_BINARY if arm == "old" else NEW_BINARY
    with (
        tempfile.TemporaryFile(mode="w+") as stdout,
        tempfile.TemporaryFile(mode="w+") as stderr,
    ):
        child = subprocess.Popen(
            [str(binary)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr
        )
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        stdout.seek(0)
        stderr.seek(0)
        if child.returncode:
            raise RuntimeError(f"{arm} {groups} {strategy} {mode}: {stderr.read()}")
        result = json.loads(stdout.read())
        result["peak_rss_bytes"] = (
            usage.ru_maxrss if platform.system() == "Darwin" else usage.ru_maxrss * 1024
        )
        return result


def main():
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "notes": [
            "All arms use identical topology, transition computations and target semantics; previous and current no-index use per-search reverse, indexed arm reuses index and disables redundant reverse.",
            "RSS is fresh-process peak including graph and index; timings use a warmed index and incoming CSR, in fresh processes, after one warm-up search. First search excludes graph/index build; index build includes cold incoming CSR.",
            "Each terminal of a mismatched group is rejected; 16 or 80 selected paths complete. Path node, edge and state digest is compared across every arm.",
        ],
        "cases": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for groups, depth in CASES:
        for strategy in ("dfs", "bfs"):
            corridor = depth if strategy == "dfs" else {16: 256, 80: 128}[groups]
            runs = {arm: {"timing": [], "rss": [], "build": []} for arm in ARMS}
            for mode in ("build", "rss", "timing"):
                for repeat in range(3):
                    order = ARMS if repeat % 2 == 0 else ARMS[::-1]
                    for arm in order:
                        runs[arm][mode].append(
                            execute(arm, groups, corridor, strategy, mode)
                        )
            all_runs = [
                sample
                for arm in ARMS
                for mode in ("rss", "timing")
                for sample in runs[arm][mode]
            ]
            assert len({(s["paths"], s["digest"]) for s in all_runs}) == 1
            result = {
                "groups": groups,
                "corridor": corridor,
                "strategy": strategy,
                "arms": {
                    arm: {
                        "median_seconds": statistics.median(
                            s["seconds_per_search"] for s in runs[arm]["timing"]
                        ),
                        "median_first_search_seconds": statistics.median(
                            s["first_search_seconds"] for s in runs[arm]["rss"]
                        ),
                        "median_index_build_seconds": statistics.median(
                            s["index_build_seconds"] for s in runs[arm]["rss"]
                        ),
                        "median_peak_rss_bytes": statistics.median(
                            s["peak_rss_bytes"] for s in runs[arm]["rss"]
                        ),
                        "median_build_peak_rss_bytes": statistics.median(
                            s["peak_rss_bytes"] for s in runs[arm]["build"]
                        ),
                        "timing_runs": runs[arm]["timing"],
                        "rss_runs": runs[arm]["rss"],
                        "build_runs": runs[arm]["build"],
                    }
                    for arm in ARMS
                },
            }
            report["cases"].append(result)
            OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
            print(
                groups,
                strategy,
                "; ".join(
                    f"{arm}: {result['arms'][arm]['median_seconds']:.4f}s / {result['arms'][arm]['median_peak_rss_bytes'] / 1048576:.1f}MiB"
                    for arm in ARMS
                ),
                flush=True,
            )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    prepare_baseline() if "--prepare-baseline" in sys.argv else main()
