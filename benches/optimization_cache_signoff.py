"""Focused cache and BFS state-release measurements against native search."""
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
OUTPUT = ROOT / "benches/results/optimization-cache-signoff.json"
CASES = ["cpu-dfs", "vec-unique", "vec-bfs", "vec-bfs-parallel"]
ARMS = ("baseline", "release_only", "bounded_cache")

def execute(case, mode, arm):
    env = os.environ.copy()
    env.update(RX_NATIVE_CASE=case, RX_NATIVE_MODE=mode, RX_NATIVE_ARM=str(arm))
    with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(mode="w+") as stderr:
        child = subprocess.Popen([str(BINARY)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        stdout.seek(0); stderr.seek(0)
        if child.returncode: raise RuntimeError(f"{case} {mode} {arm}: {stderr.read()}")
        result = json.loads(stdout.read())
        result["peak_rss_bytes"] = usage.ru_maxrss if platform.system() == "Darwin" else usage.ru_maxrss * 1024
        return result

def main():
    files=["crates/rxgraph/examples/reconvergence_native.rs", "crates/rxgraph/examples/reconvergence_bench.rs", "crates/rxgraph/examples/reconvergence_bench/scale.rs", "crates/rxgraph/examples/reconvergence_bench/heap.rs", "crates/rxgraph/src/traversal/memo.rs", "crates/rxgraph/src/traversal/engine/cursor.rs"]
    report={"started_utc":datetime.now(timezone.utc).isoformat(),"platform":platform.platform(),
        "rustc":subprocess.check_output(["rustc","--version"],text=True).strip(),
        "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "source_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
        "notes":["3 isolated >=2-second uninstrumented windows and 3 fresh process RSS measurements per arm; rotated order.",
                 "RSS is process-wide high-water including graph construction, not incremental heap.",
                 "Memory runs check every output path, edge, state, and buffer exactly; state cache is per-search and capped at 4096 entries.",
                 "Vec fixture uses compact exact metadata keys because buffer values are determined by the metadata and constants."],"cases":[]}
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    for case in CASES:
        times=[[],[],[]]; memory=[[],[],[]]
        for repeat in range(3):
            for arm in ((0,1,2) if repeat%2==0 else (2,1,0)): memory[arm].append(execute(case,"rss",arm))
        for repeat in range(3):
            for arm in ((0,1,2) if repeat%2==0 else (2,1,0)):
                sample=execute(case,"timing",arm); assert sample["window_seconds"]>=2; times[arm].append(sample)
        row={"case":case,"arms":{name:{"median_seconds":statistics.median(x["seconds_per_search"] for x in times[i]),
              "median_peak_rss_bytes":statistics.median(x["peak_rss_bytes"] for x in memory[i]),"timing_runs":times[i],"rss_runs":memory[i]} for i,name in enumerate(ARMS)}}
        report["cases"].append(row);OUTPUT.write_text(json.dumps(report,indent=2)+"\n")
        print(f"{case}: "+", ".join(f"{name}: {v['median_seconds']:.6f}s / {v['median_peak_rss_bytes']/1048576:.1f} MiB" for name,v in row["arms"].items()),flush=True)
    report["finished_utc"]=datetime.now(timezone.utc).isoformat();OUTPUT.write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__": main()
