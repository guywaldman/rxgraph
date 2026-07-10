"""Shared, vectorized benchmark data generation and cache management."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

PROFILE_PATH = Path(__file__).with_name("profiles.json")
ROW_GROUP_SIZE = 262_144


@dataclass(frozen=True, slots=True)
class SearchShape:
    starts: int
    depth: int
    fanout: int

    @property
    def edge_count(self) -> int:
        return self.starts * self.depth * (self.fanout + 1)

    @property
    def chain_nodes(self) -> int:
        return self.starts * self.depth

    @property
    def decoys(self) -> int:
        return self.fanout * 16

    @property
    def node_count(self) -> int:
        return self.chain_nodes + self.decoys + 1


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    search: SearchShape
    topology_nodes: int | None
    warmups: int
    runs: int
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class PreparedData:
    node_path: Path
    edge_path: Path
    manifest_path: Path
    cache_hit: bool
    setup_seconds: float
    size_bytes: int
    spec: dict[str, Any]


def profiles() -> dict[str, Profile]:
    raw = json.loads(PROFILE_PATH.read_text())
    return {
        name: Profile(
            name=name,
            search=SearchShape(**values["search"]),
            topology_nodes=values["topology_nodes"],
            warmups=values["warmups"],
            runs=values["runs"],
        )
        for name, values in raw["profiles"].items()
    }


def selected_profiles(name: str, scale: float = 1.0) -> tuple[Profile, ...]:
    if scale <= 0:
        raise ValueError("benchmark scale must be positive")
    available = profiles()
    if name == "all":
        selected = available["standard"], available["large"]
    elif name in available:
        selected = (available[name],)
    else:
        raise ValueError(f"unknown benchmark profile {name!r}")
    return tuple(scaled_profile(profile, scale) for profile in selected)


def scaled_profile(profile: Profile, scale: float) -> Profile:
    if scale <= 0:
        raise ValueError("benchmark scale must be positive")
    if scale == 1:
        return profile
    shape = profile.search
    return Profile(
        name=profile.name,
        search=SearchShape(
            starts=max(1, round(shape.starts * scale)),
            depth=shape.depth,
            fanout=shape.fanout,
        ),
        topology_nodes=(
            max(1, round(profile.topology_nodes * scale))
            if profile.topology_nodes is not None
            else None
        ),
        warmups=profile.warmups,
        runs=profile.runs,
        scale=scale,
    )


def cache_root() -> Path:
    override = os.environ.get("RXGRAPH_BENCH_CACHE")
    return Path(override) if override else Path.cwd() / ".cache" / "bench"


def clear_cache() -> None:
    shutil.rmtree(cache_root(), ignore_errors=True)


def prepare_search(
    profile: Profile, codec: Literal["lz4", "zstd"] = "lz4"
) -> PreparedData:
    shape = profile.search
    spec = {
        "schema_version": 1,
        "kind": "stateful-search",
        "profile": profile.name,
        "scale": profile.scale,
        "starts": shape.starts,
        "depth": shape.depth,
        "fanout": shape.fanout,
        "warmups": profile.warmups,
        "runs": profile.runs,
        "codec": codec,
        "row_group_size": ROW_GROUP_SIZE,
        "columns": {
            "nodes": {"id": "u64"},
            "edges": {
                "id": "u64",
                "src": "u64",
                "dest": "u64",
                "cost": "u64",
                "allowed": "bool",
            },
        },
    }
    return _prepare(
        spec, lambda nodes, edges: _write_search(shape, codec, nodes, edges)
    )


def prepare_topology(profile: Profile) -> PreparedData | None:
    if profile.topology_nodes is None:
        return None
    spec = {
        "schema_version": 1,
        "kind": "topology",
        "profile": profile.name,
        "scale": profile.scale,
        "nodes": profile.topology_nodes,
        "codec": "lz4",
        "row_group_size": ROW_GROUP_SIZE,
        "columns": {
            "nodes": {"id": "u64"},
            "edges": {"id": "u64", "src": "u64", "dest": "u64"},
        },
    }
    return _prepare(
        spec,
        lambda nodes, edges: _write_topology(profile.topology_nodes, nodes, edges),
    )


def _prepare(spec: dict[str, Any], write: Any) -> PreparedData:
    started = time.perf_counter()
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    key = hashlib.sha256(encoded).hexdigest()[:20]
    root = cache_root()
    path = root / f"v{spec['schema_version']}" / key
    manifest = path / "manifest.json"
    nodes, edges = path / "nodes.parquet", path / "edges.parquet"
    if _valid(manifest, nodes, edges, spec):
        return PreparedData(
            nodes,
            edges,
            manifest,
            True,
            time.perf_counter() - started,
            _tree_size(path),
            spec,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=path.parent))
    try:
        tmp_nodes, tmp_edges = tmp / "nodes.parquet", tmp / "edges.parquet"
        write(tmp_nodes, tmp_edges)
        payload = {"spec": spec, "rows": _row_counts(tmp_nodes, tmp_edges)}
        (tmp / "manifest.json").write_text(json.dumps(payload, sort_keys=True))
        if _valid(manifest, nodes, edges, spec):
            shutil.rmtree(tmp)
        elif path.exists():
            stale = path.parent / f".{key}-stale-{os.getpid()}"
            path.replace(stale)
            try:
                tmp.replace(path)
            except BaseException:
                stale.replace(path)
                raise
            shutil.rmtree(stale)
        else:
            tmp.replace(path)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)

    return PreparedData(
        nodes,
        edges,
        manifest,
        False,
        time.perf_counter() - started,
        _tree_size(path),
        spec,
    )


def _valid(manifest: Path, nodes: Path, edges: Path, spec: dict[str, Any]) -> bool:
    if not all(path.is_file() for path in (manifest, nodes, edges)):
        return False
    try:
        saved = json.loads(manifest.read_text())
        if saved.get("spec") != spec:
            return False
        return _schema_matches(nodes, spec["columns"]["nodes"]) and _schema_matches(
            edges, spec["columns"]["edges"]
        )
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return False


def _schema_matches(path: Path, expected: dict[str, str]) -> bool:
    actual = {
        name: _dtype_name(dtype) for name, dtype in pl.read_parquet_schema(path).items()
    }
    return actual == expected


def _dtype_name(dtype: pl.DataType) -> str:
    return {pl.UInt64: "u64", pl.Boolean: "bool"}.get(dtype, str(dtype))


def _write_search(shape: SearchShape, codec: str, nodes: Path, edges: Path) -> None:
    pl.LazyFrame().select(
        pl.int_range(0, shape.node_count, dtype=pl.UInt64).alias("id")
    ).sink_parquet(nodes, compression=codec, row_group_size=ROW_GROUP_SIZE)

    width = shape.fanout + 1
    slots = pl.LazyFrame().select(
        pl.int_range(0, shape.edge_count, dtype=pl.UInt64).alias("id")
    )
    slots = (
        slots.with_columns(
            (pl.col("id") // (shape.depth * width)).alias("case"),
            (pl.col("id") % (shape.depth * width)).alias("within_case"),
        )
        .with_columns(
            (pl.col("within_case") // width).alias("level"),
            (pl.col("within_case") % width).alias("lane"),
        )
        .with_columns(
            (pl.col("case") * shape.depth + pl.col("level")).alias("src"),
            (pl.col("lane") == 0).alias("allowed"),
        )
        .with_columns(
            pl.when(pl.col("allowed"))
            .then(
                pl.when(pl.col("level") + 1 == shape.depth)
                .then(pl.lit(shape.node_count - 1, dtype=pl.UInt64))
                .otherwise(pl.col("src") + 1)
            )
            .otherwise(
                pl.lit(shape.chain_nodes, dtype=pl.UInt64)
                + (
                    (pl.col("case") * 131 + pl.col("level") * 67 + pl.col("lane") - 1)
                    % shape.decoys
                )
            )
            .alias("dest"),
            pl.when(pl.col("allowed"))
            .then(pl.lit(1, dtype=pl.UInt64))
            .otherwise(pl.lit(shape.depth + 1, dtype=pl.UInt64))
            .alias("cost"),
        )
        .select("id", "src", "dest", "cost", "allowed")
    )
    slots.sink_parquet(edges, compression=codec, row_group_size=ROW_GROUP_SIZE)


def _write_topology(node_count: int, nodes: Path, edges: Path) -> None:
    pl.LazyFrame().select(
        pl.int_range(0, node_count, dtype=pl.UInt64).alias("id")
    ).sink_parquet(nodes, compression="lz4", row_group_size=ROW_GROUP_SIZE)
    frames = []
    for step in range(1, 6):
        frame = (
            pl.LazyFrame()
            .select(pl.int_range(0, node_count - step, dtype=pl.UInt64).alias("src"))
            .with_columns((pl.col("src") + step).alias("dest"))
        )
        if step > 1:
            frame = frame.filter(pl.col("dest") % step == 0)
        frames.append(frame)
    pl.concat(frames).with_row_index("id").with_columns(
        pl.col("id").cast(pl.UInt64)
    ).select("id", "src", "dest").sink_parquet(
        edges, compression="lz4", row_group_size=ROW_GROUP_SIZE
    )


def _row_counts(nodes: Path, edges: Path) -> dict[str, int]:
    return {
        "nodes": int(pl.scan_parquet(nodes).select(pl.len()).collect().item()),
        "edges": int(pl.scan_parquet(edges).select(pl.len()).collect().item()),
    }


def _tree_size(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage cached rxgraph benchmark datasets."
    )
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()
    if args.clear:
        clear_cache()


if __name__ == "__main__":
    main()
