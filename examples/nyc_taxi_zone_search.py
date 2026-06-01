"""NYC TLC yellow-taxi zone search with list and struct kernel state.

Downloads one month of public NYC yellow-taxi trip records, builds a graph with
one raw taxi trip per edge, and searches for observed zone-to-zone trip chains.

Default data:
    https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet

Data source:
    https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
"""

from datetime import datetime

import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

import polars as pl
import rxgraph as rxg

DATA_DIR = Path(tempfile.gettempdir()) / "rxgraph" / "nyc-taxi"
TLC_BASE = "https://d37ci6vzurychx.cloudfront.net"
ZONE_LOOKUP_URL = f"{TLC_BASE}/misc/taxi_zone_lookup.csv"


def main() -> None:
    args = parse_args()

    # Load data
    zones = load_zones()
    trips = load_trips(args.month, zones, args.row_limit)
    print(f"zones={zones['id'].count()}, trips={trips['id'].count()}")
    print("-" * 40)

    graph = rxg.Graph(zones, trips)

    source_id = zone_id(zones, args.source)
    target_id = zone_id(zones, args.target)

    s = lambda name: rxg.col(f"state.{name}")
    d = lambda name: rxg.col(f"dest.{name}")
    e = lambda name: rxg.col(f"edge.{name}")

    next_zones = pl.concat_list([s("zones"), rxg.col("dest.id")])
    next_fares = pl.concat_list([s("fare_samples"), e("total_amount")])

    start = datetime.now()

    result = graph.search(
        start_nodes=[source_id],
        visit=(
            (s("hops") < args.max_depth)
            & (e("trip_distance") >= args.min_distance)
            & (e("trip_distance") <= args.max_distance)
            & (e("total_amount") <= args.max_total)
            & (~s("zones").list.contains(rxg.col("dest.id")))
            & (e("tags").list.set_intersection(s("required_tags")).list.len() > 0)
            & (
                e("tags").list.filter(pl.element().str.contains("duration:")).list.len()
                > 0
            )
        ),
        next_state={
            "zones": next_zones,
            "boroughs": pl.concat_list([s("boroughs"), d("borough")])
            .list.unique()
            .list.sort(),
            "payment_tags": s("payment_tags")
            .list.set_union(
                e("tags").list.filter(pl.element().str.contains("payment:"))
            )
            .list.sort(),
            "trip_tags": s("trip_tags").list.set_union(e("tags")).list.sort(),
            "fare_samples": next_fares,
            "fare_sum": next_fares.list.sum(),
            "fare_mean": next_fares.list.mean(),
            "trips": pl.concat_list([s("trips"), e("trip")]),
            "current_zone": d("zone_info").struct.with_fields(
                (s("hops") + 1).alias("hop"),
                e("total_amount").alias("last_total_amount"),
            ),
            "current_borough": d("zone_info").struct.field("borough"),
            "current_zone_json": d("zone_info").struct.json_encode(),
            "hops": s("hops") + 1,
        },
        stop=(rxg.col("dest.id") == target_id) & (s("hops") >= args.min_hops),
        initial_state={
            "zones": [source_id],
            "boroughs": [zone_borough(zones, source_id)],
            "required_tags": args.required_tag,
            "payment_tags": [],
            "trip_tags": [],
            "fare_samples": [],
            "fare_sum": 0.0,
            "fare_mean": 0.0,
            "trips": [],
            "hops": 0,
        },
        max_depth=args.max_depth,
        max_paths=args.max_paths,
        strategy="bfs",
        parallel=args.parallel == "on",
    )

    elapsed = datetime.now() - start
    print(f"Found {len(result.paths)} results in {elapsed.microseconds / 1000}ms")
    print(
        {
            "start_nodes": result.stats.start_nodes,
            "path_entries": result.stats.path_entries,
            "evaluated_edges": result.stats.evaluated_edges,
            "accepted_edges": result.stats.accepted_edges,
            "stopped_paths": result.stats.stopped_paths,
            "rejected_edges": result.stats.rejected_edges,
            "skipped_revisits": result.stats.skipped_revisits,
            "max_depth": result.stats.max_depth,
        }
    )
    print("-" * 40)

    print(
        f"NYC yellow taxi {args.month}; "
        f"zones={graph.node_count:,} raw_trip_edges={graph.edge_count:,}"
    )
    print(
        f"{zone_name(zones, source_id)} -> {zone_name(zones, target_id)}; "
        f"paths={len(result.paths):,} evaluated_edges={result.stats.evaluated_edges:,}"
    )
    print(f"required tags: {', '.join(args.required_tag)}")

    for index, path in enumerate(result.paths, start=1):
        state = path.state
        zones_seen = [zone_name(zones, zone) for zone in state["zones"]]
        print(f"\npath {index}:")
        print("zones:", " -> ".join(zones_seen))
        print("boroughs:", ", ".join(state["boroughs"]))
        print("payment tags:", ", ".join(state["payment_tags"]))
        print("trip tags:", ", ".join(state["trip_tags"]))
        print(f"fare sum=${state['fare_sum']:.2f} mean=${state['fare_mean']:.2f}")
        print("current zone:", state["current_zone"])
        for trip in state["trips"]:
            print(
                f"  {trip['pickup_zone']} -> {trip['dropoff_zone']} "
                f"{trip['payment']} ${trip['total_amount']:.2f} "
                f"{trip['trip_distance']:.1f}mi tags=[{', '.join(trip['tags'])}]"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default="2024-01")
    parser.add_argument("--source", default="JFK Airport")
    parser.add_argument("--target", default="Times Sq/Theatre District")
    parser.add_argument("--required-tag", action="append")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-hops", type=int, default=1)
    parser.add_argument("--max-paths", type=int, default=5)
    parser.add_argument("--min-distance", type=float, default=0.2)
    parser.add_argument("--max-distance", type=float, default=30.0)
    parser.add_argument("--max-total", type=float, default=120.0)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--parallel", choices=["on", "off"], default="on")
    args = parser.parse_args()
    args.required_tag = args.required_tag or ["payment:credit_card"]
    return args


def load_zones() -> pl.DataFrame:
    path = data("taxi_zone_lookup.csv", ZONE_LOOKUP_URL)
    return (
        pl.read_csv(path)
        .rename(
            {
                "LocationID": "id",
                "Borough": "borough",
                "Zone": "zone",
            }
        )
        .select(
            pl.col("id").cast(pl.UInt64),
            "borough",
            "zone",
            "service_zone",
        )
        .with_columns(pl.struct(["borough", "zone", "service_zone"]).alias("zone_info"))
    )


def load_trips(
    month: str,
    zones: pl.DataFrame,
    row_limit: int | None,
) -> pl.DataFrame:
    path = data(
        f"yellow_tripdata_{month}.parquet",
        f"{TLC_BASE}/trip-data/yellow_tripdata_{month}.parquet",
    )
    trips = pl.scan_parquet(path)
    if row_limit is not None:
        trips = trips.head(row_limit)

    trips = (
        trips.filter(
            pl.col("PULocationID").is_not_null()
            & pl.col("DOLocationID").is_not_null()
            & (pl.col("trip_distance") > 0)
            & (pl.col("total_amount") > 0)
        )
        .with_row_index("id")
        .with_columns(
            payment_label().alias("payment"),
            (
                (
                    pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")
                ).dt.total_seconds()
                / 60
            )
            .round(1)
            .alias("duration_min"),
            pl.when(pl.col("trip_distance") < 2)
            .then(pl.lit("distance:short"))
            .when(pl.col("trip_distance") < 8)
            .then(pl.lit("distance:medium"))
            .otherwise(pl.lit("distance:long"))
            .alias("distance_tag"),
            pl.when(pl.col("total_amount") < 20)
            .then(pl.lit("fare:low"))
            .when(pl.col("total_amount") < 60)
            .then(pl.lit("fare:medium"))
            .otherwise(pl.lit("fare:high"))
            .alias("fare_tag"),
        )
        .with_columns(
            pl.concat_str([pl.lit("payment:"), pl.col("payment")]).alias("payment_tag"),
            pl.when(pl.col("duration_min") < 15)
            .then(pl.lit("duration:quick"))
            .when(pl.col("duration_min") < 45)
            .then(pl.lit("duration:normal"))
            .otherwise(pl.lit("duration:slow"))
            .alias("duration_tag"),
        )
        .with_columns(
            pl.concat_list(
                ["payment_tag", "distance_tag", "fare_tag", "duration_tag"]
            ).alias("tags")
        )
        .select(
            pl.col("id").cast(pl.UInt64),
            pl.col("PULocationID").cast(pl.UInt64).alias("src"),
            pl.col("DOLocationID").cast(pl.UInt64).alias("dest"),
            "payment",
            pl.col("passenger_count").fill_null(0).cast(pl.UInt64),
            pl.col("trip_distance").cast(pl.Float64),
            pl.col("duration_min").cast(pl.Float64),
            pl.col("fare_amount").cast(pl.Float64),
            pl.col("tip_amount").cast(pl.Float64),
            pl.col("total_amount").cast(pl.Float64),
            "tags",
        )
        .collect()
    )

    src_zones = zones.select(
        pl.col("id").alias("src"),
        pl.col("zone").alias("pickup_zone"),
        pl.col("borough").alias("pickup_borough"),
    )
    dest_zones = zones.select(
        pl.col("id").alias("dest"),
        pl.col("zone").alias("dropoff_zone"),
        pl.col("borough").alias("dropoff_borough"),
    )

    return (
        trips.join(src_zones, on="src", how="inner")
        .join(dest_zones, on="dest", how="inner")
        .with_columns(
            pl.struct(
                [
                    "pickup_zone",
                    "dropoff_zone",
                    "pickup_borough",
                    "dropoff_borough",
                    "payment",
                    "passenger_count",
                    "trip_distance",
                    "duration_min",
                    "fare_amount",
                    "tip_amount",
                    "total_amount",
                    "tags",
                ]
            ).alias("trip")
        )
    )


def payment_label() -> pl.Expr:
    return (
        pl.when(pl.col("payment_type") == 1)
        .then(pl.lit("credit_card"))
        .when(pl.col("payment_type") == 2)
        .then(pl.lit("cash"))
        .when(pl.col("payment_type") == 3)
        .then(pl.lit("no_charge"))
        .when(pl.col("payment_type") == 4)
        .then(pl.lit("dispute"))
        .when(pl.col("payment_type") == 5)
        .then(pl.lit("unknown"))
        .when(pl.col("payment_type") == 6)
        .then(pl.lit("voided_trip"))
        .otherwise(pl.lit("other"))
    )


def data(name: str, url: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    if path.exists():
        print(f"using cached data: {url}")
    else:
        print(f"downloading {url}", file=sys.stderr)
        urllib.request.urlretrieve(url, path)
    return path


def zone_id(zones: pl.DataFrame, zone: str) -> int:
    if zone.isdigit():
        return int(zone)
    rows = zones.filter(pl.col("zone") == zone).select("id").to_series().to_list()
    if not rows:
        raise SystemExit(f"unknown taxi zone: {zone}")
    return rows[0]


def zone_name(zones: pl.DataFrame, zone: int) -> str:
    return zones.filter(pl.col("id") == zone).select("zone").item()


def zone_borough(zones: pl.DataFrame, zone: int) -> str:
    return zones.filter(pl.col("id") == zone).select("borough").item()


if __name__ == "__main__":
    main()
