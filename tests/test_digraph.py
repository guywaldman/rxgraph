import itertools
import random
from dataclasses import dataclass
from math import ceil, dist

import rxgraph as rxg

DESTINATIONS_COUNT = 100
RANDOM_SEED = 7
MAX_COORDINATE = 2**10
MAX_ROUTE_COST = 100


@dataclass(frozen=True)
class Destination:
    name: str
    pos: tuple[int, int]
    can_travel: bool = True


@dataclass(frozen=True)
class TravelRoute:
    src: str
    dest: str
    distance: int
    cost: int


def generate_mock_data() -> tuple[list[Destination], list[TravelRoute]]:
    """Generate travel routes, where there is a travel distance and a budget constraint."""
    rng = random.Random(RANDOM_SEED)
    destinations = [
        Destination(
            name=f"dest-{i}",
            pos=(rng.randint(1, MAX_COORDINATE), rng.randint(1, MAX_COORDINATE)),
            can_travel=rng.random() > 0.01,
        )
        for i in range(DESTINATIONS_COUNT)
    ]
    routes = [
        TravelRoute(
            src=src.name,
            dest=dest.name,
            distance=ceil(dist(src.pos, dest.pos)),
            cost=rng.randint(0, MAX_ROUTE_COST),
        )
        for src, dest in itertools.combinations(destinations, 2)
        if src.name != dest.name
    ]
    return destinations, routes


def test_digraph_shortest_path() -> None:
    destinations, routes = generate_mock_data()
    graph = rxg.DiGraph.from_edges(
        [
            (
                route.src,
                route.dest,
                {"distance": route.distance, "cost": route.cost},
            )
            for route in routes
        ],
        nodes=[
            (
                destination.name,
                {"can_travel": destination.can_travel},
            )
            for destination in destinations
        ],
    )

    assert graph.node_count == DESTINATIONS_COUNT
    assert graph.edge_count == 2 * len(routes)
    assert graph.shortest_path("dest-0", "dest-99") == ["dest-0", "dest-99"]
    assert graph.shortest_path("dest-99", "dest-0") == ["dest-99", "dest-0"]


def test_digraph_traversal() -> None:
    MAX_BUDGET = 100
    MAX_DIST = 2_000

    destinations, routes = generate_mock_data()
    graph = rxg.DiGraph.from_edges(
        [
            (
                route.src,
                route.dest,
                {"dist": route.distance, "cost": route.cost},
            )
            for route in routes
        ],
        nodes=[
            (
                destination.name,
                {"can_travel": destination.can_travel},
            )
            for destination in destinations
        ],
    )

    result = graph.search(
        start_nodes=[destination.name for destination in destinations[:5]],
        visit=(
            (rxg.col("state.cost") + rxg.col("edge.cost") <= MAX_BUDGET)
            & (rxg.col("state.dist") + rxg.col("edge.dist") <= MAX_DIST)
            & rxg.col("dest.can_travel")
        ),
        next_state={
            "cost": rxg.col("state.cost") + rxg.col("edge.cost"),
            "dist": rxg.col("state.dist") + rxg.col("edge.dist"),
        },
        initial_state={"cost": 0, "dist": 0},
        max_paths=5,
    )

    assert result.paths
    assert all(path.state["cost"] <= MAX_BUDGET for path in result.paths)
    assert all(path.state["dist"] <= MAX_DIST for path in result.paths)
