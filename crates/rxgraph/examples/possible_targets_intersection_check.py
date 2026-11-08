"""Small independent model for path-local ending bounds and lazy replay.

Run: python3 crates/rxgraph/examples/possible_targets_intersection_check.py
This deliberately models the algorithm separately from rxgraph's Rust engine.
"""

from collections import defaultdict, deque
from dataclasses import dataclass

A, B = 1, 2
TARGET_NODE = {4: A, 5: B}


@dataclass(frozen=True)
class Edge:
    id: int
    src: int
    dest: int


EDGES = [
    Edge(0, 0, 1), Edge(1, 0, 2), Edge(2, 1, 3), Edge(3, 2, 3),
    Edge(4, 3, 4), Edge(5, 4, 5), Edge(6, 3, 5), Edge(7, 3, 6),
    Edge(8, 6, 3), Edge(9, 3, 1), Edge(10, 2, 5), Edge(11, 4, 1),
    Edge(12, 1, 4),
]
OUTGOING = defaultdict(list)
INCOMING = defaultdict(list)
for edge in EDGES:
    OUTGOING[edge.src].append(edge)
    INCOMING[edge.dest].append(edge)


def reachability():
    """Overapproximate target reachability, including each target itself."""
    result = defaultdict(int)
    for target, bit in TARGET_NODE.items():
        seen = {target}
        pending = [target]
        while pending:
            node = pending.pop()
            result[node] |= bit
            for edge in INCOMING[node]:
                if edge.src not in seen:
                    seen.add(edge.src)
                    pending.append(edge.src)
    return result


REACHABLE = reachability()


def refinement(edge, route):
    """Cheap, exact conservative ending restriction, or None for unknown."""
    if edge.id == 0:
        return B
    if edge.id == 7:
        return 0 if route == 1 else B
    if edge.id == 8:
        return B
    return None


def next_route(route, edge):
    if edge.src == 0:
        return edge.dest
    return route


def transition(state, edge):
    """Exact kernel: mask, route, depth, and path-dependent signature."""
    mask, route, depth, signature = state
    bound = refinement(edge, route)
    if bound is not None:
        mask &= bound
    if not mask:
        return "reject", None
    route = next_route(route, edge)
    child = (mask, route, depth + 1, signature * 17 + edge.id + 1)
    if TARGET_NODE.get(edge.dest, 0) & mask:
        return "complete", child
    return "continue", child


@dataclass(frozen=True)
class Entry:
    nodes: tuple
    edges: tuple
    route: int
    bound: int | None
    state: tuple | None
    history: tuple
    next_edge: int = 0


def replay(edges, initial_mask):
    state = (initial_mask, 0, 0, 0)
    history = [state]
    for i, edge in enumerate(edges):
        kind, state = transition(state, edge)
        if kind == "reject":
            return kind, None, ()
        history.append(state)
        if kind == "complete" and i != len(edges) - 1:
            raise AssertionError("candidate traversal crossed an exact completion")
    return kind, state, tuple(history)


def walk(strategy, mode, visits, limit, initial_bound=A | B, max_depth=7):
    """Serial cursor; lazy mode evaluates exact state at each candidate ending."""
    if initial_bound == 0 or limit == 0:
        return [], {}
    initial_mask = A | B if initial_bound is None else initial_bound
    initial_state = (initial_mask, 0, 0, 0)
    frontier = deque([Entry((0,), (), 0, initial_bound, initial_state, (initial_state,))])
    paths = []
    seen_bounds = defaultdict(set)
    while frontier:
        item = frontier.popleft() if strategy == "bfs" else frontier.pop()
        node = item.nodes[-1]
        if item.next_edge >= len(OUTGOING[node]) or len(item.edges) == max_depth:
            continue
        edge = OUTGOING[node][item.next_edge]
        resumed = Entry(item.nodes, item.edges, item.route, item.bound, item.state,
                        item.history, item.next_edge + 1)
        if strategy == "bfs":
            frontier.appendleft(resumed)
        else:
            frontier.append(resumed)
        if item.nodes.count(edge.dest) >= visits:
            continue
        route = next_route(item.route, edge)
        bound = item.bound
        if mode != "baseline":
            constraint = refinement(edge, item.route)
            if constraint is not None:
                bound = constraint if bound is None else bound & constraint
            if bound is not None:
                bound &= REACHABLE[edge.dest]
                if not bound:
                    continue
            seen_bounds[edge.dest].add(bound)
        nodes = item.nodes + (edge.dest,)
        edges = item.edges + (edge,)
        state = item.state
        history = item.history
        if mode == "lazy":
            if bound is None or TARGET_NODE.get(edge.dest, 0) & bound:
                kind, state, history = replay(edges, initial_mask)
            else:
                kind, state, history = "continue", None, ()
        else:
            kind, state = transition(state, edge)
            if kind != "reject":
                history = history + (state,)
        if kind == "reject":
            continue
        if kind == "complete":
            paths.append((nodes, tuple(e.id for e in edges), state, history))
            if limit is not None and len(paths) == limit:
                break
        else:
            child = Entry(nodes, edges, route, bound, state, history)
            frontier.append(child)
    return paths, seen_bounds


def main():
    assert REACHABLE[4] == A | B, "a target may also be an intermediate node"
    checks = 0
    for strategy in ("dfs", "bfs"):
        for visits in (1, 2):
            for limit in (None, 0, 1, 2, 5):
                baseline, _ = walk(strategy, "baseline", visits, limit)
                bounded, seen = walk(strategy, "bounded", visits, limit)
                lazy, _ = walk(strategy, "lazy", visits, limit)
                assert baseline == bounded == lazy, (strategy, visits, limit)
                if limit is None:
                    assert B in seen[3] and A | B in seen[3], "reconverging paths require independent bounds"
                    assert any(4 in nodes[1:-1] and nodes[-1] == 5 for nodes, *_ in baseline), "target A must be traversable as an intermediate"
                checks += 1
        for mode in ("baseline", "bounded", "lazy"):
            paths, _ = walk(strategy, mode, visits=1, limit=None, initial_bound=0)
            assert paths == []
            checks += 1
        baseline, _ = walk(strategy, "baseline", visits=1, limit=None, initial_bound=None)
        for mode in ("bounded", "lazy"):
            actual, _ = walk(strategy, mode, visits=1, limit=None, initial_bound=None)
            assert baseline == actual, (strategy, "unknown initial bound", mode)
            checks += 1
    print(f"{checks} comparisons passed: BFS/DFS, branch-local bounds, target-as-intermediate, cycles, visits, limits, histories, unknown/empty bounds")


if __name__ == "__main__":
    main()
