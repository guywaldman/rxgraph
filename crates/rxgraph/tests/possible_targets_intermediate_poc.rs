//! Standalone semantic check for a deferred candidate walker. This is not the
//! benchmark's terminal-only lazy implementation or a production traversal.
use anyhow::Result;
use arrow::array::record_batch;
use rxgraph::{EdgeCtx, Graph, GraphId, Kernel, NodeId, RunOptions, Transition, TraversalStrategy};
use std::collections::VecDeque;

// Ordered outgoing edges: includes target 1 as an intermediate node, target 3,
// a cycle between 1 and 2, and a dead branch to 4.
const EDGES: [(u32, u32); 7] = [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (2, 1), (2, 3)];
const TARGETS: u8 = 0b11;
fn bit(node: u32) -> u8 {
    match node {
        1 => 1,
        3 => 2,
        _ => 0,
    }
}
fn refine(edge: usize) -> Option<u8> {
    match edge {
        2 => Some(0),
        1 => Some(TARGETS),
        4 | 6 => Some(2),
        5 => Some(1),
        _ => None,
    }
}
fn apply(edge: usize, state: u64) -> Transition<u64> {
    let (src, dest) = EDGES[edge];
    let next = state + edge as u64 + 1;
    if dest == 3 || (src == 2 && dest == 1) {
        Transition::Complete(next)
    } else {
        Transition::Continue(next)
    }
}
struct Exact;
impl Kernel for Exact {
    type State = u64;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<u64> {
        Ok(0)
    }
    fn transition(&self, cx: &EdgeCtx<'_, u64>) -> Result<Transition<u64>> {
        Ok(apply(cx.edge() as usize, *cx.state()))
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Path {
    nodes: Vec<u64>,
    edges: Vec<u64>,
    state: u64,
    targets: u8,
}
fn convert(paths: Vec<rxgraph::GraphPath<'_, u64>>) -> Vec<Path> {
    fn ids(v: Vec<GraphId<'_>>) -> Vec<u64> {
        v.into_iter()
            .map(|i| match i {
                GraphId::U64(n) => n,
                _ => unreachable!(),
            })
            .collect()
    }
    paths
        .into_iter()
        .map(|p| Path {
            nodes: ids(p.nodes),
            edges: ids(p.edges),
            state: p.state,
            targets: TARGETS,
        })
        .collect()
}
fn extend(parent: &Path, edge: usize, max_visits: usize) -> Option<(Path, bool)> {
    let (_, dest) = EDGES[edge];
    if parent.nodes.iter().filter(|&&n| n == dest as u64).count() >= max_visits {
        return None;
    }
    let targets = parent.targets & refine(edge).unwrap_or(parent.targets);
    if targets == 0 {
        return None;
    }
    let mut child = parent.clone();
    child.nodes.push(dest as u64);
    child.edges.push(edge as u64);
    child.targets = targets;
    if bit(dest) & targets == 0 {
        return Some((child, false));
    }
    // Replay the exact transitions when a possible ending is encountered.
    // A Continue result at target 1 leaves this prefix eligible for expansion.
    let mut state = 0;
    let mut complete = false;
    for (i, &e) in child.edges.iter().enumerate() {
        match apply(e as usize, state) {
            Transition::Reject => return None,
            Transition::Continue(next) => state = next,
            Transition::Complete(next) => {
                assert_eq!(
                    i + 1,
                    child.edges.len(),
                    "earlier completed prefix must already have stopped"
                );
                state = next;
                complete = true;
            }
        }
    }
    child.state = state;
    Some((child, complete))
}
fn lazy(strategy: TraversalStrategy, max_visits: usize, limit: Option<usize>) -> Vec<Path> {
    let root = Path {
        nodes: vec![0],
        edges: vec![],
        state: 0,
        targets: TARGETS,
    };
    let mut out = Vec::new();
    if limit == Some(0) {
        return out;
    }
    if matches!(strategy, TraversalStrategy::DepthFirst) {
        let mut stack = vec![(root, 0usize)];
        while let Some((path, next)) = stack.last_mut() {
            let outgoing = EDGES
                .iter()
                .enumerate()
                .filter(|(_, e)| e.0 == *path.nodes.last().unwrap() as u32)
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            if *next == outgoing.len() {
                stack.pop();
                continue;
            }
            let edge = outgoing[*next];
            *next += 1;
            if let Some((child, complete)) = extend(path, edge, max_visits) {
                if complete {
                    out.push(child);
                    if limit.is_some_and(|n| out.len() >= n) {
                        break;
                    }
                } else if child.edges.len() < 6 {
                    stack.push((child, 0));
                }
            }
        }
    } else {
        let mut queue = VecDeque::from([root]);
        'search: while let Some(path) = queue.pop_front() {
            if path.edges.len() >= 6 {
                continue;
            }
            for (edge, _) in EDGES
                .iter()
                .enumerate()
                .filter(|(_, e)| e.0 == *path.nodes.last().unwrap() as u32)
            {
                if let Some((child, complete)) = extend(&path, edge, max_visits) {
                    if complete {
                        out.push(child);
                        if limit.is_some_and(|n| out.len() >= n) {
                            break 'search;
                        }
                    } else {
                        queue.push_back(child);
                    }
                }
            }
        }
    }
    out
}
#[test]
fn candidate_target_can_continue_and_cycles_preserve_order_and_limits() -> Result<()> {
    let graph = Graph::new(
        record_batch!(("id", UInt64, [0, 1, 2, 3, 4]))?,
        record_batch!(
            ("id", UInt64, [0, 1, 2, 3, 4, 5, 6]),
            ("src", UInt64, [0, 0, 0, 1, 1, 2, 2]),
            ("dest", UInt64, [1, 2, 4, 2, 3, 1, 3])
        )?,
    )?;
    for strategy in [
        TraversalStrategy::DepthFirst,
        TraversalStrategy::BreadthFirst,
    ] {
        for visits in [1, 2] {
            for limit in [None, Some(0), Some(1), Some(2), Some(20)] {
                let cfg = RunOptions {
                    start_nodes: vec![0u64.into()],
                    strategy,
                    parallel: false,
                    max_depth: Some(6),
                    max_visits_per_node: visits,
                    max_paths: limit,
                    ..Default::default()
                };
                let expected = convert(graph.search_paths_with(Exact, cfg)?.paths);
                let actual = lazy(strategy, visits, limit);
                let rows = |paths: &[Path]| {
                    paths
                        .iter()
                        .map(|p| (p.nodes.clone(), p.edges.clone(), p.state))
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    rows(&actual),
                    rows(&expected),
                    "{strategy:?}, visits={visits}, limit={limit:?}"
                );
                if limit.is_none() {
                    assert!(
                        actual.iter().any(|p| p.nodes == [0, 1, 3]),
                        "target 1 must be traversed when exact transition continues"
                    );
                    assert!(
                        actual.iter().any(|p| p.nodes == [0, 2, 1]),
                        "target 1 can also complete"
                    );
                }
            }
        }
    }
    Ok(())
}
