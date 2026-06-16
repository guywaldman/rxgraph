use anyhow::{Context, Result};

use crate::{
    graph::{EdgeId, Graph, GraphRepo, NodeId, OwnedGraphId},
    traversal::{
        EdgeCtx, GraphPath, Kernel, SearchResult,
        config::{TraversalConfig, TraversalStrategy},
        engine::{self, PathEntry, SearchAdapter},
        kernel::PayloadCache,
    },
};

/// Execution knobs for a native [`Kernel`] search.
///
/// This carries everything needed to drive a search except the kernel itself.
/// The DSL entry point [`Graph::search`] builds one of these from a
/// [`TraversalConfig`] and delegates to [`Graph::search_with`].
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// External node IDs where traversal starts.
    pub start_nodes: Vec<OwnedGraphId>,
    /// Maximum accepted-edge depth. `None` means unbounded.
    pub max_depth: Option<usize>,
    /// Maximum returned path count. Treated as a soft early-stop limit under
    /// parallel traversal; the returned vector is truncated exactly.
    pub max_paths: Option<usize>,
    /// Search order.
    pub strategy: TraversalStrategy,
    /// Maximum revisits allowed per node inside one path.
    pub max_revisits_per_node: usize,
    /// Whether Rayon-backed parallel traversal is enabled.
    pub parallel: bool,
    /// Whether returned paths include per-node state history.
    pub intermediate_states: bool,
    /// Whether to report search progress on stderr.
    pub progress: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            start_nodes: Vec::new(),
            max_depth: None,
            max_paths: None,
            strategy: TraversalStrategy::default(),
            max_revisits_per_node: 0,
            parallel: true,
            intermediate_states: false,
            progress: false,
        }
    }
}

impl Graph {
    /// Runs a configured DSL traversal and materializes matching paths.
    ///
    /// Start nodes are resolved from external IDs to compact internal IDs before
    /// search. Returned paths are materialized back to external IDs after `stop`
    /// accepts a path.
    pub fn search(&self, config: TraversalConfig) -> Result<SearchResult<'_>> {
        let TraversalConfig {
            kernel,
            start_nodes,
            max_depth,
            max_paths,
            strategy,
            max_revisits_per_node,
            parallel,
            intermediate_states,
            progress,
        } = config;
        let kernel = kernel.bind(self)?;
        let run = RunOptions {
            start_nodes,
            max_depth,
            max_paths,
            strategy,
            max_revisits_per_node,
            parallel,
            intermediate_states,
            progress,
        };
        self.search_with(kernel, run)
    }

    /// Runs a native [`Kernel`] traversal and materializes matching paths.
    ///
    /// The engine is monomorphized over `K`, so per-edge kernel calls are
    /// statically dispatched.
    pub fn search_with<K: Kernel + Sync>(
        &self,
        kernel: K,
        run: RunOptions,
    ) -> Result<SearchResult<'_>>
    where
        K::State: Send + Sync + Clone,
    {
        let adapter = GraphSearchAdapter {
            graph: self,
            kernel: &kernel,
        };
        let result = engine::search(&adapter, run)?;
        Ok(SearchResult {
            paths: result.paths,
            stats: result.stats,
        })
    }
}

struct GraphSearchAdapter<'g, 'k, K> {
    graph: &'g Graph,
    kernel: &'k K,
}

impl<'g, 'k, K> SearchAdapter for GraphSearchAdapter<'g, 'k, K>
where
    K: Kernel,
{
    type State = K::State;
    type Path = GraphPath<'g>;
    type Cache = PayloadCache;

    fn resolve_node(&self, external: crate::GraphId<'_>) -> Result<Option<NodeId>> {
        Ok(self.graph.repo.internal_node(external))
    }

    fn initial_state(&self, node: NodeId) -> Result<Self::State> {
        Ok(self.kernel.initial_state(self.graph, node))
    }

    fn out_degree(&self, node: NodeId) -> Result<usize> {
        Ok(self.graph.repo.out_degree(node))
    }

    fn for_each_outgoing<F>(&self, node: NodeId, mut visit: F) -> Result<()>
    where
        F: FnMut(EdgeId, NodeId) -> Result<bool>,
    {
        let (edges, dests) = self.graph.repo.outgoing_slice(node);
        for (&edge, &dest) in edges.iter().zip(dests) {
            if !visit(edge, dest)? {
                break;
            }
        }
        Ok(())
    }

    fn make_cache(&self) -> Self::Cache {
        PayloadCache::new()
    }

    fn eval_edge(
        &self,
        src: NodeId,
        edge: EdgeId,
        dest: NodeId,
        state: &Self::State,
        cache: &Self::Cache,
    ) -> Result<Option<(Self::State, bool)>> {
        let cx = EdgeCtx::new(self.graph, src, dest, edge, state, cache);
        if !self.kernel.visit(&cx)? {
            return Ok(None);
        }
        let state = self.kernel.next_state(&cx)?;
        let stop = self.kernel.stop(&cx.with_state(&state))?;
        Ok(Some((state, stop)))
    }

    fn materialize(
        &self,
        arena: &[PathEntry<Self::State>],
        mut path: usize,
        intermediate_states: bool,
    ) -> Result<Self::Path> {
        let mut nodes = Vec::with_capacity(arena[path].depth + 1);
        let mut edges = Vec::with_capacity(arena[path].depth);
        let state = self.kernel.state_row(&arena[path].state);
        let mut states = intermediate_states.then(|| Vec::with_capacity(nodes.capacity()));

        loop {
            nodes.push(
                self.graph
                    .repo
                    .external_node(arena[path].node)
                    .context("path references missing node")?,
            );
            if let Some(edge) = arena[path].incoming_edge {
                edges.push(
                    self.graph
                        .repo
                        .external_edge(edge)
                        .context("path references missing edge")?,
                );
            }
            if let Some(states) = &mut states {
                states.push(self.kernel.state_row(&arena[path].state));
            }
            match arena[path].parent {
                Some(parent) => path = parent,
                None => break,
            }
        }

        nodes.reverse();
        edges.reverse();
        if let Some(states) = &mut states {
            states.reverse();
        }
        Ok(GraphPath {
            nodes,
            edges,
            state,
            intermediate_states: states,
        })
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::record_batch;

    use super::*;
    use crate::{
        dsl::{DslExpr as e, DslKernel, Value},
        graph::{EDGE_DEST_COL, EDGE_SRC_COL, GraphId, ID_COL, Repo},
        traversal::{
            SearchStats,
            config::{TraversalConfigBuilder, TraversalStrategy},
        },
    };

    fn graph() -> Graph {
        Graph {
            repo: Repo::from_tables(
                record_batch!(
                    (ID_COL, Utf8, ["a", "b", "c", "d"]),
                    ("kind", Utf8, ["start", "mid", "mid", "end"])
                )
                .unwrap(),
                record_batch!(
                    (ID_COL, Utf8, ["ab", "ac", "bd", "cd", "ba"]),
                    (EDGE_SRC_COL, Utf8, ["a", "a", "b", "c", "b"]),
                    (EDGE_DEST_COL, Utf8, ["b", "c", "d", "d", "a"]),
                    ("ok", Boolean, [true, false, true, true, true])
                )
                .unwrap(),
            )
            .unwrap(),
        }
    }

    fn branching_graph() -> Graph {
        Graph {
            repo: Repo::from_tables(
                record_batch!(
                    (ID_COL, Utf8, ["s", "a", "b", "c", "d", "e", "f", "z"]),
                    (
                        "kind",
                        Utf8,
                        ["start", "mid", "mid", "mid", "mid", "mid", "mid", "end"]
                    )
                )
                .unwrap(),
                record_batch!(
                    (
                        ID_COL,
                        Utf8,
                        ["sa", "sb", "sc", "ad", "ae", "bf", "cz", "dz", "ez", "fz"]
                    ),
                    (
                        EDGE_SRC_COL,
                        Utf8,
                        ["s", "s", "s", "a", "a", "b", "c", "d", "e", "f"]
                    ),
                    (
                        EDGE_DEST_COL,
                        Utf8,
                        ["a", "b", "c", "d", "e", "f", "z", "z", "z", "z"]
                    ),
                    (
                        "ok",
                        Boolean,
                        [true, true, true, true, true, true, true, true, true, true]
                    )
                )
                .unwrap(),
            )
            .unwrap(),
        }
    }

    fn integer_graph() -> Graph {
        Graph {
            repo: Repo::from_tables(
                record_batch!((ID_COL, UInt64, [1, 2, 3])).unwrap(),
                record_batch!(
                    (ID_COL, UInt64, [10, 20]),
                    (EDGE_SRC_COL, UInt64, [1, 2]),
                    (EDGE_DEST_COL, UInt64, [2, 3])
                )
                .unwrap(),
            )
            .unwrap(),
        }
    }

    fn traversal(visit: e, stop: e) -> TraversalConfig {
        TraversalConfigBuilder::new(DslKernel::new(visit, [], stop, []))
            .with_start_nodes(["a".to_string()])
            .with_strategy(TraversalStrategy::BreadthFirst)
            .with_parallelism(false)
            .build()
    }

    fn path_set(result: &SearchResult<'_>) -> Vec<Vec<String>> {
        let mut paths = result
            .paths
            .iter()
            .map(|p| p.nodes.iter().map(|&n| id_label(n)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        paths.sort();
        paths
    }

    fn id_label(id: GraphId<'_>) -> String {
        match id {
            GraphId::U64(value) => value.to_string(),
            GraphId::Str(value) => value.to_owned(),
        }
    }

    fn state_u64(state: &[(String, Value)], name: &str) -> u64 {
        match state
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
        {
            Some(Value::U64(value)) => *value,
            other => panic!("expected {name} to be U64, got {other:?}"),
        }
    }

    #[test]
    fn returns_stopped_paths() {
        let graph = graph();
        let result = graph
            .search(traversal(
                e::bool_lit(true),
                e::dest("kind").eq(e::string_lit("end")),
            ))
            .unwrap();
        assert_eq!(result.paths.len(), 2);
        assert_eq!(
            result.paths[0].nodes,
            vec![GraphId::Str("a"), GraphId::Str("b"), GraphId::Str("d")]
        );
        assert_eq!(
            result.paths[1].nodes,
            vec![GraphId::Str("a"), GraphId::Str("c"), GraphId::Str("d")]
        );
        assert_eq!(
            result.stats,
            SearchStats {
                start_nodes: 1,
                path_entries: 5,
                evaluated_edges: 4,
                accepted_edges: 4,
                rejected_edges: 0,
                skipped_revisits: 1,
                stopped_paths: 2,
                max_depth: 2,
                materialized_node_payloads: 0,
                materialized_edge_payloads: 0,
                ..SearchStats::default()
            }
        );
    }

    #[test]
    fn filters_edges_and_limits_depth() {
        let config =
            TraversalConfigBuilder::new(DslKernel::new(e::edge("ok"), [], e::bool_lit(true), []))
                .with_start_nodes(["a".to_string()])
                .with_max_depth(1)
                .with_parallelism(false)
                .build();

        let graph = graph();
        let result = graph.search(config).unwrap();
        assert_eq!(
            result
                .paths
                .iter()
                .map(|p| p.nodes.clone())
                .collect::<Vec<_>>(),
            vec![vec![GraphId::Str("a"), GraphId::Str("b")]]
        );
        assert_eq!(result.stats.evaluated_edges, 2);
        assert_eq!(result.stats.accepted_edges, 1);
        assert_eq!(result.stats.rejected_edges, 1);
        assert_eq!(result.stats.max_depth, 1);
    }

    #[test]
    fn returns_final_state_and_optional_intermediate_states() {
        let kernel = DslKernel::new(
            e::bool_lit(true),
            [("hops".into(), e::state("hops").plus(e::uint_lit(1)))],
            e::dest("kind").eq(e::string_lit("end")),
            [("hops".into(), Value::U64(0))],
        );

        let graph = graph();
        let without_history = graph
            .search(
                TraversalConfigBuilder::new(kernel.clone())
                    .with_start_nodes(["a".to_string()])
                    .with_strategy(TraversalStrategy::BreadthFirst)
                    .with_parallelism(false)
                    .build(),
            )
            .unwrap();

        assert_eq!(state_u64(&without_history.paths[0].state, "hops"), 2);
        assert!(without_history.paths[0].intermediate_states.is_none());

        let with_history = graph
            .search(
                TraversalConfigBuilder::new(kernel)
                    .with_start_nodes(["a".to_string()])
                    .with_strategy(TraversalStrategy::BreadthFirst)
                    .with_parallelism(false)
                    .with_intermediate_states(true)
                    .build(),
            )
            .unwrap();

        let states = with_history.paths[0].intermediate_states.as_ref().unwrap();
        assert_eq!(states.len(), with_history.paths[0].nodes.len());
        assert_eq!(
            states
                .iter()
                .map(|state| state_u64(state, "hops"))
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(states.last().unwrap(), &with_history.paths[0].state);
    }

    #[test]
    fn rejects_revisits_by_default() {
        let graph = graph();
        let result = graph
            .search(traversal(e::bool_lit(true), e::bool_lit(false)))
            .unwrap();
        assert!(result.paths.is_empty());
        assert_eq!(result.stats.skipped_revisits, 1);
        assert_eq!(result.stats.stopped_paths, 0);
    }

    #[test]
    fn reports_unknown_start_node() {
        let config = TraversalConfigBuilder::new(DslKernel::new(
            e::bool_lit(true),
            [],
            e::bool_lit(true),
            [("x".to_string(), Value::U64(0))],
        ))
        .with_start_nodes(["missing".to_string()])
        .build();

        assert!(
            graph()
                .search(config)
                .unwrap_err()
                .to_string()
                .contains("unknown start node")
        );
    }

    #[test]
    fn integer_ids_materialize_and_dsl_ids_work() {
        let graph = integer_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(DslKernel::new(
                    e::bool_lit(true),
                    [],
                    e::dest_id().eq(e::uint_lit(3)),
                    [],
                ))
                .with_start_nodes([1u64])
                .with_strategy(TraversalStrategy::BreadthFirst)
                .with_parallelism(false)
                .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 1);
        assert_eq!(
            result.paths[0].nodes,
            vec![GraphId::U64(1), GraphId::U64(2), GraphId::U64(3)]
        );
        assert_eq!(
            result.paths[0].edges,
            vec![GraphId::U64(10), GraphId::U64(20)]
        );
    }

    #[test]
    fn string_special_id_columns_work() {
        let graph = graph();
        let result = graph
            .search(traversal(
                e::edge_id().ne(e::string_lit("ac")),
                e::dest_id().eq(e::string_lit("d")),
            ))
            .unwrap();

        assert_eq!(path_set(&result), vec![vec!["a", "b", "d"]]);
    }

    #[test]
    fn parallel_bfs_matches_serial_path_set() {
        let graph = branching_graph();
        let base = DslKernel::new(
            e::bool_lit(true),
            [],
            e::dest("kind").eq(e::string_lit("end")),
            [],
        );
        let serial = graph
            .search(
                TraversalConfigBuilder::new(base.clone())
                    .with_start_nodes(["s".to_string()])
                    .with_strategy(TraversalStrategy::BreadthFirst)
                    .with_parallelism(false)
                    .build(),
            )
            .unwrap();
        let parallel = graph
            .search(
                TraversalConfigBuilder::new(base)
                    .with_start_nodes(["s".to_string()])
                    .with_strategy(TraversalStrategy::BreadthFirst)
                    .with_parallelism(true)
                    .build(),
            )
            .unwrap();

        assert_eq!(path_set(&parallel), path_set(&serial));
        assert_eq!(
            parallel.stats.accepted_edges + parallel.stats.rejected_edges,
            parallel.stats.evaluated_edges
        );
        assert_eq!(parallel.stats.stopped_paths, parallel.paths.len());
    }

    #[test]
    fn parallel_bfs_truncates_max_paths() {
        let graph = branching_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(DslKernel::new(
                    e::bool_lit(true),
                    [],
                    e::dest("kind").eq(e::string_lit("end")),
                    [],
                ))
                .with_start_nodes(["s".to_string()])
                .with_strategy(TraversalStrategy::BreadthFirst)
                .with_max_paths(2)
                .with_parallelism(true)
                .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 2);
    }

    #[test]
    fn parallel_dfs_returns_valid_paths() {
        let graph = branching_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(DslKernel::new(
                    e::bool_lit(true),
                    [],
                    e::dest("kind").eq(e::string_lit("end")),
                    [],
                ))
                .with_start_nodes(["s".to_string()])
                .with_strategy(TraversalStrategy::DepthFirst)
                .with_parallelism(true)
                .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 4);
        for path in &result.paths {
            assert_eq!(path.nodes.first(), Some(&GraphId::Str("s")));
            assert_eq!(path.nodes.last(), Some(&GraphId::Str("z")));
        }
        assert_eq!(path_set(&result).len(), 4);
    }

    /// This test case is needed because parallel DFS was highly unoptimized for 1 thread previously,
    /// and remediation required adding [`DfsSeed`] so parent links remain valid after splitting work out of a temporary arena.
    #[test]
    fn parallel_dfs_single_thread_branch_returns_valid_paths() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            let graph = branching_graph();
            let parallel = graph
                .search(
                    TraversalConfigBuilder::new(DslKernel::new(
                        e::bool_lit(true),
                        [],
                        e::dest("kind").eq(e::string_lit("end")),
                        [],
                    ))
                    .with_start_nodes(["s".to_string()])
                    .with_strategy(TraversalStrategy::DepthFirst)
                    .with_parallelism(true)
                    .build(),
                )
                .unwrap();
            let serial = graph
                .search(
                    TraversalConfigBuilder::new(DslKernel::new(
                        e::bool_lit(true),
                        [],
                        e::dest("kind").eq(e::string_lit("end")),
                        [],
                    ))
                    .with_start_nodes(["s".to_string()])
                    .with_strategy(TraversalStrategy::DepthFirst)
                    .with_parallelism(false)
                    .build(),
                )
                .unwrap();

            assert_eq!(parallel.paths.len(), 4);
            assert_eq!(path_set(&parallel), path_set(&serial));
        });
    }

    #[test]
    fn parallel_dfs_soft_stop_returns_exact_max_paths() {
        let graph = branching_graph();
        let result = graph
            .search(
                TraversalConfigBuilder::new(DslKernel::new(
                    e::bool_lit(true),
                    [],
                    e::dest("kind").eq(e::string_lit("end")),
                    [],
                ))
                .with_start_nodes(["s".to_string()])
                .with_strategy(TraversalStrategy::DepthFirst)
                .with_max_paths(2)
                .with_parallelism(true)
                .build(),
            )
            .unwrap();

        assert_eq!(result.paths.len(), 2);
    }

    #[test]
    fn builder_parallelism_defaults_on_and_can_be_disabled() {
        let kernel = DslKernel::new(e::bool_lit(true), [], e::bool_lit(true), []);
        assert!(TraversalConfigBuilder::new(kernel.clone()).build().parallel);
        assert!(
            !TraversalConfigBuilder::new(kernel)
                .with_parallelism(false)
                .build()
                .parallel
        );
    }
}
