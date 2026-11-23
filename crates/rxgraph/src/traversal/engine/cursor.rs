//! Resumable traversal state. Adapters are borrowed only while advancing a cursor.
use super::*;

const WORK_QUANTUM: usize = 8192;

pub(crate) struct Cursor<S, C> {
    cfg: RunConfig,
    arena: Vec<PathEntry<S>>,
    frames: Vec<DfsFrame>,
    frontier: VecDeque<usize>,
    bfs_edge: usize,
    bfs_count: Option<usize>,
    bfs_prefetched: usize,
    progress: Option<Progress>,
    next_start: usize,
    visits: Visits,
    cache: Option<C>,
    initialized: bool,
    done: bool,
    pub(crate) stats: SearchStats,
}

enum Visits {
    Bits(Vec<u64>),
    Dense(DenseVisits),
    Sparse(SparseVisits),
}
impl ActiveVisits for Visits {
    fn count(&self, n: NodeId) -> usize {
        match self {
            Self::Bits(v) => usize::from(v[n as usize / 64] & (1 << (n % 64)) != 0),
            Self::Dense(v) => v.count(n),
            Self::Sparse(v) => v.count(n),
        }
    }
    fn increment(&mut self, n: NodeId) {
        match self {
            Self::Bits(v) => v[n as usize / 64] |= 1 << (n % 64),
            Self::Dense(v) => v.increment(n),
            Self::Sparse(v) => v.increment(n),
        }
    }
    fn decrement(&mut self, n: NodeId) {
        match self {
            Self::Bits(v) => v[n as usize / 64] &= !(1 << (n % 64)),
            Self::Dense(v) => v.decrement(n),
            Self::Sparse(v) => v.decrement(n),
        }
    }
}

impl<S: Clone, C> Cursor<S, C> {
    pub(crate) fn new(run: RunConfig) -> Result<Self> {
        if run.max_visits_per_node == 0 {
            bail!("max_visits_per_node must be at least 1");
        }
        let progress = Some(Progress::new(run.progress));
        Ok(Self {
            cfg: run,
            arena: Vec::new(),
            frames: Vec::new(),
            frontier: VecDeque::new(),
            bfs_edge: 0,
            bfs_count: None,
            bfs_prefetched: 0,
            progress,
            next_start: 0,
            visits: Visits::Sparse(SparseVisits(HashMap::new())),
            cache: None,
            initialized: false,
            done: false,
            stats: SearchStats::default(),
        })
    }
    pub(crate) fn close(&mut self) {
        if let Some(progress) = self.progress.take() {
            progress.finish(&self.stats);
        }
        self.done = true;
        self.arena = Vec::new();
        self.frames = Vec::new();
        self.frontier = VecDeque::new();
        self.visits = Visits::Sparse(SparseVisits(HashMap::new()));
        self.cache = None;
        self.cfg.start_nodes = Vec::new();
    }
    fn initialize<A: SearchAdapter<State = S, Cache = C>>(&mut self, adapter: &A) -> Result<()> {
        if self.initialized {
            return Ok(());
        }
        self.initialized = true;
        self.cache = Some(adapter.make_cache());
        if self.cfg.max_paths == Some(0) || self.cfg.start_nodes.is_empty() {
            self.done = true;
            return Ok(());
        }
        if matches!(self.cfg.strategy, TraversalStrategy::DepthFirst) {
            if let Some(n) = adapter.dense_node_count() {
                self.visits = if self.cfg.max_visits_per_node == 1 {
                    Visits::Bits(vec![0; n.div_ceil(64)])
                } else {
                    Visits::Dense(DenseVisits(vec![0; n]))
                };
            }
        } else {
            let (arena, frontier, stats) = initial_arena(adapter, &self.cfg)?;
            self.arena = arena;
            self.frontier = frontier;
            self.stats = stats;
        }
        Ok(())
    }
    fn start<A: SearchAdapter<State = S, Cache = C>>(&mut self, adapter: &A) -> Result<bool> {
        let Some(external) = self.cfg.start_nodes.get(self.next_start) else {
            return Ok(false);
        };
        let node = adapter
            .resolve_node(external.as_ref())?
            .with_context(|| format!("unknown start node {external}"))?;
        self.next_start += 1;
        self.arena.push(PathEntry {
            node,
            incoming_edge: None,
            parent: None,
            depth: 0,
            state: adapter.initial_state(node)?,
            visits: VisitFingerprint::for_node(node),
        });
        if self.cfg.max_depth > 0 {
            adapter.prefetch_outgoing(&[node])?;
        }
        self.frames.push(DfsFrame {
            next_edge: 0,
            edge_count: if self.cfg.max_depth == 0 {
                0
            } else {
                adapter.out_degree(node)?
            },
        });
        self.visits.increment(node);
        self.stats.start_nodes += 1;
        self.stats.path_entries += 1;
        Ok(true)
    }
    /// May return an empty work chunk; public next_batch skips these.
    fn advance<A: SearchAdapter<State = S, Cache = C>>(
        &mut self,
        adapter: &A,
        limit: usize,
    ) -> Result<Vec<A::Path>> {
        let deadline = self.cfg.deadline;
        deadline.check()?;
        self.initialize(adapter)?;
        let limit = self.batch_limit(limit);
        let paths = if self.done {
            Vec::new()
        } else {
            // Specialize once per work chunk to keep timeout checks out of untimed edge loops.
            match (self.cfg.strategy, deadline.enabled()) {
                (TraversalStrategy::DepthFirst, false) => {
                    self.advance_dfs::<_, false>(adapter, limit)?
                }
                (TraversalStrategy::DepthFirst, true) => {
                    self.advance_dfs::<_, true>(adapter, limit)?
                }
                (TraversalStrategy::BreadthFirst, false) => {
                    self.advance_bfs::<_, false>(adapter, limit)?
                }
                (TraversalStrategy::BreadthFirst, true) => {
                    self.advance_bfs::<_, true>(adapter, limit)?
                }
            }
        };
        deadline.check()?;
        if self.at_limit() {
            self.done = true;
        }
        if let Some(progress) = &mut self.progress {
            progress.tick(&self.stats);
        }
        Ok(paths)
    }
    fn at_limit(&self) -> bool {
        self.cfg
            .max_paths
            .is_some_and(|n| self.stats.stopped_paths >= n)
    }
    fn batch_limit(&self, requested: usize) -> usize {
        self.cfg.max_paths.map_or(requested, |max| {
            requested.min(max.saturating_sub(self.stats.stopped_paths))
        })
    }
    fn accept<A: SearchAdapter<State = S, Cache = C>>(
        &mut self,
        adapter: &A,
        parent: usize,
        edge: EdgeEval<S>,
        paths: &mut Vec<A::Path>,
    ) -> Result<Option<usize>> {
        let stop = edge.stop;
        let child = push_entry(&mut self.arena, parent, edge);
        self.stats.accepted_edges += 1;
        self.stats.path_entries += 1;
        self.stats.max_depth = self.stats.max_depth.max(self.arena[child].depth);
        if stop {
            paths.push(adapter.materialize(&self.arena, child, self.cfg.intermediate_states)?);
            self.stats.stopped_paths += 1;
            self.arena.pop();
            Ok(None)
        } else {
            Ok(Some(child))
        }
    }
    fn advance_dfs<A: SearchAdapter<State = S, Cache = C>, const TIMED: bool>(
        &mut self,
        adapter: &A,
        limit: usize,
    ) -> Result<Vec<A::Path>> {
        let mut paths = Vec::new();
        let mut work = 0;
        let mut checkpoint = if TIMED { 1 } else { WORK_QUANTUM };
        while work < WORK_QUANTUM && paths.len() < limit {
            if TIMED && work == checkpoint {
                self.cfg.deadline.check()?;
                checkpoint = WORK_QUANTUM.min(work + Deadline::POLL_INTERVAL);
            }
            if self.frames.is_empty() && !self.start(adapter)? {
                self.done = true;
                break;
            }
            let parent = self.arena.len() - 1;
            let frame = self.frames.last().expect("active frame");
            if frame.next_edge == frame.edge_count {
                self.frames.pop();
                let entry = self.arena.pop().expect("active entry");
                self.visits.decrement(entry.node);
                work += 1;
                continue;
            }
            let start = frame.next_edge;
            let end = frame.edge_count.min(start + (checkpoint - work));
            let node = self.arena[parent].node;
            let mut accepted = None;
            let mut consumed = 0;
            adapter.for_each_range(node, start, end, |edge, dest| {
                consumed += 1;
                if self.visits.count(dest) >= self.cfg.max_visits_per_node {
                    self.stats.skipped_revisits += 1;
                    return Ok(true);
                }
                self.stats.evaluated_edges += 1;
                let Some((state, stop)) = adapter.eval_edge(
                    node,
                    edge,
                    dest,
                    &self.arena[parent].state,
                    self.cache.as_ref().expect("initialized cache"),
                )?
                else {
                    self.stats.rejected_edges += 1;
                    return Ok(true);
                };
                accepted = self.accept(
                    adapter,
                    parent,
                    EdgeEval {
                        edge,
                        dest,
                        state,
                        stop,
                    },
                    &mut paths,
                )?;
                Ok(accepted.is_none() && paths.len() < limit)
            })?;
            work += consumed;
            self.frames.last_mut().expect("active frame").next_edge += consumed;
            if let Some(child) = accepted {
                let dest = self.arena[child].node;
                self.visits.increment(dest);
                let edge_count = if self.arena[child].depth >= self.cfg.max_depth {
                    0
                } else {
                    adapter.prefetch_outgoing(&[dest])?;
                    adapter.out_degree(dest)?
                };
                self.frames.push(DfsFrame {
                    next_edge: 0,
                    edge_count,
                });
            }
        }
        Ok(paths)
    }
    fn advance_bfs<A: SearchAdapter<State = S, Cache = C>, const TIMED: bool>(
        &mut self,
        adapter: &A,
        limit: usize,
    ) -> Result<Vec<A::Path>> {
        let mut paths = Vec::new();
        let mut work = 0;
        let mut checkpoint = if TIMED { 1 } else { WORK_QUANTUM };
        while work < WORK_QUANTUM && paths.len() < limit {
            if TIMED && work == checkpoint {
                self.cfg.deadline.check()?;
                checkpoint = WORK_QUANTUM.min(work + Deadline::POLL_INTERVAL);
            }
            let Some(&parent) = self.frontier.front() else {
                self.done = true;
                break;
            };
            if self.bfs_prefetched == 0 {
                let depth = self.arena[parent].depth;
                // A BFS frontier has no shallower entries after this one.
                if depth >= self.cfg.max_depth {
                    self.done = true;
                    break;
                }
                let nodes = self
                    .frontier
                    .iter()
                    .take(WORK_QUANTUM)
                    .take_while(|&&p| self.arena[p].depth == depth)
                    .map(|&p| self.arena[p].node)
                    .collect::<Vec<_>>();
                self.bfs_prefetched = nodes.len();
                adapter.prefetch_outgoing(&nodes)?;
            }
            let node = self.arena[parent].node;
            let count = match self.bfs_count {
                Some(count) => count,
                None => {
                    let count = adapter.out_degree(node)?;
                    self.bfs_count = Some(count);
                    count
                }
            };
            if self.bfs_edge == count {
                self.frontier.pop_front();
                if !self.cfg.intermediate_states && adapter.can_release_expanded_states() {
                    adapter.release_expanded_state(&mut self.arena[parent].state);
                }
                self.bfs_edge = 0;
                self.bfs_count = None;
                self.bfs_prefetched -= 1;
                work += 1;
                continue;
            }
            let end = count.min(self.bfs_edge + (checkpoint - work));
            let mut consumed = 0;
            adapter.for_each_range(node, self.bfs_edge, end, |edge, dest| {
                consumed += 1;
                let edge = eval_arena_edge(
                    adapter,
                    &self.arena,
                    parent,
                    (edge, dest),
                    self.cfg.max_visits_per_node,
                    &mut self.stats,
                    self.cache.as_ref().expect("initialized cache"),
                )?;
                if let Some(edge) = edge
                    && let Some(child) = self.accept(adapter, parent, edge, &mut paths)?
                {
                    self.frontier.push_back(child);
                }
                Ok(paths.len() < limit)
            })?;
            work += consumed;
            self.bfs_edge += consumed;
        }
        Ok(paths)
    }
    pub(crate) fn next_batch<A: SearchAdapter<State = S, Cache = C>>(
        &mut self,
        adapter: &A,
        size: usize,
    ) -> Result<Option<Vec<A::Path>>> {
        if size == 0 {
            self.close();
            bail!("batch_size must be at least 1");
        }
        let result = (|| {
            loop {
                if self.done {
                    return Ok(None);
                }
                let paths = self.advance(adapter, size)?;
                if !paths.is_empty() {
                    return Ok(Some(paths));
                }
            }
        })();
        if result.is_err() || self.done {
            self.close();
        }
        result
    }
}

mod parallel;
pub(crate) use parallel::GraphCursor;
