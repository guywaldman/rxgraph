//! Resumable traversal state. Adapters are borrowed only while advancing a cursor.
use super::*;

const WORK_QUANTUM: usize = 8192;
const DFS_SCAN_DEPTH: usize = 24;

pub(crate) struct Cursor<S, C> {
    cfg: RunConfig,
    arena: Vec<PathEntry<S>>,
    frames: Vec<DfsFrame>,
    frontier: VecDeque<usize>,
    bfs_slots: BfsSlots,
    bfs_edge: usize,
    bfs_count: Option<usize>,
    bfs_prefetched: usize,
    progress: Option<Progress>,
    next_start: usize,
    visits: Option<SparseVisits>,
    cache: Option<C>,
    initialized: bool,
    done: bool,
    pub(crate) stats: SearchStats,
}

/// The frontier and each child entry hold one reference to a path slot.
/// Released slots form a freelist through their former parent links.
#[derive(Default)]
struct BfsSlots {
    owners: Vec<usize>,
    free: Option<usize>,
}

impl BfsSlots {
    fn insert<S>(&mut self, arena: &mut Vec<PathEntry<S>>, entry: PathEntry<S>) -> usize {
        let parent = entry.parent.expect("BFS continuation has a parent");
        self.owners[parent] += 1;
        if let Some(slot) = self.free {
            self.free = arena[slot].parent;
            arena[slot] = entry;
            self.owners[slot] = 1;
            slot
        } else {
            let slot = arena.len();
            arena.push(entry);
            self.owners.push(1);
            slot
        }
    }

    fn release<S>(&mut self, arena: &mut [PathEntry<S>], mut path: usize) {
        loop {
            self.owners[path] -= 1;
            if self.owners[path] != 0 {
                break;
            }
            arena[path].state = None;
            let parent = std::mem::replace(&mut arena[path].parent, self.free);
            self.free = Some(path);
            let Some(parent) = parent else { break };
            path = parent;
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
            bfs_slots: BfsSlots::default(),
            bfs_edge: 0,
            bfs_count: None,
            bfs_prefetched: 0,
            progress,
            next_start: 0,
            visits: None,
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
        self.bfs_slots = BfsSlots::default();
        self.visits = None;
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
        if matches!(self.cfg.strategy, TraversalStrategy::BreadthFirst) {
            let (arena, frontier, stats) = initial_arena(adapter, &self.cfg)?;
            self.bfs_slots.owners = vec![1; arena.len()];
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
            state: Some(adapter.initial_state(node)?),
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
    fn accept<A: SearchAdapter<State = S, Cache = C>, const BFS: bool>(
        &mut self,
        adapter: &A,
        parent: usize,
        edge: EdgeEval<S>,
        paths: &mut Vec<A::Path>,
    ) -> Result<Option<usize>> {
        let depth = self.arena[parent].depth + 1;
        self.stats.accepted_edges += 1;
        self.stats.path_entries += 1;
        self.stats.max_depth = self.stats.max_depth.max(depth);
        if !edge.stop && depth >= self.cfg.max_depth {
            return Ok(None);
        }
        let stop = edge.stop;
        let entry = child_entry(&self.arena, parent, edge);
        if BFS && !stop {
            return Ok(Some(self.bfs_slots.insert(&mut self.arena, entry)));
        }
        let child = self.arena.len();
        self.arena.push(entry);
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
                if let Some(visits) = &mut self.visits {
                    visits.decrement(entry.node);
                    if self.arena.len() <= DFS_SCAN_DEPTH / 2 {
                        self.visits = None;
                    }
                }
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
                let can_visit = match &self.visits {
                    Some(visits) => visits.count(dest) < self.cfg.max_visits_per_node,
                    None => {
                        can_visit_arena(&self.arena, parent, dest, self.cfg.max_visits_per_node)
                    }
                };
                if !can_visit {
                    self.stats.skipped_revisits += 1;
                    return Ok(true);
                }
                self.stats.evaluated_edges += 1;
                let Some((state, stop)) = adapter.eval_edge(
                    node,
                    edge,
                    dest,
                    self.arena[parent].state(),
                    self.cache.as_ref().expect("initialized cache"),
                )?
                else {
                    self.stats.rejected_edges += 1;
                    return Ok(true);
                };
                accepted = self.accept::<_, false>(
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
                if let Some(visits) = &mut self.visits {
                    visits.increment(dest);
                } else if self.arena.len() > DFS_SCAN_DEPTH {
                    // Shallow searches use the path fingerprint and skip the map entirely.
                    // Track only active ancestors once scanning paths becomes expensive.
                    let mut visits = SparseVisits::default();
                    for entry in &self.arena {
                        visits.increment(entry.node);
                    }
                    self.visits = Some(visits);
                }
                adapter.prefetch_outgoing(&[dest])?;
                self.frames.push(DfsFrame {
                    next_edge: 0,
                    edge_count: adapter.out_degree(dest)?,
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
        let prefetches = adapter.prefetches_outgoing();
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
                if prefetches {
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
                    self.arena[parent].state = None;
                }
                self.bfs_slots.release(&mut self.arena, parent);
                self.bfs_edge = 0;
                self.bfs_count = None;
                self.bfs_prefetched -= usize::from(prefetches);
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
                    && let Some(child) =
                        self.accept::<_, true>(adapter, parent, edge, &mut paths)?
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
