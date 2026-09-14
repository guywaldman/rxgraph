use super::*;
type EvaluatedChunk<S> = (Vec<(usize, EdgeEval<S>)>, SearchStats);

/// Parallel DFS keeps one active stack and scratch cache per worker across pulls.
pub(crate) struct ParallelDfs<S, C> {
    workers: Vec<Cursor<S, C>>,
    max_paths: Option<usize>,
    emitted: usize,
    stats: SearchStats,
    progress: Option<Progress>,
}
impl<S: Clone, C> ParallelDfs<S, C> {
    fn new(mut run: RunOptions) -> Result<Self> {
        let max_paths = run.max_paths;
        let progress = Some(Progress::new(run.progress));
        let starts = std::mem::take(&mut run.start_nodes);
        let count = rayon::current_num_threads().min(starts.len()).max(1);
        run.max_paths = None;
        run.parallel = false;
        run.progress = false;
        let mut workers = (0..count)
            .map(|_| Cursor::new(run.clone()))
            .collect::<Result<Vec<_>>>()?;
        for (i, start) in starts.into_iter().enumerate() {
            workers[i % count].cfg.start_nodes.push(start);
        }
        Ok(Self {
            workers,
            max_paths,
            emitted: 0,
            stats: SearchStats::default(),
            progress,
        })
    }
    fn next_batch<A>(&mut self, adapter: &A, size: usize) -> Result<Option<Vec<A::Path>>>
    where
        A: SearchAdapter<State = S, Cache = C> + Sync,
        S: Send + Sync,
        C: Send,
        A::Path: Send,
    {
        loop {
            let remaining = self
                .max_paths
                .map_or(size, |n| n.saturating_sub(self.emitted).min(size));
            let active = self.workers.iter().filter(|w| !w.done).count();
            if active == 0 || remaining == 0 {
                return Ok(None);
            }
            let mut index = 0;
            let jobs = self
                .workers
                .iter_mut()
                .filter(|w| !w.done)
                .map(|w| {
                    let quota = remaining / active + usize::from(index < remaining % active);
                    index += 1;
                    (w, quota)
                })
                .collect::<Vec<_>>();
            let results = jobs
                .into_par_iter()
                .map(|(worker, quota)| {
                    if quota == 0 {
                        return Ok(Vec::new());
                    }
                    let before = worker.stats.evaluated_edges;
                    let result = worker.advance(adapter, quota);
                    worker.stats.parallel_edges += worker.stats.evaluated_edges - before;
                    result
                })
                .collect::<Result<Vec<_>>>();
            self.stats = SearchStats::default();
            for worker in &self.workers {
                merge_stats(&mut self.stats, worker.stats);
            }
            if let Some(progress) = &mut self.progress {
                progress.tick(&self.stats);
            }
            let paths = results?.into_iter().flatten().collect::<Vec<_>>();
            self.emitted += paths.len();
            if !paths.is_empty() {
                return Ok(Some(paths));
            }
        }
    }
}

pub(crate) enum GraphCursor<S, C> {
    Serial(Cursor<S, C>),
    ParallelDfs(ParallelDfs<S, C>),
    ParallelBfs(ParallelBfs<S, C>),
}
impl<S: Clone, C> GraphCursor<S, C> {
    pub(crate) fn new(run: RunOptions) -> Result<Self> {
        if run.max_visits_per_node == 0 {
            bail!("max_visits_per_node must be at least 1");
        }
        if run.parallel
            && matches!(run.strategy, TraversalStrategy::DepthFirst)
            && should_parallelize_dfs(&run)
        {
            Ok(Self::ParallelDfs(ParallelDfs::new(run)?))
        } else if run.parallel && matches!(run.strategy, TraversalStrategy::BreadthFirst) {
            Ok(Self::ParallelBfs(ParallelBfs {
                cursor: Cursor::new(run)?,
                pending: VecDeque::new(),
            }))
        } else {
            Ok(Self::Serial(Cursor::new(run)?))
        }
    }
    pub(crate) fn stats(&self) -> SearchStats {
        match self {
            Self::Serial(c) => c.stats,
            Self::ParallelDfs(c) => c.stats,
            Self::ParallelBfs(c) => c.cursor.stats,
        }
    }
    pub(crate) fn close(&mut self) {
        match self {
            Self::Serial(c) => c.close(),
            Self::ParallelDfs(c) => {
                c.workers = Vec::new();
                if let Some(progress) = c.progress.take() {
                    progress.finish(&c.stats);
                }
            }
            Self::ParallelBfs(c) => {
                c.cursor.close();
                c.pending = VecDeque::new();
            }
        }
    }
    pub(crate) fn next_batch<A>(&mut self, adapter: &A, size: usize) -> Result<Option<Vec<A::Path>>>
    where
        A: SearchAdapter<State = S, Cache = C> + Sync,
        S: Send + Sync,
        C: Send,
        A::Path: Send,
    {
        if size == 0 {
            self.close();
            bail!("batch_size must be at least 1");
        }
        let result = match self {
            Self::Serial(c) => c.next_batch(adapter, size),
            Self::ParallelDfs(c) => c.next_batch(adapter, size),
            Self::ParallelBfs(c) => c.next_batch(adapter, size),
        };
        if result.is_err() || matches!(result, Ok(None)) {
            self.close();
        }
        result
    }
}

pub(crate) struct ParallelBfs<S, C> {
    cursor: Cursor<S, C>,
    pending: VecDeque<(usize, EdgeEval<S>)>,
}
impl<S: Clone, C> ParallelBfs<S, C> {
    fn next_batch<A>(&mut self, adapter: &A, size: usize) -> Result<Option<Vec<A::Path>>>
    where
        A: SearchAdapter<State = S, Cache = C> + Sync,
        S: Send + Sync,
        C: Send,
        A::Path: Send,
    {
        if self.cursor.done {
            return Ok(None);
        }
        self.cursor.initialize(adapter)?;
        let c = &mut self.cursor;
        let mut paths = Vec::new();
        loop {
            if c.done || c.cfg.max_paths.is_some_and(|n| c.stats.stopped_paths >= n) {
                c.done = true;
                break;
            }
            while let Some((parent, edge)) = self.pending.pop_front() {
                let stop = edge.stop;
                let child = push_entry(&mut c.arena, parent, edge);
                c.stats.accepted_edges += 1;
                c.stats.path_entries += 1;
                c.stats.max_depth = c.stats.max_depth.max(c.arena[child].depth);
                if stop {
                    paths.push(adapter.materialize(&c.arena, child, c.cfg.intermediate_states)?);
                    c.stats.stopped_paths += 1;
                    c.arena.pop();
                    if paths.len() >= size
                        || c.cfg.max_paths.is_some_and(|n| c.stats.stopped_paths >= n)
                    {
                        break;
                    }
                } else {
                    c.frontier.push_back(child);
                }
            }
            if !paths.is_empty() {
                break;
            }
            let Some(&first) = c.frontier.front() else {
                c.done = true;
                break;
            };
            // Small frontiers use the same direct cursor as serial search. Recheck
            // after each bounded chunk so growing frontiers can still use workers.
            let depth = c.arena[first].depth;
            let mut frontier_edges = 0;
            if c.frontier.len() < MIN_PAR_FRONTIER {
                for &parent in &c.frontier {
                    if c.arena[parent].depth != depth || frontier_edges >= MIN_PAR_EDGES {
                        break;
                    }
                    frontier_edges += adapter.out_degree(c.arena[parent].node)?;
                }
            }
            if c.frontier.len() < MIN_PAR_FRONTIER && frontier_edges < MIN_PAR_EDGES {
                paths = c.advance_bfs(adapter, size)?;
                if let Some(progress) = &mut c.progress {
                    progress.tick(&c.stats);
                }
                if !paths.is_empty() {
                    break;
                }
                continue;
            }
            c.bfs_prefetched = 0;
            let mut jobs = Vec::new();
            let frontier_count = c.frontier.len();
            let nodes = c
                .frontier
                .iter()
                .take(WORK_QUANTUM)
                .filter(|&&p| c.arena[p].depth == depth && depth < c.cfg.max_depth)
                .map(|&p| c.arena[p].node)
                .collect::<Vec<_>>();
            adapter.prefetch_outgoing(&nodes)?;
            let mut edge_total = 0;
            while edge_total < WORK_QUANTUM {
                let Some(&parent) = c.frontier.front() else {
                    break;
                };
                if c.arena[parent].depth != depth {
                    break;
                }
                let count = match c.bfs_count {
                    Some(count) => count,
                    None => {
                        let count = if depth >= c.cfg.max_depth {
                            0
                        } else {
                            adapter.out_degree(c.arena[parent].node)?
                        };
                        c.bfs_count = Some(count);
                        count
                    }
                };
                if c.bfs_edge == count {
                    c.frontier.pop_front();
                    c.bfs_edge = 0;
                    c.bfs_count = None;
                    continue;
                }
                let end = count.min(
                    c.bfs_edge
                        + (WORK_QUANTUM - edge_total)
                            .min(WORK_QUANTUM.div_ceil(rayon::current_num_threads())),
                );
                jobs.push((parent, c.bfs_edge, end));
                edge_total += end - c.bfs_edge;
                c.bfs_edge = end;
            }
            let parallel = edge_total >= MIN_PAR_EDGES || frontier_count >= MIN_PAR_FRONTIER;
            let evaluate = |jobs: &[(usize, usize, usize)]| -> Result<EvaluatedChunk<S>> {
                let cache = adapter.make_cache();
                let mut stats = SearchStats::default();
                let mut edges = Vec::new();
                for &(parent, start, end) in jobs {
                    adapter.for_each_range(c.arena[parent].node, start, end, |edge, dest| {
                        if let Some(edge) = eval_arena_edge(
                            adapter,
                            &c.arena,
                            parent,
                            (edge, dest),
                            &c.cfg,
                            &mut stats,
                            &cache,
                        )? {
                            edges.push((parent, edge));
                        }
                        Ok(true)
                    })?;
                }
                if parallel {
                    stats.parallel_edges = stats.evaluated_edges;
                }
                Ok((edges, stats))
            };
            let results = if parallel {
                let chunk = jobs.len().div_ceil(rayon::current_num_threads()).max(1);
                jobs.par_chunks(chunk)
                    .map(evaluate)
                    .collect::<Result<Vec<_>>>()?
            } else {
                vec![evaluate(&jobs)?]
            };
            for (edges, stats) in results {
                self.pending.extend(edges);
                merge_stats(&mut c.stats, stats);
            }
            if let Some(progress) = &mut c.progress {
                progress.tick(&c.stats);
            }
        }
        if paths.is_empty() {
            Ok(None)
        } else {
            Ok(Some(paths))
        }
    }
}
