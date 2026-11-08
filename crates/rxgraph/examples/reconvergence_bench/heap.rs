//! Shared state/kernel fixtures for the memory and latency experiments.
use std::{
    any::Any,
    cell::RefCell,
    collections::HashMap,
    hash::Hash,
    marker::PhantomData,
    sync::{Arc, atomic::Ordering},
};

use crate::fixture::{State, WorkKernel};
use anyhow::Result;
use rxgraph::{EdgeCtx, EdgeId, Graph, Kernel, NodeId, Transition};

pub(crate) trait Buffer: Clone + Eq + Hash + Send + Sync + 'static {
    fn new(values: Vec<u64>) -> Self;
    fn values(&self) -> &[u64];
}
impl Buffer for Vec<u64> {
    fn new(values: Vec<u64>) -> Self {
        values
    }
    fn values(&self) -> &[u64] {
        self
    }
}
impl Buffer for Arc<Vec<u64>> {
    fn new(values: Vec<u64>) -> Self {
        Arc::new(values)
    }
    fn values(&self) -> &[u64] {
        self
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct HeapState<B> {
    pub(crate) meta: State<0>,
    pub(crate) buffer: B,
}

pub(crate) struct HeapKernel<'a, B, const COUNT: bool> {
    pub(crate) inner: WorkKernel<'a, 0, COUNT>,
    pub(crate) words: usize,
    pub(crate) marker: PhantomData<B>,
}
impl<B: Buffer, const COUNT: bool> Kernel for HeapKernel<'_, B, COUNT> {
    type State = HeapState<B>;
    fn initial_state(&self, graph: &Graph, start: NodeId) -> Result<Self::State> {
        let meta = self.inner.initial_state(graph, start)?;
        let mut values = (0..self.words).map(|i| i as u64 + 19).collect::<Vec<_>>();
        values[0] = meta.score;
        Ok(HeapState {
            meta,
            buffer: B::new(values),
        })
    }
    fn transition(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Transition<Self::State>> {
        if COUNT {
            self.inner.calls.fetch_add(1, Ordering::Relaxed);
        }
        let step = cx.edge_field(&self.inner.case.step).unwrap();
        let choice = cx.edge_field(&self.inner.case.choice).unwrap();
        let mut meta = cx.state().meta;
        meta.depth += 1;
        meta.score = meta.score.wrapping_add(step + 1);
        if step < self.inner.case.route_bits * 2 && choice != 0 {
            meta.route = meta.route * 2 + choice;
        }
        let mut values = cx.state().buffer.values().to_vec();
        values[0] = meta.score;
        let state = HeapState {
            meta,
            buffer: B::new(values),
        };
        Ok(if cx.edge_field(&self.inner.case.terminal).unwrap() {
            Transition::Complete(state)
        } else {
            Transition::Continue(state)
        })
    }
}

type FlatCache<S> = HashMap<(EdgeId, S), Transition<S>>;
type EdgeCache<S> = HashMap<EdgeId, HashMap<S, Transition<S>>>;
thread_local! {
    static CACHE: RefCell<Option<Box<dyn Any>>> = const { RefCell::new(None) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CacheMode {
    None,
    CloneKey,
    Borrowed,
}
impl CacheMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::CloneKey => "clone-key",
            Self::Borrowed => "borrowed-key",
        }
    }
}

/// A per-thread, per-search cache for deterministic kernels.
pub(crate) struct Memo<K, const BORROWED: bool>(pub(crate) K);
impl<K: Kernel, const BORROWED: bool> Kernel for Memo<K, BORROWED>
where
    K::State: Eq + Hash + 'static,
{
    type State = K::State;
    fn initial_state(&self, graph: &Graph, start: NodeId) -> Result<Self::State> {
        self.0.initial_state(graph, start)
    }
    fn transition(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Transition<Self::State>> {
        CACHE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if BORROWED {
                let cache = slot.get_or_insert_with(|| Box::new(EdgeCache::<K::State>::new()));
                let cache = cache.downcast_mut::<EdgeCache<K::State>>().unwrap();
                let per_edge = cache.entry(cx.edge()).or_default();
                if let Some(value) = per_edge.get(cx.state()) {
                    return Ok(value.clone());
                }
                let value = self.0.transition(cx)?;
                per_edge.insert(cx.state().clone(), value.clone());
                Ok(value)
            } else {
                let cache = slot.get_or_insert_with(|| Box::new(FlatCache::<K::State>::new()));
                let cache = cache.downcast_mut::<FlatCache<K::State>>().unwrap();
                let key = (cx.edge(), cx.state().clone());
                if let Some(value) = cache.get(&key) {
                    return Ok(value.clone());
                }
                let value = self.0.transition(cx)?;
                cache.insert(key, value.clone());
                Ok(value)
            }
        })
    }
}

fn take_cache<S: 'static, const COUNT: bool>(mode: CacheMode) -> (usize, usize) {
    CACHE.with(|slot| {
        let Some(cache) = slot.borrow_mut().take() else {
            return (0, 0);
        };
        if !COUNT {
            drop(cache);
            return (0, 0);
        }
        match mode {
            CacheMode::None => unreachable!(),
            CacheMode::CloneKey => {
                let cache = cache.downcast::<FlatCache<S>>().unwrap();
                (cache.len(), cache.capacity())
            }
            CacheMode::Borrowed => {
                let cache = cache.downcast::<EdgeCache<S>>().unwrap();
                cache
                    .values()
                    .fold((0, 0), |(n, cap), m| (n + m.len(), cap + m.capacity()))
            }
        }
    })
}

/// Clear every physical worker's cache. Dropping the caches is included in timing.
pub(crate) fn clear<S: 'static, const COUNT: bool>(
    mode: CacheMode,
    parallel: bool,
) -> (usize, usize) {
    if mode == CacheMode::None {
        return (0, 0);
    }
    if parallel {
        rayon::broadcast(|_| take_cache::<S, COUNT>(mode))
            .into_iter()
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    } else {
        take_cache::<S, COUNT>(mode)
    }
}

/// Initialize worker-local cache and hashing runtime before measuring a search.
pub(crate) fn prime(parallel: bool) {
    let init = || {
        CACHE.with(|slot| assert!(slot.borrow().is_none()));
        std::hint::black_box(HashMap::<u8, u8>::new());
    };
    if parallel {
        rayon::broadcast(|_| init());
    } else {
        init();
    }
}
