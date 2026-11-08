//! Graph-local, type-safe storage for derived data.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt,
    sync::{Arc, Condvar, Mutex},
};

use anyhow::Result;

type CachedValue = Arc<dyn Any + Send + Sync>;
type Entry = Arc<CacheEntry>;
type Entries = HashMap<TypeId, HashMap<String, Entry>>;

struct CacheEntry {
    state: Mutex<EntryState>,
    ready: Condvar,
}

enum EntryState {
    Initializing,
    Ready(CachedValue),
    Failed(Arc<anyhow::Error>),
}

#[derive(Debug)]
struct SharedError(Arc<anyhow::Error>);

impl fmt::Display for SharedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for SharedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref().as_ref())
    }
}

/// The map lock is held only during entry lookup/removal, so expensive
/// initializers for different keys do not block each other.
#[derive(Default)]
pub(crate) struct DerivedCache {
    entries: Mutex<Entries>,
}

impl fmt::Debug for DerivedCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DerivedCache").finish_non_exhaustive()
    }
}

impl DerivedCache {
    pub(crate) fn get<T: Send + Sync + 'static>(
        &self,
        key: &str,
        init: impl FnOnce() -> Result<T>,
    ) -> Result<Arc<T>> {
        let (entry, initialize) = {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let by_key = entries.entry(TypeId::of::<T>()).or_default();
            if let Some(entry) = by_key.get(key) {
                (Arc::clone(entry), false)
            } else {
                let entry = Arc::new(CacheEntry {
                    state: Mutex::new(EntryState::Initializing),
                    ready: Condvar::new(),
                });
                by_key.insert(key.to_owned(), Arc::clone(&entry));
                (entry, true)
            }
        };

        if initialize {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(init));
            match result {
                Ok(Ok(value)) => {
                    let value = Arc::new(value);
                    let mut state = entry.state.lock().unwrap_or_else(|e| e.into_inner());
                    *state = EntryState::Ready(value.clone());
                    entry.ready.notify_all();
                    return Ok(value);
                }
                Ok(Err(error)) => {
                    let error = Arc::new(error);
                    self.fail::<T>(key, &entry, Arc::clone(&error));
                    return Err(anyhow::Error::new(SharedError(error)));
                }
                Err(panic) => {
                    self.fail::<T>(
                        key,
                        &entry,
                        Arc::new(anyhow::anyhow!("cache initializer panicked")),
                    );
                    std::panic::resume_unwind(panic);
                }
            }
        }
        let mut state = entry.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match &*state {
                EntryState::Initializing => {
                    state = entry.ready.wait(state).unwrap_or_else(|e| e.into_inner());
                }
                EntryState::Ready(value) => {
                    // Entries are keyed by TypeId, so only this type can occupy it.
                    return Ok(Arc::clone(value)
                        .downcast::<T>()
                        .expect("cache entry type matches its key"));
                }
                EntryState::Failed(error) => {
                    return Err(anyhow::Error::new(SharedError(Arc::clone(error))));
                }
            }
        }
    }

    fn fail<T: Send + Sync + 'static>(&self, key: &str, entry: &Entry, error: Arc<anyhow::Error>) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(by_key) = entries.get_mut(&TypeId::of::<T>()) {
            if by_key
                .get(key)
                .is_some_and(|existing| Arc::ptr_eq(existing, entry))
            {
                by_key.remove(key);
            }
            if by_key.is_empty() {
                entries.remove(&TypeId::of::<T>());
            }
        }
        let mut state = entry.state.lock().unwrap_or_else(|e| e.into_inner());
        *state = EntryState::Failed(error);
        entry.ready.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::{
        any::TypeId,
        sync::{
            Arc, Barrier, Condvar, Mutex,
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
        time::{Duration, Instant},
    };

    use anyhow::{Result, anyhow};
    use arrow::array::record_batch;

    use crate::Graph;

    fn graph() -> Result<Graph> {
        Graph::new(
            record_batch!(("id", UInt64, [0, 1]), ("value", UInt64, [10, 20]))?,
            record_batch!(
                ("id", UInt64, [0]),
                ("src", UInt64, [0]),
                ("dest", UInt64, [1])
            )?,
        )
    }

    #[test]
    fn cache_is_keyed_by_type_and_key_and_scoped_to_the_graph() -> Result<()> {
        let a = graph()?;
        let b = graph()?;
        let first = a.cached("same", || Ok(41_u64))?;
        let repeated = a.cached("same", || Ok(99_u64))?;
        assert!(Arc::ptr_eq(&first, &repeated));
        assert_eq!(*repeated, 41);
        assert_eq!(*a.cached("different", || Ok(42_u64))?, 42);
        assert_eq!(*a.cached("same", || Ok(String::from("value")))?, "value");
        assert!(!Arc::ptr_eq(&first, &b.cached("same", || Ok(41_u64))?));

        let weak = Arc::downgrade(&first);
        drop(first);
        drop(repeated);
        assert!(weak.upgrade().is_some());
        drop(a);
        assert!(weak.upgrade().is_none());
        Ok(())
    }

    #[test]
    fn concurrent_same_key_initializes_once() -> Result<()> {
        let graph = Arc::new(graph()?);
        let starts = Arc::new(Barrier::new(12));
        let calls = AtomicUsize::new(0);
        let values = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..12 {
                let graph = Arc::clone(&graph);
                let starts = Arc::clone(&starts);
                let calls = &calls;
                handles.push(scope.spawn(move || {
                    starts.wait();
                    graph.cached("shared", || {
                        calls.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_millis(10));
                        Ok(vec![1_u64, 2])
                    })
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Result<Vec<_>>>()
        })?;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(values.iter().all(|value| Arc::ptr_eq(value, &values[0])));
        Ok(())
    }

    #[test]
    fn concurrent_failure_is_shared_and_next_call_retries() -> Result<()> {
        const WORKERS: usize = 8;
        let graph = Arc::new(graph()?);
        let ready = Arc::new(Barrier::new(WORKERS + 1));
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let calls = AtomicUsize::new(0);
        std::thread::scope(|scope| -> Result<()> {
            let mut handles = Vec::new();
            for _ in 0..WORKERS {
                let graph = Arc::clone(&graph);
                let ready = Arc::clone(&ready);
                let gate = Arc::clone(&gate);
                let calls = &calls;
                handles.push(scope.spawn(move || {
                    ready.wait();
                    graph.cached::<u64>("failure", || {
                        calls.fetch_add(1, Ordering::SeqCst);
                        let (lock, cv) = &*gate;
                        let mut released = lock.lock().unwrap();
                        while !*released {
                            released = cv.wait(released).unwrap();
                        }
                        Err(anyhow!("temporary failure").context("cache build"))
                    })
                }));
            }
            ready.wait();
            let deadline = Instant::now() + Duration::from_secs(5);
            let joined = loop {
                let entries = graph.repo.derived_cache.entries.lock().unwrap();
                let count = entries
                    .get(&TypeId::of::<u64>())
                    .and_then(|by_key| by_key.get("failure"))
                    .map_or(0, Arc::strong_count);
                drop(entries);
                if count == WORKERS + 1 {
                    break true;
                }
                if Instant::now() >= deadline {
                    break false;
                }
                std::thread::sleep(Duration::from_millis(1));
            };
            let (lock, cv) = &*gate;
            *lock.lock().unwrap() = true;
            cv.notify_all();
            let values = handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>();
            assert!(joined, "all callers should join the first initialization");
            let first = format!("{:#}", values[0].as_ref().unwrap_err());
            assert!(first.contains("cache build") && first.contains("temporary failure"));
            assert!(
                values
                    .iter()
                    .all(|value| { value.as_ref().is_err_and(|e| format!("{e:#}") == first) })
            );
            Ok(())
        })?;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(*graph.cached("failure", || Ok(9_u64))?, 9);
        Ok(())
    }

    #[test]
    fn errors_and_panics_do_not_poison_entries() -> Result<()> {
        let graph = graph()?;
        assert!(
            graph
                .cached::<u32>("error", || Err(anyhow!("not ready")))
                .is_err()
        );
        assert_eq!(*graph.cached("error", || Ok(17_u32))?, 17);

        let original = graph
            .cached::<u32>("typed-error", || {
                Err(std::io::Error::other("failure").into())
            })
            .unwrap_err();
        assert!(original.chain().any(|cause| cause.is::<std::io::Error>()));

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = graph.cached::<u32>("panic", || panic!("initializer panicked"));
        }));
        assert_eq!(*graph.cached("panic", || Ok(21_u32))?, 21);
        Ok(())
    }

    #[test]
    fn initializer_panic_wakes_waiters_and_allows_retry() -> Result<()> {
        let graph = Arc::new(graph()?);
        let (entered, started) = mpsc::channel();
        let (unblock, proceed) = mpsc::channel();
        std::thread::scope(|scope| -> Result<()> {
            let first_graph = Arc::clone(&graph);
            let first = scope.spawn(move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = first_graph.cached::<u32>("panic-waiter", || {
                        entered.send(()).unwrap();
                        proceed.recv().unwrap();
                        panic!("initializer panicked")
                    });
                }))
            });
            started.recv_timeout(Duration::from_secs(2))?;
            let waiter_graph = Arc::clone(&graph);
            let waiter = scope.spawn(move || waiter_graph.cached("panic-waiter", || Ok(12_u32)));
            let deadline = Instant::now() + Duration::from_secs(5);
            let joined = loop {
                let entries = graph.repo.derived_cache.entries.lock().unwrap();
                let count = entries
                    .get(&TypeId::of::<u32>())
                    .and_then(|by_key| by_key.get("panic-waiter"))
                    .map_or(0, Arc::strong_count);
                drop(entries);
                if count == 3 {
                    break true;
                }
                if Instant::now() >= deadline {
                    break false;
                }
                std::thread::sleep(Duration::from_millis(1));
            };
            unblock.send(())?;
            assert!(first.join().unwrap().is_err());
            let result = waiter.join().unwrap();
            assert!(joined, "waiter should join the panicking initializer");
            assert!(result.unwrap_err().to_string().contains("panicked"));
            Ok(())
        })?;
        assert_eq!(*graph.cached("panic-waiter", || Ok(13_u32))?, 13);
        Ok(())
    }

    #[test]
    fn different_keys_can_initialize_concurrently_and_reenter() -> Result<()> {
        let graph = Arc::new(graph()?);
        let (entered, started) = mpsc::channel();
        let (unblock, proceed) = mpsc::channel();
        std::thread::scope(|scope| -> Result<()> {
            let first_graph = Arc::clone(&graph);
            let first = scope.spawn(move || {
                first_graph.cached("first", || -> Result<u32> {
                    entered.send(())?;
                    proceed.recv_timeout(Duration::from_secs(2))?;
                    Ok(1)
                })
            });
            started.recv_timeout(Duration::from_secs(2))?;
            let second_graph = Arc::clone(&graph);
            let second = scope.spawn(move || {
                second_graph.cached("second", || -> Result<u32> {
                    unblock.send(())?;
                    Ok(2)
                })
            });
            assert_eq!(*first.join().unwrap()?, 1);
            assert_eq!(*second.join().unwrap()?, 2);
            Ok(())
        })?;
        assert_eq!(
            *graph.cached("outer", || Ok(*graph.cached("inner", || Ok(4_u32))? + 1))?,
            5
        );
        Ok(())
    }

    #[test]
    fn replacing_payloads_preserves_topology_cache() -> Result<()> {
        let mut graph = graph()?;
        let cached = graph.cached("data-v1", || Ok(5_u32))?;
        let invalid_nodes = record_batch!(("value", UInt64, [30]))?;
        let edges = record_batch!(("cost", UInt64, [1]))?;
        assert!(graph.set_payloads(invalid_nodes, edges.clone()).is_err());
        let nodes = record_batch!(("value", UInt64, [30, 40]))?;
        graph.set_payloads(nodes, edges)?;
        assert!(Arc::ptr_eq(
            &cached,
            &graph.cached("data-v1", || Ok(6_u32))?
        ));
        assert_eq!(*graph.cached("data-v2", || Ok(6_u32))?, 6);
        Ok(())
    }
}
