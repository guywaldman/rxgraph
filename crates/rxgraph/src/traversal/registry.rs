//! Name -> kernel registry.
//!
//! Kernels are registered under a string name with a factory that parses a
//! JSON params blob into a runnable search. Registration has two paths:
//!
//! - **Link-time** via [`inventory`], so a plugin crate can `inventory::submit!`
//!   a [`KernelEntry`] and have it discovered automatically.
//! - **Runtime** via [`register_kernel`], intended for a future `dlopen` path.
//!
//! Both paths resolve to a [`BoxedRun`]: a single boxed seam invoked once per
//! whole search. The inner [`Graph::search_with`](crate::Graph::search_with)
//! call stays monomorphized, so there is no per-edge virtual dispatch.

use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use anyhow::{Result, anyhow, bail};

use crate::{
    Graph,
    traversal::{Kernel, RunOptions, SearchResult},
};

// Re-exported so plugin crates can `rxgraph::inventory::submit!` without adding
// their own `inventory` dependency.
pub use inventory;

/// Object-safe seam that runs a fully-constructed kernel against a graph.
///
/// Used internally as the boxed type behind [`BoxedRun`]. Implemented by the
/// closure produced by [`boxed_run`].
pub trait RunKernel: Send + Sync {
    /// Runs the kernel against `graph` with `run` options.
    fn run<'g>(&self, graph: &'g Graph, run: RunOptions) -> Result<SearchResult<'g>>;
}

impl<F> RunKernel for F
where
    F: for<'g> Fn(&'g Graph, RunOptions) -> Result<SearchResult<'g>> + Send + Sync,
{
    fn run<'g>(&self, graph: &'g Graph, run: RunOptions) -> Result<SearchResult<'g>> {
        self(graph, run)
    }
}

/// A boxed, name-erased runnable search.
///
/// One virtual call per whole search; the inner traversal is statically
/// dispatched over the concrete kernel.
pub type BoxedRun = Box<dyn RunKernel>;

/// Factory function: parses params into a runnable kernel.
pub type MakeFn = fn(&serde_json::Value) -> Result<BoxedRun>;

/// A link-time registered kernel factory.
///
/// Submit one with [`inventory::submit!`] so [`build_kernel`] can find it:
///
/// ```ignore
/// rxgraph::inventory::submit! {
///     rxgraph::KernelEntry {
///         name: "my_kernel",
///         make: |params| Ok(rxgraph::boxed_run(MyKernel::from_params(params)?)),
///     }
/// }
/// ```
pub struct KernelEntry {
    /// Unique name used to look the kernel up.
    pub name: &'static str,
    /// Factory that builds a runnable kernel from JSON params.
    pub make: MakeFn,
}

inventory::collect!(KernelEntry);

/// Wraps a concrete [`Kernel`] into a [`BoxedRun`].
///
/// The kernel is cloned for each search so the boxed closure can be reused.
pub fn boxed_run<K>(kernel: K) -> BoxedRun
where
    K: Kernel + Clone + Send + Sync + 'static,
    K::State: Send + Sync + Clone,
{
    struct Runner<K>(K);

    impl<K> RunKernel for Runner<K>
    where
        K: Kernel + Clone + Send + Sync + 'static,
        K::State: Send + Sync + Clone,
    {
        fn run<'g>(&self, graph: &'g Graph, run: RunOptions) -> Result<SearchResult<'g>> {
            graph.search_with(self.0.clone(), run)
        }
    }

    Box::new(Runner(kernel))
}

type RuntimeMap = HashMap<String, MakeFn>;

fn runtime_registry() -> &'static Mutex<RuntimeMap> {
    static REGISTRY: OnceLock<Mutex<RuntimeMap>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Registers a kernel factory at runtime.
///
/// Errors on a duplicate name. Runtime entries take precedence over link-time
/// [`inventory`] entries in [`build_kernel`].
pub fn register_kernel(name: impl Into<String>, make: MakeFn) -> Result<()> {
    let name = name.into();
    let mut registry = runtime_registry()
        .lock()
        .map_err(|_| anyhow!("kernel registry poisoned"))?;
    if registry.contains_key(&name) {
        bail!("kernel {name:?} is already registered");
    }
    registry.insert(name, make);
    Ok(())
}

/// Builds a runnable kernel by name from JSON params.
///
/// Looks up the runtime registry first, then link-time [`inventory`] entries.
/// Errors list the available kernel names when `name` is unknown.
pub fn build_kernel(name: &str, params: &serde_json::Value) -> Result<BoxedRun> {
    let runtime = {
        let registry = runtime_registry()
            .lock()
            .map_err(|_| anyhow!("kernel registry poisoned"))?;
        registry.get(name).copied()
    };
    if let Some(make) = runtime {
        return make(params);
    }

    for entry in inventory::iter::<KernelEntry> {
        if entry.name == name {
            return (entry.make)(params);
        }
    }

    bail!(
        "unknown kernel {name:?}; available kernels: {}",
        available_names().join(", ")
    )
}

fn available_names() -> Vec<String> {
    let mut names = Vec::new();
    if let Ok(registry) = runtime_registry().lock() {
        names.extend(registry.keys().cloned());
    }
    names.extend(
        inventory::iter::<KernelEntry>
            .into_iter()
            .map(|e| e.name.to_string()),
    );
    names.sort();
    names.dedup();
    names
}
