//! Example native traversal kernel for `rxgraph`.
//!
//! This crate shows the **static plugin path** (Model A): you write a Rust crate
//! that depends on `rxgraph`, implement the [`rxgraph::Kernel`] trait, and
//! register it under a name with `inventory::submit!`. Because your kernel is
//! compiled together with the engine, every per-edge call is statically
//! dispatched - there is no per-edge virtual dispatch.
//!
//! Once registered, the kernel is reachable by name through
//! [`rxgraph::build_kernel`] (and, via the parallel Python work, through
//! `graph.search(kernel="hop_budget", params={...})`).
//!
//! The kernel implemented here is [`HopBudget`]: starting from a node, walk the
//! graph and emit a path as soon as it reaches a node whose boolean payload
//! column (named by `target_col`) is `true`, or once it has taken `max_hops`
//! edges. It demonstrates:
//!
//! - carrying typed per-path state (`hops: u64`),
//! - reading a node payload column with a typed [`EdgeCtx`] getter
//!   ([`EdgeCtx::dest_bool`]),
//! - parsing a JSON params blob into kernel configuration,
//! - registering the kernel by name with `inventory::submit!`.

use anyhow::{Context, Result};
use rxgraph::{EdgeCtx, Graph, Kernel, NodeId, StateRow, Value};

/// Per-path state carried by [`HopBudget`].
///
/// Kept deliberately tiny and `Copy`; the engine clones state for every child
/// path, so cheap state keeps search fast.
#[derive(Clone, Copy, Debug, Default)]
pub struct HopState {
    /// Number of edges accepted so far on this path.
    hops: u64,
}

/// Stop after `max_hops` edges, or earlier at a node flagged by `target_col`.
///
/// `target_col` names a boolean node payload column; the first destination node
/// where that column is `true` ends (and emits) the path.
#[derive(Clone, Debug)]
pub struct HopBudget {
    /// Maximum number of edges any emitted path may contain.
    pub max_hops: u64,
    /// Name of the boolean node column that marks a target node.
    pub target_col: String,
}

impl HopBudget {
    /// Builds a [`HopBudget`] from a JSON params object.
    ///
    /// Expected shape: `{ "max_hops": <u64>, "target_col": "<column>" }`.
    pub fn from_params(params: &serde_json::Value) -> Result<Self> {
        let max_hops = params
            .get("max_hops")
            .and_then(serde_json::Value::as_u64)
            .context("param `max_hops` (u64) is required")?;
        let target_col = params
            .get("target_col")
            .and_then(serde_json::Value::as_str)
            .context("param `target_col` (string) is required")?
            .to_string();
        Ok(Self {
            max_hops,
            target_col,
        })
    }
}

impl Kernel for HopBudget {
    type State = HopState;

    fn initial_state(&self, _graph: &Graph, _start: NodeId) -> Self::State {
        HopState::default()
    }

    fn visit(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<bool> {
        // Accept an edge only while we still have hop budget left. `cx.state()`
        // is the *parent* path's state, so accepting this edge would make
        // `hops + 1` edges - reject once that would exceed `max_hops`.
        Ok(cx.state().hops < self.max_hops)
    }

    fn next_state(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Self::State> {
        Ok(HopState {
            hops: cx.state().hops + 1,
        })
    }

    fn stop(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<bool> {
        // `cx` here carries the *child* state produced by `next_state`.
        // Emit the path if we reached the hop budget...
        if cx.state().hops >= self.max_hops {
            return Ok(true);
        }
        // ...or if the destination node is flagged as a target. A missing/null
        // value reads as `false`.
        Ok(cx.dest_bool(&self.target_col)?.unwrap_or(false))
    }

    fn state_row(&self, state: &Self::State) -> StateRow {
        vec![("hops".to_string(), Value::U64(state.hops))]
    }
}

// Register the kernel under the name "hop_budget" so `rxgraph::build_kernel`
// (and the Python `kernel="hop_budget"` selector) can find it at link time.
// `inventory` is re-exported by `rxgraph`, so this crate needs no direct
// `inventory` dependency.
rxgraph::inventory::submit! {
    rxgraph::KernelEntry {
        name: "hop_budget",
        make: |params| Ok(rxgraph::boxed_run(HopBudget::from_params(params)?)),
    }
}

/// Optional Python wiring (enabled with `--features python`).
///
/// Builds a `cdylib` Python extension module named `rxgraph_kernel_example`.
/// Importing it from Python is enough to pull this object file in and run the
/// `inventory::submit!` registration above, so the kernel becomes selectable by
/// name. See the crate README for the maturin build.
#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    #[pymodule]
    fn rxgraph_kernel_example(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
        // No symbols to export: importing the module is what registers the
        // kernel (via the `inventory::submit!` above).
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::record_batch;
    use rxgraph::{Graph, GraphId, RunOptions};

    // Required graph identity columns: nodes need `id`; edges need `id`, `src`,
    // `dest`. (These names are part of the public data model.)
    /// a -> b -> c -> d, with `target` true only at node `c`.
    fn line_graph() -> Graph {
        Graph::new(
            record_batch!(
                ("id", Utf8, ["a", "b", "c", "d"]),
                ("target", Boolean, [false, false, true, false])
            )
            .unwrap(),
            record_batch!(
                ("id", Utf8, ["ab", "bc", "cd"]),
                ("src", Utf8, ["a", "b", "c"]),
                ("dest", Utf8, ["b", "c", "d"])
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn run(graph: &Graph, params: serde_json::Value) -> rxgraph::SearchResult<'_> {
        // Resolve the kernel by name exactly like the Python selector does.
        let kernel = rxgraph::build_kernel("hop_budget", &params).unwrap();
        kernel
            .run(
                graph,
                RunOptions {
                    start_nodes: vec!["a".into()],
                    parallel: false,
                    ..RunOptions::default()
                },
            )
            .unwrap()
    }

    #[test]
    fn registered_under_its_name() {
        // build_kernel succeeds => inventory::submit! registration was linked in.
        assert!(rxgraph::build_kernel("hop_budget", &serde_json::json!({})).is_err());
        assert!(
            rxgraph::build_kernel(
                "hop_budget",
                &serde_json::json!({"max_hops": 3, "target_col": "target"})
            )
            .is_ok()
        );
    }

    #[test]
    fn stops_at_target_node_before_hop_budget() {
        let graph = line_graph();
        let result = run(
            &graph,
            serde_json::json!({"max_hops": 10, "target_col": "target"}),
        );

        // First emitted path ends at the target node `c` after 2 hops.
        assert_eq!(result.paths.len(), 1);
        let path = &result.paths[0];
        assert_eq!(
            path.nodes,
            vec![GraphId::Str("a"), GraphId::Str("b"), GraphId::Str("c")]
        );
        assert_eq!(path.edges, vec![GraphId::Str("ab"), GraphId::Str("bc")]);
        assert_eq!(path.state, vec![("hops".to_string(), rxgraph::Value::U64(2))]);
    }

    #[test]
    fn stops_at_hop_budget_when_no_target_reached() {
        let graph = line_graph();
        // Budget of 1 hop is exhausted at node `b`, before reaching target `c`.
        let result = run(
            &graph,
            serde_json::json!({"max_hops": 1, "target_col": "target"}),
        );

        assert_eq!(result.paths.len(), 1);
        let path = &result.paths[0];
        assert_eq!(path.nodes, vec![GraphId::Str("a"), GraphId::Str("b")]);
        assert_eq!(path.edges, vec![GraphId::Str("ab")]);
        assert_eq!(path.state, vec![("hops".to_string(), rxgraph::Value::U64(1))]);
    }
}
