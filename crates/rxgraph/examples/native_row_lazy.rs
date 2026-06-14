use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

use anyhow::{Context, Result};
use rxgraph::{
    GraphId, NodeId, RunOptions, TraversalStrategy, search_native,
    traversal::native::{self, GraphStore, OutgoingEdge},
};

fn main() -> Result<()> {
    let store = FraudStore::sample();
    let result = run_search(&store, false)?;

    println!(
        "paths={} evaluated_edges={} loaded_nodes={:?} loaded_edges={:?} loaded_outgoing={:?}",
        result.paths.len(),
        result.stats.evaluated_edges,
        store.loaded_nodes(),
        store.loaded_edges(),
        store.loaded_outgoing(),
    );
    for path in result.paths {
        let labels = path
            .nodes
            .iter()
            .map(|node| node.payload.label)
            .collect::<Vec<_>>();
        println!(
            "path labels={labels:?} risk={} checkpoints={:?}",
            path.state.total_risk, path.state.checkpoints
        );
    }

    Ok(())
}

fn run_search(
    store: &FraudStore,
    intermediate_states: bool,
) -> Result<native::SearchResult<'_, Account, Transfer, RiskState>> {
    search_native(
        store,
        RiskKernel {
            max_risk: 7,
            require_checkpoint: true,
        },
        RunOptions {
            start_nodes: vec![0_u64.into()],
            strategy: TraversalStrategy::BreadthFirst,
            intermediate_states,
            ..RunOptions::default()
        },
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Account {
    label: &'static str,
    target: bool,
    blocked: bool,
    checkpoint: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Transfer {
    risk: u64,
    allowed: bool,
}

#[derive(Clone, Debug)]
struct TransferRow {
    src: NodeId,
    dest: NodeId,
    transfer: Transfer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RiskState {
    total_risk: u64,
    visited: BTreeSet<NodeId>,
    checkpoints: BTreeMap<NodeId, &'static str>,
}

#[derive(Clone, Debug)]
struct RiskKernel {
    max_risk: u64,
    require_checkpoint: bool,
}

impl native::Kernel for RiskKernel {
    type Node = Account;
    type Edge = Transfer;
    type State = RiskState;

    fn initial_state(
        &self,
        cx: &native::StartCtx<'_, Self::Node, Self::Edge>,
    ) -> Result<Self::State> {
        let account = cx.node()?;
        let mut visited = BTreeSet::new();
        visited.insert(cx.id());
        let mut checkpoints = BTreeMap::new();
        if account.checkpoint {
            checkpoints.insert(cx.id(), account.label);
        }
        Ok(RiskState {
            total_risk: 0,
            visited,
            checkpoints,
        })
    }

    fn visit(&self, cx: &native::EdgeCtx<'_, Self::Node, Self::Edge, Self::State>) -> Result<bool> {
        let transfer = cx.edge()?;
        let dest = cx.dest()?;
        Ok(transfer.allowed
            && !dest.blocked
            && cx.state().total_risk.saturating_add(transfer.risk) <= self.max_risk)
    }

    fn next_state(
        &self,
        cx: &native::EdgeCtx<'_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<Self::State> {
        let transfer = cx.edge()?;
        let dest = cx.dest()?;
        let mut next = cx.state().clone();
        next.total_risk += transfer.risk;
        next.visited.insert(cx.dest_id());
        if dest.checkpoint {
            next.checkpoints.insert(cx.dest_id(), dest.label);
        }
        Ok(next)
    }

    fn stop(&self, cx: &native::EdgeCtx<'_, Self::Node, Self::Edge, Self::State>) -> Result<bool> {
        Ok(cx.dest()?.target && (!self.require_checkpoint || !cx.state().checkpoints.is_empty()))
    }
}

struct FraudStore {
    accounts: Vec<Account>,
    transfers: Vec<TransferRow>,
    outgoing_index: Vec<Vec<usize>>,
    account_cache: Vec<OnceLock<Account>>,
    transfer_cache: Vec<OnceLock<Transfer>>,
    outgoing_cache: Vec<OnceLock<Vec<OutgoingEdge>>>,
    account_loads: RefCell<BTreeSet<NodeId>>,
    transfer_loads: RefCell<BTreeSet<u32>>,
    outgoing_loads: RefCell<BTreeSet<NodeId>>,
}

impl FraudStore {
    fn sample() -> Self {
        let accounts = vec![
            Account {
                label: "origin",
                target: false,
                blocked: false,
                checkpoint: false,
            },
            Account {
                label: "merchant",
                target: false,
                blocked: false,
                checkpoint: true,
            },
            Account {
                label: "sink",
                target: true,
                blocked: false,
                checkpoint: false,
            },
            Account {
                label: "cold-wallet",
                target: false,
                blocked: false,
                checkpoint: false,
            },
            Account {
                label: "blocked",
                target: true,
                blocked: true,
                checkpoint: false,
            },
        ];
        let transfers = vec![
            TransferRow {
                src: 0,
                dest: 1,
                transfer: Transfer {
                    risk: 2,
                    allowed: true,
                },
            },
            TransferRow {
                src: 1,
                dest: 2,
                transfer: Transfer {
                    risk: 4,
                    allowed: true,
                },
            },
            TransferRow {
                src: 1,
                dest: 4,
                transfer: Transfer {
                    risk: 1,
                    allowed: true,
                },
            },
            TransferRow {
                src: 3,
                dest: 2,
                transfer: Transfer {
                    risk: 1,
                    allowed: true,
                },
            },
        ];
        Self::new(accounts, transfers)
    }

    fn new(accounts: Vec<Account>, transfers: Vec<TransferRow>) -> Self {
        let mut outgoing_index = vec![Vec::new(); accounts.len()];
        for (edge, transfer) in transfers.iter().enumerate() {
            outgoing_index[transfer.src as usize].push(edge);
        }
        let account_cache = (0..accounts.len()).map(|_| OnceLock::new()).collect();
        let transfer_cache = (0..transfers.len()).map(|_| OnceLock::new()).collect();
        let outgoing_cache = (0..accounts.len()).map(|_| OnceLock::new()).collect();
        Self {
            accounts,
            transfers,
            outgoing_index,
            account_cache,
            transfer_cache,
            outgoing_cache,
            account_loads: RefCell::new(BTreeSet::new()),
            transfer_loads: RefCell::new(BTreeSet::new()),
            outgoing_loads: RefCell::new(BTreeSet::new()),
        }
    }

    fn loaded_nodes(&self) -> BTreeSet<NodeId> {
        self.account_loads.borrow().clone()
    }

    fn loaded_edges(&self) -> BTreeSet<u32> {
        self.transfer_loads.borrow().clone()
    }

    fn loaded_outgoing(&self) -> BTreeSet<NodeId> {
        self.outgoing_loads.borrow().clone()
    }
}

impl GraphStore for FraudStore {
    type Node = Account;
    type Edge = Transfer;

    fn resolve_node(&self, external: GraphId<'_>) -> Result<Option<NodeId>> {
        Ok(match external {
            GraphId::U64(value) if (value as usize) < self.accounts.len() => Some(value as NodeId),
            _ => None,
        })
    }

    fn external_node(&self, internal: NodeId) -> Result<Option<GraphId<'_>>> {
        Ok(((internal as usize) < self.accounts.len()).then_some(GraphId::U64(internal as u64)))
    }

    fn external_edge(&self, internal: rxgraph::EdgeId) -> Result<Option<GraphId<'_>>> {
        Ok(((internal as usize) < self.transfers.len()).then_some(GraphId::U64(internal as u64)))
    }

    fn outgoing(&self, src: NodeId) -> Result<&[OutgoingEdge]> {
        let src_index = src as usize;
        let edge_ids = self
            .outgoing_index
            .get(src_index)
            .with_context(|| format!("node row {src} is out of range"))?;
        let outgoing = self
            .outgoing_cache
            .get(src_index)
            .context("outgoing cache row is missing")?
            .get_or_init(|| {
                self.outgoing_loads.borrow_mut().insert(src);
                edge_ids
                    .iter()
                    .map(|&edge| OutgoingEdge {
                        edge: edge as rxgraph::EdgeId,
                        dest: self.transfers[edge].dest,
                    })
                    .collect()
            });
        Ok(outgoing)
    }

    fn node(&self, id: NodeId) -> Result<&Self::Node> {
        let index = id as usize;
        let account = self
            .accounts
            .get(index)
            .with_context(|| format!("node row {id} is out of range"))?;
        Ok(self
            .account_cache
            .get(index)
            .context("account cache row is missing")?
            .get_or_init(|| {
                self.account_loads.borrow_mut().insert(id);
                account.clone()
            }))
    }

    fn edge(&self, id: rxgraph::EdgeId) -> Result<&Self::Edge> {
        let index = id as usize;
        let transfer = self
            .transfers
            .get(index)
            .with_context(|| format!("edge row {id} is out of range"))?;
        Ok(self
            .transfer_cache
            .get(index)
            .context("transfer cache row is missing")?
            .get_or_init(|| {
                self.transfer_loads.borrow_mut().insert(id);
                transfer.transfer.clone()
            }))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn native_example_returns_native_state_and_payloads() {
        let store = FraudStore::sample();
        let result = run_search(&store, true).unwrap();

        assert_eq!(result.paths.len(), 1);
        let path = &result.paths[0];
        assert_eq!(
            path.nodes
                .iter()
                .map(|node| node.payload.label)
                .collect::<Vec<_>>(),
            vec!["origin", "merchant", "sink"]
        );
        assert_eq!(path.state.total_risk, 6);
        assert_eq!(path.state.visited, BTreeSet::from([0, 1, 2]));
        assert_eq!(path.state.checkpoints, BTreeMap::from([(1, "merchant")]));
        assert_eq!(
            path.nodes
                .iter()
                .map(|node| node.state.as_ref().unwrap().total_risk)
                .collect::<Vec<_>>(),
            vec![0, 2, 6]
        );
    }

    #[test]
    fn native_example_loads_only_reached_rows() {
        let store = FraudStore::sample();
        let result = run_search(&store, false).unwrap();

        assert_eq!(result.stats.evaluated_edges, 3);
        assert_eq!(store.loaded_outgoing(), BTreeSet::from([0, 1]));
        assert_eq!(store.loaded_nodes(), BTreeSet::from([0, 1, 2, 4]));
        assert_eq!(store.loaded_edges(), BTreeSet::from([0, 1, 2]));
    }
}
