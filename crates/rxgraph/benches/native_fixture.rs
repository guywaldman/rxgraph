//! Shared frozen workloads for eager and streaming measurements.
use anyhow::Result;
use arrow::{
    array::UInt64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use rxgraph::{
    traversal::{RunTypedKernel, native},
    *,
};
use std::sync::Arc;
#[derive(Clone, serde::Serialize)]
pub(crate) struct State {
    spent: u64,
    history: Vec<u64>,
}
#[derive(Clone)]
pub(crate) struct Budget {
    budget: u64,
    target: u64,
    large: bool,
    cost: EdgeField<u64>,
}
impl Kernel for Budget {
    type State = State;
    fn initial_state(&self, _: &Graph, _: NodeId) -> Result<State> {
        Ok(self.initial())
    }
    fn transition(&self, cx: &EdgeCtx<'_, State>) -> Result<Transition<State>> {
        Ok(self.step(
            cx.state(),
            cx.edge_field(&self.cost).unwrap_or(0),
            cx.dest_id(),
        ))
    }
}
impl Budget {
    fn initial(&self) -> State {
        State {
            spent: 0,
            history: if self.large { vec![7; 256] } else { vec![] },
        }
    }
    fn step(&self, old: &State, cost: u64, dest: Option<GraphId<'_>>) -> Transition<State> {
        let spent = old.spent + cost;
        if spent > self.budget {
            return Transition::Reject;
        }
        let mut state = old.clone();
        state.spent = spent;
        if dest == Some(GraphId::U64(self.target)) {
            Transition::Complete(state)
        } else {
            Transition::Continue(state)
        }
    }
}
#[derive(Clone)]
pub(crate) struct Cost(u64);
impl TryFrom<ArrowRow<'_>> for Cost {
    type Error = anyhow::Error;
    fn try_from(row: ArrowRow<'_>) -> Result<Self> {
        Ok(Self(row.u64("cost")?.unwrap()))
    }
}
impl TypedKernel for Budget {
    type Node = ();
    type Edge = Cost;
    type State = State;
    fn edge_fields(&self) -> Vec<PayloadField> {
        vec![PayloadField::new("cost")]
    }
    fn initial_state(&self, _: &native::StartCtx<'_, (), Cost>) -> Result<State> {
        Ok(self.initial())
    }
    fn transition(
        &self,
        cx: &native::EdgeCtx<'_, '_, (), Cost, State>,
    ) -> Result<Transition<State>> {
        Ok(self.step(cx.state(), cx.edge()?.0, cx.dest_external_id()?))
    }
}
fn table(names: &[&str], columns: Vec<Vec<u64>>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(
            names
                .iter()
                .map(|n| Field::new(*n, DataType::UInt64, false))
                .collect::<Vec<_>>(),
        )),
        columns
            .into_iter()
            .map(|c| Arc::new(UInt64Array::from(c)) as _)
            .collect(),
    )
    .unwrap()
}

pub(crate) struct Case<'a> {
    pub name: String,
    pub mode: &'a str,
    pub graph: &'a Arc<Graph>,
    pub kernel: &'a Budget,
    pub typed: &'a dyn RunTypedKernel,
    pub cache: &'a TypedPayloadCache,
    pub paths: &'a ParquetPaths,
    pub run: RunOptions,
}
pub(crate) fn with_cases(mut measure: impl FnMut(Case<'_>) -> Result<()>) -> Result<()> {
    let cases = [
        ("first", 1, 2, 1),
        ("reject", 64, 2, 128),
        ("no_match", 32, 64, 1),
        ("cycle", 16, 128, 2),
        ("wide", 64, 3, 64),
        ("many", 2048, 2, 1),
        ("history", 64, 32, 1),
        ("large_state", 32, 32, 1),
    ];
    for (name, starts, depth, fanout) in cases {
        let terminal = starts * depth;
        let (mut src, mut dest, mut costs) = (vec![], vec![], vec![]);
        for start in 0..starts {
            for level in 0..depth {
                let node = start * depth + level;
                src.push(node);
                dest.push(if level == depth - 1 {
                    terminal
                } else {
                    node + 1
                });
                costs.push(1);
                for _ in 1..fanout {
                    src.push(node);
                    dest.push(if name == "cycle" {
                        start * depth
                    } else {
                        terminal
                    });
                    costs.push(if name == "cycle" || name == "wide" {
                        1
                    } else {
                        depth + 1
                    });
                }
            }
        }
        let nodes = table(&["id"], vec![(0..=terminal).collect()]);
        let edges = table(
            &["id", "src", "dest", "cost"],
            vec![(0..src.len() as u64).collect(), src, dest, costs],
        );
        let graph = Arc::new(Graph::new(nodes.clone(), edges.clone())?);
        let kernel = Budget {
            budget: if name == "reject" { 0 } else { depth },
            target: if name == "no_match" {
                terminal + 1
            } else {
                terminal
            },
            large: name == "large_state",
            cost: graph.edge_field("cost")?,
        };
        let dir = std::env::temp_dir().join(format!("rxgraph-native-bench-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let paths = ParquetPaths {
            nodes: dir.join("nodes.parquet"),
            edges: dir.join("edges.parquet"),
        };
        for (path, batch) in [(&paths.nodes, &nodes), (&paths.edges, &edges)] {
            let mut writer = parquet::arrow::ArrowWriter::try_new(
                std::fs::File::create(path)?,
                batch.schema(),
                None,
            )?;
            writer.write(batch)?;
            writer.close()?;
        }
        for strategy in [
            TraversalStrategy::DepthFirst,
            TraversalStrategy::BreadthFirst,
        ] {
            for mode in [
                "bound-serial",
                "bound-parallel",
                "typed-cold",
                "typed-warm",
                "typed-lazy",
            ] {
                let run = RunOptions {
                    start_nodes: (0..starts).map(|i| (i * depth).into()).collect(),
                    max_depth: Some(depth as usize),
                    max_paths: if name == "first" { Some(1) } else { None },
                    strategy,
                    parallel: mode == "bound-parallel",
                    intermediate_states: name == "history",
                    ..Default::default()
                };
                let typed = boxed_typed_run(kernel.clone());
                let cache = TypedPayloadCache::default();
                measure(Case {
                    name: format!("{name}/{strategy:?}/{mode}"),
                    mode,
                    graph: &graph,
                    kernel: &kernel,
                    typed: typed.as_ref(),
                    cache: &cache,
                    paths: &paths,
                    run,
                })?;
            }
        }
        std::fs::remove_dir_all(dir)?;
    }
    Ok(())
}
