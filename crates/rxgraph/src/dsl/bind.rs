use anyhow::{Context, Result};

use crate::{
    dsl::{
        DslKernel, StateRow, StateValue, StateValues, Value,
        arrow_value::ColumnReader,
        eval::EvalCtx,
        expr::{ColumnRef, Expr},
        ops::scalar::ScalarOp,
    },
    graph::{EDGE_DEST_COL, EDGE_SRC_COL, Graph, GraphId, GraphRepo, ID_COL, NodeId},
    traversal::{EdgeCtx, Kernel},
};

#[derive(Debug)]
pub(crate) struct BoundKernel {
    visit: Expr<BoundColumn>,
    next_state: Vec<(usize, Expr<BoundColumn>)>,
    stop: Expr<BoundColumn>,
    names: Vec<String>,
    initial_state: StateValues,
}

impl BoundKernel {
    pub(crate) fn bind(graph: &Graph, kernel: DslKernel) -> Result<Self> {
        let names = state_names(&kernel.initial_state, &kernel.next_state);
        let mut bind = |column| BoundColumn::bind(graph, column, &names);

        Ok(Self {
            visit: kernel.visit.try_map_column(&mut bind)?,
            next_state: kernel
                .next_state
                .into_iter()
                .map(|(name, expr)| {
                    Ok((
                        state_index(&names, &name).unwrap(),
                        expr.try_map_column(&mut bind)?,
                    ))
                })
                .collect::<Result<_>>()?,
            stop: kernel.stop.try_map_column(&mut bind)?,
            names: names.clone(),
            initial_state: normalize_state(kernel.initial_state, &names),
        })
    }

    pub(crate) fn initial_state(&self) -> &StateValues {
        &self.initial_state
    }

    pub(crate) fn visit(&self, ctx: &EvalCtx<'_>) -> Result<bool> {
        self.visit.eval(ctx)?.truthy()
    }

    pub(crate) fn next_state(
        &self,
        current: &[StateValue],
        ctx: &EvalCtx<'_>,
    ) -> Result<StateValues> {
        let mut next = current.iter().cloned().collect::<StateValues>();
        for (index, expr) in &self.next_state {
            next[*index] = StateValue::new(expr.eval(ctx)?);
        }
        Ok(next)
    }

    pub(crate) fn stop(&self, ctx: &EvalCtx<'_>) -> Result<bool> {
        self.stop.eval(ctx)?.truthy()
    }

    pub(crate) fn state_row(&self, state: &[StateValue]) -> StateRow {
        self.names
            .iter()
            .cloned()
            .zip(state.iter().map(StateValue::to_value))
            .collect()
    }
}

/// The DSL is just another kernel: every per-edge call rebuilds an [`EvalCtx`]
/// from the [`EdgeCtx`] and delegates to the inherent methods above.
impl Kernel for BoundKernel {
    type State = StateValues;

    fn initial_state(&self, _graph: &Graph, _start: NodeId) -> Self::State {
        // The DSL's initial state is independent of the start node.
        self.initial_state().clone()
    }

    fn visit(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<bool> {
        let ctx = EvalCtx::new(cx.graph(), cx.src(), cx.dest(), cx.edge(), cx.state());
        BoundKernel::visit(self, &ctx)
    }

    fn next_state(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<Self::State> {
        let ctx = EvalCtx::new(cx.graph(), cx.src(), cx.dest(), cx.edge(), cx.state());
        BoundKernel::next_state(self, cx.state(), &ctx)
    }

    fn stop(&self, cx: &EdgeCtx<'_, Self::State>) -> Result<bool> {
        let ctx = EvalCtx::new(cx.graph(), cx.src(), cx.dest(), cx.edge(), cx.state());
        BoundKernel::stop(self, &ctx)
    }

    fn state_row(&self, state: &Self::State) -> StateRow {
        BoundKernel::state_row(self, state)
    }
}

#[derive(Debug, Clone)]
pub(crate) enum BoundColumn {
    SrcId,
    DestId,
    EdgeId,
    Src(ColumnReader),
    Dest(ColumnReader),
    Edge(ColumnReader),
    State(usize),
    MissingState,
}

impl BoundColumn {
    fn bind(graph: &Graph, column: ColumnRef, names: &[String]) -> Result<Self> {
        Ok(match column {
            ColumnRef::SrcId => Self::SrcId,
            ColumnRef::DestId => Self::DestId,
            ColumnRef::EdgeId => Self::EdgeId,
            ColumnRef::SrcField(name) if name == ID_COL => Self::SrcId,
            ColumnRef::SrcField(name) => Self::Src(ColumnReader::bind(&graph.repo.nodes, &name)?),
            ColumnRef::DestField(name) if name == ID_COL => Self::DestId,
            ColumnRef::DestField(name) => Self::Dest(ColumnReader::bind(&graph.repo.nodes, &name)?),
            ColumnRef::EdgeField(name) if name == ID_COL => Self::EdgeId,
            ColumnRef::EdgeField(name) if name == EDGE_SRC_COL => Self::SrcId,
            ColumnRef::EdgeField(name) if name == EDGE_DEST_COL => Self::DestId,
            ColumnRef::EdgeField(name) => Self::Edge(ColumnReader::bind(&graph.repo.edges, &name)?),
            ColumnRef::State(name) => state_index(names, &name)
                .map(Self::State)
                .unwrap_or(Self::MissingState),
        })
    }

    pub(crate) fn value(&self, ctx: &EvalCtx<'_>) -> Result<Value> {
        match self {
            Self::SrcId => graph_id_value(
                ctx.graph
                    .repo
                    .external_node(ctx.src)
                    .context("missing src id")?,
            ),
            Self::DestId => graph_id_value(
                ctx.graph
                    .repo
                    .external_node(ctx.dest)
                    .context("missing dest id")?,
            ),
            Self::EdgeId => graph_id_value(
                ctx.graph
                    .repo
                    .external_edge(ctx.edge)
                    .context("missing edge id")?,
            ),
            Self::Src(reader) => reader.value(ctx.src as usize),
            Self::Dest(reader) => reader.value(ctx.dest as usize),
            Self::Edge(reader) => reader.value(ctx.edge as usize),
            Self::State(index) => Ok(ctx.state[*index].to_value()),
            Self::MissingState => Ok(Value::Null),
        }
    }

    pub(crate) fn eval_scalar_literal(
        &self,
        ctx: &EvalCtx<'_>,
        op: ScalarOp,
        literal: &Value,
        reverse: bool,
    ) -> Result<Option<Value>> {
        match self {
            Self::Src(reader) => reader.eval_scalar_literal(ctx.src as usize, op, literal, reverse),
            Self::Dest(reader) => {
                reader.eval_scalar_literal(ctx.dest as usize, op, literal, reverse)
            }
            Self::Edge(reader) => {
                reader.eval_scalar_literal(ctx.edge as usize, op, literal, reverse)
            }
            Self::SrcId | Self::DestId | Self::EdgeId | Self::State(_) | Self::MissingState => {
                Ok(None)
            }
        }
    }
}

fn graph_id_value(id: GraphId<'_>) -> Result<Value> {
    Ok(match id {
        GraphId::U64(value) => Value::U64(value),
        GraphId::Str(value) => Value::Str(std::sync::Arc::from(value)),
    })
}

fn state_names(initial: &StateRow, next: &[(String, Expr<ColumnRef>)]) -> Vec<String> {
    let mut names = initial
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    names.extend(next.iter().map(|(name, _)| name.clone()));
    names.sort();
    names.dedup();
    names
}

fn normalize_state(state: StateRow, names: &[String]) -> StateValues {
    names
        .iter()
        .map(|name| {
            state
                .binary_search_by(|(key, _)| key.as_str().cmp(name))
                .ok()
                .map(|i| StateValue::new(state[i].1.clone()))
                .unwrap_or_else(|| StateValue::new(Value::Null))
        })
        .collect::<StateValues>()
}

fn state_index(names: &[String], name: &str) -> Option<usize> {
    names.binary_search_by(|key| key.as_str().cmp(name)).ok()
}
