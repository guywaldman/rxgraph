use anyhow::{Context, Result};

use crate::{
    dsl::{
        DslKernel, StateRow, StateValue, StateValues, Value,
        arrow_value::ColumnReader,
        compiled::{CompiledBool, CompiledExpr, FastScalar, FastScalarKernel},
        eval::EvalCtx,
        expr::{ColumnRef, Expr},
    },
    graph::{EDGE_DEST_COL, EDGE_SRC_COL, Graph, GraphId, GraphRepo, ID_COL},
};

#[derive(Debug)]
pub(crate) struct BoundKernel {
    visit: CompiledBool,
    next_state: Vec<(usize, CompiledExpr)>,
    stop: CompiledBool,
    fast_scalar: Option<FastScalarKernel>,
    names: Vec<String>,
    initial_state: StateValues,
}

impl BoundKernel {
    pub(crate) fn bind(graph: &Graph, kernel: DslKernel) -> Result<Self> {
        let names = state_names(&kernel.initial_state, &kernel.next_state);
        let mut bind = |column| BoundColumn::bind(graph, column, &names);
        let initial_state = normalize_state(kernel.initial_state, &names);
        let visit = kernel.visit.try_map_column(&mut bind)?;
        let next_state = kernel
            .next_state
            .into_iter()
            .map(|(name, expr)| {
                Ok((
                    state_index(&names, &name).unwrap(),
                    expr.try_map_column(&mut bind)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let stop = kernel.stop.try_map_column(&mut bind)?;
        let fast_scalar =
            FastScalarKernel::compile(&visit, &next_state, &stop, initial_state.as_slice());

        Ok(Self {
            visit: CompiledBool::compile(visit)?,
            next_state: next_state
                .into_iter()
                .map(|(index, expr)| Ok((index, CompiledExpr::compile(expr)?)))
                .collect::<Result<_>>()?,
            stop: CompiledBool::compile(stop)?,
            fast_scalar,
            names: names.clone(),
            initial_state,
        })
    }

    pub(crate) fn initial_state(&self) -> &StateValues {
        &self.initial_state
    }

    pub(crate) fn fast_scalar(&self) -> Option<&FastScalarKernel> {
        self.fast_scalar.as_ref()
    }

    pub(crate) fn visit(&self, ctx: &EvalCtx<'_>) -> Result<bool> {
        self.visit.eval(ctx)
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
        self.stop.eval(ctx)
    }

    pub(crate) fn state_row(&self, state: &[StateValue]) -> StateRow {
        self.names
            .iter()
            .cloned()
            .zip(state.iter().map(StateValue::to_value))
            .collect()
    }

    pub(crate) fn fast_state_row(&self, state: &[FastScalar]) -> StateRow {
        let fast = self
            .fast_scalar
            .as_ref()
            .expect("fast state row requires a fast scalar kernel");
        self.names
            .iter()
            .cloned()
            .zip(
                fast.state_values(state)
                    .into_iter()
                    .map(|value| value.to_value()),
            )
            .collect()
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
