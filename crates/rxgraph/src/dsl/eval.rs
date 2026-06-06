use crate::{
    dsl::{StateValue, Value},
    graph::{EdgeId, Graph, NodeId},
};

pub(crate) struct EvalCtx<'a> {
    pub(crate) graph: &'a Graph,
    pub(crate) src: NodeId,
    pub(crate) dest: NodeId,
    pub(crate) edge: EdgeId,
    pub(crate) state: &'a [StateValue],
    element: Option<&'a Value>,
}

impl<'a> EvalCtx<'a> {
    pub(crate) fn new(
        graph: &'a Graph,
        src: NodeId,
        dest: NodeId,
        edge: EdgeId,
        state: &'a [StateValue],
    ) -> Self {
        Self {
            graph,
            src,
            dest,
            edge,
            state,
            element: None,
        }
    }

    pub(crate) fn with_state<'b>(&'b self, state: &'b [StateValue]) -> EvalCtx<'b> {
        EvalCtx {
            graph: self.graph,
            src: self.src,
            dest: self.dest,
            edge: self.edge,
            state,
            element: self.element,
        }
    }

    pub(crate) fn with_element<'b>(&'b self, element: &'b Value) -> EvalCtx<'b> {
        EvalCtx {
            graph: self.graph,
            src: self.src,
            dest: self.dest,
            edge: self.edge,
            state: self.state,
            element: Some(element),
        }
    }

    pub(crate) fn element(&self) -> Option<&'a Value> {
        self.element
    }
}
