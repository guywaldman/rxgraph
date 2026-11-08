use std::{
    fs::File,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Result, ensure};
use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, ListArray, RecordBatch, StringArray, StructArray,
        UInt64Array, record_batch,
    },
    buffer::NullBuffer,
    datatypes::{DataType, Field, Int64Type, Schema},
};
use parquet::arrow::ArrowWriter;
use rxgraph::{
    ArrowRow, Graph, ParquetPaths, PayloadField, RunOptions, Transition, TypedKernel,
    TypedPayloadCache, Value, boxed_typed_run, traversal::native,
};

fn payload_batch() -> Result<RecordBatch> {
    let items =
        ListArray::from_iter_primitive::<Int64Type, _, _>(vec![Some(vec![Some(3), Some(7)]), None]);
    let fields = vec![
        Arc::new(Field::new("name", DataType::Utf8, true)),
        Arc::new(Field::new("number", DataType::UInt64, true)),
    ];
    let profile = StructArray::new(
        fields.into(),
        vec![
            Arc::new(StringArray::from(vec![Some("nested"), None])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![Some(42), None])) as ArrayRef,
        ],
        Some(NullBuffer::from(vec![true, false])),
    );
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![Some("hello"), None])),
        Arc::new(UInt64Array::from(vec![Some(12), None])),
        Arc::new(BooleanArray::from(vec![Some(true), None])),
        Arc::new(items),
        Arc::new(profile),
    ];
    let names = ["text", "number", "flag", "items", "profile"];
    let schema = Schema::new(
        names
            .iter()
            .zip(&columns)
            .map(|(name, col)| Field::new(*name, col.data_type().clone(), true))
            .collect::<Vec<_>>(),
    );
    Ok(RecordBatch::try_new(Arc::new(schema), columns)?)
}

#[test]
fn optional_getters_and_presence_handle_absent_null_values_and_types() -> Result<()> {
    let batch = payload_batch()?;
    let row = ArrowRow::new(&batch, 0);
    assert!(Arc::ptr_eq(row.schema(), batch.schema_ref()));
    for name in ["text", "number", "flag", "items", "profile"] {
        assert!(row.has(name), "missing {name}");
        assert!(row.schema().index_of(name).is_ok());
    }
    assert!(!row.has("absent"));
    assert!(row.schema().index_of("absent").is_err());

    assert_eq!(row.opt_string("text")?, Some("hello".to_owned()));
    assert_eq!(row.opt_u64("number")?, Some(12));
    assert_eq!(row.opt_bool("flag")?, Some(true));
    let list = row.opt_list_items("items")?.unwrap();
    assert_eq!(list.len(), 2);
    assert_eq!(list.u64(0)?, Some(3));
    assert_eq!(list.u64(1)?, Some(7));
    let profile = row.opt_struct("profile")?.unwrap();
    assert!(profile.has("name"));
    assert!(profile.has("number"));
    assert!(!profile.has("absent"));
    assert_eq!(profile.string("name")?, Some("nested".to_owned()));
    assert_eq!(profile.u64("number")?, Some(42));
    assert_eq!(
        profile.string("absent").unwrap_err().to_string(),
        "missing struct field \"absent\""
    );

    assert_eq!(row.opt_string("absent")?, None);
    assert_eq!(row.opt_u64("absent")?, None);
    assert_eq!(row.opt_bool("absent")?, None);
    assert!(row.opt_list_items("absent")?.is_none());
    assert!(row.opt_struct("absent")?.is_none());
    assert_eq!(
        row.string("absent").unwrap_err().to_string(),
        "missing payload column \"absent\""
    );
    assert_eq!(
        row.u64("absent").unwrap_err().to_string(),
        "missing payload column \"absent\""
    );

    assert!(row.opt_string("number").is_err());
    assert!(row.opt_u64("text").is_err());
    assert!(row.opt_bool("number").is_err());
    assert!(row.opt_list_items("flag").is_err());
    assert!(row.opt_struct("items").is_err());

    let null = ArrowRow::new(&batch, 1);
    assert_eq!(null.opt_string("text")?, None);
    assert_eq!(null.opt_u64("number")?, None);
    assert_eq!(null.opt_bool("flag")?, None);
    assert!(null.opt_list_items("items")?.is_none());
    assert!(null.opt_struct("profile")?.is_none());
    Ok(())
}

#[test]
fn graph_rows_expose_presence_for_untyped_kernels() -> Result<()> {
    let nodes = record_batch!(("id", UInt64, [0, 1]), ("name", Utf8, ["a", "b"]))?;
    let edges = record_batch!(
        ("id", UInt64, [0]),
        ("src", UInt64, [0]),
        ("dest", UInt64, [1]),
        ("cost", UInt64, [5])
    )?;
    let graph = Graph::new(nodes, edges)?;
    let node = graph.node_at(1)?;
    assert!(node.has("name"));
    assert!(!node.has("cost"));
    assert_eq!(node.opt_string("name")?, Some("b".into()));
    let edge = graph.edge_at(0)?;
    assert!(edge.has("cost"));
    assert!(!edge.has("name"));
    assert_eq!(edge.opt_u64("cost")?, Some(5));
    assert!(graph.node_at(2).is_err());
    assert!(graph.edge_at(1).is_err());
    Ok(())
}

#[derive(Clone)]
struct ProjectedNode {
    present: bool,
    text: Option<String>,
}

impl TryFrom<ArrowRow<'_>> for ProjectedNode {
    type Error = anyhow::Error;

    fn try_from(row: ArrowRow<'_>) -> Result<Self> {
        ensure!(
            !row.has("original"),
            "source name leaked through projection"
        );
        ensure!(row.schema().index_of("original").is_err());
        let present = row.has("alias");
        ensure!(present == row.schema().index_of("alias").is_ok());
        Ok(Self {
            present,
            text: row.opt_string("alias")?,
        })
    }
}

#[derive(Clone)]
struct ProjectionKernel {
    include_alias: bool,
}

#[derive(Clone, serde::Serialize)]
struct ProjectionState {
    present: bool,
    text_len: u64,
}

impl TypedKernel for ProjectionKernel {
    type Node = ProjectedNode;
    type Edge = ();
    type State = ProjectionState;

    fn node_fields(&self) -> Vec<PayloadField> {
        if self.include_alias {
            vec![PayloadField::aliased("original", "alias")]
        } else {
            Vec::new()
        }
    }

    fn initial_state(
        &self,
        _: &native::StartCtx<'_, Self::Node, Self::Edge>,
    ) -> Result<Self::State> {
        Ok(ProjectionState {
            present: false,
            text_len: 0,
        })
    }

    fn transition(
        &self,
        cx: &native::EdgeCtx<'_, '_, Self::Node, Self::Edge, Self::State>,
    ) -> Result<Transition<Self::State>> {
        let node = cx.dest()?;
        Ok(Transition::Complete(ProjectionState {
            present: node.present,
            text_len: node.text.as_ref().map_or(0, |text| text.len() as u64),
        }))
    }
}

#[test]
fn typed_eager_and_lazy_decoders_see_projected_aliases() -> Result<()> {
    let nodes = record_batch!(("id", UInt64, [0, 1]), ("original", Utf8, ["a", "dest"]))?;
    let edges = record_batch!(
        ("id", UInt64, [0]),
        ("src", UInt64, [0]),
        ("dest", UInt64, [1])
    )?;
    let id = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let dir = std::env::temp_dir();
    let paths = ParquetPaths {
        nodes: dir.join(format!(
            "rxgraph-presence-nodes-{}-{id}.parquet",
            std::process::id()
        )),
        edges: dir.join(format!(
            "rxgraph-presence-edges-{}-{id}.parquet",
            std::process::id()
        )),
    };
    for (path, batch) in [(&paths.nodes, &nodes), (&paths.edges, &edges)] {
        let mut writer = ArrowWriter::try_new(File::create(path)?, batch.schema(), None)?;
        writer.write(batch)?;
        writer.close()?;
    }
    let graph = Arc::new(Graph::new(nodes, edges)?);
    let file_graph = Arc::new(Graph::from_parquet_topology(
        paths.nodes.clone(),
        paths.edges.clone(),
    )?);
    let cache = TypedPayloadCache::default();
    let run = RunOptions {
        start_nodes: vec![0_u64.into()],
        parallel: false,
        ..Default::default()
    };
    for include_alias in [false, true] {
        let runner = boxed_typed_run(ProjectionKernel { include_alias });
        let eager = runner.run_eager_cached(&graph, &cache, run.clone())?;
        let lazy = runner.run_parquet_lazy(&file_graph, paths.clone(), run.clone())?;
        let expected = vec![
            ("present".to_string(), Value::Bool(include_alias)),
            (
                "text_len".to_string(),
                Value::U64(if include_alias { 4 } else { 0 }),
            ),
        ];
        assert_eq!(eager.paths.len(), 1);
        assert_eq!(eager.paths[0].state, expected);
        assert_eq!(lazy.paths[0].state, expected);

        let mut eager_stream =
            runner.stream_eager_cached(Arc::clone(&graph), &cache, run.clone())?;
        let mut lazy_stream =
            runner.stream_parquet_lazy(Arc::clone(&file_graph), paths.clone(), run.clone())?;
        assert_eq!(
            eager_stream.next_batch(1)?.unwrap().paths[0].state,
            expected
        );
        assert_eq!(lazy_stream.next_batch(1)?.unwrap().paths[0].state, expected);
    }
    std::fs::remove_file(paths.nodes)?;
    std::fs::remove_file(paths.edges)?;
    Ok(())
}
