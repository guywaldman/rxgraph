use std::{alloc::System, hint::black_box, sync::Arc};

use anyhow::Result;
use arrow::{
    array::{ArrayRef, RecordBatch, StringArray, StructArray},
    datatypes::{DataType, Field},
};
use rxgraph::ArrowRow;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[test]
fn row_and_struct_presence_checks_do_not_allocate() -> Result<()> {
    let inner = StructArray::from(vec![(
        Arc::new(Field::new("name", DataType::Utf8, false)),
        Arc::new(StringArray::from(vec!["value"])) as ArrayRef,
    )]);
    let batch = RecordBatch::try_from_iter([("profile", Arc::new(inner) as ArrayRef)])?;
    let row = ArrowRow::new(&batch, 0);
    let profile = row.struct_("profile")?.unwrap();

    let region = Region::new(GLOBAL);
    let mut found = 0usize;
    for _ in 0..1_000 {
        found += usize::from(row.has(black_box("profile")));
        found += usize::from(row.has(black_box("absent")));
        found += usize::from(profile.has(black_box("name")));
        found += usize::from(profile.has(black_box("absent")));
    }
    black_box(found);
    let allocations = region.change();

    assert_eq!(found, 2_000);
    assert_eq!(allocations.allocations, 0);
    assert_eq!(allocations.bytes_allocated, 0);
    Ok(())
}
