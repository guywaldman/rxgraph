//! Compare per-row optional-column probes on a million-row Arrow batch.
use std::{hint::black_box, sync::Arc, time::Instant};

use anyhow::{Context, Result, bail};
use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use rxgraph::ArrowRow;
use serde_json::json;

const ROWS: usize = 1_000_000;
const OPTIONAL: [&str; 5] = [
    "missing_string",
    "missing_u64",
    "missing_bool",
    "missing_list",
    "missing_struct",
];

#[inline(never)]
fn original_probe(row: &ArrowRow<'_>, col: &str) -> bool {
    row.u64(col).is_ok()
        || row.bool(col).is_ok()
        || row.string(col).is_ok()
        || row.list_items(col).is_ok()
        || row.struct_(col).is_ok()
}

#[inline(never)]
fn decode(batch: &RecordBatch, arm: &str) -> Result<(u64, usize)> {
    let mut checksum = 0u64;
    let mut present = 0usize;
    for i in 0..ROWS {
        let row = ArrowRow::new(black_box(batch), black_box(i));
        checksum = checksum.wrapping_add(row.u64("id")?.context("missing id")?);
        let fields = black_box(&OPTIONAL);
        match arm {
            "probe" => {
                for field in fields {
                    present += usize::from(original_probe(&row, black_box(field)));
                }
            }
            "has" => {
                for field in fields {
                    present += usize::from(row.has(black_box(field)));
                }
            }
            "optional" => {
                present += usize::from(row.opt_string(black_box(fields[0]))?.is_some());
                present += usize::from(row.opt_u64(black_box(fields[1]))?.is_some());
                present += usize::from(row.opt_bool(black_box(fields[2]))?.is_some());
                present += usize::from(row.opt_list_items(black_box(fields[3]))?.is_some());
                present += usize::from(row.opt_struct(black_box(fields[4]))?.is_some());
            }
            _ => bail!("unknown arm {arm:?}"),
        }
    }
    Ok(black_box((checksum, present)))
}

fn main() -> Result<()> {
    let arm = std::env::args()
        .nth(1)
        .context("expected probe, has or optional")?;
    let ids = Arc::new(UInt64Array::from_iter_values(0..ROWS as u64)) as ArrayRef;
    let batch = RecordBatch::try_from_iter([("id", ids)])?;
    let start = Instant::now();
    let mut passes = 0u64;
    let (checksum, present) = loop {
        let result = decode(&batch, &arm)?;
        assert_eq!(result.0, (ROWS as u64 - 1) * ROWS as u64 / 2);
        assert_eq!(result.1, 0);
        passes += 1;
        if start.elapsed().as_secs_f64() >= 2.0 {
            break result;
        }
    };
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({"arm": arm, "rows": ROWS, "absent_columns": OPTIONAL.len(), "checksum": checksum, "present": present, "passes": passes, "window_seconds": seconds, "seconds_per_pass": seconds / passes as f64})
    );
    Ok(())
}
