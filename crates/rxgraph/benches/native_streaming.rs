//! Stream creation, first-batch latency, and full drain on the same workloads.
mod native_fixture;
use anyhow::Result;
use rxgraph::*;
use std::{hint::black_box, sync::Arc, time::Instant};
fn main() -> Result<()> {
    let mut report = serde_json::Map::new();
    native_fixture::with_cases(|case| {
        for size in [1, 64, 1024, 4096] {
            let execute = || -> Result<(usize, Option<f64>, SearchStats)> {
                let start = Instant::now();
                let mut first = None;
                let mut count = 0;
                if case.mode.starts_with("bound") {
                    let mut stream = case.graph.search_batches_with(
                        case.kernel.clone(),
                        case.run.clone(),
                        size,
                    )?;
                    for batch in stream.by_ref() {
                        let batch = batch?;
                        if first.is_none() {
                            first = Some(start.elapsed().as_secs_f64());
                        }
                        count += batch.len();
                        black_box(&batch);
                    }
                    Ok((count, first, stream.stats()))
                } else {
                    let mut stream = match case.mode {
                        "typed-cold" => case.typed.stream_eager_cached(
                            Arc::clone(case.graph),
                            &TypedPayloadCache::default(),
                            case.run.clone(),
                        )?,
                        "typed-warm" => case.typed.stream_eager_cached(
                            Arc::clone(case.graph),
                            case.cache,
                            case.run.clone(),
                        )?,
                        _ => case.typed.stream_parquet_lazy(
                            Arc::clone(case.graph),
                            case.paths.clone(),
                            case.run.clone(),
                        )?,
                    };
                    while let Some(batch) = stream.next_batch(size)? {
                        if first.is_none() {
                            first = Some(start.elapsed().as_secs_f64());
                        }
                        count += batch.paths.len();
                        black_box(&batch);
                    }
                    Ok((count, first, stream.stats()))
                }
            };
            let expected = execute()?.0;
            execute()?;
            let mut samples = vec![];
            let mut firsts = vec![];
            for _ in 0..9 {
                let start = Instant::now();
                let (n, first, _) = execute()?;
                samples.push(start.elapsed().as_secs_f64());
                firsts.push(first);
                assert_eq!(n, expected);
            }
            let (_, _, stats) = execute()?;
            let mut sorted = samples.clone();
            sorted.sort_by(f64::total_cmp);
            report.insert(format!("{}/batch-{size}",case.name),serde_json::json!({"median_seconds":sorted[4],"p90_seconds":sorted[8],"samples_seconds":samples,"first_batch_seconds":firsts,"result_size":expected,"parallel_edges":stats.parallel_edges}));
        }
        Ok(())
    })?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
