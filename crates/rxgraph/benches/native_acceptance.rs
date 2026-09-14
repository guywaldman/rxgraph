//! Fixed native workloads timed with the normal system allocator.
mod native_fixture;
use anyhow::Result;
use rxgraph::*;
use std::{hint::black_box, time::Instant};
fn main() -> Result<()> {
    let mut report = serde_json::Map::new();
    native_fixture::with_cases(|case| {
        let native_fixture::Case {
            name,
            mode,
            graph,
            kernel,
            typed,
            cache,
            paths,
            run,
        } = case;
        let execute = || -> Result<(usize, SearchStats)> {
            if mode.starts_with("bound") {
                let result = graph.search_paths_with(kernel.clone(), run.clone())?;
                black_box(&result);
                Ok((result.paths.len(), result.stats))
            } else {
                let result = match mode {
                    "typed-cold" => typed.run_eager(graph, run.clone())?,
                    "typed-warm" => typed.run_eager_cached(graph, cache, run.clone())?,
                    _ => typed.run_parquet_lazy(graph, paths.clone(), run.clone())?,
                };
                black_box(&result);
                Ok((result.paths.len(), result.stats))
            }
        };
        let start = Instant::now();
        let expected = execute()?;
        let first = start.elapsed().as_secs_f64();
        execute()?;
        let mut samples = vec![];
        for _ in 0..9 {
            let start = Instant::now();
            let result = execute()?;
            samples.push(start.elapsed().as_secs_f64());
            assert_eq!(result.0, expected.0);
        }
        samples.sort_by(f64::total_cmp);
        report.insert(name,serde_json::json!({"samples_seconds":samples,"median_seconds":samples[4],"p90_seconds":samples[8],"first_search_seconds":first,"result_size":expected.0,"evaluated_edges":expected.1.evaluated_edges,"parallel_edges":expected.1.parallel_edges,"lazy_payload_read_calls":expected.1.lazy_payload_read_calls}));
        Ok(())
    })?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
