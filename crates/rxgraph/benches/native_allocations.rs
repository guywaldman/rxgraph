//! Allocation profile, kept separate from the uninstrumented timing executable.
mod native_fixture;
use anyhow::Result;
use rxgraph::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::{alloc::System, hint::black_box};
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
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
        execute()?;
        let region = Region::new(GLOBAL);
        let result = execute()?;
        let alloc = region.change();
        report.insert(name,serde_json::json!({"allocations":alloc.allocations,"bytes_allocated":alloc.bytes_allocated,"result_size":result.0,"lazy_payload_read_calls":result.1.lazy_payload_read_calls}));
        Ok(())
    })?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
