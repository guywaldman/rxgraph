//! Isolate the cost of hashing the benchmark's 4 KiB state.
//! cargo run --release -p rxgraph --example reconvergence_hash_probe --offline
// Keep the explicit build/hash/finish sequence used by a hash-map lookup.
#![allow(clippy::manual_hash_one)]
#[path = "reconvergence_bench.rs"]
#[allow(dead_code)]
mod fixture;
#[path = "reconvergence_bench/heap.rs"]
#[allow(dead_code)]
mod heap;

use std::{
    collections::hash_map::RandomState,
    hash::{BuildHasher, Hash, Hasher},
    hint::black_box,
    sync::Arc,
    time::Instant,
};

use anyhow::Result;
use fixture::State;
use heap::HeapState;
use serde_json::json;

#[derive(Clone, Copy)]
enum Op {
    MetadataHash,
    VecHash,
    ArcHash,
    VecClone,
    ArcClone,
}
impl Op {
    fn label(self) -> &'static str {
        match self {
            Self::MetadataHash => "metadata-hash",
            Self::VecHash => "vec-full-hash",
            Self::ArcHash => "arc-full-hash",
            Self::VecClone => "vec-clone-drop",
            Self::ArcClone => "arc-clone-drop",
        }
    }
}

fn window(
    op: Op,
    vec: &HeapState<Vec<u64>>,
    arc: &HeapState<Arc<Vec<u64>>>,
    builder: &RandomState,
    seconds: f64,
) -> (f64, usize) {
    let start = Instant::now();
    let mut iterations = 0;
    loop {
        for _ in 0..8192 {
            match op {
                Op::MetadataHash => {
                    let mut h = builder.build_hasher();
                    black_box(&arc.meta).hash(&mut h);
                    black_box(h.finish());
                }
                Op::VecHash => {
                    let mut h = builder.build_hasher();
                    black_box(vec).hash(&mut h);
                    black_box(h.finish());
                }
                Op::ArcHash => {
                    let mut h = builder.build_hasher();
                    black_box(arc).hash(&mut h);
                    black_box(h.finish());
                }
                Op::VecClone => {
                    let value = black_box(&vec.buffer).clone();
                    black_box(&value);
                    drop(value);
                }
                Op::ArcClone => {
                    let value = Arc::clone(black_box(&arc.buffer));
                    black_box(&value);
                    drop(value);
                }
            }
        }
        iterations += 8192;
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= seconds {
            return (elapsed, iterations);
        }
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn main() -> Result<()> {
    anyhow::ensure!(!cfg!(debug_assertions), "run with --release");
    let samples: usize = std::env::var("RX_HASH_SAMPLES")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(5);
    let seconds: f64 = std::env::var("RX_HASH_SECONDS")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(2.0);
    anyhow::ensure!(
        samples >= 3 && seconds >= 1.0,
        "use >=3 samples and >=1s windows"
    );
    let meta = State {
        depth: 24,
        route: 0,
        score: 307,
        carried: [],
    };
    let mut values = (0..512).map(|v| v + 19).collect::<Vec<u64>>();
    values[0] = meta.score;
    let vec = HeapState {
        meta,
        buffer: values.clone(),
    };
    let arc = HeapState {
        meta,
        buffer: Arc::new(values),
    };
    let builder = RandomState::new();
    let mut vec_hash = builder.build_hasher();
    let mut arc_hash = builder.build_hasher();
    vec.hash(&mut vec_hash);
    arc.hash(&mut arc_hash);
    assert_eq!(vec_hash.finish(), arc_hash.finish());
    let ops = [
        Op::MetadataHash,
        Op::VecHash,
        Op::ArcHash,
        Op::VecClone,
        Op::ArcClone,
    ];
    for op in ops {
        window(op, &vec, &arc, &builder, 0.25);
    }
    let mut durations = vec![Vec::new(); ops.len()];
    let mut counts = vec![Vec::new(); ops.len()];
    for sample in 0..samples {
        for position in 0..ops.len() {
            let p = if sample % 2 == 0 {
                position
            } else {
                ops.len() - position - 1
            };
            let i = (p + sample) % ops.len();
            let (elapsed, count) = window(ops[i], &vec, &arc, &builder, seconds);
            durations[i].push(elapsed);
            counts[i].push(count);
        }
        eprintln!("hash probe: sample {}/{} complete", sample + 1, samples);
    }
    let rows = ops
        .iter()
        .enumerate()
        .map(|(i, op)| {
            let ns = durations[i]
                .iter()
                .zip(&counts[i])
                .map(|(&s, &n)| s * 1e9 / n as f64)
                .collect::<Vec<_>>();
            eprintln!("{}: {:.1} ns/op", op.label(), median(&ns));
            json!({"operation":op.label(), "median_ns":median(&ns), "samples_ns":ns,
            "measurement_window_seconds":durations[i], "iterations_per_window":counts[i]})
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"buffer_bytes":4096, "samples":samples,
        "minimum_window_seconds":seconds, "ops":rows,
        "method":"release; default hasher; one hot state; no allocator instrumentation; clone includes drop; loop checks time every 8192 operations"}))?
    );
    Ok(())
}
