set shell := ["sh", "-cu"]

venv := ".venv"
python := venv / "bin/python"
maturin := venv / "bin/maturin"
ruff := venv / "bin/ruff"
prek := venv / "bin/prek"
py_sources := "python tests benches examples"
python_manifest := "crates/rxgraph-python/Cargo.toml"

default: test

setup:
    test -x {{python}} || uv venv {{venv}}
    uv sync --locked --group dev --no-install-project

lock:
    uv lock

lock-check:
    uv lock --check

build-maturin: setup
    {{maturin}} develop --manifest-path {{python_manifest}} --release

fmt:
    cargo fmt --all
    {{ruff}} format {{py_sources}}

fmt-check: setup
    cargo fmt --all --check
    {{ruff}} format --check {{py_sources}}

lint: setup
    cargo clippy --workspace --all-targets -- -D warnings
    {{ruff}} check {{py_sources}}

test-rust:
    cargo test --workspace --locked

test-python *args: build-maturin
    {{python}} -m pytest {{args}}

test: test-rust test-python

package-rust:
    cargo publish -p rxgraph --dry-run --locked --allow-dirty

package-python: setup
    {{maturin}} build --manifest-path {{python_manifest}} --release --out dist

ci: lock-check fmt-check lint test package-rust package-python memcheck

precommit: setup
    {{prek}} run --all-files

install-hooks: setup
    {{prek}} install

bench *args: build-maturin
    {{python}} -m benches.main --cache {{args}}

bench-memory-rust *args: build-maturin
    cargo bench -p rxgraph --bench memory

memcheck *args: build-maturin
    {{python}} -m benches.memory_rss {{args}}

profile script: setup
    @command -v flamegraph >/dev/null || { echo "Install cargo-flamegraph first: cargo install flamegraph"; exit 1; }
    @{{maturin}} develop --manifest-path {{python_manifest}} --profile profiling
    @mkdir -p dist
    @flamegraph --output dist/flamegraph.svg -- {{python}} {{script}}
    @echo "Wrote dist/flamegraph.svg"
