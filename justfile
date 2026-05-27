set shell := ["zsh", "-cu"]

venv := ".venv"
python := venv / "bin/python"
maturin := venv / "bin/maturin"
ruff := venv / "bin/ruff"
prek := venv / "bin/prek"
py_sources := "python tests benches"
python_manifest := "crates/rxgraph-python/Cargo.toml"

default: test

setup:
    test -x {{python}} || uv venv {{venv}}
    uv sync --locked --group dev --no-install-project

lock:
    uv lock

lock-check:
    uv lock --check

develop: develop-release

develop-release: setup
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

test-python *args: develop-release
    {{python}} -m pytest {{args}}

test: test-rust test-python

package-rust:
    cargo publish -p rxgraph --dry-run --locked --allow-dirty

package-python: setup
    {{maturin}} build --manifest-path {{python_manifest}} --release --out dist

ci: lock-check fmt-check lint test package-rust package-python

precommit: setup
    {{prek}} run --all-files

install-hooks: setup
    {{prek}} install

bench *args: develop-release
    {{python}} -m benches.main {{args}}
