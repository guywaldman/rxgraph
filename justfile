venv := ".venv"
python := venv / "bin/python"

default: test

venv:
    test -x {{python}} || uv venv {{venv}}

dev: venv
    uv pip install --python {{python}} maturin pytest
    {{venv}}/bin/maturin develop --release

test: dev
    {{python}} -m pytest

bench-python: dev
    {{python}} scripts/bench_python.py
