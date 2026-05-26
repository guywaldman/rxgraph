venv := ".venv"
python := venv / "bin/python"

default: test

venv:
    test -x {{python}} || uv venv {{venv}}

develop: venv
    uv pip install --python {{python}} maturin pytest
    {{venv}}/bin/maturin develop --release

test: develop
    {{python}} -m pytest
