import doctest
from pathlib import Path

import rxgraph as rxg


def test_rxgraph_docstrings() -> None:
    result = doctest.testmod(rxg, optionflags=doctest.ELLIPSIS)

    assert result.failed == 0


def test_readme_examples() -> None:
    readme = Path(__file__).parents[1] / "README.md"
    result = doctest.testfile(str(readme), module_relative=False)

    assert result.failed == 0
