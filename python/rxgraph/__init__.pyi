from dataclasses import dataclass

@dataclass
class FieldSchema:
    id: str
    type: str

@dataclass
class Schema:
    fields: list[FieldSchema]

class Graph:
    """A class representing a reactive graph. This class is implemented in Rust and exposed to Python using PyO3."""

    def __init__(self, node_schema: Schema) -> None:
        """Initializes a new instance of the Graph class."""
