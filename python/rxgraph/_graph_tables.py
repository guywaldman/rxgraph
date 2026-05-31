from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar, cast

import polars as pl

NodeT = TypeVar("NodeT", bound=Hashable)
GraphId: TypeAlias = int | str
AttributeMap: TypeAlias = Mapping[str, object]
AttributeDict: TypeAlias = dict[str, object]
TableInput: TypeAlias = pl.DataFrame | list[tuple[str, pl.DataFrame]]
NodeInput: TypeAlias = NodeT | tuple[NodeT, AttributeMap]
EdgeInput: TypeAlias = tuple[NodeT, NodeT] | tuple[NodeT, NodeT, AttributeMap]
NullableGraphId: TypeAlias = GraphId | None

ID_COL = "id"
SRC_COL = "src"
DEST_COL = "dest"
NODE_RESERVED_ATTRS = frozenset({ID_COL})
EDGE_RESERVED_ATTRS = frozenset({ID_COL, SRC_COL, DEST_COL})
REVERSE_EDGE_SUFFIX = "__rev"
UINT64_MAX = (1 << 64) - 1

@dataclass(frozen=True)
class GraphTables(Generic[NodeT]):
    nodes: pl.DataFrame
    edges: pl.DataFrame
    label_to_id: dict[NodeT, int]
    id_to_label: list[NodeT]
    edge_id_to_label: dict[GraphId, GraphId] | None


def normalize_table(value: TableInput) -> pl.DataFrame:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(
                "rxgraph expects one node DataFrame and one edge DataFrame"
            )
        return value[0][1]
    return value


def build_labeled_tables(
    edges: Iterable[EdgeInput[NodeT]],
    nodes: Iterable[NodeInput[NodeT]] | None,
    *,
    bidirectional: bool = False,
) -> GraphTables[NodeT]:
    builder = _LabeledTableBuilder[NodeT](track_edge_labels=bidirectional)

    if nodes is not None:
        for node in nodes:
            label, attrs = _parse_node(node)
            builder.add_node(label, attrs)

    parsed_edges = [_parse_edge(edge) for edge in edges]
    for edge_id, (src, dest, attrs) in enumerate(parsed_edges):
        builder.add_edge(src, dest, attrs, edge_id)
    if bidirectional:
        for edge_id, (src, dest, attrs) in enumerate(parsed_edges):
            builder.add_edge(dest, src, attrs, edge_id)

    return builder.build()


def build_bidirectional_edges(
    edges: TableInput,
) -> tuple[pl.DataFrame, dict[GraphId, GraphId]]:
    edge_table = normalize_table(edges)

    edge_ids = cast(list[NullableGraphId], edge_table[ID_COL].to_list())
    reverse_ids = _reverse_edge_ids(edge_ids, edge_table.schema[ID_COL])
    reverse_table = edge_table.with_columns(
        pl.Series(ID_COL, reverse_ids, dtype=edge_table.schema[ID_COL]),
        pl.col(DEST_COL).alias(SRC_COL),
        pl.col(SRC_COL).alias(DEST_COL),
    )
    edge_id_to_label = {
        edge_id: edge_id for edge_id in edge_ids if edge_id is not None
    }
    edge_id_to_label.update(
        {
            reverse_id: edge_id
            for reverse_id, edge_id in zip(reverse_ids, edge_ids, strict=True)
            if edge_id is not None
        }
    )
    return pl.concat([edge_table, reverse_table], how="vertical"), edge_id_to_label


class _LabeledTableBuilder(Generic[NodeT]):
    label_to_id: dict[NodeT, int]
    id_to_label: list[NodeT]
    node_attrs: list[AttributeDict]
    edge_srcs: list[int]
    edge_dests: list[int]
    edge_attrs: list[AttributeDict]
    edge_id_to_label: dict[GraphId, GraphId] | None

    def __init__(self, *, track_edge_labels: bool) -> None:
        self.label_to_id = {}
        self.id_to_label = []
        self.node_attrs = []
        self.edge_srcs = []
        self.edge_dests = []
        self.edge_attrs = []
        self.edge_id_to_label = {} if track_edge_labels else None

    def add_node(self, label: NodeT, attrs: AttributeMap | None = None) -> int:
        if label in self.label_to_id:
            if attrs:
                self.node_attrs[self.label_to_id[label]].update(_attrs(attrs, "node"))
            return self.label_to_id[label]

        node_id = len(self.id_to_label)
        self.label_to_id[label] = node_id
        self.id_to_label.append(label)
        self.node_attrs.append(_attrs(attrs or {}, "node"))
        return node_id

    def add_edge(
        self,
        src: NodeT,
        dest: NodeT,
        attrs: AttributeMap | None,
        label: GraphId,
    ) -> None:
        edge_id = len(self.edge_srcs)
        self.edge_srcs.append(self.add_node(src))
        self.edge_dests.append(self.add_node(dest))
        self.edge_attrs.append(_attrs(attrs or {}, "edge"))
        if self.edge_id_to_label is not None:
            self.edge_id_to_label[edge_id] = label

    def build(self) -> GraphTables[NodeT]:
        node_data = _rows_to_columns(self.node_attrs)
        node_data[ID_COL] = list(range(len(self.id_to_label)))
        edge_data = _rows_to_columns(self.edge_attrs)
        edge_data[ID_COL] = list(range(len(self.edge_srcs)))
        edge_data[SRC_COL] = self.edge_srcs
        edge_data[DEST_COL] = self.edge_dests

        return GraphTables(
            nodes=pl.DataFrame(node_data, schema_overrides={ID_COL: pl.UInt64}),
            edges=pl.DataFrame(
                edge_data,
                schema_overrides={
                    ID_COL: pl.UInt64,
                    SRC_COL: pl.UInt64,
                    DEST_COL: pl.UInt64,
                },
            ),
            label_to_id=self.label_to_id,
            id_to_label=self.id_to_label,
            edge_id_to_label=self.edge_id_to_label,
        )


def _parse_node(node: NodeInput[NodeT]) -> tuple[NodeT, AttributeMap | None]:
    if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], Mapping):
        return node[0], cast(AttributeMap, node[1])
    return node, None


def _parse_edge(edge: EdgeInput[NodeT]) -> tuple[NodeT, NodeT, AttributeMap | None]:
    if len(edge) == 2:
        return edge[0], edge[1], None
    attrs = edge[2]
    if isinstance(attrs, Mapping):
        return edge[0], edge[1], cast(AttributeMap, attrs)
    raise ValueError("edges must be (src, dest) or (src, dest, attrs) tuples")


def _attrs(attrs: AttributeMap, kind: str) -> AttributeDict:
    reserved = NODE_RESERVED_ATTRS if kind == "node" else EDGE_RESERVED_ATTRS
    overlap = reserved.intersection(attrs)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"{kind} attributes cannot use reserved keys: {names}")
    return dict(attrs)


def _rows_to_columns(rows: list[AttributeDict]) -> dict[str, list[object]]:
    keys = sorted(
        {
            key
            for row in rows
            for key in row
            if any(r.get(key) is not None for r in rows)
        }
    )
    return {key: [row.get(key) for row in rows] for key in keys}


def _reverse_edge_ids(edge_ids: list[NullableGraphId], dtype: pl.DataType) -> list[GraphId]:
    used = set(edge_ids)
    if dtype.is_integer():
        return _reverse_integer_edge_ids(
            cast(list[int | None], edge_ids),
            cast(set[int | None], used),
        )
    if dtype == pl.String:
        return _reverse_string_edge_ids(
            cast(list[str | None], edge_ids),
            cast(set[str | None], used),
        )
    raise ValueError("DiGraph edge ids must be integers or strings")


def _reverse_integer_edge_ids(edge_ids: list[int | None], used: set[int | None]) -> list[int]:
    max_id = max((edge_id for edge_id in edge_ids if edge_id is not None), default=-1)
    next_id = max_id + 1
    if next_id + len(edge_ids) - 1 <= UINT64_MAX:
        return list(range(next_id, next_id + len(edge_ids)))

    reverse_ids: list[int] = []
    candidate = 0
    for _ in edge_ids:
        while candidate in used:
            candidate += 1
        reverse_ids.append(candidate)
        used.add(candidate)
    return reverse_ids


def _reverse_string_edge_ids(edge_ids: list[str | None], used: set[str | None]) -> list[str]:
    reverse_ids: list[str] = []
    for edge_id in edge_ids:
        base = f"{edge_id}{REVERSE_EDGE_SUFFIX}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        reverse_ids.append(candidate)
        used.add(candidate)
    return reverse_ids
