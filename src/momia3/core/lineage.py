"""Lineage views: edges table ↔ networkx directed graph (§2.4).

Algorithms may operate on a :class:`networkx.DiGraph` internally, but the
on-disk representation is always the edges table.
"""

from __future__ import annotations

import networkx as nx
import polars as pl
import pyarrow as pa

from momia3 import schema as _schema

__all__ = ["edges_to_nx", "nx_to_edges"]


def edges_to_nx(edges_df: pl.DataFrame) -> nx.DiGraph:
    """Convert an edges table to a directed lineage graph.

    Parameters
    ----------
    edges_df : polars.DataFrame
        Table with ``parent_key``, ``child_key``, ``event``, ``confidence``
        columns (see :func:`momia3.schema.edges_schema`).

    Returns
    -------
    networkx.DiGraph
        Nodes are ``cell_key`` ints; edge attributes are ``event`` (str)
        and ``confidence`` (float).
    """
    missing = {"parent_key", "child_key"} - set(edges_df.columns)
    if missing:
        raise _schema.SchemaError(f"edges table missing columns: {sorted(missing)}")
    optional = [c for c in ("event", "confidence") if c in edges_df.columns]
    graph = nx.DiGraph()
    for row in edges_df.select(["parent_key", "child_key", *optional]).iter_rows(
        named=True
    ):
        graph.add_edge(
            int(row["parent_key"]),
            int(row["child_key"]),
            event=row.get("event"),
            confidence=(
                None if row.get("confidence") is None else float(row["confidence"])
            ),
        )
    return graph


def nx_to_edges(graph: nx.DiGraph) -> pl.DataFrame:
    """Convert a lineage graph back to a schema-conforming edges table.

    Parameters
    ----------
    graph : networkx.DiGraph
        Nodes are ``cell_key`` ints; edge attributes may include ``event``
        (defaults to ``"continue"``) and ``confidence`` (defaults to 1.0).

    Returns
    -------
    polars.DataFrame
        Edges table validated against
        :func:`momia3.schema.edges_schema`; ``event`` is categorical.
    """
    parents: list[int] = []
    children: list[int] = []
    events: list[str] = []
    confidences: list[float] = []
    for parent, child, data in graph.edges(data=True):
        parents.append(int(parent))
        children.append(int(child))
        events.append(str(data.get("event") or "continue"))
        confidence = data.get("confidence")
        confidences.append(1.0 if confidence is None else float(confidence))
    table = pa.table(
        {
            "parent_key": pa.array(parents, type=pa.uint64()),
            "child_key": pa.array(children, type=pa.uint64()),
            "event": pa.array(events, type=pa.string()),
            "confidence": pa.array(confidences, type=pa.float32()),
        }
    )
    return pl.from_arrow(_schema.validate_edges(table))
