"""polars-based parquet IO for the cells/edges/tracks tables (§2.3–§2.4).

Writes always validate against the canonical Arrow schema
(:mod:`momia3.schema`) and are performed through pyarrow so that field
metadata (units/semantics) is preserved in the parquet file. Reads return
:class:`polars.DataFrame`; categorical columns come back as
``pl.Categorical``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Sequence

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from momia3 import schema as _schema

logger = logging.getLogger(__name__)

_VALIDATORS = {
    "cells": _schema.validate_cells,
    "edges": _schema.validate_edges,
    "tracks": _schema.validate_tracks,
}

_DEFAULT_SUBSET = {
    "cells": ("cell_key",),
    "edges": ("parent_key", "child_key", "event"),
    "tracks": ("track_id",),
}

__all__ = [
    "write_table",
    "read_table",
    "write_cells",
    "write_edges",
    "write_tracks",
    "read_cells",
    "read_edges",
    "read_tracks",
    "partition_by_fov",
    "merge_frames",
    "append_table",
]


def _to_arrow(df: pl.DataFrame | pa.Table) -> pa.Table:
    if isinstance(df, pa.Table):
        return df
    if isinstance(df, pl.DataFrame):
        return df.to_arrow()
    raise TypeError(f"expected polars.DataFrame or pyarrow.Table, got {type(df).__name__}")


def write_table(
    df: pl.DataFrame | pa.Table,
    path: str | Path,
    kind: str,
    *,
    channels: Sequence[str] = (),
) -> Path:
    """Validate and write one of the momia3 tables to parquet.

    Parameters
    ----------
    df : polars.DataFrame or pyarrow.Table
        Table content. Must satisfy the canonical schema of ``kind``.
    path : str or pathlib.Path
        Destination parquet file.
    kind : {"cells", "edges", "tracks"}
        Which schema to validate against.
    channels : sequence of str, optional
        Channel names whose intensity clusters are additionally validated
        (cells tables only).

    Returns
    -------
    pathlib.Path
        The written path.

    Raises
    ------
    ValueError
        If ``kind`` is unknown.
    momia3.schema.SchemaError
        If the table fails validation.
    """
    if kind not in _VALIDATORS:
        raise ValueError(f"unknown table kind {kind!r}; expected one of {_VALIDATORS}")
    validator = _VALIDATORS[kind]
    table = _to_arrow(df)
    validated = (
        validator(table, channels) if kind == "cells" else validator(table)  # type: ignore[misc]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(validated, str(path), compression="zstd")
    logger.info("wrote %s table (%d rows) to %s", kind, validated.num_rows, path)
    return path


def read_table(path: str | Path) -> pl.DataFrame:
    """Read a parquet table into a :class:`polars.DataFrame`.

    Parameters
    ----------
    path : str or pathlib.Path
        Parquet file previously written by :func:`write_table` (or any
        schema-compatible parquet).

    Returns
    -------
    polars.DataFrame
    """
    return pl.read_parquet(str(path))


def write_cells(
    df: pl.DataFrame | pa.Table,
    path: str | Path,
    *,
    channels: Sequence[str] = (),
) -> Path:
    """Validate and write ``cells.parquet``. See :func:`write_table`."""
    return write_table(df, path, "cells", channels=channels)


def write_edges(df: pl.DataFrame | pa.Table, path: str | Path) -> Path:
    """Validate and write ``edges.parquet``. See :func:`write_table`."""
    return write_table(df, path, "edges")


def write_tracks(df: pl.DataFrame | pa.Table, path: str | Path) -> Path:
    """Validate and write ``tracks.parquet``. See :func:`write_table`."""
    return write_table(df, path, "tracks")


def read_cells(path: str | Path) -> pl.DataFrame:
    """Read ``cells.parquet`` into polars."""
    return read_table(path)


def read_edges(path: str | Path) -> pl.DataFrame:
    """Read ``edges.parquet`` into polars."""
    return read_table(path)


def read_tracks(path: str | Path) -> pl.DataFrame:
    """Read ``tracks.parquet`` into polars."""
    return read_table(path)


def partition_by_fov(df: pl.DataFrame) -> Iterator[tuple[int, pl.DataFrame]]:
    """Yield ``(fov_id, sub_frame)`` pairs ordered by ``fov_id``.

    Parameters
    ----------
    df : polars.DataFrame
        Any table carrying a ``fov_id`` column (cells or tracks).

    Yields
    ------
    tuple of (int, polars.DataFrame)
        ``fov_id`` and the rows belonging to that fov, preserving row order
        within each partition.
    """
    if "fov_id" not in df.columns:
        raise _schema.SchemaError("table has no 'fov_id' column")
    parts = df.partition_by("fov_id", as_dict=True, maintain_order=True)
    for key in sorted(parts):
        yield int(key[0]), parts[key]


def merge_frames(
    a: pl.DataFrame,
    b: pl.DataFrame,
    subset: Sequence[str] = ("cell_key",),
) -> pl.DataFrame:
    """Concatenate two tables, keeping the ``b`` row on key collisions.

    Categorical columns are temporarily cast to strings for the merge
    (re-encoded on the next schema-validating write), which keeps merges
    robust across unequal categorical dictionaries.

    Parameters
    ----------
    a, b : polars.DataFrame
        Input tables (schemas may differ in extra columns; the merge is
        diagonal).
    subset : sequence of str
        Key columns used for de-duplication.

    Returns
    -------
    polars.DataFrame
        Merged table sorted by the first subset column.
    """
    def _decategorize(df: pl.DataFrame) -> pl.DataFrame:
        casts = [
            pl.col(name).cast(pl.String)
            for name, dtype in zip(df.columns, df.dtypes, strict=True)
            if dtype == pl.Categorical
        ]
        return df.with_columns(casts) if casts else df

    merged = pl.concat([_decategorize(a), _decategorize(b)], how="diagonal_relaxed")
    merged = merged.unique(subset=list(subset), keep="last", maintain_order=True)
    return merged.sort(subset[0])


def append_table(
    df: pl.DataFrame,
    path: str | Path,
    kind: str,
    *,
    subset: Sequence[str] | None = None,
    channels: Sequence[str] = (),
) -> Path:
    """Merge ``df`` into an existing parquet table (or create it) and rewrite.

    Parameters
    ----------
    df : polars.DataFrame
        New rows.
    path : str or pathlib.Path
        Target parquet file; read and merged when it already exists.
    kind : {"cells", "edges", "tracks"}
        Table kind (schema + default dedup keys).
    subset : sequence of str, optional
        Dedup key columns; defaults to the natural key of ``kind``
        (``cell_key`` / ``(parent_key, child_key, event)`` / ``track_id``).
    channels : sequence of str, optional
        Channel names forwarded to :func:`write_table`.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    if kind not in _VALIDATORS:
        raise ValueError(f"unknown table kind {kind!r}; expected one of {_VALIDATORS}")
    path = Path(path)
    keys = tuple(subset) if subset is not None else _DEFAULT_SUBSET[kind]
    if path.exists():
        existing = read_table(path)
        merged = merge_frames(existing, df, subset=keys)
    else:
        merged = df
    return write_table(merged, path, kind, channels=channels)
