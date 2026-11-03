"""Arrow schema definitions for the momia3 tables (ARCHITECTURE.md §2.3–§2.4).

This module is the *single source of truth* for the three on-disk tables:

- ``cells.parquet``  — one row per cell instance, nested geometry columns,
  scalar morphology columns carrying ``unit``/``semantic`` field metadata,
  per-channel intensity clusters and optional extension columns.
- ``edges.parquet``  — lineage edge list ``(parent_key, child_key, event,
  confidence)`` with a categorical ``event`` column.
- ``tracks.parquet`` — per-track summary with a categorical ``fate`` column.

Builders (:func:`empty_cells_table`, :func:`empty_edges_table`,
:func:`empty_tracks_table`) create zero-row tables with the canonical
schema; validators (:func:`validate_cells`, :func:`validate_edges`,
:func:`validate_tracks`) check required columns, cast them to the canonical
types, and pass through extra (e.g. channel-specific) columns unchanged.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc

SCHEMA_VERSION: str = "1"
"""Version tag embedded in schema-level metadata."""

__all__ = [
    "SCHEMA_VERSION",
    "SchemaError",
    "BBOX_T",
    "CENTROID_T",
    "OUTLINE_T",
    "MIDLINE_T",
    "WIDTH_PROFILE_T",
    "COORDS_T",
    "PUNCTA_T",
    "EVENT_T",
    "FATE_T",
    "EVENT_VALUES",
    "FATE_VALUES",
    "INTENSITY_STATS",
    "IDENTITY_COLUMNS",
    "GEOMETRY_COLUMNS",
    "SCALAR_MORPHOLOGY_COLUMNS",
    "cells_schema",
    "edges_schema",
    "tracks_schema",
    "empty_cells_table",
    "empty_edges_table",
    "empty_tracks_table",
    "validate_cells",
    "validate_edges",
    "validate_tracks",
]


class SchemaError(ValueError):
    """Raised when a table does not conform to a momia3 table schema."""


# ---------------------------------------------------------------------------
# Nested geometry types (§2.3)
# ---------------------------------------------------------------------------

BBOX_T: pa.DataType = pa.struct(
    [("x0", pa.int32()), ("y0", pa.int32()), ("x1", pa.int32()), ("y1", pa.int32())]
)
"""Axis-aligned bounding box in pixels (x0/y0 inclusive, x1/y1 exclusive)."""

CENTROID_T: pa.DataType = pa.struct([("x", pa.float32()), ("y", pa.float32())])
"""Subpixel centroid in pixels."""

OUTLINE_T: pa.DataType = pa.struct(
    [("x", pa.list_(pa.float32())), ("y", pa.list_(pa.float32()))]
)
"""Subpixel closed outline in global coordinates."""

MIDLINE_T: pa.DataType = pa.list_(OUTLINE_T)
"""Multi-branch midline: list of branches, each an outline-like polyline."""

WIDTH_PROFILE_T: pa.DataType = pa.list_(pa.float32())
"""Width along the resampled midline, in microns."""

COORDS_T: pa.DataType = pa.struct(
    [("x", pa.list_(pa.int32())), ("y", pa.list_(pa.int32()))]
)
"""Mask pixel coordinates (optional; normally derived from the labels zarr)."""

PUNCTA_T: pa.DataType = pa.list_(pa.struct([("x", pa.float32()), ("y", pa.float32())]))
"""FISH puncta localizations, subpixel."""

# ---------------------------------------------------------------------------
# Categorical types (§2.4)
# ---------------------------------------------------------------------------

EVENT_VALUES: tuple[str, ...] = (
    "continue",
    "division",
    "merge",
    "appear",
    "disappear",
    "gap",
)
"""Allowed values of the ``edges.event`` column."""

FATE_VALUES: tuple[str, ...] = ("divide", "disappear", "edge", "incomplete")
"""Allowed values of the ``tracks.fate`` column."""

EVENT_T: pa.DataType = pa.dictionary(pa.int32(), pa.string())
"""Categorical type of the ``edges.event`` column."""

FATE_T: pa.DataType = pa.dictionary(pa.int32(), pa.string())
"""Categorical type of the ``tracks.fate`` column."""

INTENSITY_STATS: tuple[str, ...] = (
    "mean",
    "max",
    "min",
    "std",
    "p25",
    "p75",
    "integrated",
)
"""Statistics computed per channel (``{ch}_{stat}`` columns)."""


# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

IDENTITY_COLUMNS: tuple[str, ...] = (
    "cell_key",
    "fov_id",
    "frame",
    "label",
    "track_id",
    "parent_track",
    "touching_edge",
    "include",
)
"""Identity / bookkeeping columns of the cells table."""

GEOMETRY_COLUMNS: tuple[str, ...] = (
    "bbox",
    "centroid",
    "outline",
    "midline",
    "width_profile",
    "coords",
)
"""Nested geometry columns of the cells table."""

# (name, unit, semantic) for every scalar morphology column (§2.3).
_SCALAR_MORPHOLOGY: tuple[tuple[str, str, str], ...] = (
    ("area", "px^2", "morphology.size"),
    ("perimeter", "px", "morphology.size"),
    ("major_axis_length", "px", "morphology.size"),
    ("minor_axis_length", "px", "morphology.size"),
    ("aspect_ratio", "1", "morphology.shape"),
    ("eccentricity", "1", "morphology.shape"),
    ("solidity", "1", "morphology.shape"),
    ("ellipse_dev", "1", "morphology.shape"),
    ("sagitta_ratio", "1", "morphology.shape"),
    ("extent", "1", "morphology.shape"),
    ("circularity", "1", "morphology.shape"),
    ("compactness", "1", "morphology.shape"),
    ("hu_0", "1", "morphology.hu_moment"),
    ("hu_1", "1", "morphology.hu_moment"),
    ("hu_2", "1", "morphology.hu_moment"),
    ("hu_3", "1", "morphology.hu_moment"),
    ("hu_4", "1", "morphology.hu_moment"),
    ("hu_5", "1", "morphology.hu_moment"),
    ("hu_6", "1", "morphology.hu_moment"),
    ("curv_min", "rad/px", "morphology.curvature"),
    ("curv_max", "rad/px", "morphology.curvature"),
    ("curv_mean", "rad/px", "morphology.curvature"),
    ("curv_q1", "rad/px", "morphology.curvature"),
    ("curv_q3", "rad/px", "morphology.curvature"),
    ("length_um", "um", "morphology.length"),
    ("width_median_um", "um", "morphology.width"),
    ("width_std_um", "um", "morphology.width"),
    ("width_max_um", "um", "morphology.width"),
    ("width_q1_um", "um", "morphology.width"),
    ("width_q3_um", "um", "morphology.width"),
    ("sinuosity", "1", "morphology.shape"),
    ("branched", "1", "morphology.topology"),
    ("pole1_x", "px", "morphology.pole"),
    ("pole1_y", "px", "morphology.pole"),
    ("pole2_x", "px", "morphology.pole"),
    ("pole2_y", "px", "morphology.pole"),
    ("septum_pos", "1", "division.septum_position"),
)

SCALAR_MORPHOLOGY_COLUMNS: tuple[str, ...] = tuple(n for n, _, _ in _SCALAR_MORPHOLOGY)
"""Names of all scalar morphology (float32) columns."""


def _meta(**entries: str) -> dict[bytes, bytes]:
    """Build pyarrow field metadata from string key/value pairs."""
    return {k.encode("utf-8"): v.encode("utf-8") for k, v in entries.items()}


def _intensity_fields(channels: Iterable[str]) -> list[pa.Field]:
    """Build the per-channel intensity column cluster (§2.3)."""
    fields: list[pa.Field] = []
    for ch in channels:
        for stat in INTENSITY_STATS:
            fields.append(
                pa.field(
                    f"{ch}_{stat}",
                    pa.float32(),
                    metadata=_meta(
                        unit="intensity", semantic=f"intensity.{stat}", channel=ch
                    ),
                )
            )
        fields.append(
            pa.field(
                f"puncta_{ch}",
                PUNCTA_T,
                metadata=_meta(semantic="puncta.fish", unit="px", channel=ch),
            )
        )
    return fields


# ---------------------------------------------------------------------------
# Schema factories
# ---------------------------------------------------------------------------


def cells_schema(channels: Sequence[str] = ()) -> pa.Schema:
    """Canonical schema of ``cells.parquet``.

    Parameters
    ----------
    channels : sequence of str, optional
        Channel names; each contributes an intensity cluster
        ``{ch}_mean/{ch}_max/{ch}_min/{ch}_std/{ch}_p25/{ch}_p75/
        {ch}_integrated`` plus an optional ``puncta_{ch}`` column.

    Returns
    -------
    pyarrow.Schema
        Schema with identity, geometry, scalar morphology, extension and
        (per-channel) intensity columns. Scalar morphology fields carry
        ``unit``/``semantic`` metadata; schema-level metadata records the
        spec section and schema version.
    """
    fields: list[pa.Field] = [
        pa.field(
            "cell_key",
            pa.uint64(),
            metadata=_meta(semantic="identity.primary_key", spec="ARCHITECTURE.md §2.1"),
        ),
        pa.field(
            "fov_id",
            pa.uint32(),
            metadata=_meta(semantic="identity.partition_key"),
        ),
        pa.field(
            "frame", pa.uint32(), metadata=_meta(semantic="identity.frame", unit="frame")
        ),
        pa.field("label", pa.uint32(), metadata=_meta(semantic="identity.label")),
        pa.field("track_id", pa.uint32(), metadata=_meta(semantic="identity.track")),
        pa.field(
            "parent_track", pa.uint32(), metadata=_meta(semantic="lineage.parent_track")
        ),
        pa.field(
            "touching_edge", pa.bool_(), metadata=_meta(semantic="qc.touching_edge")
        ),
        pa.field("include", pa.bool_(), metadata=_meta(semantic="qc.include")),
        pa.field(
            "bbox", BBOX_T, metadata=_meta(semantic="geometry.bbox", unit="px")
        ),
        pa.field(
            "centroid",
            CENTROID_T,
            metadata=_meta(semantic="geometry.centroid", unit="px"),
        ),
        pa.field(
            "outline",
            OUTLINE_T,
            metadata=_meta(
                semantic="geometry.outline",
                unit="px",
                description="subpixel closed outline, global coordinates",
            ),
        ),
        pa.field(
            "midline",
            MIDLINE_T,
            metadata=_meta(
                semantic="geometry.midline",
                unit="px",
                description="multi-branch midline; main branch = longest geodesic",
            ),
        ),
        pa.field(
            "width_profile",
            WIDTH_PROFILE_T,
            metadata=_meta(semantic="geometry.width_profile", unit="um"),
        ),
        pa.field(
            "coords",
            COORDS_T,
            metadata=_meta(
                semantic="geometry.coords",
                unit="px",
                description="optional; defaults to derivation from labels zarr",
            ),
        ),
    ]
    for name, unit, semantic in _SCALAR_MORPHOLOGY:
        fields.append(
            pa.field(name, pa.float32(), metadata=_meta(unit=unit, semantic=semantic))
        )
    # Optional extension columns (§2.3, nullable).
    fields.append(
        pa.field(
            "embedding",
            pa.list_(pa.float32()),
            metadata=_meta(semantic="ml.embedding"),
        )
    )
    fields.append(
        pa.field(
            "annotation", pa.int8(), metadata=_meta(semantic="annotation.manual")
        )
    )
    fields.extend(_intensity_fields(channels))
    return pa.schema(
        fields,
        metadata=_meta(
            description="momia3 cells table: one row per cell instance",
            spec="ARCHITECTURE.md §2.3",
            schema_version=SCHEMA_VERSION,
        ),
    )


def edges_schema() -> pa.Schema:
    """Canonical schema of ``edges.parquet`` (§2.4).

    Returns
    -------
    pyarrow.Schema
        ``(parent_key: u64, child_key: u64, event: categorical, confidence:
        f32)``. ``parent_key == 0`` marks the root sentinel of ``appear``
        events (see :data:`momia3.keys.ROOT_KEY`).
    """
    return pa.schema(
        [
            pa.field(
                "parent_key", pa.uint64(), metadata=_meta(semantic="lineage.parent")
            ),
            pa.field(
                "child_key", pa.uint64(), metadata=_meta(semantic="lineage.child")
            ),
            pa.field(
                "event",
                EVENT_T,
                metadata=_meta(
                    semantic="lineage.event", values="|".join(EVENT_VALUES)
                ),
            ),
            pa.field(
                "confidence",
                pa.float32(),
                metadata=_meta(semantic="lineage.confidence", unit="1"),
            ),
        ],
        metadata=_meta(
            description="momia3 lineage edge table",
            spec="ARCHITECTURE.md §2.4",
            schema_version=SCHEMA_VERSION,
        ),
    )


def tracks_schema() -> pa.Schema:
    """Canonical schema of ``tracks.parquet`` (§2.4).

    Returns
    -------
    pyarrow.Schema
        ``(track_id, fov_id, birth_frame, last_frame, division_frame?,
        parent_track?, n_frames, fate: categorical)``.
    """
    return pa.schema(
        [
            pa.field(
                "track_id", pa.uint32(), metadata=_meta(semantic="identity.track")
            ),
            pa.field(
                "fov_id",
                pa.uint32(),
                metadata=_meta(semantic="identity.partition_key"),
            ),
            pa.field(
                "birth_frame",
                pa.uint32(),
                metadata=_meta(semantic="track.birth_frame", unit="frame"),
            ),
            pa.field(
                "last_frame",
                pa.uint32(),
                metadata=_meta(semantic="track.last_frame", unit="frame"),
            ),
            pa.field(
                "division_frame",
                pa.uint32(),
                metadata=_meta(semantic="track.division_frame", unit="frame"),
            ),
            pa.field(
                "parent_track",
                pa.uint32(),
                metadata=_meta(semantic="lineage.parent_track"),
            ),
            pa.field(
                "n_frames",
                pa.uint32(),
                metadata=_meta(semantic="track.n_frames", unit="frame"),
            ),
            pa.field(
                "fate",
                FATE_T,
                metadata=_meta(semantic="track.fate", values="|".join(FATE_VALUES)),
            ),
        ],
        metadata=_meta(
            description="momia3 track summary table",
            spec="ARCHITECTURE.md §2.4",
            schema_version=SCHEMA_VERSION,
        ),
    )


# ---------------------------------------------------------------------------
# Empty table builders
# ---------------------------------------------------------------------------


def _empty_table(schema: pa.Schema) -> pa.Table:
    """Create a zero-row table with exactly ``schema``."""
    return pa.Table.from_arrays(
        [pa.array([], type=field.type) for field in schema], schema=schema
    )


def empty_cells_table(channels: Sequence[str] = ()) -> pa.Table:
    """Zero-row cells table with the canonical schema.

    Parameters
    ----------
    channels : sequence of str, optional
        Channel names forwarded to :func:`cells_schema`.

    Returns
    -------
    pyarrow.Table
        Empty table conforming to :func:`cells_schema`.
    """
    return _empty_table(cells_schema(channels))


def empty_edges_table() -> pa.Table:
    """Zero-row edges table with the canonical schema."""
    return _empty_table(edges_schema())


def empty_tracks_table() -> pa.Table:
    """Zero-row tracks table with the canonical schema."""
    return _empty_table(tracks_schema())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _cast_column(col: pa.ChunkedArray, target: pa.DataType, name: str) -> pa.ChunkedArray:
    """Cast one column to the canonical type, normalizing categoricals.

    Parameters
    ----------
    col : pyarrow.ChunkedArray
        Input column.
    target : pyarrow.DataType
        Canonical type from the schema.
    name : str
        Column name (for error messages).

    Returns
    -------
    pyarrow.ChunkedArray
        Column with type ``target``.

    Raises
    ------
    SchemaError
        If the column cannot be converted to the target type.
    """
    if col.type == target:
        return col
    try:
        return col.cast(target)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as exc:
        if pa.types.is_dictionary(target):
            # Normalize any string/categorical representation to the canonical
            # dictionary type: decode to plain utf8, re-encode, cast index type.
            try:
                values = col.combine_chunks().cast(pa.string())
                encoded = pc.dictionary_encode(values)
                return encoded.cast(target)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
                pass
        raise SchemaError(
            f"column {name!r}: cannot cast {col.type} to {target}"
        ) from exc


def _validate(table: pa.Table, schema: pa.Schema, table_name: str) -> pa.Table:
    """Validate ``table`` against ``schema``; return a canonicalized copy.

    Required columns must be present and castable to the canonical types.
    Extra columns (e.g. channel intensity clusters) are passed through
    unchanged, appended after the canonical columns.
    """
    if not isinstance(table, pa.Table):
        raise TypeError(
            f"expected pyarrow.Table for {table_name}, got {type(table).__name__}"
        )
    missing = [f.name for f in schema if f.name not in table.column_names]
    if missing:
        raise SchemaError(f"{table_name} table missing required columns: {missing}")
    arrays = [
        _cast_column(table.column(field.name), field.type, field.name)
        for field in schema
    ]
    out = pa.Table.from_arrays(arrays, schema=schema)
    for extra in table.column_names:
        if extra not in schema.names:
            out = out.append_column(extra, table.column(extra))
    return out


def validate_cells(table: pa.Table, channels: Sequence[str] = ()) -> pa.Table:
    """Validate a cells table; return a canonicalized copy.

    Parameters
    ----------
    table : pyarrow.Table
        Candidate cells table (e.g. ``polars.DataFrame.to_arrow()``).
    channels : sequence of str, optional
        Channels whose intensity clusters should also be validated. Extra
        columns beyond the canonical schema are always passed through.

    Returns
    -------
    pyarrow.Table
        Table with canonical column order, types and field metadata.

    Raises
    ------
    SchemaError
        If required columns are missing or not castable.
    TypeError
        If ``table`` is not a :class:`pyarrow.Table`.
    """
    return _validate(table, cells_schema(channels), "cells")


def validate_edges(table: pa.Table) -> pa.Table:
    """Validate an edges table; return a canonicalized copy.

    Raises
    ------
    SchemaError
        If required columns are missing or not castable.
    """
    return _validate(table, edges_schema(), "edges")


def validate_tracks(table: pa.Table) -> pa.Table:
    """Validate a tracks table; return a canonicalized copy.

    Raises
    ------
    SchemaError
        If required columns are missing or not castable.
    """
    return _validate(table, tracks_schema(), "tracks")
