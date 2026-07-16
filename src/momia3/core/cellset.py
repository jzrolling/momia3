"""CellSet: a polars DataFrame conforming to the cells schema (§2.3) with
vectorized scalar access and per-cell geometry views."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from momia3 import schema as _schema
from momia3.storage import tables as _tables

logger = logging.getLogger(__name__)

__all__ = ["CellSet", "CellGeometry"]


@dataclass(frozen=True)
class CellGeometry:
    """Geometry of one cell instance as numpy arrays.

    Attributes
    ----------
    outline : numpy.ndarray or None
        ``(N, 2)`` float32 array of ``(x, y)`` outline vertices (subpixel,
        global coordinates).
    midline : list of numpy.ndarray or None
        One ``(N, 2)`` float32 array per midline branch.
    width_profile : numpy.ndarray or None
        ``(N,)`` float32 widths along the midline, in microns.
    bbox : tuple of int or None
        ``(x0, y0, x1, y1)`` in pixels.
    centroid : tuple of float or None
        ``(x, y)`` in pixels.
    """

    outline: np.ndarray | None
    midline: list[np.ndarray] | None
    width_profile: np.ndarray | None
    bbox: tuple[int, int, int, int] | None
    centroid: tuple[float, float] | None


def _xy_to_array(struct: dict[str, Any] | None, dtype: np.dtype) -> np.ndarray | None:
    """Convert an ``{"x": [...], "y": [...]}`` struct to an ``(N, 2)`` array."""
    if struct is None:
        return None
    x = np.asarray(struct["x"], dtype=dtype)
    y = np.asarray(struct["y"], dtype=dtype)
    return np.stack([x, y], axis=1)


class CellSet:
    """Wrapper around a cells-schema polars DataFrame.

    Parameters
    ----------
    df : polars.DataFrame
        Table conforming to :func:`momia3.schema.cells_schema`.
    validate : bool
        Validate/canonicalize against the cells schema on construction
        (default True). Internal views pass False to skip re-validation.

    Raises
    ------
    momia3.schema.SchemaError
        If ``df`` fails schema validation.
    """

    def __init__(self, df: pl.DataFrame, *, validate: bool = True) -> None:
        if validate:
            canonical = _schema.validate_cells(df.to_arrow())
            self._df = pl.from_arrow(canonical)
        else:
            self._df = df

    # ------------------------------------------------------------------
    # basics
    # ------------------------------------------------------------------
    @property
    def df(self) -> pl.DataFrame:
        """Underlying polars DataFrame (do not mutate in place)."""
        return self._df

    def __len__(self) -> int:
        return self._df.height

    def scalar_features(self) -> pl.DataFrame:
        """Vectorized view of identity + scalar columns (geometry dropped).

        Returns
        -------
        polars.DataFrame
            All non-geometry columns — identity/bookkeeping, scalar
            morphology, intensity clusters and extensions. Cheap zero-copy
            projection suitable for bulk statistics.
        """
        drop = [c for c in _schema.GEOMETRY_COLUMNS if c in self._df.columns]
        return self._df.drop(drop)

    def get_geometry(self, cell_key: int) -> CellGeometry:
        """Geometry arrays of one cell instance.

        Parameters
        ----------
        cell_key : int
            Composite key (see :mod:`momia3.keys`).

        Returns
        -------
        CellGeometry
            Outline/midline/width profile as numpy arrays (None for null
            geometry fields), plus bbox and centroid.

        Raises
        ------
        KeyError
            If no row carries ``cell_key``.
        """
        rows = self._df.filter(pl.col("cell_key") == int(cell_key))
        if rows.height == 0:
            raise KeyError(f"no cell with key {cell_key}")
        row = rows.row(0, named=True)

        outline = _xy_to_array(row.get("outline"), np.float32)
        midline_raw = row.get("midline")
        midline = (
            None
            if midline_raw is None
            else [
                branch
                for branch in (
                    _xy_to_array(b, np.float32) for b in midline_raw
                )
                if branch is not None
            ]
        )
        width_raw = row.get("width_profile")
        width_profile = (
            None if width_raw is None else np.asarray(width_raw, dtype=np.float32)
        )
        bbox_raw = row.get("bbox")
        bbox = (
            None
            if bbox_raw is None
            else (
                int(bbox_raw["x0"]),
                int(bbox_raw["y0"]),
                int(bbox_raw["x1"]),
                int(bbox_raw["y1"]),
            )
        )
        centroid_raw = row.get("centroid")
        centroid = (
            None
            if centroid_raw is None
            else (float(centroid_raw["x"]), float(centroid_raw["y"]))
        )
        return CellGeometry(
            outline=outline,
            midline=midline,
            width_profile=width_profile,
            bbox=bbox,
            centroid=centroid,
        )

    def filter(self, *predicates: pl.Expr, **equals: Any) -> CellSet:
        """Return a filtered view.

        Parameters
        ----------
        *predicates : polars.Expr
            Boolean expressions, e.g. ``pl.col("length_um") > 3``.
        **equals
            Column equality shortcuts, e.g. ``fov_id=1, frame=7``.

        Returns
        -------
        CellSet
        """
        df = self._df.filter(*predicates) if predicates else self._df
        for column, value in equals.items():
            if column not in df.columns:
                raise KeyError(f"unknown column {column!r}")
            df = df.filter(pl.col(column) == value)
        return CellSet(df, validate=False)

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    def to_parquet(self, path: str | Path, *, channels: tuple[str, ...] = ()) -> Path:
        """Validate and write to parquet (schema metadata preserved)."""
        return _tables.write_cells(self._df, path, channels=channels)

    @classmethod
    def from_parquet(cls, path: str | Path, *, validate: bool = True) -> CellSet:
        """Read a cells parquet file.

        Parameters
        ----------
        path : str or pathlib.Path
            Parquet file written by :meth:`to_parquet` (or any
            schema-compatible source).
        validate : bool
            Re-validate/canonicalize against the cells schema (default).
        """
        return cls(_tables.read_cells(path), validate=validate)
