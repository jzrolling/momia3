"""Shared rod-cell frame derived from stored midline/outline geometry.

Every rod-like measurement (rod coordinates, 1-D profiles, demographic
projection) needs the same quantities: the main midline branch, its
uniform arc-length resampling, local tangents/normals, half widths and
the outline ring KD-tree. Building them once per cell and sharing the
result avoids re-parsing the geometry columns and rebuilding KD-trees
for every downstream operator.

Coordinate convention follows the rest of ``momia3``: ``(x, y)`` in
global image pixels, ``x`` = column, ``y`` = row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
from scipy import ndimage
from scipy.spatial import cKDTree

__all__ = [
    "RodFrame",
    "build_rod_frame",
    "rod_frames",
    "main_branch",
    "outline_ring",
    "sample_line_profile",
]

_EPS = 1e-12


def _to_pa(table: pa.Table | Any) -> pa.Table:
    if isinstance(table, pa.Table):
        return table
    to_arrow = getattr(table, "to_arrow", None)
    if callable(to_arrow):
        return to_arrow()
    raise TypeError(
        f"expected pyarrow.Table or polars.DataFrame, got {type(table).__name__}"
    )


def main_branch(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """Longest midline branch of one cells-table row as ``(x, y)``."""
    midline = row.get("midline")
    if not midline:
        return None
    best_x: np.ndarray | None = None
    best_y: np.ndarray | None = None
    best_len = -1.0
    for branch in midline:
        if branch is None:
            continue
        x = np.asarray(branch["x"], dtype=np.float64)
        y = np.asarray(branch["y"], dtype=np.float64)
        if x.size < 2:
            continue
        length = float(np.hypot(np.diff(x), np.diff(y)).sum())
        if length > best_len:
            best_len = length
            best_x, best_y = x, y
    if best_x is None or best_y is None:
        return None
    return best_x, best_y


def outline_ring(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """Closed ``(x, y)`` outline ring of one cells-table row (first point
    repeated at the end)."""
    outline = row.get("outline")
    if outline is None:
        return None
    x = np.asarray(outline["x"], dtype=np.float64)
    y = np.asarray(outline["y"], dtype=np.float64)
    if x.size < 3:
        return None
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.concatenate([x, x[:1]])
        y = np.concatenate([y, y[:1]])
    return x, y


def _arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    seg = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _resample(
    x: np.ndarray, y: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = _arc_length(x, y)
    total = float(s[-1])
    if total <= _EPS:
        return np.repeat(x[:1], n), np.repeat(y[:1], n), np.zeros(n)
    targets = np.linspace(0.0, total, n)
    return np.interp(targets, s, x), np.interp(targets, s, y), targets / total


@dataclass
class RodFrame:
    """Precomputed geometry of one rod-like cell.

    Attributes
    ----------
    cell_key, label : int
        Identity of the cell.
    x, y : numpy.ndarray
        Main-branch midline resampled to ``n_points`` arc-length positions.
    l : numpy.ndarray
        Normalized longitudinal coordinate ``[0, 1]`` of the samples.
    nx, ny : numpy.ndarray
        Unit left normals at the samples.
    width_px : numpy.ndarray
        Full width in pixels at each sample (width profile when present,
        otherwise twice the nearest-outline distance).
    outline_x, outline_y : numpy.ndarray or None
        Closed outline ring, if stored.
    chord_px : float
        Straight pole-to-pole chord length of the resampled midline.
    pixel_microns : float
        Physical pixel size.
    """

    cell_key: int
    label: int
    x: np.ndarray
    y: np.ndarray
    l: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    width_px: np.ndarray
    outline_x: np.ndarray | None
    outline_y: np.ndarray | None
    chord_px: float
    pixel_microns: float
    _tree: Any = field(default=None, repr=False, compare=False)

    def tree(self) -> cKDTree:
        """KD-tree over the resampled midline points (built once)."""
        if self._tree is None:
            self._tree = cKDTree(np.column_stack([self.x, self.y]))
        return self._tree

    def outline_distance(self) -> np.ndarray:
        """Distance from every midline sample to the nearest outline point."""
        if self.outline_x is None or self.outline_y is None:
            return np.zeros(self.n_samples, dtype=np.float64)
        ring_tree = cKDTree(np.column_stack([self.outline_x, self.outline_y]))
        dist, _ = ring_tree.query(np.column_stack([self.x, self.y]))
        return np.asarray(dist, dtype=np.float64)

    @property
    def n_samples(self) -> int:
        return int(self.x.size)

    def project(self, px: np.ndarray, py: np.ndarray) -> dict[str, np.ndarray]:
        """Project pixel coordinates onto the frame (rod coordinates).

        Mirrors :func:`momia3.ops.localization.rod_coordinates`: nearest
        midline sample gives the longitudinal coordinate; the signed
        normal offset divided by the local half width gives the radial
        coordinate; pole distances are normalized by the pole chord.
        """
        pixel_xy = np.column_stack(
            [np.asarray(px, dtype=np.float64), np.asarray(py, dtype=np.float64)]
        )
        _, nearest = self.tree().query(pixel_xy)
        nearest = np.asarray(nearest, dtype=np.int64)
        l_pix = self.l[nearest]
        delta = pixel_xy - np.column_stack([self.x, self.y])[nearest]
        signed = delta[:, 0] * self.nx[nearest] + delta[:, 1] * self.ny[nearest]
        half = np.maximum(self.width_px * 0.5, _EPS)
        radial = signed / half[nearest]
        pole1 = (
            np.hypot(
                np.asarray(px, dtype=np.float64) - self.x[0],
                np.asarray(py, dtype=np.float64) - self.y[0],
            )
            / self.chord_px
        )
        pole2 = (
            np.hypot(
                np.asarray(px, dtype=np.float64) - self.x[-1],
                np.asarray(py, dtype=np.float64) - self.y[-1],
            )
            / self.chord_px
        )
        return {
            "longitudinal": l_pix.astype(np.float32),
            "radial": radial.astype(np.float32),
            "signed_distance": signed.astype(np.float32),
            "width_px": self.width_px[nearest].astype(np.float32),
            "pole1_distance": np.minimum(pole1, 1e6).astype(np.float32),
            "pole2_distance": np.minimum(pole2, 1e6).astype(np.float32),
        }


def build_rod_frame(
    cell_key: int,
    label: int,
    row: Mapping[str, Any],
    n_points: int = 32,
    pixel_microns: float = 1.0,
) -> RodFrame | None:
    """Build a :class:`RodFrame` from one cells-table row, or ``None``
    when the row carries no usable midline."""
    if n_points < 3:
        raise ValueError("n_points must be >= 3")
    if pixel_microns <= 0:
        raise ValueError("pixel_microns must be positive")
    branch = main_branch(row)
    if branch is None:
        return None
    bx, by = branch
    rx, ry, l_norm = _resample(bx, by, int(n_points))
    tx = np.gradient(rx, edge_order=2)
    ty = np.gradient(ry, edge_order=2)
    tnorm = np.hypot(tx, ty)
    tnorm[tnorm <= _EPS] = 1.0
    nx = -ty / tnorm
    ny = tx / tnorm
    width_px = np.zeros(int(n_points), dtype=np.float64)
    wp = row.get("width_profile")
    if wp:
        widths = np.asarray(wp, dtype=np.float64)
        if widths.size == 1:
            width_px = np.full(int(n_points), widths[0] / pixel_microns)
        else:
            width_px = np.interp(
                l_norm, np.linspace(0.0, 1.0, widths.size), widths
            ) / pixel_microns
    else:
        ring = outline_ring(row)
        if ring is not None:
            ring_tree = cKDTree(np.column_stack(ring))
            dist, _ = ring_tree.query(np.column_stack([rx, ry]))
            width_px = 2.0 * np.maximum(dist, 0.0)
    chord = max(
        float(np.hypot(rx[-1] - rx[0], ry[-1] - ry[0])), _EPS
    )
    ring = outline_ring(row)
    return RodFrame(
        cell_key=int(cell_key),
        label=int(label),
        x=rx,
        y=ry,
        l=l_norm,
        nx=nx,
        ny=ny,
        width_px=width_px,
        outline_x=None if ring is None else ring[0],
        outline_y=None if ring is None else ring[1],
        chord_px=chord,
        pixel_microns=float(pixel_microns),
    )


def rod_frames(
    cells_table: pa.Table | Any,
    n_points: int = 32,
    pixel_microns: float = 1.0,
) -> dict[int, RodFrame]:
    """Build rod frames for the target frame of ``cells_table``.

    With a multi-frame table only the first frame (ascending) is used,
    matching the frame-selection rule of the geometry operators.
    """
    table = _to_pa(cells_table)
    if table.num_rows == 0:
        return {}
    # pyarrow's ``to_numpy`` imports pandas lazily (even with
    # ``zero_copy_only=True``); ``to_pylist`` on a non-nullable integer
    # column avoids that ~2 s one-off cost.
    frames = np.asarray(table.column("frame").to_pylist(), dtype=np.int64)
    unique = np.unique(frames)
    target = int(unique[0]) if unique.size else 0
    columns = [
        c
        for c in ("cell_key", "frame", "label", "midline", "width_profile", "outline")
        if c in table.column_names
    ]
    out: dict[int, RodFrame] = {}
    for row in table.select(columns).to_pylist():
        if unique.size > 1 and int(row["frame"]) != target:
            continue
        frame = build_rod_frame(
            int(row["cell_key"]),
            int(row["label"]),
            row,
            n_points=n_points,
            pixel_microns=pixel_microns,
        )
        if frame is not None:
            out[frame.cell_key] = frame
    return out


def sample_line_profile(
    spline_image: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    half_widths: np.ndarray,
    mode: str,
    n_lateral: int,
    *,
    closed: bool,
) -> np.ndarray:
    """Sample a channel across the normal of a resampled line.

    Shared core of ``midline_profile`` (open line, widths from cell
    geometry) and ``outline_profile`` (closed membrane ring, constant
    band). The caller supplies already-resampled line points and
    half-widths in pixels plus a spline-prefiltered image, so the cubic
    prefilter runs once per frame rather than once per cell.
    """
    if mode not in {"mean", "max"}:
        raise ValueError(f"mode must be 'mean' or 'max', got {mode!r}")
    if n_lateral < 1:
        raise ValueError("n_lateral must be >= 1")
    if not (xs.shape == ys.shape == half_widths.shape):
        raise ValueError("xs, ys and half_widths must have the same shape")
    if closed:
        tx = (np.roll(xs, -1) - np.roll(xs, 1)) * 0.5
        ty = (np.roll(ys, -1) - np.roll(ys, 1)) * 0.5
    else:
        tx = np.gradient(xs, edge_order=2)
        ty = np.gradient(ys, edge_order=2)
    tnorm = np.hypot(tx, ty)
    tnorm[tnorm <= _EPS] = 1.0
    nx = -ty / tnorm
    ny = tx / tnorm
    offsets = np.linspace(-1.0, 1.0, n_lateral)
    grid_x = xs[:, None] + offsets[None, :] * half_widths[:, None] * nx[:, None]
    grid_y = ys[:, None] + offsets[None, :] * half_widths[:, None] * ny[:, None]
    values = ndimage.map_coordinates(
        spline_image,
        [grid_y.ravel(), grid_x.ravel()],
        order=3,
        mode="nearest",
        prefilter=False,
    ).reshape(len(xs), n_lateral)
    return values.mean(axis=1) if mode == "mean" else values.max(axis=1)
