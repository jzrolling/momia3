"""Localization / distribution operators (research foundation for CIFT).

These operators turn the common subcellular-distribution representations
from the microscopy literature into plain-function operators:

- rod cell coordinates and 1-D long-axis profiles (MicrobeJ, Oufti,
  BacStalk, ColiCoords, BactMAP, demograph);
- radial and compartment intensity statistics (CellProfiler / Cell Painting);
- point-process statistics (Ripley K, DBSCAN-style puncta clusters);
- kernel mean embeddings, MMD and sliced Wasserstein distances;
- boundary Fourier coefficients for periodic membrane signals.

Each registry adapter carries provenance metadata (algorithm source,
reference, retrieval date and implementation notes).  The functions are
pure; they accept arrays / Arrow tables and return arrays or JSON-friendly
dicts.  They are deliberately small building blocks for a later CIFT layer,
not a complete distribution model.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
from scipy import ndimage
from scipy.spatial import cKDTree

from ._batch import FrameBatch
from .rod_frame import rod_frames

logger = logging.getLogger(__name__)

__all__ = [
    "rod_coordinates",
    "rod_profile",
    "radial_intensity_distribution",
    "compartment_stats",
]

_EPS = 1e-12


def _to_pa(table: pa.Table | Any) -> pa.Table:
    if isinstance(table, pa.Table):
        return table
    to_arrow = getattr(table, "to_arrow", None)
    if callable(to_arrow):
        return to_arrow()
    raise TypeError(f"expected pyarrow.Table or polars.DataFrame, got {type(table).__name__}")


def _arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    seg = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(seg)])


def rod_coordinates(
    cells_table: pa.Table | Any,
    labels_2d: np.ndarray,
    n_points: int = 32,
    pixel_microns: float = 1.0,
) -> dict[int, dict[str, np.ndarray]]:
    """Cell-coordinate transform for rod-like cells.

    For every foreground pixel of a cell, project it onto the main midline
    branch and return normalized longitudinal coordinate l in [0, 1],
    signed radial coordinate r (r=0 at the midline, |r|=1 at the outline),
    signed distance in pixels, width at that longitudinal position, and
    normalized distances to the two poles.  This is the same family of
    coordinate systems used by MicrobeJ, Oufti, BacStalk, ColiCoords,
    BactMAP and momia2 demograph, implemented without assuming a straight
    axis: the projection uses the actual midline and outline.

    Parameters
    ----------
    cells_table:
        Canonical cells table with cell_key, frame, label, midline,
        width_profile and outline columns.
    labels_2d:
        Instance labels of the same frame.
    n_points:
        Number of samples along the midline.
    pixel_microns:
        Pixel size used to convert width_profile from um to px.

    Returns
    -------
    dict[int, dict[str, numpy.ndarray]]
        cell_key -> arrays longitudinal, radial, signed_distance,
        width_px, pole1_distance, pole2_distance, all in ascending pixel
        order of numpy.where on the labels image.
    """
    if n_points < 4:
        raise ValueError("n_points must be >= 4")
    if pixel_microns <= 0:
        raise ValueError("pixel_microns must be positive")
    table = _to_pa(cells_table)
    labels = np.asarray(labels_2d)
    batch = FrameBatch(labels)
    frames = rod_frames(table, n_points=n_points, pixel_microns=pixel_microns)
    out: dict[int, dict[str, np.ndarray]] = {}
    for key, frame in frames.items():
        try:
            xs, ys = batch.pixel_xy(frame.label)
        except KeyError:
            continue
        out[int(key)] = frame.project(xs, ys)
    return out


def rod_profile(
    cells_table: pa.Table | Any,
    image_2d: np.ndarray,
    labels_2d: np.ndarray,
    n_bins: int = 64,
    mode: str = "mean",
    pixel_microns: float = 1.0,
) -> dict[int, np.ndarray]:
    """One-dimensional intensity profile along the rod long axis.

    Foreground pixels are binned by the normalized longitudinal coordinate
    from rod_coordinates, then averaged or max-pooled in each bin.  This is
    the profile family used by MicrobeJ and Oufti, but follows the actual
    (possibly curved) midline instead of a straight long axis.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    if mode not in {"mean", "max"}:
        raise ValueError("mode must be 'mean' or 'max'")
    image = np.asarray(image_2d, dtype=np.float64)
    labels = np.asarray(labels_2d)
    if image.shape != labels.shape:
        raise ValueError("image and labels must have the same shape")
    table = _to_pa(cells_table)
    batch = FrameBatch(labels)
    frames = rod_frames(
        table, n_points=max(n_bins, 8), pixel_microns=pixel_microns
    )
    width = labels.shape[1]
    out: dict[int, np.ndarray] = {}
    flat_image = image.reshape(-1)
    for key, frame in frames.items():
        try:
            pixels = batch.pixels(frame.label)
        except KeyError:
            continue
        values = flat_image[pixels]
        if values.size == 0:
            continue
        xs = pixels % width
        ys = pixels // width
        rec = frame.project(xs, ys)
        bins = np.clip((rec["longitudinal"] * n_bins).astype(np.int64), 0, n_bins - 1)
        counts = np.bincount(bins, minlength=n_bins).astype(np.float64)
        if mode == "mean":
            sums = np.bincount(bins, weights=values, minlength=n_bins)
            prof = np.divide(sums, counts, out=np.full(n_bins, np.nan), where=counts > 0)
        else:
            prof = np.full(n_bins, -np.inf)
            np.maximum.at(prof, bins, values)
            prof[~np.isfinite(prof)] = np.nan
        valid = np.isfinite(prof)
        if valid.any() and not valid.all():
            idx = np.arange(n_bins, dtype=np.float64)
            prof = np.interp(idx, idx[valid], prof[valid])
        out[int(key)] = prof.astype(np.float32)
    return out


def radial_intensity_distribution(
    labels_2d: np.ndarray,
    image_2d: np.ndarray,
    n_bins: int = 8,
) -> dict[int, np.ndarray]:
    """Per-cell radial intensity distribution (CellProfiler analogue).

    Intensity is binned by distance from the cell centroid, normalized by
    the cell's maximum centroid distance.  The result is rotation invariant
    and scale-normalized, but intentionally only a radial summary; use
    compartment_stats or the CIFT field for non-radial patterns.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    labels = np.asarray(labels_2d)
    image = np.asarray(image_2d, dtype=np.float64)
    if labels.shape != image.shape:
        raise ValueError("labels and image must have the same shape")
    out: dict[int, np.ndarray] = {}
    yy, xx = np.mgrid[: labels.shape[0], : labels.shape[1]]
    for label in np.unique(labels):
        if int(label) == 0:
            continue
        mask = labels == label
        ys = yy[mask].astype(np.float64)
        xs = xx[mask].astype(np.float64)
        cy, cx = ys.mean(), xs.mean()
        radius = np.hypot(ys - cy, xs - cx)
        scale = float(radius.max())
        if scale <= _EPS:
            continue
        bins = np.clip((radius / scale * n_bins).astype(np.int64), 0, n_bins - 1)
        values = image[mask]
        sums = np.bincount(bins, weights=values, minlength=n_bins)
        counts = np.bincount(bins, minlength=n_bins).astype(np.float64)
        out[int(label)] = np.divide(sums, counts, out=np.full(n_bins, np.nan), where=counts > 0).astype(np.float32)
    return out


def compartment_stats(
    labels_2d: np.ndarray,
    image_2d: np.ndarray,
    compartment_mask: np.ndarray,
    name: str = "compartment",
) -> dict[str, np.ndarray]:
    """Per-cell intensity statistics inside a compartment mask.

    This is the Cell Painting / CellProfiler compartment analogue: the
    caller supplies a membrane, nucleoid, pole or other binary compartment
    mask, and the operator returns mean, integrated intensity, covered area
    and compartment fraction for every cell label in ascending order.
    """
    labels = np.asarray(labels_2d)
    image = np.asarray(image_2d, dtype=np.float64)
    comp = np.asarray(compartment_mask).astype(bool)
    if labels.shape != image.shape or labels.shape != comp.shape:
        raise ValueError("labels, image and compartment mask must share shape")
    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids > 0]
    mean = np.full(label_ids.size, np.nan, dtype=np.float32)
    total = np.full(label_ids.size, np.nan, dtype=np.float32)
    area = np.zeros(label_ids.size, dtype=np.float32)
    frac = np.zeros(label_ids.size, dtype=np.float32)
    for i, label in enumerate(label_ids):
        cell = labels == label
        inside = cell & comp
        area[i] = float(inside.sum())
        if inside.any():
            vals = image[inside]
            mean[i] = float(vals.mean())
            total[i] = float(vals.sum())
        cell_area = float(cell.sum())
        frac[i] = area[i] / cell_area if cell_area > 0 else 0.0
    return {
        f"{name}_mean": mean,
        f"{name}_integrated": total,
        f"{name}_area": area,
        f"{name}_fraction": frac,
    }
