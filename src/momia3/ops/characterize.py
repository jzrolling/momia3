"""Single-cell characterization operators (ARCHITECTURE.md §4.6).

- :func:`demographic_projection` — resample one channel into normalized
  cell coordinates (midline arc ``l ∈ [0, 1]`` × lateral ``d ∈ [-1, 1]``),
  a Coli-Coords/BacStalk-style demographic map that upgrades
  ``momia2/utils/fish.py::create_canvas``.
- :func:`growth_stats` — per-track elongation rates (log-linear fit of
  ``length_um`` vs frame), interdivision frames, and an adder/sizer
  correlation summary.
- :func:`detect_septum` — fractional midline position of the deepest width
  dip in a width profile (``None`` when no constriction is found).
- :func:`orientation_coherence` — local nematic orientational order
  parameter ``S`` of cell major axes.
- :func:`find_puncta` — LoG-response spot detection with 3×3 quadratic
  subpixel refinement (pure-numpy port of ``momia2/utils/fish.py``, no
  string serialization anywhere).
- :func:`assign_pole_age` — MicrobeJ-style pole-age inheritance across
  ``continue``/``division`` lineage edges (``poles.age`` of §4.6).
- :func:`track_demographic_mean` — per-track mean of
  :func:`demographic_projection` maps.
- :func:`efd_shape_space` — PCA shape space over elliptic Fourier
  descriptors of the per-cell outlines.
- :func:`contact_metrics` — shared-boundary contact fraction and
  shared-boundary-weighted neighbour alignment (contact-network part of
  ``population.cording``, §4.6).

All functions are pure; tables are accepted as ``pyarrow.Table`` or
``polars.DataFrame`` and scalars read directly from the canonical columns.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
from pydantic import BaseModel, Field
from scipy import ndimage, signal
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

from momia3.ops.outline import elliptic_fourier

logger = logging.getLogger(__name__)

__all__ = [
    "PunctaParams",
    "demographic_projection",
    "growth_stats",
    "detect_septum",
    "orientation_coherence",
    "find_puncta",
    "assign_pole_age",
    "track_demographic_mean",
    "efd_shape_space",
    "contact_metrics",
]

_EPS = 1e-12


class PunctaParams(BaseModel):
    """Parameters of :func:`find_puncta`."""

    smooth_sigma: float = Field(
        0.5, ge=0.0, description="Pre-detection Gaussian smoothing sigma (px)"
    )
    log_sigma: float = Field(1.0, gt=0, description="Laplacian-of-Gaussian sigma (px)")
    min_distance: int = Field(1, ge=1, description="Minimum spot separation (px)")
    threshold_rel: float = Field(
        0.1, ge=0.0, le=1.0, description="Relative LoG-response threshold"
    )
    threshold_abs: float | None = Field(
        None, ge=0.0, description="Absolute LoG-response threshold (overrides rel)"
    )
    max_iterations: int = Field(
        2, ge=1, le=8, description="Max subpixel relaxation iterations"
    )


# ---------------------------------------------------------------------------
# table helpers
# ---------------------------------------------------------------------------


def _to_pa(table: pa.Table | pl.DataFrame) -> pa.Table:
    """Coerce a polars DataFrame or Arrow table to ``pa.Table``."""
    if isinstance(table, pa.Table):
        return table
    if isinstance(table, pl.DataFrame):
        return table.to_arrow()
    raise TypeError(
        f"expected pyarrow.Table or polars.DataFrame, got {type(table).__name__}"
    )


def _column(table: pa.Table, name: str) -> np.ndarray:
    """Read a flat numeric column as a float numpy array (NaN for null)."""
    return np.asarray(
        table.column(name).to_numpy(zero_copy_only=False), dtype=np.float64
    )


def _int_column(table: pa.Table, name: str) -> np.ndarray:
    """Read a flat integer column as int64 (``-1`` for null)."""
    arr = table.column(name)
    if arr.null_count:
        out = np.full(len(arr), -1, dtype=np.int64)
        valid = np.asarray(arr.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
        out[valid] = np.asarray(arr.drop_null().to_numpy(zero_copy_only=False))
        return out
    return np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.int64)


def _main_branch_xy(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract the main (longest) midline branch as ``(x, y)`` arrays."""
    midline = row.get("midline")
    if not midline:
        return None
    best: dict[str, Any] | None = None
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
            best = branch
    if best is None:
        return None
    return (
        np.asarray(best["x"], dtype=np.float64),
        np.asarray(best["y"], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# demographic projection
# ---------------------------------------------------------------------------


def _polyline_arclength(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative arc length along an open polyline, starting at 0."""
    seg = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _outline_to_closed_xy(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """Read the outline column as a closed ``(x, y)`` ring."""
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


def demographic_projection(
    cells_table: pa.Table | pl.DataFrame,
    image_2d: np.ndarray,
    labels_2d: np.ndarray,
    n_l: int = 64,
    n_d: int = 16,
) -> dict[int, np.ndarray]:
    """Resample channel intensities into normalized per-cell coordinates.

    For every cell in ``cells_table`` (restricted to rows whose ``frame``
    matches the frame of ``labels_2d`` when several frames are present),
    sample ``image_2d`` on a regular grid of ``n_l × n_d`` points: ``l``
    runs along the main midline branch (arc-length parameterized,
    ``l ∈ [0, 1]``) and ``d`` runs laterally between the two outline
    boundaries (``d ∈ [-1, 1]``, nearest outline point at ``|d| = 1``).
    Sampling uses :func:`scipy.ndimage.map_coordinates` with cubic spline
    interpolation.

    Parameters
    ----------
    cells_table : pyarrow.Table or polars.DataFrame
        Cells table carrying ``cell_key``, ``frame``, ``label`` and the
        ``midline``/``outline`` geometry columns.
    image_2d : numpy.ndarray
        Single channel frame ``(y, x)`` to resample.
    labels_2d : numpy.ndarray
        Instance labels of the same frame; only cells whose label is
        present are projected.
    n_l : int
        Number of samples along the midline.
    n_d : int
        Number of samples across the cell width.

    Returns
    -------
    dict of int to numpy.ndarray
        ``cell_key -> (n_l, n_d) float32`` intensity map. Rows are
        midline positions ``l = 0 .. 1``, columns lateral positions
        ``d = -1 .. 1``. Cells lacking midline/outline geometry or absent
        from ``labels_2d`` are skipped.
    """
    table = _to_pa(cells_table)
    image = np.asarray(image_2d, dtype=np.float64)
    labels = np.asarray(labels_2d)
    present = set(np.unique(labels).tolist()) - {0}
    frames = _int_column(table, "frame")
    unique_frames = np.unique(frames)
    target_frame = int(unique_frames[0]) if unique_frames.size else 0

    l_targets = np.linspace(0.0, 1.0, n_l)
    d_targets = np.linspace(-1.0, 1.0, n_d)
    out: dict[int, np.ndarray] = {}
    # Spline coefficients are image-global: prefilter once instead of letting
    # every map_coordinates call re-filter the full frame (the dominant cost).
    # Reproduce scipy's map_coordinates(mode='nearest') prepadding exactly:
    # it pads the input by 12 px with edge values, filters with mode='mirror',
    # then samples at coordinates shifted by that padding.
    npad = 12
    coeffs = ndimage.spline_filter(np.pad(image, npad, mode="edge"), order=3)
    coords_parts: list[np.ndarray] = []
    keys: list[int] = []
    sizes: list[int] = []

    for row in table.select(["cell_key", "frame", "label", "midline", "outline"]).to_pylist():
        if unique_frames.size > 1 and int(row["frame"]) != target_frame:
            continue
        if int(row["label"]) not in present:
            continue
        branch = _main_branch_xy(row)
        ring = _outline_to_closed_xy(row)
        if branch is None or ring is None:
            continue
        mx, my = branch
        total = _polyline_arclength(mx, my)[-1]
        if total <= _EPS:
            continue
        s = _polyline_arclength(mx, my) / total
        # Resample the midline uniformly in arc length.
        xs = np.interp(l_targets, s, mx)
        ys = np.interp(l_targets, s, my)
        # Tangents (central differences on the uniform samples).
        tx = np.gradient(xs, l_targets, edge_order=2)
        ty = np.gradient(ys, l_targets, edge_order=2)
        tnorm = np.hypot(tx, ty)
        tnorm[tnorm <= _EPS] = 1.0
        nx_, ny_ = -ty / tnorm, tx / tnorm  # left normal (x, y)

        ox, oy = ring
        tree = cKDTree(np.column_stack([ox, oy]))
        # Distance from each midline sample to the outline (half width).
        dist, _ = tree.query(np.column_stack([xs, ys]))
        half_width = np.where(dist <= _EPS, 1.0, dist)

        # Sample grid: d = -1 -> -normal side boundary, d = +1 -> +normal.
        grid_x = xs[:, None] + (d_targets[None, :] * half_width[:, None]) * nx_[:, None]
        grid_y = ys[:, None] + (d_targets[None, :] * half_width[:, None]) * ny_[:, None]
        coords_parts.append(np.stack([grid_y.ravel(), grid_x.ravel()]))
        keys.append(int(row["cell_key"]))
        sizes.append(int(grid_x.size))

    if coords_parts:
        # One batched cubic-spline sample for all cells (prefilter=False reuses
        # the global coefficients computed above).
        values = ndimage.map_coordinates(
            coeffs,
            np.concatenate(coords_parts, axis=1) + npad,
            order=3,
            mode="nearest",
            prefilter=False,
        )
        start = 0
        for key, size in zip(keys, sizes):
            out[key] = values[start : start + size].reshape(n_l, n_d).astype(np.float32)
            start += size

    logger.debug("demographic_projection: %d cells projected", len(out))
    return out


# ---------------------------------------------------------------------------
# growth statistics
# ---------------------------------------------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, NaN when undefined (too few points / no spread)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2 or x.std() <= _EPS or y.std() <= _EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def growth_stats(
    tracks_table: pa.Table | pl.DataFrame,
    cells_table: pa.Table | pl.DataFrame,
) -> dict[str, Any]:
    """Per-track growth dynamics plus a population adder/sizer summary.

    For each track (rows of ``cells_table`` grouped by ``track_id``), fit
    ``log(length_um)`` linearly against ``frame``; the slope is the
    elongation rate (per frame). Tracks with a ``division`` fate also
    contribute their birth length, division length, added length and
    interdivision time to the adder/sizer correlation summary.

    Parameters
    ----------
    tracks_table : pyarrow.Table or polars.DataFrame
        Canonical tracks table (``track_id``, ``birth_frame``,
        ``last_frame``, ``division_frame``, ``fate``).
    cells_table : pyarrow.Table or polars.DataFrame
        Cells table with ``track_id``, ``frame`` and ``length_um``
        (rows with null ``track_id`` or non-finite ``length_um`` are
        ignored).

    Returns
    -------
    dict
        ``per_track``
            List of dicts ``{track_id, birth_frame, last_frame, n_frames,
            elongation_rate, elongation_intercept, interdivision_frames,
            birth_length_um, division_length_um, added_length_um}``
            (growth/d division fields NaN when undefined).
        ``summary``
            Dict with ``n_tracks``, ``mean_elongation_rate``,
            ``mean_interdivision_frames``, and Pearson correlations
            ``corr_birth_added`` (birth vs added length — adder test) and
            ``corr_birth_division`` (birth vs division length — sizer
            test).
    """
    tracks = _to_pa(tracks_table)
    cells = _to_pa(cells_table)
    c_track = _int_column(cells, "track_id")
    c_frame = _int_column(cells, "frame")
    c_length = _column(cells, "length_um")

    per_track: dict[int, dict[str, Any]] = {}
    for track_id in np.unique(c_track[c_track >= 0]):
        tid = int(track_id)
        sel = c_track == tid
        frames = c_frame[sel].astype(np.float64)
        lengths = c_length[sel]
        ok = np.isfinite(lengths) & (lengths > 0)
        rate = float("nan")
        intercept = float("nan")
        if ok.sum() >= 2:
            slope, intercept_v = np.polyfit(frames[ok], np.log(lengths[ok]), 1)
            rate, intercept = float(slope), float(intercept_v)
        per_track[tid] = {
            "track_id": tid,
            "birth_frame": int(frames.min()),
            "last_frame": int(frames.max()),
            "n_frames": int(frames.size),
            "elongation_rate": rate,
            "elongation_intercept": intercept,
            "interdivision_frames": float("nan"),
            "birth_length_um": float("nan"),
            "division_length_um": float("nan"),
            "added_length_um": float("nan"),
        }

    fate_col = tracks.column("fate")
    if pa.types.is_dictionary(fate_col.type):
        dict_arr = fate_col.combine_chunks()
        fate_values = np.asarray(dict_arr.dictionary.to_numpy(zero_copy_only=False))
        fate_index = np.asarray(dict_arr.indices.to_numpy(zero_copy_only=False))
        fates = np.where(fate_index >= 0, fate_values[np.clip(fate_index, 0, None)], None)
    else:
        fates = np.asarray(fate_col.to_numpy(zero_copy_only=False))
    t_ids = _int_column(tracks, "track_id")
    t_birth = _int_column(tracks, "birth_frame")
    t_last = _int_column(tracks, "last_frame")
    t_div = _int_column(tracks, "division_frame")

    for i, tid in enumerate(t_ids):
        rec = per_track.get(int(tid))
        if rec is None:
            continue
        fate = fates[i]
        rec["birth_frame"] = int(t_birth[i])
        rec["last_frame"] = int(t_last[i])
        if fate == "divide" and t_div[i] >= 0:
            rec["interdivision_frames"] = float(t_div[i] - t_birth[i])
            sel = (c_track == int(tid)) & (c_frame == t_birth[i])
            birth_len = c_length[sel]
            sel = (c_track == int(tid)) & (c_frame == t_div[i])
            div_len = c_length[sel]
            if birth_len.size and np.isfinite(birth_len[0]):
                rec["birth_length_um"] = float(birth_len[0])
            if div_len.size and np.isfinite(div_len[0]):
                rec["division_length_um"] = float(div_len[0])
            if np.isfinite(rec["birth_length_um"]) and np.isfinite(
                rec["division_length_um"]
            ):
                rec["added_length_um"] = float(
                    rec["division_length_um"] - rec["birth_length_um"]
                )

    track_list = [per_track[tid] for tid in sorted(per_track)]
    rates = np.array([r["elongation_rate"] for r in track_list])
    interdiv = np.array([r["interdivision_frames"] for r in track_list])
    births = np.array([r["birth_length_um"] for r in track_list])
    added = np.array([r["added_length_um"] for r in track_list])
    division = np.array([r["division_length_um"] for r in track_list])
    summary = {
        "n_tracks": len(track_list),
        "n_dividing": int(np.isfinite(interdiv).sum()),
        "mean_elongation_rate": (
            float(np.nanmean(rates)) if np.isfinite(rates).any() else float("nan")
        ),
        "mean_interdivision_frames": (
            float(np.nanmean(interdiv))
            if np.isfinite(interdiv).any()
            else float("nan")
        ),
        "corr_birth_added": _pearson(births, added),
        "corr_birth_division": _pearson(births, division),
    }
    return {"per_track": track_list, "summary": summary}


# ---------------------------------------------------------------------------
# septum detection
# ---------------------------------------------------------------------------


def detect_septum(
    width_profile: np.ndarray | list[float],
    min_prominence: float,
) -> float | None:
    """Locate a septum as the deepest width dip along the midline.

    Parameters
    ----------
    width_profile : array-like of float
        Width along the (arc-length ordered) midline; unit is irrelevant
        as long as it is consistent. Non-finite samples are interpolated
        from neighbours before the search.
    min_prominence : float
        Minimum prominence of the dip, in the same unit as
        ``width_profile`` (passed to :func:`scipy.signal.find_peaks` on the
        inverted profile).

    Returns
    -------
    float or None
        Fractional midline position in ``[0, 1]`` of the deepest dip, or
        ``None`` when no dip clears ``min_prominence``.
    """
    w = np.asarray(width_profile, dtype=np.float64).ravel()
    if w.size < 3:
        return None
    if not np.isfinite(w).all():
        good = np.isfinite(w)
        if good.sum() < 3:
            return None
        w = np.interp(np.arange(w.size), np.flatnonzero(good), w[good])
    peaks, props = signal.find_peaks(-w, prominence=float(min_prominence))
    if peaks.size == 0:
        return None
    deepest = int(peaks[int(np.argmax(props["prominences"]))])
    return float(deepest / (w.size - 1))


# ---------------------------------------------------------------------------
# orientational order
# ---------------------------------------------------------------------------


def _cell_angles(
    labels_2d: np.ndarray, table: pa.Table
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell ``(label, angle_rad, centroid[yx])`` of the major axis.

    Angles are axis orientations in ``[0, pi)`` measured from the +x axis
    in the image plane, derived from the binary-mask inertia tensor.
    """
    labels = np.asarray(labels_2d)
    out_label: list[int] = []
    out_angle: list[float] = []
    out_yx: list[tuple[float, float]] = []
    for row in table.select(["label", "centroid"]).to_pylist():
        lab = int(row["label"])
        mask = labels == lab
        if not mask.any():
            continue
        yy, xx = np.nonzero(mask)
        if yy.size < 3:
            continue
        cy, cx = yy.mean(), xx.mean()
        y, x = yy - cy, xx - cx
        mu20 = float((x * x).mean())
        mu02 = float((y * y).mean())
        mu11 = float((x * y).mean())
        theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)  # from +x axis
        out_label.append(lab)
        out_angle.append(float(theta % np.pi))
        out_yx.append((float(cy), float(cx)))
    return (
        np.asarray(out_label, dtype=np.int64),
        np.asarray(out_angle, dtype=np.float64),
        np.asarray(out_yx, dtype=np.float64).reshape(-1, 2),
    )


def orientation_coherence(
    labels_2d: np.ndarray,
    cells_table: pa.Table | pl.DataFrame,
) -> float:
    """Local nematic order parameter of cell major axes.

    Each cell's orientation is compared with the mean orientation of its
    ``k = min(6, n - 1)`` nearest neighbours (centroid distances, aligned
    as headless axes). The returned coherence is the mean local order
    parameter

    ``S = mean_i | <exp(2 i (theta_i - theta_j))>_j |``,

    in ``[0, 1]``; 1 = perfectly aligned neighbourhood, 0 = isotropic.

    Parameters
    ----------
    labels_2d : numpy.ndarray
        Instance label image.
    cells_table : pyarrow.Table or polars.DataFrame
        Cells of that frame (``label`` + ``centroid`` columns).

    Returns
    -------
    float
        Mean local order parameter; 0.0 when fewer than two cells are
        measurable.
    """
    table = _to_pa(cells_table)
    _, angles, yx = _cell_angles(labels_2d, table)
    n = angles.size
    if n < 2:
        return 0.0
    k = min(6, n - 1)
    tree = cKDTree(yx)
    _, neigh = tree.query(yx, k=k + 1)  # first hit is the cell itself
    local = np.empty(n, dtype=np.float64)
    for i in range(n):
        js = np.atleast_1d(neigh[i])[1:]
        diff = angles[i] - angles[js]
        local[i] = np.abs(np.exp(2j * diff).mean())
    return float(local.mean())


# ---------------------------------------------------------------------------
# puncta detection (momia2 fish.py port)
# ---------------------------------------------------------------------------


def _quadratic_peak_offset(q1: float, q2: float, q3: float) -> float | None:
    """Vertex offset of the parabola through ``(-1, q1) (0, q2) (1, q3)``.

    Returns the sub-sample offset in ``[-1, 1]`` of the quadratic maximum,
    or ``None`` when the centre is not a strict local maximum (the legacy
    code raised ``ValueError`` here; returning ``None`` keeps the detector
    total). Ported from ``momia2/utils/fish.py::quadratic_maxima_approximation``.
    """
    if q2 <= max(q1, q3):
        return None
    denom = q1 - 2.0 * q2 + q3
    if abs(denom) <= _EPS:
        return None
    return float(0.5 * (q1 - q3) / denom)


def _subpixel_quadratic(
    row: int,
    col: int,
    energy: np.ndarray,
    max_iterations: int,
) -> tuple[float, float, bool]:
    """Refine a pixel peak to subpixel precision on a 3×3 neighbourhood.

    Iteratively re-centres on the 3×3 argmax (up to ``max_iterations``);
    once the centre pixel is the local maximum, fits separable 1D
    quadratics along rows and columns. Ported from
    ``momia2/utils/fish.py::subpixel_approximation_quadratic`` (pure
    numpy, row/col order).
    """
    r, c = float(row), float(col)
    ri, ci = row, col
    stable = False
    for _ in range(max_iterations):
        r0, r1 = max(ri - 1, 0), min(ri + 2, energy.shape[0])
        c0, c1 = max(ci - 1, 0), min(ci + 2, energy.shape[1])
        cube = energy[r0:r1, c0:c1]
        centre_r, centre_c = ri - r0, ci - c0
        if cube[centre_r, centre_c] == cube.max():
            stable = True
            dr = _quadratic_peak_offset(
                float(cube[centre_r - 1, centre_c]) if centre_r > 0 else -np.inf,
                float(cube[centre_r, centre_c]),
                float(cube[centre_r + 1, centre_c])
                if centre_r < cube.shape[0] - 1
                else -np.inf,
            )
            dc = _quadratic_peak_offset(
                float(cube[centre_r, centre_c - 1]) if centre_c > 0 else -np.inf,
                float(cube[centre_r, centre_c]),
                float(cube[centre_r, centre_c + 1])
                if centre_c < cube.shape[1] - 1
                else -np.inf,
            )
            if dr is not None:
                r = ri + dr
            if dc is not None:
                c = ci + dc
            break
        flat = int(np.argmax(cube))
        ri = r0 + flat // cube.shape[1]
        ci = c0 + flat % cube.shape[1]
    return r, c, stable


def find_puncta(
    image_2d: np.ndarray,
    mask_2d: np.ndarray | None = None,
    params: PunctaParams | None = None,
) -> np.ndarray:
    """Detect subpixel point sources (FISH puncta) in one channel.

    Pipeline (port of ``momia2/utils/fish.py::find_puncta`` with the
    string serialization removed): Gaussian pre-smoothing, min-max
    normalization, Laplacian-of-Gaussian response, local-maximum picking
    on the inverted response (bright spots on dark background are LoG
    minima), then 3×3 quadratic subpixel refinement.

    Parameters
    ----------
    image_2d : numpy.ndarray
        2D intensity image (bright spots).
    mask_2d : numpy.ndarray, optional
        Boolean/0-1 region in which to accept peaks (e.g. one cell or the
        foreground mask). ``None`` searches the whole frame.
    params : PunctaParams, optional
        Detection parameters; defaults match the legacy behaviour
        (``smooth_sigma=0.5``, ``log_sigma=1``, ``min_distance=1``,
        ``threshold_rel=0.1``).

    Returns
    -------
    numpy.ndarray
        ``(K, 2)`` float64 array of ``(y, x)`` subpixel spot coordinates,
        sorted by decreasing LoG response. ``(0, 2)`` when nothing is
        found.
    """
    params = params or PunctaParams()
    image = np.asarray(image_2d, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"image_2d must be 2D, got shape {image.shape}")

    smoothed = ndimage.gaussian_filter(image, sigma=params.smooth_sigma)
    span = float(smoothed.max() - smoothed.min())
    if span <= _EPS:
        return np.empty((0, 2), dtype=np.float64)
    smoothed = (smoothed - smoothed.min()) / span
    log_resp = ndimage.gaussian_laplace(smoothed, sigma=params.log_sigma)
    response = -log_resp  # bright spots -> maxima
    if mask_2d is not None:
        mask = np.asarray(mask_2d) > 0
        response = np.where(mask, response, -np.inf)

    kwargs: dict[str, Any] = {
        "min_distance": params.min_distance,
        "exclude_border": False,
    }
    if params.threshold_abs is not None:
        kwargs["threshold_abs"] = params.threshold_abs
    else:
        kwargs["threshold_rel"] = params.threshold_rel
    peaks = peak_local_max(response, **kwargs)
    if peaks.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    coords: list[tuple[float, float, float]] = []
    for row, col in peaks:
        r, c, stable = _subpixel_quadratic(
            int(row), int(col), smoothed, params.max_iterations
        )
        if stable:
            coords.append((r, c, float(response[row, col])))
    if not coords:
        return np.empty((0, 2), dtype=np.float64)
    coords.sort(key=lambda item: -item[2])
    out = np.array([(r, c) for r, c, _ in coords], dtype=np.float64)
    logger.debug("find_puncta: %d spots (from %d pixel peaks)", len(out), len(peaks))
    return out


# ---------------------------------------------------------------------------
# pole age inheritance (§4.6 poles.age)
# ---------------------------------------------------------------------------


def _optional_float_column(table: pa.Table, name: str) -> np.ndarray:
    """Read a float column (NaN for null); all-NaN when the column is absent."""
    if name not in table.column_names:
        return np.full(table.num_rows, np.nan, dtype=np.float64)
    return _column(table, name)


def _centroid_xy(table: pa.Table) -> tuple[np.ndarray, np.ndarray]:
    """Read the ``centroid`` struct column as ``(x, y)`` float arrays.

    Null structs (or a missing column) yield NaN coordinates.
    """
    if "centroid" not in table.column_names:
        nan = np.full(table.num_rows, np.nan, dtype=np.float64)
        return nan, nan.copy()
    col = table.column("centroid").combine_chunks()
    valid = np.asarray(col.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    x = np.asarray(col.field("x").to_numpy(zero_copy_only=False), dtype=np.float64)
    y = np.asarray(col.field("y").to_numpy(zero_copy_only=False), dtype=np.float64)
    return np.where(valid, x, np.nan), np.where(valid, y, np.nan)


def _string_column(table: pa.Table, name: str) -> np.ndarray:
    """Decode a string/dictionary column to an object array (``""`` for null)."""
    values = table.column(name).to_pylist()
    return np.array([v if v is not None else "" for v in values], dtype=object)


def assign_pole_age(
    cells_table: pa.Table | pl.DataFrame,
    edges_table: pa.Table | pl.DataFrame,
) -> pl.DataFrame:
    """Propagate pole age (frames since the pole was formed) along a lineage.

    MicrobeJ-style new/old-pole tracking over the canonical cells and
    edges tables. Frame-0 cells seed both poles at age 0; ages then
    propagate forward along ``continue`` and ``division`` edges:

    - ``continue``: the daughter pole nearer (Euclidean) to the mother's
      ``pole1`` inherits the ``pole1`` identity and gets
      ``mother_pole1_age + 1``; the other daughter pole inherits
      ``pole2``. Pole indices may therefore swap between mother and
      daughter (identity follows proximity, not index).
    - ``division`` (mother with exactly two daughters in the same frame):
      for each daughter, the pole *farther* from the other daughter's
      centroid is the inherited (old) pole and receives the age of the
      mother's same-side pole (the mother pole nearer to the daughter's
      inherited pole) plus one; the septum-side pole is new (age 0).

    Cells whose mother is missing from the cells table, lacks finite pole
    coordinates, or has unknown ages keep ``-1`` (unknown); non-lineage
    events (``appear``/``merge``/``gap``/``disappear``) do not propagate.

    Parameters
    ----------
    cells_table : pyarrow.Table or polars.DataFrame
        Cells table with ``cell_key``, ``frame``, the pole columns
        ``pole1_x/pole1_y/pole2_x/pole2_y`` and (for the division rule)
        ``centroid``.
    edges_table : pyarrow.Table or polars.DataFrame
        Canonical edges table (``parent_key``, ``child_key``, ``event``).

    Returns
    -------
    polars.DataFrame
        Input cells table with two appended int32 columns
        ``pole1_age``/``pole2_age`` (``-1`` = unknown). Row order is
        preserved.
    """
    cells = _to_pa(cells_table)
    edges = _to_pa(edges_table)
    keys = _int_column(cells, "cell_key")
    frames = _int_column(cells, "frame")
    p1x = _optional_float_column(cells, "pole1_x")
    p1y = _optional_float_column(cells, "pole1_y")
    p2x = _optional_float_column(cells, "pole2_x")
    p2y = _optional_float_column(cells, "pole2_y")
    cx, cy = _centroid_xy(cells)

    age1 = np.where(frames == 0, 0, -1).astype(np.int64)
    age2 = age1.copy()

    key_to_idx = {int(k): i for i, k in enumerate(keys)}
    parent = _int_column(edges, "parent_key")
    child = _int_column(edges, "child_key")
    events = _string_column(edges, "event")
    m_idx = np.array([key_to_idx.get(int(k), -1) for k in parent], dtype=np.int64)
    c_idx = np.array([key_to_idx.get(int(k), -1) for k in child], dtype=np.int64)
    valid = (m_idx >= 0) & (c_idx >= 0)

    child_frame = np.full(parent.size, -1, dtype=np.int64)
    child_frame[valid] = frames[c_idx[valid]]
    is_cont = valid & (events == "continue")
    is_div = valid & (events == "division")

    # Mothers always live in earlier frames, so sweeping child frames in
    # ascending order guarantees mother ages are final when read.
    for f in np.unique(child_frame[valid]):
        batch = np.flatnonzero(is_cont & (child_frame == f))
        if batch.size:
            m, c = m_idx[batch], c_idx[batch]
            ok = (
                (age1[m] >= 0)
                & (age2[m] >= 0)
                & np.isfinite(p1x[m]) & np.isfinite(p1y[m])
                & np.isfinite(p2x[m]) & np.isfinite(p2y[m])
                & np.isfinite(p1x[c]) & np.isfinite(p1y[c])
                & np.isfinite(p2x[c]) & np.isfinite(p2y[c])
            )
            d11 = np.hypot(p1x[c] - p1x[m], p1y[c] - p1y[m])
            d21 = np.hypot(p2x[c] - p1x[m], p2y[c] - p1y[m])
            swap = d21 < d11  # daughter pole2 is the mother's-pole1 side
            new1 = np.where(swap, age2[m] + 1, age1[m] + 1)
            new2 = np.where(swap, age1[m] + 1, age2[m] + 1)
            age1[c] = np.where(ok, new1, -1)
            age2[c] = np.where(ok, new2, -1)

        batch = np.flatnonzero(is_div & (child_frame == f))
        if batch.size:
            by_mother: dict[int, list[int]] = {}
            for i in batch:
                by_mother.setdefault(int(m_idx[i]), []).append(int(c_idx[i]))
            for m, daughters in by_mother.items():
                if len(daughters) != 2:
                    continue  # malformed division; ages stay unknown
                if age1[m] < 0 or age2[m] < 0:
                    continue
                if not np.isfinite([p1x[m], p1y[m], p2x[m], p2y[m]]).all():
                    continue
                d_a, d_b = daughters
                for d, other in ((d_a, d_b), (d_b, d_a)):
                    if not np.isfinite(
                        [p1x[d], p1y[d], p2x[d], p2y[d], cx[other], cy[other]]
                    ).all():
                        continue
                    d1 = float(np.hypot(p1x[d] - cx[other], p1y[d] - cy[other]))
                    d2 = float(np.hypot(p2x[d] - cx[other], p2y[d] - cy[other]))
                    if d1 >= d2:  # pole1 is the far (inherited) pole
                        inh_x, inh_y, inh_is_p1 = p1x[d], p1y[d], True
                    else:
                        inh_x, inh_y, inh_is_p1 = p2x[d], p2y[d], False
                    dm1 = float(np.hypot(inh_x - p1x[m], inh_y - p1y[m]))
                    dm2 = float(np.hypot(inh_x - p2x[m], inh_y - p2y[m]))
                    inherited_age = int(age1[m] if dm1 <= dm2 else age2[m]) + 1
                    if inh_is_p1:
                        age1[d], age2[d] = inherited_age, 0
                    else:
                        age1[d], age2[d] = 0, inherited_age

    out = (
        cells_table.clone()
        if isinstance(cells_table, pl.DataFrame)
        else pl.from_arrow(cells)
    )
    return out.with_columns(
        [
            pl.Series("pole1_age", age1.astype(np.int32)),
            pl.Series("pole2_age", age2.astype(np.int32)),
        ]
    )


# ---------------------------------------------------------------------------
# per-track demographic means
# ---------------------------------------------------------------------------


def track_demographic_mean(
    projections: dict[int, np.ndarray],
    cells_table: pa.Table | pl.DataFrame,
) -> pl.DataFrame:
    """Mean demographic map of every track.

    Aggregates :func:`demographic_projection` outputs over the cells table
    grouped by ``track_id``: every cell whose ``cell_key`` has a
    projection and whose ``track_id`` is non-null contributes one map to
    its track's mean. Cells without a track or a projection are skipped,
    as are maps whose shape differs from the first map seen for the
    track.

    Parameters
    ----------
    projections : dict of int to numpy.ndarray
        ``cell_key -> (n_l, n_d)`` intensity maps, as returned by
        :func:`demographic_projection`.
    cells_table : pyarrow.Table or polars.DataFrame
        Cells table with ``cell_key`` and ``track_id``.

    Returns
    -------
    polars.DataFrame
        One row per track, sorted by ``track_id``, with columns
        ``track_id`` (uint32), ``n_cells`` (uint32) and ``mean_map``
        (list of float32, the row-major flattened ``n_l * n_d`` mean
        map).
    """
    cells = _to_pa(cells_table)
    keys = _int_column(cells, "cell_key")
    tids = _int_column(cells, "track_id")

    stacks: dict[int, list[np.ndarray]] = {}
    shapes: dict[int, tuple[int, ...]] = {}
    for key, tid in zip(keys, tids):
        tid = int(tid)
        if tid < 0:
            continue
        proj = projections.get(int(key))
        if proj is None:
            continue
        arr = np.asarray(proj, dtype=np.float32)
        if tid in shapes and arr.shape != shapes[tid]:
            continue
        shapes.setdefault(tid, arr.shape)
        stacks.setdefault(tid, []).append(arr)

    records: list[dict[str, Any]] = []
    for tid in sorted(stacks):
        mean = np.stack(stacks[tid]).mean(axis=0, dtype=np.float64).astype(np.float32)
        records.append(
            {
                "track_id": tid,
                "n_cells": len(stacks[tid]),
                "mean_map": mean.ravel().tolist(),
            }
        )
    return pl.DataFrame(
        records,
        schema={
            "track_id": pl.UInt32,
            "n_cells": pl.UInt32,
            "mean_map": pl.List(pl.Float32),
        },
    )


# ---------------------------------------------------------------------------
# EFD shape space
# ---------------------------------------------------------------------------


def efd_shape_space(
    cells_table: pa.Table | pl.DataFrame,
    n_harmonics: int = 10,
    n_components: int = 3,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """PCA shape space over elliptic Fourier descriptors of the outlines.

    Every cell's ``outline`` struct is described by
    :func:`momia3.ops.outline.elliptic_fourier` (``n_harmonics``
    Kuhl–Giardina-normalized harmonics, flattened to ``4 * n_harmonics``
    features). The feature matrix of all cells with outlines is
    standardized (per-feature mean/std; degenerate features get std 1)
    and decomposed by PCA via :func:`numpy.linalg.svd`.

    Parameters
    ----------
    cells_table : pyarrow.Table or polars.DataFrame
        Cells table with the ``outline`` geometry column.
    n_harmonics : int
        Number of EFD harmonics per outline.
    n_components : int
        Number of principal components to keep (columns
        ``efd_pc1 .. efd_pc{n_components}``).

    Returns
    -------
    table : polars.DataFrame
        Input cells table with appended float32 columns
        ``efd_pc1 .. efd_pc{n_components}``; cells without an outline
        (or beyond the available rank) carry nulls. Row order is
        preserved.
    info : dict
        ``explained_variance_ratio`` (list of floats, one per returned
        component), ``n_harmonics``, ``n_components`` and
        ``n_cells_used``.
    """
    cells = _to_pa(cells_table)
    n = cells.num_rows
    feats: list[np.ndarray] = []
    rows_idx: list[int] = []
    if "outline" in cells.column_names and n:
        for i, row in enumerate(cells.select(["outline"]).to_pylist()):
            ring = _outline_to_closed_xy(row)
            if ring is None:
                continue
            ox, oy = ring
            coeff = elliptic_fourier(
                np.column_stack([ox, oy]), n_harmonics=int(n_harmonics)
            )
            feats.append(coeff.ravel())
            rows_idx.append(i)

    k = 0
    scores = np.empty((0, 0), dtype=np.float64)
    ratio = np.zeros(0, dtype=np.float64)
    if feats:
        X = np.stack(feats)
        center = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale <= _EPS] = 1.0
        Xs = (X - center) / scale
        U, S, _ = np.linalg.svd(Xs, full_matrices=False)
        total = float((S**2).sum())
        ratio = (S**2 / total) if total > _EPS else np.zeros_like(S)
        k = min(int(n_components), S.size)
        scores = U[:, :k] * S[:k]

    columns: list[pl.Series] = []
    for c in range(int(n_components)):
        values: list[float | None] = [None] * n
        if c < k:
            for row_i, score in zip(rows_idx, scores[:, c]):
                values[row_i] = float(score)
        columns.append(pl.Series(f"efd_pc{c + 1}", values, dtype=pl.Float32))

    out = (
        cells_table.clone()
        if isinstance(cells_table, pl.DataFrame)
        else pl.from_arrow(cells)
    )
    info: dict[str, Any] = {
        "explained_variance_ratio": [float(v) for v in ratio[:k]],
        "n_harmonics": int(n_harmonics),
        "n_components": int(n_components),
        "n_cells_used": len(feats),
    }
    return out.with_columns(columns), info


# ---------------------------------------------------------------------------
# contact metrics (§4.6 population.cording contact network)
# ---------------------------------------------------------------------------


def contact_metrics(
    labels_2d: np.ndarray,
    cells_table: pa.Table | pl.DataFrame,
) -> pl.DataFrame:
    """Shared-boundary contact fraction and neighbour alignment per cell.

    Shared-boundary pixel counts are computed from 4-neighbourhood
    label-pair differences of ``labels_2d`` (down and right shifts,
    counted with :func:`numpy.bincount` over dense label codes — no
    O(n²) pixel-level comparisons). Each unit shared edge contributes 1
    px to the shared perimeter of both cells.

    - ``contact_fraction`` — shared perimeter divided by the cell's
      ``perimeter`` column (0 when the perimeter is missing/zero or the
      cell is absent from ``labels_2d``).
    - ``mean_neighbor_alignment`` — shared-boundary-weighted mean of
      ``|cos(delta_orientation)|`` over the cell's neighbours, using the
      ``weighted_orientation`` column (1 = perfectly parallel
      neighbourhood, ~0 = perpendicular); 0 when the cell has no
      neighbours.

    ``cells_table`` is expected to hold the cells of the same frame as
    ``labels_2d``; with multi-frame tables the last row wins per label.

    Parameters
    ----------
    labels_2d : numpy.ndarray
        2D instance label image; 0 is background.
    cells_table : pyarrow.Table or polars.DataFrame
        Cells table with ``label``, ``perimeter`` and
        ``weighted_orientation``.

    Returns
    -------
    polars.DataFrame
        Input cells table with appended float32 columns
        ``contact_fraction`` and ``mean_neighbor_alignment``. Row order
        is preserved.
    """
    cells = _to_pa(cells_table)
    labels = np.asarray(labels_2d)
    if labels.ndim != 2:
        raise ValueError(f"labels_2d must be 2D, got shape {labels.shape}")
    li = labels.astype(np.int64, copy=False)
    # Dense recode keeps the pair matrix small for sparse label ids.
    uniq, inverse = np.unique(li, return_inverse=True)
    dense = inverse.reshape(li.shape)
    n_u = int(uniq.size)

    codes: list[np.ndarray] = []
    for sl_a, sl_b in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        a0, b0 = li[sl_a].ravel(), li[sl_b].ravel()
        d0, d1 = dense[sl_a].ravel(), dense[sl_b].ravel()
        mask = (a0 != b0) & (a0 > 0) & (b0 > 0)
        codes.append(d0[mask] * n_u + d1[mask])
    all_codes = np.concatenate(codes)
    if all_codes.size:
        shared = (
            np.bincount(all_codes, minlength=n_u * n_u)
            .reshape(n_u, n_u)
            .astype(np.float64)
        )
    else:
        shared = np.zeros((n_u, n_u), dtype=np.float64)
    shared = shared + shared.T  # symmetric shared-boundary lengths
    shared_perim = shared.sum(axis=1)

    lab_col = _int_column(cells, "label")
    perim = _optional_float_column(cells, "perimeter")
    orient = _optional_float_column(cells, "weighted_orientation")

    # Map table rows to dense label ids (labels absent from the image -> -1).
    pos = np.searchsorted(uniq, np.clip(lab_col, 0, None))
    pos_clip = np.clip(pos, 0, n_u - 1)
    found = (lab_col > 0) & (uniq[pos_clip] == lab_col)
    dense_of_row = np.where(found, pos_clip, -1)

    theta = np.full(n_u, np.nan, dtype=np.float64)
    theta[dense_of_row[found]] = orient[found]
    cos_abs = np.abs(np.cos(theta[:, None] - theta[None, :]))
    cos_abs = np.nan_to_num(cos_abs, nan=0.0)
    alignment = np.where(
        shared_perim > 0,
        (shared * cos_abs).sum(axis=1) / np.maximum(shared_perim, _EPS),
        0.0,
    )

    row_idx = np.clip(dense_of_row, 0, None)
    sp_row = np.where(found, shared_perim[row_idx], 0.0)
    al_row = np.where(found, alignment[row_idx], 0.0)
    perim_ok = np.isfinite(perim) & (perim > _EPS)
    cf_row = np.where(perim_ok, sp_row / np.maximum(perim, _EPS), 0.0)

    out = (
        cells_table.clone()
        if isinstance(cells_table, pl.DataFrame)
        else pl.from_arrow(cells)
    )
    return out.with_columns(
        [
            pl.Series("contact_fraction", cf_row.astype(np.float32)),
            pl.Series("mean_neighbor_alignment", al_row.astype(np.float32)),
        ]
    )
