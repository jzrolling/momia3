"""Sub-pixel outline extraction, curvature features and elliptic Fourier
descriptors (ARCHITECTURE.md §4.4 / §5.1).

Pipeline per label (pure function of the inputs):

1. bbox crop of the instance mask (with padding),
2. Gaussian smoothing of the mask — ``sigma`` is a *real* parameter here
   (momia2 accepted it but ignored it, always smoothing with ``sigma=1``),
3. marching squares via :func:`skimage.measure.find_contours` at ``level``,
4. optional Ramer–Douglas–Peucker simplification
   (:func:`skimage.measure.approximate_polygon`, ``tolerance``),
5. closed B-spline smoothing (``scipy.interpolate.splprep`` with
   ``per=1``, smoothing factor ``smooth``) resampled to a uniform point
   count ``n_points``,
6. optional snake refinement against the intensity image
   (:func:`skimage.segmentation.active_contour`, ``alpha``/``beta``/
   ``gamma``) for high-precision mode.

Coordinate convention: outlines are ``(N, 2)`` float64 arrays of
``(x, y)`` points in *global* image coordinates. "Closed" means the ring
is implicitly closed (the last point connects back to the first); the
first point is *not* duplicated at the end.

References
----------
[1] Ducret, A., Quardokus, E. M. & Brun, Y. V. MicrobeJ, a tool to automate
    high-content screening for bacterial cell biology. Nat. Microbiol. 1,
    16209 (2016).
[2] Zhu, J. MOMIA: Mycobacteria-Optimized Microscopy Image Analysis (PyPI,
    2021); https://pypi.org/project/momia.
[3] Zhu, J. momia2: Mycobacteria-optimized microscopy image analysis version 2
    (PyPI, 2023); https://pypi.org/project/momia2.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import polars as pl
import pyarrow as pa
from scipy import interpolate, ndimage
from skimage import measure, segmentation

from momia3 import keys as _keys
from momia3 import schema as _schema

logger = logging.getLogger(__name__)

__all__ = [
    "extract_outlines",
    "curvature_features",
    "elliptic_fourier",
    "ellipse_deviation",
    "chord_sagitta_ratio",
    "shape_descriptors",
    "fill_cells_geometry",
]

CURVATURE_COLUMNS: tuple[str, ...] = (
    "curv_min",
    "curv_max",
    "curv_mean",
    "curv_q1",
    "curv_q3",
)
"""Scalar curvature columns written by :func:`fill_cells_geometry`."""

SHAPE_COLUMNS: tuple[str, ...] = ("ellipse_dev", "sagitta_ratio")
"""Simple-shape descriptor columns written by fill_cells_geometry."""

_DEFAULT_PARAMS: dict[str, Any] = {
    "sigma": 1.0,
    "level": 0.5,
    "simplify": False,
    "tolerance": 0.75,
    "smooth": 1.0,
    # Pixel-equidistant outline sampling: output ~perimeter / downsample_factor
    # points, and fit the spline with s = smooth_scale * n_down. Evaluating the
    # spline at the natural marching-squares count keeps the pixel staircase.
    "n_points": None,
    "downsample_factor": 2.0,
    "smooth_scale": 0.03,
    # Opt-in constriction refinement: after the base B-spline smoothing, pull
    # the outline back toward the raw marching-squares contour for 1-2 rounds,
    # with a light re-smoothing each round. Restores septum neck depth that the
    # global smoothing rounds away, while keeping the outline smooth.
    # Deprecated: the mask-target constriction pull was superseded by the
    # image-guided refine_outline_* rounds below; kept for backwards
    # compatibility, candidate for removal.
    "constriction_iters": 0,  # 0 = off
    "constriction_strength": 0.6,  # pull fraction toward the raw contour per round
    "constriction_smooth_scale": 0.01,  # light spline stiffness per round (x n_out)
    # Image-guided variant: snap outline samples to the dark cell edge where the
    # mask boundary still includes bright septum background (phase contrast).
    "constriction_image_iters": 0,  # 0 = off
    "constriction_image_max_shift": 3.0,  # px, max inward move per round
    "constriction_image_strength": 0.8,
    "constriction_image_threshold_frac": 0.55,  # dark threshold between cell and background
    # Combined outline refinement (momia v1 / MicrobeJ style): one subpixel move
    # per round to the local maximum of a Sobel edge map along the normal, then
    # the intensity-threshold constriction step above, then light re-smoothing.
    # "refine_outline_mode": "edge" | "intensity" | "edge+intensity".
    "refine_outline_iters": 1,  # adopted default: one edge+intensity constriction round
    "refine_outline_mode": "edge+intensity",
    "refine_edge_step": 0.5,  # px sampling offset along the normal
    "refine_edge_cutoff": 0.2,  # normalized Sobel magnitude below which no move
    "refine_edge_max_shift": 1.5,  # px cap per round
    "refine_edge_max_round_shift": 1.5,  # revert points moving farther than this in one round
    "refine_edge_pre_expand": 0.3,  # px outward expansion before the first round (momia v1)
    "active_contour": False,
    "alpha": 0.015,
    "beta": 10.0,
    "gamma": 0.001,
    "pad": 2,
}


# ---------------------------------------------------------------------------
# table helpers
# ---------------------------------------------------------------------------


def _to_pa(table: pa.Table | pl.DataFrame) -> pa.Table:
    """Coerce a polars DataFrame or Arrow table to ``pa.Table``."""
    if isinstance(table, pa.Table):
        return table
    if isinstance(table, pl.DataFrame):
        return table.to_arrow()
    raise TypeError(f"expected pyarrow.Table or polars.DataFrame, got {type(table).__name__}")


def _back(table: pa.Table, like: pa.Table | pl.DataFrame) -> pa.Table | pl.DataFrame:
    """Return ``table`` in the same container type as ``like``."""
    return pl.from_arrow(table) if isinstance(like, pl.DataFrame) else table


def _upsert_column(table: pa.Table, name: str, arr: pa.Array) -> pa.Table:
    """Replace or append a column, keeping canonical schema metadata."""
    canonical = _schema.cells_schema()
    field = canonical.field(name) if name in canonical.names else pa.field(name, arr.type)
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), field, arr)
    return table.append_column(field, arr)


def _use_cell_keys(mapping: Mapping[int, Any]) -> bool:
    """Heuristic: dicts may be keyed by ``label`` or by ``cell_key``.

    ``cell_key`` values exceed :data:`momia3.keys.LABEL_MAX`; if any key
    does, the whole mapping is treated as cell-key keyed.
    """
    return any(int(k) > _keys.LABEL_MAX for k in mapping)


# ---------------------------------------------------------------------------
# outline extraction
# ---------------------------------------------------------------------------


def _resample_closed_arclength(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed ring to ``n`` points uniform in arc length."""
    ring = np.vstack([points, points[:1]])
    seg = np.hypot(np.diff(ring[:, 0]), np.diff(ring[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return np.repeat(points[:1], n, axis=0)
    targets = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(targets, s, ring[:, 0])
    y = np.interp(targets, s, ring[:, 1])
    return np.column_stack([x, y])


def _smooth_closed(points: np.ndarray, smooth: float, n_points: int | None) -> np.ndarray:
    """Closed B-spline smoothing resampled to ``n_points`` (or input count).

    Falls back to arc-length resampling without smoothing when the spline
    fit fails (degenerate/too few points).
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    n = len(pts) if n_points is None else int(n_points)
    if len(pts) < 4:
        return _resample_closed_arclength(pts, n)
    try:
        tck, _ = interpolate.splprep(
            [pts[:, 0], pts[:, 1]], s=float(smooth), per=1, k=3, quiet=1
        )
        uu = np.linspace(0.0, 1.0, n, endpoint=False)
        x, y = interpolate.splev(uu, tck, der=0)
        return np.column_stack([x, y])
    except (ValueError, TypeError) as exc:
        logger.debug("closed spline fit failed (%s); using linear resample", exc)
        return _resample_closed_arclength(pts, n)


def _outline_normals(ring: np.ndarray) -> np.ndarray:
    """Unit normals of a closed ring (direction sign is resolved by caller)."""
    p = np.asarray(ring, dtype=np.float64)
    tang = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1)[:, None], 1e-12)
    return nrm


def _ring_from_binary_mask(work: np.ndarray, params: Mapping[str, Any]) -> np.ndarray | None:
    """Same contour pipeline as _outline_single, but starting from a mask.

    Used by the image-guided constriction rounds: the intensity-thresholded
    mask already excludes the bright septum bridge, so re-extracting its ring
    gives the deeper waist directly.
    """
    sigma = float(params.get("sigma", 1.0))
    smoothed = ndimage.gaussian_filter(work.astype(np.float64), sigma) if sigma > 0 else work.astype(np.float64)
    contours = measure.find_contours(smoothed, level=float(params.get("level", 0.5)))
    if not contours:
        return None
    ring_rc = max(contours, key=len)
    ring_xy = np.column_stack([ring_rc[:, 1], ring_rc[:, 0]])
    factor = float(params.get("downsample_factor", 2.0) or 0.0)
    if factor > 0 and len(ring_xy) > 8:
        closed = np.vstack([ring_xy, ring_xy[:1]])
        perimeter = float(np.sum(np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))))
        n_down = max(16, int(round(perimeter / factor)))
        base_ring = _resample_closed_arclength(ring_xy, n_down)
    else:
        base_ring = ring_xy
        n_down = len(ring_xy)
    n_out = n_down if params["n_points"] is None else int(params["n_points"])
    smooth_scale = params.get("smooth_scale")
    smooth = (
        float(smooth_scale) * n_down
        if smooth_scale is not None
        else float(params["smooth"])
    )
    return _smooth_closed(base_ring, smooth, n_out)


def _constriction_intensity_step(
    ring: np.ndarray,
    intensity: np.ndarray,
    mask: np.ndarray,
    max_shift: float,
    strength: float,
    threshold_frac: float,
    params: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Deepen concave notches toward the intensity-defined dark cell edge.

    The binary mask may bridge a bright septum gap; thresholding the image at a
    level between cell body and background cuts that gap. The thresholded mask
    is re-contoured with the normal outline pipeline, and only *concave*
    boundary points that move *inward* are taken (capped at max_shift). This
    keeps the overall cell size and only sharpens real constrictions.
    """
    pts = np.asarray(ring, dtype=np.float64).copy()
    h, w = intensity.shape
    nrm = _outline_normals(pts)

    def sample_mask(pp):
        cols = np.clip(np.round(pp[:, 0]).astype(int), 0, w - 1)
        rows = np.clip(np.round(pp[:, 1]).astype(int), 0, h - 1)
        return mask[rows, cols].astype(np.float64)

    m_plus = sample_mask(pts + 0.8 * nrm)
    m_minus = sample_mask(pts - 0.8 * nrm)
    outward = nrm * np.where(m_plus <= m_minus, 1.0, -1.0)[:, None]
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    dev = pts - 0.5 * (prev + nxt)
    concave = np.sum(dev * outward, axis=1) < 0.0

    dark = float(np.percentile(intensity[mask], 25.0)) if np.any(mask) else 0.0
    outside = ~mask
    bright = float(np.percentile(intensity[outside], 75.0)) if np.any(outside) else dark + 1.0
    if not np.isfinite(dark) or not np.isfinite(bright) or bright - dark < 1e-9:
        return pts
    frac = float(np.clip(threshold_frac, 0.05, 0.95))
    refined = None
    for _ in range(4):
        cand = mask & (intensity < dark + frac * (bright - dark))
        n_labels = int(ndimage.label(cand)[1])
        if n_labels == 1 and cand.sum() >= 0.45 * mask.sum():
            refined = cand
            break
        frac = min(0.95, frac + 0.1)
    if refined is None:
        return pts
    ref_params = dict(params or {})
    ref_params["smooth_scale"] = 0.5 * float(ref_params.get("smooth_scale", 0.03) or 0.03)
    ref = _ring_from_binary_mask(refined, ref_params)
    if ref is None or len(ref) < 8:
        return pts
    from scipy.spatial import cKDTree as _cKDTree

    ref_closed = np.vstack([ref, ref[:1]])
    _, nearest = _cKDTree(ref_closed).query(pts)
    target = ref_closed[nearest]
    delta = target - pts
    d_out = np.sum(delta * outward, axis=1)
    shift = np.linalg.norm(delta, axis=1)
    inward_amt = -d_out
    # Uniform halo shrink moves every point by roughly the same amount; compare
    # with the local (circular) median to keep only locally-deeper notches.
    local = ndimage.median_filter(inward_amt, size=11, mode="wrap")
    excess = inward_amt - local
    accept = concave & (excess > 0.2) & (shift <= max_shift)
    moved = pts.copy()
    moved[accept] = pts[accept] + strength * delta[accept]
    return moved


def _sobel_edge_map(intensity: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Normalized Sobel edge energy map.

    Preprocessing follows MicrobeJ ImMask [1]: gaussian -> ImageJ Sobel
    (FloatProcessor.findEdges) -> gaussian, with the same role for the two
    sigmas. The momia v1 implementation [2] uses the same edge image.
    """
    from skimage import filters

    img = np.asarray(intensity, dtype=np.float64)
    lo, hi = np.percentile(img, [2.0, 98.0])
    img = np.clip((img - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    img = ndimage.gaussian_filter(img, sigma=sigma)
    edge = filters.sobel(img)
    edge = ndimage.gaussian_filter(edge, sigma=sigma)
    lo, hi = np.percentile(edge, [2.0, 99.5])
    edge = np.clip((edge - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return edge


def _edge_snap_step(
    ring: np.ndarray,
    edge: np.ndarray,
    step: float,
    cutoff: float,
    max_shift: float,
) -> np.ndarray:
    """One subpixel step of each outline sample to the local Sobel maximum.

    Mirrors momia v1 contour_optimization [2]: sample the edge map at the
    current position and +/- one step along the normal, keep the central sample
    when it is the maximum (subpixel quadratic correction), otherwise move half
    a step toward the better side. Weak edges (below cutoff) are left alone.
    The MicrobeJ high-precision mode [1] solves the same normal-line maximum
    problem with a single full-profile pass (ContourRoi.getCorrectedContour in
    MicrobeJ 5.13p).
    """
    pts = np.asarray(ring, dtype=np.float64).copy()
    h, w = edge.shape
    nrm = _outline_normals(pts)
    step = max(float(step), 0.1)

    def sample(pp):
        cols = np.clip(pp[:, 0], 0, w - 1.0)
        rows = np.clip(pp[:, 1], 0, h - 1.0)
        return ndimage.map_coordinates(edge, [rows, cols], order=1, mode="nearest")

    v_in = sample(pts + step * nrm)
    v0 = sample(pts)
    v_out = sample(pts - step * nrm)
    best = np.maximum(np.maximum(v_in, v0), v_out)
    weak = best < float(cutoff)
    denom = v_in - 2.0 * v0 + v_out
    denom = np.where(np.abs(denom) < 1e-9, np.inf, denom)
    sub = 0.5 * (v_in - v_out) / denom * step  # parabolic vertex offset (px)
    sub = np.clip(sub, -step, step)
    move = np.zeros_like(pts)
    center_max = (v0 >= v_in) & (v0 >= v_out) & ~weak
    move[center_max] = sub[center_max, None] * nrm[center_max]
    in_max = (v_in > v_out) & (v_in > v0) & ~weak
    move[in_max] = 0.5 * step * nrm[in_max]
    out_max = (v_out > v_in) & (v_out > v0) & ~weak
    move[out_max] = -0.5 * step * nrm[out_max]
    shift = np.linalg.norm(move, axis=1)
    too_far = shift > float(max_shift)
    move[too_far] = 0.0
    return pts + move


def _outline_single(

    crop: np.ndarray,
    intensity_crop: np.ndarray | None,
    params: Mapping[str, Any],
) -> np.ndarray | None:
    """Extract the outermost sub-pixel outline of one cropped instance mask.

    Parameters
    ----------
    crop : numpy.ndarray of bool
        Padded bounding-box crop of the instance (see
        :func:`extract_outlines`; the crop carries the same ``pad`` margin
        the full-frame version used).
    intensity_crop : numpy.ndarray or None
        Matching intensity crop for active-contour refinement.
    params : mapping
        Resolved parameter dict (see :func:`extract_outlines`).

    Returns
    -------
    numpy.ndarray or None
        ``(N, 2)`` ``(x, y)`` coordinates **in crop space** (the caller
        adds the crop origin), or None when no contour could be found
        (e.g. single-pixel objects).
    """
    if not crop.any():
        return None
    work = crop.astype(np.float64)
    sigma = float(params["sigma"])
    smoothed = ndimage.gaussian_filter(work, sigma=sigma) if sigma > 0 else work

    contours = measure.find_contours(smoothed, level=float(params["level"]))
    if not contours:
        # Fallback for tiny/thin objects: raw mask at the half level.
        contours = measure.find_contours(work, level=0.5)
        if not contours:
            logger.debug("no contour found for cropped instance mask")
            return None
    ring_rc = max(contours, key=len)  # (row, col)

    if bool(params["simplify"]) and len(ring_rc) > 4:
        ring_rc = measure.approximate_polygon(
            ring_rc, tolerance=float(params["tolerance"])
        )

    # (row, col) -> (x, y), crop coordinates
    ring_xy = np.column_stack([ring_rc[:, 1], ring_rc[:, 0]])
    raw_contour = ring_xy  # pre-smoothing marching-squares ring
    # Pixel-equidistant anti-staircase smoothing: downsample the marching-
    # squares ring to ~downsample_factor px spacing, then fit a stiffer
    # closed B-spline (s proportional to the point count).
    factor = float(params.get("downsample_factor", 2.0) or 0.0)
    if factor > 0 and len(ring_xy) > 8:
        closed = np.vstack([ring_xy, ring_xy[:1]])
        perimeter = float(
            np.sum(np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1])))
        )
        n_down = max(16, int(round(perimeter / factor)))
        base_ring = _resample_closed_arclength(ring_xy, n_down)
    else:
        base_ring = ring_xy
        n_down = len(ring_xy)
    n_out = n_down if params["n_points"] is None else int(params["n_points"])
    smooth_scale = params.get("smooth_scale")
    smooth = (
        float(smooth_scale) * n_down
        if smooth_scale is not None
        else float(params["smooth"])
    )
    ring_xy = _smooth_closed(base_ring, smooth, n_out)

    # Combined refinement (momia v1 / MicrobeJ style + intensity constriction).
    refine_iters = int(params.get("refine_outline_iters", 0) or 0)
    refine_mode = str(params.get("refine_outline_mode", "edge+intensity")).lower()
    if refine_iters > 0 and intensity_crop is not None and len(ring_xy) > 16:
        icrop = np.asarray(intensity_crop, dtype=np.float64)
        edge_map = _sobel_edge_map(icrop, sigma=1.0)
        step = float(params.get("refine_edge_step", 0.5))
        cutoff = float(params.get("refine_edge_cutoff", 0.2))
        max_shift = float(params.get("refine_edge_max_shift", 1.5))
        expand = float(params.get("refine_edge_pre_expand", 0.3))
        if expand > 0.0:
            ring_xy = ring_xy + expand * _outline_normals(ring_xy)
        light = max(float(params.get("constriction_smooth_scale", 0.01)) * n_out, 1.0)
        for _ in range(refine_iters):
            before = ring_xy.copy()
            if refine_mode in ("edge", "edge+intensity"):
                ring_xy = _edge_snap_step(ring_xy, edge_map, step, cutoff, max_shift)
                far = np.linalg.norm(ring_xy - before, axis=1) > float(
                    params.get("refine_edge_max_round_shift", 1.5)
                )
                ring_xy[far] = before[far]
            if refine_mode in ("intensity", "edge+intensity"):
                thr_frac = float(params.get("constriction_image_threshold_frac", 0.55))
                strength_i = float(np.clip(params.get("constriction_image_strength", 0.8), 0.0, 1.0))
                ring_xy = _constriction_intensity_step(
                    ring_xy, icrop, np.asarray(crop, dtype=bool),
                    float(params.get("constriction_image_max_shift", 3.0)),
                    strength_i, thr_frac, params,
                )
            ring_xy = _smooth_closed(ring_xy, light, n_out)

    # Opt-in: restore septum constriction lost to the global spline smoothing.
    # Snap toward the raw marching-squares ring, then re-smooth lightly; 1-2
    # rounds deepen the necks without bringing back the pixel staircase.
    constriction_iters = int(params.get("constriction_iters", 0) or 0)
    if constriction_iters > 0 and len(ring_xy) > 16 and len(raw_contour) > 8:
        # Target the *un-Gaussian-smoothed* mask contour: sigma=1 blurs thin
        # necks, so pulling toward the sigma=1 ring barely changes anything.
        # On the raw binary mask the septum waist is sharper.
        raw_target = raw_contour
        try:
            raw_contours = measure.find_contours(work, level=0.5)
            if raw_contours:
                raw_ring_rc = max(raw_contours, key=len)
                raw_target = np.column_stack([raw_ring_rc[:, 1], raw_ring_rc[:, 0]])
        except Exception as exc:  # pragma: no cover - fall back to smoothed ring
            logger.debug("raw mask contour failed (%s)", exc)
        raw_closed = np.vstack([raw_target, raw_target[:1]])
        raw_seg = np.hypot(np.diff(raw_closed[:, 0]), np.diff(raw_closed[:, 1]))
        raw_cum = np.concatenate([[0.0], np.cumsum(raw_seg)])
        raw_total = float(raw_cum[-1])
        strength = float(np.clip(params.get("constriction_strength", 0.6), 0.0, 1.0))
        light_smooth = float(params.get("constriction_smooth_scale", 0.01)) * n_out
        # Arc-length correspondence (not nearest-point): ordering along the
        # raw ring is preserved, so the pull cannot jump across a thin neck.
        start = 0
        if raw_total > 1e-9:
            from scipy.spatial import cKDTree as _cKDTree

            _, nearest = _cKDTree(raw_closed).query(ring_xy[0])
            start = int(nearest)
        for _ in range(constriction_iters):
            ring_closed = np.vstack([ring_xy, ring_xy[:1]])
            ring_seg = np.hypot(np.diff(ring_closed[:, 0]), np.diff(ring_closed[:, 1]))
            ring_cum = np.concatenate([[0.0], np.cumsum(ring_seg)])
            frac = ring_cum[:-1] / max(float(ring_cum[-1]), 1e-9)
            target_s = (raw_cum[start] + frac * raw_total) % max(raw_total, 1e-9)
            target = np.column_stack([
                np.interp(target_s, raw_cum, raw_closed[:, 0]),
                np.interp(target_s, raw_cum, raw_closed[:, 1]),
            ])
            ring_xy = ring_xy + strength * (target - ring_xy)
            ring_xy = _smooth_closed(ring_xy, max(light_smooth, 1.0), n_out)

    # Image-guided variant: deepen the septum waist where the mask boundary
    # still covers bright background (phase contrast).
    image_iters = int(params.get("constriction_image_iters", 0) or 0)
    if image_iters > 0 and intensity_crop is not None and len(ring_xy) > 16:
        icrop = np.asarray(intensity_crop, dtype=np.float64)
        max_shift = float(params.get("constriction_image_max_shift", 3.0))
        strength_i = float(np.clip(params.get("constriction_image_strength", 0.8), 0.0, 1.0))
        thr_frac = float(params.get("constriction_image_threshold_frac", 0.55))
        light_smooth_i = float(params.get("constriction_smooth_scale", 0.01)) * n_out
        for _ in range(image_iters):
            ring_xy = _constriction_intensity_step(
                ring_xy, icrop, np.asarray(crop, dtype=bool),
                max_shift, strength_i, thr_frac, params,
            )
            ring_xy = _smooth_closed(ring_xy, max(light_smooth_i, 1.0), n_out)

    if bool(params["active_contour"]) and intensity_crop is not None:
        icrop = np.asarray(intensity_crop, dtype=np.float64)
        span = float(icrop.max() - icrop.min())
        if span > 0:
            icrop = (icrop - icrop.min()) / span
        snake_rc = np.column_stack([ring_xy[:, 1], ring_xy[:, 0]])
        try:
            snake = segmentation.active_contour(
                icrop,
                snake_rc,
                alpha=float(params["alpha"]),
                beta=float(params["beta"]),
                gamma=float(params["gamma"]),
                boundary_condition="periodic",
            )
            ring_xy = np.column_stack([snake[:, 1], snake[:, 0]])
            # Restore uniform point spacing after snake deformation.
            ring_xy = _resample_closed_arclength(ring_xy, len(ring_xy))
        except (ValueError, TypeError) as exc:
            logger.debug("active contour refinement failed (%s); keeping spline", exc)

    return ring_xy


def extract_outlines(
    labels_2d: np.ndarray,
    intensity_2d: np.ndarray | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[int, np.ndarray]:
    """Extract sub-pixel closed outlines for every labelled instance.

    Parameters
    ----------
    labels_2d : numpy.ndarray of int
        2-D instance label image (0 = background).
    intensity_2d : numpy.ndarray of float, optional
        Intensity image used only when ``params["active_contour"]`` is
        True.
    params : mapping, optional
        ``sigma`` (mask smoothing, px; default 1.0 — a real parameter,
        unlike momia2 which ignored it), ``level`` (marching-squares
        level, default 0.5), ``simplify``/``tolerance`` (RDP),
        ``downsample_factor`` (px spacing of the working ring, default 2.0),
        ``smooth_scale`` (B-spline s = smooth_scale * n_down, default 0.05),
        ``smooth`` (absolute B-spline smoothing factor, used only when
        ``smooth_scale`` is None), ``n_points`` (output count; None =
        ~perimeter / downsample_factor),
        ``active_contour``/``alpha``/``beta``/``gamma`` (snake
        refinement), ``pad`` (bbox padding, px).

    Returns
    -------
    dict of int to numpy.ndarray
        ``label -> (N, 2)`` float64 ``(x, y)`` outline in global image
        coordinates, implicitly closed (first point not duplicated).
    """
    p: dict[str, Any] = {**_DEFAULT_PARAMS, **dict(params or {})}
    labels = np.asarray(labels_2d)
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64)
    pad = int(p["pad"])
    img_h, img_w = labels.shape
    out: dict[int, np.ndarray] = {}
    # ONE bounding-box pass for the whole frame: ``find_objects`` returns the
    # tight bbox of every label, so per-label work crops to ``bbox + pad``
    # instead of re-scanning the full image per label (the O(n_labels x
    # pixels) ``labels == lab`` + ``nonzero`` pattern that dominated the
    # batch cost; see benchmarks/benchmark_report.md §2).
    slices = ndimage.find_objects(labels)
    for lab, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        y0 = max(sl[0].start - pad, 0)
        y1 = min(sl[0].stop + pad, img_h)
        x0 = max(sl[1].start - pad, 0)
        x1 = min(sl[1].stop + pad, img_w)
        crop = labels[y0:y1, x0:x1] == lab
        intensity_crop = (
            None if intensity_2d is None else np.asarray(intensity_2d)[y0:y1, x0:x1]
        )
        ring = _outline_single(crop, intensity_crop, p)
        if ring is not None:
            out[lab] = ring + np.array([x0, y0], dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# curvature
# ---------------------------------------------------------------------------


def _bend_angle_closed(data: np.ndarray, window: int, deg: bool = False) -> np.ndarray:
    """Windowed bend angle of a closed ring (complex-number method).

    Port of ``momia2/utils/linalg.py::bend_angle_closed``: for each point
    ``p2`` the angle between the chords ``p1 -> p2`` and ``p1 -> p3`` with
    ``p1``/``p3`` taken ``window`` samples back/forward along the ring,
    computed as the argument of a complex ratio.
    """
    d = np.asarray(data, dtype=np.float64)
    n = len(d)
    w = max(1, min(int(window), (n - 1) // 2))
    p1 = np.concatenate((d[-w:], d[:-w])).T
    p2 = d.T
    p3 = np.concatenate((d[w:], d[:w])).T
    p1p2 = p1[0] + 1j * p1[1] - (p2[0] + 1j * p2[1])
    p1p3 = p1[0] + 1j * p1[1] - (p3[0] + 1j * p3[1])
    return np.angle(p1p3 / p1p2, deg=deg)


def curvature_features(outline: np.ndarray, window: int = 3) -> dict[str, float]:
    """Windowed bend-angle statistics of a closed outline.

    Parameters
    ----------
    outline : numpy.ndarray
        ``(N, 2)`` ``(x, y)`` closed outline.
    window : int
        Half-window in samples (clamped to ``(N - 1) // 2``).

    Returns
    -------
    dict of str to float
        ``curv_min``, ``curv_max``, ``curv_mean``, ``curv_q1``,
        ``curv_q3`` in radians per window (matching the schema ``curv_*``
        columns).
    """
    angles = _bend_angle_closed(outline, window, deg=False)
    return {
        "curv_min": float(np.min(angles)),
        "curv_max": float(np.max(angles)),
        "curv_mean": float(np.mean(angles)),
        "curv_q1": float(np.percentile(angles, 25)),
        "curv_q3": float(np.percentile(angles, 75)),
    }


# ---------------------------------------------------------------------------
# elliptic Fourier descriptors
# ---------------------------------------------------------------------------


def _outline_principals(
    ring: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Centroid, unit major axis, aspect ratio and polygon area of an outline.

    The major axis and aspect ratio come from the covariance of the outline
    vertices (equivalent to the second-moment ratio for a closed curve); the
    centroid/area use the shoelace formula. Used only to fill optional
    arguments when the caller does not pass mask-derived values.
    """
    r = np.asarray(ring, dtype=np.float64)
    x, y = r[:, 0], r[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    signed = float(cross.sum())
    area = abs(signed) / 2.0
    if area > 1e-9:
        cx = float(((x + np.roll(x, -1)) * cross).sum() / (3.0 * signed))
        cy = float(((y + np.roll(y, -1)) * cross).sum() / (3.0 * signed))
    else:
        cx, cy = float(x.mean()), float(y.mean())
    d = r - np.array([cx, cy], dtype=np.float64)[None, :]
    cov = (d.T @ d) / max(len(r), 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    l1 = max(float(evals[order[0]]), 1e-12)
    l2 = max(float(evals[order[1]]), 1e-12)
    axis = np.asarray(evecs[:, order[0]], dtype=np.float64)
    nrm = float(np.hypot(axis[0], axis[1]))
    axis = np.array([1.0, 0.0]) if nrm <= 1e-12 else axis / nrm
    return np.array([cx, cy], dtype=np.float64), axis, float(np.sqrt(l1 / l2)), area


def ellipse_deviation(
    outline: np.ndarray,
    *,
    centroid: np.ndarray | None = None,
    axis: np.ndarray | None = None,
    aspect_ratio: float | None = None,
    area: float | None = None,
) -> float:
    """Mean relative deviation of an outline from its best-fit ellipse.

    Simple cocci/oval/straight-rod masks are close to an ellipse (small
    value); curved, lobed or branched shapes deviate strongly. This is the
    cheap pre-gate used by the midline router; the same descriptor is written
    as a per-cell feature (ellipse_dev) by fill_cells_geometry.

    Parameters
    ----------
    outline : (N, 2) numpy.ndarray
        Closed (x, y) outline in local crop coordinates.
    centroid, axis, aspect_ratio, area : optional
        Mask-derived context. When omitted they are estimated from the
        outline (polygon area/centroid, second-moment major axis).
    """
    ring = np.asarray(outline, dtype=np.float64)
    if len(ring) < 8:
        return 1.0
    c0, ax0, ar0, area0 = _outline_principals(ring)
    c = c0 if centroid is None else np.asarray(centroid, dtype=np.float64)
    ax = ax0 if axis is None else np.asarray(axis, dtype=np.float64)
    ar = ar0 if aspect_ratio is None else float(aspect_ratio)
    a_area = area0 if area is None else float(area)
    if a_area < 8 or ar <= 0:
        return 1.0
    a = float(np.sqrt(a_area * ar / np.pi))
    b = max(a / ar, 1e-6)
    rel = ring - c[None, :]
    nrm = float(np.hypot(ax[0], ax[1]))
    if nrm <= 1e-12:
        return 1.0
    ax = ax / nrm
    perp = np.array([-ax[1], ax[0]], dtype=np.float64)
    u = (rel @ ax) / max(a, 1e-6)
    v = (rel @ perp) / max(b, 1e-6)
    return float(np.mean(np.abs(np.hypot(u, v) - 1.0)))


def chord_sagitta_ratio(
    outline: np.ndarray,
    *,
    centroid: np.ndarray | None = None,
    axis: np.ndarray | None = None,
    n_samples: int = 9,
) -> float:
    """Max chord-midpoint sagitta away from the major axis, normalized by width.

    A straight rod gives a small value even when its outline is slightly
    asymmetric; a bent (banana) cell bows the perpendicular chord midpoints
    away from the axis. Scale-free; no skeleton/distance transform needed.
    """
    ring = np.asarray(outline, dtype=np.float64)
    if len(ring) < 8:
        return 1.0
    c0, ax0, _, _ = _outline_principals(ring)
    c = c0 if centroid is None else np.asarray(centroid, dtype=np.float64)
    axis_v = ax0 if axis is None else np.asarray(axis, dtype=np.float64)
    nrm = float(np.hypot(axis_v[0], axis_v[1]))
    if nrm <= 1e-12:
        return 1.0
    axis_v = axis_v / nrm
    perp = np.array([-axis_v[1], axis_v[0]], dtype=np.float64)
    t = (ring - c[None, :]) @ axis_v
    tmin, tmax = np.percentile(t, [2.0, 98.0])
    if tmax - tmin < 2.0:
        return 1.0
    ts = np.linspace(tmin + 0.15 * (tmax - tmin), tmax - 0.15 * (tmax - tmin), n_samples)
    a_pt = ring
    b_pt = np.roll(ring, -1, axis=0)
    e = b_pt - a_pt
    p0 = c[None, :] + ts[:, None] * axis_v[None, :]
    hits_fwd = np.full((len(ts), 2), np.nan)
    hits_bwd = np.full((len(ts), 2), np.nan)
    for slot, dvec in ((0, perp), (1, -perp)):
        den = dvec[0] * (-e[:, 1]) - (-e[:, 0]) * dvec[1]
        ok = np.abs(den) > 1e-12
        rhs = a_pt[None, :, :] - p0[:, None, :]
        tpar = (rhs[..., 0] * (-e[None, :, 1]) - (-e[None, :, 0]) * rhs[..., 1]) / np.where(ok, den, 1.0)[None, :]
        uu = (dvec[0] * rhs[..., 1] - dvec[1] * rhs[..., 0]) / np.where(ok, den, 1.0)[None, :]
        valid = ok[None, :] & (uu >= -0.05) & (uu <= 1.05) & (tpar > 1e-6)
        tpar = np.where(valid, tpar, np.inf)
        sel = np.argmin(tpar, axis=1)
        tval = tpar[np.arange(len(ts)), sel]
        hit = p0 + tval[:, None] * np.asarray(dvec, dtype=np.float64)[None, :]
        hit[~np.isfinite(tval)] = np.nan
        if slot == 0:
            hits_fwd = hit
        else:
            hits_bwd = hit
    good = np.isfinite(hits_fwd[:, 0]) & np.isfinite(hits_bwd[:, 0])
    if int(np.count_nonzero(good)) < 3:
        return 1.0
    mid = 0.5 * (hits_fwd[good] + hits_bwd[good])
    devs = np.abs((mid - p0[good]) @ perp)
    widths = np.linalg.norm(hits_fwd[good] - hits_bwd[good], axis=1)
    width = float(np.median(widths))
    if width <= 1e-6:
        return 1.0
    return float(np.max(devs)) / width


def shape_descriptors(
    outline: np.ndarray,
    *,
    centroid: np.ndarray | None = None,
    axis: np.ndarray | None = None,
    aspect_ratio: float | None = None,
    area: float | None = None,
) -> dict[str, float]:
    """Simple-shape descriptors of one outline (cells-table features).

    Returns ellipse_dev and sagitta_ratio, the two curvature-aware gate
    metrics of the degenerate/equidistant midline router, so downstream
    analyses can use the same quantities as single-cell features.
    """
    ring = np.asarray(outline, dtype=np.float64)
    c0, ax0, ar0, area0 = _outline_principals(ring)
    kwargs = {
        "centroid": c0 if centroid is None else np.asarray(centroid, dtype=np.float64),
        "axis": ax0 if axis is None else np.asarray(axis, dtype=np.float64),
        "aspect_ratio": ar0 if aspect_ratio is None else float(aspect_ratio),
        "area": area0 if area is None else float(area),
    }
    return {
        "ellipse_dev": ellipse_deviation(ring, **kwargs),
        "sagitta_ratio": chord_sagitta_ratio(
            ring, centroid=kwargs["centroid"], axis=kwargs["axis"]
        ),
    }


def elliptic_fourier(outline: np.ndarray, n_harmonics: int = 10) -> np.ndarray:
    """Elliptic Fourier descriptors with Kuhl–Giardina normalization.

    The descriptors are invariant to translation (by construction, from
    contour deltas) and normalized for rotation, scale and starting point
    using the first harmonic (Kuhl & Giardina 1982).

    Parameters
    ----------
    outline : numpy.ndarray
        ``(N, 2)`` ``(x, y)`` outline (closed or open; closed by joining
        the last point to the first).
    n_harmonics : int
        Number of harmonics to compute.

    Returns
    -------
    numpy.ndarray
        ``(n_harmonics, 4)`` float64 array of ``(a_n, b_n, c_n, d_n)``
        coefficients. A degenerate outline (zero perimeter) yields zeros.
    """
    xy = np.asarray(outline, dtype=np.float64)
    if len(xy) < 2:
        return np.zeros((int(n_harmonics), 4), dtype=np.float64)
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[:1]])
    deltas = np.diff(xy, axis=0)
    dx, dy = deltas[:, 0], deltas[:, 1]
    dt = np.hypot(dx, dy)
    keep = dt > 0
    if not np.all(keep):
        dx, dy, dt = dx[keep], dy[keep], dt[keep]
    if dt.size == 0:
        return np.zeros((int(n_harmonics), 4), dtype=np.float64)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    period = t[-1]

    coeffs = np.zeros((int(n_harmonics), 4), dtype=np.float64)
    t0, t1 = t[:-1], t[1:]
    for n in range(1, int(n_harmonics) + 1):
        omega = 2.0 * np.pi * n / period
        pref = period / (2.0 * n * n * np.pi * np.pi)
        cos_d = np.cos(omega * t1) - np.cos(omega * t0)
        sin_d = np.sin(omega * t1) - np.sin(omega * t0)
        rx = dx / dt
        ry = dy / dt
        coeffs[n - 1] = [
            pref * np.sum(rx * cos_d),
            pref * np.sum(rx * sin_d),
            pref * np.sum(ry * cos_d),
            pref * np.sum(ry * sin_d),
        ]

    # --- Kuhl-Giardina normalization -----------------------------------
    a1, b1, c1, d1 = coeffs[0]
    denom = a1 * a1 - b1 * b1 + c1 * c1 - d1 * d1
    if abs(denom) < 1e-15 and abs(a1 * b1 + c1 * d1) < 1e-15:
        return coeffs  # degenerate first harmonic; return raw
    theta1 = 0.5 * np.arctan2(2.0 * (a1 * b1 + c1 * d1), denom)

    # Rotate each harmonic's phase by n * theta1.
    for n in range(1, int(n_harmonics) + 1):
        a, b, c, d = coeffs[n - 1]
        cn, sn = np.cos(n * theta1), np.sin(n * theta1)
        coeffs[n - 1] = [
            a * cn + b * sn,
            -a * sn + b * cn,
            c * cn + d * sn,
            -c * sn + d * cn,
        ]

    # Rotate the coordinate axes so the first harmonic's x-component
    # carries the full semi-major axis.
    psi = np.arctan2(coeffs[0, 2], coeffs[0, 0])
    cp, sp = np.cos(psi), np.sin(psi)
    for n in range(int(n_harmonics)):
        a, b, c, d = coeffs[n]
        coeffs[n] = [
            a * cp + c * sp,
            b * cp + d * sp,
            -a * sp + c * cp,
            -b * sp + d * cp,
        ]

    # Scale invariance and sign convention.
    scale = float(np.hypot(coeffs[0, 0], coeffs[0, 2]))
    if scale > 0:
        coeffs /= scale
    if coeffs[0, 0] < 0:
        coeffs = -coeffs
    return coeffs


# ---------------------------------------------------------------------------
# table writer
# ---------------------------------------------------------------------------


def fill_cells_geometry(
    cells_table: pa.Table | pl.DataFrame,
    outlines: Mapping[int, np.ndarray],
    curvatures: Mapping[int, Mapping[str, float]] | None = None,
) -> pa.Table | pl.DataFrame:
    """Write outline geometry and curvature scalars into a cells table.

    Parameters
    ----------
    cells_table : pyarrow.Table or polars.DataFrame
        Cells table (schema §2.3). The ``outline`` struct column and the
        curv_* and shape descriptor scalar columns are replaced/created.
    outlines : mapping of int to numpy.ndarray
        ``label -> (N, 2)`` outline, or ``cell_key -> outline``. Keying
        is inferred: any key larger than
        :data:`momia3.keys.LABEL_MAX` switches the whole mapping to
        cell-key mode.
    curvatures : mapping of int to mapping, optional
        Precomputed curvature stats keyed like ``outlines``. When None,
        :func:`curvature_features` with the default window is computed
        from each outline.

    Returns
    -------
    pyarrow.Table or polars.DataFrame
        Updated table in the same container type as the input. Rows
        without a matching outline keep null geometry and null curvature.
    """
    table = _to_pa(cells_table)
    by_key = _use_cell_keys(outlines)
    row_ids = table.column("cell_key" if by_key else "label").to_pylist()

    if curvatures is None:
        curvatures = {k: curvature_features(v) for k, v in outlines.items()}
    shapes = {k: shape_descriptors(v) for k, v in outlines.items()}

    outline_vals: list[dict[str, list[float]] | None] = []
    curv_vals: dict[str, list[float | None]] = {c: [] for c in CURVATURE_COLUMNS}
    shape_vals: dict[str, list[float | None]] = {c: [] for c in SHAPE_COLUMNS}
    for rid in row_ids:
        ring = outlines.get(rid)
        if ring is None:
            outline_vals.append(None)
            for c in CURVATURE_COLUMNS:
                curv_vals[c].append(None)
            for c in SHAPE_COLUMNS:
                shape_vals[c].append(None)
            continue
        arr = np.asarray(ring, dtype=np.float64)
        outline_vals.append(
            {"x": arr[:, 0].astype(np.float32).tolist(),
             "y": arr[:, 1].astype(np.float32).tolist()}
        )
        stats = curvatures.get(rid, {})
        for c in CURVATURE_COLUMNS:
            value = stats.get(c)
            curv_vals[c].append(None if value is None else float(value))
        desc = shapes.get(rid, {})
        for c in SHAPE_COLUMNS:
            value = desc.get(c)
            shape_vals[c].append(None if value is None else float(value))

    table = _upsert_column(table, "outline", pa.array(outline_vals, type=_schema.OUTLINE_T))
    for c in CURVATURE_COLUMNS:
        table = _upsert_column(table, c, pa.array(curv_vals[c], type=pa.float32()))
    for c in SHAPE_COLUMNS:
        table = _upsert_column(table, c, pa.array(shape_vals[c], type=pa.float32()))
    return _back(table, cells_table)
