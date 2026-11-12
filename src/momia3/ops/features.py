"""Aggregated per-cell feature computation (ARCHITECTURE.md §4.3–§4.6).

One entry point, :func:`cell_features`, computes every feature group on a
shared :class:`~momia3.ops._batch.FrameBatch` (one label sort per frame)
and a shared :class:`~momia3.ops.rod_frame.RodFrame` (one midline
resampling + KDTree per cell), so nested or repeated features never
re-derive the same artifact.

Feature groups (all optional, requested through ``groups``):

``shape``
    Feret (max/min), radius min/mean/max, convex area/perimeter, filled
    area, equivalent diameter, Euler number, skeleton length/endpoints/
    branch points, angularity, roundness and 30 Zernike moment magnitudes.
``geometry``
    Rod-frame scalars: constriction depth, septum width/position, pole
    width (needs a cells table with ``midline``/``width_profile``).
``intensity``
    Median absolute deviation, intensity-weighted mass displacement,
    fraction above an Otsu threshold, local ring background and SNR.
``texture``
    The 13 Haralick GLCM features plus Haralick granularity.
``coloc``
    Pearson/Spearman/Manders/overlap (shared with
    :func:`momia3.ops.measure.colocalization`), a Costes-style threshold
    bisection, object Jaccard/Dice and a 12-bin angular distribution.
``puncta``
    LoG spot detection reused from :func:`momia3.ops.characterize.find_puncta`:
    count, mean intensity, mean area and mean longitudinal/lateral /
    pole-distance coordinates.

All returned arrays are ``float32`` in ascending label order (the row
order of :func:`momia3.ops.measure.labels_to_cells`). Prerequisites of an
explicitly requested group raise ``ValueError`` instead of silently
returning NaNs.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
from numba import njit
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage.filters import threshold_otsu
from skimage.measure import regionprops_table
from skimage.morphology import skeletonize

from ._batch import FrameBatch
from .characterize import PunctaParams, find_puncta
from .measure import colocalization
from .rod_frame import RodFrame, rod_frames
from skimage.feature import graycomatrix

logger = logging.getLogger(__name__)

__all__ = [
    "FEATURE_GROUPS",
    "FeaturePlan",
    "HARALICK13_PROPS",
    "cell_features",
    "shape_features",
    "geometry_features",
    "intensity_features",
    "texture_features",
    "coloc_features",
    "puncta_features",
]

FEATURE_GROUPS: tuple[str, ...] = (
    "shape",
    "geometry",
    "intensity",
    "texture",
    "coloc",
    "puncta",
)
"""Every supported feature group."""

HARALICK13_PROPS: tuple[str, ...] = (
    "asm",
    "contrast",
    "correlation",
    "sum_variance",
    "inverse_difference_moment",
    "sum_average",
    "sum_entropy",
    "entropy",
    "difference_variance",
    "difference_entropy",
    "information_measure_1",
    "information_measure_2",
    "max_correlation",
)
"""The 13 Haralick GLCM texture features (§4.3)."""

_EPS = 1e-12


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeaturePlan:
    """Validated feature-group request.

    Parameters
    ----------
    groups : sequence of str
        Subset of :data:`FEATURE_GROUPS`; duplicates are removed while
        preserving the requested order.
    """

    groups: tuple[str, ...] = FEATURE_GROUPS

    def __post_init__(self) -> None:
        unknown = [g for g in self.groups if g not in FEATURE_GROUPS]
        if unknown:
            raise ValueError(
                f"unknown feature groups: {unknown}; expected a subset of "
                f"{FEATURE_GROUPS}"
            )
        object.__setattr__(
            self,
            "groups",
            tuple(dict.fromkeys(str(g) for g in self.groups)),
        )

    def has(self, group: str) -> bool:
        """True when ``group`` is part of the plan."""
        return group in self.groups

    def cache_key(self) -> tuple[str, ...]:
        """Stable, order-preserving cache token of the requested groups."""
        return self.groups


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _empty(n: int, names: Iterable[str]) -> dict[str, np.ndarray]:
    return {name: np.full(n, np.nan, dtype=np.float32) for name in names}


def _regionprops(batch: FrameBatch) -> dict[str, np.ndarray]:
    """One regionprops pass for the shared shape properties."""
    return regionprops_table(
        batch.labels_i,
        properties=[
            "label",
            "area",
            "area_convex",
            "area_filled",
            "equivalent_diameter_area",
            "euler_number",
            "feret_diameter_max",
            "perimeter",
            "centroid",
            "bbox",
        ],
    )


def _convex_perimeter(crop: np.ndarray) -> float:
    """Perimeter of the convex hull of a boolean crop (0 for degenerate)."""
    pts = np.column_stack(np.nonzero(crop)).astype(np.float64)
    if pts.shape[0] < 3:
        return 0.0
    try:
        return float(ConvexHull(pts).area)
    except Exception:  # collinear or duplicated points
        return 0.0


def _skeleton_topology(crop: np.ndarray) -> tuple[float, float, float]:
    """``(length, endpoints, branchpoints)`` of a boolean crop skeleton."""
    skel = skeletonize(crop)
    if not skel.any():
        return 0.0, 0.0, 0.0
    count = ndimage.convolve(
        skel.astype(np.int32),
        np.ones((3, 3), dtype=np.int32),
        mode="constant",
        cval=0,
    ) - skel.astype(np.int32)
    return float(skel.sum()), float(np.sum(skel & (count == 1))), float(
        np.sum(skel & (count > 2))
    )


def _factorial(value: int) -> float:
    return float(np.prod(np.arange(2, value + 1), dtype=np.float64)) if value > 0 else 1.0


_ZERNIKE_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (n, m) for n in range(10) for m in range(n % 2, n + 1, 2)
)[:30]
"""``(n, m)`` pairs of the first 30 rotation-invariant Zernike moments."""


def _zernike_terms(n: int, m: int) -> tuple[tuple[int, float], ...]:
    """``(power, coefficient)`` terms of the radial polynomial ``R_nm``."""
    return tuple(
        (
            n - 2 * k,
            (-1) ** k
            * _factorial(n - k)
            / (
                _factorial(k)
                * _factorial((n + m) // 2 - k)
                * _factorial((n - m) // 2 - k)
            ),
        )
        for k in range((n - m) // 2 + 1)
    )


_ZERNIKE_TERMS: tuple[tuple[tuple[int, float], ...], ...] = tuple(
    _zernike_terms(n, m) for n, m in _ZERNIKE_PAIRS
)
"""Radial coefficient tables, built once at import instead of per cell."""


def _zernike_moments(crop: np.ndarray) -> np.ndarray:
    """Magnitudes of the first 30 Zernike moments of a boolean crop."""
    out = np.zeros(len(_ZERNIKE_PAIRS), dtype=np.float64)
    ys, xs = np.nonzero(crop)
    if ys.size == 0:
        return out
    cy, cx = float(ys.mean()), float(xs.mean())
    scale = float(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)) / 2.0
    if scale <= _EPS:
        return out
    yy = (ys - cy) / scale
    xx = (xs - cx) / scale
    r = np.hypot(xx, yy)
    theta = np.arctan2(yy, xx)
    inside = r <= 1.0
    if inside.sum() == 0:
        return out
    r, theta = r[inside], theta[inside]
    rho = r / max(float(r.max()), _EPS)
    nt = inside.sum()
    for idx, (n, m) in enumerate(_ZERNIKE_PAIRS):
        radial = np.zeros_like(rho)
        for power, coeff in _ZERNIKE_TERMS[idx]:
            radial += coeff * rho ** power
        # ``cos``/``sin`` replace the complex exponential; the magnitude is
        # identical while avoiding complex temporaries.
        re = float(np.sum(radial * np.cos(m * theta)))
        im = float(np.sum(radial * np.sin(m * theta)))
        out[idx] = abs((n + 1) / np.pi * (re + 1j * im) / nt)
    return out


_GLCM_INDEX_CACHE: dict[int, tuple[np.ndarray, ...]] = {}


def _glcm_indices(levels: int) -> tuple[np.ndarray, ...]:
    """Cached coordinate/index arrays for a ``levels``-bin GLCM.

    Values are immutable for a given ``levels`` and shared across all calls,
    which keeps :func:`_haralick13` free of per-call allocations.
    """
    cached = _GLCM_INDEX_CACHE.get(levels)
    if cached is not None:
        return cached
    i, j = np.indices((levels, levels))
    i_f = i.astype(np.float64)
    j_f = j.astype(np.float64)
    diff = i_f - j_f
    cached = (
        i_f,
        j_f,
        i_f * j_f,
        diff * diff,
        1.0 / (1.0 + diff * diff),
        (i + j).ravel(),
        np.abs(i - j).ravel(),
        np.arange(2 * levels, dtype=np.float64),
        np.arange(levels, dtype=np.float64),
    )
    _GLCM_INDEX_CACHE[levels] = cached
    return cached


def _haralick13(glcm: np.ndarray) -> dict[str, float]:
    """The 13 Haralick features from one normalized GLCM."""
    p = np.asarray(glcm, dtype=np.float64)
    levels = p.shape[0]
    eps = _EPS
    (
        i,
        j,
        i_j,
        diff2,
        inv_diff2,
        plus_index,
        minus_index,
        k_plus,
        k_minus,
    ) = _glcm_indices(levels)
    px = p.sum(axis=1)
    py = p.sum(axis=0)
    mu_x = float(np.sum(i * p))
    mu_y = float(np.sum(j * p))
    sigma_x = float(np.sqrt(np.sum((i - mu_x) ** 2 * p)))
    sigma_y = float(np.sqrt(np.sum((j - mu_y) ** 2 * p)))
    flat = p.ravel()
    # C-order accumulation matches the former explicit loops exactly.
    pxplusy = np.bincount(plus_index, weights=flat, minlength=2 * levels)
    pxminusy = np.bincount(minus_index, weights=flat, minlength=levels)
    sum_average = float(np.sum(k_plus * pxplusy))
    sum_entropy = -float(
        np.sum(pxplusy[pxplusy > 0] * np.log2(pxplusy[pxplusy > 0] + eps))
    )
    sum_variance = float(np.sum((k_plus - sum_entropy) ** 2 * pxplusy))
    hx = -float(np.sum(px[px > 0] * np.log2(px[px > 0] + eps)))
    hy = -float(np.sum(py[py > 0] * np.log2(py[py > 0] + eps)))
    hxy = -float(np.sum(p[p > 0] * np.log2(p[p > 0] + eps)))
    pxy = px[:, None] * py[None, :]
    hxy1 = -float(np.sum(p * np.log2(pxy + eps)))
    hxy2 = -float(np.sum(pxy * np.log2(pxy + eps)))
    # Max correlation coefficient (Haralick 1973, eq. 3.14): second-largest
    # eigenvalue of the normalized GLCM product. The product is symmetric up
    # to rounding, so its symmetrized form is used with the cheaper
    # Hermitian solver.
    q = (p / np.maximum(px[:, None], eps)) @ (p / np.maximum(py[:, None], eps)).T
    eig = np.linalg.eigvalsh(0.5 * (q + q.T))
    max_corr = float(np.sqrt(np.maximum(eig[-2], 0.0))) if eig.size > 1 else 0.0
    diff_mean = float(np.sum(k_minus * pxminusy))
    diff_var = float(np.sum((k_minus - diff_mean) ** 2 * pxminusy))
    return {
        "asm": float(np.sum(p * p)),
        "contrast": float(np.sum(diff2 * p)),
        "correlation": float(
            (np.sum(i_j * p) - mu_x * mu_y) / (sigma_x * sigma_y + eps)
        ),
        "sum_variance": sum_variance,
        "inverse_difference_moment": float(np.sum(p * inv_diff2)),
        "sum_average": sum_average,
        "sum_entropy": sum_entropy,
        "entropy": hxy,
        "difference_variance": diff_var,
        "difference_entropy": -float(
            np.sum(pxminusy[pxminusy > 0] * np.log2(pxminusy[pxminusy > 0] + eps))
        ),
        "information_measure_1": float((hxy - hxy1) / max(hx, hy, eps)),
        "information_measure_2": float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * (hxy2 - hxy))))),
        "max_correlation": max_corr,
    }


# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------


def shape_features(
    batch: FrameBatch, *, include_zernike: bool = True
) -> dict[str, np.ndarray]:
    """Extended shape descriptors of every instance.

    Adds the Momia2/CellProfiler regionprops extensions and the shape
    topology/Zernike descriptors on top of the canonical scalar columns.
    """
    n = batch.n_labels
    props = _regionprops(batch)
    area = props["area"].astype(np.float64)
    feret = props["feret_diameter_max"].astype(np.float64)
    convex_area = props["area_convex"].astype(np.float64)
    perimeter = props["perimeter"].astype(np.float64)
    centroid_x = props["centroid-1"].astype(np.float64)
    centroid_y = props["centroid-0"].astype(np.float64)

    radius_min = np.empty(n, dtype=np.float64)
    radius_mean = np.empty(n, dtype=np.float64)
    radius_max = np.empty(n, dtype=np.float64)
    convex_perim = np.empty(n, dtype=np.float64)
    skel_len = np.empty(n, dtype=np.float64)
    skel_end = np.empty(n, dtype=np.float64)
    skel_branch = np.empty(n, dtype=np.float64)
    zernike = np.full((n, len(_ZERNIKE_PAIRS)), np.nan, dtype=np.float64)

    for i, label_id in enumerate(batch.label_ids):
        lid = int(label_id)
        xs, ys = batch.pixel_xy(lid)
        r = np.hypot(xs - centroid_x[i], ys - centroid_y[i])
        radius_min[i] = float(r.min())
        radius_mean[i] = float(r.mean())
        radius_max[i] = float(r.max())
        crop = batch.crop(lid) == lid
        convex_perim[i] = _convex_perimeter(crop)
        skel_len[i], skel_end[i], skel_branch[i] = _skeleton_topology(crop)
        if include_zernike:
            zernike[i] = _zernike_moments(crop)

    angularity = perimeter / np.maximum(convex_perim, _EPS)
    roundness = 4.0 * area / (np.pi * np.maximum(feret**2, _EPS))
    out: dict[str, np.ndarray] = {
        "feret_max": feret.astype(np.float32),
        # Minor Feret diameter approximated from the convex area as in
        # CellProfiler's FeretDiameterMin (2*A_convex / FeretMax).
        "feret_min": np.minimum(
            feret, 2.0 * convex_area / np.maximum(feret, _EPS)
        ).astype(np.float32),
        "radius_min": radius_min.astype(np.float32),
        "radius_mean": radius_mean.astype(np.float32),
        "radius_max": radius_max.astype(np.float32),
        "convex_area": convex_area.astype(np.float32),
        "convex_perimeter": convex_perim.astype(np.float32),
        "filled_area": props["area_filled"].astype(np.float32),
        "equivalent_diameter": props["equivalent_diameter_area"].astype(np.float32),
        "euler_number": props["euler_number"].astype(np.float32),
        "skeleton_length": skel_len.astype(np.float32),
        "skeleton_endpoints": skel_end.astype(np.float32),
        "skeleton_branchpoints": skel_branch.astype(np.float32),
        "angularity": angularity.astype(np.float32),
        "roundness": roundness.astype(np.float32),
    }
    for idx in range(zernike.shape[1]):
        out[f"zernike_{idx}"] = zernike[:, idx].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# geometry (rod frame)
# ---------------------------------------------------------------------------


def _geometry_scalars(frame: RodFrame, pixel_microns: float) -> dict[str, float]:
    width = np.asarray(frame.width_px, dtype=np.float64)
    if width.size == 0:
        return {
            "constriction_depth": float("nan"),
            "septum_width_um": float("nan"),
            "septum_position": float("nan"),
            "pole_width_um": float("nan"),
        }
    # Exclude the pole-cap taper (10% at each end) so the constriction is
    # an interior minimum rather than the cap narrowing.
    edge = max(1, width.size // 10)
    interior = width[edge:-edge] if width.size > 2 * edge else width
    median = float(np.median(interior)) if interior.size else float("nan")
    dip_idx = int(np.argmin(interior)) + (edge if width.size > 2 * edge else 0)
    septum_width = float(width[dip_idx])
    depth = (
        float((median - septum_width) / median) if median > _EPS else float("nan")
    )
    n = max(1, min(3, width.size // 8 or 1))
    pole_width = float(np.mean(np.concatenate([width[:n], width[-n:]])))
    position = float(dip_idx / max(width.size - 1, 1))
    return {
        "constriction_depth": depth,
        "septum_width_um": septum_width * pixel_microns,
        "septum_position": position,
        "pole_width_um": pole_width * pixel_microns,
    }


def geometry_features(
    cells_table: pa.Table,
    batch: FrameBatch,
    *,
    n_points: int = 32,
    pixel_microns: float | None = None,
    frames: Mapping[int, RodFrame] | None = None,
) -> dict[str, np.ndarray]:
    """Rod-frame geometry scalars aligned to ``batch.label_ids``.

    ``frames`` lets a caller pass rod frames it already built (as
    :func:`cell_features` does) instead of resampling the midline again.
    """
    n = batch.n_labels
    names = ("constriction_depth", "septum_width_um", "septum_position", "pole_width_um")
    out = _empty(n, names)
    pm = float(batch.pixel_microns if pixel_microns is None else pixel_microns)
    if frames is None:
        frames = rod_frames(cells_table, n_points=n_points, pixel_microns=pm)
    by_label = {frame.label: frame for frame in frames.values()}
    for i, label_id in enumerate(batch.label_ids):
        frame = by_label.get(int(label_id))
        if frame is None:
            continue
        for name, value in _geometry_scalars(frame, pm).items():
            out[name][i] = value
    return out


# ---------------------------------------------------------------------------
# intensity
# ---------------------------------------------------------------------------


def _channel_threshold(batch: FrameBatch, image: np.ndarray) -> float:
    """Otsu threshold of the foreground pixels (0 when undefined)."""
    foreground = np.asarray(image)[batch.labels > 0]
    if foreground.size < 2 or float(foreground.max() - foreground.min()) < _EPS:
        return 0.0
    try:
        return float(threshold_otsu(foreground))
    except ValueError:
        return 0.0


def intensity_features(
    batch: FrameBatch,
    channels: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Distribution and background features per channel."""
    n = batch.n_labels
    out: dict[str, np.ndarray] = {}
    labels = batch.labels
    background = {
        name: float(np.median(np.asarray(image)[labels == 0]))
        if np.any(labels == 0)
        else 0.0
        for name, image in channels.items()
    }
    for name, raw in channels.items():
        image = np.asarray(raw, dtype=np.float64)
        threshold = batch.cache.memoize(
            ("otsu", name), lambda image=image: _channel_threshold(batch, image)
        )
        bg = background[name]
        mad = np.full(n, np.nan, dtype=np.float64)
        displacement = np.full(n, np.nan, dtype=np.float64)
        pct_above = np.full(n, np.nan, dtype=np.float64)
        local_bg = np.full(n, np.nan, dtype=np.float64)
        snr = np.full(n, np.nan, dtype=np.float64)
        for i, label_id in enumerate(batch.label_ids):
            lid = int(label_id)
            values = batch.values(lid, image)
            if values.size == 0:
                continue
            med = float(np.median(values))
            mad[i] = float(np.median(np.abs(values - med)))
            pct_above[i] = float(np.mean(values > threshold))
            xs, ys = batch.pixel_xy(lid)
            total = float(values.sum())
            if total > _EPS:
                wx = float(np.sum(values * xs) / total)
                wy = float(np.sum(values * ys) / total)
                cx, cy = float(xs.mean()), float(ys.mean())
                displacement[i] = float(np.hypot(wx - cx, wy - cy))
            crop = batch.crop(lid, pad=2)
            mask = crop == lid
            ring = ndimage.binary_dilation(mask, iterations=2) & ~mask
            if ring.any():
                y0, x0 = batch.slices[lid - 1][0].start - 2, batch.slices[lid - 1][1].start - 2
                y0, x0 = max(y0, 0), max(x0, 0)
                ring_values = image[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]][ring]
                if ring_values.size:
                    local_bg[i] = float(np.median(ring_values))
            if np.isfinite(local_bg[i]):
                std = float(values.std())
                snr[i] = (
                    float((values.mean() - local_bg[i]) / std) if std > _EPS else np.nan
                )
        out[f"{name}_mad"] = mad.astype(np.float32)
        out[f"{name}_mass_displacement"] = displacement.astype(np.float32)
        out[f"{name}_pct_above"] = pct_above.astype(np.float32)
        out[f"{name}_local_background"] = local_bg.astype(np.float32)
        out[f"{name}_snr"] = snr.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# texture
# ---------------------------------------------------------------------------


def texture_features(
    batch: FrameBatch,
    channels: Mapping[str, np.ndarray],
    *,
    levels: int = 64,
    distances: tuple[int, ...] = (1,),
    angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> dict[str, np.ndarray]:
    """13 Haralick features + granularity per channel and cell."""
    n = batch.n_labels
    out: dict[str, np.ndarray] = {}
    for name, raw in channels.items():
        image = np.asarray(raw, dtype=np.float64)
        values = {
            f"{name}_haralick_{prop}": np.full(n, np.nan, dtype=np.float64)
            for prop in HARALICK13_PROPS
        }
        granularity = np.full(n, np.nan, dtype=np.float64)
        for i, label_id in enumerate(batch.label_ids):
            lid = int(label_id)
            y0, x0 = batch.slices[lid - 1][0].start, batch.slices[lid - 1][1].start
            crop_mask = batch.crop(lid)
            crop = image[
                y0 : y0 + crop_mask.shape[0], x0 : x0 + crop_mask.shape[1]
            ]
            cmin, cmax = float(crop.min()), float(crop.max())
            if cmax - cmin < _EPS:
                quantized = np.zeros(crop.shape, dtype=np.uint8)
            else:
                quantized = np.floor(
                    (crop - cmin) / (cmax - cmin) * (levels - 1)
                ).astype(np.uint8)
            if min(quantized.shape) > max(distances):
                glcm = graycomatrix(
                    quantized,
                    distances=list(distances),
                    angles=list(angles),
                    levels=levels,
                    symmetric=True,
                    normed=True,
                )
                # Average the per-angle marginal statistics by summing the
                # individual angle matrices into one normalized GLCM.
                p = glcm.mean(axis=3)[:, :, 0]
                p = p / max(float(p.sum()), _EPS)
                for prop, value in _haralick13(p).items():
                    values[f"{name}_haralick_{prop}"][i] = value
            if crop.size > 1:
                dy = np.abs(np.diff(crop, axis=0))
                dx = np.abs(np.diff(crop, axis=1))
                total = float(np.sum(dy**2) + np.sum(dx**2))
                granularity[i] = total / max(float(dy.size + dx.size), 1.0)
        for key, arr in values.items():
            out[key] = arr.astype(np.float32)
        out[f"{name}_granularity"] = granularity.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# colocalization
# ---------------------------------------------------------------------------


def _pair_key(a: str, b: str, name: str) -> str:
    return f"coloc_{a}_{b}_{name}"


def coloc_features(
    batch: FrameBatch,
    channels: Mapping[str, np.ndarray],
    *,
    n_angle_bins: int = 12,
) -> dict[str, np.ndarray]:
    """Pairwise colocalization extensions (Costes, objects, angular)."""
    if len(channels) < 2:
        raise ValueError("coloc features need at least two channels")
    n = batch.n_labels
    out: dict[str, np.ndarray] = {}
    labels = batch.labels
    thresholds = {
        name: batch.cache.memoize(
            ("otsu", name),
            lambda image=np.asarray(image, dtype=np.float64): _channel_threshold(batch, image),
        )
        for name, image in channels.items()
    }
    for a_name, b_name in itertools.combinations(channels, 2):
        a = np.asarray(channels[a_name], dtype=np.float64)
        b = np.asarray(channels[b_name], dtype=np.float64)
        base = colocalization(labels, a, b, a_name, b_name, batch=batch)
        for key, arr in base.items():
            out[key] = arr
        keys = [
            _pair_key(a_name, b_name, "costes_tau_a"),
            _pair_key(a_name, b_name, "costes_tau_b"),
            _pair_key(a_name, b_name, "costes_pearson"),
            _pair_key(a_name, b_name, "object_jaccard"),
            _pair_key(a_name, b_name, "object_dice"),
        ]
        keys += [
            _pair_key(a_name, b_name, f"angle_{k}") for k in range(n_angle_bins)
        ]
        for key in keys:
            out[key] = np.full(n, np.nan, dtype=np.float32)
        ta, tb = thresholds[a_name], thresholds[b_name]
        for i, label_id in enumerate(batch.label_ids):
            lid = int(label_id)
            va = batch.values(lid, a)
            vb = batch.values(lid, b)
            if va.size == 0:
                continue
            tau_a, tau_b, p = _costes_thresholds(va, vb)
            out[_pair_key(a_name, b_name, "costes_tau_a")][i] = tau_a
            out[_pair_key(a_name, b_name, "costes_tau_b")][i] = tau_b
            out[_pair_key(a_name, b_name, "costes_pearson")][i] = p
            ma = va >= ta
            mb = vb >= tb
            union = int(np.sum(ma | mb))
            inter = int(np.sum(ma & mb))
            out[_pair_key(a_name, b_name, "object_jaccard")][i] = (
                inter / union if union else np.nan
            )
            out[_pair_key(a_name, b_name, "object_dice")][i] = (
                2 * inter / max(int(ma.sum()) + int(mb.sum()), 1)
            )
            xs, ys = batch.pixel_xy(lid)
            bright = va >= ta
            if bright.any():
                cx, cy = float(xs.mean()), float(ys.mean())
                ang = np.mod(np.arctan2(ys[bright] - cy, xs[bright] - cx), np.pi)
                hist, _ = np.histogram(ang, bins=n_angle_bins, range=(0.0, np.pi))
                hist = hist.astype(np.float64) / max(hist.sum(), 1.0)
                for k in range(n_angle_bins):
                    out[_pair_key(a_name, b_name, f"angle_{k}")][i] = hist[k]
    return out


@njit(cache=True)
def _costes_kernel(
    va: np.ndarray,
    vb: np.ndarray,
    taus: np.ndarray,
    vmin_a: float,
    vmin_b: float,
    span_a: float,
    span_b: float,
    eps: float,
) -> tuple[float, float]:
    """Compiled Costes bisection: ``(best_tau, best_pearson)``."""
    n = va.shape[0]
    best_tau = 0.0
    best_p = np.nan
    for s in range(taus.shape[0]):
        tau = taus[s]
        ta = vmin_a + tau * span_a
        tb = vmin_b + tau * span_b
        count = 0
        for k in range(n):
            if va[k] >= ta and vb[k] >= tb:
                count += 1
        if count < 3:
            break
        sx = 0.0
        sy = 0.0
        for k in range(n):
            if va[k] >= ta and vb[k] >= tb:
                sx += va[k]
                sy += vb[k]
        mx = sx / count
        my = sy / count
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        for k in range(n):
            if va[k] >= ta and vb[k] >= tb:
                dx = va[k] - mx
                dy = vb[k] - my
                sxx += dx * dx
                syy += dy * dy
                sxy += dx * dy
        # Population standard deviation, matching ``numpy.ndarray.std``.
        if math.sqrt(sxx / count) <= eps or math.sqrt(syy / count) <= eps:
            continue
        p = sxy / math.sqrt(sxx * syy)
        if math.isfinite(p) and p > 0.0:
            best_tau = tau
            best_p = p
        elif math.isfinite(best_p):
            break
    return best_tau, best_p


def _costes_thresholds(
    va: np.ndarray, vb: np.ndarray, n_steps: int = 20
) -> tuple[float, float, float]:
    """Bisection-style Costes thresholds and masked Pearson coefficient."""
    va = np.asarray(va, dtype=np.float64)
    vb = np.asarray(vb, dtype=np.float64)
    if va.size < 2:
        return float("nan"), float("nan"), float("nan")
    vmin_a = float(va.min())
    vmin_b = float(vb.min())
    span_a = float(va.max() - vmin_a)
    span_b = float(vb.max() - vmin_b)
    if span_a <= _EPS or span_b <= _EPS:
        return 0.0, 0.0, float("nan")
    taus = np.linspace(0.0, 0.9, n_steps)
    best_tau, best_p = _costes_kernel(
        np.ascontiguousarray(va),
        np.ascontiguousarray(vb),
        taus,
        vmin_a,
        vmin_b,
        span_a,
        span_b,
        _EPS,
    )
    return float(best_tau), float(best_tau), float(best_p)


# ---------------------------------------------------------------------------
# puncta
# ---------------------------------------------------------------------------


def puncta_features(
    batch: FrameBatch,
    channels: Mapping[str, np.ndarray],
    *,
    frames: Mapping[int, RodFrame] | None = None,
    params: PunctaParams | None = None,
) -> dict[str, np.ndarray]:
    """LoG puncta count/intensity/area and (optional) rod coordinates."""
    n = batch.n_labels
    params = params or PunctaParams()
    out: dict[str, np.ndarray] = {}
    by_label = {frame.label: frame for frame in (frames or {}).values()}
    for name, raw in channels.items():
        image = np.asarray(raw, dtype=np.float64)
        count = np.zeros(n, dtype=np.float64)
        mean_int = np.full(n, np.nan, dtype=np.float64)
        area = np.full(n, np.nan, dtype=np.float64)
        ld_l = np.full(n, np.nan, dtype=np.float64)
        ld_d = np.full(n, np.nan, dtype=np.float64)
        pole_sep = np.full(n, np.nan, dtype=np.float64)
        spot_area = float(np.pi * params.log_sigma**2)
        for i, label_id in enumerate(batch.label_ids):
            lid = int(label_id)
            sl = batch.slices[lid - 1]
            y0, x0 = sl[0].start, sl[1].start
            crop_mask = batch.crop(lid)
            crop = image[y0 : y0 + crop_mask.shape[0], x0 : x0 + crop_mask.shape[1]]
            spots = find_puncta(crop, crop_mask, params)
            spots = spots[np.isfinite(spots).all(axis=1)]
            count[i] = float(spots.shape[0])
            if spots.shape[0] == 0:
                continue
            ys = np.clip(np.round(spots[:, 0]).astype(int), 0, crop.shape[0] - 1)
            xs = np.clip(np.round(spots[:, 1]).astype(int), 0, crop.shape[1] - 1)
            mean_int[i] = float(np.mean(crop[ys, xs]))
            area[i] = spot_area
            frame = by_label.get(lid)
            if frame is not None:
                gx = np.clip(spots[:, 1] + x0, 0, batch.labels.shape[1] - 1)
                gy = np.clip(spots[:, 0] + y0, 0, batch.labels.shape[0] - 1)
                proj = frame.project(gx, gy)
                ld_l[i] = float(np.mean(proj["longitudinal"]))
                ld_d[i] = float(np.mean(proj["radial"]))
                pole_sep[i] = float(
                    np.mean(
                        np.minimum(proj["pole1_distance"], proj["pole2_distance"])
                    )
                )
        out[f"puncta_{name}_count"] = count.astype(np.float32)
        out[f"puncta_{name}_intensity_mean"] = mean_int.astype(np.float32)
        out[f"puncta_{name}_area_mean"] = area.astype(np.float32)
        out[f"puncta_{name}_ld_l_mean"] = ld_l.astype(np.float32)
        out[f"puncta_{name}_ld_d_mean"] = ld_d.astype(np.float32)
        out[f"puncta_{name}_pole_separation"] = pole_sep.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# aggregation entry
# ---------------------------------------------------------------------------


def cell_features(
    labels_2d: np.ndarray,
    intensity: Mapping[str, np.ndarray] | None = None,
    cells_table: pa.Table | None = None,
    *,
    groups: Sequence[str] | None = None,
    pixel_microns: float = 1.0,
    n_midline_points: int = 32,
    texture_levels: int = 64,
    puncta_params: PunctaParams | None = None,
) -> dict[str, np.ndarray]:
    """Compute every requested feature group on one frame.

    Parameters
    ----------
    labels_2d : numpy.ndarray
        2D instance label image (0 = background).
    intensity : mapping of str to numpy.ndarray, optional
        Channel name -> 2D intensity image. ``intensity``, ``texture``,
        ``coloc`` and ``puncta`` groups require at least one channel;
        ``coloc`` requires two.
    cells_table : pyarrow.Table, optional
        Cells table carrying ``midline``/``width_profile``; required by
        ``geometry`` and the rod coordinates of ``puncta``.
    groups : sequence of str, optional
        Feature groups to compute; ``None`` means all supported groups.
        Explicitly requested groups whose prerequisites are missing raise
        ``ValueError``.
    pixel_microns : float
        Physical pixel size used for geometry widths.
    n_midline_points : int
        Rod-frame resampling count for shared geometry/rod coordinates.
    texture_levels : int
        GLCM quantization levels.
    puncta_params : PunctaParams, optional
        LoG spot-detection parameters.

    Returns
    -------
    dict of str to numpy.ndarray
        Feature columns as ``float32`` arrays in ascending label order.
    """
    labels = np.asarray(labels_2d)
    if labels.ndim != 2:
        raise ValueError(f"cell_features expects a 2D label image, got {labels.shape}")
    channels = {
        name: np.asarray(image) for name, image in dict(intensity or {}).items()
    }
    for name, image in channels.items():
        if image.shape != labels.shape:
            raise ValueError(
                f"channel {name!r} shape {image.shape} does not match "
                f"labels shape {labels.shape}"
            )
    plan = FeaturePlan(tuple(groups) if groups is not None else FEATURE_GROUPS)
    if plan.has("geometry") and cells_table is None:
        raise ValueError("geometry features require a cells_table")
    for group in ("intensity", "texture", "coloc", "puncta"):
        if plan.has(group) and not channels:
            raise ValueError(f"{group} features require at least one channel")
    if plan.has("coloc") and len(channels) < 2:
        raise ValueError("coloc features require at least two channels")

    batch = FrameBatch(labels, channels, pixel_microns=pixel_microns)
    if batch.n_labels == 0:
        return {}

    result: dict[str, np.ndarray] = {}
    rod: dict[int, RodFrame] = {}
    if plan.has("geometry") or plan.has("puncta"):
        if cells_table is not None:
            try:
                rod = rod_frames(
                    cells_table,
                    n_points=max(int(n_midline_points), 4),
                    pixel_microns=float(pixel_microns),
                )
            except Exception as exc:  # no usable rod geometry
                logger.debug("cell_features: rod frames unavailable (%s)", exc)
                rod = {}

    if plan.has("shape"):
        result.update(shape_features(batch))
    if plan.has("geometry"):
        # Reuse the rod frames built above when they were resampled with the
        # same number of points; otherwise let geometry_features rebuild.
        shared = rod if rod and int(n_midline_points) >= 4 else None
        result.update(
            geometry_features(
                cells_table,
                batch,
                n_points=n_midline_points,
                pixel_microns=pixel_microns,
                frames=shared,
            )
        )
    if plan.has("intensity"):
        result.update(intensity_features(batch, channels))
    if plan.has("texture"):
        result.update(texture_features(batch, channels, levels=int(texture_levels)))
    if plan.has("coloc"):
        result.update(coloc_features(batch, channels))
    if plan.has("puncta"):
        result.update(
            puncta_features(batch, channels, frames=rod, params=puncta_params)
        )
    logger.debug(
        "cell_features: %d labels, %d columns, groups=%s",
        batch.n_labels,
        len(result),
        plan.groups,
    )
    return result
