"""Measurement operators (ARCHITECTURE.md §4.3).

- :func:`labels_to_cells` — one :func:`skimage.measure.regionprops_table`
  call per frame into a ``cells.parquet``-conforming Arrow table
  (morphology + per-channel intensity clusters); geometry columns
  (outline/midline/width_profile/coords) are left null for the
  outline/midline ops to fill.
- :func:`intensity_stats` — vectorized per-channel intensity statistics.
- :func:`texture_haralick` — per-cell Haralick texture features
  (CellProfiler ``MeasureTexture`` analogue).
- :func:`texture_lbp` — per-cell uniform-LBP histogram features.
- :func:`add_features` — append per-cell feature columns to a cells table.

All per-cell arrays are ordered by ascending label id, matching the row
order of :func:`labels_to_cells`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pyarrow as pa
from scipy import ndimage, stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops_table

from .. import keys as _keys
from .. import schema as _schema
from ._batch import FrameBatch

logger = logging.getLogger(__name__)

__all__ = [
    "labels_to_cells",
    "intensity_stats",
    "colocalization",
    "INTENSITY_MOMENT_STATS",
    "intensity_moments",
    "texture_haralick",
    "texture_lbp",
    "add_features",
    "HARALICK_PROPS",
]

_EPS = 1e-12

# regionprops properties needed for the schema scalar morphology columns
# plus the raw weighted moments used by the weighted-orientation port.
_REGIONPROPS: tuple[str, ...] = (
    "label",
    "bbox",
    "centroid",
    "area",
    "perimeter",
    "major_axis_length",
    "minor_axis_length",
    "eccentricity",
    "solidity",
    "extent",
    "moments_hu",
    "weighted_moments",
)

HARALICK_PROPS: tuple[str, ...] = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
)
"""Haralick properties computed by :func:`texture_haralick`."""


# ---------------------------------------------------------------------------
# labels -> cells table
# ---------------------------------------------------------------------------


def _weighted_orientation(props: dict[str, np.ndarray]) -> np.ndarray:
    """Weighted orientation from raw weighted moments (momia2 port).

    Direct port of ``momia2/core/patch.py::get_regionprop`` (L868-898):
    central moments are derived from the raw weighted moments as
    ``mu20 = M20 - M10**2 / M00`` etc., then
    ``theta = 0.5 * arctan2(2*mu11, mu20 - mu02)``. Angles follow the
    skimage row/column axis convention (momia2's ``M_ji`` mapping).
    """
    m00 = props["weighted_moments-0-0"]
    m10 = props["weighted_moments-1-0"]
    m01 = props["weighted_moments-0-1"]
    m20 = props["weighted_moments-2-0"]
    m02 = props["weighted_moments-0-2"]
    m11 = props["weighted_moments-1-1"]
    m00_safe = np.where(m00 == 0, _EPS, m00)
    mu20 = m20 - (m10**2 / m00_safe)
    mu02 = m02 - (m01**2 / m00_safe)
    mu11 = m11 - (m10 * m01 / m00_safe)
    return 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)


def labels_to_cells(
    labels_2d: np.ndarray,
    fov_id: int,
    frame: int,
    intensity: dict[str, np.ndarray] | None = None,
    pixel_microns: float = 1.0,
) -> pa.Table:
    """Measure one frame's instance labels into a canonical cells table.

    A single :func:`skimage.measure.regionprops_table` call computes all
    morphology columns; per-channel intensity clusters are appended with
    :func:`intensity_stats`. Geometry columns (``outline``, ``midline``,
    ``width_profile``, ``coords``) and midline-derived scalar columns
    (``curv_*``, ``length_um``, ``width_*_um``, ``sinuosity``,
    ``branched``, ``pole*``, ``septum_pos``) are left null — the
    outline/midline ops fill them.

    Parameters
    ----------
    labels_2d : numpy.ndarray
        2D instance label image; 0 is background.
    fov_id, frame : int
        Coordinates of the frame, encoded into ``cell_key`` via
        :func:`momia3.keys.encode_keys`.
    intensity : dict of str to numpy.ndarray, optional
        Channel name -> 2D intensity image (same shape as ``labels_2d``).
        Each channel contributes a ``{ch}_mean/max/min/std/p25/p75/
        integrated`` cluster. The first channel also weights the weighted
        moments; without channels the binary mask is used (momia2's
        fallback), making ``weighted_orientation`` equal to the
        mask orientation.
    pixel_microns : float
        Physical pixel size in microns. Recorded for downstream µm
        conversions (``length_um``/``width_*_um`` are computed by the
        midline op); scalar columns here stay in pixels per the schema
        metadata. Must be positive.

    Returns
    -------
    pyarrow.Table
        Cells table validated against :func:`momia3.schema.validate_cells`
        (one row per instance, ascending label order). ``weighted_orientation``
        is appended as an extra float32 column (the schema has no
        orientation column; extras pass validation unchanged).
    """
    labels = np.asarray(labels_2d)
    if labels.ndim != 2:
        raise ValueError(f"labels_to_cells expects a 2D label image, got {labels.shape}")
    if pixel_microns <= 0:
        raise ValueError(f"pixel_microns must be positive, got {pixel_microns}")
    channels = list(intensity.keys()) if intensity else []
    if intensity:
        for name, img in intensity.items():
            if np.asarray(img).shape != labels.shape:
                raise ValueError(
                    f"intensity channel {name!r} shape {np.asarray(img).shape} "
                    f"does not match labels shape {labels.shape}"
                )

    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids > 0]
    if label_ids.size == 0:
        logger.warning("labels_to_cells: no instances in frame %d", frame)
        return _schema.empty_cells_table(channels)

    labels_i = labels.astype(np.int32, copy=False)
    if intensity:
        reference = np.asarray(intensity[channels[0]], dtype=np.float32)
    else:
        reference = (labels_i > 0).astype(np.float32)
    props: dict[str, np.ndarray] = regionprops_table(
        labels_i, intensity_image=reference, properties=list(_REGIONPROPS)
    )
    n = int(props["label"].size)
    height, width = labels.shape

    area = props["area"].astype(np.float64)
    perimeter = props["perimeter"].astype(np.float64)
    major = props["major_axis_length"].astype(np.float64)
    minor = props["minor_axis_length"].astype(np.float64)
    # momia2 convention (CellProfiler-style): aspect_ratio = minor/major in (0, 1].
    aspect_ratio = minor / np.maximum(major, _EPS)
    circularity = 4.0 * np.pi * area / np.maximum(perimeter**2, _EPS)
    compactness = perimeter**2 / (4.0 * np.pi * np.maximum(area, _EPS))
    touching_edge = (
        (props["bbox-0"] <= 0)
        | (props["bbox-1"] <= 0)
        | (props["bbox-2"] >= height)
        | (props["bbox-3"] >= width)
    )
    cell_keys = _keys.encode_keys(
        int(fov_id), int(frame), label_ids.astype(np.int64)
    )
    weighted_orientation = _weighted_orientation(props).astype(np.float32)

    columns: dict[str, Any] = {
        "cell_key": cell_keys,
        "fov_id": np.full(n, int(fov_id), dtype=np.uint32),
        "frame": np.full(n, int(frame), dtype=np.uint32),
        "label": label_ids.astype(np.uint32),
        "track_id": [None] * n,
        "parent_track": [None] * n,
        "touching_edge": touching_edge.tolist(),
        "include": [True] * n,
        "bbox": [
            {
                "x0": int(props["bbox-1"][i]),
                "y0": int(props["bbox-0"][i]),
                "x1": int(props["bbox-3"][i]),
                "y1": int(props["bbox-2"][i]),
            }
            for i in range(n)
        ],
        "centroid": [
            {"x": float(props["centroid-1"][i]), "y": float(props["centroid-0"][i])}
            for i in range(n)
        ],
        "outline": [None] * n,
        "midline": [None] * n,
        "width_profile": [None] * n,
        "coords": [None] * n,
        "area": area.astype(np.float32),
        "perimeter": perimeter.astype(np.float32),
        "major_axis_length": major.astype(np.float32),
        "minor_axis_length": minor.astype(np.float32),
        "aspect_ratio": aspect_ratio.astype(np.float32),
        "eccentricity": props["eccentricity"].astype(np.float32),
        "solidity": props["solidity"].astype(np.float32),
        "extent": props["extent"].astype(np.float32),
        "circularity": circularity.astype(np.float32),
        "compactness": compactness.astype(np.float32),
        "embedding": [None] * n,
        "annotation": [None] * n,
    }
    for hu_idx in range(7):
        columns[f"hu_{hu_idx}"] = props[f"moments_hu-{hu_idx}"].astype(np.float32)
    # Midline/curvature-derived scalar columns stay null (filled by ops/midline.py).
    for name in _schema.SCALAR_MORPHOLOGY_COLUMNS:
        columns.setdefault(name, [None] * n)

    # Per-channel intensity clusters, aligned to ascending label order.
    if intensity:
        for ch in channels:
            columns.update(intensity_stats(labels_i, intensity[ch], ch))
            columns[f"puncta_{ch}"] = [None] * n  # filled by characterize ops

    sch = _schema.cells_schema(channels)
    table = pa.Table.from_arrays(
        [pa.array(columns[field.name], type=field.type) for field in sch],
        schema=sch,
    )
    table = _schema.validate_cells(table, channels)
    # Extra derived column (no schema slot; passes validation unchanged).
    table = table.append_column(
        "weighted_orientation", pa.array(weighted_orientation, type=pa.float32())
    )
    logger.debug(
        "labels_to_cells: fov=%d frame=%d instances=%d channels=%s",
        fov_id,
        frame,
        n,
        channels,
    )
    return table


# ---------------------------------------------------------------------------
# Intensity statistics (vectorized)
# ---------------------------------------------------------------------------


def intensity_stats(
    labels_2d: np.ndarray, channel_image: np.ndarray, prefix: str
) -> dict[str, np.ndarray]:
    """Per-cell intensity statistics of one channel (vectorized).

    Uses scipy.ndimage.find_objects bounding boxes and computes all
    statistics on the cropped instance pixels; this avoids scanning the full
    frame once per label (the previous percentile loop did exactly that).

    Parameters
    ----------
    labels_2d : numpy.ndarray
        2D instance label image; 0 is background.
    channel_image : numpy.ndarray
        2D intensity image, same shape as labels_2d.
    prefix : str
        Channel name; output keys are prefix_mean, prefix_max, prefix_min,
        prefix_std, prefix_p25, prefix_p75, prefix_integrated.

    Returns
    -------
    dict of str to numpy.ndarray
        Float32 arrays of length n_instances, ascending label order.
    """
    labels = np.asarray(labels_2d)
    image = np.asarray(channel_image, dtype=np.float64)
    if labels.shape != image.shape:
        raise ValueError(
            f"labels shape {labels.shape} does not match image shape {image.shape}"
        )
    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids > 0]
    stats = {
        "mean": np.empty(label_ids.size, dtype=np.float64),
        "max": np.empty(label_ids.size, dtype=np.float64),
        "min": np.empty(label_ids.size, dtype=np.float64),
        "std": np.empty(label_ids.size, dtype=np.float64),
        "p25": np.empty(label_ids.size, dtype=np.float64),
        "p75": np.empty(label_ids.size, dtype=np.float64),
        "integrated": np.empty(label_ids.size, dtype=np.float64),
    }
    if label_ids.size:
        slices = ndimage.find_objects(labels)
        for i, label_id in enumerate(label_ids):
            idx = int(label_id) - 1
            sl = slices[idx] if idx < len(slices) else None
            if sl is None:
                for key in ("mean", "max", "min", "std", "p25", "p75"):
                    stats[key][i] = np.nan
                stats["integrated"][i] = 0.0
                continue
            sub_labels = labels[sl]
            values = image[sl][sub_labels == label_id]
            if values.size == 0:
                for key in ("mean", "max", "min", "std", "p25", "p75"):
                    stats[key][i] = np.nan
                stats["integrated"][i] = 0.0
                continue
            stats["mean"][i] = values.mean()
            stats["max"][i] = values.max()
            stats["min"][i] = values.min()
            stats["std"][i] = values.std()
            p25, p75 = np.percentile(values, [25.0, 75.0])
            stats["p25"][i] = p25
            stats["p75"][i] = p75
            stats["integrated"][i] = values.sum()

    return {
        f"{prefix}_mean": stats["mean"].astype(np.float32),
        f"{prefix}_max": stats["max"].astype(np.float32),
        f"{prefix}_min": stats["min"].astype(np.float32),
        f"{prefix}_std": stats["std"].astype(np.float32),
        f"{prefix}_p25": stats["p25"].astype(np.float32),
        f"{prefix}_p75": stats["p75"].astype(np.float32),
        f"{prefix}_integrated": stats["integrated"].astype(np.float32),
    }


INTENSITY_MOMENT_STATS: tuple[str, ...] = (
    "median",
    "cv",
    "skewness",
    "kurtosis",
    "entropy",
)
"""Distribution moments computed by intensity_moments.

entropy is the Shannon entropy (bits) of a 64-bin intensity histogram of
the cell pixels. cv is std / mean; undefined values are NaN rather than
silently dropped.
"""


def _intensity_moment(pixels: np.ndarray, stat: str) -> float:
    """One distribution moment of a 1-D pixel vector (NaN when undefined)."""
    values = np.asarray(pixels, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    mean = float(values.mean())
    std = float(values.std())
    if stat == "median":
        return float(np.median(values))
    if stat == "cv":
        return float(std / mean) if abs(mean) > _EPS else float("nan")
    if stat == "skewness":
        return float(stats.skew(values)) if values.size > 2 and std > _EPS else float("nan")
    if stat == "kurtosis":
        return float(stats.kurtosis(values)) if values.size > 3 and std > _EPS else float("nan")
    if stat == "entropy":
        counts, _ = np.histogram(values, bins=64)
        p = counts[counts > 0].astype(np.float64)
        p /= p.sum()
        return float(-np.sum(p * np.log2(p)))
    raise ValueError(f"unknown intensity moment {stat!r}")


def intensity_moments(
    labels_2d: np.ndarray,
    channel_image: np.ndarray,
    prefix: str = "ch0",
    stats: tuple[str, ...] = INTENSITY_MOMENT_STATS,
) -> dict[str, np.ndarray]:
    """Per-cell distribution moments of one channel.

    Extends the canonical intensity_stats cluster with the moments used by
    the legacy momia2 profiling workflows:

    - median: median pixel intensity;
    - cv: coefficient of variation std / mean;
    - skewness: scipy stats.skew (biased);
    - kurtosis: scipy stats.kurtosis (Fisher=True);
    - entropy: Shannon entropy (bits) of a 64-bin histogram.

    Parameters
    ----------
    labels_2d : numpy.ndarray
        2D instance label image; 0 is background.
    channel_image : numpy.ndarray
        Intensity image with the same shape as labels_2d.
    prefix : str
        Output column prefix (normally the channel name).
    stats : tuple of str
        Subset of INTENSITY_MOMENT_STATS to compute. Unknown names raise
        ValueError.

    Returns
    -------
    dict of str to numpy.ndarray
        {prefix}_{stat} -> float32 array in ascending label order. Empty
        label images return zero-length arrays. Undefined moments are NaN.
    """
    labels = np.asarray(labels_2d)
    image = np.asarray(channel_image, dtype=np.float64)
    if labels.shape != image.shape:
        raise ValueError(
            f"labels shape {labels.shape} does not match image shape {image.shape}"
        )
    unknown = [s for s in stats if s not in INTENSITY_MOMENT_STATS]
    if unknown:
        raise ValueError(f"unknown intensity moments: {unknown}")
    batch = FrameBatch(labels)
    out = {
        f"{prefix}_{stat}": np.empty(batch.n_labels, dtype=np.float32) for stat in stats
    }
    if batch.n_labels == 0:
        return out
    for i, label_id in enumerate(batch.label_ids):
        pixels = batch.values(int(label_id), image)
        for stat in stats:
            out[f"{prefix}_{stat}"][i] = _intensity_moment(pixels, stat)
    return out


# ---------------------------------------------------------------------------
# Per-cell colocalization
# ---------------------------------------------------------------------------


def colocalization(
    labels_2d,
    image_a,
    image_b,
    prefix_a="A",
    prefix_b="B",
    threshold_a=0.0,
    threshold_b=0.0,
    *,
    batch: FrameBatch | None = None,
):
    """Per-cell Pearson/Spearman/Manders/overlap colocalization.

    Pearson, Manders A/B and overlap are computed from per-label pixel
    sums (:class:`~momia3.ops._batch.FrameBatch`, one label sort per
    frame); Spearman ranks are computed per cell with
    scipy.stats.rankdata. Pearson/Manders/overlap return NaN for
    degenerate cells (empty, zero variance, zero denominator). Manders
    coefficients are thresholded independently for each channel.

    Parameters
    ----------
    labels_2d:
        Instance label image, 0 is background.
    image_a, image_b:
        Two intensity images of the same shape.
    prefix_a, prefix_b:
        Output column prefixes.
    threshold_a, threshold_b:
        Channel thresholds for the Manders denominators.
    batch:
        Optional pre-built :class:`~momia3.ops._batch.FrameBatch` over
        ``labels_2d``. Callers that already hold one (e.g. feature groups)
        pass it in to avoid re-sorting the frame per channel pair; the
        label set and pixel index map must match ``labels_2d``.

    Returns
    -------
    dict[str, np.ndarray]
        Arrays in ascending label order: pearson, spearman, manders_a,
        manders_b, overlap.
    """
    from scipy.stats import rankdata

    labels = np.asarray(labels_2d)
    a = np.asarray(image_a, dtype=np.float64)
    b = np.asarray(image_b, dtype=np.float64)
    if labels.shape != a.shape or labels.shape != b.shape:
        raise ValueError("labels, image_a and image_b must share shape")
    if batch is None:
        batch = FrameBatch(labels)
    elif batch.labels.shape != labels.shape:
        raise ValueError("pre-built batch does not match labels shape")
    n = batch.n_labels
    keys = [
        f"{prefix_a}_{prefix_b}_pearson",
        f"{prefix_a}_{prefix_b}_spearman",
        f"{prefix_a}_{prefix_b}_manders_a",
        f"{prefix_a}_{prefix_b}_manders_b",
        f"{prefix_a}_{prefix_b}_overlap",
    ]
    if n == 0:
        return {key: np.empty(0, dtype=np.float32) for key in keys}

    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    t_a = float(threshold_a)
    t_b = float(threshold_b)
    counts = np.empty(n, dtype=np.float64)
    sum_a = np.empty(n, dtype=np.float64)
    sum_b = np.empty(n, dtype=np.float64)
    sum_aa = np.empty(n, dtype=np.float64)
    sum_bb = np.empty(n, dtype=np.float64)
    sum_ab = np.empty(n, dtype=np.float64)
    manders_a_num = np.empty(n, dtype=np.float64)
    manders_b_num = np.empty(n, dtype=np.float64)
    spearman = np.full(n, np.nan)
    for i, label_id in enumerate(batch.label_ids):
        # Only the label's own pixels are touched: the frame is never
        # rescanned once per label.
        px = batch.pixels(int(label_id))
        va = flat_a[px]
        vb = flat_b[px]
        counts[i] = float(px.size)
        sum_a[i] = va.sum()
        sum_b[i] = vb.sum()
        sum_aa[i] = np.sum(va * va)
        sum_bb[i] = np.sum(vb * vb)
        sum_ab[i] = np.sum(va * vb)
        manders_a_num[i] = np.sum(va[vb >= t_b])
        manders_b_num[i] = np.sum(vb[va >= t_a])
        if px.size < 2:
            continue
        ra = rankdata(va)
        rb = rankdata(vb)
        if ra.std() <= _EPS or rb.std() <= _EPS:
            continue
        spearman[i] = float(np.corrcoef(ra, rb)[0, 1])

    mean_a = sum_a / np.maximum(counts, 1)
    mean_b = sum_b / np.maximum(counts, 1)
    cov = sum_ab / np.maximum(counts, 1) - mean_a * mean_b
    var_a = sum_aa / np.maximum(counts, 1) - mean_a * mean_a
    var_b = sum_bb / np.maximum(counts, 1) - mean_b * mean_b
    denom = np.sqrt(np.maximum(var_a * var_b, 0.0))
    pearson = np.full(n, np.nan)
    valid = (denom > _EPS) & (counts > 1)
    pearson[valid] = cov[valid] / denom[valid]

    manders_a = np.full(n, np.nan)
    manders_b = np.full(n, np.nan)
    good_a = np.abs(sum_a) > _EPS
    good_b = np.abs(sum_b) > _EPS
    manders_a[good_a] = manders_a_num[good_a] / sum_a[good_a]
    manders_b[good_b] = manders_b_num[good_b] / sum_b[good_b]
    overlap = np.full(n, np.nan)
    denom_overlap = np.sqrt(np.maximum(sum_aa * sum_bb, 0.0))
    ok = denom_overlap > _EPS
    overlap[ok] = sum_ab[ok] / denom_overlap[ok]

    return {
        keys[0]: pearson.astype(np.float32),
        keys[1]: spearman.astype(np.float32),
        keys[2]: manders_a.astype(np.float32),
        keys[3]: manders_b.astype(np.float32),
        keys[4]: overlap.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Texture features
# ---------------------------------------------------------------------------


def texture_haralick(
    labels_2d: np.ndarray,
    image: np.ndarray,
    levels: int = 64,
    distances: tuple[int, ...] = (1,),
    angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> dict[str, np.ndarray]:
    """Per-cell Haralick texture features (CellProfiler analogue).

    For every instance, the intensity image is cropped to the instance
    bounding box, quantized to ``levels`` gray levels (min-max within the
    crop), and the symmetric, normalized GLCM is computed over
    ``distances`` x ``angles``; each property is averaged over all
    distance/angle combinations.

    Parameters
    ----------
    labels_2d : numpy.ndarray
        2D instance label image; 0 is background.
    image : numpy.ndarray
        2D grayscale image, same shape as ``labels_2d``.
    levels : int
        Number of gray levels for quantization.
    distances, angles : tuple
        GLCM offsets (see :func:`skimage.feature.graycomatrix`).

    Returns
    -------
    dict of str to numpy.ndarray
        ``haralick_{prop}`` float32 arrays of length ``n_instances``,
        ascending label order. Tiny instances whose crop cannot host the
        requested offset yield NaN-free 0.0 values.
    """
    labels = np.asarray(labels_2d)
    image = np.asarray(image, dtype=np.float32)
    if labels.shape != image.shape:
        raise ValueError(
            f"labels shape {labels.shape} does not match image shape {image.shape}"
        )
    if levels < 2:
        raise ValueError(f"levels must be >= 2, got {levels}")
    features: dict[str, list[float]] = {f"haralick_{p}": [] for p in HARALICK_PROPS}
    slices = ndimage.find_objects(labels)
    for label_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        crop = image[sl]
        cmin, cmax = float(crop.min()), float(crop.max())
        if cmax - cmin < _EPS:
            quantized = np.zeros(crop.shape, dtype=np.uint8)
        else:
            quantized = np.floor(
                (crop - cmin) / (cmax - cmin) * (levels - 1)
            ).astype(np.uint8)
        values: dict[str, float] = {p: 0.0 for p in HARALICK_PROPS}
        if min(quantized.shape) > max(distances):
            glcm = graycomatrix(
                quantized,
                distances=list(distances),
                angles=list(angles),
                levels=levels,
                symmetric=True,
                normed=True,
            )
            for p in HARALICK_PROPS:
                values[p] = float(np.mean(graycoprops(glcm, p)))
        for p in HARALICK_PROPS:
            features[f"haralick_{p}"].append(values[p])
    return {k: np.asarray(v, dtype=np.float32) for k, v in features.items()}


def texture_lbp(
    labels_2d: np.ndarray, image: np.ndarray, P: int = 8, R: float = 1.0
) -> dict[str, np.ndarray]:
    """Per-cell uniform-LBP histogram features.

    Computes :func:`skimage.feature.local_binary_pattern` once on the
    whole image (min-max quantized to uint8 to avoid float-artifact codes,
    ``method="uniform"``), then builds a normalized ``P + 2``-bin histogram
    of the LBP codes inside each instance.

    Parameters
    ----------
    labels_2d : numpy.ndarray
        2D instance label image; 0 is background.
    image : numpy.ndarray
        2D grayscale image, same shape as ``labels_2d``.
    P : int
        Number of circularly symmetric neighbor points.
    R : float
        Radius of the sampling circle (px).

    Returns
    -------
    dict of str to numpy.ndarray
        ``lbp_0`` .. ``lbp_{P+1}`` float32 arrays of length
        ``n_instances`` (each histogram row sums to 1), ascending label
        order.
    """
    labels = np.asarray(labels_2d)
    image = np.asarray(image, dtype=np.float32)
    if labels.shape != image.shape:
        raise ValueError(
            f"labels shape {labels.shape} does not match image shape {image.shape}"
        )
    if P < 1:
        raise ValueError(f"P must be >= 1, got {P}")
    imin, imax = float(image.min()), float(image.max())
    if imax - imin < _EPS:
        quantized = np.zeros(image.shape, dtype=np.uint8)
    else:
        quantized = np.round((image - imin) / (imax - imin) * 255.0).astype(np.uint8)
    lbp = local_binary_pattern(quantized, P, R, method="uniform").astype(np.int64)

    n_bins = P + 2
    features: dict[str, list[float]] = {f"lbp_{b}": [] for b in range(n_bins)}
    slices = ndimage.find_objects(labels)
    for label_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        codes = lbp[sl][labels[sl] == label_id]
        hist = np.bincount(codes, minlength=n_bins).astype(np.float64)
        hist /= max(float(codes.size), 1.0)
        for b in range(n_bins):
            features[f"lbp_{b}"].append(float(hist[b]))
    return {k: np.asarray(v, dtype=np.float32) for k, v in features.items()}


# ---------------------------------------------------------------------------
# Feature-column helper
# ---------------------------------------------------------------------------


def add_features(cells_table: pa.Table, feature_dict: dict[str, Any]) -> pa.Table:
    """Append per-cell feature columns to a cells table.

    Parameters
    ----------
    cells_table : pyarrow.Table
        A cells table (e.g. from :func:`labels_to_cells`).
    feature_dict : dict of str to array-like
        Column name -> per-cell values; every array must have one value
        per table row (ascending label order). Non-float arrays are
        appended with their inferred Arrow type; float arrays are stored
        as float32.

    Returns
    -------
    pyarrow.Table
        New table with the feature columns appended.

    Raises
    ------
    ValueError
        If a column name already exists or an array length does not match
        the row count.
    """
    out = cells_table
    for name, values in feature_dict.items():
        if name in out.column_names:
            raise ValueError(f"column {name!r} already exists in the cells table")
        arr = np.asarray(values)
        if arr.shape[0] != out.num_rows:
            raise ValueError(
                f"feature {name!r} has {arr.shape[0]} values but the table has "
                f"{out.num_rows} rows"
            )
        if np.issubdtype(arr.dtype, np.floating):
            column = pa.array(arr.astype(np.float32), type=pa.float32())
        else:
            column = pa.array(arr)
        out = out.append_column(name, column)
    return out
