"""Segmentation operators (ARCHITECTURE.md §4.2).

Traditional threshold segmentation (ported from ``momia2/segment/mask.py``
as pure functions with explicit pydantic parameters instead of
``configparser``), AI-segmenter protocol/registry and the Omnipose backend parameter model (implemented in momia3.models.omnipose_lite), probability-map
post-processing (ported from ``momia2/segment/_helper.py``), normalized
per-instance distance maps, weighted-window patch stitching (ported
from momia2's ``Patchifier`` as pure functions), and distance-transform
declumping for crowded fields (:func:`declump`).

Probability-map convention: ``prob`` has shape ``(c, y, x)`` with channel
0 = background, 1 = edge, 2 = cell interior (seed), matching the momia2
3-class U-Net output layout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, Union, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field
from scipy import ndimage
from skimage import feature, filters, measure, morphology, segmentation

logger = logging.getLogger(__name__)


def _remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """``remove_small_objects`` with momia2's ``min_size`` semantics.

    skimage >= 0.26 deprecated ``min_size`` in favor of ``max_size`` with
    off-by-one semantics (objects with size ``<= max_size`` are removed).
    """
    if min_size <= 1:
        return mask
    return morphology.remove_small_objects(mask, max_size=min_size - 1)

__all__ = [
    "ThresholdParams",
    "OmniposeParams",
    "ProbToLabelsParams",
    "DeclumpParams",
    "Segmenter",
    "SEGMENTERS",
    "PatchGrid",
    "threshold_segment",
    "prob_to_labels",
    "compute_dist",
    "declump",
    "register_segmenter",
    "get_segmenter",
    "build_patch_grid",
    "patchify",
    "unpatchify",
]


# ---------------------------------------------------------------------------
# Threshold segmentation (ported from momia2/segment/mask.py)
# ---------------------------------------------------------------------------


class ThresholdParams(BaseModel):
    """Parameters of :func:`threshold_segment`.

    Defaults reproduce momia2's ``mask_composite`` configuration
    (``sauvola_window_size=9``, ``sauvola_k=0.05``, ``min_size=10``,
    ``overlap_threshold=0.5``, dilation on, opening off).
    """

    method: Literal["isodata", "sauvola", "local", "composite"] = Field(
        "composite", description="Thresholding algorithm"
    )
    dark_foreground: bool = Field(
        True, description="Cells darker than background (phase contrast)"
    )
    block_size: int = Field(15, ge=3, description="Window for method='local'")
    local_offset: float = Field(
        0.0, description="Offset subtracted from the local threshold (method='local')"
    )
    sauvola_window: int = Field(9, ge=3, description="Sauvola window size (px)")
    sauvola_k: float = Field(0.05, gt=0, description="Sauvola sensitivity k")
    min_size: int = Field(
        10, ge=0, description="Minimum object size kept in method='composite' (px)"
    )
    opening: bool = Field(
        False, description="Binary opening to delink neighbors (composite)"
    )
    dilation: bool = Field(
        True, description="Final binary dilation by 1 px (composite)"
    )
    overlap_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum overlap fraction with the global mask (composite)",
    )


class OmniposeParams(BaseModel):
    """Parameters of the Omnipose segmentation backend (segment.omnipose).

    The Omnipose network outputs flow/dist/boundary channels and is handled by
    the label-producing adapter in momia3.models.omnipose_lite.segmenter; it
    does not go through prob_to_labels.
    """

    weights_path: str | None = Field(
        None,
        description="Omnipose checkpoint; None resolves MOMIA3_OMNIPOSE_WEIGHTS "
        "or ~/.cellpose/models/bact_phase_omnitorch_0",
    )
    device: str | None = Field(
        None, description="torch device (cuda/mps/cpu); None picks the best available"
    )
    half: bool = Field(
        False, description="CUDA bf16 autocast (fp16 overflows in CPnet)"
    )
    niter: int = Field(200, ge=1, description="Euler integration iterations")
    mask_threshold: float = Field(
        0.0, description="Cell-mask threshold on the distance channel"
    )
    diam_threshold: float = Field(
        12.0, gt=0, description="Mask-reconstruction diameter threshold (px)"
    )
    min_size: int = Field(15, ge=0, description="Minimum instance size (px)")
    max_size: int | None = Field(
        None, ge=1, description="Maximum instance size (px); None disables"
    )
    hole_size: int = Field(3, ge=0, description="Fill holes up to this size (px)")


def _binarize(image: np.ndarray, threshold: np.ndarray | float, dark: bool) -> np.ndarray:
    """Apply a scalar or per-pixel threshold honoring polarity."""
    if dark:
        return image < threshold
    return image > threshold


def _mask_isodata(image: np.ndarray, dark: bool) -> np.ndarray:
    """Global isodata threshold (momia2 ``mask_isodata``)."""
    return _binarize(image, filters.threshold_isodata(image), dark)


def _mask_sauvola(image: np.ndarray, dark: bool, window: int, k: float) -> np.ndarray:
    """Local Sauvola threshold (momia2 ``mask_sauvola``)."""
    window = max(3, window | 1)  # sauvola requires an odd window
    return _binarize(image, filters.threshold_sauvola(image, window_size=window, k=k), dark)


def _mask_local(image: np.ndarray, dark: bool, block_size: int, offset: float) -> np.ndarray:
    """Local adaptive threshold (momia2 ``mask_local`` + explicit offset)."""
    block_size = max(3, block_size | 1)
    th = filters.threshold_local(image, block_size=block_size, offset=offset)
    return _binarize(image, th, dark)


def _mask_composite(image: np.ndarray, params: ThresholdParams) -> np.ndarray:
    """Composite local+global mask (momia2 ``mask_composite``, vectorized).

    Union of the Sauvola (local) and isodata (global) masks, small-object
    removal, then regionprops overlap filtering: connected components whose
    overlap fraction with the global mask is below
    ``params.overlap_threshold`` are discarded (momia2 computed the same
    quantity as ``mean_intensity`` of the binary global mask per region).
    """
    local_mask = _mask_sauvola(
        image, params.dark_foreground, params.sauvola_window, params.sauvola_k
    )
    global_mask = _mask_isodata(image, params.dark_foreground)
    init = _remove_small(local_mask | global_mask, params.min_size)
    if params.opening:
        init = morphology.opening(init)

    labels = measure.label(init)
    n = int(labels.max())
    if n == 0:
        logger.warning("composite mask is empty after small-object removal")
        return np.zeros(image.shape, dtype=bool)
    index = np.arange(1, n + 1)
    overlap = ndimage.sum_labels(global_mask.astype(np.float64), labels, index=index)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.float64)
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = (overlap / np.maximum(sizes, 1.0)) >= params.overlap_threshold
    binary = keep[labels]
    if params.dilation:
        binary = morphology.dilation(binary)
    return binary


def threshold_segment(
    image: np.ndarray, params: ThresholdParams | None = None
) -> np.ndarray:
    """Binarize a 2D image into a boolean foreground mask.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale image (dark rods on bright background by default).
    params : ThresholdParams, optional
        Algorithm and thresholds.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` on foreground (cells).
    """
    params = params or ThresholdParams()
    work = np.asarray(image, dtype=np.float32)
    if work.ndim != 2:
        raise ValueError(f"threshold_segment expects a 2D image, got {work.shape}")
    if params.method == "isodata":
        mask = _mask_isodata(work, params.dark_foreground)
    elif params.method == "sauvola":
        mask = _mask_sauvola(
            work, params.dark_foreground, params.sauvola_window, params.sauvola_k
        )
    elif params.method == "local":
        mask = _mask_local(
            work, params.dark_foreground, params.block_size, params.local_offset
        )
    else:
        mask = _mask_composite(work, params)
    logger.debug("threshold_segment: method=%s", params.method)
    return np.asarray(mask, dtype=bool)


# ---------------------------------------------------------------------------
# Probability-map post-processing (ported from momia2/segment/_helper.py)
# ---------------------------------------------------------------------------


class ProbToLabelsParams(BaseModel):
    """Parameters of :func:`prob_to_labels`.

    ``seed_thresholds`` generalizes momia2's single ``seed_min`` to a
    ladder: strict (high) seeds are placed first, then components from
    lower thresholds that do not overlap existing markers are added, which
    recovers dim cells without merging touching ones.
    """

    bg_channel: int = Field(0, ge=0, description="Background probability channel")
    edge_channel: int = Field(1, ge=0, description="Edge probability channel")
    seed_channel: int = Field(2, ge=0, description="Cell-interior (seed) channel")
    fg_threshold: float = Field(
        0.4,
        gt=0,
        lt=1,
        description="Foreground where background prob < fg_threshold",
    )
    fg_erosion_radius: int = Field(
        1, ge=0, description="Foreground erosion radius (px), 0 disables"
    )
    seed_thresholds: tuple[float, ...] = Field(
        (0.5, 0.3), description="Seed-probability threshold ladder"
    )
    edge_max: float = Field(
        0.75, gt=0, lt=1, description="Seeds excluded where edge prob >= edge_max"
    )
    min_seed_size: int = Field(20, ge=0, description="Minimum seed component size (px)")
    seed_opening_radius: int = Field(
        1, ge=0, description="Seed binary-opening radius (px), 0 disables"
    )
    min_cell_size: int = Field(10, ge=0, description="Minimum final instance size (px)")
    compactness: float = Field(
        0.0, ge=0.0, description="Watershed compactness (0 = pure watershed)"
    )
    watershed_line: bool = Field(
        False, description="Separate instances by 1-px watershed lines"
    )


def _extract_seeds(prob: np.ndarray, params: ProbToLabelsParams) -> np.ndarray:
    """Labeled seed markers from the threshold ladder."""
    seed_prob = prob[params.seed_channel]
    edge_prob = prob[params.edge_channel]
    markers = np.zeros(seed_prob.shape, dtype=np.int32)
    next_id = 0
    for threshold in sorted(params.seed_thresholds, reverse=True):
        seed = (seed_prob > threshold) & (edge_prob < params.edge_max)
        if params.seed_opening_radius > 0:
            seed = morphology.opening(seed, morphology.disk(params.seed_opening_radius))
        seed = _remove_small(seed, params.min_seed_size)
        components = measure.label(seed)
        for comp in range(1, int(components.max()) + 1):
            comp_mask = components == comp
            # Ladder rule: only add components not overlapping placed markers.
            if np.any(markers[comp_mask] > 0):
                continue
            next_id += 1
            markers[comp_mask] = next_id
    return markers


def prob_to_labels(
    prob: np.ndarray, params: ProbToLabelsParams | None = None
) -> np.ndarray:
    """Instance labels from a 3-class probability map ``(c, y, x)``.

    Ports momia2's ``prediction2foreground`` + ``prediction2seed`` +
    marker-controlled watershed:

    1. Foreground = background probability below ``fg_threshold``
       (optionally eroded, as in ``prediction2foreground``).
    2. Seeds = interior probability above a threshold ladder, excluding
       high edge-probability pixels (as in ``prediction2seed``), with
       opening and small-component removal.
    3. Marker-controlled watershed on ``1 - interior probability`` inside
       the foreground mask.
    4. Instances smaller than ``min_cell_size`` are removed and labels are
       relabeled sequentially.

    Parameters
    ----------
    prob : numpy.ndarray
        Probability map of shape ``(c, y, x)``, ``c >= 3``.
    params : ProbToLabelsParams, optional
        Thresholds and morphology parameters.

    Returns
    -------
    numpy.ndarray
        Instance labels, uint32, shape ``(y, x)``; 0 is background.
    """
    params = params or ProbToLabelsParams()
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 3:
        raise ValueError(f"prob_to_labels expects (c, y, x), got {prob.shape}")
    n_channels = prob.shape[0]
    for name, ch in (
        ("bg_channel", params.bg_channel),
        ("edge_channel", params.edge_channel),
        ("seed_channel", params.seed_channel),
    ):
        if ch >= n_channels:
            raise ValueError(f"{name}={ch} out of range for {n_channels} channels")

    foreground = prob[params.bg_channel] < params.fg_threshold
    if params.fg_erosion_radius > 0:
        foreground = morphology.erosion(
            foreground, morphology.disk(params.fg_erosion_radius)
        )
    markers = _extract_seeds(prob, params)
    if int(markers.max()) == 0:
        logger.warning("prob_to_labels: no seeds found; returning empty labels")
        return np.zeros(prob.shape[1:], dtype=np.uint32)

    elevation = 1.0 - prob[params.seed_channel]
    labels = segmentation.watershed(
        elevation,
        markers=markers,
        mask=foreground,
        compactness=params.compactness,
        watershed_line=params.watershed_line,
    )
    if params.min_cell_size > 0:
        labels = _remove_small(labels, params.min_cell_size)
    labels, _, _ = segmentation.relabel_sequential(labels)
    logger.debug("prob_to_labels: %d instance(s)", int(labels.max()))
    return labels.astype(np.uint32)


# ---------------------------------------------------------------------------
# Normalized distance map (ported from momia2 compute_dist, vectorized)
# ---------------------------------------------------------------------------


def compute_dist(labels: np.ndarray, max_value: float = 7.0) -> np.ndarray:
    """Normalized per-instance Euclidean distance transform.

    Port of momia2's ``compute_dist`` (momia2/segment/_helper.py),
    replacing the per-instance spline/contour logic and full-image loops
    with a bounding-box-cropped :func:`scipy.ndimage.distance_transform_edt`
    per label: distances are computed inside each instance only (instance
    borders act as zeros), then the whole map is scaled to
    ``[0, max_value]`` by its global maximum (momia2's
    ``normalize_image(min_perc=0, max_perc=100, min_v=0, max_v=7)``).

    Parameters
    ----------
    labels : numpy.ndarray
        2D instance label image; 0 is background.
    max_value : float
        Upper bound of the normalized output range.

    Returns
    -------
    numpy.ndarray
        Normalized distance map, float32, same shape as ``labels``.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"compute_dist expects a 2D label image, got {labels.shape}")
    out = np.zeros(labels.shape, dtype=np.float32)
    slices = ndimage.find_objects(labels)
    for label_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        crop = labels[sl] == label_id
        dist = ndimage.distance_transform_edt(crop)
        region = out[sl]
        region[crop] = np.maximum(region[crop], dist[crop].astype(np.float32))
    peak = float(out.max())
    if peak > 0:
        out *= np.float32(max_value / peak)
    return out


# ---------------------------------------------------------------------------
# Declumping (EDT + peak_local_max + marker watershed)
# ---------------------------------------------------------------------------


class DeclumpParams(BaseModel):
    """Parameters of :func:`declump`.

    Defaults target bacterial rods: ``threshold_abs=2`` px suppresses
    shallow distance ridges on mask boundary noise, ``min_distance=5`` px
    enforces one seed per cell body, and 1-px watershed lines keep the
    split instances disjoint for downstream outline/midline extraction.
    When a precomputed ``dist`` is passed (e.g. the ``[0, 7]`` normalized
    map of :func:`compute_dist`), ``threshold_abs`` is expressed in that
    map's units and should be rescaled accordingly.
    """

    min_distance: int = Field(
        5, ge=1, description="Minimum spacing between seed peaks (px)"
    )
    threshold_abs: float = Field(
        2.0,
        ge=0.0,
        description="Minimum distance-map value eligible as a seed peak",
    )
    smooth_sigma: float = Field(
        1.0,
        ge=0.0,
        description=(
            "Gaussian sigma (px) applied to the distance map before peak "
            "detection; suppresses pixel-level local maxima along the EDT "
            "ridge of elongated cells (0 disables). The watershed itself "
            "always runs on the unsmoothed map."
        ),
    )
    h_maxima: float | None = Field(
        1.0,
        ge=0.0,
        description=(
            "Optional h-maxima prominence (distance-map units) applied "
            "before peak detection: local maxima standing less than h above "
            "their surroundings are suppressed, which removes secondary "
            "ridge bumps on elongated cells that smoothing and min_distance "
            "alone cannot (None disables)."
        ),
    )
    footprint_radius: int | None = Field(
        None,
        ge=1,
        description=(
            "Optional disk radius used as the peak_local_max footprint "
            "(overrides min_distance when set)"
        ),
    )
    exclude_border: bool = Field(
        False, description="Exclude seed peaks touching the image border"
    )
    min_cell_size: int = Field(
        10, ge=0, description="Minimum final instance size (px)"
    )
    compactness: float = Field(
        0.0, ge=0.0, description="Watershed compactness (0 = pure watershed)"
    )
    watershed_line: bool = Field(
        True, description="Separate instances by 1-px watershed lines"
    )


def declump(
    mask: np.ndarray,
    dist: np.ndarray | None = None,
    *,
    params: DeclumpParams | None = None,
) -> np.ndarray:
    """Split merged foreground blobs into per-cell instances.

    Distance-transform declumping for crowded fields where plain connected
    components fuse touching cells into one blob (recall collapse):

    1. Distance map: Euclidean distance transform of the foreground, or
       the caller-provided ``dist`` (e.g. an AI segmenter's distance head
       or :func:`compute_dist` output).
    2. Seeds: :func:`skimage.feature.peak_local_max` on the distance map
       (``min_distance`` / ``threshold_abs`` / optional disk footprint).
    3. Marker-controlled watershed on ``-dist``, restricted to the
       foreground mask.
    4. Instances smaller than ``min_cell_size`` are removed and labels are
       relabeled sequentially.

    Robustness fallback: if no seed peak survives the thresholds (e.g.
    very small or flat objects), the foreground's plain connected
    components are returned instead of an empty image, so declumping never
    loses cells relative to connected-component labeling.

    Parameters
    ----------
    mask : numpy.ndarray
        2D boolean foreground mask, or a 2D label image (any nonzero pixel
        counts as foreground; input labels are ignored — the point of
        declumping is to re-split merged components).
    dist : numpy.ndarray, optional
        Precomputed 2D distance map aligned with ``mask``; when omitted,
        the EDT of the foreground is computed internally.
    params : DeclumpParams, optional
        Seed and watershed parameters.

    Returns
    -------
    numpy.ndarray
        Instance labels, uint32, shape ``(y, x)``; 0 is background.
    """
    params = params or DeclumpParams()
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"declump expects a 2D mask/label image, got {mask.shape}")
    foreground = mask > 0
    if not bool(foreground.any()):
        logger.warning("declump: empty foreground; returning empty labels")
        return np.zeros(mask.shape, dtype=np.uint32)

    if dist is None:
        dist = ndimage.distance_transform_edt(foreground)
    else:
        dist = np.asarray(dist, dtype=np.float32)
        if dist.shape != foreground.shape:
            raise ValueError(
                f"declump: dist shape {dist.shape} does not match mask {foreground.shape}"
            )

    footprint = (
        morphology.disk(params.footprint_radius)
        if params.footprint_radius is not None
        else None
    )
    peak_map = (
        ndimage.gaussian_filter(dist, params.smooth_sigma)
        if params.smooth_sigma > 0
        else dist
    )
    if params.h_maxima is not None and params.h_maxima > 0:
        # h_maxima returns a binary mask of maxima with prominence >= h;
        # gate the (smoothed) map with it so secondary ridge bumps never
        # reach peak_local_max, while surviving maxima keep their heights.
        prominent = morphology.h_maxima(peak_map, params.h_maxima)
        peak_map = np.where(prominent, peak_map, 0.0)
    coords = feature.peak_local_max(
        peak_map,
        min_distance=None if footprint is not None else params.min_distance,
        threshold_abs=params.threshold_abs,
        exclude_border=params.exclude_border,
        footprint=footprint,
    )

    if coords.shape[0] == 0:
        logger.warning(
            "declump: no seed peaks (threshold_abs=%s); falling back to "
            "connected components",
            params.threshold_abs,
        )
        labels = measure.label(foreground)
    else:
        seed_mask = np.zeros(foreground.shape, dtype=bool)
        seed_mask[coords[:, 0], coords[:, 1]] = True
        # Thicken single-pixel peaks to 3x3 crosses (merged when adjacent).
        seed_mask = morphology.dilation(seed_mask)
        markers = measure.label(seed_mask)
        labels = segmentation.watershed(
            -dist,
            markers=markers,
            mask=foreground,
            compactness=params.compactness,
            watershed_line=params.watershed_line,
        )

    if params.min_cell_size > 0:
        labels = _remove_small(labels, params.min_cell_size)
    labels, _, _ = segmentation.relabel_sequential(labels)
    logger.debug("declump: %d instance(s)", int(labels.max()))
    return labels.astype(np.uint32)


# ---------------------------------------------------------------------------
# AI segmenter protocol and registry (Phase 3 stub)
# ---------------------------------------------------------------------------


@runtime_checkable
class Segmenter(Protocol):
    """Protocol for AI segmentation backends (§4.2).

    Implementations (``omnipose``, ``cellpose_sam``, ``unet_torch`` — to be
    added under ``momia3.models``) turn a 2D image into a ``(c, y, x)``
    probability map consumed by :func:`prob_to_labels`.
    """

    def predict(self, image: np.ndarray, params: BaseModel | None = None) -> np.ndarray:
        """Predict a ``(c, y, x)`` probability map from a 2D image."""
        ...


SEGMENTERS: dict[str, Union[Segmenter, Callable[..., np.ndarray]]] = {}
"""Registry of named AI segmentation backends; populated by ``momia3.models``."""


def register_segmenter(
    name: str, segmenter: Union[Segmenter, Callable[..., np.ndarray]]
) -> None:
    """Register an AI segmentation backend under ``name``.

    Raises
    ------
    KeyError
        If ``name`` is already registered.
    """
    if name in SEGMENTERS:
        raise KeyError(f"segmenter {name!r} already registered")
    SEGMENTERS[name] = segmenter
    logger.debug("registered segmenter %r", name)


def get_segmenter(name: str) -> Union[Segmenter, Callable[..., np.ndarray]]:
    """Look up a registered segmenter by name.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    try:
        return SEGMENTERS[name]
    except KeyError:
        raise KeyError(
            f"segmenter {name!r} not registered; available: {sorted(SEGMENTERS)}"
        ) from None


# ---------------------------------------------------------------------------
# Patchify / unpatchify (ported from momia2 Patchifier, pure functions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchGrid:
    """Immutable patch-stitching geometry.

    Attributes
    ----------
    image_shape : tuple of int
        Original ``(h, w)`` image shape.
    patch_size : int
        Square patch side (px).
    overlap : int
        Overlap between adjacent patches (px).
    stride : int
        Patch stride (``patch_size - overlap``).
    padded_shape : tuple of int
        Padded canvas shape covering the patch grid.
    coords : tuple of tuple of int
        Top-left ``(y, x)`` of every patch on the padded canvas.
    window : numpy.ndarray
        2D Hann weighting window, floored at 1e-3 so every covered pixel
        carries non-zero weight (deviation from momia2, whose Hann window
        is exactly 0 on the first/last row and column, zeroing border
        pixels in weighted mode).
    """

    image_shape: tuple[int, int]
    patch_size: int
    overlap: int
    stride: int
    padded_shape: tuple[int, int]
    coords: tuple[tuple[int, int], ...]
    window: np.ndarray = field(compare=False)


def _hann_window(size: int, floor: float = 1e-3) -> np.ndarray:
    """Floored 2D Hann window (see :class:`PatchGrid`)."""
    hann_1d = np.hanning(size).astype(np.float32)
    window = np.outer(hann_1d, hann_1d)
    return np.clip(window, np.float32(floor), np.float32(1.0))


def build_patch_grid(
    image_shape: tuple[int, int], patch_size: int, overlap: int
) -> PatchGrid:
    """Compute the deterministic patch grid for an image shape.

    Ports momia2 ``Patchifier._generate_patch_coords``: coordinates step by
    ``stride`` and the last coordinate is clamped to the far edge so the
    grid always covers the padded canvas.

    Parameters
    ----------
    image_shape : tuple of int
        ``(h, w)`` of the unpadded image.
    patch_size : int
        Square patch side (px).
    overlap : int
        Overlap between adjacent patches (px), ``0 <= overlap < patch_size``.

    Returns
    -------
    PatchGrid
        Grid geometry shared by :func:`patchify` and :func:`unpatchify`.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    if h < 1 or w < 1:
        raise ValueError(f"image_shape must be positive, got {image_shape}")
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    if not 0 <= overlap < patch_size:
        raise ValueError(f"overlap must be in [0, patch_size), got {overlap}")
    stride = patch_size - overlap
    padded_h, padded_w = max(h, patch_size), max(w, patch_size)

    def _axis_coords(length: int) -> list[int]:
        starts = list(range(0, length - patch_size + 1, stride))
        if not starts:
            starts = [0]
        if starts[-1] + patch_size < length:
            starts.append(length - patch_size)
        return sorted(set(starts))

    ys = _axis_coords(padded_h)
    xs = _axis_coords(padded_w)
    coords = tuple((y, x) for y in ys for x in xs)
    return PatchGrid(
        image_shape=(h, w),
        patch_size=patch_size,
        overlap=overlap,
        stride=stride,
        padded_shape=(padded_h, padded_w),
        coords=coords,
        window=_hann_window(patch_size),
    )


def patchify(
    image: np.ndarray, patch_size: int, overlap: int
) -> tuple[np.ndarray, PatchGrid]:
    """Split an image into overlapping patches.

    Parameters
    ----------
    image : numpy.ndarray
        Image of shape ``(h, w)`` or ``(h, w, c)``.
    patch_size : int
        Square patch side (px).
    overlap : int
        Overlap between adjacent patches (px).

    Returns
    -------
    patches : numpy.ndarray
        ``(n, patch_size, patch_size)`` or ``(n, patch_size, patch_size, c)``
        patches in row-major grid order (deterministic; momia2's random
        rotation augmentation is intentionally not ported).
    grid : PatchGrid
        Geometry needed by :func:`unpatchify`.
    """
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError(f"patchify expects (h, w) or (h, w, c), got {image.shape}")
    grid = build_patch_grid(image.shape[:2], patch_size, overlap)
    pad_h = grid.padded_shape[0] - grid.image_shape[0]
    pad_w = grid.padded_shape[1] - grid.image_shape[1]
    pad_width: list[tuple[int, int]] = [(0, pad_h), (0, pad_w)]
    if image.ndim == 3:
        pad_width.append((0, 0))
    padded = np.pad(image, pad_width, mode="symmetric")
    size = grid.patch_size
    patches = np.stack(
        [padded[y : y + size, x : x + size] for y, x in grid.coords], axis=0
    )
    return patches, grid


def unpatchify(
    patches: np.ndarray, grid: PatchGrid, mode: str = "weighted"
) -> np.ndarray:
    """Stitch patches back into an image.

    Parameters
    ----------
    patches : numpy.ndarray
        ``(n, patch_size, patch_size[, c])`` patches in :func:`patchify`
        order.
    grid : PatchGrid
        Geometry returned by :func:`patchify` (or :func:`build_patch_grid`).
    mode : {"weighted", "mean", "max", "min"}
        Merge mode; ``"weighted"`` uses the grid's floored Hann window.

    Returns
    -------
    numpy.ndarray
        Stitched image cropped back to ``grid.image_shape``.
    """
    if mode not in ("weighted", "mean", "max", "min"):
        raise ValueError(f"mode must be weighted/mean/max/min, got {mode!r}")
    patches = np.asarray(patches)
    if patches.ndim not in (3, 4):
        raise ValueError(
            f"unpatchify expects (n, h, w) or (n, h, w, c), got {patches.shape}"
        )
    if patches.shape[0] != len(grid.coords):
        raise ValueError(
            f"got {patches.shape[0]} patches but grid has {len(grid.coords)} coords"
        )
    size = grid.patch_size
    if patches.shape[1] != size or patches.shape[2] != size:
        raise ValueError(
            f"patch shape {patches.shape[1:3]} does not match patch_size {size}"
        )
    has_channels = patches.ndim == 4
    ph, pw = grid.padded_shape
    canvas_shape = (ph, pw, patches.shape[3]) if has_channels else (ph, pw)
    canvas = np.zeros(canvas_shape, dtype=np.float32)
    weight_map = np.zeros(canvas_shape, dtype=np.float32)
    window = grid.window[:, :, None] if has_channels else grid.window

    for i, (y, x) in enumerate(grid.coords):
        region = np.s_[y : y + size, x : x + size]
        patch = patches[i].astype(np.float32, copy=False)
        if mode == "weighted":
            canvas[region] += patch * window
            weight_map[region] += window
        elif mode == "mean":
            canvas[region] += patch
            weight_map[region] += 1.0
        elif mode == "max":
            if i == 0:
                canvas.fill(-np.inf)
            canvas[region] = np.maximum(canvas[region], patch)
        else:  # min
            if i == 0:
                canvas.fill(np.inf)
            canvas[region] = np.minimum(canvas[region], patch)

    if mode in ("weighted", "mean"):
        weight_map = np.where(weight_map < 1e-6, 1.0, weight_map)
        canvas = canvas / weight_map

    h, w = grid.image_shape
    if has_channels:
        return canvas[:h, :w, :]
    return canvas[:h, :w]
