"""Preprocessing operators (ARCHITECTURE.md §4.1).

Pure functions on 2D/3D numpy arrays with pydantic v2 parameter models:

- :func:`illumination_correct` — simplified BaSiC-style flatfield/darkfield
  correction (documented simplification: darkfield = temporal median,
  flatfield = gaussian-smoothed temporal mean of the darkfield-subtracted
  stack; the iterative low-rank + smooth-surface optimization of BaSiC is
  replaced by this one-shot estimate, which is sufficient for slowly
  varying illumination).
- :func:`rolling_ball_background_fft` — FFT-convolution approximation of
  the classical rolling-ball background, ported from
  ``momia2/external/mosaic_picasso_torch.py::rolling_ball_fft_background``
  with a mandatory numpy/scipy path (``momia3.backend.Ops`` has no
  rolling-ball primitive, so the kernel convolution is done directly with
  :func:`scipy.signal.fftconvolve`; both paths compute the same normalized
  parabola/ball kernel convolution).
- :func:`drift_correct` — phase cross-correlation drift correction:
  subpixel shifts are estimated **once** per frame pair on the reference
  channel, accumulated into a timeseries, and the *same* cumulative shift
  is applied to every channel via a single Fourier-shift resampling per
  frame.
- :func:`bleach_correct` — single/double exponential bleach correction.
- :func:`normalize` — unified percentile/gamma normalization with an
  explicit output dtype.
- :func:`denoise` — skimage wrappers (NLM / gaussian / bilateral).

All functions are deterministic, have no side effects, and log through
the module logger.
"""

from __future__ import annotations

import logging
from typing import Literal, Union

import numpy as np
from numba import njit, prange
from pydantic import BaseModel, Field
from scipy import ndimage, optimize, signal
from skimage import restoration
from skimage.registration import phase_cross_correlation

from ..backend import Ops

logger = logging.getLogger(__name__)

__all__ = [
    "IlluminationParams",
    "RollingBallParams",
    "DriftParams",
    "BleachParams",
    "NormalizeParams",
    "DenoiseParams",
    "illumination_correct",
    "rolling_ball_background_fft",
    "shrink_img_local_min",
    "drift_correct",
    "bleach_correct",
    "normalize",
    "denoise",
]

_EPS = 1e-12


def _estimate_noise(image: np.ndarray) -> float:
    """Robust noise-sigma estimate (MAD of the high-pass residual).

    Self-contained replacement for ``skimage.restoration.estimate_sigma``
    (which requires the optional PyWavelets dependency): the image is
    high-pass filtered with a 3x3 median filter and the median absolute
    deviation of the residual is scaled by the gaussian consistency
    constant 0.6745.
    """
    residual = image - ndimage.median_filter(image, size=3)
    mad = float(np.median(np.abs(residual)))
    return mad / 0.6745


# ---------------------------------------------------------------------------
# Parameter models
# ---------------------------------------------------------------------------


class IlluminationParams(BaseModel):
    """Parameters of :func:`illumination_correct`."""

    smooth_sigma: float = Field(
        20.0, gt=0, description="Gaussian sigma (px) of the flatfield surface"
    )
    clip_min: float | None = Field(
        0.0,
        description="Lower clip bound of the corrected stack; None disables clipping",
    )


class RollingBallParams(BaseModel):
    """Parameters of :func:`rolling_ball_background_fft`."""

    radius: int = Field(50, ge=1, description="Rolling-ball kernel radius (px)")
    kind: Literal["parabola", "ball"] = Field(
        "parabola", description="Kernel profile: paraboloid or hemisphere"
    )
    shrink_factor: int | None = Field(
        4,
        ge=1,
        description=(
            "Local-min shrink factor applied before the FFT convolution "
            "(momia2 production-path amortization): the background is estimated "
            "on the shrunk image with a radius/shrink_factor kernel and resized "
            "back. None or 1 keeps the exact full-resolution estimate."
        ),
    )


class DriftParams(BaseModel):
    """Parameters of :func:`drift_correct`."""

    upsample_factor: int = Field(
        10, ge=1, description="Subpixel upsampling of the phase-correlation peak"
    )
    reference_channel: str | None = Field(
        None,
        description=(
            "Channel used to estimate shifts when the input is a dict of "
            "channels; None uses the first channel (sorted by name)"
        ),
    )


class BleachParams(BaseModel):
    """Parameters of :func:`bleach_correct`."""

    order: Literal[1, 2] = Field(
        1, description="Number of exponential components of the decay model"
    )


class NormalizeParams(BaseModel):
    """Parameters of :func:`normalize`."""

    lower_percentile: float = Field(
        1.0, ge=0.0, le=100.0, description="Lower intensity percentile"
    )
    upper_percentile: float = Field(
        99.9, ge=0.0, le=100.0, description="Upper intensity percentile"
    )
    gamma: float = Field(1.0, gt=0, description="Gamma applied after rescaling")
    dtype: Literal["uint8", "uint16", "float32"] = Field(
        "float32", description="Explicit output dtype"
    )


class DenoiseParams(BaseModel):
    """Parameters of :func:`denoise`."""

    method: Literal["nlm", "gaussian", "bilateral"] = Field(
        "nlm", description="Denoising algorithm"
    )
    sigma: float = Field(
        1.0, gt=0, description="Gaussian sigma / bilateral spatial sigma (px)"
    )
    h: float | None = Field(
        None,
        gt=0,
        description="NLM filtering strength; None derives it from the estimated noise",
    )
    patch_size: int = Field(5, ge=1, description="NLM patch size (px)")
    patch_distance: int = Field(6, ge=1, description="NLM search radius (px)")


# ---------------------------------------------------------------------------
# Illumination correction (simplified BaSiC)
# ---------------------------------------------------------------------------


def illumination_correct(
    stack: np.ndarray, params: IlluminationParams | None = None
) -> np.ndarray:
    """Flatfield/darkfield correction of a timelapse stack (simplified BaSiC).

    Simplification of the BaSiC low-rank smooth model: instead of
    iteratively fitting a low-rank intensity component plus a smooth
    shading surface, the darkfield is estimated once as the per-pixel
    temporal median and the flatfield as the gaussian-smoothed temporal
    mean of the darkfield-subtracted stack. The flatfield is normalized to
    unit mean so corrected intensities keep their original scale.

    Parameters
    ----------
    stack : numpy.ndarray
        Timelapse of shape ``(t, y, x)``, any real dtype.
    params : IlluminationParams, optional
        Smoothing sigma and clipping bound.

    Returns
    -------
    numpy.ndarray
        Corrected stack, float32, shape ``(t, y, x)``:
        ``(stack - darkfield) / flatfield``.
    """
    params = params or IlluminationParams()
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"illumination_correct expects (t, y, x), got {stack.shape}")
    if stack.shape[0] < 2:
        logger.warning("illumination_correct on a single frame; returning float copy")
        return stack.astype(np.float32)

    work = stack.astype(np.float32, copy=False)
    darkfield = np.median(work, axis=0)
    subtracted = work - darkfield[None, :, :]
    flatfield = np.mean(subtracted, axis=0)
    flatfield = ndimage.gaussian_filter(flatfield, sigma=params.smooth_sigma)
    flatfield = np.where(
        np.abs(flatfield) < _EPS, np.float32(1.0), flatfield / np.mean(flatfield)
    )
    corrected = subtracted / flatfield[None, :, :].astype(np.float32)
    if params.clip_min is not None:
        corrected = np.maximum(corrected, np.float32(params.clip_min))
    logger.debug(
        "illumination_correct: t=%d, sigma=%.1f", stack.shape[0], params.smooth_sigma
    )
    return corrected.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Rolling-ball background (FFT approximation)
# ---------------------------------------------------------------------------


def _rolling_ball_kernel(radius: int, kind: str) -> np.ndarray:
    """Normalized 2D rolling-ball kernel (paraboloid or hemisphere).

    Same kernel definition as
    ``momia2/external/mosaic_picasso_torch.py::rolling_ball_fft_background``.
    """
    size = 2 * radius + 1
    ax = np.arange(size, dtype=np.float32) - float(radius)
    d2 = ax[:, None] ** 2 + ax[None, :] ** 2
    r2 = float(radius) ** 2
    if kind == "ball":
        kernel = np.sqrt(np.clip(r2 - d2, 0.0, None))
    elif kind == "parabola":
        kernel = np.clip(r2 - d2, 0.0, None)
    else:
        raise ValueError(f"kind must be 'ball' or 'parabola', got {kind!r}")
    return kernel / (kernel.sum() + _EPS)


def shrink_img_local_min(image: np.ndarray, shrink_factor: int = 4) -> np.ndarray:
    """Block-minimum downsample (pure-function port of momia2's numba
    ``shrink_img_local_min``).

    The image is divided into ``shrink_factor`` × ``shrink_factor`` blocks
    over its top-left ``(r // s) * s`` × ``(c // s) * c`` region (trailing
    rows/columns are dropped, exactly as in momia2) and each block is
    replaced by its minimum — dark objects (cells) therefore do not raise
    the estimated background after upsampling. Vectorized equivalent of
    momia2's ``@jit`` double loop; values are identical.

    Parameters
    ----------
    image : numpy.ndarray
        2D image ``(y, x)``.
    shrink_factor : int
        Block side (px).

    Returns
    -------
    numpy.ndarray
        Shrunk image ``(y // s, x // s)``, same dtype as ``image``.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"shrink_img_local_min expects a 2D image, got {image.shape}")
    s = int(shrink_factor)
    if s < 1:
        raise ValueError(f"shrink_factor must be >= 1, got {shrink_factor}")
    r, c = image.shape
    r_s, c_s = int(r / s), int(c / s)
    if r_s == 0 or c_s == 0:
        raise ValueError(
            f"shrink_factor {s} too large for image shape {image.shape}"
        )
    cropped = image[: r_s * s, : c_s * s]
    return cropped.reshape(r_s, s, c_s, s).min(axis=(1, 3))


@njit(cache=True)
def _shrink_min_nb(frame: np.ndarray, s: int, out: np.ndarray) -> np.ndarray:
    """Numba hot-path twin of :func:`shrink_img_local_min` (float32).

    Single streaming pass over the image; min is exact, so values are
    identical to the vectorized numpy reference (verified in
    ``tests/test_preprocess.py``) while avoiding its two reduction
    temporaries (~2.6 ms → ~0.3 ms at 1024²).
    """
    r_s, c_s = out.shape
    for bi in range(r_s):
        y0 = bi * s
        for bj in range(c_s):
            x0 = bj * s
            m = frame[y0, x0]
            for dy in range(s):
                for dx in range(s):
                    v = frame[y0 + dy, x0 + dx]
                    if v < m:
                        m = v
            out[bi, bj] = m
    return out


@njit(cache=True, parallel=True)
def _upsample_subtract_nb(
    background_s: np.ndarray, frame: np.ndarray, out: np.ndarray
) -> np.ndarray:
    """Fused bilinear upsample of the shrunk background + subtraction.

    Center-aligned bilinear interpolation with edge clamping — the same
    coordinate convention and border handling as
    ``skimage.transform.resize(..., order=1, anti_aliasing=True)`` at
    upscale factors (anti-aliasing is a no-op when upsampling, so the
    interpolation math is identical). Fusing the upsample with the
    subtraction in one numba pass avoids skimage's generic warping
    overhead, which measured ~12.5 ms for 256²→1024² — slower than the
    exact full-resolution FFT path itself and the reason a plain
    ``resize`` composition caps the shrink-path speedup far below its
    target. Numerics match the resize-based composition to float32
    rounding (Pearson ≈ 1.0; see ``tests/test_preprocess.py``). Rows are
    independent, so the outer loop runs in parallel; each output pixel is
    written exactly once, keeping results deterministic.
    """
    h_s, w_s = background_s.shape
    h, w = frame.shape
    scale_y = h_s / h
    scale_x = w_s / w
    one = np.float32(1.0)
    for i in prange(h):
        sy = (i + 0.5) * scale_y - 0.5
        if sy < 0.0:
            sy = 0.0
        elif sy > h_s - 1.0:
            sy = h_s - 1.0
        y0 = int(sy)
        y1 = min(y0 + 1, h_s - 1)
        fy = np.float32(sy - y0)
        wy0 = one - fy
        for j in range(w):
            sx = (j + 0.5) * scale_x - 0.5
            if sx < 0.0:
                sx = 0.0
            elif sx > w_s - 1.0:
                sx = w_s - 1.0
            x0 = int(sx)
            x1 = min(x0 + 1, w_s - 1)
            fx = np.float32(sx - x0)
            wx0 = one - fx
            top = background_s[y0, x0] * wx0 + background_s[y0, x1] * fx
            bot = background_s[y1, x0] * wx0 + background_s[y1, x1] * fx
            out[i, j] = frame[i, j] - (top * wy0 + bot * fy)
    return out


def rolling_ball_background_fft(
    image: np.ndarray, params: RollingBallParams | None = None
) -> np.ndarray:
    """Rolling-ball background subtraction via FFT convolution.

    The smooth background is estimated by convolving the image with a
    normalized parabola/ball kernel and subtracted. This is the FFT-based
    approximation ported from momia2's ``rolling_ball_fft_background``;
    the numpy/scipy path (:func:`scipy.signal.fftconvolve`) is mandatory
    and matches the torch FFT path of momia2 up to float precision (the
    centered ``mode="same"`` crop is equivalent to momia2's manual
    ``kh//2`` crop because the kernel is symmetric).

    When ``params.shrink_factor`` is greater than 1 (the default), the
    convolution runs on a local-min shrunk image
    (:func:`shrink_img_local_min`, the same amortization momia2's
    production path applies to its numba kernel) with a
    ``radius / shrink_factor`` kernel (matching momia2's
    ``rolling_ball.radius`` scaling) and the background is upsampled back
    with a fused bilinear-upsample + subtract numba kernel
    (:func:`_upsample_subtract_nb`; same center-aligned bilinear math as
    ``skimage.transform.resize(order=1)``, whose generic warping overhead
    otherwise caps the speedup — documented deviation). The shrunk
    estimate tracks the exact full-resolution one at Pearson ≥ 0.995 on
    the benchmark shading phantom (see ``tests/test_preprocess.py``). Set
    ``shrink_factor`` to None or 1 for the exact full-resolution path.

    Parameters
    ----------
    image : numpy.ndarray
        2D image ``(y, x)`` or 3D stack ``(t, y, x)`` (filtered per frame).
    params : RollingBallParams, optional
        Kernel radius, profile and shrink factor.

    Returns
    -------
    numpy.ndarray
        Background-subtracted image, float32, same shape as ``image``.
    """
    params = params or RollingBallParams()
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError(
            f"rolling_ball_background_fft expects 2D or (t, y, x), got {image.shape}"
        )
    work = image.astype(np.float32, copy=False)
    shrink = params.shrink_factor if params.shrink_factor is not None else 1

    if shrink > 1:
        radius_s = max(1, int(round(params.radius / shrink)))
        kernel = _rolling_ball_kernel(radius_s, params.kind)

        def _subtract(frame: np.ndarray) -> np.ndarray:
            frame = np.ascontiguousarray(frame, dtype=np.float32)
            r_s, c_s = frame.shape[0] // shrink, frame.shape[1] // shrink
            if r_s == 0 or c_s == 0:
                raise ValueError(
                    f"shrink_factor {shrink} too large for image shape {frame.shape}"
                )
            shrunk = np.empty((r_s, c_s), dtype=np.float32)
            _shrink_min_nb(frame, shrink, shrunk)
            background_s = np.ascontiguousarray(
                signal.fftconvolve(shrunk, kernel, mode="same"), dtype=np.float32
            )
            out = np.empty_like(frame)
            return _upsample_subtract_nb(background_s, frame, out)

    else:
        kernel = _rolling_ball_kernel(params.radius, params.kind)

        def _subtract(frame: np.ndarray) -> np.ndarray:
            background = signal.fftconvolve(frame, kernel, mode="same")
            return frame - background

    if image.ndim == 2:
        out = _subtract(work)
    else:
        out = np.empty_like(work)
        for t in range(work.shape[0]):
            out[t] = _subtract(work[t])
    logger.debug(
        "rolling_ball_background_fft: radius=%d kind=%s shrink=%s shape=%s",
        params.radius,
        params.kind,
        shrink,
        image.shape,
    )
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Drift correction (phase cross-correlation)
# ---------------------------------------------------------------------------


def _fourier_shift_image(image: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    """Subpixel shift of a 2D image, dispatched through ``backend.Ops``.

    Falls back to the scipy FFT path (``ndimage.fourier_shift`` applied to
    the image's FFT) when torch is unavailable; both are the Fourier shift
    theorem with wrap-around borders.
    """
    ops = Ops()
    return ops.fourier_shift(image, shift)


def drift_correct(
    stack: Union[np.ndarray, dict[str, np.ndarray]],
    params: DriftParams | None = None,
) -> tuple[Union[np.ndarray, dict[str, np.ndarray]], np.ndarray]:
    """Correct whole-field drift of a timelapse by phase cross-correlation.

    Pairwise shifts between consecutive frames are estimated **once** on
    the reference channel with
    :func:`skimage.registration.phase_cross_correlation`
    (subpixel, ``upsample_factor``), accumulated into a per-frame
    timeseries, and the *same* cumulative shift is applied to every
    channel of a frame via a single Fourier-shift resampling
    (:meth:`momia3.backend.Ops.fourier_shift`).

    Parameters
    ----------
    stack : numpy.ndarray or dict of str to numpy.ndarray
        Single-channel timelapse ``(t, y, x)``, or a dict mapping channel
        names to ``(t, y, x)`` stacks of identical shape.
    params : DriftParams, optional
        Upsampling factor and reference-channel selection.

    Returns
    -------
    corrected : numpy.ndarray or dict of str to numpy.ndarray
        Drift-corrected stack(s), float32, same structure as ``stack``.
    shifts : numpy.ndarray
        Cumulative ``(t, 2)`` float32 shifts ``(dy, dx)`` relative to
        frame 0; ``shifts[0] == (0, 0)``.
    """
    params = params or DriftParams()
    if isinstance(stack, dict):
        if not stack:
            raise ValueError("drift_correct received an empty channel dict")
        names = sorted(stack)
        ref_name = params.reference_channel or names[0]
        if ref_name not in stack:
            raise KeyError(f"reference channel {ref_name!r} not in channels {names}")
        channels = dict(stack)
    else:
        channels = {"image": np.asarray(stack)}
        ref_name = "image"

    reference = np.asarray(channels[ref_name])
    if reference.ndim != 3:
        raise ValueError(f"drift_correct expects (t, y, x), got {reference.shape}")
    n_frames = reference.shape[0]
    for name, arr in channels.items():
        if np.asarray(arr).shape != reference.shape:
            raise ValueError(
                f"channel {name!r} shape {np.asarray(arr).shape} does not match "
                f"reference shape {reference.shape}"
            )

    shifts = np.zeros((n_frames, 2), dtype=np.float32)
    for t in range(1, n_frames):
        pair_shift, _, _ = phase_cross_correlation(
            reference[t - 1].astype(np.float32, copy=False),
            reference[t].astype(np.float32, copy=False),
            upsample_factor=params.upsample_factor,
        )
        # phase_cross_correlation(ref, mov) returns the *correction* that
        # aligns mov onto ref; cumulative sum registers frame t onto frame 0.
        shifts[t] = shifts[t - 1] + np.asarray(pair_shift, dtype=np.float32)

    corrected: dict[str, np.ndarray] = {}
    for name, arr in channels.items():
        work = np.asarray(arr)
        out = np.empty(reference.shape, dtype=np.float32)
        for t in range(n_frames):
            if np.all(shifts[t] == 0.0):
                out[t] = work[t].astype(np.float32, copy=False)
            else:
                out[t] = _fourier_shift_image(
                    work[t].astype(np.float32, copy=False), tuple(shifts[t])
                )
        corrected[name] = out

    logger.debug(
        "drift_correct: %d frame(s), max |shift| = %.2f px",
        n_frames,
        float(np.abs(shifts).max()) if n_frames else 0.0,
    )
    if isinstance(stack, dict):
        return corrected, shifts
    return corrected["image"], shifts


# ---------------------------------------------------------------------------
# Bleach correction (exponential fit)
# ---------------------------------------------------------------------------


def _double_exp(t: np.ndarray, a1: float, tau1: float, a2: float, tau2: float) -> np.ndarray:
    """Double-exponential decay model."""
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)


def bleach_correct(
    stack: np.ndarray, params: BleachParams | None = None
) -> np.ndarray:
    """Bleach correction by exponential fitting of the mean intensity trace.

    Fits ``I(t) = a * exp(-t / tau)`` (``order=1``, via a robust log-linear
    fit) or ``I(t) = a1 * exp(-t/tau1) + a2 * exp(-t/tau2)`` (``order=2``,
    via :func:`scipy.optimize.curve_fit`) to the per-frame mean intensity
    and divides the fitted decay out, preserving the frame-0 level.

    Parameters
    ----------
    stack : numpy.ndarray
        Timelapse of shape ``(t, y, x)``.
    params : BleachParams, optional
        Number of exponential components.

    Returns
    -------
    numpy.ndarray
        Bleach-corrected stack, float32, same shape as ``stack``.
    """
    params = params or BleachParams()
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"bleach_correct expects (t, y, x), got {stack.shape}")
    work = stack.astype(np.float32, copy=False)
    n_frames = work.shape[0]
    if n_frames < 2:
        return work.copy()

    t = np.arange(n_frames, dtype=np.float64)
    trace = work.reshape(n_frames, -1).mean(axis=1).astype(np.float64)

    fitted: np.ndarray
    if params.order == 1 or np.any(trace <= 0):
        if params.order == 2 and np.any(trace <= 0):
            logger.warning(
                "non-positive mean intensities; falling back to order=1 fit"
            )
        slope, intercept = np.polyfit(t, np.log(np.maximum(trace, _EPS)), 1)
        fitted = np.exp(intercept + slope * t)
    else:
        span = max(float(n_frames - 1), 1.0)
        try:
            popt, _ = optimize.curve_fit(
                _double_exp,
                t,
                trace,
                p0=(trace[0] / 2.0, span / 3.0, trace[0] / 2.0, span * 3.0),
                bounds=([0.0, 1e-3, 0.0, 1e-3], [np.inf, np.inf, np.inf, np.inf]),
                maxfev=10000,
            )
            fitted = _double_exp(t, *popt)
        except (RuntimeError, ValueError) as exc:
            logger.warning("double-exponential fit failed (%s); using order=1", exc)
            slope, intercept = np.polyfit(t, np.log(trace), 1)
            fitted = np.exp(intercept + slope * t)

    fitted = np.maximum(fitted, _EPS)
    gain = (fitted[0] / fitted).astype(np.float32)
    corrected = work * gain[:, None, None]
    logger.debug("bleach_correct: order=%d, frames=%d", params.order, n_frames)
    return corrected.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize(image: np.ndarray, params: NormalizeParams | None = None) -> np.ndarray:
    """Percentile/gamma normalization with an explicit output dtype.

    Maps the ``[lower_percentile, upper_percentile]`` intensity range to
    ``[0, 1]``, applies ``gamma``, and casts to the requested dtype with
    bounds clipping (``uint8`` → ``[0, 255]``, ``uint16`` → ``[0, 65535]``,
    ``float32`` → ``[0, 1]``).

    Parameters
    ----------
    image : numpy.ndarray
        2D image or ``(t, y, x)`` stack; percentiles are computed over the
        whole array so a stack shares one intensity mapping.
    params : NormalizeParams, optional
        Percentiles, gamma and output dtype.

    Returns
    -------
    numpy.ndarray
        Normalized image with dtype ``params.dtype``.
    """
    params = params or NormalizeParams()
    if params.upper_percentile < params.lower_percentile:
        raise ValueError(
            f"upper_percentile ({params.upper_percentile}) must be >= "
            f"lower_percentile ({params.lower_percentile})"
        )
    work = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(work, [params.lower_percentile, params.upper_percentile])
    if hi - lo < _EPS:
        logger.warning("normalize: degenerate intensity range; returning zeros")
        out = np.zeros_like(work)
    else:
        out = np.clip((work - lo) / (hi - lo), 0.0, 1.0)
    if params.gamma != 1.0:
        out = out ** np.float32(params.gamma)
    if params.dtype == "uint8":
        return np.round(out * 255.0).astype(np.uint8)
    if params.dtype == "uint16":
        return np.round(out * 65535.0).astype(np.uint16)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------


def denoise(image: np.ndarray, params: DenoiseParams | None = None) -> np.ndarray:
    """Denoise a 2D image with skimage algorithms.

    Parameters
    ----------
    image : numpy.ndarray
        2D image, any real dtype (converted to float32 internally).
    params : DenoiseParams, optional
        Algorithm selection and parameters. ``"nlm"`` uses
        :func:`skimage.restoration.denoise_nl_means` with the noise level
        estimated by a MAD high-pass estimator (self-contained;
        ``skimage.restoration.estimate_sigma`` needs the optional
        PyWavelets dependency);
        ``"gaussian"`` uses :func:`scipy.ndimage.gaussian_filter`;
        ``"bilateral"`` uses :func:`skimage.restoration.denoise_bilateral`.

    Returns
    -------
    numpy.ndarray
        Denoised image, float32.
    """
    params = params or DenoiseParams()
    work = np.asarray(image, dtype=np.float32)
    if work.ndim != 2:
        raise ValueError(f"denoise expects a 2D image, got {work.shape}")

    if params.method == "gaussian":
        out = ndimage.gaussian_filter(work, sigma=params.sigma)
    elif params.method == "bilateral":
        out = restoration.denoise_bilateral(
            work,
            sigma_color=params.sigma * float(work.std() + _EPS),
            sigma_spatial=params.sigma,
        )
    else:  # nlm
        sigma_est = _estimate_noise(work)
        h = params.h if params.h is not None else max(0.8 * sigma_est, _EPS)
        out = restoration.denoise_nl_means(
            work,
            h=h,
            sigma=sigma_est,
            patch_size=params.patch_size,
            patch_distance=params.patch_distance,
            fast_mode=True,
            preserve_range=True,
        )
    logger.debug("denoise: method=%s", params.method)
    return np.asarray(out, dtype=np.float32)
