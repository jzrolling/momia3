"""Tests for momia3.ops.preprocess (§4.1)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from momia3.ops import preprocess as pp


def _structured_image(seed: int = 0, shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Smooth random image with a well-defined phase-correlation peak."""
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.random(shape), 4.0).astype(np.float32)


def _fshift(image: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    """Subpixel shift via scipy's Fourier-domain fourier_shift."""
    return np.fft.ifft2(
        ndimage.fourier_shift(np.fft.fft2(image), shift)
    ).real.astype(np.float32)


# ---------------------------------------------------------------------------
# drift_correct
# ---------------------------------------------------------------------------


def test_drift_correct_recovers_synthetic_shift() -> None:
    img = _structured_image()
    true_shift = (3.5, -2.25)
    moved = _fshift(img, true_shift)
    stack = np.stack([img, moved])

    corrected, shifts = pp.drift_correct(stack, pp.DriftParams(upsample_factor=10))

    assert corrected.shape == stack.shape
    assert shifts.shape == (2, 2)
    assert np.allclose(shifts[0], 0.0)
    # Estimated cumulative shift ≈ -true motion.
    assert np.abs(shifts[1] - (-np.asarray(true_shift))).max() < 0.5
    # Corrected frames re-align to well below 0.5 px.
    residual, _, _ = phase_cross_correlation(
        corrected[0], corrected[1], upsample_factor=10
    )
    assert np.abs(residual).max() < 0.5


def test_drift_correct_multichannel_shared_shift() -> None:
    img_a = _structured_image(seed=1)
    img_b = _structured_image(seed=2)
    true_shift = (-2.0, 4.5)
    stack_a = np.stack([img_a, _fshift(img_a, true_shift)])
    stack_b = np.stack([img_b, _fshift(img_b, true_shift)])

    corrected, shifts = pp.drift_correct(
        {"ch_a": stack_a, "ch_b": stack_b},
        pp.DriftParams(reference_channel="ch_a"),
    )

    assert set(corrected) == {"ch_a", "ch_b"}
    for channel in ("ch_a", "ch_b"):
        residual, _, _ = phase_cross_correlation(
            corrected[channel][0], corrected[channel][1], upsample_factor=10
        )
        assert np.abs(residual).max() < 0.5, channel
    # Reference-channel selection is honored.
    np.testing.assert_allclose(shifts[1], -np.asarray(true_shift), atol=0.5)


def test_drift_correct_three_frames_cumulative() -> None:
    img = _structured_image(seed=3)
    stack = np.stack(
        [img, _fshift(img, (1.0, 1.0)), _fshift(img, (2.0, 2.0))]
    )
    corrected, shifts = pp.drift_correct(stack)
    np.testing.assert_allclose(shifts[:, 0], [0.0, -1.0, -2.0], atol=0.3)
    for t in (1, 2):
        residual, _, _ = phase_cross_correlation(
            corrected[0], corrected[t], upsample_factor=10
        )
        assert np.abs(residual).max() < 0.5


# ---------------------------------------------------------------------------
# rolling_ball_background_fft
# ---------------------------------------------------------------------------


def test_rolling_ball_removes_smooth_gradient() -> None:
    yy, xx = np.mgrid[0:128, 0:128]
    background = 0.6 + 0.25 * (xx / 128.0) + 0.15 * np.sin(yy / 128.0 * np.pi)
    cells = np.zeros((128, 128), dtype=np.float32)
    cells[40:50, 40:60] = -0.4
    cells[84:96, 88:100] = -0.3
    image = (background + cells).astype(np.float32)

    # shrink_factor=None: contrast preservation is an exact-path property
    # (the production shrink path trades it for speed, like momia2's).
    out = pp.rolling_ball_background_fft(
        image, pp.RollingBallParams(radius=15, shrink_factor=None)
    )

    assert out.dtype == np.float32
    assert out.shape == image.shape
    # Interior background is flattened (borders excluded: circular FFT conv;
    # a dilation margin excludes the halo the kernel leaves around cells).
    inner = np.s_[16:-16, 16:-16]
    halo = ndimage.binary_dilation(cells < 0, iterations=18)
    bg_region = ~(cells[inner] < 0) & ~halo[inner]
    before = float(image[inner][bg_region].std())
    after = float(out[inner][bg_region].std())
    assert after < before * 0.35
    # Cell contrast is preserved.
    assert float(out[45, 50]) < -0.2
    assert float(out[90, 94]) < -0.15


def test_rolling_ball_stack_and_ball_kernel() -> None:
    image = _structured_image(seed=4)
    stack = np.stack([image, image])
    out = pp.rolling_ball_background_fft(
        stack, pp.RollingBallParams(radius=20, kind="ball", shrink_factor=None)
    )
    assert out.shape == stack.shape
    single = pp.rolling_ball_background_fft(
        image, pp.RollingBallParams(radius=20, kind="ball", shrink_factor=None)
    )
    np.testing.assert_allclose(out[0], single, rtol=1e-5, atol=1e-6)


def test_shrink_img_local_min_values_and_remainder() -> None:
    rng = np.random.default_rng(11)
    image = rng.integers(0, 1000, size=(18, 22)).astype(np.float32)

    shrunk = pp.shrink_img_local_min(image, 4)

    # Trailing remainder rows/cols are dropped (momia2 semantics).
    assert shrunk.shape == (4, 5)
    # Block-min values match a manual reference.
    ref = np.empty((4, 5), dtype=np.float32)
    for bi in range(4):
        for bj in range(5):
            ref[bi, bj] = image[bi * 4 : (bi + 1) * 4, bj * 4 : (bj + 1) * 4].min()
    np.testing.assert_array_equal(shrunk, ref)
    # The numba hot-path twin produces identical values.
    twin = np.empty_like(ref)
    pp._shrink_min_nb(np.ascontiguousarray(image), 4, twin)
    np.testing.assert_array_equal(twin, ref)

    with pytest.raises(ValueError, match="2D"):
        pp.shrink_img_local_min(np.zeros((4, 4, 4), dtype=np.float32), 2)
    with pytest.raises(ValueError, match=">= 1"):
        pp.shrink_img_local_min(image, 0)
    with pytest.raises(ValueError, match="too large"):
        pp.shrink_img_local_min(image, 32)


def test_rolling_ball_shrink_none_and_one_are_exact_path() -> None:
    image = _structured_image(seed=9)
    exact_none = pp.rolling_ball_background_fft(
        image, pp.RollingBallParams(radius=15, shrink_factor=None)
    )
    exact_one = pp.rolling_ball_background_fft(
        image, pp.RollingBallParams(radius=15, shrink_factor=1)
    )
    np.testing.assert_array_equal(exact_none, exact_one)
    # The default production path (shrink=4) differs from the exact path.
    fast = pp.rolling_ball_background_fft(image, pp.RollingBallParams(radius=15))
    assert not np.array_equal(fast, exact_none)


def test_rolling_ball_shrink_fidelity_vs_exact() -> None:
    """Production shrink path tracks the exact estimator (fidelity contract).

    Phantom mirrors the benchmark regime (smooth shading + thin bright
    rods + noise, radius 50). Compared in background space: full-frame
    Pearson >= 0.995 and interior (one kernel radius in) >= 0.999. The
    border band is where the FFT estimator itself is zero-padding biased
    (exact-vs-truth is only ~0.87 full-frame; both paths recover to
    ~0.9997 at a one-radius margin), so the strict bound lives interior.
    """
    size, radius = 512, 50
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    image = 0.55 + 0.25 * np.sin(yy / size * np.pi) * np.cos(xx / size * np.pi * 0.7)
    for _ in range(12):
        cy, cx = rng.uniform(0.15, 0.85, 2) * size
        ang = rng.uniform(0, np.pi)
        wy, wx = np.cos(ang), np.sin(ang)
        along = (yy - cy) * wy + (xx - cx) * wx
        perp = -(yy - cy) * wx + (xx - cx) * wy
        image[(np.abs(along) < rng.uniform(14, 30) / 2) & (np.abs(perp) < 2.2)] += 0.45
    image += rng.normal(0, 0.05, image.shape)
    image = np.clip(image, 0, 1).astype(np.float32)

    exact = pp.rolling_ball_background_fft(
        image, pp.RollingBallParams(radius=radius, shrink_factor=None)
    )
    fast = pp.rolling_ball_background_fft(image, pp.RollingBallParams(radius=radius))

    assert fast.dtype == np.float32
    assert fast.shape == image.shape
    bg_exact = image - exact
    bg_fast = image - fast

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    assert _pearson(bg_fast, bg_exact) >= 0.995
    inner = np.s_[radius:-radius, radius:-radius]
    assert _pearson(bg_fast[inner], bg_exact[inner]) >= 0.999
    # Stack input goes through the same shrunk estimator per frame.
    stack_fast = pp.rolling_ball_background_fft(
        np.stack([image, image]), pp.RollingBallParams(radius=radius)
    )
    np.testing.assert_array_equal(stack_fast[0], fast)


# ---------------------------------------------------------------------------
# bleach_correct
# ---------------------------------------------------------------------------


def test_bleach_correct_flattens_single_exponential() -> None:
    base = _structured_image(seed=5, shape=(64, 64)) + 0.5
    t = np.arange(12, dtype=np.float64)
    decay = 2.0 * np.exp(-t / 6.0)
    stack = (base[None] * decay[:, None, None]).astype(np.float32)

    corrected = pp.bleach_correct(stack, pp.BleachParams(order=1))

    means = corrected.reshape(12, -1).mean(axis=1)
    slope = float(np.polyfit(t, means, 1)[0])
    assert abs(slope) / float(means.mean()) < 0.01
    # Frame-0 level is preserved.
    np.testing.assert_allclose(corrected[0], stack[0], rtol=1e-5, atol=1e-6)


def test_bleach_correct_double_exponential() -> None:
    base = _structured_image(seed=6, shape=(64, 64)) + 0.5
    t = np.arange(16, dtype=np.float64)
    decay = 1.2 * np.exp(-t / 2.5) + 0.8 * np.exp(-t / 40.0)
    stack = (base[None] * decay[:, None, None]).astype(np.float32)

    corrected = pp.bleach_correct(stack, pp.BleachParams(order=2))

    means = corrected.reshape(16, -1).mean(axis=1)
    cv = float(means.std() / means.mean())
    assert cv < 0.02


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_normalize_dtype_and_range() -> None:
    rng = np.random.default_rng(7)
    image = rng.uniform(0, 100, (64, 64)).astype(np.float32)

    out16 = pp.normalize(
        image,
        pp.NormalizeParams(
            lower_percentile=0.0, upper_percentile=100.0, dtype="uint16"
        ),
    )
    assert out16.dtype == np.uint16
    assert int(out16.min()) >= 0
    assert int(out16.max()) <= 65535
    assert int(out16.max()) == 65535  # exact percentile bounds hit the top

    out8 = pp.normalize(
        image,
        pp.NormalizeParams(lower_percentile=0.0, upper_percentile=100.0, dtype="uint8"),
    )
    assert out8.dtype == np.uint8
    assert int(out8.max()) == 255

    outf = pp.normalize(image, pp.NormalizeParams(dtype="float32"))
    assert outf.dtype == np.float32
    assert 0.0 <= float(outf.min()) <= float(outf.max()) <= 1.0


def test_normalize_gamma_and_bounds() -> None:
    image = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
    lin = pp.normalize(
        image, pp.NormalizeParams(lower_percentile=0, upper_percentile=100, gamma=1.0)
    )
    gam = pp.normalize(
        image, pp.NormalizeParams(lower_percentile=0, upper_percentile=100, gamma=2.0)
    )
    np.testing.assert_allclose(gam, lin**2, rtol=1e-5, atol=1e-6)

    constant = np.full((8, 8), 3.0, dtype=np.float32)
    out = pp.normalize(constant)
    assert np.all(out == 0.0)

    with pytest.raises(ValueError, match="upper_percentile"):
        pp.normalize(image, pp.NormalizeParams(lower_percentile=90, upper_percentile=10))


# ---------------------------------------------------------------------------
# illumination_correct
# ---------------------------------------------------------------------------


def test_illumination_correct_reduces_vignetting() -> None:
    """Radial vignette + moving dark cell sweeping the field.

    The temporal median captures the vignetted background as darkfield;
    the mean of the median-subtracted stack retains the vignette-shaped
    average cell coverage, so the smoothed flatfield divides the vignette
    out of the cell response.
    """
    n_frames, size = 16, 96
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (size / 2)
    vignette = (1.0 - 0.55 * np.clip(radius, 0, 1) ** 2).astype(np.float32)

    rng = np.random.default_rng(8)
    stack = np.empty((n_frames, size, size), dtype=np.float32)
    cell_centers: list[tuple[int, int]] = []
    grid = np.linspace(12, size - 12, 4)
    for t in range(n_frames):
        cy, cx = int(grid[t // 4]), int(grid[t % 4])
        cell_centers.append((cy, cx))
        signal = np.full((size, size), 0.8, dtype=np.float32)
        cell = (yy - cy) ** 2 + (xx - cx) ** 2 <= 6.0**2
        signal[cell] = 0.3  # dark cell, depth 0.5
        frame = vignette * signal
        frame += rng.normal(0.0, 0.005, (size, size)).astype(np.float32)
        stack[t] = frame

    corrected = pp.illumination_correct(
        stack, pp.IlluminationParams(smooth_sigma=8.0, clip_min=None)
    )

    assert corrected.shape == stack.shape
    assert corrected.dtype == np.float32
    assert np.isfinite(corrected).all()

    # Raw cell contrast (local annulus background - cell center) scales with
    # the vignette at the cell's position.
    raw_response = np.array(
        [
            float(
                np.median(stack[t][(yy - cy) ** 2 + (xx - cx) ** 2 > 100.0])
                - stack[t, cy, cx]
            )
            for t, (cy, cx) in enumerate(cell_centers)
        ]
    )
    # In the corrected stack the background sits near zero, so the cell
    # response is the absolute value at the cell center.
    corrected_response = np.abs(
        np.array([corrected[t, cy, cx] for t, (cy, cx) in enumerate(cell_centers)])
    )
    raw_cv = float(raw_response.std() / raw_response.mean())
    corrected_cv = float(corrected_response.std() / corrected_response.mean())
    assert corrected_cv < raw_cv * 0.6


def test_illumination_correct_single_frame_passthrough() -> None:
    stack = _structured_image(seed=9)[None]
    out = pp.illumination_correct(stack)
    assert out.shape == stack.shape
    np.testing.assert_allclose(out[0], stack[0], rtol=1e-6)


# ---------------------------------------------------------------------------
# denoise
# ---------------------------------------------------------------------------


def test_denoise_methods_reduce_noise() -> None:
    smooth = _structured_image(seed=10, shape=(64, 64))
    rng = np.random.default_rng(10)
    noisy = smooth + rng.normal(0.0, 0.05, smooth.shape).astype(np.float32)

    for method in ("gaussian", "bilateral", "nlm"):
        out = pp.denoise(noisy, pp.DenoiseParams(method=method))
        assert out.shape == noisy.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        assert float(np.abs(out - smooth).mean()) < float(
            np.abs(noisy - smooth).mean()
        ), method

def _background_phantom() -> np.ndarray:
    """Smooth illumination gradient + vignetting + one dark cell."""
    y, x = np.mgrid[0:128, 0:128].astype(np.float32)
    bg = 100.0 + 20.0 * np.exp(
        -(((x - 64.0) ** 2 + (y - 64.0) ** 2) / (2.0 * 45.0 ** 2))
    )
    cell = 35.0 * np.exp(
        -(((x - 40.0) ** 2) / (2.0 * 4.0 ** 2) + ((y - 64.0) ** 2) / (2.0 * 3.0 ** 2))
    )
    return (bg - cell).astype(np.float32)


def test_background_correct_methods_suppress_gradient_and_keep_cell() -> None:
    img = _background_phantom()
    for method in ("rolling_ball", "tophat", "tile_percentile", "tile_median"):
        out = pp.background_correct(
            img,
            pp.BackgroundParams(
                method=method, radius=25, block_size=16, percentile=20.0
            ),
        )
        assert out.shape == img.shape
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))
        patch = (slice(20, 45), slice(85, 110))
        assert out[patch].std() < 0.5 * img[patch].std() + 1.0
        assert out[64, 40] < out[64, 110] - 5.0
        assert out.mean() == pytest.approx(img.mean(), rel=0.15)


def test_background_correct_stack_and_mean_flag() -> None:
    img = _background_phantom()
    stack = np.stack([img, img + 5.0])
    out = pp.background_correct(
        stack, pp.BackgroundParams(method="tophat", radius=15, post_blur_sigma=0.8)
    )
    assert out.shape == stack.shape
    out_flat = pp.background_correct(
        img, pp.BackgroundParams(method="tile_median", restore_mean=False)
    )
    assert abs(float(out_flat.mean())) < 20.0


def test_background_correct_rejects_unknown_method() -> None:
    with pytest.raises(Exception):
        pp.BackgroundParams(method="not-a-method")

def _band_phantom() -> np.ndarray:
    """Large-scale illumination + one cell-scale dark blob + sub-px dust."""
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:192, 0:192].astype(np.float32)
    bg = 100.0 + 0.20 * x + 0.10 * y
    cell = 40.0 * np.exp(
        -(((x - 90.0) ** 2) / (2.0 * 6.0 ** 2) + ((y - 90.0) ** 2) / (2.0 * 3.0 ** 2))
    )
    dust = 25.0 * np.exp(
        -(((x - 150.0) ** 2 + (y - 40.0) ** 2) / (2.0 * 0.7 ** 2))
    )
    return (bg - cell + dust + rng.normal(0, 0.2, bg.shape)).astype(np.float32)


def test_dual_bandpass_suppresses_gradient_and_keeps_cell() -> None:
    img = _band_phantom()
    out = pp.dual_bandpass(
        img, pp.DualBandpassParams(min_structure_px=2.0, max_structure_px=100.0)
    )
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    patch = (slice(120, 140), slice(40, 60))
    assert out[patch].std() < 0.4 * img[patch].std() + 1.0
    # the cell-scale dark blob is still darker than its neighbourhood
    assert out[90, 90] < out[90, 130] - 5.0


def test_dual_bandpass_validation_and_stack() -> None:
    img = _band_phantom()
    with pytest.raises(ValueError):
        pp.dual_bandpass(img, pp.DualBandpassParams(min_structure_px=100.0, max_structure_px=2.0))
    stack = np.stack([img, img * 0.5])
    out = pp.dual_bandpass(
        stack, pp.DualBandpassParams(min_structure_px=2.0, max_structure_px=100.0)
    )
    assert out.shape == stack.shape


def test_background_correct_dual_bandpass_method() -> None:
    img = _band_phantom()
    out = pp.background_correct(
        img,
        pp.BackgroundParams(
            method="dual_bandpass",
            min_structure_px=2.0,
            max_structure_px=100.0,
            post_blur_sigma=0.8,
        ),
    )
    assert out.shape == img.shape
    assert np.all(np.isfinite(out))

