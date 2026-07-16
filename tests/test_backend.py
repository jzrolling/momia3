"""Tests for momia3.backend — numpy/torch dual backend dispatch."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from momia3 import backend

torch_required = pytest.mark.skipif(not backend.HAS_TORCH, reason="torch not installed")


def _blobs(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    rng = np.random.default_rng(42)
    image = rng.normal(0.0, 0.05, shape).astype(np.float32)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    image += np.exp(-(((yy - 32) ** 2 + (xx - 28) ** 2) / (2 * 4.0**2))).astype(
        np.float32
    )
    return image


def test_has_torch_flag() -> None:
    assert isinstance(backend.HAS_TORCH, bool)


def test_get_device() -> None:
    device = backend.get_device()
    if backend.HAS_TORCH:
        assert isinstance(device, backend.torch.device)
    else:
        assert device == "cpu"


def test_gaussian_numpy_matches_scipy() -> None:
    image = _blobs()
    out = backend.Ops(prefer_torch=False).gaussian_filter(image, sigma=2.0)
    expected = ndimage.gaussian_filter(image, sigma=2.0, mode="mirror")
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


@torch_required
def test_gaussian_torch_vs_numpy() -> None:
    image = _blobs()
    numpy_out = backend.Ops(prefer_torch=False).gaussian_filter(image, sigma=2.0)
    torch_out = backend.Ops(prefer_torch=True).gaussian_filter(image, sigma=2.0)
    np.testing.assert_allclose(numpy_out, torch_out, rtol=1e-4, atol=1e-6)


@torch_required
def test_conv2d_torch_vs_numpy() -> None:
    image = _blobs()
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]])
    numpy_out = backend.Ops(prefer_torch=False).conv2d(image, kernel)
    torch_out = backend.Ops(prefer_torch=True).conv2d(image, kernel)
    np.testing.assert_allclose(numpy_out, torch_out, rtol=1e-4, atol=1e-5)


def test_conv2d_numpy_matches_scipy() -> None:
    image = _blobs()
    kernel = np.arange(9, dtype=np.float32).reshape(3, 3) - 4.0
    out = backend.Ops(prefer_torch=False).conv2d(image, kernel)
    expected = ndimage.convolve(image, kernel, mode="mirror")
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_conv2d_rejects_even_kernel() -> None:
    with pytest.raises(ValueError, match="odd"):
        backend.Ops(prefer_torch=False).conv2d(_blobs(), np.ones((2, 2)))


def test_phase_corr_numpy() -> None:
    ref = _blobs()
    mov = np.roll(ref, shift=(5, -7), axis=(0, 1))
    dy, dx = backend.Ops(prefer_torch=False).fft_shift_phase_corr(ref, mov)
    assert (dy, dx) == (-5.0, 7.0)


@torch_required
def test_phase_corr_torch() -> None:
    ref = _blobs()
    mov = np.roll(ref, shift=(5, -7), axis=(0, 1))
    dy, dx = backend.Ops(prefer_torch=True).fft_shift_phase_corr(ref, mov)
    assert (dy, dx) == (-5.0, 7.0)


def test_fourier_shift_realigns() -> None:
    ops = backend.Ops(prefer_torch=False)
    ref = _blobs()
    mov = np.roll(ref, shift=(5, -7), axis=(0, 1))
    dy, dx = ops.fft_shift_phase_corr(ref, mov)
    aligned = ops.fourier_shift(mov, (dy, dx))
    np.testing.assert_allclose(aligned, ref, atol=1e-4)


@torch_required
def test_fourier_shift_torch() -> None:
    ops = backend.Ops(prefer_torch=True)
    ref = _blobs()
    shifted = ops.fourier_shift(ref, (3.0, -4.0))
    np.testing.assert_allclose(shifted, np.roll(ref, (3, -4), axis=(0, 1)), atol=1e-4)


@torch_required
def test_to_xp_roundtrip() -> None:
    image = _blobs()
    tensor = backend.to_xp(image)
    assert isinstance(tensor, backend.torch.Tensor)
    back = backend.to_xp(tensor)
    assert isinstance(back, np.ndarray)
    np.testing.assert_array_equal(back, image)


def test_to_xp_without_torch() -> None:
    if backend.HAS_TORCH:
        pytest.skip("torch installed; numpy-only path not active")
    image = _blobs()
    out = backend.to_xp(image)
    assert isinstance(out, np.ndarray)


def test_ops_rejects_non_2d() -> None:
    ops = backend.Ops(prefer_torch=False)
    with pytest.raises(ValueError):
        ops.gaussian_filter(np.zeros((2, 8, 8), dtype=np.float32), 1.0)
    with pytest.raises(ValueError):
        ops.fourier_shift(np.zeros(8, dtype=np.float32), (1.0, 1.0))
