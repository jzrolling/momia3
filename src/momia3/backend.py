"""numpy/torch dual backend dispatch (ARCHITECTURE.md §3.3).

Rules from the spec:

- Whole-image operations (filtering, convolution, FFT phase correlation)
  run on torch on the selected device when torch is installed, and fall
  back to numpy/scipy otherwise. Both paths agree within a tolerance of
  1e-4 (enforced by the test suite).
- The module imports and works without torch: ``HAS_TORCH`` flags
  availability and every torch code path is guarded.

Boundary-condition note: the scipy paths use ``mode="mirror"`` (reflect
*excluding* the edge pixel), which is exactly torch's ``F.pad(...,
mode="reflect")`` semantics — that is what makes the two backends agree.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

try:  # optional GPU backend (§3.3: torch is the only GPU backend)
    import torch
    import torch.nn.functional as _F

    HAS_TORCH: bool = True
except ImportError:  # pragma: no cover - depends on environment
    torch = None  # type: ignore[assignment]
    _F = None  # type: ignore[assignment]
    HAS_TORCH = False

__all__ = ["HAS_TORCH", "get_device", "to_xp", "Ops"]

_EPS = 1e-12


def get_device() -> Any:
    """Best available torch device: cuda → mps → cpu.

    Returns
    -------
    torch.device or str
        A :class:`torch.device` when torch is installed; the string
        ``"cpu"`` otherwise (callers must not assume torch semantics
        without checking :data:`HAS_TORCH`).
    """
    if HAS_TORCH:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return "cpu"


def to_xp(arr: Any, device: Any = None) -> Any:
    """Convert between numpy arrays and torch tensors.

    numpy → torch uses :func:`torch.from_numpy` (zero-copy on CPU) followed
    by an optional device transfer; torch → numpy detaches and moves back
    to CPU (zero-copy for CPU tensors). Without torch installed, numpy
    input is returned unchanged.

    Parameters
    ----------
    arr : numpy.ndarray or torch.Tensor
        Input array.
    device : torch.device or str, optional
        Target device for numpy → torch conversion; ``None`` keeps CPU.

    Returns
    -------
    torch.Tensor or numpy.ndarray
    """
    if HAS_TORCH and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    if HAS_TORCH:
        tensor = torch.from_numpy(np.ascontiguousarray(arr))
        return tensor.to(device) if device is not None else tensor
    logger.debug("torch unavailable; to_xp returning numpy input")
    return np.asarray(arr)


def _gaussian_kernel1d(sigma: float, truncate: float) -> np.ndarray:
    """Normalized 1D gaussian kernel matching ``scipy.ndimage``'s definition."""
    radius = int(truncate * float(sigma) + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    return kernel / kernel.sum()


class Ops:
    """Unified entry point for device-accelerated whole-image ops (§3.3).

    Parameters
    ----------
    device : torch.device or str, optional
        Compute device; defaults to :func:`get_device`.
    prefer_torch : bool
        When True (default) and torch is installed, ops run on torch;
        otherwise they run on numpy/scipy. Exposed mainly so tests can
        force both backends and compare.

    Notes
    -----
    All ops are 2D whole-image operations and return float32 arrays.
    """

    def __init__(self, device: Any = None, *, prefer_torch: bool = True) -> None:
        self.device = device if device is not None else get_device()
        self.use_torch: bool = bool(prefer_torch and HAS_TORCH)
        if prefer_torch and not HAS_TORCH:
            logger.debug("torch not installed; Ops using numpy/scipy fallback")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_2d(name: str, arr: np.ndarray) -> None:
        if arr.ndim != 2:
            raise ValueError(f"{name} expects a 2D image, got shape {arr.shape}")

    def _tensor(self, arr: np.ndarray) -> Any:
        return torch.as_tensor(
            np.ascontiguousarray(arr, dtype=np.float32), device=self.device
        )

    @staticmethod
    def _numpy(tensor: Any) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # gaussian filter
    # ------------------------------------------------------------------
    def gaussian_filter(
        self, arr: np.ndarray, sigma: float, truncate: float = 4.0
    ) -> np.ndarray:
        """2D gaussian blur, scipy-compatible within 1e-4.

        Parameters
        ----------
        arr : numpy.ndarray
            2D image.
        sigma : float
            Gaussian standard deviation in pixels (same for both axes).
        truncate : float
            Kernel radius in units of ``sigma`` (scipy convention).

        Returns
        -------
        numpy.ndarray
            Filtered image, float32, same shape as ``arr``. Boundary mode
            is "mirror" (reflect excluding the edge pixel) in both
            backends.
        """
        arr = np.asarray(arr)
        self._check_2d("gaussian_filter", arr)
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if self.use_torch:
            kernel = torch.as_tensor(
                _gaussian_kernel1d(sigma, truncate),
                dtype=torch.float32,
                device=self.device,
            )
            pad = kernel.numel() // 2
            t = self._tensor(arr).reshape(1, 1, *arr.shape)
            t = _F.pad(t, (pad, pad, 0, 0), mode="reflect")
            t = _F.conv2d(t, kernel.view(1, 1, 1, -1))
            t = _F.pad(t, (0, 0, pad, pad), mode="reflect")
            t = _F.conv2d(t, kernel.view(1, 1, -1, 1))
            return self._numpy(t.reshape(arr.shape))
        blurred = ndimage.gaussian_filter(
            arr.astype(np.float32, copy=False),
            sigma=sigma,
            mode="mirror",
            truncate=truncate,
        )
        return blurred.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # convolution
    # ------------------------------------------------------------------
    def conv2d(self, arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """True 2D convolution with mirror boundary conditions.

        Parameters
        ----------
        arr : numpy.ndarray
            2D image.
        kernel : numpy.ndarray
            2D convolution kernel with odd side lengths (centered origin).

        Returns
        -------
        numpy.ndarray
            Convolved image, float32, same shape as ``arr``.
        """
        arr = np.asarray(arr)
        kernel = np.asarray(kernel)
        self._check_2d("conv2d", arr)
        self._check_2d("conv2d kernel", kernel)
        kh, kw = kernel.shape
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError(f"kernel sides must be odd, got {kernel.shape}")
        if self.use_torch:
            ph, pw = kh // 2, kw // 2
            # torch conv2d computes cross-correlation; flip for convolution.
            k = torch.as_tensor(
                np.ascontiguousarray(kernel[::-1, ::-1], dtype=np.float32),
                device=self.device,
            )
            t = self._tensor(arr).reshape(1, 1, *arr.shape)
            t = _F.pad(t, (pw, pw, ph, ph), mode="reflect")
            t = _F.conv2d(t, k.view(1, 1, kh, kw))
            return self._numpy(t.reshape(arr.shape))
        convolved = ndimage.convolve(
            arr.astype(np.float32, copy=False),
            kernel.astype(np.float32, copy=False),
            mode="mirror",
        )
        return convolved.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # FFT phase correlation + fourier shift
    # ------------------------------------------------------------------
    def fft_shift_phase_corr(
        self, ref: np.ndarray, mov: np.ndarray
    ) -> tuple[float, float]:
        """Integer-pixel translation of ``mov`` relative to ``ref``.

        Uses Fourier phase correlation of the normalized cross-power
        spectrum ``F(ref) * conj(F(mov))`` (same convention as
        ``skimage.registration.phase_cross_correlation``): the returned
        ``(dy, dx)`` is the *correction* such that ``shift(mov, (dy, dx))``
        aligns with ``ref`` (numpy ``np.roll`` convention). Subpixel
        refinement is left to the preprocess ops (Phase 3).

        Parameters
        ----------
        ref, mov : numpy.ndarray
            2D images of identical shape.

        Returns
        -------
        tuple of float
            ``(dy, dx)`` translation in pixels.
        """
        ref = np.asarray(ref)
        mov = np.asarray(mov)
        self._check_2d("fft_shift_phase_corr", ref)
        self._check_2d("fft_shift_phase_corr", mov)
        if ref.shape != mov.shape:
            raise ValueError(f"shape mismatch: ref {ref.shape} vs mov {mov.shape}")
        if self.use_torch:
            f_ref = torch.fft.fft2(self._tensor(ref))
            f_mov = torch.fft.fft2(self._tensor(mov))
            cross = f_ref * torch.conj(f_mov)
            cross = cross / torch.clamp(torch.abs(cross), min=_EPS)
            corr = torch.fft.ifft2(cross).real
            peak = int(torch.argmax(corr.flatten()).item())
        else:
            f_ref = np.fft.fft2(ref.astype(np.float32, copy=False))
            f_mov = np.fft.fft2(mov.astype(np.float32, copy=False))
            cross = f_ref * np.conj(f_mov)
            cross /= np.maximum(np.abs(cross), _EPS)
            corr = np.fft.ifft2(cross).real
            peak = int(np.argmax(corr))
        height, width = ref.shape
        dy, dx = divmod(peak, width)
        if dy > height // 2:
            dy -= height
        if dx > width // 2:
            dx -= width
        return float(dy), float(dx)

    def fourier_shift(
        self, arr: np.ndarray, shift: tuple[float, float]
    ) -> np.ndarray:
        """Translate a 2D image by ``(dy, dx)`` via the Fourier shift theorem.

        Supports subpixel shifts; wraps around at the borders (like
        ``np.roll``). Used by drift correction (§4.1): one reference-channel
        shift, applied to all channels.

        Parameters
        ----------
        arr : numpy.ndarray
            2D image.
        shift : tuple of float
            ``(dy, dx)`` in pixels.

        Returns
        -------
        numpy.ndarray
            Shifted image, float32, same shape.
        """
        arr = np.asarray(arr)
        self._check_2d("fourier_shift", arr)
        dy, dx = float(shift[0]), float(shift[1])
        height, width = arr.shape
        if self.use_torch:
            fy = torch.fft.fftfreq(height, device=self.device).view(-1, 1)
            fx = torch.fft.fftfreq(width, device=self.device).view(1, -1)
            phase = torch.exp(-2j * np.pi * (fy * dy + fx * dx))
            shifted = torch.fft.ifft2(torch.fft.fft2(self._tensor(arr)) * phase)
            return self._numpy(shifted.real)
        fy = np.fft.fftfreq(height).reshape(-1, 1)
        fx = np.fft.fftfreq(width).reshape(1, -1)
        phase = np.exp(-2j * np.pi * (fy * dy + fx * dx))
        shifted = np.fft.ifft2(np.fft.fft2(arr.astype(np.float32)) * phase)
        return shifted.real.astype(np.float32, copy=False)
