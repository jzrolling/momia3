"""Omnipose label-segmenter adapter (torch imported lazily).

The Omnipose network does not produce the 3-class probability map consumed by
momia3.ops.segment.prob_to_labels; it outputs flow/dist/boundary channels whose
post-processing (Euler integration + mask reconstruction) lives in
momia3.models.omnipose_lite.pipeline.FastOmnipose. This adapter exposes that
label-producing path to the momia3 pipeline as segment.omnipose.

Importing this module is torch-free; the first call to omnipose_labels loads
torch/cellpose_omni and caches one runner per (weights, device, half) tuple.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from momia3.models.omnipose_lite.weights import resolve_omnipose_weights

__all__ = [
    "OmniposeSegmenter",
    "clear_runners",
    "omnipose_labels",
    "resolve_device",
]

_RUNNERS: dict[tuple[str, str, bool], Any] = {}


def resolve_device(device: str | None = None) -> str:
    """Best available device: explicit string, else cuda -> mps -> cpu."""
    if device:
        return str(device)
    try:
        from momia3.backend import get_device

        return str(get_device())
    except Exception:  # pragma: no cover - backend always importable
        return "cpu"


def _get_runner(weights_path: str, device: str, half: bool) -> Any:
    key = (weights_path, device, half)
    runner = _RUNNERS.get(key)
    if runner is None:
        from momia3.models.omnipose_lite.pipeline import FastOmnipose

        runner = FastOmnipose(weights_path, device=device, half=half)
        _RUNNERS[key] = runner
    return runner


def clear_runners() -> None:
    """Drop cached runners (useful in tests or when switching devices)."""
    _RUNNERS.clear()


class OmniposeSegmenter:
    """Label-producing Omnipose segmenter for the momia3 pipeline.

    Parameters are the typed fields of
    momia3.ops.segment.OmniposeParams. A runner is loaded lazily on the first
    call and reused for subsequent frames.
    """

    def __init__(self, params: Any) -> None:
        self.weights_path = resolve_omnipose_weights(
            getattr(params, "weights_path", None)
        )
        self.device = resolve_device(getattr(params, "device", None))
        self.half = bool(getattr(params, "half", False))
        self.niter = int(getattr(params, "niter", 200))
        self.mask_threshold = float(getattr(params, "mask_threshold", 0.0))
        self.diam_threshold = float(getattr(params, "diam_threshold", 12.0))
        self.min_size = int(getattr(params, "min_size", 15))
        self.max_size = getattr(params, "max_size", None)
        self.hole_size = int(getattr(params, "hole_size", 3))
        self._runner: Any = None

    def _ensure_runner(self) -> Any:
        if self._runner is None:
            self._runner = _get_runner(self.weights_path, self.device, self.half)
        return self._runner

    def predict_labels(self, image: np.ndarray) -> np.ndarray:
        """Segment one 2D image and return uint32 instance labels."""
        runner = self._ensure_runner()
        return runner.segment(
            np.asarray(image),
            niter=self.niter,
            mask_threshold=self.mask_threshold,
            diam_threshold=self.diam_threshold,
            min_size=self.min_size,
            max_size=self.max_size,
            hole_size=self.hole_size,
        )

def omnipose_labels(image: np.ndarray, params: Any) -> np.ndarray:
    """One-shot segmentation: build/reuse the adapter and return labels."""
    return OmniposeSegmenter(params).predict_labels(np.asarray(image))
