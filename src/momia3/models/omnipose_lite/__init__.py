"""Lightweight Omnipose inference backend (optional torch dependency).

Usage::

    from momia3.models.omnipose_lite import FastOmnipose
    fast = FastOmnipose(device="mps")  # weights resolved automatically
    masks = fast.segment(image)

The Omnipose checkpoint is not bundled; see
momia3.models.omnipose_lite.weights.resolve_omnipose_weights.

Torch is imported lazily: importing this package and the segmenter adapter is
torch-free, only the pipeline/runner symbols require torch + cellpose_omni.
"""

from __future__ import annotations

from momia3.models.omnipose_lite.segmenter import (
    OmniposeSegmenter,
    clear_runners,
    omnipose_labels,
    resolve_device,
)
from momia3.models.omnipose_lite.weights import (
    DEFAULT_MODEL,
    ENV_VAR,
    resolve_omnipose_weights,
)

_LAZY = {
    "CPnetRunner": ("pipeline", "CPnetRunner"),
    "FastOmnipose": ("pipeline", "FastOmnipose"),
    "OmniposeLite": ("pipeline", "FastOmnipose"),
    "StageTimings": ("pipeline", "StageTimings"),
    "cleanup_fast": ("pipeline", "cleanup_fast"),
    "div_rescale_gpu": ("pipeline", "div_rescale_gpu"),
    "euler_final_coords": ("pipeline", "euler_final_coords"),
    "get_masks_fast": ("pipeline", "get_masks_fast"),
    "img_to_gpu_channel": ("pipeline", "img_to_gpu_channel"),
    "normalize_channel_gpu": ("pipeline", "normalize_channel_gpu"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    from importlib import import_module

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr)
    globals()[name] = value
    return value


__all__ = [
    "CPnetRunner",
    "DEFAULT_MODEL",
    "ENV_VAR",
    "FastOmnipose",
    "OmniposeLite",
    "OmniposeSegmenter",
    "StageTimings",
    "cleanup_fast",
    "clear_runners",
    "div_rescale_gpu",
    "euler_final_coords",
    "get_masks_fast",
    "img_to_gpu_channel",
    "normalize_channel_gpu",
    "omnipose_labels",
    "resolve_device",
    "resolve_omnipose_weights",
]
