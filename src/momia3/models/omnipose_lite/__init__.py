"""Lightweight Omnipose inference backend (optional torch dependency).

Usage::

    from momia3.models.omnipose_lite import FastOmnipose
    fast = FastOmnipose(device="mps")  # weights resolved automatically
    masks = fast.segment(image)

The Omnipose checkpoint is not bundled; see
momia3.models.omnipose_lite.weights.resolve_omnipose_weights.
"""

from __future__ import annotations

from momia3.models.omnipose_lite.pipeline import (
    CPnetRunner,
    FastOmnipose,
    StageTimings,
    cleanup_fast,
    div_rescale_gpu,
    euler_final_coords,
    get_masks_fast,
    img_to_gpu_channel,
    normalize_channel_gpu,
)
from momia3.models.omnipose_lite.weights import (
    DEFAULT_MODEL,
    ENV_VAR,
    resolve_omnipose_weights,
)

OmniposeLite = FastOmnipose

__all__ = [
    "CPnetRunner",
    "DEFAULT_MODEL",
    "ENV_VAR",
    "FastOmnipose",
    "OmniposeLite",
    "StageTimings",
    "cleanup_fast",
    "div_rescale_gpu",
    "euler_final_coords",
    "get_masks_fast",
    "img_to_gpu_channel",
    "normalize_channel_gpu",
    "resolve_omnipose_weights",
]
