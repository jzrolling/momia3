"""BaSiC illumination correction backed by the vendored BaSiCpy_torch.

The package is resident at momia3.vendor.basicpy_torch, but imported lazily:
momia3 core and this module import without torch. The function names keep the
one-shot fit/apply API of the original BaSiC wrapper used by the lab.

Reference: Peng, T. et al. "BaSiC: a background and shading correction method
for optical microscopy images." Nature Communications 8, 14836 (2017),
DOI 10.1038/s41467-017-01613-0. Upstream: https://github.com/peng-lab/BaSiCPy
(MIT). The vendored PyTorch adaptation and its provenance are documented in
momia3/vendor/basicpy_torch/README.md.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

__all__ = [
    "BasicPyParams",
    "basicpy_fit",
    "basicpy_transform",
    "basicpy_correct",
]


class BasicPyParams(BaseModel):
    """Parameters passed to the vendored BaSiCpy_torch BaSiC model."""

    get_darkfield: bool = Field(
        False, description="Estimate a darkfield component in addition to the flatfield"
    )
    smoothness_flatfield: float = Field(
        1.0, ge=0.0, description="Smoothness weight of the flatfield surface"
    )
    smoothness_darkfield: float = Field(
        1.0, ge=0.0, description="Smoothness weight of the darkfield surface"
    )
    sparse_cost_darkfield: float = Field(
        0.01, ge=0.0, description="Sparse L1 weight of the darkfield"
    )
    fitting_mode: Literal["ladmap", "approximate"] = Field(
        "ladmap", description="BaSiC fitting algorithm"
    )
    max_iterations: int = Field(
        500, ge=1, description="Maximum iterations of the inner optimizer"
    )
    working_size: int | None = Field(
        128, ge=16, description="Working resolution used while fitting; None keeps full size"
    )
    device: Literal["auto", "cpu", "cuda", "mps", "tpu"] = Field(
        "cpu", description="Torch device; cpu is fastest for small stacks on MPS machines"
    )


def _import_basicpy() -> Any:
    try:
        from momia3.vendor.basicpy_torch import BaSiC
    except ImportError as exc:  # pragma: no cover - depends on optional torch
        raise ImportError(
            "BaSiCpy_torch requires torch; install the momia3 gpu/inference extra "
            "or run this operator in an environment with torch"
        ) from exc
    return BaSiC


def _check_stack(stack: np.ndarray) -> np.ndarray:
    arr = np.asarray(stack)
    if arr.ndim != 3:
        raise ValueError(f"BaSiC expects a (t, y, x) stack, got {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError("BaSiC requires at least two frames to fit a shading model")
    return arr.astype(np.float32, copy=False)


def basicpy_fit(stack: np.ndarray, params: BasicPyParams | None = None) -> Any:
    """Fit a BaSiCpy_torch model on a (t, y, x) stack and return the model."""
    params = params or BasicPyParams()
    arr = _check_stack(stack)
    BaSiC = _import_basicpy()
    model = BaSiC(**params.model_dump(exclude_none=True))
    model.fit(arr)
    return model


def basicpy_transform(stack: np.ndarray, model: Any) -> np.ndarray:
    """Apply a fitted BaSiCpy_torch model to a (t, y, x) or (y, x) image."""
    arr = np.asarray(stack)
    if arr.ndim not in (2, 3):
        raise ValueError(f"BaSiC transform expects 2-D or (t, y, x), got {arr.shape}")
    out = model.transform(arr.astype(np.float32, copy=False))
    return np.asarray(out, dtype=np.float32)


def basicpy_correct(stack: np.ndarray, params: BasicPyParams | None = None) -> np.ndarray:
    """Fit and apply BaSiC on the same (t, y, x) stack."""
    params = params or BasicPyParams()
    arr = _check_stack(stack)
    return basicpy_transform(arr, basicpy_fit(arr, params))
