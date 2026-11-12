"""Tests for the vendored BaSiCpy_torch illumination correction."""

from __future__ import annotations

import numpy as np
import pytest

from momia3.ops import basic_correction as bc


def _shaded_stack(n=4, size=64):
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:size, 0:size]
    flat = 0.7 + 0.3 * (xx / (size - 1)) + 0.15 * (yy / (size - 1))
    dark = 10.0
    return np.stack(
        [flat * (100.0 + 50.0 * rng.random((size, size))) + dark for _ in range(n)]
    ).astype(np.float32)


def test_params_and_torch_free_error():
    params = bc.BasicPyParams(device="cpu", max_iterations=2)
    assert params.device == "cpu"
    with pytest.raises(ImportError, match="requires torch"):
        bc.basicpy_fit(_shaded_stack(), params)


def test_fit_apply_reduces_flatfield_gradient():
    pytest.importorskip("torch")
    pytest.importorskip("momia3.vendor.basicpy_torch")
    stack = _shaded_stack()
    params = bc.BasicPyParams(
        device="cpu",
        get_darkfield=True,
        smoothness_flatfield=1.0,
        max_iterations=50,
    )
    model = bc.basicpy_fit(stack, params)
    corrected = bc.basicpy_transform(stack, model)
    assert corrected.shape == stack.shape
    assert corrected.dtype == np.float32
    assert np.isfinite(corrected).all()
    assert corrected.std() < stack.std()
