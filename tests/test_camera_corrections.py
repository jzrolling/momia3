"""Tests for camera dark/bias, hot-pixel and vignetting corrections."""

from __future__ import annotations

import numpy as np
import pytest

from momia3.ops.preprocess import (
    DarkCorrectionParams,
    HotPixelParams,
    VignettingParams,
    dark_correct,
    hot_pixel_correct,
    vignetting_correct,
)


def test_dark_correct_explicit_and_estimated_offset():
    yy, xx = np.mgrid[0:32, 0:32]
    image = 100.0 + 5.0 * xx / 31.0 + 25.0
    out = dark_correct(image, DarkCorrectionParams(offset=25.0), dark=None)
    assert np.isclose(float(out.mean()), 102.5, atol=0.5)
    estimated = dark_correct(image, DarkCorrectionParams(border_px=2), dark=None)
    assert estimated.mean() < image.mean()
    assert out.shape == image.shape


def test_dark_correct_accepts_dark_frame_and_stack():
    image = np.full((3, 16, 16), 50.0, dtype=np.float32)
    dark = np.full((16, 16), 7.0, dtype=np.float32)
    out = dark_correct(image, DarkCorrectionParams(offset=0.0), dark=dark)
    assert out.shape == image.shape
    assert np.allclose(out, 43.0)


def test_hot_pixel_correct_hot_and_dead():
    image = np.full((32, 32), 100.0, dtype=np.float32)
    image[5, 5] = 5000.0
    image[20, 20] = -4000.0
    out = hot_pixel_correct(image, HotPixelParams(threshold=5.0, polarity="both"))
    assert np.isclose(float(out[5, 5]), 100.0)
    assert np.isclose(float(out[20, 20]), 100.0)
    assert np.isclose(float(out[10, 10]), 100.0)


def test_hot_pixel_correct_stack_shape():
    image = np.full((2, 16, 16), 10.0, dtype=np.float32)
    image[:, 3, 3] = 999.0
    out = hot_pixel_correct(image)
    assert out.shape == image.shape
    assert np.allclose(out[:, 3, 3], 10.0)


def test_vignetting_correct_provided_gain_and_stack_estimate():
    yy, xx = np.mgrid[0:64, 0:64]
    gain = 0.4 + 0.6 * (1.0 - ((xx - 32) ** 2 + (yy - 32) ** 2) / (32**2 + 32**2))
    image = (100.0 * gain).astype(np.float32)
    corrected = vignetting_correct(image, gain=gain)
    assert corrected.shape == image.shape
    assert corrected.std() < image.std()
    stack = np.stack([image, image, image])
    estimated = vignetting_correct(stack, VignettingParams(smooth_sigma=3.0))
    assert estimated.shape == stack.shape
    assert estimated[0].std() < image.std()


def test_vignetting_single_frame_requires_gain():
    with pytest.raises(ValueError, match="ambiguous"):
        vignetting_correct(np.ones((16, 16), dtype=np.float32))
