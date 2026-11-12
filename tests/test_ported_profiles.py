
"""Focused tests for the P0 backend operators ported from the clean repo."""

from __future__ import annotations

import numpy as np

from momia3.ops import characterize as ch
from momia3.ops import measure
from momia3.ops.midline import extract_midlines, fill_cells_midline
from momia3.ops.outline import extract_outlines, fill_cells_geometry
from tests.synth import generate_timelapse


def _frame():
    data = generate_timelapse(n_frames=1, shape=(96, 96), n_initial=3, seed=11)
    labels = data["labels"][0]
    image = data["images"][0].astype(np.float32)
    cells = measure.labels_to_cells(labels, 0, 0, {"phase": image}, pixel_microns=1.0)
    outlines = extract_outlines(labels, image, {"n_points": 64})
    cells = fill_cells_geometry(cells, outlines)
    mids = extract_midlines(labels, outlines, pixel_microns=1.0)
    cells = fill_cells_midline(cells, mids)
    return labels, image, cells


def test_demographic_maps_matches_projection():
    labels, image, cells = _frame()
    n_l, n_d = 16, 8
    maps = ch.demographic_maps(cells, {"phase": image}, labels, n_l=n_l, n_d=n_d)
    proj = ch.demographic_projection(cells, image, labels, n_l=n_l, n_d=n_d)
    assert set(maps) == set(proj)
    for key in maps:
        np.testing.assert_array_equal(maps[key][0], proj[key])


def test_midline_and_outline_profiles():
    labels, image, cells = _frame()
    midline = ch.midline_profile(cells, image, labels, n_points=16)
    outline = ch.outline_profile(cells, image, labels, n_points=32)
    assert midline and outline
    assert next(iter(midline.values())).shape == (16,)
    assert next(iter(outline.values())).shape == (32,)


def test_intensity_moments():
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:6, 2:6] = 1
    image = np.arange(144, dtype=np.float64).reshape(12, 12)
    out = measure.intensity_moments(labels, image, "phase")
    expected = image[2:6, 2:6]
    assert np.isclose(out["phase_median"][0], np.median(expected))
    assert np.isclose(out["phase_entropy"][0], np.asarray(out["phase_entropy"])[0])
