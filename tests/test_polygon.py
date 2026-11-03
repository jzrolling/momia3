"""Tests for momia3.ops.polygon (inside_polygon numpy/PIL backends)."""

from __future__ import annotations

import numpy as np
import pytest

from momia3.ops import polygon


SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


def test_inside_polygon_numpy_explicit_points() -> None:
    pts = np.array([[5.0, 5.0], [-1.0, 5.0], [11.0, 5.0], [9.9, 9.9]])
    mask = polygon.inside_polygon(SQUARE, pts)
    assert mask.dtype == bool
    assert mask.tolist() == [True, False, False, True]


def test_inside_polygon_requires_points_or_grid() -> None:
    with pytest.raises(ValueError):
        polygon.inside_polygon(SQUARE)


def test_inside_polygon_rejects_mixed_modes() -> None:
    with pytest.raises(ValueError):
        polygon.inside_polygon(SQUARE, np.array([[1.0, 1.0]]), nx=10, ny=10)


def test_inside_polygon_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        polygon.inside_polygon(SQUARE, nx=10, ny=10, backend="magic")


def test_rasterization_backends_agree_on_grid() -> None:
    nx = ny = 20
    scale = 2.0
    pil = polygon.inside_polygon(
        SQUARE, x0=0.0, y0=0.0, nx=nx, ny=ny, scale=scale, backend="pil"
    )
    np_mask = polygon.inside_polygon(
        SQUARE, x0=0.0, y0=0.0, nx=nx, ny=ny, scale=scale, backend="numpy"
    )
    assert pil.shape == np_mask.shape == (ny, nx)
    inter = np.logical_and(pil, np_mask).sum()
    union = np.logical_or(pil, np_mask).sum()
    assert inter / union > 0.95


def test_rasterization_falls_back_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> np.ndarray:
        raise RuntimeError("PIL unavailable")

    monkeypatch.setattr(polygon, "inside_polygon_pil", _boom)
    mask = polygon.inside_polygon(
        SQUARE, x0=0.0, y0=0.0, nx=12, ny=12, scale=1.0, backend="auto"
    )
    assert mask.shape == (12, 12)
    assert mask[5, 5]


def test_grid_points_order_and_centres() -> None:
    pts = polygon.polygon_grid_points(1.0, 2.0, 3, 2, 1.0)
    assert pts.shape == (6, 2)
    assert np.allclose(pts[0], [1.5, 2.5])
    assert np.allclose(pts[2], [3.5, 2.5])
    assert np.allclose(pts[3], [1.5, 3.5])
