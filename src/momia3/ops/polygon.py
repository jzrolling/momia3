"""Polygon containment and rasterization helpers.

Shared by momia3.ops.outline and momia3.ops.midline. Two explicit backends
mirror the two implementations used during development:

- numpy: exact point-in-polygon test for arbitrary query points, via
  matplotlib.path.Path.contains_points (matplotlib is imported lazily).
- pil: fast rasterization of a polygon onto a regular supersampled grid, via
  PIL.ImageDraw.polygon (50-100x faster than skimage.draw.polygon for the
  many-vertex smooth outlines produced by the outline stage).

The grid convention matches the SDF skeleton rasterizer: grid cell (row, col)
samples the original-frame point (x0 + (col + 0.5) / scale,
y0 + (row + 0.5) / scale).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "inside_polygon",
    "inside_polygon_numpy",
    "inside_polygon_pil",
    "polygon_grid_points",
    "resample_closed_polygon",
    "smooth_closed_polygon",
]


def resample_closed_polygon(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polygon to ``n`` points uniform in arc length."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or n < 2:
        raise ValueError("expected (N, 2) polygon points with N >= 2 and n >= 2")
    ring = np.vstack([points, points[:1]])
    lengths = np.hypot(np.diff(ring[:, 0]), np.diff(ring[:, 1]))
    distance = np.concatenate([[0.0], np.cumsum(lengths)])
    if distance[-1] <= 0:
        raise ValueError("polygon perimeter must be positive")
    targets = np.linspace(0.0, distance[-1], int(n), endpoint=False)
    x = np.interp(targets, distance, ring[:, 0])
    y = np.interp(targets, distance, ring[:, 1])
    return np.column_stack([x, y])


def smooth_closed_polygon(
    points: np.ndarray, smooth: float = 1.0, spacing: float = 1.0
) -> np.ndarray:
    """Smooth and arc-length resample a closed polygon with a cubic B-spline."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) >= 2 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    keep = np.concatenate([[True], np.any(np.diff(points, axis=0) != 0, axis=1)])
    points = points[keep]
    if len(points) < 3:
        raise ValueError("at least three distinct polygon points are required")
    perimeter = float(np.hypot(*np.diff(np.vstack([points, points[:1]]), axis=0).T).sum())
    n = max(16, int(np.ceil(perimeter / spacing)))
    points = resample_closed_polygon(points, n)
    if smooth <= 0:
        return points
    from scipy import interpolate

    tck, _ = interpolate.splprep(
        [points[:, 0], points[:, 1]], s=float(smooth), per=1, k=3, quiet=1
    )
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    x, y = interpolate.splev(u, tck, der=0)
    return np.column_stack([x, y])


def inside_polygon_numpy(ring: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Exact inside test of a closed polygon for arbitrary query points.

    Parameters
    ----------
    ring : (N, 2) numpy.ndarray
        Closed (x, y) polygon; the last point connects back to the first.
    points : (M, 2) numpy.ndarray
        Query points in the same coordinate frame.

    Returns
    -------
    numpy.ndarray of bool
        (M,) mask, True where the point is inside the polygon.
    """
    from matplotlib.path import Path

    ring_arr = np.asarray(ring, dtype=np.float64)
    points_arr = np.asarray(points, dtype=np.float64)
    if ring_arr.ndim != 2 or ring_arr.shape[1] != 2 or len(ring_arr) < 3:
        raise ValueError("ring must be a closed (N, 2) polygon with N >= 3")
    if points_arr.size == 0:
        return np.zeros(0, dtype=bool)
    if points_arr.ndim != 2 or points_arr.shape[1] != 2:
        raise ValueError("points must be an (M, 2) array")
    return Path(ring_arr).contains_points(points_arr)


def polygon_grid_points(
    x0: float,
    y0: float,
    nx: int,
    ny: int,
    scale: float,
) -> np.ndarray:
    """Query points (row-major) of the grid-cell centres for a rasterization."""
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number")
    gx = float(x0) + (np.arange(int(nx)) + 0.5) / float(scale)
    gy = float(y0) + (np.arange(int(ny)) + 0.5) / float(scale)
    xx, yy = np.meshgrid(gx, gy)
    return np.column_stack([xx.ravel(), yy.ravel()])


def inside_polygon_pil(
    ring: np.ndarray,
    x0: float,
    y0: float,
    nx: int,
    ny: int,
    scale: float,
) -> np.ndarray:
    """Rasterize a polygon onto a regular grid with the PIL backend.

    Parameters
    ----------
    ring : (N, 2) numpy.ndarray
        Closed (x, y) polygon in the original coordinate frame.
    x0, y0 : float
        Origin of the raster grid in the original coordinate frame.
    nx, ny : int
        Grid size in columns/rows.
    scale : float
        Supersampling factor (raster cells per original pixel).

    Returns
    -------
    numpy.ndarray of bool
        (ny, nx) mask of grid-cell centres inside the polygon.
    """
    from PIL import Image, ImageDraw

    ring_arr = np.asarray(ring, dtype=np.float64)
    if ring_arr.ndim != 2 or ring_arr.shape[1] != 2 or len(ring_arr) < 3:
        raise ValueError("ring must be a closed (N, 2) polygon with N >= 3")
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number")
    col_idx = (ring_arr[:, 0] - float(x0)) * float(scale) - 0.5
    row_idx = (ring_arr[:, 1] - float(y0)) * float(scale) - 0.5
    canvas = Image.new("L", (int(nx), int(ny)), 0)
    ImageDraw.Draw(canvas).polygon(
        list(zip(col_idx.tolist(), row_idx.tolist())), fill=1
    )
    return np.asarray(canvas, dtype=np.uint8).astype(bool)


def inside_polygon(
    ring: np.ndarray,
    points: np.ndarray | None = None,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    nx: int | None = None,
    ny: int | None = None,
    scale: float = 1.0,
    backend: str = "auto",
) -> np.ndarray:
    """Unified polygon inside test / rasterizer with explicit backends.

    Two calling modes:

    - points given: exact point-in-polygon for (M, 2) query points
      (numpy backend only).
    - grid arguments given (nx/ny and optional x0/y0/scale): rasterize the
      polygon onto the grid; backend selects "pil", "numpy" or "auto"
      (PIL first, numpy fallback).

    Returns
    -------
    numpy.ndarray of bool
        (M,) mask in explicit-points mode, (ny, nx) mask in rasterization mode.
    """
    mode = str(backend or "auto").lower()
    if mode not in ("auto", "pil", "numpy"):
        raise ValueError("backend must be 'auto', 'pil' or 'numpy'")
    if points is not None:
        if nx is not None or ny is not None:
            raise ValueError("pass either points or grid arguments, not both")
        return inside_polygon_numpy(ring, points)
    if nx is None or ny is None:
        raise ValueError("rasterization mode needs nx and ny")
    if mode in ("auto", "pil"):
        try:
            return inside_polygon_pil(ring, x0, y0, int(nx), int(ny), float(scale))
        except Exception as exc:  # pragma: no cover - backend availability
            if mode == "pil":
                raise
            logger.debug("PIL rasterization failed (%s); using numpy fallback", exc)
    grid = polygon_grid_points(x0, y0, int(nx), int(ny), float(scale))
    return inside_polygon_numpy(ring, grid).reshape(int(ny), int(nx))
