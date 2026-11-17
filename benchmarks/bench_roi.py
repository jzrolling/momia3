"""Benchmark spatial indexing and napari Shapes at dense ROI counts.

Run with a native Qt binding (not QT_QPA_PLATFORM=offscreen):

    env -u QT_QPA_PLATFORM PYTHONPATH=src .venv/bin/python benchmarks/bench_roi.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from momia3.roi_index import ROIGridIndex

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "roi_benchmark_results.json"


def _timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def bench_grid(n: int = 20000) -> dict:
    rng = np.random.default_rng(11)
    centers = rng.uniform(0, 2048, (n, 2))
    sizes = rng.uniform(10, 80, (n, 2))
    boxes = np.column_stack([centers - sizes / 2, centers + sizes / 2])
    ids = [f"r{i}" for i in range(n)]
    index = ROIGridIndex(cell_size=64)
    _, build_ms = _timed(lambda: index.build(zip(ids, map(tuple, boxes))))

    queries = []
    for _ in range(400):
        x0, y0 = rng.uniform(0, 1900, 2)
        queries.append((x0, y0, x0 + 256, y0 + 256))

    def brute(query):
        x0, y0, x1, y1 = query
        return [roi_id for roi_id, b in zip(ids, boxes) if b[0] <= x1 and b[2] >= x0 and b[1] <= y1 and b[3] >= y0]

    grid_times = []
    brute_times = []
    for query in queries:
        _, ms = _timed(lambda query=query: index.query(query))
        grid_times.append(ms)
        _, ms = _timed(lambda query=query: brute(query))
        brute_times.append(ms)
    return {
        "n": n,
        "build_ms": round(build_ms, 2),
        "grid_p50_ms": round(float(statistics.median(grid_times)), 3),
        "grid_p95_ms": round(_pct(grid_times, 95), 3),
        "linear_p50_ms": round(float(statistics.median(brute_times)), 3),
        "linear_p95_ms": round(_pct(brute_times, 95), 3),
        "grid_speedup_p50": round(float(statistics.median(brute_times) / max(statistics.median(grid_times), 1e-9)), 1),
    }


def _rect_coords(n: int, rng) -> np.ndarray:
    centers = rng.uniform(20, 1000, (n, 2))
    sizes = rng.uniform(4, 18, (n, 2))
    x0, y0 = centers[:, 0] - sizes[:, 0] / 2, centers[:, 1] - sizes[:, 1] / 2
    x1, y1 = centers[:, 0] + sizes[:, 0] / 2, centers[:, 1] + sizes[:, 1] / 2
    return np.stack([np.column_stack([x0, y0]), np.column_stack([x1, y0]), np.column_stack([x1, y1]), np.column_stack([x0, y1])], axis=1)


def _poly_coords(n: int, rng) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    radii = rng.uniform(3, 12, (n, 1))
    centers = rng.uniform(20, 1000, (n, 2))
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    return centers[:, None, :] + unit[None, :, :] * radii[:, :, None]


def bench_napari(ns=(5000, 20000)) -> list[dict]:
    import napari
    from qtpy.QtWidgets import QApplication
    from napari.layers import Points, Shapes

    app = QApplication.instance() or QApplication([])
    viewer = napari.Viewer(show=False)
    rng = np.random.default_rng(7)
    results = []

    for n in ns:
        rects = _rect_coords(n, rng)
        layer, construct_ms = _timed(lambda: Shapes(rects, shape_type="rectangle", edge_color="coral", face_color="transparent", name=f"rect-{n}"))
        _, select_ms = _timed(lambda: setattr(layer, "selected_data", set(range(n))))
        positions = rects[:, 0, :]
        _, hit_ms = _timed(lambda: [layer.get_value(positions[i], view_direction=None, dims_displayed=[0, 1]) for i in range(200)])
        results.append({"type": "rectangle", "n": n, "construct_ms": round(construct_ms, 1), "select_all_ms": round(select_ms, 1), "hit_200_ms": round(hit_ms, 1)})
        layer.selected_data = set()
        layer.data = []

    for n in ns:
        polys = _poly_coords(n, rng)
        layer, construct_ms = _timed(lambda: Shapes(polys, shape_type="polygon", edge_color="coral", face_color="transparent", name=f"poly-{n}"))
        _, select_ms = _timed(lambda: setattr(layer, "selected_data", set(range(n))))
        positions = polys[:, 0, :]
        _, hit_ms = _timed(lambda: [layer.get_value(positions[i], view_direction=None, dims_displayed=[0, 1]) for i in range(200)])
        results.append({"type": "polygon", "n": n, "construct_ms": round(construct_ms, 1), "select_all_ms": round(select_ms, 1), "hit_200_ms": round(hit_ms, 1)})
        layer.selected_data = set()
        layer.data = []

    n = max(ns)
    for label, coords, shape_type in (
        ("ellipse", _rect_coords(n, rng), "ellipse"),
        ("polyline", _poly_coords(n, rng), "path"),
    ):
        layer, construct_ms = _timed(lambda coords=coords, shape_type=shape_type, label=label: Shapes(coords, shape_type=shape_type, edge_color="coral", face_color="transparent", name=f"{label}-{n}"))
        _, select_ms = _timed(lambda: setattr(layer, "selected_data", set(range(n))))
        positions = coords[:, 0, :]
        _, hit_ms = _timed(lambda: [layer.get_value(positions[i], view_direction=None, dims_displayed=[0, 1]) for i in range(200)])
        results.append({"type": label, "n": n, "construct_ms": round(construct_ms, 1), "select_all_ms": round(select_ms, 1), "hit_200_ms": round(hit_ms, 1)})
        layer.selected_data = set()
        layer.data = []

    dots = rng.uniform(0, 1000, (n, 2))
    points, construct_ms = _timed(lambda: Points(dots, size=3, name=f"dots-{n}"))
    _, select_ms = _timed(lambda: setattr(points, "selected_data", set(range(n))))
    results.append({"type": "dot", "n": n, "construct_ms": round(construct_ms, 1), "select_all_ms": round(select_ms, 1), "hit_200_ms": 0.0})
    viewer.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-napari", action="store_true")
    parser.add_argument("--n", type=int, nargs="*", default=[5000, 20000])
    args = parser.parse_args()
    result = {"grid": [bench_grid(n) for n in args.n]}
    if not args.no_napari:
        result["napari"] = bench_napari(tuple(args.n))
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
