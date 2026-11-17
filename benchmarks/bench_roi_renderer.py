"""Benchmark the custom columnar/VisPy ROI overlay.

Run with a native Qt binding:

    env -u QT_QPA_PLATFORM PYTHONPATH=src .venv/bin/python benchmarks/bench_roi_renderer.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import QApplication
from vispy import scene

from momia3.gui.roi_overlay import ROIVispyOverlay
from momia3.roi_geometry import ROIBuffer, ROIRecord

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "roi_renderer_results.json"
SHOT = ROOT / "docs" / "gui_mockups" / "roi-overlay-20k.png"


def _timed(fn, repeat=1):
    values = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        values.append((time.perf_counter() - start) * 1000)
    return result, values


def _mean_ms(fn, repeat=5) -> float:
    _, values = _timed(fn, repeat)
    return float(statistics.mean(values))


def make_records(n: int, seed: int = 4) -> list[ROIRecord]:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 2048, (n, 2))
    kinds = ("rectangle", "polygon", "ellipse", "polyline", "dot")
    records: list[ROIRecord] = []
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ellipse_angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    for index, center in enumerate(centers):
        kind = kinds[index % len(kinds)]
        radius = rng.uniform(4, 22)
        if kind == "dot":
            points = center[None, :]
        elif kind == "rectangle":
            x0, y0 = center - radius
            x1, y1 = center + radius
            points = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        elif kind == "ellipse":
            points = center[None, :] + np.column_stack([np.cos(ellipse_angles), np.sin(ellipse_angles)]) * np.array([radius, radius * 0.6])
        elif kind == "polygon":
            points = center[None, :] + np.column_stack([np.cos(angles), np.sin(angles)]) * rng.uniform(0.65, 1.0, (32, 1)) * radius
        else:
            x = np.linspace(-radius, radius, 32)
            points = np.column_stack([x, 0.35 * radius * np.sin(x / max(radius, 1) * 3)]) + center
        records.append(ROIRecord(f"r{index}", kind, points, frame=0))
    return records


def bench(n: int) -> dict:
    app = QApplication.instance() or QApplication([])
    records = make_records(n)
    buffer = ROIBuffer()
    _, build_ms = _timed(lambda: buffer.build(records))
    canvas = scene.SceneCanvas(keys=None, show=False, size=(1000, 800), bgcolor="#181818")
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.set_range((0, 2048), (0, 2048), margin=0)
    overlay = ROIVispyOverlay(view)
    _, update_ms = _timed(lambda: overlay.update(buffer))
    _, render_ms = _timed(canvas.render, repeat=5)
    viewport = (600, 500, 1100, 1000)
    _, query_ms = _timed(lambda: buffer.visible_mask(viewport))
    mask = buffer.visible_mask(viewport)
    _, cull_ms = _timed(lambda: overlay.update(buffer, mask))
    _, cull_render_ms = _timed(canvas.render, repeat=5)
    selected = buffer.selected_segments(range(min(1000, n)))
    _, selection_ms = _timed(lambda: overlay.set_selection(selected))
    hover_positions = np.random.default_rng(3).uniform(0, 2048, (1000, 2))
    _, hover_ms = _timed(lambda: [buffer.index.query((x - 4, y - 4, x + 4, y + 4)) for x, y in hover_positions])
    frame = canvas.render()
    if n >= 20000:
        try:
            from PIL import Image
            Image.fromarray(frame).save(SHOT)
        except Exception:
            pass
    canvas.close()
    return {
        "n": n,
        "segments": int(len(buffer.segments)),
        "dots": int(len(buffer.dot_centers)),
        "build_ms": round(build_ms[0], 1),
        "update_all_ms": round(update_ms[0], 1),
        "render_all_ms": round(statistics.mean(render_ms), 1),
        "cull_mask_ms": round(query_ms[0], 2),
        "cull_visible_rois": int(mask.sum()),
        "cull_update_ms": round(cull_ms[0], 1),
        "cull_render_ms": round(statistics.mean(cull_render_ms), 1),
        "selection_1000_ms": round(selection_ms[0], 2),
        "hover_1000_ms": round(hover_ms[0], 1),
    }


def main() -> int:
    ns = [int(arg) for arg in sys.argv[1:]] or [5000, 20000]
    results = [bench(n) for n in ns]
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
