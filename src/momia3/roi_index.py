"""Spatial index for large ROI sets.

The grid stores only bounding boxes.  Exact hit tests remain type-specific
(rectangle/ellipse/dot/polygon/polyline); this index narrows them to a handful
of candidates without scanning the full ROI set.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from math import floor
from typing import TypeAlias

BBox: TypeAlias = tuple[float, float, float, float]


def _intersects(a: BBox, b: BBox) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


class ROIGridIndex:
    def __init__(self, cell_size: float = 64.0) -> None:
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.cell_size = float(cell_size)
        self._boxes: dict[Hashable, BBox] = {}
        self._cells: dict[tuple[int, int], list[Hashable]] = {}
        self._roi_cells: dict[Hashable, list[tuple[int, int]]] = {}

    def __len__(self) -> int:
        return len(self._boxes)

    def clear(self) -> None:
        self._boxes.clear()
        self._cells.clear()
        self._roi_cells.clear()

    def add(self, roi_id: Hashable, bbox: BBox) -> None:
        if roi_id in self._boxes:
            self.remove(roi_id)
        self._boxes[roi_id] = bbox
        cells = self._covered_cells(bbox)
        self._roi_cells[roi_id] = cells
        for cell in cells:
            self._cells.setdefault(cell, []).append(roi_id)

    def remove(self, roi_id: Hashable) -> None:
        if roi_id not in self._boxes:
            return
        for cell in self._roi_cells.pop(roi_id):
            bucket = self._cells.get(cell)
            if bucket is None:
                continue
            try:
                bucket.remove(roi_id)
            except ValueError:
                pass
            if not bucket:
                self._cells.pop(cell, None)
        self._boxes.pop(roi_id, None)

    def query(self, bbox: BBox) -> list[Hashable]:
        candidates: set[Hashable] = set()
        for cell in self._covered_cells(bbox):
            bucket = self._cells.get(cell)
            if bucket:
                candidates.update(bucket)
        return [roi_id for roi_id in candidates if _intersects(self._boxes[roi_id], bbox)]

    def query_viewport(self, x0: float, y0: float, x1: float, y1: float) -> list[Hashable]:
        return self.query((x0, y0, x1, y1))

    def build(self, items: Iterable[tuple[Hashable, BBox]]) -> None:
        self.clear()
        for roi_id, bbox in items:
            self.add(roi_id, bbox)

    def _covered_cells(self, bbox: BBox) -> list[tuple[int, int]]:
        size = self.cell_size
        cx0, cx1 = floor(bbox[0] / size), floor(bbox[2] / size)
        cy0, cy1 = floor(bbox[1] / size), floor(bbox[3] / size)
        return [(cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)]
