"""Columnar ROI geometry for the custom HyperView overlay.

ROIs are kept in flat NumPy arrays so rendering never needs one Python object
per ROI.  Each cell stores its outline as segments; dots live in a separate
marker array.  Slice/visibility filtering happens by indexing the segment
owner array instead of rebuilding Python shape objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .roi_index import ROIGridIndex

CLOSED_KINDS = {"rectangle", "ellipse", "polygon"}
OPEN_KINDS = {"polyline"}
DOT_KINDS = {"dot"}
SUPPORTED_KINDS = CLOSED_KINDS | OPEN_KINDS | DOT_KINDS


@dataclass
class ROIRecord:
    roi_id: str
    kind: str
    points: np.ndarray
    bbox: tuple[float, float, float, float] | None = None
    frame: int = 0
    color: str = "#67d9c4"

    def __post_init__(self) -> None:
        self.kind = self.kind.lower()
        if self.kind not in SUPPORTED_KINDS:
            raise ValueError(f"unsupported ROI kind: {self.kind}")
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("ROI points must be an (N, 2) array")
        if len(self.points) < (1 if self.kind in DOT_KINDS else 2):
            raise ValueError(f"not enough vertices for {self.kind}")
        if self.bbox is None:
            self.bbox = (
                float(self.points[:, 0].min()),
                float(self.points[:, 1].min()),
                float(self.points[:, 0].max()),
                float(self.points[:, 1].max()),
            )


class ROIBuffer:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.kinds: list[str] = []
        self.frames: np.ndarray = np.empty(0, dtype=np.int32)
        self.colors: np.ndarray = np.empty((0, 4), dtype=np.float32)
        self.bboxes: np.ndarray = np.empty((0, 4), dtype=np.float32)
        self.segments: np.ndarray = np.empty((0, 2, 2), dtype=np.float32)
        self.segment_owners: np.ndarray = np.empty(0, dtype=np.int32)
        self.segment_offsets: np.ndarray = np.zeros(1, dtype=np.int64)
        self.fill_vertices: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.fill_faces: np.ndarray = np.empty((0, 3), dtype=np.int32)
        self.fill_face_owners: np.ndarray = np.empty(0, dtype=np.int32)
        self.dot_centers: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.dot_owners: np.ndarray = np.empty(0, dtype=np.int32)
        self.index = ROIGridIndex(cell_size=64.0)

    def __len__(self) -> int:
        return len(self.ids)

    def build(self, records: Iterable[ROIRecord]) -> "ROIBuffer":
        ids: list[str] = []
        kinds: list[str] = []
        frames: list[int] = []
        colors: list[np.ndarray] = []
        bboxes: list[tuple[float, float, float, float]] = []
        parts: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        fill_vertices: list[np.ndarray] = []
        fill_faces: list[np.ndarray] = []
        fill_owners: list[np.ndarray] = []
        fill_offset = 0
        dot_centers: list[np.ndarray] = []
        dot_owners: list[int] = []
        offset = 0
        offsets = [0]
        for index, record in enumerate(records):
            ids.append(record.roi_id)
            kinds.append(record.kind)
            frames.append(int(record.frame))
            colors.append(_parse_color(record.color))
            bboxes.append(record.bbox)
            if record.kind in DOT_KINDS:
                dot_centers.append(record.points[0])
                dot_owners.append(index)
            else:
                points = record.points
                if record.kind in CLOSED_KINDS:
                    points = np.vstack([points, points[:1]])
                edges = np.stack([points[:-1], points[1:]], axis=1).astype(np.float32, copy=False)
                parts.append(edges)
                owners.append(np.full(len(edges), index, dtype=np.int32))
                offset += len(edges)
                if record.kind in CLOSED_KINDS and len(points) >= 3:
                    vertices, faces = _triangulate(points, record.kind)
                    if len(faces):
                        fill_vertices.append(vertices)
                        fill_faces.append(faces + fill_offset)
                        fill_owners.append(np.full(len(faces), index, dtype=np.int32))
                        fill_offset += len(vertices)
            offsets.append(offset)
        self.ids = ids
        self.kinds = kinds
        self.frames = np.asarray(frames, dtype=np.int32)
        self.colors = np.asarray(colors, dtype=np.float32) if colors else np.empty((0, 4), dtype=np.float32)
        self.bboxes = np.asarray(bboxes, dtype=np.float32)
        self.segments = np.concatenate(parts, axis=0) if parts else np.empty((0, 2, 2), dtype=np.float32)
        self.segment_owners = np.concatenate(owners, axis=0) if owners else np.empty(0, dtype=np.int32)
        self.segment_offsets = np.asarray(offsets, dtype=np.int64)
        self.fill_vertices = np.concatenate(fill_vertices, axis=0) if fill_vertices else np.empty((0, 2), dtype=np.float32)
        self.fill_faces = np.concatenate(fill_faces, axis=0) if fill_faces else np.empty((0, 3), dtype=np.int32)
        self.fill_face_owners = np.concatenate(fill_owners, axis=0) if fill_owners else np.empty(0, dtype=np.int32)
        self.dot_centers = np.asarray(dot_centers, dtype=np.float32) if dot_centers else np.empty((0, 2), dtype=np.float32)
        self.dot_owners = np.asarray(dot_owners, dtype=np.int32)
        self.index.build(zip(ids, bboxes))
        return self

    def visible_mask(self, bbox: tuple[float, float, float, float], frame: int | None = None) -> np.ndarray:
        ids = set(self.index.query(bbox))
        mask = np.fromiter((roi_id in ids for roi_id in self.ids), dtype=bool, count=len(self.ids))
        if frame is not None:
            mask &= self.frames == frame
        return mask

    def segments_for(self, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            return self.segments
        owner_mask = mask[self.segment_owners]
        return self.segments[owner_mask]

    def dots_for(self, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            return self.dot_centers
        return self.dot_centers[mask[self.dot_owners]]

    def selected_segments(self, indices: Iterable[int]) -> np.ndarray:
        mask = np.zeros(len(self), dtype=bool)
        mask[list(indices)] = True
        return self.segments_for(mask)


def _triangulate(points: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 3:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    # Centroid fan is linear-time and sufficient for the cell-like outlines
    # used here; avoids per-ROI constrained-Delaunay overhead at 20k scale.
    center = points.mean(axis=0, keepdims=True)
    vertices = np.vstack([points, center]).astype(np.float32, copy=False)
    n = len(points)
    faces = np.array([[i, (i + 1) % n, n] for i in range(n)], dtype=np.int32)
    return vertices, faces


def _parse_color(value: str | tuple[float, ...]) -> np.ndarray:
    if isinstance(value, (tuple, list)):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if len(arr) == 4:
            return arr
        if len(arr) == 3:
            return np.array([arr[0], arr[1], arr[2], 1.0], dtype=np.float32)
    text = str(value).lstrip("#")
    try:
        if len(text) == 8:
            return np.array([int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16), int(text[6:8], 16)], dtype=np.float32) / 255.0
        if len(text) == 6:
            return np.array([int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16), 255], dtype=np.float32) / 255.0
    except ValueError:
        pass
    return np.array([0.40, 0.85, 0.77, 1.0], dtype=np.float32)


def synthetic_rois(width: float, height: float, count: int = 5000, seed: int = 7) -> list[ROIRecord]:
    """Mixed five-shape ROI set for the HyperView demo overlay."""
    rng = np.random.default_rng(seed)
    kinds = ("rectangle", "polygon", "ellipse", "polyline", "dot")
    records: list[ROIRecord] = []
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    ellipse_angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    for index in range(count):
        kind = kinds[index % len(kinds)]
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        radius = rng.uniform(2.5, 8.0)
        if kind == "rectangle":
            points = np.array([[cx - radius, cy - radius], [cx + radius, cy - radius], [cx + radius, cy + radius], [cx - radius, cy + radius]], dtype=np.float32)
        elif kind == "ellipse":
            points = np.column_stack([cx + radius * np.cos(ellipse_angles), cy + 0.7 * radius * np.sin(ellipse_angles)]).astype(np.float32)
        elif kind == "polygon":
            radial = rng.uniform(0.65, 1.0, len(angles))
            points = np.column_stack([cx + radius * radial * np.cos(angles), cy + radius * radial * np.sin(angles)]).astype(np.float32)
        elif kind == "polyline":
            x = np.linspace(cx - radius, cx + radius, 9)
            points = np.column_stack([x, cy + 0.3 * radius * np.sin((x - cx) / max(radius, 1) * 3)]).astype(np.float32)
        else:
            points = np.array([[cx, cy]], dtype=np.float32)
        records.append(ROIRecord(f"demo-{index:05d}", kind, points, frame=0))
    return records
