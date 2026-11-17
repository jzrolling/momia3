"""Batched Vispy overlay for the custom ROI renderer."""

from __future__ import annotations

import numpy as np
from vispy.scene.visuals import Line, Markers, Mesh

from ..roi_geometry import ROIBuffer

ROI_COLOR = "#67d9c4"
DOT_COLOR = "#f2c45e"
SELECTION_COLOR = "#f2b45e"
HOVER_COLOR = "#ffffff"
FILL_COLOR = (0.40, 0.85, 0.77, 0.16)


class ROIVispyOverlay:
    def __init__(self, view, roi_color: str = ROI_COLOR, dot_color: str = DOT_COLOR, selection_color: str = SELECTION_COLOR) -> None:
        self.roi_color = roi_color
        self.dot_color = dot_color
        self.selection_color = selection_color
        dummy = np.zeros((1, 2), dtype=np.float32)
        empty = dummy
        self._dummy = dummy
        self.fill = Mesh(
            vertices=np.zeros((1, 2), dtype=np.float32),
            faces=np.zeros((1, 3), dtype=np.int32),
            color=FILL_COLOR,
            parent=view.scene,
        )
        self.fill.visible = False
        self.fill.order = 9999
        # Alpha blend overlapping ROIs instead of letting depth testing merge/cull them.
        self.fill.set_gl_state(blend=True, blend_func=("src_alpha", "one_minus_src_alpha"), depth_test=False)
        self.line = Line(
            pos=empty,
            color=roi_color,
            width=1.0,
            connect="segments",
            parent=view.scene,
        )
        self.line.visible = False
        self.line.order = 10000
        self.dots = Markers(
            pos=empty,
            face_color=dot_color,
            size=3,
            parent=view.scene,
        )
        self.dots.visible = False
        self.dots.order = 10001
        self.selection = Line(
            pos=empty,
            color=selection_color,
            width=2.0,
            connect="segments",
            parent=view.scene,
        )
        self.selection.visible = False
        self.selection.order = 10002
        self.hover = Line(
            pos=empty,
            color=HOVER_COLOR,
            width=2.0,
            connect="segments",
            parent=view.scene,
        )
        self.hover.visible = False
        self.hover.order = 10003
        self.handles = Markers(
            pos=empty,
            face_color="#f2b45e",
            edge_color="#111111",
            size=8,
            symbol="square",
            parent=view.scene,
        )
        self.handles.visible = False
        self.handles.order = 10004
        self.draft = Line(
            pos=empty,
            color="#ffffff",
            width=1.5,
            connect="segments",
            parent=view.scene,
        )
        self.draft.visible = False
        self.draft.order = 10005
        self.draft_points = Markers(
            pos=empty,
            face_color="#ffffff",
            edge_color="#111111",
            size=6,
            parent=view.scene,
        )
        self.draft_points.visible = False
        self.draft_points.order = 10006

    def update(self, buffer: ROIBuffer, mask: np.ndarray | None = None) -> None:
        segments = buffer.segments_for(mask)
        dots = buffer.dots_for(mask)
        colors = buffer.colors
        segment_owners = buffer.segment_owners
        dot_owners = buffer.dot_owners
        if mask is not None:
            segment_owners = segment_owners[mask[segment_owners]]
            dot_owners = dot_owners[mask[dot_owners]]
        if len(buffer.fill_faces):
            face_colors = colors[buffer.fill_face_owners].copy()
            face_colors[:, 3] = 0.16
            self.fill.set_data(
                vertices=buffer.fill_vertices,
                faces=buffer.fill_faces,
                face_colors=face_colors,
            )
            self.fill.visible = True
        else:
            self.fill.visible = False
        if len(segments):
            segment_colors = np.repeat(colors[segment_owners], 2, axis=0)
            self.line.set_data(
                pos=segments.reshape(-1, 2).astype(np.float32, copy=False),
                color=segment_colors,
                width=1.0,
                connect="segments",
            )
            self.line.visible = True
        else:
            self.line.visible = False
        if len(dots):
            self.dots.set_data(pos=dots, face_color=colors[dot_owners], size=3)
            self.dots.visible = True
        else:
            self.dots.visible = False

    def set_selection(self, segments: np.ndarray) -> None:
        if not len(segments):
            self.selection.visible = False
            return
        self.selection.set_data(
            pos=segments.reshape(-1, 2).astype(np.float32, copy=False),
            color=self.selection_color,
            width=2.0,
            connect="segments",
        )
        self.selection.visible = True

    def set_hover(self, segments: np.ndarray) -> None:
        if not len(segments):
            self.hover.visible = False
            return
        self.hover.set_data(
            pos=segments.reshape(-1, 2).astype(np.float32, copy=False),
            color=HOVER_COLOR,
            width=2.0,
            connect="segments",
        )
        self.hover.visible = True

    def set_handles(self, points: np.ndarray) -> None:
        if not len(points):
            self.handles.visible = False
            return
        self.handles.set_data(
            pos=points.astype(np.float32, copy=False),
            face_color="#f2b45e",
            edge_color="#111111",
            size=8,
            symbol="square",
        )
        self.handles.visible = True

    def set_draft(self, segments: np.ndarray, points: np.ndarray) -> None:
        if len(segments):
            self.draft.set_data(
                pos=segments.reshape(-1, 2).astype(np.float32, copy=False),
                color="#ffffff",
                width=1.5,
                connect="segments",
            )
            self.draft.visible = True
        else:
            self.draft.visible = False
        if len(points):
            self.draft_points.set_data(pos=points.astype(np.float32, copy=False), face_color="#ffffff", edge_color="#111111", size=6)
            self.draft_points.visible = True
        else:
            self.draft_points.visible = False

    def clear_draft(self) -> None:
        self.draft.visible = False
        self.draft_points.visible = False
