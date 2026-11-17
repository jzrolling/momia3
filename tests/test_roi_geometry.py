import numpy as np

from momia3.roi_geometry import ROIBuffer, ROIRecord


def _records():
    return [
        ROIRecord("rect", "rectangle", np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)),
        ROIRecord("poly", "polygon", np.array([[20, 20], [30, 20], [25, 30]], float)),
        ROIRecord("line", "polyline", np.array([[40, 40], [50, 45], [55, 40]], float)),
        ROIRecord("ellipse", "ellipse", np.array([[60, 60], [70, 60], [70, 72], [60, 72]], float)),
        ROIRecord("dot", "dot", np.array([[80, 80]], float)),
    ]


def test_buffer_flattens_closed_and_open_geometry():
    buffer = ROIBuffer().build(_records())
    assert len(buffer) == 5
    assert len(buffer.dot_centers) == 1
    assert buffer.dot_owners.tolist() == [4]
    assert len(buffer.segments) == 4 + 3 + 2 + 4
    assert len(buffer.fill_vertices) > 0
    assert len(buffer.fill_faces) > 0
    assert 2 not in buffer.fill_face_owners.tolist()
    assert buffer.colors.shape == (5, 4)
    assert set(buffer.segment_owners.tolist()) == {0, 1, 2, 3}


def test_buffer_viewport_and_frame_filter():
    buffer = ROIBuffer().build(_records())
    mask = buffer.visible_mask((-1, -1, 31, 31))
    assert mask.tolist() == [True, True, False, False, False]
    assert len(buffer.segments_for(mask)) == 7
    assert len(buffer.selected_segments([0])) == 4
    assert buffer.dots_for(mask).shape == (0, 2)
