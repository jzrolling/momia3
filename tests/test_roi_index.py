import numpy as np

from momia3.roi_index import ROIGridIndex, _intersects


def test_grid_query_matches_bruteforce():
    rng = np.random.default_rng(3)
    boxes = []
    for index in range(1500):
        x0, y0 = rng.uniform(-100, 1000, 2)
        x1, y1 = x0 + rng.uniform(2, 80), y0 + rng.uniform(2, 80)
        boxes.append((f"r{index}", (x0, y0, x1, y1)))
    index = ROIGridIndex(cell_size=32)
    index.build(boxes)
    for _ in range(100):
        x0, y0 = rng.uniform(-100, 1000, 2)
        query = (x0, y0, x0 + rng.uniform(5, 200), y0 + rng.uniform(5, 200))
        expected = {roi_id for roi_id, bbox in boxes if _intersects(bbox, query)}
        assert set(index.query(query)) == expected


def test_grid_add_remove_and_replace():
    index = ROIGridIndex(cell_size=10)
    index.add("a", (0, 0, 5, 5))
    index.add("b", (15, 15, 25, 25))
    assert set(index.query((0, 0, 30, 30))) == {"a", "b"}
    index.add("a", (100, 100, 110, 110))
    assert index.query((0, 0, 30, 30)) == ["b"]
    index.remove("b")
    assert index.query((0, 0, 30, 30)) == []
    assert len(index) == 1


def test_grid_handles_negative_coordinates_and_empty_query():
    index = ROIGridIndex(cell_size=16)
    index.add("neg", (-20, -20, -5, -5))
    assert set(index.query((-30, -30, 0, 0))) == {"neg"}
    assert index.query((100, 100, 110, 110)) == []
