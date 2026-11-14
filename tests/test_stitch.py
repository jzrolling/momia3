import numpy as np

from momia3.stitch.torch_stitch import StitchConfig, _intersection, _neighbor_edges


def test_neighbor_edges_grid():
    positions = np.array(
        [[2113.846, 0.0], [2113.846, 1256.923], [1056.923, 0.0],
         [1056.923, 1256.923], [0.0, 0.0], [0.0, 1256.923]]
    )
    edges = _neighbor_edges(positions, np.array([1080.0, 1280.0]))
    assert edges == [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]


def test_intersection_overlap_and_offsets():
    shape, padding, offsets = _intersection(
        np.array([0.0, 0.0]),
        np.array([0.0, 1256.923]),
        np.array([1080.0, 1280.0]),
        np.zeros(2),
    )
    assert tuple(shape) == (1080, 23)
    assert padding.tolist() == [0.0, 0.0]
    assert offsets[0].tolist() == [0.0, 1256.923]
    assert offsets[1].tolist() == [0.0, 0.0]


def test_default_config():
    cfg = StitchConfig()
    assert cfg.channel == 0
    assert cfg.window_limit == 2
    assert cfg.tile_size == 1024
