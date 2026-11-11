"""Tests for ImageJ ROI conversion and closed-polygon processing."""

from __future__ import annotations

import struct
import zipfile

import numpy as np
import tifffile

from momia3.models.probability.rois import (
    prepare_roi_dataset,
    rasterize_roi_polygons,
    read_imagej_polygon,
    read_imagej_roi_zip,
)
from momia3.ops.polygon import resample_closed_polygon, smooth_closed_polygon


def _imagej_polygon(points: np.ndarray) -> bytes:
    points = np.asarray(points, dtype=np.int16)
    left = int(points[:, 0].min())
    top = int(points[:, 1].min())
    right = int(points[:, 0].max())
    bottom = int(points[:, 1].max())
    n = len(points)
    data = bytearray(64 + 4 * n)
    data[:4] = b"Iout"
    struct.pack_into(">H", data, 4, 228)
    data[6] = 0
    struct.pack_into(">hhhh", data, 8, top, left, bottom, right)
    struct.pack_into(">H", data, 16, n)
    x = points[:, 0] - left
    y = points[:, 1] - top
    data[64 : 64 + 2 * n] = x.astype(">i2").tobytes()
    data[64 + 2 * n :] = y.astype(">i2").tobytes()
    return bytes(data)


def test_imagej_polygon_decode_and_zip(tmp_path) -> None:
    expected = np.array([[12, 15], [30, 15], [30, 35], [12, 35]], dtype=float)
    payload = _imagej_polygon(expected)
    np.testing.assert_array_equal(read_imagej_polygon(payload), expected)
    archive = tmp_path / "sample_rois.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("sample.roi", payload)
    polygons = read_imagej_roi_zip(archive)
    assert len(polygons) == 1
    np.testing.assert_array_equal(polygons[0][1], expected)


def test_polygon_resample_smooth_and_overlap_ownership() -> None:
    square = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=float)
    ring = smooth_closed_polygon(square, smooth=1.0, spacing=0.5)
    assert len(ring) >= 16
    assert np.all(np.isfinite(ring))
    assert resample_closed_polygon(ring, 40).shape == (40, 2)

    shifted = square + np.array([15, 0])
    labels, qc = rasterize_roi_polygons(
        [("a", square), ("b", shifted)], (50, 55), spacing=1.0
    )
    assert labels.shape == (50, 55)
    assert int(labels.max()) == 2
    assert set(np.unique(labels)) == {0, 1, 2}
    assert sum(row["overlap_pixels"] for row in qc) > 0
    assert all(row["final_area"] > 0 for row in qc)


def test_prepare_roi_dataset_ignores_existing_masks(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    image = np.zeros((48, 48), dtype=np.uint16)
    image[10:30, 10:30] = 100
    tifffile.imwrite(source / "sample.tif", image)
    tifffile.imwrite(source / "sample_multilabel.tif", image)
    polygon = np.array([[10, 10], [30, 10], [30, 30], [10, 30]])
    with zipfile.ZipFile(source / "sample_roi.zip", "w") as z:
        z.writestr("cell.roi", _imagej_polygon(polygon))
    output = tmp_path / "output"
    manifest = prepare_roi_dataset(source, output)
    assert manifest["n_images"] == 1
    labels = tifffile.imread(output / "other/sample_masks.tif")
    assert labels.shape == image.shape
    assert int(labels.max()) == 1
    assert labels.dtype == np.uint16
