from pathlib import Path

import numpy as np
import zarr
import tifffile

from momia3.io import is_supported, open_image


def _write_ome_zarr(root: Path) -> None:
    data = np.arange(2 * 3 * 4 * 8 * 9, dtype=np.uint16).reshape(2, 3, 4, 8, 9)
    group = zarr.open_group(str(root), mode="w")
    group.create_array("0", data=data, chunks=(1, 1, 1, 8, 9))
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "t"}, {"name": "c"}, {"name": "z"},
                {"name": "y"}, {"name": "x"},
            ],
            "datasets": [{"path": "0"}],
        }
    ]
    group.attrs["omero"] = {"channels": [{"label": "A"}, {"label": "B"}, {"label": "C"}]}


def test_ome_zarr_is_lazy_and_splits_channels(tmp_path: Path):
    root = tmp_path / "demo.ome.zarr"
    _write_ome_zarr(root)
    assert is_supported(root)
    loaded = open_image(root)
    assert loaded.source == "ome-zarr"
    assert loaded.axes == ("T", "Z", "Y", "X")
    assert [channel.name for channel in loaded.channels] == ["A", "B", "C"]
    channel = loaded.channels[0].data
    assert channel.shape == (2, 4, 8, 9)
    assert hasattr(channel, "dask")
    assert channel[:1, :1, :2, :2].compute().shape == (1, 1, 2, 2)


def test_ome_tiff_is_lazy_and_splits_channels(tmp_path: Path):
    path = tmp_path / "demo.ome.tif"
    data = np.arange(2 * 2 * 3 * 5 * 6, dtype=np.uint16).reshape(2, 2, 3, 5, 6)
    tifffile.imwrite(
        path,
        data,
        ome=True,
        photometric="minisblack",
        metadata={"axes": "TCZYX", "Channel": {"Name": ["DAPI", "CD3"]}},
    )
    assert is_supported(path)
    loaded = open_image(path)
    assert loaded.source == "ome-tiff"
    assert loaded.axes == ("T", "Z", "Y", "X")
    assert [channel.name for channel in loaded.channels] == ["DAPI", "CD3"]
    assert loaded.channels[0].data.shape == (2, 3, 5, 6)
    assert hasattr(loaded.channels[0].data, "dask")
