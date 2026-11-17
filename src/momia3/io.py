"""Lazy image loading for the project sidebar.

Supported inputs are OME-TIFF/TIFF and OME-Zarr/Zarr folders.  The loader
returns channel views backed by the source array, so the napari canvas can
slice them without materialising the whole image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import zarr

DEFAULT_LUTS = ("blue", "green", "magenta", "yellow", "cyan", "red", "gray")
TIFF_SUFFIXES = {".tif", ".tiff"}


@dataclass
class ChannelData:
    data: Any
    name: str
    lut: str = "gray"
    contrast_limits: tuple[float, float] | None = None


@dataclass
class LoadedImage:
    path: Path
    axes: str
    channels: list[ChannelData]
    source: str
    scale: tuple[float, ...] | None = None
    keepalive: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


def is_supported(path: Path) -> bool:
    if path.is_dir():
        return path.name.lower().endswith(".zarr") and any((path / name).exists() for name in ("zarr.json", ".zgroup", ".zattrs"))
    return path.suffix.lower() in TIFF_SUFFIXES


def open_image(path: str | Path) -> LoadedImage:
    path = Path(path)
    if path.is_dir():
        return _open_zarr(path)
    if path.suffix.lower() in TIFF_SUFFIXES:
        return _open_tiff(path)
    raise ValueError(f"unsupported image: {path}")


def _open_zarr(path: Path) -> LoadedImage:
    group = zarr.open_group(str(path), mode="r")
    attrs = dict(group.attrs)
    array, axes = _find_multiscale(group, attrs)
    if axes is None:
        axes = tuple(attrs.get("_ARRAY_DIMENSIONS", ())) or _infer_axes(array.shape)
    axes = _normalise_axes(axes, array.ndim)
    names = _omero_names(attrs, _channel_count(array.shape, axes))
    return _make_loaded(path, array, axes, names=names, source="ome-zarr", keepalive=group)


def _open_tiff(path: Path) -> LoadedImage:
    import tifffile

    tif = tifffile.TiffFile(str(path))
    series = tif.series[0]
    axes = _normalise_axes(tuple(series.axes or _infer_axes(series.shape)), series.ndim)
    keepalive: Any = tif
    from_tiff = False
    try:
        store = tif.aszarr(series=0)
        array = zarr.open(store, mode="r")
        keepalive = (tif, store)
        from_tiff = True
    except Exception:
        array = series.asarray()
    names = _tiff_names(tif, _channel_count(array.shape, axes))
    loaded = _make_loaded(path, array, axes, names=names, source="ome-tiff" if tif.is_ome else "tiff", keepalive=keepalive)
    loaded.metadata["lazy"] = from_tiff
    return loaded


def _find_multiscale(group: Any, attrs: dict[str, Any]) -> tuple[Any, tuple[str, ...] | None]:
    multiscales = attrs.get("multiscales") or []
    if multiscales:
        meta = multiscales[0]
        datasets = meta.get("datasets") or []
        if datasets:
            node = group
            for part in str(datasets[0].get("path", "")).split("/"):
                if part:
                    node = node[part]
            axes_spec = meta.get("axes") or ()
            axes = []
            for axis in axes_spec:
                axes.append(str(axis.get("name", "")) if isinstance(axis, dict) else str(axis))
            return node, tuple(axes) if all(axes) else None
    candidates = []
    try:
        keys = list(group.array_keys())
    except Exception:
        keys = []
    for key in keys:
        try:
            array = group[key]
            candidates.append(array)
        except Exception:
            continue
    if not candidates:
        for key in group.keys():
            try:
                node = group[key]
                if hasattr(node, "shape"):
                    candidates.append(node)
            except Exception:
                continue
    if not candidates:
        raise ValueError("no array found in zarr group")
    array = max(candidates, key=lambda item: (item.ndim, np.prod(item.shape)))
    return array, tuple(array.attrs.get("_ARRAY_DIMENSIONS", ())) or None


def _make_loaded(path: Path, array: Any, axes: tuple[str, ...], names: list[str], source: str, keepalive: Any = None) -> LoadedImage:
    axes = _normalise_axes(axes, array.ndim)
    try:
        channel_axis = axes.index("C")
    except ValueError:
        channel_axis = None
    channels: list[ChannelData] = []
    out_axes = axes
    lazy = _as_lazy(array)
    if channel_axis is None:
        channels.append(ChannelData(lazy, names[0] if names else "Channel 1", DEFAULT_LUTS[0], _estimate_limits(lazy)))
    else:
        out_axes = axes[:channel_axis] + axes[channel_axis + 1 :]
        import dask.array as da

        for index in range(array.shape[channel_axis]):
            view = da.take(lazy, index, axis=channel_axis)
            name = names[index] if index < len(names) else f"Channel {index + 1}"
            channels.append(ChannelData(view, name, DEFAULT_LUTS[index % len(DEFAULT_LUTS)], _estimate_limits(view)))
    return LoadedImage(path=path, axes=out_axes, channels=channels, source=source, keepalive=keepalive)


def _as_lazy(array: Any):
    if array.__class__.__module__.startswith("dask"):
        return array
    try:
        import dask.array as da

        chunks = getattr(array, "chunks", None)
        return da.from_array(array, chunks=chunks, name=f"momia3-{id(array)}")
    except Exception:
        return array


def _estimate_limits(array: Any) -> tuple[float, float] | None:
    try:
        shape = tuple(int(v) for v in array.shape)
        if len(shape) < 2:
            return None
        selector = tuple(0 if i < len(shape) - 2 else slice(0, min(512, shape[i])) for i in range(len(shape)))
        sample = np.asarray(array[selector])
        if not sample.size:
            return None
        lo, hi = np.percentile(sample, (0.5, 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        return float(lo), float(hi)
    except Exception:
        return None


def _normalise_axes(axes: tuple[Any, ...], ndim: int) -> tuple[str, ...]:
    normalised = tuple(str(axis).upper() for axis in axes if str(axis))
    if len(normalised) != ndim:
        return _infer_axes(tuple(0 for _ in range(ndim)))
    return normalised


def _infer_axes(shape: tuple[int, ...]) -> tuple[str, ...]:
    ndim = len(shape)
    if ndim == 2:
        return ("Y", "X")
    if ndim == 3:
        return ("C", "Y", "X") if shape[0] <= 16 else ("Z", "Y", "X")
    if ndim == 4:
        return ("C", "Z", "Y", "X") if shape[0] <= 16 else ("T", "C", "Y", "X")
    if ndim == 5:
        return ("T", "C", "Z", "Y", "X")
    return tuple(f"D{i}" for i in range(ndim - 2)) + ("Y", "X")


def _channel_count(shape: tuple[int, ...], axes: tuple[str, ...]) -> int:
    try:
        return int(shape[axes.index("C")])
    except ValueError:
        return 1


def _tiff_names(tif: Any, count: int) -> list[str]:
    names: list[str] = []
    if tif.ome_metadata:
        try:
            root = ElementTree.fromstring(tif.ome_metadata)
            for element in root.iter():
                if element.tag.rsplit("}", 1)[-1] != "Channel":
                    continue
                name = element.attrib.get("Name") or element.attrib.get("ID")
                if name:
                    names.append(name)
        except Exception:
            pass
    return names or [f"Channel {i + 1}" for i in range(count)]


def _omero_names(attrs: dict[str, Any], count: int) -> list[str]:
    omero = attrs.get("omero") or {}
    channels = omero.get("channels") or []
    names = [str(item.get("label") or item.get("name") or f"Channel {i + 1}") for i, item in enumerate(channels)]
    return names or [f"Channel {i + 1}" for i in range(count)]
