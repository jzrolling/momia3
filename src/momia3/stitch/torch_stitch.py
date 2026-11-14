"""Torch-accelerated ASHLAR-style stitching for OME-TIFF tile series.

This is the prototype used for the exemplar-001 tile-stitch benchmark.  It
mirrors ASHLAR's alignment structure: multi-window phase correlation, a
neighbor graph from stage positions, and spanning-tree position propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

import networkx as nx
import numpy as np
import tifffile
from scipy import ndimage as ndi

__all__ = ["StitchConfig", "StitchResult", "stitch_ome_tiff"]

_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
_LAPLACE = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


@dataclass(frozen=True)
class StitchConfig:
    channel: int = 0
    pixel_size_um: float | None = None
    max_shift_um: float = 100.0
    window_limit: int = 2
    tile_size: int = 1024
    compression: str = "adobe_deflate"
    dfp_path: str | Path | None = None
    ffp_path: str | Path | None = None


@dataclass
class StitchResult:
    positions: np.ndarray
    edge_shifts: dict[tuple[int, int], np.ndarray]
    edge_errors: dict[tuple[int, int], float]
    output_path: Path
    tile_shape: tuple[int, int] = field(default=())


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for torch tile stitching; install momia3[gpu] "
            "or momia3[inference]"
        ) from exc
    return torch


def _read_stage(path, pixel_size_um):
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"no TIFF series in {path}")
        tile_shape = tuple(int(v) for v in tif.series[0].shape[-2:])
        if not tif.ome_metadata:
            raise ValueError(f"{path} has no OME-XML stage positions")
        root = ET.fromstring(tif.ome_metadata)
        pixels = root.find(".//ome:Pixels", _NS)
        if pixels is None:
            raise ValueError(f"{path} has no OME Pixels element")
        if pixel_size_um is None:
            value = pixels.attrib.get("PhysicalSizeX")
            pixel_size_um = float(value) if value else 1.0
        positions = []
        for image in root.findall("ome:Image", _NS):
            plane = image.find("ome:Pixels/ome:Plane", _NS)
            if plane is None:
                raise ValueError("OME Image has no Plane positions")
            # ASHLAR flips stage Y into image row coordinates.
            y = -float(plane.attrib["PositionY"])
            x = float(plane.attrib["PositionX"])
            positions.append([y / pixel_size_um, x / pixel_size_um])
    return np.asarray(positions, dtype=np.float64), tile_shape, float(pixel_size_um)


def _read_channel(path, channel):
    with tifffile.TiffFile(path) as tif:
        tiles = []
        for series in tif.series:
            arr = series.asarray()
            if arr.ndim == 3:
                if channel >= arr.shape[0]:
                    raise ValueError(f"channel {channel} out of range for {arr.shape}")
                arr = arr[channel]
            elif arr.ndim != 2:
                raise ValueError(f"unsupported series shape {arr.shape}")
            tiles.append(np.asarray(arr, dtype=np.float32))
    return np.stack(tiles)


def _intersection(p1, p2, tile_shape, min_size, shift=None):
    corners = np.vstack([p1, p2])
    if shift is not None:
        corners[1] = corners[1] + np.asarray(shift)
    corners2 = corners + tile_shape
    max_shape = (corners2 - corners).max(axis=0)
    min_size = np.clip(np.asarray(min_size, dtype=float), 1, max_shape)
    position = corners.max(axis=0)
    initial_shape = np.floor(corners2.min(axis=0) - position).astype(int)
    shape = np.ceil(np.maximum(initial_shape, min_size)).astype(int)
    padding = shape - initial_shape
    offsets = np.maximum(position - corners - padding, 0)
    return shape, padding, offsets


def _crop(img, offset, shape):
    y, x = np.round(offset).astype(int)
    return img[y : y + shape[0], x : x + shape[1]]


def _window_sizes(positions, tile_shape, edge):
    i, j = edge
    smin = _intersection(positions[i], positions[j], tile_shape, np.zeros(2))[0]
    smax = np.round(tile_shape * 0.1).astype(int)
    sizes = [smin]
    while any(sizes[-1] < smax):
        sizes.append(sizes[-1] * 2)
    return sizes


def _neighbor_edges(positions, tile_shape):
    edges = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            lo = np.maximum(positions[i], positions[j])
            hi = np.minimum(positions[i] + tile_shape, positions[j] + tile_shape)
            overlap = hi - lo
            if np.all(overlap > 0) and overlap.max() > 0.5 * tile_shape.max():
                edges.append((i, j))
    return edges


def _hann(torch, shape):
    cache = getattr(_hann, "_cache", None)
    if cache is None:
        cache = {}
        _hann._cache = cache
    if shape not in cache:
        wy = torch.hann_window(shape[0], periodic=True)
        wx = torch.hann_window(shape[1], periodic=True)
        cache[shape] = wy[:, None] * wx[None, :]
    return cache[shape]


def _shift_batch(x, dy, dx):
    torch = _require_torch()
    n, h, w = x.shape
    yy = torch.arange(h, dtype=torch.float32)[None, :, None] - dy[:, None, None]
    xx = torch.arange(w, dtype=torch.float32)[None, None, :] - dx[:, None, None]
    valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    yi = yy.clamp(0, h - 1).long()
    xi = xx.clamp(0, w - 1).long()
    batch = torch.arange(n)[:, None, None]
    return x[batch, yi, xi] * valid.float()


def _batch_register(a, b):
    torch = _require_torch()
    x = torch.as_tensor(a)[:, None]
    y = torch.as_tensor(b)[:, None]
    kernel = torch.tensor(_LAPLACE, dtype=torch.float32)[None, None]
    x = torch.nn.functional.conv2d(x, kernel, padding=1)
    y = torch.nn.functional.conv2d(y, kernel, padding=1)
    win = _hann(torch, a.shape[1:])
    x = x[:, 0] * win
    y = y[:, 0] * win
    corr = torch.fft.ifft2(torch.fft.fft2(x) * torch.conj(torch.fft.fft2(y))).real.abs()
    n, h, w = corr.shape
    vals, idx = corr.reshape(n, -1).max(1)
    iy, ix = idx // w, idx % w

    def at(yy, xx):
        return corr[torch.arange(n), yy % h, xx % w]

    den_y = 2 * vals - at(iy - 1, ix) - at(iy + 1, ix)
    fy = 0.5 * (at(iy + 1, ix) - at(iy - 1, ix)) / torch.where(
        den_y.abs() > 1e-8, den_y, torch.ones_like(den_y)
    )
    den_x = 2 * vals - at(iy, ix - 1) - at(iy, ix + 1)
    fx = 0.5 * (at(iy, ix + 1) - at(iy, ix - 1)) / torch.where(
        den_x.abs() > 1e-8, den_x, torch.ones_like(den_x)
    )
    fy, fx = fy.clamp(-0.5, 0.5), fx.clamp(-0.5, 0.5)
    by = torch.where(iy > h // 2, iy - h, iy)
    bx = torch.where(ix > w // 2, ix - w, ix)
    cy = torch.stack([by, by, by - h, by - h], 1).reshape(-1)
    cx = torch.stack([bx, bx - w, bx, bx - w], 1).reshape(-1)
    x4 = x[:, None].expand(n, 4, h, w).reshape(n * 4, h, w)
    y4 = y[:, None].expand(n, 4, h, w).reshape(n * 4, h, w)
    corr4 = (x4 * _shift_batch(y4, cy.float(), cx.float())).reshape(n * 4, -1).sum(1).abs().reshape(n, 4)
    norm = torch.linalg.norm(x.reshape(n, -1), dim=1) * torch.linalg.norm(y.reshape(n, -1), dim=1)
    err = -torch.log((corr4 / norm[:, None]).clamp_min(1e-12))
    best = err.argmin(1)
    ar = torch.arange(n)
    out_y = cy.reshape(n, 4)[ar, best].float() + fy
    out_x = cx.reshape(n, 4)[ar, best].float() + fx
    return torch.stack([out_y, out_x], 1).numpy(), err[ar, best].numpy()


def _nccw(a, b):
    a = ndi.convolve(a.astype(np.float32), _LAPLACE)
    b = ndi.convolve(b.astype(np.float32), _LAPLACE)
    corr = abs(float(np.sum(a * b)))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return -np.log(corr / norm) if corr > 0 and norm > 0 else float("inf")


def _estimate_edges(tiles, positions, edges, tile_shape, cfg):
    items, meta = [], []
    for edge_index, (i, j) in enumerate(edges):
        sizes = _window_sizes(positions, tile_shape, (i, j))[: max(1, cfg.window_limit)]
        for size in sizes:
            shape, padding, offsets = _intersection(positions[i], positions[j], tile_shape, size)
            items.append((_crop(tiles[i], offsets[0], shape), _crop(tiles[j], offsets[1], shape)))
            sy = 1 if positions[i][0] >= positions[j][0] else -1
            sx = 1 if positions[i][1] >= positions[j][1] else -1
            meta.append((edge_index, padding * np.array([sy, sx])))
    shifts = np.zeros((len(items), 2))
    errors = np.zeros(len(items))
    groups = {}
    for k, (a, _b) in enumerate(items):
        groups.setdefault(a.shape, []).append(k)
    for shape, idxs in groups.items():
        sh, er = _batch_register(
            np.stack([items[k][0] for k in idxs]), np.stack([items[k][1] for k in idxs])
        )
        for n, k in enumerate(idxs):
            shifts[k], errors[k] = sh[n], er[n]
    edge_shifts, edge_errors = {}, {}
    for edge_index, (i, j) in enumerate(edges):
        idxs = [k for k, m in enumerate(meta) if m[0] == edge_index]
        best = min(idxs, key=lambda k: errors[k])
        shift = shifts[best] + meta[best][1]
        shape, _padding, offsets = _intersection(positions[i], positions[j], tile_shape, np.zeros(2), shift)
        a, b = _crop(tiles[i], offsets[0], shape), _crop(tiles[j], offsets[1], shape)
        edge_shifts[(i, j)] = shift
        edge_errors[(i, j)] = _nccw(a, b)
    return edge_shifts, edge_errors


def _solve_positions(positions, edges, edge_shifts, edge_errors):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(positions)))
    graph.add_weighted_edges_from([(i, j, float(edge_errors[(i, j)])) for i, j in edges])
    tree = nx.Graph()
    tree.add_nodes_from(graph)
    for component in nx.connected_components(graph):
        sub = graph.subgraph(component)
        center = nx.center(sub)[0]
        for path in nx.single_source_dijkstra_path(sub, center).values():
            nx.add_path(tree, path)
    root = nx.center(tree)[0]
    shifts = {root: np.zeros(2)}
    for source, dest in nx.traversal.bfs_edges(tree, root):
        if source not in shifts:
            source, dest = dest, source
        shift = edge_shifts[tuple(sorted((source, dest)))]
        if source > dest:
            shift = -shift
        shifts[dest] = shifts[source] + shift
    solved = np.array([shifts[k] for k in range(len(positions))]) + positions
    return solved - solved.min(axis=0)


def _assemble(tiles, positions, tile_shape, dfp, ffp):
    size = tuple(np.ceil(positions.max(axis=0) + tile_shape).astype(int))
    canvas = np.zeros(size, dtype=np.float32)
    weights = np.zeros(size, dtype=np.float32)
    for i, image in enumerate(tiles):
        if dfp is not None and ffp is not None:
            image = (image - dfp) / np.maximum(ffp, 1e-3)
        y0, x0 = np.floor(positions[i]).astype(int)
        frac = positions[i] - [y0, x0]
        if np.any(frac):
            image = ndi.shift(image, frac, order=1, mode="constant")
        y1, x1 = min(size[0], y0 + image.shape[0]), min(size[1], x0 + image.shape[1])
        sub = image[: y1 - y0, : x1 - x0]
        yy, xx = np.mgrid[0 : sub.shape[0], 0 : sub.shape[1]]
        wy = np.minimum(yy, sub.shape[0] - 1 - yy) + 1
        wx = np.minimum(xx, sub.shape[1] - 1 - xx) + 1
        wmap = np.clip(np.minimum(wy, wx) / 32.0, 0.02, 1.0)
        canvas[y0:y1, x0:x1] += sub * wmap
        weights[y0:y1, x0:x1] += wmap
    return canvas / np.maximum(weights, 1e-6)


def _write_pyramid(path, canvas, tile_size, compression):
    lo, hi = np.percentile(canvas, (0.5, 99.5))
    data = np.round(np.clip((canvas - lo) / max(hi - lo, 1.0), 0, 1) * 65535).astype(np.uint16)
    levels = [data]
    while max(levels[-1].shape) > 1024:
        arr = levels[-1]
        arr = arr[: arr.shape[0] // 2 * 2, : arr.shape[1] // 2 * 2]
        levels.append(np.round(arr.reshape(arr.shape[0] // 2, 2, arr.shape[1] // 2, 2).mean(axis=(1, 3))).astype(np.uint16))
    with tifffile.TiffWriter(path, ome=True, bigtiff=True) as writer:
        writer.write(data=levels[0], metadata={"Pixels": {"PhysicalSizeX": 1.0, "PhysicalSizeY": 1.0}}, subifds=len(levels) - 1, tile=(tile_size, tile_size), compression=compression, predictor=True, photometric="minisblack")
        for level in levels[1:]:
            writer.write(data=level, subfiletype=1, tile=(tile_size, tile_size), compression=compression, predictor=True, photometric="minisblack")


def stitch_ome_tiff(input_path, output_path, config=None):
    """Stitch one channel of an OME-TIFF tile series and write a pyramid."""
    cfg = config or StitchConfig()
    positions, tile_shape, _pixel_size = _read_stage(input_path, cfg.pixel_size_um)
    tiles = _read_channel(input_path, cfg.channel)
    if len(tiles) != len(positions):
        raise ValueError(f"{len(tiles)} series but {len(positions)} stage positions")
    tile_shape = np.asarray(tile_shape, dtype=np.float64)
    edges = _neighbor_edges(positions, tile_shape)
    if not edges:
        raise ValueError("no neighboring tiles found from stage positions")
    edge_shifts, edge_errors = _estimate_edges(tiles, positions, edges, tile_shape, cfg)
    solved = _solve_positions(positions, edges, edge_shifts, edge_errors)
    dfp = tifffile.imread(cfg.dfp_path).astype(np.float32)[cfg.channel] if cfg.dfp_path else None
    ffp = tifffile.imread(cfg.ffp_path).astype(np.float32)[cfg.channel] if cfg.ffp_path else None
    canvas = _assemble(tiles, solved, tile_shape, dfp, ffp)
    out = Path(output_path)
    _write_pyramid(out, canvas, cfg.tile_size, cfg.compression)
    return StitchResult(positions=solved, edge_shifts=edge_shifts, edge_errors=edge_errors, output_path=out, tile_shape=tuple(int(v) for v in tile_shape))
