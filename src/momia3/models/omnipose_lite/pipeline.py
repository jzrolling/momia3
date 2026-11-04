"""Fast Omnipose inference pipeline (momia3.models.omnipose_lite).

Lightweight orchestration around the reference Omnipose CPnet weights: no RGB
flow rendering, no boundary/flow-error passes, torch-native Euler integration.
The checkpoint is downloaded separately and resolved by
momia3.models.omnipose_lite.weights.resolve_omnipose_weights; the model never
ships inside this repository.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage import measure

import fastremap

from cellpose_omni import resnet_torch

from momia3.models.omnipose_lite.weights import resolve_omnipose_weights

_CROSS4 = ndi.generate_binary_structure(2, 1)
_SQUARE8 = ndi.generate_binary_structure(2, 2)


def _quantiles(x: torch.Tensor, qs):
    q = torch.tensor(qs, device=x.device, dtype=x.dtype)
    lo, hi = torch.quantile(x, q)
    return float(lo), float(hi)


def normalize_channel_gpu(ch0: torch.Tensor) -> torch.Tensor:
    """Percentile normalisation matching normalize_img(omni=True).

    Reference skips channels whose 1st/99th percentile difference is tiny and
    uses 0.01/99.99 percentiles for the actual normalisation.
    """
    p1, p99 = _quantiles(ch0, [0.01, 0.99])
    if p99 > p1 + 1e-3:
        lo, hi = _quantiles(ch0, [0.0001, 0.9999])
        if abs(hi - lo) < 1e-12:
            return torch.zeros_like(ch0)
        return torch.clamp((ch0 - lo) / (hi - lo), 0, 1)
    return ch0


def img_to_gpu_channel(img: np.ndarray, device) -> torch.Tensor:
    a = np.asarray(img)
    t = torch.as_tensor(a, device=device)
    if a.ndim == 3:
        t = t.float().mean(dim=-1)
    elif a.ndim != 2:
        raise ValueError(f"unsupported image shape {a.shape}")
    t = t.float()
    return normalize_channel_gpu(t)


def _pad_amounts(size: int, div: int = 16, extra: int = 1):
    lpad = int(div * np.ceil(size / div) - size)
    p1 = extra * div // 2 + lpad // 2
    p2 = extra * div // 2 + lpad - lpad // 2
    return p1, p2


class CPnetRunner:
    """Loads an Omnipose 1.x checkpoint into a torch CPnet and runs single images."""

    def __init__(
        self,
        weights_path: str | None = None,
        device="cuda",
        nclasses: int = 4,
        half: bool = False,
    ):
        if weights_path is None:
            weights_path = resolve_omnipose_weights()
        self.device = torch.device(device)
        self.nclasses = nclasses
        # fp16 overflows in CPnet; bf16 autocast keeps fp32 weights and runs mixed precision
        self.bf16 = bool(half) and self.device.type == "cuda"
        self.net = resnet_torch.CPnet(
            [2, 32, 64, 128, 256], nclasses, sz=3,
            residual_on=True, style_on=True, concatenation=False,
            dim=2, kernel_size=2, scale_factor=2, dilation=1,
            checkpoint=False, dropout=False,
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state)
        self.net.to(self.device).eval()

    @torch.no_grad()
    def run_gpu(self, ch0: torch.Tensor):
        """Normalised (H, W) GPU tensor -> (nclasses, H, W) GPU tensor."""
        H, W = int(ch0.shape[0]), int(ch0.shape[1])
        py1, py2 = _pad_amounts(H)
        px1, px2 = _pad_amounts(W)
        t = ch0[None, None]
        t = F.pad(t, (px1, px2, py1, py2), mode="reflect")
        x = torch.zeros((1, 2, t.shape[-2], t.shape[-1]), device=self.device, dtype=t.dtype)
        x[:, 0] = t[:, 0]
        if self.bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y, _style = self.net(x)
        else:
            y, _style = self.net(x)
        y = y[0][:, py1:py1 + H, px1:px1 + W]
        return y.float()


@torch.no_grad()
def div_rescale_gpu(dP: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Torch port of omnipose.core.div_rescale (p=1)."""
    f = dP * mask
    mag = torch.sqrt((f * f).sum(dim=0)).clamp_min(1e-12)
    f = f / mag
    gy, gx = torch.gradient(f, dim=(1, 2), edge_order=1)
    div = gy + gx
    lo, hi = _quantiles(div, [0.0001, 0.9999])
    if abs(hi - lo) < 1e-12:
        return f
    divn = torch.clamp((div - lo) / (hi - lo), 0, 1)
    return f * divn


@torch.no_grad()
def euler_final_coords(
    flow: torch.Tensor,
    coords: np.ndarray,
    niter: int = 200,
    suppress: bool = True,
):
    """Torch port of omnipose.core.follow_flows/steps_batch (single image).

    flow: (2, H, W) float32 GPU tensor, channel order (y, x)
    coords: (2, N) int foreground coordinates (y, x)

    Returns final (y, x) pixel coordinates per foreground pixel (float32, len N).
    """
    H, W = int(flow.shape[1]), int(flow.shape[2])
    dev = flow.device
    ys = coords[0].astype(np.int64)
    xs = coords[1].astype(np.int64)
    f = torch.empty((1, 2, H, W), device=dev, dtype=flow.dtype)
    f[0, 0] = flow[1] * (2.0 / (W - 1))
    f[0, 1] = flow[0] * (2.0 / (H - 1))
    pt = torch.empty((1, 1, xs.size, 2), device=dev, dtype=flow.dtype)
    pt[..., 0] = torch.as_tensor(xs, device=dev).to(flow.dtype) * (2.0 / (W - 1)) - 1.0
    pt[..., 1] = torch.as_tensor(ys, device=dev).to(flow.dtype) * (2.0 / (H - 1)) - 1.0
    mode = "nearest" if suppress else "bilinear"
    dpt0 = F.grid_sample(f, pt, mode=mode, align_corners=True) if suppress else None
    for t in range(int(niter)):
        dpt = F.grid_sample(f, pt, mode=mode, align_corners=True)
        if suppress:
            dpt = (dpt + dpt0) / 2.0
            dpt0 = dpt.clone()
            dpt = dpt / (t + 1)
        pt[..., 0].add_(dpt[:, 0])
        pt[..., 1].add_(dpt[:, 1])
        pt.clamp_(-1.0, 1.0)
    x_final = (pt[..., 0] + 1.0) * 0.5 * (W - 1)
    y_final = (pt[..., 1] + 1.0) * 0.5 * (H - 1)
    return y_final[0, 0].float().cpu().numpy(), x_final[0, 0].float().cpu().numpy()


_BORDER_CACHE: dict = {}


def border_band(shape):
    """5-pixel border band; equal to scipy binary_dilation(border_value=1, iterations=5)."""
    band = _BORDER_CACHE.get(shape)
    if band is None:
        band = np.zeros(shape, dtype=bool)
        band[:5, :] = True
        band[-5:, :] = True
        band[:, :5] = True
        band[:, -5:] = True
        _BORDER_CACHE[shape] = band
    return band


def hysteresis_fast(dist: np.ndarray, low: float, high: float) -> np.ndarray:
    """Equivalent to skimage.filters.apply_hysteresis_threshold(dist, low, high)."""
    lowmask = dist >= low
    highmask = dist >= high
    lab, n = ndi.label(lowmask, structure=ndi.generate_binary_structure(2, 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[lab[highmask]] = True
    return keep[lab]


def get_masks_fast(p, bd, dist, mask, inds, nclasses=4, cluster=False,
                   diam_threshold=12.0, eps=None, min_samples=5, newinds=None):
    """Copy of omnipose.core.get_masks with a cached border band.

    If newinds (N,2) final pixel coordinates are provided, p is not needed.
    """
    if nclasses > 1:
        d = 2 * (mask.ndim + 1) * float(np.mean(np.abs(dist[mask])))
    else:
        d = 0.0
    if eps is None:
        eps = 2 ** 0.5
    if d <= diam_threshold:
        cluster = True

    cell_px = tuple(inds)
    if newinds is None:
        newinds = p[(Ellipsis,) + cell_px].T
    out = np.zeros(dist.shape, np.uint32)
    if cluster:
        from dbscan import DBSCAN as new_DBSCAN
        from sklearn.neighbors import NearestNeighbors
        labels, _ = new_DBSCAN(newinds, eps=eps, min_samples=min_samples)
        neighbors = NearestNeighbors(n_neighbors=5).fit(newinds)
        o_inds = np.where(labels == -1)[0]
        if len(o_inds):
            outliers = [newinds[i] for i in o_inds]
            nearest_dists, nearest_indices = neighbors.kneighbors(outliers)
            nearest_labels = labels[nearest_indices]
            nearest_idx = [np.where(n != -1)[0][0] if np.any(n != -1) else 0 for n in nearest_labels]
            labels[o_inds] = [nl[i] if nd[i] < eps else -1
                              for i, nl, nd in zip(nearest_idx, nearest_labels, nearest_dists)]
        out[cell_px] = labels + 1
    else:
        new_px = tuple(np.rint(newinds.T).astype(int))
        skelmask = np.zeros(dist.shape, dtype=bool)
        skelmask[new_px] = True
        band = border_band(skelmask.shape)
        border_px = skelmask & band
        if nclasses == mask.ndim + 2:
            border_px = border_px & ~(bd > -1)
        skelmask[band] = border_px[band]
        labels = ndi.label(skelmask, structure=_SQUARE8)[0]
        out[cell_px] = labels[new_px]
    return out


def _fill_small_holes(msk: np.ndarray, hsz: float) -> np.ndarray:
    """Exact port of skimage.morphology.remove_small_holes(msk, area_threshold=hsz).

    skimage's implementation labels the complement (padded so the outside is one
    large component), removes complement components with area < area_threshold,
    and inverts back.
    """
    pad = np.pad(msk, 1, mode="constant")
    holes = ~pad
    lab, n = ndi.label(holes, structure=ndi.generate_binary_structure(2, 1))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    # deprecated area_threshold -> new max_size = area_threshold - 1 (size <= max_size removed)
    fill = sizes <= (hsz - 1)
    fill[0] = False
    out = pad | fill[lab]
    return out[1:-1, 1:-1]


def cleanup_fast(labels, min_size=15, max_size=None, hole_size=3, dim=2):
    """Port of omnipose.core.fill_holes_and_remove_small_masks without ncolor."""
    labels = np.asarray(labels)
    if labels.dtype != np.int32:
        labels = labels.astype(np.int32, copy=False)
    out = np.zeros(labels.shape, dtype=np.int32)
    ncomp = int(labels.max()) if labels.size else 0
    if ncomp == 0:
        return out, 0
    areas = np.bincount(labels.ravel(), minlength=ncomp + 1)
    slices = ndi.find_objects(labels)
    j = 0
    fill_holes = hole_size > 0
    for i, slc in enumerate(slices):
        if slc is None:
            continue
        npix = int(areas[i + 1])
        if npix == 0:
            continue
        too_small = npix < min_size
        too_big = False if max_size is None else npix > max_size
        if (min_size > 0) and (too_small or too_big):
            continue
        sub = out[slc]
        hsz = npix * hole_size / 100.0 if fill_holes else 0.0
        if hsz >= 1.0:
            msk = _fill_small_holes(labels[slc] == (i + 1), hsz)
            sub[msk] = j + 1
        else:
            sub[labels[slc] == (i + 1)] = j + 1
        j += 1
    return out, j


@torch.no_grad()
def euler_suppressed_gpu(flow: torch.Tensor, coords: np.ndarray, niter: int = 200,
                         suppress: bool = True) -> np.ndarray:
    """Legacy wrapper returning a full (2, H, W) position array."""
    ys = coords[0].astype(np.int64)
    xs = coords[1].astype(np.int64)
    y, x = euler_final_coords(flow, coords, niter=niter, suppress=suppress)
    p = np.zeros((2,) + tuple(flow.shape[1:]), dtype=np.float32)
    p[0, ys, xs] = y
    p[1, ys, xs] = x
    return p


@dataclass
class StageTimings:
    totals: dict = field(default_factory=dict)

    def add(self, name: str, dt: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + dt

    def reset(self) -> None:
        self.totals.clear()


class FastOmnipose:
    """P1 pipeline: reference weights, GPU pre/post processing, no rendering."""

    def __init__(self, weights_path=None, device="cuda", half: bool = False, runner=None):
        self.device = torch.device(device)
        if runner is not None:
            self.runner = runner
        else:
            if weights_path is None:
                weights_path = resolve_omnipose_weights()
            self.runner = CPnetRunner(weights_path, device=self.device, half=half)
        self.nclasses = 4
        self.timings = StageTimings()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _gpu_stages(self, img, niter, mask_threshold, diam_threshold):
        """Preprocess + network + threshold + div/euler; returns CPU-ready state."""
        ch0 = img_to_gpu_channel(img, self.device)
        y = self.runner.run_gpu(ch0)
        self._sync()
        dist = y[2]
        bd = y[3]
        dist_np = dist.float().cpu().numpy()
        iscell_np = hysteresis_fast(dist_np, mask_threshold - 1.0, mask_threshold)
        if not bool(np.any(iscell_np)):
            return None, dist_np.shape
        coords = np.array(np.nonzero(iscell_np)).astype(np.int32)
        iscell_t = torch.from_numpy(iscell_np).to(self.device)
        flow = div_rescale_gpu(y[:2], iscell_t)
        yf, xf = euler_final_coords(flow, coords, niter=niter, suppress=True)
        bd_np = bd.float().cpu().numpy()
        newinds = np.stack([yf, xf], axis=1)
        return (dist_np, iscell_np, coords, newinds, bd_np), dist_np.shape

    @staticmethod
    def _cpu_finish(state, shape, nclasses, diam_threshold, min_size, max_size, hole_size):
        if state is None:
            return np.zeros(shape, dtype=np.uint32)
        dist_np, iscell_np, coords, newinds, bd_np = state
        labels = get_masks_fast(None, bd_np, dist_np, iscell_np, coords, nclasses,
                                cluster=False, diam_threshold=diam_threshold,
                                newinds=newinds)
        labels, _n = cleanup_fast(labels, min_size=min_size, max_size=max_size,
                                  hole_size=hole_size, dim=2)
        labels = (labels * iscell_np).astype(np.uint32)
        fastremap.renumber(labels, in_place=True)
        return labels.astype(np.uint32)

    def segment_batch(self, images, niter: int = 40, mask_threshold: float = 0.0,
                      diam_threshold: float = 12.0, min_size: int = 15,
                      max_size=None, hole_size: int = 3):
        """Process many images, overlapping CPU post-processing with GPU work.

        GPU stages run on the calling thread; mask reconstruction / cleanup runs
        on a worker thread for the previous frame. Returns labels in input order.
        """
        from concurrent.futures import ThreadPoolExecutor

        ex = ThreadPoolExecutor(max_workers=1)
        futs = []
        try:
            for img in images:
                state, shape = self._gpu_stages(img, niter, mask_threshold, diam_threshold)
                futs.append(ex.submit(self._cpu_finish, state, shape, self.nclasses,
                                      diam_threshold, min_size, max_size, hole_size))
            return [f.result() for f in futs]
        finally:
            ex.shutdown()

    def segment(
        self,
        img: np.ndarray,
        niter: int = 200,
        mask_threshold: float = 0.0,
        diam_threshold: float = 12.0,
        min_size: int = 15,
        max_size=None,
        hole_size: int = 3,
    ) -> np.ndarray:
        T = self.timings
        t0 = time.perf_counter()
        ch0 = img_to_gpu_channel(img, self.device)
        T.add("preprocess", time.perf_counter() - t0)

        t0 = time.perf_counter()
        y = self.runner.run_gpu(ch0)
        self._sync()
        T.add("network", time.perf_counter() - t0)
        dP = y[:2]
        dist = y[2]
        bd = y[3]

        t0 = time.perf_counter()
        dist_np = dist.float().cpu().numpy()
        iscell_np = hysteresis_fast(dist_np, mask_threshold - 1.0, mask_threshold)
        T.add("threshold", time.perf_counter() - t0)
        if not bool(np.any(iscell_np)):
            return np.zeros(dist_np.shape, dtype=np.uint32)

        t0 = time.perf_counter()
        coords = np.array(np.nonzero(iscell_np)).astype(np.int32)
        iscell_t = torch.from_numpy(iscell_np).to(self.device)
        flow = div_rescale_gpu(dP, iscell_t)
        T.add("div_rescale", time.perf_counter() - t0)

        t0 = time.perf_counter()
        yf, xf = euler_final_coords(flow, coords, niter=niter, suppress=True)
        self._sync()
        T.add("euler", time.perf_counter() - t0)
        newinds = np.stack([yf, xf], axis=1)

        t0 = time.perf_counter()
        labels = get_masks_fast(
            None, bd.float().cpu().numpy(), dist_np, iscell_np, coords, self.nclasses,
            cluster=False, diam_threshold=diam_threshold, newinds=newinds,
        )
        T.add("get_masks", time.perf_counter() - t0)

        t0 = time.perf_counter()
        labels, _n = cleanup_fast(
            labels, min_size=min_size, max_size=max_size, hole_size=hole_size, dim=2
        )
        labels = (labels * iscell_np).astype(np.uint32)
        fastremap.renumber(labels, in_place=True)
        T.add("cleanup", time.perf_counter() - t0)
        return labels.astype(np.uint32)


# Preferred public name of the backend; FastOmnipose is kept as an alias.
OmniposeLite = FastOmnipose
