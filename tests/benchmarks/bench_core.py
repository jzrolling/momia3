"""Shared benchmark implementations: momia3 vs the momia2 reference.

Three operation groups are benchmarked (plan §7):

1. **Rolling-ball background** — momia3 ``rolling_ball_background_fft`` vs
   momia2's production path ``rolling_ball_bg_subtraction`` (gaussian -> 4x
   local-min shrink -> numba min-envelope kernel -> upsample) and vs the raw
   numba kernel ``rolling_ball_bg`` at true radius. Correctness is assessed
   two ways: (a) *algorithmic fidelity* — momia3 is a documented port of
   momia2's torch FFT reference
   (``external/mosaic_picasso_torch.py::rolling_ball_fft_background``), so we
   transcribe that reference to numpy and require Pearson >= 0.98;
   (b) *cross-estimator* agreement with the production min-envelope
   background, full-image and interior (one ball radius cropped, where the
   FFT zero-padding cannot leak).
2. **Midline extraction** — momia3 batch ``extract_midlines`` vs momia2
   per-mask ``find_contour_marching_squares`` + ``extract_midline`` (both
   include outline extraction: production-vs-production), zhang skeleton on
   both sides. Correctness: per-mask main-branch arc length agreement.
3. **Overlap matrix** — momia3 ``overlap_matrix`` (single bincount) vs
   momia2 ``_calculate_overlap_matrix`` (``np.unique(axis=0)`` + pandas).
   Correctness: exact equality of the (label_t, label_t1, pixels) triples.

momia2 is loaded read-only through ``benchmarks/_momia2_loader.py``; see that
module for the import strategy and the purity guards.

Scope trims (per task instructions): 3 repeats everywhere except the raw
numba kernel (1 repeat, JIT-warmed — a single r=100 pass costs ~1 min); the
end-to-end per-frame stage cost is not re-measured here but aggregated from
``benchmarks/robustness_matrix.json`` stage columns in
``benchmarks/run_robustness.py``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from _momia2_loader import load_momia2  # noqa: E402

UINT16_SCALE = 65535.0
PIXEL_MICRONS = 0.065


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _time_call(fn, repeats: int) -> dict:
    """Wall-time statistics (ms) over `repeats` calls (fn returns None)."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return {"mean_ms": float(np.mean(samples)), "min_ms": float(np.min(samples)),
            "repeats": repeats}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float("nan")


def _arc_length(points: np.ndarray) -> float:
    if points is None or len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(points, float), axis=0), axis=1)))


# ---------------------------------------------------------------------------
# synthetic data builders
# ---------------------------------------------------------------------------


def make_rb_image(rng: np.random.Generator, size: int, noise: float,
                  n_rods: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Smooth shading + bright rods + gaussian noise, uint16 intensity scale."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    shading = 0.55 + 0.25 * np.sin(yy / size * np.pi) * np.cos(xx / size * np.pi * 0.7)
    img = shading.copy()
    for _ in range(n_rods):
        cy, cx = rng.uniform(0.15, 0.85, 2) * size
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(14, 30)
        wy, wx = np.cos(ang), np.sin(ang)
        along = (yy - cy) * wy + (xx - cx) * wx
        perp = -(yy - cy) * wx + (xx - cx) * wy
        img[(np.abs(along) < length / 2) & (np.abs(perp) < 2.2)] += 0.45
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 1) * UINT16_SCALE, shading * UINT16_SCALE


def torch_reference_background(frame: np.ndarray, radius: int, kind: str) -> np.ndarray:
    """Numpy transcription of momia2 ``external/mosaic_picasso_torch.py``
    ``rolling_ball_fft_background`` (L139-170): same kernel formula, same
    full-FFT convolution with center crop, float32 throughout."""
    size = 2 * radius + 1
    ax = np.arange(size, dtype=np.float32) - float(radius)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    d2 = xx**2 + yy**2
    r2 = np.float32(radius) ** 2
    kernel = (np.sqrt(np.clip(r2 - d2, 0.0, None)) if kind == "ball"
              else np.clip(r2 - d2, 0.0, None))
    kernel = (kernel / (kernel.sum() + 1e-12)).astype(np.float32)
    h, w = frame.shape
    kh, kw = kernel.shape
    pad_h, pad_w = h + kh - 1, w + kw - 1
    f_img = np.fft.rfft2(frame.astype(np.float32), s=(pad_h, pad_w))
    f_ker = np.fft.rfft2(kernel, s=(pad_h, pad_w))
    conv = np.fft.irfft2(f_img * f_ker, s=(pad_h, pad_w))
    return conv[kh // 2 : kh // 2 + h, kw // 2 : kw // 2 + w]


def _segment_mask(tile_shape, cy, cx, ang, length, half_width) -> np.ndarray:
    yy, xx = np.mgrid[0 : tile_shape[0], 0 : tile_shape[1]].astype(np.float64)
    wy, wx = np.cos(ang), np.sin(ang)
    along = (yy - cy) * wy + (xx - cx) * wx
    perp = -(yy - cy) * wx + (xx - cx) * wy
    return (np.abs(along) < length / 2) & (np.abs(perp) <= half_width)


def _banana_mask(tile_shape, cy, cx, ang, arc_radius, arc_len,
                 half_width, phase1=0.0, phase2=0.0) -> np.ndarray:
    """Smoothly curved rod (circular-arc centerline) as a boolean mask.

    The half-width is modulated by two sines along the arc (wavelengths
    arc_len/3 and arc_len/5, combined amplitude ~1.5 px). This is load-bearing
    for the momia2 comparison: momia2 simplifies the skeleton with
    ``simplify_polygon(tolerance=0.9)`` and its ``find_poles``
    (utils/midline.py L301) then requires >= 9 retained points. Perfectly
    straight or purely circular rods simplify to ~4 points and crash momia2
    with ``IndexError`` (observed during benchmark development; momia3
    handles the same masks without error). The width lumps survive momia2's
    binary_opening and keep the centerline wiggling above the 0.9 px
    tolerance, as real cell masks do.
    """
    yy, xx = np.mgrid[0 : tile_shape[0], 0 : tile_shape[1]].astype(np.float64)
    # circle centre sits perpendicular to the rod's mean direction
    ny, nx = -np.sin(ang), np.cos(ang)
    oy, ox = cy + ny * arc_radius, cx + nx * arc_radius
    r = np.hypot(yy - oy, xx - ox)
    theta = np.arctan2(yy - oy, xx - ox)
    theta0 = np.arctan2(cy - oy, cx - ox)
    dtheta = np.angle(np.exp(1j * (theta - theta0)))
    s = dtheta * arc_radius  # signed arc position (px)
    width = half_width * (
        1.0
        + 0.28 * np.sin(2 * np.pi * 3 * s / arc_len + phase1)
        + 0.18 * np.sin(2 * np.pi * 5 * s / arc_len + phase2)
    )
    return (np.abs(r - arc_radius) <= width) & (np.abs(s) <= arc_len / 2)


def make_midline_masks(rng: np.random.Generator, n_capsule: int = 25,
                       n_bent: int = 25, tile: int = 128,
                       ncols: int = 10) -> tuple[np.ndarray, list[np.ndarray]]:
    """Curved rods on 128^2 tiles, packed into one labels grid.

    Masks must be *smoothly curved* (circular arcs), not straight capsules or
    kinked segments: momia2's skeleton simplification decimates straight/
    piecewise-straight skeletons to <= 4 points and its ``find_poles``
    (utils/midline.py L301) then crashes with ``IndexError`` (it requires
    >= 9 smoothed-skeleton points). Real cell masks are always curved enough;
    these arcs reproduce that property. Observed on a 171-px bent rod during
    benchmark development; momia3 handled the same masks without error.
    """
    masks: list[np.ndarray] = []
    for _ in range(n_capsule):  # gently curved
        masks.append(_banana_mask((tile, tile), tile / 2, tile / 2,
                                  rng.uniform(0, np.pi),
                                  rng.uniform(200, 400), rng.uniform(30, 60),
                                  rng.uniform(2.8, 3.6),
                                  rng.uniform(0, 6.28), rng.uniform(0, 6.28)))
    for _ in range(n_bent):  # strongly curved
        masks.append(_banana_mask((tile, tile), tile / 2, tile / 2,
                                  rng.uniform(0, np.pi),
                                  rng.uniform(50, 140), rng.uniform(30, 55),
                                  rng.uniform(2.8, 3.6),
                                  rng.uniform(0, 6.28), rng.uniform(0, 6.28)))
    nrows = int(np.ceil(len(masks) / ncols))
    labels = np.zeros((nrows * tile, ncols * tile), np.int32)
    for i, m in enumerate(masks):
        r, c = divmod(i, ncols)
        labels[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile][m] = i + 1
    return labels, masks


def make_overlap_pair(rng: np.random.Generator, size: int = 2048,
                      n_cells: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Grid-placed random rods; frame2 = per-label 1-px dilation."""
    from scipy import ndimage

    grid = int(np.ceil(np.sqrt(n_cells)))
    step = size // grid
    f1 = np.zeros((size, size), np.int32)
    label = 0
    for gy in range(grid):
        for gx in range(grid):
            if label >= n_cells:
                break
            label += 1
            cy = gy * step + step / 2 + rng.uniform(-6, 6)
            cx = gx * step + step / 2 + rng.uniform(-6, 6)
            m = _segment_mask((size, size), cy, cx, rng.uniform(0, np.pi),
                              rng.uniform(25, 60), rng.uniform(3.0, 4.5))
            f1[m] = label
    f2 = np.zeros_like(f1)
    for lb in range(1, label + 1):
        dil = ndimage.binary_dilation(f1 == lb, iterations=1)
        f2[dil & (f2 == 0)] = lb
    return f1, f2


# ---------------------------------------------------------------------------
# group 1: rolling ball
# ---------------------------------------------------------------------------


def bench_rolling_ball(ref, repeats: int = 3) -> dict:
    from momia3.ops.preprocess import RollingBallParams, rolling_ball_background_fft

    rng = np.random.default_rng(7)
    out: dict = {"repeats": repeats}

    # --- correctness -------------------------------------------------------
    frame, shading = make_rb_image(rng, 1024, noise=0.05)
    fidelity = {}
    for kind in ("parabola", "ball"):
        bg_ref = torch_reference_background(frame, 50, kind)
        bg3 = frame.astype(np.float32) - rolling_ball_background_fft(
            frame, RollingBallParams(radius=50, kind=kind, shrink_factor=None))
        fidelity[kind] = {
            "pearson": _pearson(bg_ref, bg3),
            "max_abs_diff": float(np.abs(bg_ref - bg3).max()),
            "max_rel_diff": float(np.abs(bg_ref - bg3).max() / UINT16_SCALE),
        }
    out["fidelity_vs_momia2_fft_reference"] = fidelity

    crop = np.s_[50:-50, 50:-50]
    cross = {}
    for noise in (0.05, 0.002):
        fr, sh = make_rb_image(rng, 1024, noise=noise)
        bg3 = fr.astype(np.float32) - rolling_ball_background_fft(
            fr, RollingBallParams(radius=50, kind="parabola", shrink_factor=None))
        bg3s = fr.astype(np.float32) - rolling_ball_background_fft(
            fr, RollingBallParams(radius=50, kind="parabola", shrink_factor=4))
        bg2 = _momia2_production_background(ref, fr, 50)
        cross[str(noise)] = {
            "pearson_full": _pearson(bg2, bg3),
            "pearson_interior": _pearson(bg2[crop], bg3[crop]),
            "pearson_interior_shrink": _pearson(bg2[crop], bg3s[crop]),
            "momia2_vs_true_shading": _pearson(bg2, sh),
            "momia3_vs_true_shading_interior": _pearson(bg3[crop], sh[crop]),
        }
    out["cross_estimator_vs_momia2_production"] = cross

    # --- production timing grid (momia3 default path = shrink_factor 4) -----
    prod = []
    for size in (512, 1024, 2048):
        for radius in (25, 50, 100):
            fr, _ = make_rb_image(rng, size, noise=0.02)
            img_u16 = fr.astype(np.uint16)
            params = RollingBallParams(radius=radius, kind="parabola")
            params_exact = RollingBallParams(
                radius=radius, kind="parabola", shrink_factor=None)
            ref.corrections.rolling_ball_bg_subtraction(
                img_u16, rolling_ball_radius=radius)  # warmup
            rolling_ball_background_fft(fr, params)
            rolling_ball_background_fft(fr, params_exact)
            t2 = _time_call(lambda: ref.corrections.rolling_ball_bg_subtraction(
                img_u16, rolling_ball_radius=radius), repeats)
            t3 = _time_call(lambda: rolling_ball_background_fft(fr, params),
                            repeats)
            t3e = _time_call(
                lambda: rolling_ball_background_fft(fr, params_exact), repeats)
            prod.append({"size": size, "radius": radius,
                         "momia2_ms": t2["mean_ms"], "momia3_ms": t3["mean_ms"],
                         "momia3_exact_ms": t3e["mean_ms"],
                         "shrink_factor": params.shrink_factor,
                         "speedup": t2["mean_ms"] / t3["mean_ms"]})
    out["production_timing"] = prod

    # --- production path breakdown (1024^2, r=50) ---------------------------
    from skimage import filters, transform
    fr, _ = make_rb_image(rng, 1024, noise=0.02)
    img_u16 = fr.astype(np.uint16)
    corr = ref.corrections
    ball = corr.rolling_ball(radius=50)

    def _stage_times():
        stages = {}
        t0 = time.perf_counter()
        smoothed = filters.gaussian(img_u16, sigma=1) * 65535
        stages["gaussian_x65535"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        shrunk = corr.shrink_img_local_min(smoothed, shrink_factor=ball.shrink_factor)
        stages["shrink_local_min"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        bg = corr.rolling_ball_bg(shrunk, ball.ball)
        stages["numba_kernel_on_shrunk"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        transform.resize(bg, img_u16.shape, anti_aliasing=True)
        stages["upsample_resize"] = time.perf_counter() - t0
        return stages

    _stage_times()  # warmup
    out["production_breakdown_s"] = _stage_times()

    # --- momia3 production path stage breakdown (1024^2, r=50, shrink=4) ----
    from scipy import signal as _signal

    from momia3.ops.preprocess import (
        _rolling_ball_kernel,
        _shrink_min_nb,
        _upsample_subtract_nb,
    )
    f32 = fr.astype(np.float32)
    kernel_s = _rolling_ball_kernel(12, "parabola")
    shrunk3 = np.empty((256, 256), dtype=np.float32)
    bg_s = np.ascontiguousarray(
        _signal.fftconvolve(shrunk3, kernel_s, mode="same"), dtype=np.float32)
    out3 = np.empty_like(f32)
    _shrink_min_nb(f32, 4, shrunk3)  # JIT warmup
    _upsample_subtract_nb(bg_s, f32, out3)
    out["momia3_production_breakdown_ms"] = {
        "shrink_local_min": _time_call(
            lambda: _shrink_min_nb(f32, 4, shrunk3), repeats)["mean_ms"],
        "fftconvolve_shrunk": _time_call(
            lambda: _signal.fftconvolve(shrunk3, kernel_s, mode="same"),
            repeats)["mean_ms"],
        "upsample_subtract": _time_call(
            lambda: _upsample_subtract_nb(bg_s, f32, out3), repeats)["mean_ms"],
    }

    # --- raw numba kernel, true radius (1 repeat, JIT-warmed) ---------------
    raw = []
    for radius in (25, 50, 100):
        fr, _ = make_rb_image(rng, 1024, noise=0.02)
        ball1 = corr.rolling_ball(radius=radius)
        ball1.shrink_factor = 1
        ball1.radius = float(radius)
        ball1.build()
        corr.rolling_ball_bg(fr, ball1.ball)  # warmup/JIT
        t0 = time.perf_counter()
        corr.rolling_ball_bg(fr, ball1.ball)
        t2 = (time.perf_counter() - t0) * 1e3
        params = RollingBallParams(radius=radius, kind="parabola",
                                   shrink_factor=None)
        rolling_ball_background_fft(fr, params)
        t3 = _time_call(lambda: rolling_ball_background_fft(fr, params), repeats)
        raw.append({"size": 1024, "radius": radius, "repeats": 1,
                    "momia2_ms": t2, "momia3_ms": t3["mean_ms"],
                    "speedup": t2 / t3["mean_ms"]})
    out["raw_kernel_timing"] = raw

    # --- shrink-path fidelity contract (default shrink=4 vs exact path) -----
    sframe, sshading = make_rb_image(np.random.default_rng(107), 1024, noise=0.05)
    f32 = sframe.astype(np.float32)
    bg_exact = f32 - rolling_ball_background_fft(
        sframe, RollingBallParams(radius=50, kind="parabola", shrink_factor=None))
    bg_fast = f32 - rolling_ball_background_fft(
        sframe, RollingBallParams(radius=50, kind="parabola", shrink_factor=4))
    crop = np.s_[50:-50, 50:-50]
    out["shrink_path_correctness"] = {
        "shrink_factor": 4,
        "pearson_full_vs_exact": _pearson(bg_fast, bg_exact),
        "pearson_interior_vs_exact": _pearson(bg_fast[crop], bg_exact[crop]),
        "max_abs_diff": float(np.abs(bg_fast - bg_exact).max()),
        "exact_vs_true_shading_interior": _pearson(bg_exact[crop],
                                                   sshading[crop]),
        "shrink_vs_true_shading_interior": _pearson(bg_fast[crop],
                                                    sshading[crop]),
    }
    return out


def _momia2_production_background(ref, frame_u16scale: np.ndarray,
                                  radius: int) -> np.ndarray:
    """momia2 ``rolling_ball_bg_subtraction`` internals, capturing bg_rescaled."""
    from skimage import filters, transform

    corr = ref.corrections
    smoothed = filters.gaussian(frame_u16scale, sigma=1) * 65535
    ball = corr.rolling_ball(radius=radius)
    shrunk = corr.shrink_img_local_min(smoothed, shrink_factor=ball.shrink_factor)
    bg = corr.rolling_ball_bg(shrunk, ball.ball)
    return transform.resize(bg, frame_u16scale.shape, anti_aliasing=True)


# ---------------------------------------------------------------------------
# group 2: midline
# ---------------------------------------------------------------------------


def bench_midline(ref, repeats: int = 3) -> dict:
    from momia3.ops.midline import extract_midlines

    rng = np.random.default_rng(13)
    labels, masks = make_midline_masks(rng)
    pm = PIXEL_MICRONS

    def momia2_one(m: np.ndarray):
        """Main-branch arc length (um), or None if momia2 crashes on the mask.

        momia2 is fragile here: simplify_polygon(tolerance=0.9) can decimate
        smooth skeletons below the 9 points find_poles requires, and pruning
        can empty the skeleton — both raise IndexError (utils/midline.py
        L301). Failed masks are excluded from both engines' comparison and
        counted in ``momia2_failures`` (momia3 processes all 50).
        """
        try:
            outline = ref.contour.find_contour_marching_squares(m)
            mid = ref.midline.extract_midline(m, outline, pixel_microns=pm,
                                              method="zhang")
            main = max(mid, key=_arc_length) if len(mid) else np.empty((0, 2))
            return _arc_length(main) * pm
        except Exception:
            return None

    def momia2_pass() -> list[float | None]:
        return [momia2_one(m) for m in masks]

    def momia3_pass() -> list[float]:
        res = extract_midlines(labels)
        out = []
        for i in range(len(masks)):
            branches = res[i + 1]["midline"]  # list of (N, 2) polylines
            main = max(branches, key=_arc_length) if branches else np.empty((0, 2))
            out.append(_arc_length(np.asarray(main, float)) * pm)
        return out

    def momia3_tiles_pass() -> list[float]:
        """momia3 called per 128^2 tile (momia2-equivalent semantics).

        The packed-grid batch call is momia3's production entry point, but it
        pays O(n_labels x image_pixels) in per-label full-image ``nonzero``
        scans (67% of batch time in profiling). Per-tile calls isolate the
        pure midline algorithm cost for a fair per-mask comparison.
        """
        tile = masks and 128
        out = []
        for i, m in enumerate(masks):
            r, c = divmod(i, 10)
            sub = labels[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile]
            res = extract_midlines(sub)
            branches = res[i + 1]["midline"]
            main = max(branches, key=_arc_length) if branches else np.empty((0, 2))
            out.append(_arc_length(np.asarray(main, float)) * pm)
        return out

    len2 = momia2_pass()  # warmup + reference lengths
    len3 = momia3_pass()
    len3t = momia3_tiles_pass()
    t2 = _time_call(momia2_pass, repeats)
    t3 = _time_call(momia3_pass, repeats)
    t3t = _time_call(momia3_tiles_pass, repeats)

    # correctness on the packed-grid batch result (per-tile agrees to float
    # precision; both are momia3's own algorithm)
    pairs = [(a, b) for a, b in zip(len2, len3) if a is not None and b > 0]
    tile_agree = float(np.mean([
        abs(x - y) <= 1e-6 * max(1.0, abs(x)) for x, y in zip(len3, len3t)
    ]))
    a = np.asarray([p[0] for p in pairs])
    b = np.asarray([p[1] for p in pairs])
    rel = np.abs(a - b) / np.maximum(a, 1e-9)
    return {
        "n_masks": len(masks),
        "momia2_failures": sum(1 for v in len2 if v is None),
        "compared_masks": len(pairs),
        "repeats": repeats,
        "correctness": {
            "length_pearson": _pearson(a, b),
            "length_median_rel_diff": float(np.median(rel)),
            "length_max_rel_diff": float(np.max(rel)),
        },
        "momia2_ms_per_pass": t2["mean_ms"],
        "momia3_ms_per_pass": t3["mean_ms"],
        "speedup": t2["mean_ms"] / t3["mean_ms"],
        "momia3_tile_ms_per_pass": t3t["mean_ms"],
        "speedup_per_tile": t2["mean_ms"] / t3t["mean_ms"],
        "batch_vs_tile_agreement": tile_agree,
        "momia2_ms_per_mask": t2["mean_ms"] / len(masks),
        "momia3_ms_per_mask": t3["mean_ms"] / len(masks),
    }


# ---------------------------------------------------------------------------
# group 3: overlap matrix
# ---------------------------------------------------------------------------


def bench_overlap(ref, repeats: int = 3) -> dict:
    from momia3.ops.track import overlap_matrix

    rng = np.random.default_rng(29)
    f1, f2 = make_overlap_pair(rng)

    def momia2_pass():
        return ref.tracker._calculate_overlap_matrix(f1, f2, "0", "1")

    def momia3_pass():
        return overlap_matrix(f1, f2)

    df = momia2_pass()  # warmup
    coo = momia3_pass()
    t2 = _time_call(momia2_pass, repeats)
    t3 = _time_call(momia3_pass, repeats)

    ref_pairs = {
        (int(a.split("_")[1]), int(b.split("_")[1])): int(o)
        for a, b, o in zip(df["frame1_id"], df["frame2_id"], df["overlap_area"])
    }
    got_pairs = {(int(r), int(c)): int(v)
                 for r, c, v in zip(coo.row, coo.col, coo.data)}
    return {
        "size": int(f1.shape[0]),
        "n_labels": int(f1.max()),
        "repeats": repeats,
        "correctness": {
            "n_pairs_momia2": len(ref_pairs),
            "n_pairs_momia3": len(got_pairs),
            "pairs_exact_match": ref_pairs == got_pairs,
        },
        "momia2_ms": t2["mean_ms"],
        "momia3_ms": t3["mean_ms"],
        "speedup": t2["mean_ms"] / t3["mean_ms"],
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_all(repeats: int = 3) -> dict:
    ref = load_momia2()
    results: dict = {"repeats": repeats, "groups": {}}
    for name, fn in (("rolling_ball", bench_rolling_ball),
                     ("midline", bench_midline),
                     ("overlap", bench_overlap)):
        try:
            results["groups"][name] = fn(ref, repeats=repeats)
        except Exception as exc:  # record, don't abort the other groups
            results["groups"][name] = {"error": f"{type(exc).__name__}: {exc}"}
    return results


if __name__ == "__main__":
    import json

    res = run_all(repeats=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    out = REPO_ROOT / "benchmarks" / "benchmark_results.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")
    print(json.dumps(res, indent=1)[:3000])
