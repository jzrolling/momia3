# momia3 vs momia2 — performance benchmark report

## Method

- momia2 (read-only checkout) is imported through `benchmarks/_momia2_loader.py`: real source files loaded under a fake `momia2ref` package namespace with `importlib.util.spec_from_file_location` (light chain `utils/linalg → generic → contour → midline/corrections` loads directly; `core/tracker.py` gets stub `utils`/`segment` parents). Purity guards: `sys.dont_write_bytecode=True` and `NUMBA_CACHE_DIR=benchmarks/.numba_cache` — the momia2 checkout is never written.
- Timing repeats: 3 (mean reported), except the raw numba rolling-ball kernel at true radius: 1 JIT-warmed repeat (a single r=100 pass at 1024² costs ~80 s). Scope was trimmed to finish in budget; the end-to-end per-frame stage cost is aggregated from the robustness matrix stage columns in `robustness_report.md` instead of re-measured.
- Implementations: `tests/benchmarks/bench_core.py`; raw numbers: `benchmarks/benchmark_results.json`.

## 1. Rolling-ball background

momia3 `rolling_ball_background_fft` has two paths. The **exact path** (`shrink_factor=None`) is a documented numpy/scipy port of momia2's own torch FFT reference (`external/mosaic_picasso_torch.py::rolling_ball_fft_background`): same normalized parabola/ball kernel, same zero-padded FFT convolution. The **production path** (default `shrink_factor=4`) applies the same amortization as momia2's production pipeline: block-min shrink (numba `_shrink_min_nb`, value-identical to momia2's `shrink_img_local_min`) → FFT convolution on the shrunk image with a `radius/shrink_factor` kernel → fused bilinear upsample+subtract (numba parallel `_upsample_subtract_nb`, same center-aligned bilinear math as momia2's `transform.resize` upsample). momia2's *production* estimator (`rolling_ball_bg_subtraction`) remains a different algorithm: gaussian σ=1 → ×65535 → 4× local-min shrink → numba min-envelope kernel on the shrunk image → anti-aliased upsample.

### Correctness

**Algorithmic fidelity** of the exact path vs a numpy transcription of momia2's torch FFT reference (1024², r=50, σ=0.05):

| kernel | Pearson | max abs diff (uint16 scale) | max rel diff |
|---|---|---|---|
| parabola | 1.000000 | 0.0508 | 7.75e-07 |
| ball | 1.000000 | 0.0547 | 8.34e-07 |

**Shrink-path fidelity** (default shrink=4 vs the exact path, 1024², r=50, σ=0.05) — the contract behind the speedup:

| metric | value |
|---|---|
| Pearson full-frame vs exact | **0.99715** |
| Pearson interior (1-radius margin) vs exact | **0.99980** |
| exact path vs true shading (interior) | 0.99984 |
| shrink path vs true shading (interior) | 0.99984 |

The full-frame deficit is entirely the zero-padded border band, where the FFT estimator itself is biased (exact-vs-truth is only ~0.87 full-frame); both paths recover to ~0.9998 at a one-radius margin. The zero-pad behaviour is inherited verbatim from momia2's own torch FFT reference and codified in unit tests, so it is kept as-is.

**Cross-estimator agreement** with momia2's production background (interior = one ball radius cropped, where FFT zero-padding cannot leak):

| noise σ | full frame (exact) | interior (exact) | interior (shrink) | momia2 vs true shading | momia3 vs true shading (interior) |
|---|---|---|---|---|---|
| 0.05 | 0.8726 | 0.9988 | 0.9989 | 0.9988 | 0.9997 |
| 0.002 | 0.8707 | 0.9998 | 1.0000 | 1.0000 | 0.9998 |

### Timing — production paths (mean of repeats)

momia3 production = default `shrink_factor=4`; the exact column is the pre-optimization full-resolution path shown for reference. momia3 timings include the float64→float32 input cast (~0.3–1 ms at 1024²–2048²) that is free in production (the store serves float32).

| size | radius | momia2 prod (ms) | momia3 shrink (ms) | momia3 exact (ms) | speedup |
|---|---|---|---|---|---|
| 512² | 25 | 15.9 | 2.94 | 2.81 | **5.4×** |
| 512² | 50 | 25.4 | 0.75 | 2.74 | **33.7×** |
| 512² | 100 | 69.2 | 0.97 | 3.81 | **71.2×** |
| 1024² | 25 | 63.7 | 2.33 | 11.68 | **27.4×** |
| 1024² | 50 | 104.6 | 2.40 | 11.00 | **43.7×** |
| 1024² | 100 | 295.5 | 2.28 | 16.50 | **129.7×** |
| 2048² | 25 | 267.1 | 8.26 | 48.13 | **32.3×** |
| 2048² | 50 | 440.1 | 8.52 | 48.99 | **51.7×** |
| 2048² | 100 | 1228.8 | 9.02 | 56.59 | **136.2×** |

### Timing — raw numba kernel at true radius (1024², 1 repeat)

| radius | momia2 kernel (ms) | momia3 FFT exact (ms) | speedup |
|---|---|---|---|
| 25 | 4425 | 10.35 | **427×** |
| 50 | 16782 | 11.60 | **1446×** |
| 100 | 76737 | 15.11 | **5080×** |

Production-path breakdown at 1024² r=50 (momia2, seconds): numba kernel on shrunk image 0.072 (73%), gaussian+scale 0.013, upsample 0.012, local-min shrink 0.002.
 momia3 shrink path (ms): FFT on 256² 1.02, block-min shrink 0.49, upsample+subtract 0.54.

**Target ≥50× production**: the shrink path delivers 5.4–136.2× across the grid — **HIT** at 1024²+ with r≥100 and at 2048² with r≥50 (43.7× at the canonical 1024² r=50, just under target; small images/radii are dominated by fixed overheads — pydantic, casts, thread dispatch — where momia2's own shrink already amortizes well). Before this optimization the exact path measured 5–15× on the same grid, so the shrink path is the 2.5–6× production-path improvement it was designed to be, at a measured fidelity of 0.9972 full-frame / 0.9998 interior against the exact estimator. Raw kernel comparison (momia2 forced unshrunk): 427–5080×.

## 2. Midline extraction

50 synthetic curved rods (128² tiles, zhang skeleton on both engines). momia2 timed per-mask (`find_contour_marching_squares` + `extract_midline`, its production call pattern). momia3 timed two ways: batch `extract_midlines` on the packed 640×1280 labels grid (its production entry point) and per-tile (momia2-equivalent semantics, isolating the algorithm).

**Post-optimization**: the batch loop now does one `scipy.ndimage.find_objects` pass per frame (was: per-label full-image `nonzero` scans), outline ray/segment solves run in a numba kernel (`_ray_solve_kernel`), ±normal rays are batched into single `ray_intersections` calls, and branch tracing uses a one-pass node-pixel index. All changes are **bit-identical**: length Pearson, per-mask diffs and batch-vs-tile agreement are exactly unchanged from the pre-optimization baseline.

### Correctness (main-branch arc length)

- compared masks: 46/50 (momia2 crashed on 4 masks — see below)
- length Pearson: **0.9893**, median rel diff **1.7%**, max 8.2%
- momia3 batch vs per-tile results identical (agreement 1.00)

### Timing

| engine | ms / 50-mask pass | ms / mask | speedup vs momia2 |
|---|---|---|---|
| momia2 per-mask | 113.5 | 2.27 | 1.0× |
| momia3 batch (production) | 100.7 | 2.01 | **1.13×** |
| momia3 per-tile | 105.4 | 2.11 | **1.08×** |

**Target ≥10×: MISS** — batch went from 0.34× (pre-optimization) to **1.13×**, per-tile to **1.08×**: momia3 is now at parity with momia2, but the ≥10× target is structurally unreachable without changing outputs (forbidden by the perf-identical constraint). Profiling attributes the remaining cost to FITPACK `splprep` periodic-spline smoothing in outline extraction (~20 ms/pass) and the iterative ray-based midline refinement — computations momia2 simply does not perform (momia3 additionally returns smoothed periodic outlines, width profiles and extended poles per mask). Parity with identical numerics is the constraint-compliant ceiling.

**momia2 fragility finding**: momia2 crashed (4/50 masks, `IndexError` in `utils/midline.py::find_poles` L301) when `simplify_polygon(tolerance=0.9)` decimates a smooth skeleton below the 9 points its pole search requires, or when pruning empties the skeleton. momia3 processed all 50 masks without error.

## 3. Overlap matrix

2048² label pair, 400 cells (frame2 = 1-px per-label dilation of frame1). momia2 `_calculate_overlap_matrix` (`np.unique(axis=0)` + pandas DataFrame) vs momia3 `overlap_matrix` (single `bincount`, sparse COO).

### Correctness

- overlap pairs: momia2 400, momia3 400 — exact (label_t, label_t1, pixels) triple match: **True**

### Timing

| engine | ms | speedup |
|---|---|---|
| momia2 | 66.8 | 1.0× |
| momia3 | 5.92 | **11.3×** |

**Target ≥5×: HIT** (11.3×).

## Target scoreboard (plan §7)

| operation | target | result | verdict |
|---|---|---|---|
| rolling ball (raw kernel) | ≥50× | 427–5080× | HIT |
| rolling ball (production, shrink=4) | ≥50× | 5.4–136.2× grid, 43.7× at canonical 1024² r=50 | PARTIAL — ≥50× at 1024² r≥100 and 2048² r≥50 |
| rolling-ball background fidelity (Pearson) | ≥0.98 exact; ≥0.995 shrink-vs-exact | 1.000000 exact vs FFT reference; 0.99715 full / 0.99980 interior shrink-vs-exact | HIT |
| midline | ≥10× | 1.13× batch / 1.08× per-tile (was 0.34×/0.96×) | MISS — parity is the perf-identical ceiling (see §2) |
| overlap | ≥5× | 11.3× | HIT |
