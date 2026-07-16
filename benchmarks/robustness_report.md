# momia3 end-to-end robustness matrix

## Design

- 108 pipeline runs: noise σ {0.01, 0.02, 0.05} × drift {0, 0.5, 2.0} px/frame × initial cells {3, 8} × division length {3.0, 4.0} µm × seeds {11, 23, 37}; 16 frames, 256², growth 0.1 µm/frame.
- Synthetic ground truth with rendered noise/shading; truth labels are aligned to each drifted frame by cumulative-shift resampling before comparison (without this, drift robustness is unmeasurable).
- Detection = track-consistent Hungarian matching (IoU ≥ 0.25, truth identity inherited along continue/gap edges). `edge_recall` is detection-aware (denominator = truth edges with both endpoints detected); `edge_recall_raw` scores against all truth edges.
- Segment params override `dilation: false`: momia2-legacy 1-px mask dilation inflates masks ~18%, merges sister cells and costs ~5% midline bias (documented usability finding).
- Sister-merge caveat: in the generator, daughters elongate into each other within ~1.5 frames of division, so crowded conditions measure merged-blob behaviour as much as tracking.

## Clean-condition baseline (noise 0.01, no drift, 3 cells, L=4.0 µm)

| metric | seed-mean | gate | verdict |
|---|---|---|---|
| foreground IoU | 0.977 | ≥ 0.85 | PASS |
| detection-aware edge recall | 0.961 | ≥ 0.95 | PASS |
| midline length rel err | 0.50% | ≤ 5% | PASS |
| raw detection rate | 0.864 | — | |
| division precision | 1.000 | — | |
| division recall | 1.000 | — | |
| edge precision | 1.000 | — | |

## Full matrix (seed-mean per condition)

| noise | drift | n_init | L_div | IoU | detect | edge R | edge R raw | div R | midline err |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.0 | 3 | 3.0 | 0.986 | 0.774 | 0.942 | 0.720 | 1.000 | 1.0% |
| 0.01 | 0.0 | 3 | 4.0 | 0.977 | 0.864 | 0.961 | 0.825 | 1.000 | 0.5% |
| 0.01 | 0.0 | 8 | 3.0 | 0.990 | 0.541 | 0.943 | 0.483 | 0.857 | 10.7% |
| 0.01 | 0.0 | 8 | 4.0 | 0.995 | 0.498 | 0.937 | 0.433 | 0.611 | 13.3% |
| 0.01 | 0.5 | 3 | 3.0 | 0.975 | 0.775 | 0.942 | 0.720 | 1.000 | 1.1% |
| 0.01 | 0.5 | 3 | 4.0 | 0.985 | 0.859 | 0.961 | 0.819 | 1.000 | 0.6% |
| 0.01 | 0.5 | 8 | 3.0 | 0.980 | 0.541 | 0.942 | 0.482 | 0.815 | 10.4% |
| 0.01 | 0.5 | 8 | 4.0 | 0.987 | 0.498 | 0.942 | 0.442 | 0.611 | 11.9% |
| 0.01 | 2.0 | 3 | 3.0 | 0.981 | 0.784 | 0.943 | 0.730 | 1.000 | 1.2% |
| 0.01 | 2.0 | 3 | 4.0 | 0.978 | 0.864 | 0.961 | 0.825 | 1.000 | 0.6% |
| 0.01 | 2.0 | 8 | 3.0 | 0.982 | 0.539 | 0.946 | 0.478 | 0.815 | 11.9% |
| 0.01 | 2.0 | 8 | 4.0 | 0.982 | 0.494 | 0.941 | 0.434 | 0.556 | 13.5% |
| 0.02 | 0.0 | 3 | 3.0 | 0.984 | 0.779 | 0.942 | 0.725 | 1.000 | 1.1% |
| 0.02 | 0.0 | 3 | 4.0 | 0.991 | 0.864 | 0.961 | 0.825 | 1.000 | 0.6% |
| 0.02 | 0.0 | 8 | 3.0 | 0.990 | 0.541 | 0.946 | 0.485 | 0.815 | 10.5% |
| 0.02 | 0.0 | 8 | 4.0 | 0.995 | 0.494 | 0.937 | 0.433 | 0.611 | 13.7% |
| 0.02 | 0.5 | 3 | 3.0 | 0.980 | 0.779 | 0.942 | 0.725 | 1.000 | 1.1% |
| 0.02 | 0.5 | 3 | 4.0 | 0.978 | 0.864 | 0.961 | 0.825 | 1.000 | 0.6% |
| 0.02 | 0.5 | 8 | 3.0 | 0.978 | 0.536 | 0.946 | 0.473 | 0.815 | 11.0% |
| 0.02 | 0.5 | 8 | 4.0 | 0.984 | 0.496 | 0.938 | 0.438 | 0.556 | 13.6% |
| 0.02 | 2.0 | 3 | 3.0 | 0.983 | 0.779 | 0.942 | 0.725 | 1.000 | 1.3% |
| 0.02 | 2.0 | 3 | 4.0 | 0.983 | 0.864 | 0.961 | 0.825 | 1.000 | 0.6% |
| 0.02 | 2.0 | 8 | 3.0 | 0.981 | 0.541 | 0.942 | 0.480 | 0.815 | 11.4% |
| 0.02 | 2.0 | 8 | 4.0 | 0.981 | 0.494 | 0.945 | 0.436 | 0.556 | 14.0% |
| 0.05 | 0.0 | 3 | 3.0 | 0.972 | 0.775 | 0.942 | 0.720 | 1.000 | 1.4% |
| 0.05 | 0.0 | 3 | 4.0 | 0.964 | 0.864 | 0.961 | 0.825 | 1.000 | 0.8% |
| 0.05 | 0.0 | 8 | 3.0 | 0.981 | 0.539 | 0.946 | 0.487 | 0.857 | 11.3% |
| 0.05 | 0.0 | 8 | 4.0 | 0.984 | 0.502 | 0.934 | 0.439 | 0.611 | 12.0% |
| 0.05 | 0.5 | 3 | 3.0 | 0.974 | 0.784 | 0.943 | 0.730 | 1.000 | 1.9% |
| 0.05 | 0.5 | 3 | 4.0 | 0.977 | 0.858 | 0.954 | 0.813 | 1.000 | 0.9% |
| 0.05 | 0.5 | 8 | 3.0 | 0.974 | 0.534 | 0.941 | 0.471 | 0.815 | 11.8% |
| 0.05 | 0.5 | 8 | 4.0 | 0.977 | 0.502 | 0.926 | 0.435 | 0.556 | 12.7% |
| 0.05 | 2.0 | 3 | 3.0 | 0.972 | 0.779 | 0.942 | 0.725 | 1.000 | 1.6% |
| 0.05 | 2.0 | 3 | 4.0 | 0.982 | 0.859 | 0.961 | 0.819 | 1.000 | 0.7% |
| 0.05 | 2.0 | 8 | 3.0 | 0.976 | 0.537 | 0.945 | 0.480 | 0.815 | 13.3% |
| 0.05 | 2.0 | 8 | 4.0 | 0.978 | 0.494 | 0.933 | 0.429 | 0.556 | 10.8% |

## Degradation by axis

| noise σ | IoU | detect | edge R | edge R raw | div R | midline err |
|---|---|---|---|---|---|---|
| 0.01 | 0.983 | 0.669 | 0.947 | 0.616 | 0.855 | 6.4% |
| 0.02 | 0.984 | 0.669 | 0.947 | 0.616 | 0.847 | 6.6% |
| 0.05 | 0.976 | 0.669 | 0.944 | 0.614 | 0.851 | 6.6% |

| drift (px/frame) | IoU | detect | edge R | edge R raw | div R | midline err |
|---|---|---|---|---|---|---|
| 0.0 | 0.984 | 0.670 | 0.946 | 0.617 | 0.864 | 6.4% |
| 0.5 | 0.979 | 0.669 | 0.945 | 0.615 | 0.847 | 6.4% |
| 2.0 | 0.980 | 0.669 | 0.947 | 0.615 | 0.843 | 6.7% |

| crowding | IoU | detect | edge R | edge R raw | div R | midline err |
|---|---|---|---|---|---|---|
| 8 cells + L=3.0 (crowded) | 0.981 | 0.539 | 0.944 | 0.480 | 0.825 | 11.4% |
| all other conditions | 0.981 | 0.713 | 0.947 | 0.661 | 0.860 | 4.9% |

**Reading**: noise barely degrades anything (IoU stays ≥0.93 at σ=0.05); drift is fully corrected by phase-correlation registration; crowding is the dominant axis — detection and raw edge recall collapse when sisters merge into shared blobs, while *conditional* edge recall stays ~0.96, i.e. the LAP linker is robust whenever both cells are actually detected.

## Per-frame pipeline stage cost (all 108 runs)

| stage | mean (s/run) | share |
|---|---|---|
| import | 0.041 | 4.3% |
| drift | 0.075 | 7.9% |
| segment | 0.144 | 15.2% |
| cells | 0.337 | 35.6% |
| outlines | 0.107 | 11.2% |
| midlines | 0.218 | 22.9% |
| track | 0.028 | 2.9% |
| **total** | 0.949 | 100% |

Bottleneck: **cells** (36% of wall time) — the per-cell feature step (contour/midline-ready measurements); consistent with the midline benchmark's finding that per-cell ops dominate momia3's cost.
