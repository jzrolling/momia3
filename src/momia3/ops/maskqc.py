"""Mask QC for instance label images and stacks."""

from __future__ import annotations

import numpy as np

__all__ = ["mask_qc"]


def mask_qc(labels, min_area=10.0, max_area=None, border_width=1, max_listed=20):
    """Return a JSON-friendly QC report for a 2-D label image or 3-D stack."""
    if min_area < 0:
        raise ValueError("min_area must be >= 0")
    if max_area is not None and max_area < min_area:
        raise ValueError("max_area must be >= min_area")
    if border_width < 1:
        raise ValueError("border_width must be >= 1")
    arr = np.asarray(labels)
    if arr.ndim == 2:
        stack = arr[None, ...]
    elif arr.ndim == 3:
        stack = arr
    else:
        raise ValueError(f"labels must be 2-D or 3-D, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"labels must be an integer dtype, got {arr.dtype}")

    frames = []
    issues = []
    total_cells = 0
    for t in range(stack.shape[0]):
        frame = stack[t]
        h, w = frame.shape
        flat = frame.ravel()
        ids, counts = np.unique(flat[flat > 0], return_counts=True)
        n_cells = int(ids.size)
        total_cells += n_cells
        max_id = int(ids.max()) if n_cells else 0
        missing = []
        if n_cells:
            present = set(int(x) for x in ids.tolist())
            missing = [i for i in range(1, max_id + 1) if i not in present]
        bw = min(int(border_width), h, w)
        border_band = np.concatenate(
            [
                frame[:bw, :].ravel(),
                frame[-bw:, :].ravel(),
                frame[:, :bw].ravel(),
                frame[:, -bw:].ravel(),
            ]
        )
        border_ids = np.unique(border_band[border_band > 0])
        small = ids[counts < min_area] if n_cells else np.array([], dtype=ids.dtype)
        large = (
            ids[counts > max_area]
            if (max_area is not None and n_cells)
            else np.array([], dtype=ids.dtype)
        )
        frame_issues = []
        if n_cells == 0:
            frame_issues.append("empty_frame")
        if missing:
            frame_issues.append("non_contiguous_label_ids")
        if small.size:
            frame_issues.append("small_objects")
        if large.size:
            frame_issues.append("large_objects")
        if border_ids.size:
            frame_issues.append("border_touching_objects")
        issues.extend(f"frame_{t}:{issue}" for issue in frame_issues)
        frames.append(
            {
                "frame": int(t),
                "n_cells": n_cells,
                "empty": bool(n_cells == 0),
                "max_label_id": max_id,
                "missing_label_count": int(len(missing)),
                "missing_label_ids": missing[:max_listed],
                "border_object_count": int(border_ids.size),
                "border_label_ids": sorted(int(x) for x in border_ids.tolist())[:max_listed],
                "small_object_count": int(small.size),
                "small_label_ids": sorted(int(x) for x in small.tolist())[:max_listed],
                "large_object_count": int(large.size),
                "large_label_ids": sorted(int(x) for x in large.tolist())[:max_listed],
                "area_mean": float(counts.mean()) if n_cells else 0.0,
                "area_median": float(np.median(counts)) if n_cells else 0.0,
                "area_min": int(counts.min()) if n_cells else 0,
                "area_max": int(counts.max()) if n_cells else 0,
                "issues": frame_issues,
            }
        )
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "n_frames": int(stack.shape[0]),
        "total_cells": int(total_cells),
        "min_area": float(min_area),
        "max_area": None if max_area is None else float(max_area),
        "ok": not issues,
        "issues": issues,
        "frames": frames,
    }
