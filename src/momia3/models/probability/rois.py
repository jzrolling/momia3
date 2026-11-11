"""ImageJ polygon ROI conversion to OmniPose-compatible instance masks."""

from __future__ import annotations

import json
import shutil
import struct
import zipfile
from pathlib import Path

import numpy as np
import tifffile

from momia3.ops.polygon import inside_polygon, smooth_closed_polygon


def read_imagej_polygon(data: bytes) -> np.ndarray:
    """Decode an ImageJ polygon ROI into absolute ``(x, y)`` coordinates."""
    if len(data) < 64 or data[:4] != b"Iout":
        raise ValueError("not an ImageJ ROI payload")
    roi_type = data[6]
    if roi_type != 0:
        raise ValueError(f"only ImageJ polygon ROIs are supported, got type {roi_type}")
    top, left, _bottom, _right = struct.unpack(">hhhh", data[8:16])
    n = struct.unpack(">H", data[16:18])[0]
    if n < 3 or len(data) < 64 + 4 * n:
        raise ValueError(f"invalid polygon coordinate count {n}")
    x = np.frombuffer(data, dtype=">i2", count=n, offset=64).astype(np.float64) + left
    y = np.frombuffer(data, dtype=">i2", count=n, offset=64 + 2 * n).astype(np.float64) + top
    return np.column_stack([x, y])


def read_imagej_roi_zip(path: str | Path) -> list[tuple[str, np.ndarray]]:
    """Read all polygon ROIs from an ImageJ ROI zip in deterministic order."""
    polygons = []
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.filename.endswith(".roi"):
                polygons.append((info.filename, read_imagej_polygon(archive.read(info))))
    if not polygons:
        raise ValueError(f"no .roi entries found in {path}")
    return polygons


def roi_zip_for_image(image: Path) -> Path | None:
    for suffix in ("_rois.zip", "_roi.zip"):
        candidate = image.with_name(image.stem + suffix)
        if candidate.exists():
            return candidate
    return None


def source_series(stem: str) -> str:
    """Keep related acquisition series together in downstream group splits."""
    for prefix in ("NQTF", "Mtb", "Training", "PBPlipo_KD", "PBPlipo_compl", "WT", "eco"):
        if stem.startswith(prefix):
            return prefix
    if stem.lower().startswith(("msm",)):
        return "MSM"
    if stem[:1].isdigit():
        return "numbered"
    return "other"


def _polygon_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return float(0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def rasterize_roi_polygons(
    polygons: list[tuple[str, np.ndarray]],
    shape: tuple[int, int],
    spacing: float = 0.75,
    smooth: float = 1.0,
) -> tuple[np.ndarray, list[dict]]:
    """Smooth polygons and assign overlap pixels to the nearest normalized centre."""
    h, w = shape
    labels = np.zeros(shape, dtype=np.uint32)
    score_map = np.full(shape, np.inf, dtype=np.float32)
    qc = []
    for label_id, (name, polygon) in enumerate(polygons, start=1):
        ring = smooth_closed_polygon(polygon, smooth=smooth, spacing=spacing)
        x0 = max(0, int(np.floor(ring[:, 0].min())) - 1)
        x1 = min(w, int(np.ceil(ring[:, 0].max())) + 2)
        y0 = max(0, int(np.floor(ring[:, 1].min())) - 1)
        y1 = min(h, int(np.ceil(ring[:, 1].max())) + 2)
        entry = {
            "roi": name,
            "label": label_id,
            "n_points_input": len(polygon),
            "n_points_output": len(ring),
            "polygon_area": _polygon_area(ring),
            "clipped": bool(x0 <= 0 or y0 <= 0 or x1 >= w or y1 >= h),
        }
        if x0 >= x1 or y0 >= y1:
            entry.update(mask_area=0, overlap_pixels=0, overlap_fraction=0.0)
            qc.append(entry)
            continue
        local = inside_polygon(
            ring, x0=x0, y0=y0, nx=x1 - x0, ny=y1 - y0, scale=1.0
        )
        yy, xx = np.nonzero(local)
        yy += y0
        xx += x0
        area = len(xx)
        if area == 0:
            entry.update(mask_area=0, overlap_pixels=0, overlap_fraction=0.0)
            qc.append(entry)
            continue
        centre = ring.mean(axis=0)
        scale = max(_polygon_area(ring), 1.0)
        score = ((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / scale
        existing = labels[yy, xx]
        overlap = int((existing > 0).sum())
        replace = (existing == 0) | (score < score_map[yy, xx])
        labels[yy[replace], xx[replace]] = label_id
        score_map[yy[replace], xx[replace]] = score[replace].astype(np.float32)
        entry.update(
            mask_area=area,
            overlap_pixels=overlap,
            overlap_fraction=overlap / max(area, 1),
        )
        qc.append(entry)
    for entry in qc:
        entry["final_area"] = int(np.sum(labels == entry["label"]))
    from momia3.ops.segment import cleanup_labels

    return cleanup_labels(labels), qc


def prepare_roi_dataset(
    source_root: str | Path,
    output_root: str | Path,
    spacing: float = 0.75,
    smooth: float = 1.0,
    exclude_images: dict[str, str] | None = None,
) -> dict:
    """Convert every image/ROI pair and write images, masks and a QC manifest."""
    source_root = Path(source_root)
    output_root = Path(output_root)
    exclude_images = exclude_images or {}
    rows = []
    for image in sorted(source_root.glob("*.tif")):
        if image.stem.endswith(("_masks", "_multilabel")):
            continue
        roi_zip = roi_zip_for_image(image)
        if roi_zip is None:
            raise ValueError(f"no ROI zip paired with {image.name}")
        all_polygons = read_imagej_roi_zip(roi_zip)
        polygons = []
        duplicates = []
        seen = {}
        for name, polygon in all_polygons:
            key = (polygon.shape, polygon.tobytes())
            if key in seen:
                duplicates.append({"roi": name, "duplicate_of": seen[key]})
            else:
                seen[key] = name
                polygons.append((name, polygon))
        pixels = tifffile.imread(image)
        labels, cells = rasterize_roi_polygons(
            polygons, pixels.shape, spacing=spacing, smooth=smooth
        )
        series = source_series(image.stem)
        target_dir = output_root / series
        target_dir.mkdir(parents=True, exist_ok=True)
        target_image = target_dir / image.name
        target_mask = target_dir / f"{image.stem}_masks.tif"
        shutil.copy2(image, target_image)
        tifffile.imwrite(target_mask, labels.astype(np.uint16))
        overlap_pixels = sum(cell["overlap_pixels"] for cell in cells)
        rows.append({
            "image": image.name,
            "source_image": str(image),
            "source_rois": str(roi_zip),
            "series": series,
            "output_image": str(target_image),
            "output_mask": str(target_mask),
            "shape": list(pixels.shape),
            "n_rois": len(all_polygons),
            "n_rois_used": len(polygons),
            "duplicate_rois": duplicates,
            "include": image.name not in exclude_images,
            "exclusion_reason": exclude_images.get(image.name),
            "n_labels": int(labels.max()),
            "empty_labels": int(sum(cell["mask_area"] == 0 for cell in cells)),
            "overlap_pixels": overlap_pixels,
            "overlap_roi_count": int(sum(cell["overlap_pixels"] > 0 for cell in cells)),
            "cells": cells,
        })
    manifest = {
        "version": 1,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "spacing": spacing,
        "smooth": smooth,
        "n_images": len(rows),
        "n_rois": sum(row["n_rois"] for row in rows),
        "images": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
