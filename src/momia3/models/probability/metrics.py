"""Pixel and instance metrics for three-class bacterial segmentation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def pixel_metrics(prob: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prob = np.asarray(prob)
    target = np.asarray(target)
    if prob.ndim != 3 or prob.shape[0] != 3 or prob.shape[1:] != target.shape:
        raise ValueError(f"incompatible probability/target shapes: {prob.shape}, {target.shape}")
    pred = np.argmax(prob, axis=0)
    confusion = np.bincount(
        target.ravel() * 3 + pred.ravel(), minlength=9
    ).reshape(3, 3).astype(np.float64)
    intersection = np.diag(confusion)
    pred_count = confusion.sum(axis=0)
    true_count = confusion.sum(axis=1)
    union = pred_count + true_count - intersection
    dice = 2 * intersection / np.maximum(pred_count + true_count, 1.0)
    iou = intersection / np.maximum(union, 1.0)
    result = {
        "pixel_accuracy": float(intersection.sum() / max(confusion.sum(), 1.0)),
        "mean_dice": float(dice.mean()),
        "mean_iou": float(iou.mean()),
    }
    for index, name in enumerate(("background", "edge", "interior")):
        result[f"dice_{name}"] = float(dice[index])
        result[f"iou_{name}"] = float(iou[index])
    return result


def instance_metrics(
    truth: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.5
) -> dict[str, float | int]:
    """Optimal IoU matching plus overlap-based over/under-segmentation counts."""
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    if truth.shape != pred.shape or truth.ndim != 2:
        raise ValueError(f"expected matching 2D labels, got {truth.shape}, {pred.shape}")
    n_true = int(truth.max())
    n_pred = int(pred.max())
    if n_true == 0 or n_pred == 0:
        tp = 0
        fp = n_pred
        fn = n_true
        precision = 1.0 if n_pred == 0 else 0.0
        recall = 1.0 if n_true == 0 else 0.0
        return {
            "n_true": n_true,
            "n_pred": n_pred,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "matched_mean_iou": 0.0,
            "oversegmented": 0,
            "undersegmented": 0,
        }

    intersection = np.bincount(
        truth.astype(np.int64).ravel() * (n_pred + 1) + pred.astype(np.int64).ravel(),
        minlength=(n_true + 1) * (n_pred + 1),
    ).reshape(n_true + 1, n_pred + 1)[1:, 1:].astype(np.float64)
    true_area = np.bincount(truth.ravel(), minlength=n_true + 1)[1:, None]
    pred_area = np.bincount(pred.ravel(), minlength=n_pred + 1)[1:, None]
    union = true_area + pred_area.T - intersection
    iou = intersection / np.maximum(union, 1.0)
    rows, cols = linear_sum_assignment(-iou)
    matched = iou[rows, cols] >= iou_threshold
    tp = int(matched.sum())
    fp = n_pred - tp
    fn = n_true - tp
    precision = tp / max(n_pred, 1)
    recall = tp / max(n_true, 1)
    matched_ious = iou[rows[matched], cols[matched]]

    true_covered = intersection / np.maximum(true_area, 1.0) >= 0.25
    pred_covered = intersection / np.maximum(pred_area.T, 1.0) >= 0.25
    return {
        "n_true": n_true,
        "n_pred": n_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "matched_mean_iou": float(matched_ious.mean()) if matched_ious.size else 0.0,
        "oversegmented": int((true_covered.sum(axis=1) >= 2).sum()),
        "undersegmented": int((pred_covered.sum(axis=0) >= 2).sum()),
    }


def aggregate_instance_metrics(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        return {}
    totals = {key: sum(int(row[key]) for row in rows) for key in (
        "n_true", "n_pred", "tp", "fp", "fn", "oversegmented", "undersegmented"
    )}
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    return {
        **totals,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "matched_mean_iou": float(np.mean([row["matched_mean_iou"] for row in rows])),
        "count_mae": float(np.mean([abs(int(row["n_pred"]) - int(row["n_true"])) for row in rows])),
    }
