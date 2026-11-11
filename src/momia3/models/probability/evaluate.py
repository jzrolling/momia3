"""Validation-only reconstruction tuning and final test-set evaluation."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import tifffile

from momia3.models.probability.data import Sample, labels_to_target, load_split
from momia3.models.probability.inference import load_checkpoint, predict_probability
from momia3.models.probability.metrics import (
    aggregate_instance_metrics,
    instance_metrics,
    pixel_metrics,
)
from momia3.ops.segment import ProbToLabelsParams, prob_to_result


def predict_records(model, samples: list[Sample], device: str, bf16: bool = False):
    records = []
    for sample in samples:
        image = tifffile.imread(sample.image)
        truth = tifffile.imread(sample.mask)
        records.append({
            "sample_id": sample.sample_id,
            "prob": predict_probability(model, image, device=device, bf16=bf16),
            "target": labels_to_target(truth),
            "truth": truth,
        })
    return records


def evaluate_records(
    records, params: ProbToLabelsParams | None = None
) -> tuple[dict, list[dict]]:
    params = params or ProbToLabelsParams()
    instance_rows = []
    pixel_rows = []
    details = []
    for record in records:
        result = prob_to_result(record["prob"], params)
        instances = instance_metrics(record["truth"], result.labels)
        pixels = pixel_metrics(record["prob"], record["target"])
        instance_rows.append(instances)
        pixel_rows.append(pixels)
        details.append({
            "sample_id": record["sample_id"],
            "instances": instances,
            "pixels": pixels,
            "qc": result.qc,
        })
    summary = {
        "instances": aggregate_instance_metrics(instance_rows),
        "pixels": {
            key: float(np.mean([row[key] for row in pixel_rows]))
            for key in pixel_rows[0]
        } if pixel_rows else {},
        "n_images": len(records),
        "reconstruction": params.model_dump(mode="json"),
    }
    return summary, details


def tune_reconstruction(records) -> tuple[ProbToLabelsParams, list[dict]]:
    """Select post-processing on validation probabilities, never on test data."""
    trials = []
    foreground_thresholds = (0.3, 0.4, 0.5)
    seed_ladders = ((0.5, 0.3), (0.7, 0.5))
    edge_maxima = (0.6, 0.75, 0.9)
    for fg, seeds, edge_max in itertools.product(
        foreground_thresholds, seed_ladders, edge_maxima
    ):
        params = ProbToLabelsParams(
            fg_threshold=fg,
            fg_erosion_radius=0,
            seed_thresholds=seeds,
            edge_max=edge_max,
            min_seed_size=5,
            seed_opening_radius=0,
            min_cell_size=10,
        )
        summary, _ = evaluate_records(records, params)
        instances = summary["instances"]
        trials.append({
            "params": params.model_dump(mode="json"),
            "f1": instances["f1"],
            "precision": instances["precision"],
            "recall": instances["recall"],
            "matched_mean_iou": instances["matched_mean_iou"],
        })
    best = max(trials, key=lambda row: (row["f1"], row["matched_mean_iou"]))
    return ProbToLabelsParams(**best["params"]), trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    model, _ = load_checkpoint(args.checkpoint, device=args.device)
    samples = load_split(args.split_path, args.data_root)
    params = ProbToLabelsParams(
        fg_erosion_radius=0, min_seed_size=5, seed_opening_radius=0
    )
    trials = None
    validation_summary = None
    if args.tune:
        val_records = predict_records(
            model, [s for s in samples if s.split == "val"], args.device, args.bf16
        )
        params, trials = tune_reconstruction(val_records)
        validation_summary, _ = evaluate_records(val_records, params)
    test_records = predict_records(
        model, [s for s in samples if s.split == "test"], args.device, args.bf16
    )
    summary, details = evaluate_records(test_records, params)
    payload = {
        "checkpoint": str(args.checkpoint),
        "validation": validation_summary,
        "test": summary,
        "per_image": details,
        "tuning_trials": trials,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
