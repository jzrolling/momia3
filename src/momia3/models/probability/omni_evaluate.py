"""Validation-tuned evaluation of from-scratch OmniPose field checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tifffile

from momia3.models.probability.data import load_split
from momia3.models.probability.metrics import aggregate_instance_metrics, instance_metrics
from momia3.models.probability.omni_train import OmniPoseModelSegmenter
from momia3.ops.segment import OmniposeParams


def evaluate_samples(segmenter, samples, params):
    rows = []
    details = []
    for sample in samples:
        truth = tifffile.imread(sample.mask)
        result = segmenter.predict(tifffile.imread(sample.image), params=params)
        metrics = instance_metrics(truth, result.labels)
        rows.append(metrics)
        details.append({"sample_id": sample.sample_id, "instances": metrics, "qc": result.qc})
    return {"instances": aggregate_instance_metrics(rows), "n_images": len(samples),
            "reconstruction": params.model_dump(mode="json")}, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    samples = load_split(args.split_path, args.data_root)
    segmenter = OmniPoseModelSegmenter(args.checkpoint, device=args.device)
    params = OmniposeParams()
    trials = None
    validation = None
    if args.tune:
        val = [s for s in samples if s.split == "val"]
        trials = []
        for threshold in (-0.5, 0.0, 0.5):
            candidate = OmniposeParams(mask_threshold=threshold)
            summary, _ = evaluate_samples(segmenter, val, candidate)
            trials.append({"mask_threshold": threshold, "instances": summary["instances"]})
        best = max(trials, key=lambda row: row["instances"]["f1"])
        params = OmniposeParams(mask_threshold=best["mask_threshold"])
        validation, _ = evaluate_samples(segmenter, val, params)
    test = [s for s in samples if s.split == "test"]
    summary, details = evaluate_samples(segmenter, test, params)
    payload = {
        "checkpoint": args.checkpoint,
        "validation": validation,
        "test": summary,
        "per_image": details,
        "tuning_trials": trials,
        "strict_holdout": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
