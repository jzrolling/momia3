"""Train U-Net or AttResUNet on OmniPose masks with instance-based model selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from momia3.models.probability.data import (
    AugmentParams,
    OmniPoseDataset,
    load_or_create_split,
)
from momia3.models.probability.inference import predict_probability
from momia3.models.probability.metrics import (
    aggregate_instance_metrics,
    instance_metrics,
    pixel_metrics,
)
from momia3.models.probability.networks import MODEL_NAMES, build_model
from momia3.ops.segment import ProbToLabelsParams, prob_to_result


def combined_loss(
    logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=class_weights)
    prob = torch.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (prob * one_hot).sum(dims)
    union = prob.sum(dims) + one_hot.sum(dims)
    dice = 1.0 - ((2 * intersection + 1.0) / (union + 1.0)).mean()
    return 0.5 * ce.float() + 0.5 * dice


def class_weights(dataset: OmniPoseDataset, device: torch.device) -> torch.Tensor:
    counts = np.zeros(3, dtype=np.float64)
    for target in dataset.targets:
        counts += np.bincount(target.ravel(), minlength=3)
    frequency = counts / max(counts.sum(), 1.0)
    weights = 1.0 / np.sqrt(np.maximum(frequency, 1e-8))
    weights /= weights.mean()
    weights = np.clip(weights, 0.5, 5.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def reconstruction_params() -> ProbToLabelsParams:
    return ProbToLabelsParams(
        fg_erosion_radius=0,
        min_seed_size=5,
        seed_opening_radius=0,
        min_cell_size=10,
    )


@torch.no_grad()
def validate(model, dataset: OmniPoseDataset, device, bf16: bool) -> dict:
    model.eval()
    params = reconstruction_params()
    instance_rows = []
    pixel_rows = []
    for index, sample in enumerate(dataset.samples):
        prob = predict_probability(model, dataset.images[index], device=device, bf16=bf16)
        truth = tifffile.imread(sample.mask)
        result = prob_to_result(prob, params)
        instance_rows.append(instance_metrics(truth, result.labels))
        pixel_rows.append(pixel_metrics(prob, dataset.targets[index]))
    return {
        "instances": aggregate_instance_metrics(instance_rows),
        "pixels": {
            key: float(np.mean([row[key] for row in pixel_rows]))
            for key in pixel_rows[0]
        },
    }


def _checkpoint(model, config: dict, epoch: int) -> dict:
    return {
        "format_version": 1,
        "model_name": config["model"],
        "base_channels": config["base_channels"],
        "state_dict": model.state_dict(),
        "normalization": "full_frame_percentile_1_99_clip_0_1",
        "target": "background_edge_interior_v1",
        "reconstruction": reconstruction_params().model_dump(mode="json"),
        "training": config,
        "epoch": epoch,
    }


def load_compatible_weights(model, path: str) -> dict[str, int]:
    """Load only checkpoint tensors whose names and shapes match the new model."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    current = model.state_dict()
    compatible = {}
    for key, value in state.items():
        candidates = (key, f"net.{key}") if not key.startswith("net.") else (key, key[4:])
        target_key = next(
            (
                candidate for candidate in candidates
                if candidate in current and current[candidate].shape == value.shape
            ),
            None,
        )
        if target_key is not None:
            compatible[target_key] = value
    if not compatible:
        raise ValueError(f"no compatible tensors found in {path}")
    result = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": len(compatible),
        "missing": len(result.missing_keys),
        "unexpected": len(result.unexpected_keys),
    }


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    samples = load_or_create_split(args.split_path, args.data_root, seed=args.seed)
    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]
    augment = AugmentParams(patch_size=args.patch_size)
    train_data = OmniPoseDataset(train_samples, training=True, augment=augment, seed=args.seed)
    val_data = OmniPoseDataset(val_samples, training=False, boundary_width=1)
    print(
        f"data: train={len(train_samples)} val={len(val_samples)} "
        f"test={sum(s.split == 'test' for s in samples)}",
        flush=True,
    )
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = build_model(args.model, base_channels=args.base_channels).to(device)
    if args.pretrained:
        transfer = load_compatible_weights(model, args.pretrained)
        print(f"pretrained tensor transfer: {transfer}", flush=True)
    weights = class_weights(train_data, device)
    print(f"class weights: {weights.detach().cpu().numpy().tolist()}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
    )
    best_score = -1.0
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history = []
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        losses = []
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            use_bf16 = args.bf16 and device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(images)
                loss = combined_loss(logits, targets, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - started,
        }
        should_validate = epoch == 1 or epoch % args.validate_every == 0 or epoch == args.epochs
        if should_validate:
            metrics = validate(model, val_data, device, args.bf16)
            row["validation"] = metrics
            score = float(metrics["instances"]["f1"])
            if score > best_score:
                best_score = score
                stale_epochs = 0
                torch.save(_checkpoint(model, config, epoch), best_path)
            else:
                stale_epochs += args.validate_every
            torch.save(_checkpoint(model, config, epoch), last_path)
            print(json.dumps(row), flush=True)
            history.append(row)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            if stale_epochs >= args.early_stopping:
                print(f"early stopping at epoch {epoch}", flush=True)
                break
        else:
            print(json.dumps(row), flush=True)
            history.append(row)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Optional checkpoint; only name- and shape-compatible tensors are loaded",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--early-stopping", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bf16", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
