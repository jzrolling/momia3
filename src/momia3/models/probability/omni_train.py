"""Train probability backbones with OmniPose-native field targets."""

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

from momia3.models.omnipose_lite.pipeline import FastOmnipose
from momia3.models.probability.data import AugmentParams, load_or_create_split
from momia3.models.probability.inference import load_checkpoint
from momia3.models.probability.metrics import aggregate_instance_metrics, instance_metrics
from momia3.models.probability.networks import build_model
from momia3.models.probability.omni_data import OmniPoseFieldDataset
from momia3.ops.segment import OmniposeParams, SegmentationResult


def omni_loss(logits: torch.Tensor, target: torch.Tensor):
    """Core weighted flow/direction/distance/boundary losses from OmniPose."""
    flow = logits[:, :2]
    true_flow = target[:, :2]
    weight = target[:, 4:5]
    flow_mse = ((flow - true_flow) ** 2 * weight).mean()
    pred_mag = torch.sqrt((flow**2).sum(dim=1) + 1e-12)
    true_mag = torch.sqrt((true_flow**2).sum(dim=1) + 1e-12)
    dot = (flow * true_flow).sum(dim=1)
    cos_sq = torch.where(
        target[:, 2] > 0,
        (dot / (pred_mag * true_mag + 1e-12)) ** 2,
        torch.ones_like(dot),
    )
    direction = ((1.0 - cos_sq) * target[:, 4]).mean()
    distance = ((logits[:, 2] - target[:, 2]) ** 2 * target[:, 4]).mean()
    boundary = F.binary_cross_entropy_with_logits(logits[:, 3], target[:, 3])
    total = flow_mse + direction + distance + boundary
    return total, {
        "flow_mse": flow_mse,
        "direction": direction,
        "distance": distance,
        "boundary": boundary,
    }


class ModelFieldRunner:
    """Adapt a momia3 backbone to the FastOmnipose reconstruction runner API."""

    def __init__(self, model, device="cuda", bf16=False, input_channels=1):
        self.model = model
        self.device = torch.device(device)
        self.bf16 = bf16 and self.device.type == "cuda"
        self.input_channels = input_channels

    @torch.no_grad()
    def run_gpu(self, ch0: torch.Tensor):
        h, w = ch0.shape
        pad_h = (-h) % 16
        pad_w = (-w) % 16
        x = ch0[None, None]
        if self.input_channels == 2:
            x = torch.cat((x, torch.zeros_like(x)), dim=1)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):
            y = self.model(x)
        return y[0, :, :h, :w].float()


class OmniPoseModelSegmenter:
    """Checkpoint-backed OmniPose segmenter for either supported backbone."""

    def __init__(self, checkpoint: str | Path, params: OmniposeParams | None = None,
                 device: str = "cuda", bf16: bool = True):
        self.model, self.checkpoint = load_checkpoint(checkpoint, device=device)
        self.params = params or OmniposeParams()
        runner = ModelFieldRunner(
            self.model, device=device, bf16=bf16,
            input_channels=self.checkpoint.get("in_channels", 1),
        )
        self.engine = FastOmnipose(device=device, half=bf16, runner=runner)

    def predict(self, image: np.ndarray, params: OmniposeParams | None = None) -> SegmentationResult:
        p = params or self.params
        return self.engine.segment_result(
            image,
            niter=p.niter,
            mask_threshold=p.mask_threshold,
            diam_threshold=p.diam_threshold,
            min_size=p.min_size,
            max_size=p.max_size,
            hole_size=p.hole_size,
        )


@torch.no_grad()
def validate_fields(model, dataset, runner):
    model.eval()
    losses = []
    for index in range(len(dataset)):
        image = torch.from_numpy(dataset.images[index]).to(runner.device)
        target = torch.from_numpy(dataset.targets[index])[None].to(runner.device)
        pred = runner.run_gpu(image)[None]
        _total, parts = omni_loss(pred, target)
        losses.append({key: float(value) for key, value in parts.items()})
    return {key: float(np.mean([row[key] for row in losses])) for key in losses[0]}


@torch.no_grad()
def validate_instances(engine: FastOmnipose, dataset: OmniPoseFieldDataset):
    rows = []
    for index, sample in enumerate(dataset.samples):
        result = engine.segment_result(dataset.images[index], niter=200)
        truth = tifffile.imread(sample.mask)
        rows.append(instance_metrics(truth, result.labels))
    return aggregate_instance_metrics(rows)


def _checkpoint(model, config, epoch):
    return {
        "format_version": 1,
        "model_name": config["model"],
        "base_channels": config["base_channels"],
        "n_classes": 4,
        "task": "omnipose",
        "state_dict": model.state_dict(),
        "training": config,
        "epoch": epoch,
    }


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    samples = load_or_create_split(args.split_path, args.data_root, seed=args.seed)
    train_samples = [s for s in samples if s.split == "train"]
    val_samples = [s for s in samples if s.split == "val"]
    augment = AugmentParams(
        patch_size=args.patch_size,
        scale_probability=args.scale_probability,
        scale_range=(args.scale_min, args.scale_max),
    )
    train_data = OmniPoseFieldDataset(
        train_samples, args.cache_dir, training=True, patch_size=args.patch_size,
        seed=args.seed, augment=augment,
    )
    val_data = OmniPoseFieldDataset(val_samples, args.cache_dir, training=False)
    loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = build_model(
        args.model, base_channels=args.base_channels, n_classes=4,
        dino_checkpoint=args.dino_checkpoint, dino_unfreeze_blocks=args.dino_unfreeze_blocks,
    ).to(device)
    runner = ModelFieldRunner(model, device=device, bf16=args.bf16)
    engine = FastOmnipose(device=device, half=args.bf16, runner=runner)
    backbone_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("backbone.")
    ]
    readout_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    parameter_groups = [{"params": readout_parameters, "lr": args.learning_rate}]
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": args.dino_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
    )
    history = []
    best_f1 = -1.0
    start_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        history_path = output / "history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        validated = [row for row in history if "validation_instances" in row]
        if validated:
            best_f1 = max(row["validation_instances"]["f1"] for row in validated)
        print(f"resume from epoch {start_epoch}", flush=True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    backbone_trainable = sum(parameter.numel() for parameter in backbone_parameters)
    print(
        f"data: train={len(train_data)} val={len(val_data)} epochs={args.epochs}; "
        f"trainable={trainable:,} backbone={backbone_trainable:,}", flush=True
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        rows = []
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            use_bf16 = args.bf16 and device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(images)
                total, parts = omni_loss(logits, targets)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            rows.append({
                "total": float(total.detach()),
                **{key: float(value.detach()) for key, value in parts.items()},
            })
        scheduler.step()
        row = {
            "epoch": epoch,
            "train": {key: float(np.mean([r[key] for r in rows])) for key in rows[0]},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "dino_learning_rate": float(optimizer.param_groups[-1]["lr"]),
            "seconds": time.perf_counter() - started,
        }
        if epoch % args.val_loss_every == 0 or epoch == 1:
            row["validation_fields"] = validate_fields(model, val_data, runner)
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            metrics = validate_instances(engine, val_data)
            row["validation_instances"] = metrics
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(_checkpoint(model, config, epoch), output / "best.pt")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint = _checkpoint(model, config, epoch)
            torch.save(checkpoint, output / "last.pt")
            if args.keep_checkpoints:
                snapshots = output / "checkpoints"
                snapshots.mkdir(exist_ok=True)
                torch.save(checkpoint, snapshots / f"epoch_{epoch:04d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                },
                output / "trainer_state.pt",
            )
        history.append(row)
        if epoch % args.val_loss_every == 0 or epoch == 1 or epoch == args.epochs:
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=("attresunet", "cpnet", "dinov3_s16", "dinov3_s8", "dinov3_s8_hier"), required=True)
    parser.add_argument("--dino-checkpoint", default=None)
    parser.add_argument("--dino-unfreeze-blocks", type=int, default=0)
    parser.add_argument("--dino-learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scale-probability", type=float, default=0.3)
    parser.add_argument("--scale-min", type=float, default=0.9)
    parser.add_argument("--scale-max", type=float, default=1.1)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--val-loss-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", default=None, help="Path to trainer_state.pt")
    parser.add_argument("--bf16", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
