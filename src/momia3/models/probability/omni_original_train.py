"""Train CPnet or AttResUNet through the upstream OmniPose recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import nn

from momia3.models.probability.data import load_split
from momia3.models.probability.networks import AttResUNet


class BatchMeanBCEWithLogits(nn.Module):
    def forward(self, pred, target):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        )
        return loss.reshape(loss.shape[0], -1).mean(dim=1).mean()


class AttResOmniNet(nn.Module):
    """Expose AttResUNet through the CPnet-style interface used by OmniPose."""

    def __init__(self, nchan: int = 2, base_channels: int = 32):
        super().__init__()
        self.net = AttResUNet(in_channels=nchan, n_classes=4, base_channels=base_channels)
        self.mkldnn = False

    def forward(self, x: torch.Tensor):
        style = torch.zeros((x.shape[0], 256), dtype=x.dtype, device=x.device)
        return self.net(x), style

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)


def build_original_model(model_name: str, device: str = "cuda", base_channels: int = 32):
    from cellpose_omni import models

    model = models.CellposeModel(
        gpu=torch.device(device).type == "cuda",
        device=torch.device(device),
        pretrained_model=False,
        diam_mean=30.0,
        residual_on=True,
        style_on=True,
        concatenation=False,
        nchan=2,
        nclasses=3,
        dim=2,
        omni=True,
        logits=False,
        nsample=4,
        checkpoint=False,
        dropout=False,
        kernel_size=2,
        scale_factor=2,
        dilation=1,
    )
    if model_name == "attresunet":
        network = AttResOmniNet(nchan=2, base_channels=base_channels).to(device)
        if isinstance(model.net, nn.DataParallel):
            model.net = nn.DataParallel(network)
        else:
            model.net = network
        model.net_type = "attresunet_omni_original_recipe"
    elif model_name != "cpnet":
        raise ValueError(f"unsupported model {model_name!r}")
    set_criterion = model._set_criterion

    def set_logits_criterion():
        set_criterion()
        model.BCELoss = BatchMeanBCEWithLogits()

    model._set_criterion = set_logits_criterion
    return model


def export_checkpoint(model, model_name: str, output: str | Path, base_channels: int = 32):
    network = model.net.module if isinstance(model.net, nn.DataParallel) else model.net
    if model_name == "cpnet":
        state = {f"net.{key}": value for key, value in network.state_dict().items()}
    else:
        state = network.net.state_dict()
    payload = {
        "format_version": 1,
        "model_name": model_name,
        "base_channels": base_channels,
        "in_channels": 2,
        "n_classes": 4,
        "task": "omnipose",
        "state_dict": state,
        "training": {"recipe": "upstream_cellpose_omni", "diam_mean": 30.0},
    }
    torch.save(payload, output)


def train(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({
        "normalization": "cellpose_omni.transforms.normalize_img(omni=True)",
        "augmentation": "cellpose_omni.transforms.random_rotate_and_resize",
        "loss": "omnipose.core.loss",
        "optimizer": "SGD",
        "learning_rate": 0.2,
        "momentum": 0.9,
        "weight_decay": 1e-5,
        "batch_size": 8,
        "patch_size": 224,
        "nchan": 2,
        "nclasses_output": 4,
        "diam_mean": 30.0,
        "rescale": True,
        "dataloader": False,
        "autocast": False,
    })
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    samples = load_split(args.split_path, args.data_root)
    train_samples = [sample for sample in samples if sample.split == "train"]
    train_images = [tifffile.imread(sample.image) for sample in train_samples]
    train_labels = [tifffile.imread(sample.mask).astype(np.int32) for sample in train_samples]
    model = build_original_model(args.model, args.device, args.base_channels)
    model.train(
        train_images,
        train_labels,
        train_links=[None] * len(train_samples),
        train_files=[str(sample.image) for sample in train_samples],
        test_data=None,
        test_labels=None,
        test_links=None,
        test_files=None,
        channels=[0, 0],
        channel_axis=None,
        normalize=True,
        save_path=str(output),
        save_every=args.save_every,
        save_each=False,
        learning_rate=0.2,
        n_epochs=args.epochs,
        momentum=0.9,
        SGD=True,
        weight_decay=1e-5,
        batch_size=8,
        dataloader=False,
        num_workers=0,
        rescale=True,
        min_train_masks=0,
        tyx=(224, 224),
        do_autocast=False,
    )
    export_checkpoint(model, args.model, output / "last.pt", args.base_channels)
    (output / "completed.json").write_text(json.dumps({"epochs": args.epochs}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("cpnet", "attresunet"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=100)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
