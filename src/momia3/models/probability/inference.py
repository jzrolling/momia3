"""Checkpoint loading and full-frame inference for probability-map models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from momia3.ops.segment import ProbToLabelsParams, SegmentationResult, prob_to_result
from momia3.models.probability.data import normalize_image
from momia3.models.probability.networks import build_model


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(
        payload["model_name"],
        base_channels=payload["base_channels"],
        n_classes=payload.get("n_classes", 3),
        in_channels=payload.get("in_channels", 1),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


@torch.no_grad()
def predict_probability(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str | torch.device = "cpu",
    bf16: bool = False,
) -> np.ndarray:
    """Normalize, pad to the network stride and return a (3, y, x) probability map."""
    work = normalize_image(image)
    h, w = work.shape
    pad_h = (-h) % 16
    pad_w = (-w) % 16
    if pad_h or pad_w:
        work = np.pad(work, ((0, pad_h), (0, pad_w)), mode="reflect")
    tensor = torch.from_numpy(work)[None, None].to(device)
    use_bf16 = bf16 and torch.device(device).type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(tensor)
        prob = torch.softmax(logits.float(), dim=1)
    return prob[0, :, :h, :w].cpu().numpy().astype(np.float32)


class ProbabilitySegmenter:
    """Common Segmenter adapter backed by a U-Net or AttResUNet checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        params: ProbToLabelsParams | None = None,
        device: str | None = None,
        bf16: bool = False,
    ) -> None:
        if device is None:
            from momia3.backend import get_device

            device = str(get_device())
        self.device = device
        self.bf16 = bf16
        self.params = params or ProbToLabelsParams()
        self.model, self.checkpoint = load_checkpoint(checkpoint, device=device)

    def predict_prob(self, image: np.ndarray) -> np.ndarray:
        return predict_probability(self.model, image, self.device, bf16=self.bf16)

    def predict(
        self, image: np.ndarray, params: ProbToLabelsParams | None = None
    ) -> SegmentationResult:
        return prob_to_result(self.predict_prob(image), params or self.params)

    def predict_labels(self, image: np.ndarray) -> np.ndarray:
        return self.predict(image).labels
