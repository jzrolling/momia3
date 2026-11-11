"""DINOv3 dense features with direct or hierarchical OmniPose readout."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class DinoV3Omni(nn.Module):
    def __init__(
        self,
        n_classes: int = 4,
        checkpoint: str | Path | None = None,
        backbone: nn.Module | None = None,
        patch_size: int = 16,
        hierarchical: bool = False,
        unfreeze_blocks: int = 0,
    ) -> None:
        super().__init__()
        if patch_size not in (8, 16):
            raise ValueError(f"unsupported DINOv3 patch size: {patch_size}")
        if hierarchical and patch_size != 8:
            raise ValueError("hierarchical DINOv3-OmniPose currently requires patch_size=8")
        if backbone is None:
            from dinov3.hub.backbones import dinov3_vits16

            backbone = dinov3_vits16(
                pretrained=checkpoint is not None,
                weights=str(checkpoint) if checkpoint is not None else None,
            )
        if patch_size == 8:
            previous = backbone.patch_embed.proj
            projection = nn.Conv2d(3, 384, kernel_size=8, stride=8)
            projection.weight.data = previous.weight.data[:, :, ::2, ::2].clone()
            projection.bias.data = previous.bias.data.clone()
            backbone.patch_embed.proj = projection
            backbone.patch_size = 8
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        if not 0 <= unfreeze_blocks <= len(backbone.blocks):
            raise ValueError(f"unfreeze_blocks must be in [0, {len(backbone.blocks)}]")
        for block in backbone.blocks[-unfreeze_blocks:] if unfreeze_blocks else ():
            block.requires_grad_(True)
        if unfreeze_blocks:
            backbone.norm.requires_grad_(True)
        self.unfreeze_blocks = unfreeze_blocks
        self.backbone.eval()
        self.patch_size = patch_size
        self.hierarchical = hierarchical
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        if hierarchical:
            from momia3.models.probability.networks import ConvBlock, Down, Up

            self.layer_weights = nn.Parameter(torch.zeros(4))
            self.skip0 = ConvBlock(3, 32, residual=True)
            self.skip1 = Down(32, 64, residual=True)
            self.skip2 = Down(64, 128, residual=True)
            self.up2 = Up(384, 128, 128, residual=True, attention=True)
            self.up1 = Up(128, 64, 64, residual=True, attention=True)
            self.up0 = Up(64, 32, 32, residual=True, attention=True)
            self.output = nn.Conv2d(32, n_classes, 1)
        else:
            self.readout = nn.Conv2d(384, n_classes * patch_size**2, kernel_size=1)
            self.upsample = nn.PixelShuffle(patch_size)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[1] != 3:
            raise ValueError(f"DINOv3 expects one or three input channels, got {x.shape[1]}")
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % self.patch_size, (-w) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        image = x
        x = (x - self.input_mean) / self.input_std
        context = nullcontext() if self.unfreeze_blocks else torch.no_grad()
        with context:
            layers = (2, 5, 8, 11) if self.hierarchical else 1
            features = self.backbone.get_intermediate_layers(x, n=layers, reshape=True, norm=True)
        if not self.hierarchical:
            return self.upsample(self.readout(features[0]))[..., :h, :w]
        weights = torch.softmax(self.layer_weights, dim=0)
        feature = sum(weight * value for weight, value in zip(weights, features))
        skip0 = self.skip0(image)
        skip1 = self.skip1(skip0)
        skip2 = self.skip2(skip1)
        x = self.up2(feature, skip2)
        x = self.up1(x, skip1)
        x = self.up0(x, skip0)
        return self.output(x)[..., :h, :w]
