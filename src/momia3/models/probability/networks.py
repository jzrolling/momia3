"""Compact U-Net and attention residual U-Net for three-class probability maps."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool = False):
        super().__init__()
        groups = min(8, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.residual = residual
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if residual and in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x) if self.residual else 0.0
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2), ConvBlock(in_channels, out_channels, residual=residual)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.gate = nn.Conv2d(gate_channels, inter_channels, 1, bias=False)
        self.skip = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.score = nn.Sequential(
            nn.SiLU(inplace=True), nn.Conv2d(inter_channels, 1, 1), nn.Sigmoid()
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attention = self.score(self.gate(gate) + self.skip(skip))
        return skip * attention


class Up(nn.Module):
    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        out_channels: int,
        residual: bool,
        attention: bool,
    ):
        super().__init__()
        self.attention = (
            AttentionGate(gate_channels, skip_channels, max(8, out_channels // 2))
            if attention
            else None
        )
        self.block = ConvBlock(
            gate_channels + skip_channels, out_channels, residual=residual
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        gate = F.interpolate(gate, scale_factor=2, mode="bilinear", align_corners=False)
        dy = skip.shape[-2] - gate.shape[-2]
        dx = skip.shape[-1] - gate.shape[-1]
        gate = F.pad(gate, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        if self.attention is not None:
            skip = self.attention(gate, skip)
        return self.block(torch.cat((skip, gate), dim=1))


class UNet(nn.Module):
    """Four-level U-Net producing logits for background, edge and interior."""

    def __init__(self, in_channels: int = 1, n_classes: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.input = ConvBlock(in_channels, c)
        self.down1 = Down(c, c * 2, residual=False)
        self.down2 = Down(c * 2, c * 4, residual=False)
        self.down3 = Down(c * 4, c * 8, residual=False)
        self.down4 = Down(c * 8, c * 16, residual=False)
        self.up1 = Up(c * 16, c * 8, c * 8, residual=False, attention=False)
        self.up2 = Up(c * 8, c * 4, c * 4, residual=False, attention=False)
        self.up3 = Up(c * 4, c * 2, c * 2, residual=False, attention=False)
        self.up4 = Up(c * 2, c, c, residual=False, attention=False)
        self.output = nn.Conv2d(c, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.input(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        return self.output(x)


class AttResUNet(nn.Module):
    """Residual U-Net with attention-gated decoder skip connections."""

    def __init__(self, in_channels: int = 1, n_classes: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.input = ConvBlock(in_channels, c, residual=True)
        self.down1 = Down(c, c * 2, residual=True)
        self.down2 = Down(c * 2, c * 4, residual=True)
        self.down3 = Down(c * 4, c * 8, residual=True)
        self.down4 = Down(c * 8, c * 16, residual=True)
        self.up1 = Up(c * 16, c * 8, c * 8, residual=True, attention=True)
        self.up2 = Up(c * 8, c * 4, c * 4, residual=True, attention=True)
        self.up3 = Up(c * 4, c * 2, c * 2, residual=True, attention=True)
        self.up4 = Up(c * 2, c, c, residual=True, attention=True)
        self.output = nn.Conv2d(c, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.input(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        return self.output(x)


class CPnetProbability(nn.Module):
    """Cellpose-Omni CPnet adapted to three-class logits."""

    def __init__(self, n_classes: int = 3, base_channels: int = 32):
        super().__init__()
        from cellpose_omni import resnet_torch

        c = base_channels
        self.net = resnet_torch.CPnet(
            [2, c, c * 2, c * 4, c * 8],
            n_classes,
            sz=3,
            residual_on=True,
            style_on=True,
            concatenation=False,
            dim=2,
            kernel_size=2,
            scale_factor=2,
            dilation=1,
            checkpoint=False,
            dropout=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = torch.cat((x, torch.zeros_like(x)), dim=1)
        if x.shape[1] != 2:
            raise ValueError(f"CPnet expects one or two input channels, got {x.shape[1]}")
        h, w = x.shape[-2:]
        pad_h = (-h) % 16
        pad_w = (-w) % 16
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        y, _style = self.net(x)
        return y[..., :h, :w]


MODEL_NAMES = ("unet", "attresunet", "cpnet", "dinov3_s16", "dinov3_s8", "dinov3_s8_hier")


def build_model(
    name: str, base_channels: int = 32, n_classes: int = 3, in_channels: int = 1,
    dino_checkpoint: str | None = None, dino_unfreeze_blocks: int = 0,
) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(
            in_channels=in_channels, base_channels=base_channels, n_classes=n_classes
        )
    if name in ("attresunet", "att_res_unet"):
        return AttResUNet(
            in_channels=in_channels, base_channels=base_channels, n_classes=n_classes
        )
    if name == "cpnet":
        return CPnetProbability(base_channels=base_channels, n_classes=n_classes)
    if name in ("dinov3_s16", "dinov3_s8", "dinov3_s8_hier"):
        from momia3.models.probability.dino_omni import DinoV3Omni

        patch_size = 16 if name == "dinov3_s16" else 8
        return DinoV3Omni(
            n_classes=n_classes, checkpoint=dino_checkpoint, patch_size=patch_size,
            hierarchical=name == "dinov3_s8_hier", unfreeze_blocks=dino_unfreeze_blocks,
        )
    raise ValueError(f"unknown model {name!r}; available: {MODEL_NAMES}")
