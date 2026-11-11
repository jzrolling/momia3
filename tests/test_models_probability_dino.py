"""Tests for DINOv3 OmniPose readouts and partial backbone fine-tuning."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from momia3.models.probability.dino_omni import DinoV3Omni


class FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 384, 1)
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 384, 16, 16)
        self.patch_size = 16
        self.blocks = nn.ModuleList([nn.Conv2d(384, 384, 1) for _ in range(12)])
        self.norm = nn.LayerNorm(384)

    def get_intermediate_layers(self, x, n, reshape, norm):
        assert reshape and norm
        count = 1 if isinstance(n, int) else len(n)
        return tuple(self.projection(x) for _ in range(count))


@pytest.mark.parametrize("patch_size", [8, 16])
def test_dino_readout_shape_and_frozen_backbone(patch_size):
    model = DinoV3Omni(n_classes=4, backbone=FakeDino(), patch_size=patch_size)
    model.train()
    assert not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    logits = model(torch.rand(2, 1, 49, 51))
    assert logits.shape == (2, 4, 49, 51)
    logits.mean().backward()
    assert model.readout.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_dino_hierarchical_decoder_forward_backward():
    model = DinoV3Omni(
        n_classes=4, backbone=FakeDino(), patch_size=8, hierarchical=True
    )
    logits = model(torch.rand(2, 1, 49, 51))
    assert logits.shape == (2, 4, 49, 51)
    logits.mean().backward()
    assert model.output.weight.grad is not None
    assert model.layer_weights.grad is not None
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_dino_unfreezes_only_requested_blocks():
    model = DinoV3Omni(
        n_classes=4, backbone=FakeDino(), patch_size=8, hierarchical=True,
        unfreeze_blocks=4,
    )
    assert all(not parameter.requires_grad for parameter in model.backbone.patch_embed.parameters())
    assert all(not parameter.requires_grad for block in model.backbone.blocks[:8] for parameter in block.parameters())
    assert all(parameter.requires_grad for block in model.backbone.blocks[8:] for parameter in block.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.norm.parameters())
    model.train()
    assert not model.backbone.training
