"""Isolated CPnet tests: cellpose_omni modifies global logging on import."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cellpose_omni")

from momia3.models.probability.networks import CPnetProbability  # noqa: E402
from momia3.models.probability.train import load_compatible_weights  # noqa: E402


def test_cpnet_probability_forward_backward() -> None:
    model = CPnetProbability(base_channels=4)
    image = torch.rand(1, 1, 49, 51, requires_grad=True)
    logits = model(image)
    assert logits.shape == (1, 3, 49, 51)
    logits.mean().backward()
    assert image.grad is not None


def test_cpnet_pretrained_prefix_mapping(tmp_path) -> None:
    source = CPnetProbability(base_channels=4)
    checkpoint = tmp_path / "cpnet.pt"
    torch.save(source.net.state_dict(), checkpoint)
    target = CPnetProbability(base_channels=4)
    transfer = load_compatible_weights(target, str(checkpoint))
    assert transfer["loaded"] == len(source.net.state_dict())
    assert transfer["missing"] == 0
    assert transfer["unexpected"] == 0
