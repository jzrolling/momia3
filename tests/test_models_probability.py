"""Tests for trainable probability-map models and their data pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from momia3.models.probability.data import (  # noqa: E402
    AugmentParams,
    OmniPoseDataset,
    Sample,
    labels_to_target,
    normalize_image,
    random_augment,
    split_samples,
)
from momia3.models.probability.inference import ProbabilitySegmenter  # noqa: E402
from momia3.models.probability.metrics import (  # noqa: E402
    instance_metrics,
    pixel_metrics,
)
from momia3.models.probability.networks import AttResUNet, UNet  # noqa: E402


def test_normalization_and_three_class_target() -> None:
    image = np.arange(400, dtype=np.float32).reshape(20, 20)
    normalized = normalize_image(image)
    assert normalized.dtype == np.float32
    assert 0.0 <= normalized.min() <= normalized.max() <= 1.0
    assert np.array_equal(normalize_image(np.ones((8, 8))), np.zeros((8, 8)))

    labels = np.zeros((12, 16), dtype=np.int32)
    labels[2:9, 2:7] = 1
    labels[2:9, 7:12] = 2
    labels[10, 15] = 3
    target = labels_to_target(labels)
    assert set(np.unique(target)) == {0, 1, 2}
    assert np.all(target[labels > 0] > 0)
    for label_id in (1, 2, 3):
        assert np.any(target[labels == label_id] == 2)


def test_augmentation_is_synchronized_and_deterministic() -> None:
    image = np.zeros((48, 48), dtype=np.float32)
    labels = np.zeros((48, 48), dtype=np.int32)
    labels[12:30, 15:28] = 1
    image[labels > 0] = 0.2
    image[labels == 0] = 0.8
    target = labels_to_target(labels)
    params = AugmentParams(patch_size=32, elastic_probability=0.0)
    a, at = random_augment(image, target, np.random.default_rng(9), params)
    b, bt = random_augment(image, target, np.random.default_rng(9), params)
    assert a.shape == (32, 32) and at.shape == (32, 32)
    assert np.array_equal(a, b) and np.array_equal(at, bt)
    assert set(np.unique(at)).issubset({0, 1, 2})
    assert np.all((a >= 0) & (a <= 1))


def test_dataset_augmentation_advances_between_epochs() -> None:
    dataset = OmniPoseDataset.__new__(OmniPoseDataset)
    dataset.training = True
    dataset.seed = 2026
    dataset.augment = AugmentParams(patch_size=32)
    dataset.images = [np.random.default_rng(3).random((48, 48)).astype(np.float32)]
    labels = np.zeros((48, 48), dtype=np.int32)
    labels[10:30, 12:32] = 1
    dataset.targets = [labels_to_target(labels)]

    torch.manual_seed(4)
    first = dataset[0][0].numpy()
    second = dataset[0][0].numpy()
    assert not np.array_equal(first, second)


def test_split_keeps_source_groups_together() -> None:
    samples = [
        Sample(Path(f"{group}/{i}.tif"), Path(f"{group}/{i}_masks.tif"), group)
        for group, count in (("a", 8), ("b", 5), ("c", 4), ("d", 3), ("e", 2))
        for i in range(count)
    ]
    first = split_samples(samples, seed=7)
    second = split_samples(samples, seed=7)
    assert [(s.sample_id, s.split) for s in first] == [
        (s.sample_id, s.split) for s in second
    ]
    assert {s.split for s in first} == {"train", "val", "test"}
    for group in {s.group for s in first}:
        assert len({s.split for s in first if s.group == group}) == 1


@pytest.mark.parametrize("model_class", [UNet, AttResUNet])
def test_network_forward_backward_irregular_shape(model_class) -> None:
    model = model_class(base_channels=4)
    image = torch.rand(2, 1, 49, 51, requires_grad=True)
    logits = model(image)
    assert logits.shape == (2, 3, 49, 51)
    logits.mean().backward()
    assert image.grad is not None


def test_checkpoint_probability_segmenter(tmp_path) -> None:
    model = UNet(base_channels=4)
    checkpoint = tmp_path / "unet.pt"
    torch.save(
        {
            "model_name": "unet",
            "base_channels": 4,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    image = np.random.default_rng(2).random((53, 47)).astype(np.float32)
    segmenter = ProbabilitySegmenter(
        checkpoint,
        device="cpu",
        params=None,
    )
    prob = segmenter.predict_prob(image)
    assert prob.shape == (3, 53, 47)
    np.testing.assert_allclose(prob.sum(axis=0), 1.0, atol=1e-5)
    result = segmenter.predict(image)
    assert result.labels.shape == image.shape
    assert result.qc["backend"] == "probability"


def test_perfect_pixel_and_instance_metrics() -> None:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:8, 2:8] = 1
    labels[12:18, 12:18] = 2
    target = labels_to_target(labels)
    prob = np.eye(3, dtype=np.float32)[target].transpose(2, 0, 1)
    assert pixel_metrics(prob, target)["mean_dice"] == pytest.approx(1.0)
    metrics = instance_metrics(labels, labels)
    assert metrics["f1"] == 1.0
    assert metrics["matched_mean_iou"] == 1.0
