"""Isolated tests for OmniPose-native training targets and losses."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omnipose")

from momia3.models.probability.networks import build_model  # noqa: E402
from momia3.models.probability.omni_data import (  # noqa: E402
    _rotate_field,
    _scale_field,
    labels_to_omni_target,
    random_field_augment,
)
from momia3.models.probability.omni_train import omni_loss  # noqa: E402


def test_omni_target_layout_and_values() -> None:
    labels = np.zeros((48, 48), dtype=np.int32)
    labels[8:28, 10:18] = 1
    labels[12:36, 28:34] = 2
    target = labels_to_omni_target(labels)
    assert target.shape == (5, 48, 48)
    assert target.dtype == np.float32
    assert np.all(np.isfinite(target))
    assert np.all(target[2][labels == 0] == -5.0)
    assert set(np.unique(target[3])) == {0.0, 1.0}
    assert np.max(np.linalg.norm(target[:2], axis=0)) <= 5.0 + 1e-5
    assert np.any(target[:2][:, labels > 0] != 0)


@pytest.mark.parametrize("model_name", ["attresunet", "cpnet"])
def test_field_augmentation_and_loss_backward(model_name) -> None:
    image = np.random.default_rng(3).random((64, 64)).astype(np.float32)
    target = np.zeros((5, 64, 64), dtype=np.float32)
    target[0, 20:30, 20:30] = 5.0
    target[2] = -5.0
    target[2, 15:35, 15:35] = 2.0
    target[4, 15:35, 15:35] = 1.0
    original = target.copy()
    augmented, fields = random_field_augment(
        image, target, np.random.default_rng(5), patch_size=32
    )
    assert augmented.shape == (32, 32)
    assert fields.shape == (5, 32, 32)
    assert np.all(np.isfinite(fields))
    np.testing.assert_array_equal(target, original)

    model = build_model(model_name, base_channels=4, n_classes=4)
    logits = model(torch.from_numpy(augmented)[None, None])
    total, parts = omni_loss(logits, torch.from_numpy(fields)[None])
    total.backward()
    assert torch.isfinite(total)
    assert set(parts) == {"flow_mse", "direction", "distance", "boundary"}


def test_field_affine_transforms_update_vectors_and_distances() -> None:
    image = np.zeros((33, 33), dtype=np.float32)
    image[8:25, 8:25] = 1.0
    target = np.zeros((5, 33, 33), dtype=np.float32)
    target[1, 8:25, 8:25] = 2.0
    target[2] = -5.0
    target[2, 8:25, 8:25] = 2.0
    target[4, 8:25, 8:25] = 1.0
    rotated_image, rotated = _rotate_field(image, target, 90.0)
    assert np.isclose(rotated[0, 16, 16], -2.0, atol=1e-5)
    assert np.isclose(rotated[1, 16, 16], 0.0, atol=1e-5)
    assert np.isclose(rotated_image[16, 16], 1.0, atol=1e-5)
    scaled_image, scaled = _scale_field(image, target, 2.0)
    assert scaled_image.shape == image.shape
    assert scaled.shape == target.shape
    assert np.isclose(scaled[1, 16, 16], 4.0, atol=1e-5)
    assert np.isclose(scaled[2, 16, 16], 4.0, atol=1e-5)
    _downscaled_image, downscaled = _scale_field(image, target, 0.5)
    assert np.all(downscaled[2, 0] == -5.0)


def test_original_recipe_attresunet_checkpoint_roundtrip(tmp_path) -> None:
    from momia3.models.probability.omni_original_train import (
        build_original_model,
        export_checkpoint,
    )
    from momia3.models.probability.omni_train import OmniPoseModelSegmenter

    model = build_original_model("attresunet", device="cpu", base_channels=4)
    output = model.net(torch.zeros((1, 2, 32, 32)))
    assert output[0].shape == (1, 4, 32, 32)
    checkpoint = tmp_path / "attresunet.pt"
    export_checkpoint(model, "attresunet", checkpoint, base_channels=4)
    segmenter = OmniPoseModelSegmenter(checkpoint, device="cpu", bf16=False)
    assert segmenter.model.input.conv1.in_channels == 2
    fields = segmenter.engine.runner.run_gpu(torch.zeros((32, 32)))
    assert fields.shape == (4, 32, 32)


def test_original_recipe_cpnet_checkpoint_roundtrip(tmp_path) -> None:
    from momia3.models.probability.inference import load_checkpoint
    from momia3.models.probability.omni_original_train import (
        build_original_model,
        export_checkpoint,
    )

    model = build_original_model("cpnet", device="cpu", base_channels=32)
    checkpoint = tmp_path / "cpnet.pt"
    export_checkpoint(model, "cpnet", checkpoint)
    restored, payload = load_checkpoint(checkpoint)
    assert payload["in_channels"] == 2
    assert restored(torch.zeros((1, 1, 32, 32))).shape == (1, 4, 32, 32)
