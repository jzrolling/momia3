"""Tests for the optional Omnipose-lite model backend."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi


torch = pytest.importorskip("torch")
pytest.importorskip("cellpose_omni")

from momia3.models.omnipose_lite import (  # noqa: E402
    FastOmnipose,
    OmniposeLite,
    resolve_omnipose_weights,
)


def test_backend_alias() -> None:
    assert OmniposeLite is FastOmnipose


def test_resolve_explicit_weights(tmp_path, monkeypatch) -> None:
    fake = tmp_path / "weights"
    fake.write_bytes(b"not-a-real-checkpoint")
    assert resolve_omnipose_weights(fake) == str(fake)


def test_resolve_env_weights(tmp_path, monkeypatch) -> None:
    fake = tmp_path / "weights"
    fake.write_bytes(b"not-a-real-checkpoint")
    monkeypatch.setenv("MOMIA3_OMNIPOSE_WEIGHTS", str(fake))
    assert resolve_omnipose_weights() == str(fake)


def test_resolve_missing_explicit_path_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_omnipose_weights(tmp_path / "does-not-exist")


@pytest.mark.parametrize("resample", [True, False])
def test_rescale_preserves_public_output_shape(resample) -> None:
    class FakeRunner:
        def __init__(self):
            self.input_shapes = []

        def run_gpu(self, ch0):
            self.input_shapes.append(tuple(ch0.shape))
            return torch.zeros((4, *ch0.shape), dtype=ch0.dtype)

    runner = FakeRunner()
    engine = FastOmnipose(device="cpu", runner=runner)
    image = np.zeros((20, 30), dtype=np.float32)
    result = engine.segment_result(
        image, niter=1, rescale=0.5, resample=resample,
        return_flow=True, return_boundary=True,
    )

    assert runner.input_shapes == [(10, 15)]
    assert result.labels.shape == image.shape
    assert result.foreground.shape == image.shape
    assert result.distance.shape == image.shape
    assert result.flow.shape == (2, *image.shape)
    assert result.boundary.shape == image.shape
    assert result.flow.dtype == np.float32
    assert result.boundary.dtype == np.float32
    assert result.labels.dtype == np.uint32
    assert result.qc["prediction_shape"] == [10, 15]
    assert int(result.labels.max()) == 1

    batch = engine.segment_batch(
        [image, image], niter=1, rescale=0.5, resample=resample
    )
    assert [labels.shape for labels in batch] == [image.shape, image.shape]


def test_rescale_rejects_invalid_values() -> None:
    engine = FastOmnipose(device="cpu", runner=object())
    with pytest.raises(ValueError, match="rescale"):
        engine.segment_result(np.zeros((8, 8)), rescale=0)


def test_rescale_input_matches_scipy_exactly() -> None:
    from momia3.models.omnipose_lite.pipeline import img_to_gpu_channel

    class FakeRunner:
        def run_gpu(self, ch0):
            self.input = ch0.numpy().copy()
            return torch.zeros((4, *ch0.shape), dtype=ch0.dtype)

    image = np.random.default_rng(4).normal(size=(21, 31)).astype(np.float32)
    expected = ndi.zoom(img_to_gpu_channel(image, "cpu").numpy(), 0.5, order=3)
    runner = FakeRunner()
    engine = FastOmnipose(device="cpu", runner=runner)
    engine.segment_result(image, niter=1, rescale=0.5)
    np.testing.assert_array_equal(runner.input, expected)


def test_flow_threshold_calls_upstream_qc(monkeypatch) -> None:
    import omnipose.core

    class FakeRunner:
        def run_gpu(self, ch0):
            return torch.zeros((4, *ch0.shape), dtype=ch0.dtype)

    called = {}

    def fake_remove(labels, dP, **kwargs):
        called["threshold"] = kwargs["threshold"]
        called["coords"] = kwargs["coords"]
        called["flow_shape"] = dP.shape
        return np.zeros_like(labels)

    monkeypatch.setattr(omnipose.core, "remove_bad_flow_masks", fake_remove)
    engine = FastOmnipose(device="cpu", runner=FakeRunner())
    result = engine.segment_result(
        np.zeros((12, 16), dtype=np.float32), niter=1, flow_threshold=0.4
    )

    assert called["threshold"] == 0.4
    assert called["flow_shape"] == (2, 12, 16)
    assert called["coords"].shape == (2, 12 * 16)
    assert result.qc["flow_rejected"] == 1
    assert result.qc["n_instances"] == 0
    assert result.flow is None
    assert result.boundary is None
