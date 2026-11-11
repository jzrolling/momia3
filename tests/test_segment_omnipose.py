"""Torch-free tests for the Omnipose segment-op integration."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from momia3.pipeline import steps as pipeline_steps
from momia3.pipeline.registry import StepContext
from momia3.storage.zarr_store import ExperimentStore

from momia3.ops.segment import OmniposeParams, SegmentationResult, Segmenter
from momia3.pipeline.registry import REGISTRY
from momia3.models.omnipose_lite import segmenter


def test_omnipose_params_defaults() -> None:
    params = OmniposeParams()
    assert params.weights_path is None
    assert params.device is None
    assert params.niter == 200
    assert params.rescale == 1.0
    assert params.resample is True
    assert params.mask_threshold == 0.0
    assert params.flow_threshold == 0.0
    assert params.return_flow is False
    assert params.return_boundary is False
    assert params.diam_threshold == 12.0
    assert params.min_size == 15
    assert params.max_size is None
    assert params.hole_size == 3


def test_omnipose_ops_registered() -> None:
    assert "segment.omnipose" in REGISTRY
    assert "segment.omnipose_frame" in REGISTRY
    assert REGISTRY["segment.omnipose"].shard == "frame"
    assert REGISTRY["segment.omnipose_frame"].shard == "frame"


def test_segmenter_adapter_import_is_torch_free() -> None:
    # The segmenter module must be importable without torch so the pipeline
    # registry (params models) does not require the optional dependency.
    assert hasattr(segmenter, "omnipose_labels")
    assert callable(segmenter.resolve_device)


def test_omnipose_adapter_uses_common_result(tmp_path) -> None:
    weights = tmp_path / "weights"
    weights.write_bytes(b"fake")

    class FakeRunner:
        def segment_result(self, image, **kwargs):
            assert kwargs["niter"] == 7
            assert kwargs["rescale"] == 0.5
            assert kwargs["resample"] is False
            assert kwargs["flow_threshold"] == 0.4
            assert kwargs["return_flow"] is True
            assert kwargs["return_boundary"] is True
            labels = np.zeros(image.shape, dtype=np.uint32)
            labels[4:8, 4:8] = 1
            return SegmentationResult(
                labels=labels,
                foreground=labels > 0,
                distance=np.ones(image.shape, dtype=np.float32),
                qc={"backend": "omnipose", "n_instances": 1},
            )

    adapter = segmenter.OmniposeSegmenter(
        OmniposeParams(
            weights_path=str(weights), device="cpu", niter=7,
            rescale=0.5, resample=False, flow_threshold=0.4,
            return_flow=True, return_boundary=True,
        )
    )
    adapter._runner = FakeRunner()
    image = np.zeros((16, 16), dtype=np.float32)
    result = adapter.predict(image)

    assert isinstance(adapter, Segmenter)
    assert isinstance(result, SegmentationResult)
    assert result.qc["backend"] == "omnipose"
    np.testing.assert_array_equal(adapter.predict_labels(image), result.labels)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("cellpose_omni") is None,
    reason="optional torch/cellpose_omni backend not installed",
)
def test_omnipose_segmenter_labels() -> None:
    img = np.zeros((64, 64), dtype=np.float32)
    img[16:48, 24:40] = -1.0  # dark rod on bright background
    try:
        from momia3.models.omnipose_lite.weights import resolve_omnipose_weights

        resolve_omnipose_weights()
    except FileNotFoundError:
        pytest.skip("Omnipose weights not installed")
    labels = segmenter.omnipose_labels(img, OmniposeParams(niter=2))
    assert labels.shape == img.shape
    assert labels.dtype == np.uint32


def test_run_omnipose_writes_labels(tmp_path, monkeypatch) -> None:
    """Composite runner pre-creates and fills the store labels without torch."""
    store_path = tmp_path / "experiment.zarr"
    store = ExperimentStore.create(
        store_path, pixel_microns=1.0, channels=["phase"]
    )
    store.write_images(
        0, "phase", np.full((2, 32, 32), 0.5, dtype=np.float32)
    )
    del store

    def fake_labels(image, params):
        del params
        labels = np.zeros(image.shape, dtype=np.uint32)
        labels[8:16, 8:16] = 1
        return labels

    monkeypatch.setattr(segmenter, "omnipose_labels", fake_labels)
    monkeypatch.setattr(segmenter, "resolve_device", lambda device=None: "cpu")

    params = pipeline_steps.OmniposeRunParams(channel="phase", device="cpu")
    ctx = StepContext(step_id="segment.omnipose", workspace=tmp_path)
    out = pipeline_steps.run_omnipose(
        {"experiment": str(store_path)}, params, ctx
    )
    assert out == {"experiment": store_path}
    opened = ExperimentStore.open(store_path, mode="r")
    for t in range(2):
        labels = opened.get_labels_frame(0, t)
        assert labels.shape == (32, 32)
        assert int(labels.max()) == 1
        assert int((labels > 0).sum()) == 64

