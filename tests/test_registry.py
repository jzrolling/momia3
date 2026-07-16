"""Registry completeness and schema-export tests (ARCHITECTURE.md §3.1)."""

from __future__ import annotations

import inspect
import json

import pytest
from pydantic import BaseModel, ValidationError

from momia3.ops import (
    characterize,
    measure,
    midline,
    outline,
    preprocess,
    segment,
    track,
)
from momia3.pipeline import registry
from momia3.pipeline.registry import REGISTRY

OPS_MODULES = (preprocess, segment, measure, outline, midline, characterize, track)


def _public_functions(module) -> list[str]:
    """Public plain functions of an ops module (per its ``__all__``)."""
    return [
        name
        for name in module.__all__
        if inspect.isfunction(getattr(module, name, None))
    ]


def test_every_public_ops_function_is_registered() -> None:
    """Every public function of the 7 ops modules maps to ``<module>.<function>``."""
    missing: list[str] = []
    for module in OPS_MODULES:
        mod_name = module.__name__.rsplit(".", 1)[1]
        for fn_name in _public_functions(module):
            op_name = f"{mod_name}.{fn_name}"
            if op_name not in REGISTRY:
                missing.append(op_name)
    assert not missing, f"unregistered ops: {missing}"


def test_io_ops_registered() -> None:
    """The two import ops are present."""
    assert "io.import_tiff" in REGISTRY
    assert "io.import_synth" in REGISTRY


def test_chain_ops_registered() -> None:
    """The canonical chain's composite steps are present with shard hints."""
    expected = {
        "preprocess.drift": "fov",
        "segment.threshold": "frame",
        "measure.cells": "frame",
        "outline.extract": "frame",
        "midline.extract": "frame",
        "track.link": "fov",
    }
    for name, shard in expected.items():
        assert name in REGISTRY, name
        assert REGISTRY[name].shard == shard, name


def test_specs_are_well_formed() -> None:
    """Every spec has a pydantic params model, semver and declared outputs."""
    for name, spec in REGISTRY.items():
        assert spec.name == name
        assert isinstance(spec.version, str) and spec.version
        assert issubclass(spec.params_model, BaseModel)
        assert callable(spec.runner)
        assert spec.outputs, f"{name}: an op must declare at least one output"
        assert spec.shard in (None, "frame", "fov")


def test_describe_returns_valid_json_schema() -> None:
    """describe() of every op is JSON-serializable with an object params schema."""
    for name in REGISTRY:
        description = registry.describe(name)
        blob = json.dumps(description)  # must not raise
        assert blob
        for key in ("name", "version", "description", "shard", "inputs", "outputs", "params"):
            assert key in description, f"{name}: missing {key}"
        assert description["name"] == name
        assert description["params"]["type"] == "object"
        assert "properties" in description["params"]


def test_describe_unknown_op_raises() -> None:
    with pytest.raises(KeyError):
        registry.describe("does.not_exist")


def test_to_mcp_shape() -> None:
    """MCP export entries carry name/description/inputSchema with unique names."""
    tools = registry.to_mcp()
    assert len(tools) == len(REGISTRY)
    names = [tool["name"] for tool in tools]
    assert len(set(names)) == len(names)
    for tool in tools:
        assert set(tool) >= {"name", "description", "inputSchema"}
        assert "." not in tool["name"]
        assert tool["inputSchema"]["type"] == "object"
        assert "params" in tool["inputSchema"]["properties"]
        assert "inputs" in tool["inputSchema"]["properties"]
        assert tool["x-momia3-op"] in REGISTRY


def test_to_json_roundtrip() -> None:
    """to_json() is JSON-serializable and covers the whole registry."""
    blob = json.dumps(registry.to_json())
    parsed = json.loads(blob)
    assert set(parsed["ops"]) == set(REGISTRY)


def test_params_models_enforce_constraints() -> None:
    """Spot-check pydantic validation of registered params models."""
    spec = REGISTRY["segment.threshold"]
    params = spec.params_model.model_validate({"method": "isodata", "channel": "phase"})
    assert params.method == "isodata"
    with pytest.raises(ValidationError):
        spec.params_model.model_validate({"method": "not_a_method"})
    with pytest.raises(ValidationError):
        spec.params_model.model_validate({"sauvola_k": -1.0})
    with pytest.raises(ValidationError):
        REGISTRY["io.import_synth"].params_model.model_validate({"n_frames": 0})


def test_duplicate_registration_raises() -> None:
    with pytest.raises(KeyError):
        registry.register(
            "segment.threshold",
            params_model=REGISTRY["segment.threshold"].params_model,
            inputs={},
            outputs={"x": "json"},
            runner=lambda inputs, params, ctx: {"x": {}},
        )
