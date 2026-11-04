"""Tests for the optional Omnipose-lite model backend."""

from __future__ import annotations

import pytest


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
