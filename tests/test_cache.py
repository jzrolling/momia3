"""Tests for momia3.storage.cache — content-addressed cache + provenance."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from momia3.storage.cache import CacheStore, cache_key, make_provenance


class _Params(BaseModel):
    sigma: float = Field(default=1.0, ge=0)
    mode: str = "mirror"


def test_cache_key_deterministic() -> None:
    k1 = cache_key("op.gaussian", "1.0.0", _Params(), ["abc"])
    k2 = cache_key("op.gaussian", "1.0.0", _Params(), ["abc"])
    assert k1 == k2
    assert len(k1) == 64
    int(k1, 16)  # valid hex


def test_cache_key_sensitivity() -> None:
    base = cache_key("op.gaussian", "1.0.0", _Params(sigma=1.0), [])
    assert cache_key("op.gaussian", "1.0.0", _Params(sigma=2.0), []) != base
    assert cache_key("op.gaussian", "1.0.1", _Params(sigma=1.0), []) != base
    assert cache_key("op.other", "1.0.0", _Params(sigma=1.0), []) != base
    assert cache_key("op.gaussian", "1.0.0", _Params(sigma=1.0), ["up"]) != base


def test_cache_key_upstream_order_insensitive() -> None:
    k1 = cache_key("op", "1.0.0", _Params(), ["a", "b", "c"])
    k2 = cache_key("op", "1.0.0", _Params(), ["c", "a", "b"])
    assert k1 == k2


def test_cache_key_rejects_non_model() -> None:
    with pytest.raises(TypeError):
        cache_key("op", "1.0.0", {"sigma": 1.0}, [])


def test_put_get_has(tmp_path) -> None:
    store = CacheStore(tmp_path / ".momia3_cache")
    key = cache_key("op.gaussian", "1.0.0", _Params(), ["deadbeef" * 8])
    assert not store.has(key)

    provenance = make_provenance("op.gaussian", "1.0.0", _Params(), ["deadbeef" * 8])
    payload = b"parquet-bytes"
    artifact = store.put(key, payload, suffix=".parquet", provenance=provenance)

    assert artifact.name == "artifact.parquet"
    assert store.has(key)
    assert store.get_bytes(key) == payload
    assert store.get(key) == artifact

    prov = store.read_provenance(key)
    assert prov["op"] == "op.gaussian"
    assert prov["version"] == "1.0.0"
    assert prov["params"] == {"sigma": 1.0, "mode": "mirror"}
    assert prov["upstream"] == ["deadbeef" * 8]
    assert "timestamp" in prov
    assert "git_hash" in prov
    # provenance.json is valid standalone JSON
    raw = json.loads((artifact.parent / "provenance.json").read_text())
    assert raw == prov


def test_get_missing(tmp_path) -> None:
    store = CacheStore(tmp_path / ".momia3_cache")
    key = cache_key("op", "1.0.0", _Params(), [])
    assert not store.has(key)
    with pytest.raises(KeyError):
        store.get(key)
    with pytest.raises(KeyError):
        store.read_provenance(key)


def test_malformed_key(tmp_path) -> None:
    store = CacheStore(tmp_path / ".momia3_cache")
    with pytest.raises(ValueError, match="malformed cache key"):
        store.has("../escape")
    with pytest.raises(ValueError, match="malformed cache key"):
        store.put("not-hex", b"data")
