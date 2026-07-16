"""Tests for momia3.keys — uint64 composite key encode/decode."""

from __future__ import annotations

import numpy as np
import pytest

from momia3 import keys


def test_known_answer() -> None:
    assert keys.encode_key(1, 2, 3) == (1 << 40) | (2 << 20) | 3
    assert keys.decode_key((1 << 40) | (2 << 20) | 3) == (1, 2, 3)


def test_scalar_roundtrip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        fov = int(rng.integers(0, keys.FOV_MAX))
        frame = int(rng.integers(0, keys.FRAME_MAX))
        label = int(rng.integers(0, keys.LABEL_MAX))
        assert keys.decode_key(keys.encode_key(fov, frame, label)) == (fov, frame, label)


def test_max_values() -> None:
    key = keys.encode_key(keys.FOV_MAX, keys.FRAME_MAX, keys.LABEL_MAX)
    assert key == 2**64 - 1
    assert keys.decode_key(key) == (keys.FOV_MAX, keys.FRAME_MAX, keys.LABEL_MAX)


def test_vectorized_roundtrip() -> None:
    rng = np.random.default_rng(1)
    fov = rng.integers(0, 100, size=64, dtype=np.uint32)
    frame = rng.integers(0, 1000, size=64, dtype=np.uint32)
    label = rng.integers(1, 5000, size=64, dtype=np.uint32)
    encoded = keys.encode_keys(fov, frame, label)
    assert encoded.dtype == np.uint64
    assert encoded.shape == (64,)
    dec_fov, dec_frame, dec_label = keys.decode_keys(encoded)
    assert dec_fov.dtype == np.uint32
    np.testing.assert_array_equal(dec_fov, fov)
    np.testing.assert_array_equal(dec_frame, frame)
    np.testing.assert_array_equal(dec_label, label)


def test_vectorized_broadcast() -> None:
    frame = np.arange(5, dtype=np.uint32)
    encoded = keys.encode_keys(3, frame, 7)
    assert encoded.shape == (5,)
    fov, frm, lbl = keys.decode_keys(encoded)
    np.testing.assert_array_equal(fov, np.full(5, 3, dtype=np.uint32))
    np.testing.assert_array_equal(frm, frame)
    np.testing.assert_array_equal(lbl, np.full(5, 7, dtype=np.uint32))


def test_scalar_vectorized_consistency() -> None:
    key = keys.encode_key(5, 6, 7)
    assert int(keys.encode_keys(5, 6, 7)) == key
    fov, frame, label = keys.decode_keys(np.array([key]))
    assert (int(fov[0]), int(frame[0]), int(label[0])) == (5, 6, 7)


@pytest.mark.parametrize(
    "fov, frame, label",
    [
        (keys.FOV_MAX + 1, 0, 0),
        (0, keys.FRAME_MAX + 1, 0),
        (0, 0, keys.LABEL_MAX + 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
    ],
)
def test_encode_out_of_range(fov: int, frame: int, label: int) -> None:
    with pytest.raises(ValueError):
        keys.encode_key(fov, frame, label)


def test_encode_type_errors() -> None:
    with pytest.raises(TypeError):
        keys.encode_key(0.5, 0, 0)
    with pytest.raises(TypeError):
        keys.encode_key("0", 0, 0)


def test_decode_out_of_range() -> None:
    with pytest.raises(ValueError):
        keys.decode_key(2**64)
    with pytest.raises(ValueError):
        keys.decode_key(-1)
    with pytest.raises(TypeError):
        keys.decode_key(1.5)


def test_vectorized_bounds() -> None:
    with pytest.raises(ValueError):
        keys.encode_keys(np.array([0, keys.FOV_MAX + 1]), 0, 0)
    with pytest.raises(ValueError):
        keys.encode_keys(0, np.array([-3]), 0)
    with pytest.raises(ValueError):
        keys.decode_keys(np.array([-1]))
    with pytest.raises(TypeError):
        keys.encode_keys(np.array([0.5]), 0, 0)
