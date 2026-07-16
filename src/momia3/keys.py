"""Composite uint64 primary keys for cell instances (ARCHITECTURE.md §2.1).

Layout::

    cell_key = fov_id << 40 | frame << 20 | label

Bit budget (64 bits total):

- ``fov_id`` : bits [40, 64) — 24 bits, valid range ``0 .. 2**24 - 1``
- ``frame``  : bits [20, 40) — 20 bits, valid range ``0 .. 2**20 - 1``
- ``label``  : bits [ 0, 20) — 20 bits, valid range ``0 .. 2**20 - 1``

``label == 0`` is reserved for background by convention; real cells use
``label >= 1``. ``cell_key == 0`` (fov=0, frame=0, label=0) is used as the
root sentinel parent for ``appear`` lineage events.

No string parsing is involved anywhere; both scalar and vectorized
(numpy) encode/decode variants are provided.
"""

from __future__ import annotations

import numpy as np

FOV_SHIFT: int = 40
"""Bit offset of the ``fov_id`` component."""

FRAME_SHIFT: int = 20
"""Bit offset of the ``frame`` component."""

FOV_MAX: int = (1 << 24) - 1
"""Largest encodable ``fov_id`` (inclusive)."""

FRAME_MAX: int = (1 << 20) - 1
"""Largest encodable ``frame`` (inclusive)."""

LABEL_MAX: int = (1 << 20) - 1
"""Largest encodable ``label`` (inclusive)."""

KEY_MAX: int = (1 << 64) - 1
"""Largest valid ``cell_key`` (uint64 range)."""

ROOT_KEY: int = 0
"""Sentinel parent key for ``appear`` edges (fov=0, frame=0, label=0)."""

_MASK_20: int = (1 << 20) - 1

__all__ = [
    "FOV_SHIFT",
    "FRAME_SHIFT",
    "FOV_MAX",
    "FRAME_MAX",
    "LABEL_MAX",
    "KEY_MAX",
    "ROOT_KEY",
    "encode_key",
    "decode_key",
    "encode_keys",
    "decode_keys",
]


def _check_scalar(name: str, value: int, maximum: int) -> int:
    """Validate one scalar key component.

    Parameters
    ----------
    name : str
        Component name used in the error message.
    value : int
        Component value (must be a non-negative integer).
    maximum : int
        Largest allowed value (inclusive).

    Returns
    -------
    int
        The value as a plain Python ``int``.

    Raises
    ------
    TypeError
        If ``value`` is not an integer.
    ValueError
        If ``value`` is outside ``[0, maximum]``.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got bool")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    ivalue = int(value)
    if not 0 <= ivalue <= maximum:
        raise ValueError(f"{name}={ivalue} out of range [0, {maximum}]")
    return ivalue


def encode_key(fov_id: int, frame: int, label: int) -> int:
    """Encode one ``(fov_id, frame, label)`` triple into a uint64 ``cell_key``.

    Parameters
    ----------
    fov_id : int
        Field-of-view index, ``0 .. 2**24 - 1``.
    frame : int
        Time frame index, ``0 .. 2**20 - 1``.
    label : int
        Instance label within the frame, ``0 .. 2**20 - 1``
        (``0`` is reserved for background).

    Returns
    -------
    int
        The composite key as a plain Python ``int`` (fits in uint64).

    Raises
    ------
    TypeError
        If any component is not an integer.
    ValueError
        If any component is out of range.
    """
    f = _check_scalar("fov_id", fov_id, FOV_MAX)
    t = _check_scalar("frame", frame, FRAME_MAX)
    lb = _check_scalar("label", label, LABEL_MAX)
    return (f << FOV_SHIFT) | (t << FRAME_SHIFT) | lb


def decode_key(key: int) -> tuple[int, int, int]:
    """Decode one ``cell_key`` into ``(fov_id, frame, label)``.

    Parameters
    ----------
    key : int
        Composite key in the uint64 range.

    Returns
    -------
    tuple of int
        ``(fov_id, frame, label)`` as plain Python ints.

    Raises
    ------
    TypeError
        If ``key`` is not an integer.
    ValueError
        If ``key`` is outside ``[0, 2**64 - 1]``.
    """
    k = _check_scalar("key", key, KEY_MAX)
    fov_id = (k >> FOV_SHIFT) & FOV_MAX
    frame = (k >> FRAME_SHIFT) & _MASK_20
    label = k & _MASK_20
    return fov_id, frame, label


def _as_int_array(name: str, value: int | np.ndarray, maximum: int) -> np.ndarray:
    """Coerce input to an integer ndarray and bounds-check it.

    Parameters
    ----------
    name : str
        Component name used in error messages.
    value : int or array-like of int
        Component values.
    maximum : int
        Largest allowed value (inclusive).

    Returns
    -------
    numpy.ndarray
        Integer array (dtype int64, safe for bounds comparison).

    Raises
    ------
    TypeError
        If the input is not integer typed.
    ValueError
        If any element is outside ``[0, maximum]``.
    """
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype, got {arr.dtype}")
    arr = arr.astype(np.int64, copy=False)
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) > maximum):
        raise ValueError(
            f"{name} values out of range [0, {maximum}]: "
            f"min={int(arr.min())}, max={int(arr.max())}"
        )
    return arr


def encode_keys(
    fov_id: int | np.ndarray,
    frame: int | np.ndarray,
    label: int | np.ndarray,
) -> np.ndarray:
    """Vectorized :func:`encode_key` (broadcasting supported).

    Parameters
    ----------
    fov_id, frame, label : int or array-like of int
        Key components; arrays are broadcast against each other.

    Returns
    -------
    numpy.ndarray of uint64
        Composite keys with the broadcast shape.

    Raises
    ------
    TypeError
        If any component is not integer typed.
    ValueError
        If any component value is out of range.
    """
    fov, frm, lbl = np.broadcast_arrays(
        _as_int_array("fov_id", fov_id, FOV_MAX),
        _as_int_array("frame", frame, FRAME_MAX),
        _as_int_array("label", label, LABEL_MAX),
    )
    fov_u = fov.astype(np.uint64)
    frm_u = frm.astype(np.uint64)
    lbl_u = lbl.astype(np.uint64)
    return (fov_u << np.uint64(FOV_SHIFT)) | (frm_u << np.uint64(FRAME_SHIFT)) | lbl_u


def decode_keys(keys: int | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized :func:`decode_key`.

    Parameters
    ----------
    keys : int or array-like of int
        Composite keys, each within the uint64 range.

    Returns
    -------
    tuple of numpy.ndarray
        ``(fov_id, frame, label)`` arrays of dtype uint32.

    Raises
    ------
    TypeError
        If ``keys`` is not integer typed.
    ValueError
        If any key is negative or exceeds the uint64 range.
    """
    arr = np.asarray(keys)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"keys must have an integer dtype, got {arr.dtype}")
    if arr.size and np.issubdtype(arr.dtype, np.signedinteger) and int(arr.min()) < 0:
        raise ValueError("keys must be non-negative")
    keys_u = arr.astype(np.uint64)
    fov = (keys_u >> np.uint64(FOV_SHIFT)) & np.uint64(FOV_MAX)
    frame = (keys_u >> np.uint64(FRAME_SHIFT)) & np.uint64(_MASK_20)
    label = keys_u & np.uint64(_MASK_20)
    return (
        fov.astype(np.uint32),
        frame.astype(np.uint32),
        label.astype(np.uint32),
    )
