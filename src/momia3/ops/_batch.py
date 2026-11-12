"""Frame-scoped batch access and memoization primitives.

Two small helpers keep per-instance measurement code from rescanning the
whole frame once per label:

- :class:`FrameBatch` sorts the flat label array exactly once per frame
  (``O(N log N)``) and then slices contiguous per-label pixel ranges
  (``O(n_i)`` per label), replacing the ``O(L * N)`` pattern
  ``flat_image[flat_labels == label]`` used by the percentile and
  distribution-moment operators. Pixels are returned in ascending flat
  index order, matching ``numpy.where(labels == label)``.
- :class:`ArtifactCache` is a frame-lifetime memo table for derived
  artifacts (cropped rings, percentile vectors, GLCMs, rod frames) so a
  feature plan computes a shared quantity once.

Both are internal (no registry entries, no schema change).
"""

from __future__ import annotations

from typing import Any, Callable, Hashable, Mapping

import numpy as np
from scipy import ndimage

__all__ = ["FrameBatch", "ArtifactCache"]


class ArtifactCache:
    """Frame-lifetime memo table for derived artifacts.

    Keys are hashable tokens (e.g. ``("percentiles", channel)``); values
    are opaque. The cache is deliberately not thread-shared.
    """

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: dict[Hashable, Any] = {}

    def __contains__(self, key: Hashable) -> bool:
        return key in self._data

    def get(self, key: Hashable, default: Any = None) -> Any:
        return self._data.get(key, default)

    def put(self, key: Hashable, value: Any) -> Any:
        self._data[key] = value
        return value

    def memoize(self, key: Hashable, factory: Callable[[], Any]) -> Any:
        """Return the cached value, computing it with ``factory`` once."""
        if key not in self._data:
            self._data[key] = factory()
        return self._data[key]

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


class FrameBatch:
    """Per-label pixel access for one frame.

    Parameters
    ----------
    labels_2d : numpy.ndarray
        2D instance label image (0 = background).
    channels : mapping of str to numpy.ndarray, optional
        Channel name -> 2D intensity image; shapes must match the labels.
    pixel_microns : float
        Physical pixel size recorded for downstream geometry.

    Attributes
    ----------
    labels : numpy.ndarray
        The label image as given.
    label_ids : numpy.ndarray
        Ascending positive label ids (the per-cell row order).
    slices : list of slice tuples
        Tight bounding boxes from :func:`scipy.ndimage.find_objects`;
        ``slices[label - 1]`` is the bbox of label ``label``.
    channels : dict of str to numpy.ndarray
        Validated channel images.
    cache : ArtifactCache
        Frame-scoped memo table.
    """

    __slots__ = (
        "labels",
        "labels_i",
        "label_ids",
        "slices",
        "channels",
        "pixel_microns",
        "cache",
        "_order",
        "_starts",
        "_ends",
        "_pixels",
    )

    def __init__(
        self,
        labels_2d: np.ndarray,
        channels: Mapping[str, np.ndarray] | None = None,
        pixel_microns: float = 1.0,
    ) -> None:
        labels = np.asarray(labels_2d)
        if labels.ndim != 2:
            raise ValueError(f"labels_2d must be 2D, got shape {labels.shape}")
        self.labels = labels
        self.labels_i = labels.astype(np.int64, copy=False)
        unique = np.unique(self.labels_i)
        self.label_ids = unique[unique > 0]
        self.slices = (
            ndimage.find_objects(self.labels_i) if self.label_ids.size else []
        )
        self.cache = ArtifactCache()
        self.pixel_microns = float(pixel_microns)
        validated: dict[str, np.ndarray] = {}
        for name, image in dict(channels or {}).items():
            arr = np.asarray(image)
            if arr.shape != labels.shape:
                raise ValueError(
                    f"channel {name!r} shape {arr.shape} does not match "
                    f"labels shape {labels.shape}"
                )
            validated[name] = arr
        self.channels = validated
        self._order: np.ndarray | None = None
        self._starts: np.ndarray | None = None
        self._ends: np.ndarray | None = None
        self._pixels: dict[int, np.ndarray] = {}

    @property
    def n_labels(self) -> int:
        """Number of instances (background excluded)."""
        return int(self.label_ids.size)

    def position(self, label_id: int) -> int:
        """Position of ``label_id`` in the ascending ``label_ids`` order."""
        pos = int(np.searchsorted(self.label_ids, int(label_id)))
        if pos >= self.label_ids.size or int(self.label_ids[pos]) != int(label_id):
            raise KeyError(f"label {label_id} is not present in the frame")
        return pos

    def _ensure_order(self) -> None:
        """Build the one-time stable sort of flat pixels by label."""
        if self._order is not None:
            return
        flat = self.labels_i.ravel()
        order = np.argsort(flat, kind="stable")
        self._order = order
        sorted_labels = flat[order]
        self._starts = np.searchsorted(sorted_labels, self.label_ids)
        self._ends = np.searchsorted(sorted_labels, self.label_ids, side="right")

    def pixels(self, label_id: int) -> np.ndarray:
        """Flat pixel indices of one label, ascending flat index order."""
        lab = int(label_id)
        cached = self._pixels.get(lab)
        if cached is not None:
            return cached
        self._ensure_order()
        pos = self.position(lab)
        assert self._order is not None and self._starts is not None and self._ends is not None
        idx = self._order[self._starts[pos] : self._ends[pos]]
        self._pixels[lab] = idx
        return idx

    def pixel_xy(self, label_id: int) -> tuple[np.ndarray, np.ndarray]:
        """``(x, y)`` global coordinates of one label, ascending flat order."""
        width = self.labels.shape[1]
        flat = self.pixels(label_id)
        return flat % width, flat // width

    def values(self, label_id: int, image_2d: np.ndarray) -> np.ndarray:
        """Pixel values of ``image_2d`` inside one label (1-D float64)."""
        image = np.asarray(image_2d)
        if image.shape != self.labels.shape:
            raise ValueError(
                f"image shape {image.shape} does not match labels shape "
                f"{self.labels.shape}"
            )
        flat = self.pixels(label_id)
        return image.reshape(-1)[flat].astype(np.float64, copy=False)

    def channel_values(self, label_id: int, name: str) -> np.ndarray:
        """Pixel values of one registered channel inside one label."""
        return self.values(label_id, self.channels[name])

    def crop(self, label_id: int, pad: int = 0) -> np.ndarray:
        """Padded bounding-box view of the label image for one instance."""
        sl = self.slices[int(label_id) - 1]
        if sl is None:
            raise KeyError(f"label {label_id} has no bounding box")
        y0 = max(sl[0].start - int(pad), 0)
        y1 = min(sl[0].stop + int(pad), self.labels.shape[0])
        x0 = max(sl[1].start - int(pad), 0)
        x1 = min(sl[1].stop + int(pad), self.labels.shape[1])
        return self.labels[y0:y1, x0:x1]
