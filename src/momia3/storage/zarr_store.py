"""OME-Zarr experiment store (ARCHITECTURE.md §2.2).

On-disk layout::

    experiment.zarr/
    ├── fov_{id}/
    │   ├── images/{channel}   # (t, y, x), chunks (1, 512, 512)
    │   ├── labels             # (t, y, x) uint32 instance segmentation
    │   └── prob               # (t, c, y, x) float16 AI probability maps (optional)
    └── .zattrs / zarr.json    # pixel_microns, channels, time interval, provenance

All reads are lazy: array handles are returned as :class:`zarr.Array` and
frame access slices a single time point (one chunk row), never the whole
stack.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import zarr

logger = logging.getLogger(__name__)

CHUNK_Y: int = 512
CHUNK_X: int = 512

__all__ = ["ExperimentStore", "CHUNK_Y", "CHUNK_X", "CHANNEL_ROLES"]

CHANNEL_ROLES: tuple[str, ...] = (
    "brightfield",
    "fluorescence",
    "mask",
    "label",
    "other",
)


class ExperimentStore:
    """Handle on one ``experiment.zarr`` directory.

    Parameters
    ----------
    root : zarr.Group
        Opened root group.
    path : pathlib.Path or None
        Filesystem path of the store (informational).

    Notes
    -----
    Use :meth:`create` or :meth:`open` rather than constructing directly.
    """

    def __init__(self, root: zarr.Group, path: Path | None = None) -> None:
        self._root = root
        self._path = Path(path) if path is not None else None

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        pixel_microns: float,
        channels: list[str] | tuple[str, ...],
        time_interval_s: float | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> ExperimentStore:
        """Create a new (empty) experiment store, overwriting any existing one.

        Parameters
        ----------
        path : str or pathlib.Path
            Target ``experiment.zarr`` directory.
        pixel_microns : float
            Physical pixel size in microns.
        channels : list of str
            Channel names (e.g. ``["phase", "gfp"]``).
        time_interval_s : float, optional
            Time interval between frames in seconds.
        provenance : dict, optional
            Free-form provenance record stored in root attrs.

        Returns
        -------
        ExperimentStore
            Store handle opened in append mode.
        """
        path = Path(path)
        root = zarr.open_group(str(path), mode="w")
        root.attrs["pixel_microns"] = float(pixel_microns)
        root.attrs["channels"] = [str(c) for c in channels]
        if time_interval_s is not None:
            root.attrs["time_interval_s"] = float(time_interval_s)
        if provenance is not None:
            root.attrs["provenance"] = provenance
        logger.info("created experiment store at %s", path)
        return cls(root, path)

    @classmethod
    def open(cls, path: str | Path, mode: str = "a") -> ExperimentStore:
        """Open an existing experiment store.

        Parameters
        ----------
        path : str or pathlib.Path
            Existing ``experiment.zarr`` directory.
        mode : str
            Zarr open mode: ``"r"`` read-only, ``"a"`` read/write (default).

        Returns
        -------
        ExperimentStore
        """
        path = Path(path)
        return cls(zarr.open_group(str(path), mode=mode), path)

    # ------------------------------------------------------------------
    # basic properties
    # ------------------------------------------------------------------
    @property
    def path(self) -> Path | None:
        """Filesystem path of the store, if known."""
        return self._path

    @property
    def root(self) -> zarr.Group:
        """Underlying zarr root group."""
        return self._root

    @property
    def attrs(self) -> Any:
        """Mutable mapping of root attributes."""
        return self._root.attrs

    @property
    def pixel_microns(self) -> float:
        """Physical pixel size in microns."""
        return float(self._root.attrs["pixel_microns"])

    @property
    def channels(self) -> tuple[str, ...]:
        """Registered channel names."""
        return tuple(self._root.attrs.get("channels", ()))

    @property
    def time_interval_s(self) -> float | None:
        """Time interval between frames in seconds, if recorded."""
        value = self._root.attrs.get("time_interval_s")
        return None if value is None else float(value)

    @property
    def fov_ids(self) -> tuple[int, ...]:
        """Sorted ids of all ``fov_{id}`` groups present in the store."""
        ids: list[int] = []
        for name in self._root.keys():
            if name.startswith("fov_"):
                try:
                    ids.append(int(name[len("fov_") :]))
                except ValueError:
                    logger.debug("ignoring non-fov member %r", name)
        return tuple(sorted(ids))

    # ------------------------------------------------------------------
    # channel roles (semantic typing for segmentation-source selection)
    # ------------------------------------------------------------------
    @property
    def channel_roles(self) -> dict[str, str]:
        """Mapping of channel name to semantic role.

        Roles are one of :data:`CHANNEL_ROLES`; channels without an entry
        are untyped. Stored in root attrs under ``channel_roles``.
        """
        return dict(self._root.attrs.get("channel_roles", {}))

    def set_channel_role(self, channel: str, role: str | None) -> None:
        """Assign (or clear, with ``None``) the semantic role of a channel."""
        if channel not in self.channels:
            raise KeyError(f"unknown channel {channel!r}")
        roles = self.channel_roles
        if role is None:
            roles.pop(channel, None)
        else:
            if role not in CHANNEL_ROLES:
                raise ValueError(
                    f"invalid role {role!r}; expected one of {CHANNEL_ROLES}"
                )
            roles[channel] = role
        self._root.attrs["channel_roles"] = roles

    def channels_by_role(self, role: str) -> tuple[str, ...]:
        """Channels tagged with ``role``, in registration order."""
        roles = self.channel_roles
        return tuple(c for c in self.channels if roles.get(c) == role)

    # ------------------------------------------------------------------
    # group / array access
    # ------------------------------------------------------------------
    @staticmethod
    def _fov_name(fov_id: int) -> str:
        return f"fov_{int(fov_id)}"

    def _fov_group(self, fov_id: int, create: bool = False) -> zarr.Group:
        name = self._fov_name(fov_id)
        if create:
            return self._root.require_group(name)
        if name not in self._root:
            raise KeyError(f"no such fov group: {name!r}")
        return self._root[name]

    @staticmethod
    def _spatial_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
        """Chunk shape ``(1, ..., min(512, y), min(512, x))`` per §2.2."""
        chunks = [1] * (len(shape) - 2)
        chunks.append(min(CHUNK_Y, shape[-2]))
        chunks.append(min(CHUNK_X, shape[-1]))
        return tuple(chunks)

    def images(self, fov_id: int, channel: str) -> zarr.Array:
        """Lazy handle on ``fov_{id}/images/{channel}`` (never loads data).

        Raises
        ------
        KeyError
            If the fov or channel array does not exist.
        """
        group = self._fov_group(fov_id)
        if "images" not in group or channel not in group["images"]:
            raise KeyError(f"no images array for fov={fov_id} channel={channel!r}")
        return group["images"][channel]

    def labels(self, fov_id: int) -> zarr.Array:
        """Lazy handle on ``fov_{id}/labels`` (uint32 instance labels).

        Raises
        ------
        KeyError
            If the labels array does not exist.
        """
        group = self._fov_group(fov_id)
        if "labels" not in group:
            raise KeyError(f"no labels array for fov={fov_id}")
        return group["labels"]

    def prob(self, fov_id: int) -> zarr.Array:
        """Lazy handle on the optional ``fov_{id}/prob`` probability stack.

        Raises
        ------
        KeyError
            If the prob array does not exist.
        """
        group = self._fov_group(fov_id)
        if "prob" not in group:
            raise KeyError(f"no prob array for fov={fov_id}")
        return group["prob"]

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def _register_channel(self, channel: str) -> None:
        channels = list(self.channels)
        if channel not in channels:
            channels.append(channel)
            self._root.attrs["channels"] = channels
            logger.info("registered channel %r in store attrs", channel)

    def write_images(
        self, fov_id: int, channel: str, images: np.ndarray
    ) -> zarr.Array:
        """Write (or replace) one channel stack ``(t, y, x)``.

        Parameters
        ----------
        fov_id : int
            Field-of-view index.
        channel : str
            Channel name; auto-registered in root attrs if new.
        images : numpy.ndarray
            ``(t, y, x)`` stack, typically uint16 per §2.2. Stored with
            chunks ``(1, 512, 512)`` (clipped to the frame size).

        Returns
        -------
        zarr.Array
            The written array handle.
        """
        images = np.asarray(images)
        if images.ndim != 3:
            raise ValueError(f"images must be (t, y, x), got shape {images.shape}")
        group = self._fov_group(fov_id, create=True)
        images_group = group.require_group("images")
        if channel in images_group:
            del images_group[channel]
        arr = images_group.create_array(
            channel,
            shape=images.shape,
            chunks=self._spatial_chunks(images.shape),
            dtype=images.dtype,
        )
        arr[...] = images
        self._register_channel(channel)
        logger.info(
            "wrote images fov=%d channel=%r shape=%s dtype=%s",
            fov_id,
            channel,
            images.shape,
            images.dtype,
        )
        return arr

    def write_labels(self, fov_id: int, labels: np.ndarray) -> zarr.Array:
        """Write (or replace) the instance-label stack ``(t, y, x)`` uint32.

        Parameters
        ----------
        fov_id : int
            Field-of-view index.
        labels : numpy.ndarray
            ``(t, y, x)`` integer stack; cast to uint32.

        Returns
        -------
        zarr.Array
            The written array handle.
        """
        labels = np.asarray(labels)
        if labels.ndim != 3:
            raise ValueError(f"labels must be (t, y, x), got shape {labels.shape}")
        labels = labels.astype(np.uint32, copy=False)
        group = self._fov_group(fov_id, create=True)
        if "labels" in group:
            del group["labels"]
        arr = group.create_array(
            "labels",
            shape=labels.shape,
            chunks=self._spatial_chunks(labels.shape),
            dtype=np.uint32,
        )
        arr[...] = labels
        logger.info("wrote labels fov=%d shape=%s", fov_id, labels.shape)
        return arr

    def write_prob(self, fov_id: int, prob: np.ndarray) -> zarr.Array:
        """Write (or replace) the optional probability stack ``(t, c, y, x)``.

        Parameters
        ----------
        fov_id : int
            Field-of-view index.
        prob : numpy.ndarray
            ``(t, c, y, x)`` stack; cast to float16 per §2.2.

        Returns
        -------
        zarr.Array
            The written array handle.
        """
        prob = np.asarray(prob)
        if prob.ndim != 4:
            raise ValueError(f"prob must be (t, c, y, x), got shape {prob.shape}")
        prob = prob.astype(np.float16, copy=False)
        group = self._fov_group(fov_id, create=True)
        if "prob" in group:
            del group["prob"]
        arr = group.create_array(
            "prob",
            shape=prob.shape,
            chunks=self._spatial_chunks(prob.shape),
            dtype=np.float16,
        )
        arr[...] = prob
        logger.info("wrote prob fov=%d shape=%s", fov_id, prob.shape)
        return arr

    def set_labels_frame(self, fov_id: int, t: int, frame_labels: np.ndarray) -> None:
        """Write one time point of the labels stack in place.

        If the labels array does not exist yet it is created with shape
        ``(t + 1, y, x)``; if ``t`` lies beyond the current stack depth the
        first axis is resized. This supports incremental/manual label edits
        without rewriting the whole stack.

        Parameters
        ----------
        fov_id : int
            Field-of-view index.
        t : int
            Time point (non-negative).
        frame_labels : numpy.ndarray
            ``(y, x)`` label image, cast to uint32.
        """
        t = int(t)
        if t < 0:
            raise ValueError(f"t must be non-negative, got {t}")
        frame_labels = np.asarray(frame_labels).astype(np.uint32, copy=False)
        if frame_labels.ndim != 2:
            raise ValueError(
                f"frame_labels must be (y, x), got shape {frame_labels.shape}"
            )
        try:
            arr = self.labels(fov_id)
        except KeyError:
            group = self._fov_group(fov_id, create=True)
            shape = (t + 1, frame_labels.shape[0], frame_labels.shape[1])
            arr = group.create_array(
                "labels",
                shape=shape,
                chunks=self._spatial_chunks(shape),
                dtype=np.uint32,
                fill_value=0,
            )
            logger.info("created labels array fov=%d shape=%s", fov_id, shape)
        if arr.shape[1:] != frame_labels.shape:
            raise ValueError(
                f"frame shape {frame_labels.shape} does not match labels "
                f"spatial shape {arr.shape[1:]}"
            )
        if t >= arr.shape[0]:
            new_shape = (t + 1, arr.shape[1], arr.shape[2])
            logger.info("resizing labels fov=%d %s -> %s", fov_id, arr.shape, new_shape)
            arr.resize(new_shape)
        arr[t] = frame_labels

    # ------------------------------------------------------------------
    # lazy reads
    # ------------------------------------------------------------------
    def get_frame(self, fov_id: int, channel: str, t: int) -> np.ndarray:
        """Read one time point of one channel — never loads the whole stack.

        Parameters
        ----------
        fov_id : int
            Field-of-view index.
        channel : str
            Channel name.
        t : int
            Time point, ``0 <= t < n_frames``.

        Returns
        -------
        numpy.ndarray
            ``(y, x)`` frame with the channel's dtype.

        Raises
        ------
        KeyError
            If the fov/channel does not exist.
        IndexError
            If ``t`` is out of range.
        """
        arr = self.images(fov_id, channel)
        t = int(t)
        if not 0 <= t < arr.shape[0]:
            raise IndexError(f"t={t} out of range for {arr.shape[0]} frames")
        return np.asarray(arr[t])

    def get_labels_frame(self, fov_id: int, t: int) -> np.ndarray:
        """Read one time point of the labels stack.

        Raises
        ------
        KeyError
            If the labels array does not exist.
        IndexError
            If ``t`` is out of range.
        """
        arr = self.labels(fov_id)
        t = int(t)
        if not 0 <= t < arr.shape[0]:
            raise IndexError(f"t={t} out of range for {arr.shape[0]} frames")
        return np.asarray(arr[t])

    # ------------------------------------------------------------------
    # shapes
    # ------------------------------------------------------------------
    def shape(self, fov_id: int, channel: str | None = None) -> tuple[int, int, int]:
        """``(t, y, x)`` shape of one channel (default: first registered)."""
        if channel is None:
            channels = self.channels
            if not channels:
                raise KeyError(f"no channels registered for fov={fov_id}")
            channel = channels[0]
        arr = self.images(fov_id, channel)
        return int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])

    def n_frames(self, fov_id: int) -> int:
        """Number of time frames recorded for one fov."""
        return self.shape(fov_id)[0]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ExperimentStore(path={self._path!r}, fovs={self.fov_ids}, "
            f"channels={self.channels})"
        )
