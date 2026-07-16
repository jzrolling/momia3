"""Field: an immutable view on one fov of an :class:`ExperimentStore`.

A ``Field`` is a lightweight, immutable bundle of a zarr store handle plus
an ``fov_id`` (ARCHITECTURE.md §1). All frame access is lazy and routed
through the store's single-time-point slicing — whole stacks are never
loaded into memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from momia3.storage.zarr_store import ExperimentStore

__all__ = ["Field"]


@dataclass(frozen=True)
class Field:
    """Immutable frame bundle referencing one fov in an experiment zarr store.

    Parameters
    ----------
    store : momia3.storage.zarr_store.ExperimentStore
        Open experiment store.
    fov_id : int
        Field-of-view index within the store.
    """

    store: ExperimentStore
    fov_id: int

    @property
    def channels(self) -> tuple[str, ...]:
        """Registered channel names of the experiment."""
        return self.store.channels

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(t, y, x)`` shape of the first channel."""
        return self.store.shape(self.fov_id)

    @property
    def n_frames(self) -> int:
        """Number of time frames."""
        return self.store.n_frames(self.fov_id)

    @property
    def pixel_microns(self) -> float:
        """Physical pixel size in microns."""
        return self.store.pixel_microns

    @property
    def time_interval_s(self) -> float | None:
        """Time interval between frames in seconds, if recorded."""
        return self.store.time_interval_s

    def frame(self, channel: str, t: int) -> np.ndarray:
        """Lazily read one ``(y, x)`` frame of ``channel`` at time ``t``.

        Raises
        ------
        KeyError
            If the channel does not exist for this fov.
        IndexError
            If ``t`` is out of range.
        """
        if channel not in self.channels:
            raise KeyError(
                f"unknown channel {channel!r}; available: {list(self.channels)}"
            )
        return self.store.get_frame(self.fov_id, channel, t)

    def labels(self, t: int) -> np.ndarray:
        """Lazily read the ``(y, x)`` instance-label image at time ``t``.

        Raises
        ------
        KeyError
            If no labels array exists for this fov.
        IndexError
            If ``t`` is out of range.
        """
        return self.store.get_labels_frame(self.fov_id, t)

    def set_labels(self, t: int, arr: np.ndarray) -> None:
        """Write one ``(y, x)`` label image at time ``t`` (creates/extends
        the labels stack as needed)."""
        self.store.set_labels_frame(self.fov_id, t, arr)
