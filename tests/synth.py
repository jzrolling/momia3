"""Synthetic rod-bacteria timelapse generator with ground truth (§7.1).

Simulates capsule-shaped (rod) bacteria that elongate along their long
axis, divide into two daughters at a target length (with optional
V-snapping kink), jiggle slightly, and share a whole-field drift. Frames
are rendered as phase-contrast-like grayscale images (dark rods on a
bright background), noised and digitized at 16-bit resolution.

Everything is deterministic given ``seed``; all randomness flows through
one :class:`numpy.random.Generator` in a fixed call order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from skimage.draw import disk, polygon

from momia3.keys import ROOT_KEY, encode_key

logger = logging.getLogger(__name__)

__all__ = ["generate_timelapse"]

_BACKGROUND: float = 0.85
"""Bright background intensity (phase contrast)."""

_CELL_TRANSLATE_JITTER_PX: float = 0.2
"""Per-cell, per-frame random-walk sigma (on top of whole-field drift)."""

_MAX_CELLS: int = 400
"""Safety cap on simultaneously tracked cells."""


@dataclass
class _CellState:
    """Internal per-cell simulation state (all positions in pixels)."""

    track_id: int
    label: int
    y: float
    x: float
    angle: float  # radians; long-axis direction in the (y, x) plane
    length_px: float
    width_px: float
    intensity: float

    @property
    def unit(self) -> tuple[float, float]:
        """Long-axis unit vector ``(uy, ux)``."""
        return float(np.sin(self.angle)), float(np.cos(self.angle))


def _capsule_mask(state: _CellState, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize one rod as a capsule: line segment + semicircular caps.

    Uses :func:`skimage.draw.polygon` for the rectangular body and
    :func:`skimage.draw.disk` for the two caps; coordinates are clipped to
    ``shape`` by skimage.
    """
    radius = state.width_px / 2.0
    half_segment = max(state.length_px / 2.0 - radius, 0.0)
    uy, ux = state.unit
    # Segment endpoints (pole-to-pole axis shortened by one radius each side).
    p1y, p1x = state.y - uy * half_segment, state.x - ux * half_segment
    p2y, p2x = state.y + uy * half_segment, state.x + ux * half_segment
    # Perpendicular unit vector for the rectangle body.
    py, px = -ux, uy
    rows = np.array(
        [p1y + py * radius, p2y + py * radius, p2y - py * radius, p1y - py * radius]
    )
    cols = np.array(
        [p1x + px * radius, p2x + px * radius, p2x - px * radius, p1x - px * radius]
    )
    body_r, body_c = polygon(rows, cols, shape=shape)
    cap1_r, cap1_c = disk((p1y, p1x), radius, shape=shape)
    cap2_r, cap2_c = disk((p2y, p2x), radius, shape=shape)
    rr = np.concatenate([body_r, cap1_r, cap2_r])
    cc = np.concatenate([body_c, cap1_c, cap2_c])
    return rr, cc


def _render_frame(
    states: list[_CellState],
    shape: tuple[int, int],
    rng: np.random.Generator,
    noise_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one phase-contrast frame and its ground-truth label image.

    Rods are blended with ``min`` (overlaps get darker). Noise is gaussian,
    then the frame is clipped to [0, 1] and digitized on a 16-bit grid to
    emulate camera quantization; the result is returned as float32.
    """
    image = np.full(shape, _BACKGROUND, dtype=np.float32)
    labels = np.zeros(shape, dtype=np.uint32)
    for state in states:
        rr, cc = _capsule_mask(state, shape)
        image[rr, cc] = np.minimum(image[rr, cc], np.float32(state.intensity))
        labels[rr, cc] = np.uint32(state.label)
    if noise_sigma > 0:
        image += rng.normal(0.0, noise_sigma, shape).astype(np.float32)
    np.clip(image, 0.0, 1.0, out=image)
    image = (
        np.round(image.astype(np.float64) * 65535.0).astype(np.uint16).astype(np.float32)
        / np.float32(65535.0)
    )
    return image, labels


def _divide(
    parent: _CellState,
    label_alloc,
    track_alloc,
    rng: np.random.Generator,
    vsnap: bool,
) -> tuple[_CellState, _CellState]:
    """Split one (already grown) cell into two daughters along its axis.

    Daughters get fresh labels/track ids and a small septum gap. Normal
    division adds independent angular jitter of a few degrees; V-snapping
    produces a symmetric kink of 15–35 degrees between the daughters.
    """
    gap_px = max(1.0, 0.3 * parent.width_px)
    daughter_len = (parent.length_px - gap_px) / 2.0
    offset = (daughter_len + gap_px) / 2.0
    uy, ux = parent.unit
    if vsnap:
        kink = np.deg2rad(rng.uniform(15.0, 35.0)) * rng.choice([-1.0, 1.0])
        angle1 = parent.angle + kink
        angle2 = parent.angle - kink
    else:
        angle1 = parent.angle + np.deg2rad(rng.uniform(-3.0, 3.0))
        angle2 = parent.angle + np.deg2rad(rng.uniform(-3.0, 3.0))
    daughter1 = _CellState(
        track_id=track_alloc(),
        label=label_alloc(),
        y=parent.y + uy * offset,
        x=parent.x + ux * offset,
        angle=float(angle1),
        length_px=daughter_len,
        width_px=parent.width_px,
        intensity=float(parent.intensity * rng.uniform(0.95, 1.05)),
    )
    daughter2 = _CellState(
        track_id=track_alloc(),
        label=label_alloc(),
        y=parent.y - uy * offset,
        x=parent.x - ux * offset,
        angle=float(angle2),
        length_px=daughter_len,
        width_px=parent.width_px,
        intensity=float(parent.intensity * rng.uniform(0.95, 1.05)),
    )
    return daughter1, daughter2


def generate_timelapse(
    n_frames: int = 20,
    shape: tuple[int, int] = (512, 512),
    n_initial: int = 5,
    pixel_microns: float = 0.065,
    growth_rate_um_per_frame: float = 0.05,
    division_length_um: float = 4.0,
    drift_px_per_frame: float = 0.5,
    noise_sigma: float = 0.02,
    vsnap: bool = False,
    seed: int = 0,
) -> dict:
    """Generate a synthetic growing/dividing rod-bacteria timelapse.

    Parameters
    ----------
    n_frames : int
        Number of time frames (>= 1).
    shape : tuple of int
        Frame shape ``(y, x)`` in pixels.
    n_initial : int
        Number of founder cells at frame 0.
    pixel_microns : float
        Physical pixel size in microns.
    growth_rate_um_per_frame : float
        Elongation per frame along the long axis, in microns.
    division_length_um : float
        Length at which a cell divides into two daughters.
    drift_px_per_frame : float
        Whole-field drift magnitude per frame (fixed random direction).
    noise_sigma : float
        Gaussian noise sigma in [0, 1] intensity units.
    vsnap : bool
        If True, daughters snap apart with a large symmetric kink
        (V-snapping division); otherwise a small angular jitter is used.
    seed : int
        RNG seed; output is fully deterministic given the seed.

    Returns
    -------
    dict
        ``images``
            float32 ``(t, y, x)`` in [0, 1], digitized at 16-bit
            resolution (dark rods on bright background).
        ``labels``
            uint32 ``(t, y, x)`` ground-truth instance labels; a cell
            keeps its label across frames until it divides.
        ``edges``
            Ground-truth lineage edge list of ``(parent_key, child_key,
            event)`` tuples with event in ``{"appear", "continue",
            "division"}``. ``parent_key`` is :data:`momia3.keys.ROOT_KEY`
            (0) for ``appear`` events. Keys use
            :func:`momia3.keys.encode_key` with ``fov_id=0``.
        ``truth``
            Per-cell, per-frame records (list of dicts) with
            ``cell_key``, ``fov_id``, ``frame``, ``label``, ``track_id``,
            ``length_um``, ``width_um``, ``center_y``, ``center_x`` and
            ``angle_rad``.
        ``meta``
            Echo of the simulation parameters.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if len(shape) != 2 or shape[0] < 8 or shape[1] < 8:
        raise ValueError(f"shape must be (y, x) with sides >= 8, got {shape}")
    if n_initial < 1:
        raise ValueError(f"n_initial must be >= 1, got {n_initial}")
    if pixel_microns <= 0:
        raise ValueError(f"pixel_microns must be positive, got {pixel_microns}")
    if growth_rate_um_per_frame < 0:
        raise ValueError("growth_rate_um_per_frame must be non-negative")
    if division_length_um <= 0:
        raise ValueError("division_length_um must be positive")

    rng = np.random.default_rng(seed)
    growth_px = growth_rate_um_per_frame / pixel_microns
    max_width_px = 0.85 / pixel_microns
    division_length_px = division_length_um / pixel_microns

    # Keep founders far enough from the border that a full-size rod fits.
    margin = int(np.ceil(division_length_px / 2.0 + max_width_px + 4.0))
    if 2 * margin >= min(shape):
        margin = max(min(shape) // 4, 1)
        logger.info("small frame; reduced founder margin to %d px", margin)

    drift_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    drift_dy = drift_px_per_frame * float(np.sin(drift_angle))
    drift_dx = drift_px_per_frame * float(np.cos(drift_angle))

    next_label = 1
    next_track = 1

    def label_alloc() -> int:
        nonlocal next_label
        value = next_label
        next_label += 1
        return value

    def track_alloc() -> int:
        nonlocal next_track
        value = next_track
        next_track += 1
        return value

    states: list[_CellState] = []
    edges: list[tuple[int, int, str]] = []
    for _ in range(n_initial):
        state = _CellState(
            track_id=track_alloc(),
            label=label_alloc(),
            y=float(rng.uniform(margin, shape[0] - margin)),
            x=float(rng.uniform(margin, shape[1] - margin)),
            angle=float(rng.uniform(0.0, np.pi)),
            length_px=float(rng.uniform(0.55, 0.95)) * division_length_px,
            width_px=float(rng.uniform(0.75, 0.85)) / pixel_microns,
            intensity=float(rng.uniform(0.35, 0.50)),
        )
        states.append(state)
        edges.append((ROOT_KEY, encode_key(0, 0, state.label), "appear"))

    images = np.empty((n_frames, shape[0], shape[1]), dtype=np.float32)
    labels = np.empty((n_frames, shape[0], shape[1]), dtype=np.uint32)
    truth: list[dict] = []

    for t in range(n_frames):
        frame, frame_labels = _render_frame(states, shape, rng, noise_sigma)
        images[t] = frame
        labels[t] = frame_labels
        for state in states:
            truth.append(
                {
                    "cell_key": encode_key(0, t, state.label),
                    "fov_id": 0,
                    "frame": t,
                    "label": state.label,
                    "track_id": state.track_id,
                    "length_um": state.length_px * pixel_microns,
                    "width_um": state.width_px * pixel_microns,
                    "center_y": state.y,
                    "center_x": state.x,
                    "angle_rad": state.angle,
                }
            )
        if t == n_frames - 1:
            break

        updated: list[_CellState] = []
        for state in states:
            grown = _CellState(
                track_id=state.track_id,
                label=state.label,
                y=state.y + drift_dy + float(rng.normal(0.0, _CELL_TRANSLATE_JITTER_PX)),
                x=state.x + drift_dx + float(rng.normal(0.0, _CELL_TRANSLATE_JITTER_PX)),
                angle=state.angle,
                length_px=state.length_px + growth_px,
                width_px=state.width_px,
                intensity=state.intensity,
            )
            parent_key = encode_key(0, t, state.label)
            if (
                grown.length_px * pixel_microns >= division_length_um
                and len(states) + len(updated) < _MAX_CELLS
            ):
                daughter1, daughter2 = _divide(
                    grown, label_alloc, track_alloc, rng, vsnap
                )
                edges.append((parent_key, encode_key(0, t + 1, daughter1.label), "division"))
                edges.append((parent_key, encode_key(0, t + 1, daughter2.label), "division"))
                updated.extend([daughter1, daughter2])
            else:
                edges.append((parent_key, encode_key(0, t + 1, grown.label), "continue"))
                updated.append(grown)
        states = updated

    logger.info(
        "generated timelapse: %d frames, %d final cells, %d edges (seed=%d)",
        n_frames,
        len(states),
        len(edges),
        seed,
    )
    return {
        "images": images,
        "labels": labels,
        "edges": edges,
        "truth": truth,
        "meta": {
            "n_frames": n_frames,
            "shape": tuple(shape),
            "n_initial": n_initial,
            "pixel_microns": pixel_microns,
            "growth_rate_um_per_frame": growth_rate_um_per_frame,
            "division_length_um": division_length_um,
            "drift_px_per_frame": drift_px_per_frame,
            "noise_sigma": noise_sigma,
            "vsnap": vsnap,
            "seed": seed,
        },
    }
