"""Single-step preview sandbox (GUI test-tune-commit loop).

This module implements the "preview on current frame" mechanism used by the
Pipeline dock widget. It is deliberately Qt-free so the logic is testable
headless; the widget layer only handles user interaction and napari layer
display.

Workflow
--------
1. :func:`build_sandbox_store` copies the *current frame* of the currently
   open store (all channels, plus labels if present) into a throwaway
   one-frame store inside a temporary directory.
2. :func:`preview_step` runs exactly one pipeline step against the sandbox
   through the regular :mod:`momia3.pipeline.engine`, so parameter
   validation, artifact materialization and error reporting behave exactly
   as in a real run. The real store is never touched.

Input wiring rules
------------------
- ``store`` inputs are fed from the sandbox store.
- ``table:cells`` inputs are produced by an implicit ``measure.cells`` step
  (default parameters) run against the sandbox labels — this covers
  ``outline.extract`` / ``midline.extract`` tuning without requiring a
  prior real run.
- ``array`` inputs are assembled from the sandbox store's current frame
  by argument-name convention (see :func:`_array_source_for`): ``image``/
  ``stack``/``intensity`` come from the selected channel frame, ``labels``/
  ``mask`` from the labels frame. Intermediate arrays (``dist``, ``prob``,
  ``patches``, …) and ``json``/``table:edges`` inputs are not previewable
  and raise ``ValueError``.
- Any other input kind raises ``ValueError`` (preview unsupported).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from momia3.pipeline import engine
from momia3.pipeline.registry import StepContext, get, register
from momia3.storage.zarr_store import ExperimentStore

logger = logging.getLogger(__name__)

__all__ = ["build_sandbox_store", "preview_step", "PreviewResult"]


# ---------------------------------------------------------------------------
# sandbox store
# ---------------------------------------------------------------------------


def build_sandbox_store(
    src: ExperimentStore,
    fov_id: int,
    t: int,
    dst_dir: Path,
) -> Path:
    """Copy one frame of ``src`` into a fresh one-frame store in ``dst_dir``.

    Returns the path of the sandbox ``experiment.zarr``. Labels are copied
    when the source store has a labels stack for the FOV.
    """
    store_path = dst_dir / "experiment.zarr"
    sandbox = ExperimentStore.create(
        store_path,
        pixel_microns=src.pixel_microns,
        channels=list(src.channels),
        time_interval_s=src.time_interval_s,
        provenance={"op": "preview.sandbox", "source": str(src.path)},
    )
    for channel in src.channels:
        sandbox.write_images(fov_id, channel, src.get_frame(fov_id, channel, t)[np.newaxis])
    try:
        labels_frame = src.get_labels_frame(fov_id, t)
    except KeyError:
        labels_frame = None
    if labels_frame is not None:
        sandbox.write_labels(fov_id, labels_frame[np.newaxis])
    roles = src.channel_roles
    if roles:
        sandbox.attrs["channel_roles"] = dict(roles)
    return store_path


# ---------------------------------------------------------------------------
# sandbox store source op (registry-only, GUI/preview internal)
# ---------------------------------------------------------------------------


class _SandboxStoreParams(BaseModel):
    """Parameters of the ``preview.sandbox_store`` helper op."""

    path: str = Field(description="Path of the sandbox experiment.zarr")


def _run_sandbox_store(
    inputs: dict[str, Any], params: _SandboxStoreParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of ``preview.sandbox_store`` (passes the path through)."""
    del inputs, ctx
    return {"experiment": Path(params.path)}


register(
    "preview.sandbox_store",
    params_model=_SandboxStoreParams,
    inputs={},
    outputs={"experiment": "store"},
    shard=None,
    runner=_run_sandbox_store,
    description=(
        "Yield an already-created sandbox store by path (preview helper; "
        "the store is built by build_sandbox_store beforehand)."
    ),
)


# ---------------------------------------------------------------------------
# sandbox frame array source op (preview helper)
# ---------------------------------------------------------------------------


class _FrameArrayParams(BaseModel):
    """Parameters of the ``preview.frame_array`` helper op."""

    arg: str = Field(description="Pipeline input argument name being served")
    channel: str = Field(description="Channel name; '__labels__' serves the labels frame")


def _array_source_for(arg: str) -> str | None:
    """Map a pipeline argument name to a sandbox-store frame source.

    Returns ``"channel"`` (the selected channel's current frame) or
    ``"labels"`` (the labels frame), or ``None`` when the argument is an
    intermediate array that cannot be produced from the store alone
    (``dist``, ``prob``, ``patches``, ``grid``, …).
    """
    if arg in ("image", "stack", "intensity"):
        return "channel"
    if arg in ("labels", "mask"):
        return "labels"
    return None


def _run_frame_array(
    inputs: dict[str, Any], params: _FrameArrayParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of ``preview.frame_array`` (serves one frame as a 2D array)."""
    del ctx
    sandbox = ExperimentStore.open(inputs["experiment"], mode="r")
    fov_id = sandbox.fov_ids[0]
    if params.channel == "__labels__":
        frame = sandbox.get_labels_frame(fov_id, 0)
    else:
        frame = sandbox.get_frame(fov_id, params.channel, 0)
    return {"value": np.asarray(frame)}


register(
    "preview.frame_array",
    params_model=_FrameArrayParams,
    inputs={"experiment": "store"},
    outputs={"value": "array"},
    shard=None,
    runner=_run_frame_array,
    description=(
        "Serve one frame of the sandbox store as a 2D array input (preview "
        "helper; argument-name convention of _array_source_for)."
    ),
)


# ---------------------------------------------------------------------------
# single-step preview
# ---------------------------------------------------------------------------


class PreviewResult:
    """Outcome of :func:`preview_step`."""

    def __init__(self, outputs: dict[str, Any], sandbox_store: Path | None) -> None:
        self.outputs = outputs
        self.sandbox_store = sandbox_store

    def table(self, name: str) -> Any:
        """Output ``name`` as a polars DataFrame (raises if not a table)."""
        import polars as pl

        value = self.outputs[name]
        return pl.from_arrow(value)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        keys = {k: type(v).__name__ for k, v in self.outputs.items()}
        return f"PreviewResult(outputs={keys}, sandbox_store={self.sandbox_store})"


def preview_step(
    op_name: str,
    params: dict[str, Any],
    src: ExperimentStore,
    fov_id: int,
    t: int,
    sandbox_dir: Path,
    channel: str | None = None,
) -> PreviewResult:
    """Run one pipeline step against a one-frame sandbox copy of ``src``.

    Parameters
    ----------
    op_name : str
        Registered operation to test.
    params : dict
        Parameter values (validated against the op's pydantic model).
    src : ExperimentStore
        The real (open) store providing the current frame; never mutated.
    fov_id, t : int
        FOV and time point to preview.
    sandbox_dir : pathlib.Path
        Temporary directory holding the sandbox store and artifacts. The
        caller owns its lifetime; display code must read results eagerly
        before deleting it.
    channel : str, optional
        Channel serving ``array`` inputs named ``image``/``stack``/
        ``intensity``; defaults to the store's first channel.

    Returns
    -------
    PreviewResult
        In-memory outputs of the step plus the sandbox store path when the
        step outputs a store (for labels/prob display).
    """
    spec = get(op_name)
    # resolve() so paths match the engine's resolved workspace (macOS
    # /var -> /private/var symlink would otherwise break relative_to()).
    sandbox_dir = Path(sandbox_dir).resolve()
    store_path = build_sandbox_store(src, fov_id, t, sandbox_dir / "sandbox")
    channel = channel or (src.channels[0] if src.channels else None)

    steps: list[dict[str, Any]] = [
        {
            "id": "sandbox",
            "op": "preview.sandbox_store",
            "params": {"path": str(store_path)},
            "outputs": {"experiment": "experiment.zarr"},
        }
    ]
    inputs: dict[str, str] = {}
    for arg, kind in spec.inputs.items():
        kind = kind.rstrip("?")
        if kind == "store":
            inputs[arg] = "sandbox.experiment"
        elif kind == "table:cells":
            steps.append(
                {
                    "id": "cells",
                    "op": "measure.cells",
                    "params": {},
                    "inputs": {"experiment": "sandbox.experiment"},
                    "outputs": {"cells": "cells/cells.parquet"},
                }
            )
            inputs[arg] = "cells.cells"
        elif kind == "array":
            source = _array_source_for(arg)
            if source is None:
                raise ValueError(
                    f"preview cannot produce the array input {arg!r} "
                    f"(an intermediate array, not a store frame)"
                )
            arr_channel = channel or "__labels__"
            if source == "labels":
                arr_channel = "__labels__"
            arr_id = f"arr_{arg}"
            steps.append(
                {
                    "id": arr_id,
                    "op": "preview.frame_array",
                    "params": {"arg": arg, "channel": arr_channel},
                    "inputs": {"experiment": "sandbox.experiment"},
                    "outputs": {"value": arg},
                }
            )
            inputs[arg] = f"{arr_id}.value"
        else:
            raise ValueError(
                f"preview does not support input {arg!r} of kind {kind!r} "
                f"(op {op_name!r})"
            )
    step_id = "preview"
    steps.append(
        {
            "id": step_id,
            "op": op_name,
            "params": dict(params),
            "inputs": inputs,
            "outputs": {},
        }
    )
    result = engine.run(
        {"name": "preview", "steps": steps},
        workspace=sandbox_dir,
        use_cache=False,
    )
    outputs = result.outputs[step_id]
    store_out = next(
        (Path(v) for n, v in outputs.items() if spec.outputs.get(n) == "store"),
        None,
    )
    return PreviewResult(outputs, store_out)
