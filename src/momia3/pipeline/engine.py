"""JSON DAG execution engine (ARCHITECTURE.md §3.1/§3.2).

Pipeline file format
--------------------
A pipeline is a JSON object::

    {
      "name": "my_pipeline",                  // optional
      "workspace": "path/to/workspace",       // optional; relative paths resolve
                                              // against the pipeline file's directory
      "steps": [
        {
          "id": "import",                     // unique step id (no dots)
          "op": "io.import_synth",            // registered operation name
          "params": {"n_frames": 12, ...},    // validated against the op's pydantic model
          "inputs": {"experiment": "upstream.experiment"},   // optional
          "outputs": {"experiment": "experiment.zarr"}       // optional; workspace-relative
        }
      ]
    }

``inputs`` maps an op argument name to a ``"<step_id>.<output>"`` reference;
``outputs`` maps an op output name to a workspace-relative artifact path
(defaults to ``<step_id>/<output><ext>`` when omitted).

Execution
---------
- Steps run in topological order (:func:`validate` rejects cycles,
  unknown ops, invalid params and unresolvable references beforehand).
- Every step's outputs are materialized into the workspace: zarr stores
  (written by the runner itself), parquet tables (schema-validated for the
  three canonical kinds), JSON scalars, ``.npy`` arrays.
- Content-addressed caching (§2.5): the step cache key is
  ``sha256(op + version + params + upstream step keys)``; a second
  identical run is fully cache-hit (artifacts are restored without
  recomputation). Cache entries live in ``<workspace>/.momia3_cache/`` with
  a provenance record per step; a human-readable provenance copy is written
  to ``<workspace>/provenance/<step_id>.json``.
- Parallelism (§3.2): ops with shard ``"frame"``/``"fov"`` map their work
  items over :class:`concurrent.futures.ProcessPoolExecutor` when
  ``n_workers > 1`` (serial fallback otherwise). The sharding itself is
  implemented by the runners in :mod:`momia3.pipeline.steps` through
  :meth:`momia3.pipeline.registry.StepContext.map_shards`.

Store-mutation caveat
---------------------
``segment.threshold`` writes labels *into* its input store. Its cache entry
verifies that ``fov_*/labels`` members still exist on restore; if an
intermediate store was deleted, the step (and its upstream) is recomputed.
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ValidationError

from momia3.pipeline.registry import REGISTRY, OpSpec, StepContext
from momia3.storage.cache import CacheStore, cache_key, make_provenance
from momia3.storage.tables import write_table

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineError",
    "RunResult",
    "StepRecord",
    "load_pipeline",
    "run",
    "validate",
]

_TABLE_KINDS = ("table", "table:cells", "table:edges", "table:tracks")
_DEFAULT_EXT = {
    "store": "",
    "json": ".json",
    "array": ".npy",
    "table": ".parquet",
    "table:cells": ".parquet",
    "table:edges": ".parquet",
    "table:tracks": ".parquet",
}


class PipelineError(Exception):
    """Raised when a pipeline fails validation or execution.

    Attributes
    ----------
    errors : list of str
        Human-readable error messages.
    """

    def __init__(self, errors: list[str] | str) -> None:
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(self.errors))


@dataclass
class StepRecord:
    """Bookkeeping of one executed step."""

    step_id: str
    op: str
    status: str  # "computed" | "cached"
    cache_key: str
    duration_s: float
    outputs: dict[str, str]  # output name -> workspace-relative path


@dataclass
class RunResult:
    """Result of :func:`run`."""

    name: str
    workspace: Path
    steps: list[StepRecord]
    stats: dict[str, int]  # {"computed": n, "cached": n}
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    """In-memory output values per step id (store paths, tables, json, arrays)."""


# ---------------------------------------------------------------------------
# loading / graph helpers
# ---------------------------------------------------------------------------


def load_pipeline(path: str | Path) -> dict[str, Any]:
    """Load a pipeline JSON file.

    Raises
    ------
    PipelineError
        If the file does not parse as a JSON object.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot load pipeline {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PipelineError(f"pipeline {path} must be a JSON object")
    return data


def _split_ref(ref: Any) -> tuple[str, str]:
    """Split a ``"<step_id>.<output>"`` input reference."""
    if not isinstance(ref, str) or "." not in ref:
        raise PipelineError(
            f"input reference must be '<step_id>.<output>', got {ref!r}"
        )
    step_id, _, output = ref.rpartition(".")
    if not step_id or not output:
        raise PipelineError(f"malformed input reference {ref!r}")
    return step_id, output


def _dependencies(
    steps: list[Mapping[str, Any]],
) -> dict[str, set[str]]:
    """Dependency edges ``step_id -> {producer step ids}`` from input refs."""
    deps: dict[str, set[str]] = {}
    for step in steps:
        sid = str(step.get("id", ""))
        inputs = step.get("inputs", {})
        refs: set[str] = set()
        if isinstance(inputs, Mapping):
            for ref in inputs.values():
                try:
                    producer, _ = _split_ref(ref)
                except PipelineError:
                    continue  # reported by validate()
                if producer != sid:
                    refs.add(producer)
        deps[sid] = refs
    return deps


def _topo_order(deps: Mapping[str, set[str]]) -> tuple[list[str], set[str]]:
    """Kahn topological sort; returns ``(order, cyclic_node_ids)``."""
    remaining = {sid: set(ds) for sid, ds in deps.items()}
    order: list[str] = []
    ready = sorted(sid for sid, ds in remaining.items() if not ds)
    while ready:
        sid = ready.pop(0)
        if sid not in remaining:
            continue
        order.append(sid)
        del remaining[sid]
        for other, ds in remaining.items():
            if sid in ds:
                ds.discard(sid)
                if not ds:
                    ready.append(other)
        ready.sort()
    return order, set(remaining)


def _kinds_compatible(src: str, dst: str) -> bool:
    """Lenient artifact-type compatibility between an output and an input."""
    dst = dst.rstrip("?")
    if dst == "any" or src == dst:
        return True
    if dst == "table" and src.startswith("table"):
        return True
    if dst.startswith("table") and src == "table":
        return True
    return False


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate(pipeline: Mapping[str, Any]) -> list[str]:
    """Statically validate a pipeline; returns a list of errors (empty = OK).

    Checks: ``steps`` structure, unique dot-free ids, op existence, params
    against the op's pydantic model, input-reference resolution and type
    compatibility, required inputs, output declarations (names + relative
    paths), and acyclicity. Dry-runnable — nothing is executed.

    Parameters
    ----------
    pipeline : mapping
        Parsed pipeline JSON object.

    Returns
    -------
    list of str
        Validation errors; empty when the pipeline is valid.
    """
    errors: list[str] = []
    if not isinstance(pipeline, Mapping):
        return ["pipeline must be a JSON object"]
    if "workspace" in pipeline and not isinstance(pipeline["workspace"], str):
        errors.append("'workspace' must be a string path")
    steps = pipeline.get("steps")
    if not isinstance(steps, list) or not steps:
        return errors + ["pipeline 'steps' must be a non-empty list"]

    ids: set[str] = set()
    specs: dict[str, OpSpec] = {}
    for i, step in enumerate(steps):
        where = f"steps[{i}]"
        if not isinstance(step, Mapping):
            errors.append(f"{where}: step must be an object")
            continue
        sid = step.get("id")
        if not isinstance(sid, str) or not sid:
            errors.append(f"{where}: 'id' must be a non-empty string")
            continue
        where = f"step {sid!r}"
        if "." in sid:
            errors.append(f"{where}: step ids must not contain '.'")
        if sid in ids:
            errors.append(f"{where}: duplicate step id")
        ids.add(sid)
        op_name = step.get("op")
        if not isinstance(op_name, str) or op_name not in REGISTRY:
            errors.append(
                f"{where}: unknown op {op_name!r}; "
                f"available: {sorted(REGISTRY)[:5]} ... ({len(REGISTRY)} total)"
            )
            continue
        spec = REGISTRY[op_name]
        specs[sid] = spec
        try:
            spec.params_model.model_validate(step.get("params", {}))
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            )
            errors.append(f"{where}: params failed validation — {details}")
        inputs = step.get("inputs", {})
        if not isinstance(inputs, Mapping):
            errors.append(f"{where}: 'inputs' must be an object")
            inputs = {}
        for arg in inputs:
            if arg not in spec.inputs:
                errors.append(
                    f"{where}: unknown input {arg!r} for op {op_name!r} "
                    f"(declared: {sorted(spec.inputs)})"
                )
        for arg, kind in spec.inputs.items():
            if not kind.endswith("?") and arg not in inputs:
                errors.append(f"{where}: missing required input {arg!r}")
        outputs = step.get("outputs", {})
        if not isinstance(outputs, Mapping):
            errors.append(f"{where}: 'outputs' must be an object")
            continue
        for name, rel in outputs.items():
            if name not in spec.outputs:
                errors.append(
                    f"{where}: unknown output {name!r} for op {op_name!r} "
                    f"(declared: {sorted(spec.outputs)})"
                )
                continue
            rel_path = Path(str(rel))
            if rel_path.is_absolute() or ".." in rel_path.parts:
                errors.append(
                    f"{where}: output path for {name!r} must be workspace-relative "
                    f"without '..', got {rel!r}"
                )

    # Reference resolution + type compatibility (needs all specs first).
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        sid = step.get("id")
        op_name = step.get("op")
        if not isinstance(sid, str) or sid not in specs:
            continue
        spec = specs[sid]
        inputs = step.get("inputs", {})
        if not isinstance(inputs, Mapping):
            continue
        for arg, ref in inputs.items():
            if arg not in spec.inputs:
                continue
            try:
                producer, out_name = _split_ref(ref)
            except PipelineError as exc:
                errors.append(f"step {sid!r}: {exc}")
                continue
            if producer not in specs:
                errors.append(
                    f"step {sid!r}: input {arg!r} references unknown step {producer!r}"
                )
                continue
            producer_spec = specs[producer]
            if out_name not in producer_spec.outputs:
                errors.append(
                    f"step {sid!r}: step {producer!r} has no output {out_name!r} "
                    f"(declared: {sorted(producer_spec.outputs)})"
                )
                continue
            src_kind = producer_spec.outputs[out_name]
            dst_kind = spec.inputs[arg]
            if not _kinds_compatible(src_kind, dst_kind):
                errors.append(
                    f"step {sid!r}: type mismatch on input {arg!r}: "
                    f"{producer}.{out_name} is {src_kind!r}, expected {dst_kind!r}"
                )

    order, cyclic = _topo_order(_dependencies([s for s in steps if isinstance(s, Mapping)]))
    if cyclic:
        errors.append(f"dependency cycle detected involving: {sorted(cyclic)}")
    del order
    return errors


# ---------------------------------------------------------------------------
# artifact materialization / caching
# ---------------------------------------------------------------------------


def _resolve_out_paths(
    workspace: Path,
    step_id: str,
    spec: OpSpec,
    declared: Mapping[str, Any],
) -> dict[str, Path]:
    """Workspace-absolute path of every declared output of a step."""
    paths: dict[str, Path] = {}
    for name, kind in spec.outputs.items():
        if name in declared:
            rel = Path(str(declared[name]))
        else:
            rel = Path(step_id) / f"{name}{_DEFAULT_EXT.get(kind, '.bin')}"
        paths[name] = workspace / rel
    return paths


def _normalize_value(value: Any, kind: str) -> Any:
    """Canonical in-memory form of an output value."""
    if kind == "store":
        return Path(value)
    if kind in _TABLE_KINDS:
        if isinstance(value, pl.DataFrame):
            return value.to_arrow()
        if not isinstance(value, pa.Table):
            raise PipelineError(
                f"expected a table output, got {type(value).__name__}"
            )
        return value
    if kind == "array":
        return np.asarray(value)
    return value  # json / any


def _infer_channels(table: pa.Table) -> tuple[str, ...]:
    """Channel names of a cells table, inferred from intensity clusters.

    A column prefix only counts as a channel when the full
    ``{ch}_mean/max/min/std/p25/p75/integrated`` cluster is present, so
    scalar morphology columns such as ``curv_mean`` are not mistaken for
    channels.
    """
    names = set(table.column_names)
    channels: set[str] = set()
    for col in names:
        if not col.endswith("_mean"):
            continue
        ch = col[: -len("_mean")]
        if all(
            f"{ch}_{stat}" in names
            for stat in ("max", "min", "std", "p25", "p75", "integrated")
        ):
            channels.add(ch)
    return tuple(sorted(channels))


def _materialize(name: str, kind: str, value: Any, path: Path) -> None:
    """Write one step output to its workspace artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "store":
        if not Path(value).is_dir():
            raise PipelineError(
                f"store output {name!r} was not created at {value}"
            )
    elif kind == "table:cells":
        write_table(value, path, "cells", channels=_infer_channels(value))
    elif kind == "table:edges":
        write_table(value, path, "edges")
    elif kind == "table:tracks":
        write_table(value, path, "tracks")
    elif kind == "table":
        pq.write_table(value, str(path), compression="zstd")
    elif kind == "json":
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    elif kind == "array":
        with open(path, "wb") as fh:
            np.save(fh, value)
    else:
        raise PipelineError(f"cannot materialize output {name!r} of kind {kind!r}")


def _cache_store(
    cache: CacheStore,
    key: str,
    step_id: str,
    spec: OpSpec,
    params: BaseModel,
    upstream_keys: list[str],
    workspace: Path,
    out_paths: dict[str, Path],
    values: dict[str, Any],
) -> None:
    """Write a cache entry (manifest + blobs + provenance) for a computed step."""
    entry = cache.root / key
    blobs_dir = entry / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    manifest_outputs: dict[str, Any] = {}
    for name, kind in spec.outputs.items():
        rel = str(out_paths[name].relative_to(workspace))
        info: dict[str, Any] = {"kind": kind, "path": rel}
        if kind == "json":
            info["inline"] = values[name]
        elif kind == "store":
            pass  # stores are verified on disk at restore time
        else:
            blob_name = f"{name}{_DEFAULT_EXT.get(kind, '.bin')}"
            (blobs_dir / blob_name).write_bytes(out_paths[name].read_bytes())
            info["blob"] = blob_name
        manifest_outputs[name] = info
    manifest = {
        "step_id": step_id,
        "op": spec.name,
        "version": spec.version,
        "outputs": manifest_outputs,
    }
    cache.put(
        key,
        json.dumps(manifest, indent=2).encode("utf-8"),
        suffix=".json",
        provenance=make_provenance(spec.name, spec.version, params, upstream_keys),
    )


def _try_restore(
    cache: CacheStore,
    key: str,
    workspace: Path,
    spec: OpSpec,
) -> dict[str, Any] | None:
    """Restore a step's outputs from a cache entry; None on any miss."""
    try:
        manifest = json.loads(cache.get_bytes(key))
    except (KeyError, json.JSONDecodeError):
        return None
    entry = cache.root / key
    values: dict[str, Any] = {}
    for name, info in manifest.get("outputs", {}).items():
        kind = info["kind"]
        path = workspace / info["path"]
        if kind == "store":
            if not path.is_dir():
                return None
            for pattern in spec.verify:
                if not any(path.glob(pattern)):
                    logger.info(
                        "cache entry %s: store %s missing %r; treating as miss",
                        key[:12],
                        path,
                        pattern,
                    )
                    return None
            values[name] = path
        elif kind == "json":
            values[name] = info.get("inline")
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(values[name], indent=2) + "\n", encoding="utf-8"
                )
        elif kind in _TABLE_KINDS or kind == "array":
            blob = entry / "blobs" / info.get("blob", "")
            if not blob.is_file():
                return None
            data = blob.read_bytes()
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            if kind in _TABLE_KINDS:
                values[name] = pq.read_table(io.BytesIO(data))
            else:
                values[name] = np.load(io.BytesIO(data))
        else:
            return None
    return values


def _write_step_provenance(
    workspace: Path,
    step_id: str,
    spec: OpSpec,
    params: BaseModel,
    upstream_keys: list[str],
    key: str,
    status: str,
    duration_s: float,
) -> None:
    """Human-readable per-step provenance copy under ``workspace/provenance/``."""
    record = make_provenance(spec.name, spec.version, params, upstream_keys)
    record.update(
        {"step_id": step_id, "cache_key": key, "status": status, "duration_s": duration_s}
    )
    prov_dir = workspace / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / f"{step_id}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


def run(
    pipeline: Mapping[str, Any] | str | Path,
    workspace: str | Path | None = None,
    n_workers: int = 1,
    use_cache: bool = True,
) -> RunResult:
    """Execute a pipeline in topological order.

    Parameters
    ----------
    pipeline : mapping or path
        Parsed pipeline object, or a path to a pipeline JSON file.
    workspace : str or pathlib.Path, optional
        Artifact root. Overrides the pipeline's ``workspace`` key; when
        both are absent, defaults to ``workspace/`` next to the pipeline
        file (or below the cwd for in-memory pipelines).
    n_workers : int
        Worker processes for frame/fov-sharded steps (1 = serial).
    use_cache : bool
        Honor and populate the content-addressed cache
        (``<workspace>/.momia3_cache/``).

    Returns
    -------
    RunResult
        Per-step records, aggregate stats and in-memory output values.

    Raises
    ------
    PipelineError
        On validation failures or step execution errors.
    """
    base_dir = Path.cwd()
    if isinstance(pipeline, (str, Path)):
        pipeline_path = Path(pipeline)
        pipeline = load_pipeline(pipeline_path)
        base_dir = pipeline_path.resolve().parent
    errors = validate(pipeline)
    if errors:
        raise PipelineError(errors)

    if workspace is not None:
        ws = Path(workspace).resolve()
    elif pipeline.get("workspace"):
        ws = Path(str(pipeline["workspace"]))
        if not ws.is_absolute():
            ws = (base_dir / ws).resolve()
    else:
        ws = (base_dir / "workspace").resolve()
    ws.mkdir(parents=True, exist_ok=True)
    cache = CacheStore(ws / ".momia3_cache")

    steps = [s for s in pipeline["steps"] if isinstance(s, Mapping)]
    steps_by_id = {str(s["id"]): s for s in steps}
    order, _ = _topo_order(_dependencies(steps))

    values: dict[str, dict[str, Any]] = {}
    records: list[StepRecord] = []
    stats = {"computed": 0, "cached": 0}

    for sid in order:
        step = steps_by_id[sid]
        spec = REGISTRY[str(step["op"])]
        params = spec.params_model.model_validate(step.get("params", {}))
        declared_inputs = step.get("inputs", {})
        declared_outputs = step.get("outputs", {})

        input_values: dict[str, Any] = {}
        upstream_keys: list[str] = []
        for arg, ref in sorted(declared_inputs.items()):
            producer, out_name = _split_ref(ref)
            input_values[arg] = values[producer][out_name]
            producer_record = next(r for r in records if r.step_id == producer)
            upstream_keys.append(producer_record.cache_key)
        key = cache_key(spec.name, spec.version, params, upstream_keys)

        out_paths = _resolve_out_paths(ws, sid, spec, declared_outputs)
        step_values: dict[str, Any] | None = None
        status = "computed"
        duration = 0.0
        if use_cache:
            step_values = _try_restore(cache, key, ws, spec)
        if step_values is not None:
            status = "cached"
        else:
            ctx = StepContext(
                step_id=sid,
                workspace=ws,
                outputs=out_paths,
                n_workers=max(1, int(n_workers)),
                cache=cache if use_cache else None,
            )
            started = time.perf_counter()
            try:
                raw = spec.runner(input_values, params, ctx)
            except Exception as exc:
                raise PipelineError(
                    f"step {sid!r} (op {spec.name!r}) failed: {exc}"
                ) from exc
            duration = time.perf_counter() - started
            step_values = {}
            for name, kind in spec.outputs.items():
                if name not in raw:
                    raise PipelineError(
                        f"step {sid!r}: op {spec.name!r} did not return output {name!r}"
                    )
                value = _normalize_value(raw[name], kind)
                if kind == "store":
                    out_paths[name] = Path(value)  # runner chose the actual path
                _materialize(name, kind, value, out_paths[name])
                step_values[name] = value
            if use_cache:
                _cache_store(
                    cache, key, sid, spec, params, upstream_keys, ws, out_paths, step_values
                )
        stats[status] += 1
        _write_step_provenance(ws, sid, spec, params, upstream_keys, key, status, duration)
        values[sid] = step_values
        records.append(
            StepRecord(
                step_id=sid,
                op=spec.name,
                status=status,
                cache_key=key,
                duration_s=duration,
                outputs={n: str(p.relative_to(ws)) for n, p in out_paths.items()},
            )
        )
        logger.info("step %s (%s): %s", sid, spec.name, status)

    return RunResult(
        name=str(pipeline.get("name", "pipeline")),
        workspace=ws,
        steps=records,
        stats=stats,
        outputs=values,
    )
