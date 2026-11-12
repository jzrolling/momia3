"""Operation registry (ARCHITECTURE.md §3.1).

One registry entry per operation, defined once and consumed by the CLI,
the (future) napari GUI forms and MCP tool-schema export:

- :class:`OpSpec` — immutable operation descriptor: name
  (``"segment.threshold"``), semver version (part of the cache key),
  pydantic v2 params model, input/output artifact declarations, shard hint
  (``"frame"`` / ``"fov"`` / None, §3.2) and a runner following the engine
  protocol ``run(inputs, params, ctx) -> dict[str, Any]``.
- :data:`REGISTRY` + :func:`register` — global table and registration
  decorator. Every public function of the seven ``momia3.ops`` modules is
  registered under the systematic name ``"<module>.<function>"``; composite
  store-level steps (``"segment.threshold"``, ``"measure.cells"``, …) and
  the ``io.*`` import ops live in :mod:`momia3.pipeline.steps` and
  :mod:`momia3.pipeline.io_ops` and register themselves on import.
- :class:`StepContext` — execution context handed to runners (workspace,
  resolved output paths, worker count, shard mapper).
- :func:`list_ops`, :func:`describe`, :func:`to_mcp`, :func:`to_json` —
  schema export helpers (no server, just dictionaries).

Artifact type vocabulary used in ``inputs``/``outputs`` declarations:

``store``
    OME-Zarr experiment store directory (passed as :class:`pathlib.Path`).
``table`` / ``table:cells`` / ``table:edges`` / ``table:tracks``
    Arrow table (schema-validated on write for the three canonical kinds).
``json``
    JSON-serializable scalars/dicts.
``array``
    NumPy array (materialized as ``.npy``).
``any``
    Unconstrained (validation accepts anything).

A trailing ``"?"`` on an input type marks the input as optional.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

__all__ = [
    "OpSpec",
    "REGISTRY",
    "StepContext",
    "describe",
    "get",
    "list_ops",
    "register",
    "to_json",
    "to_mcp",
]

Shard = Literal["frame", "fov"]
"""Shard hints of §3.2: ``"frame"`` maps over (fov, frame), ``"fov"`` over FOVs."""

Runner = Callable[[dict[str, Any], BaseModel, "StepContext"], dict[str, Any]]
"""Engine protocol: named resolved inputs + validated params + context -> outputs."""

_T = TypeVar("_T")


@dataclass(frozen=True)
class OpSpec:
    """Immutable descriptor of one registered operation.

    Attributes
    ----------
    name : str
        Dotted registry name, e.g. ``"segment.threshold"``.
    version : str
        Semantic version; part of the content cache key (§2.5).
    runner : callable
        Engine-protocol callable ``(inputs, params, ctx) -> outputs``.
    params_model : type of pydantic.BaseModel
        Parameter model; its JSON schema feeds CLI help, GUI forms and the
        MCP export.
    inputs : dict of str to str
        Named input declarations (arg name -> artifact type, ``"?"`` suffix
        marks optional).
    outputs : dict of str to str
        Named output declarations (output name -> artifact type).
    shard : {"frame", "fov"} or None
        Parallelization hint of §3.2.
    description : str
        Human/agent-readable summary.
    verify : tuple of str
        Glob patterns that must match at least one member of a restored
        ``store`` output for a cache hit to be honored (guards against
        stores whose derived arrays were lost).
    func : callable or None
        The underlying raw operator, for introspection.
    """

    name: str
    version: str
    runner: Runner
    params_model: type[BaseModel]
    inputs: dict[str, str]
    outputs: dict[str, str]
    shard: Shard | None = None
    description: str = ""
    verify: tuple[str, ...] = ()
    func: Callable[..., Any] | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)


REGISTRY: dict[str, OpSpec] = {}
"""Global operation table, keyed by dotted op name."""


def register(
    name: str,
    *,
    version: str = "0.1.0",
    params_model: type[BaseModel],
    inputs: Mapping[str, str],
    outputs: Mapping[str, str],
    shard: Shard | None = None,
    description: str = "",
    verify: Sequence[str] = (),
    func: Callable[..., Any] | None = None,
    provenance: Mapping[str, str] | None = None,
    runner: Runner | None = None,
) -> Callable[[Runner], Runner] | Runner:
    """Register an operation in :data:`REGISTRY`.

    Can be used as a decorator (``runner`` omitted)::

        @register("segment.threshold", params_model=ThresholdRunParams,
                  inputs={"experiment": "store"}, outputs={"experiment": "store"},
                  shard="frame", description="...")
        def run_threshold(inputs, params, ctx): ...

    or as a plain call with an explicit ``runner``.

    Raises
    ------
    KeyError
        If ``name`` is already registered.
    """

    def _build(fn: Runner) -> OpSpec:
        return OpSpec(
            name=str(name),
            version=str(version),
            runner=fn,
            params_model=params_model,
            inputs=dict(inputs),
            outputs=dict(outputs),
            shard=shard,
            description=str(description),
            verify=tuple(verify),
            func=func if func is not None else fn,
            provenance=dict(provenance or {}),
        )

    if runner is not None:
        _add(_build(runner))
        return runner

    def decorator(fn: Runner) -> Runner:
        _add(_build(fn))
        return fn

    return decorator


def _add(spec: OpSpec) -> None:
    if spec.name in REGISTRY:
        raise KeyError(f"operation {spec.name!r} already registered")
    REGISTRY[spec.name] = spec
    logger.debug("registered op %r (v%s)", spec.name, spec.version)


# ---------------------------------------------------------------------------
# execution context
# ---------------------------------------------------------------------------


@dataclass
class StepContext:
    """Execution context handed to operation runners by the engine.

    Parameters
    ----------
    step_id : str
        Id of the pipeline step being executed.
    workspace : pathlib.Path
        Artifact root of the current run.
    outputs : dict of str to pathlib.Path
        Resolved workspace path of every declared output; runners that
        create directories (zarr stores) must write to
        :meth:`output_path`.
    n_workers : int
        Worker count for :meth:`map_shards` (1 = serial).
    cache : momia3.storage.cache.CacheStore or None
        Content cache of the run (None when caching is disabled).
    """

    step_id: str
    workspace: Path
    outputs: dict[str, Path] = field(default_factory=dict)
    n_workers: int = 1
    cache: Any = None

    def output_path(self, name: str) -> Path:
        """Resolved workspace path of declared output ``name``."""
        try:
            return self.outputs[name]
        except KeyError:
            raise KeyError(
                f"step {self.step_id!r} has no declared output {name!r}"
            ) from None

    def map_shards(self, fn: Callable[[_T], Any], items: Iterable[_T]) -> list[Any]:
        """Map ``fn`` over shard items, in parallel when ``n_workers > 1``.

        Uses :class:`concurrent.futures.ProcessPoolExecutor` per §3.2 and
        falls back to a plain serial loop otherwise. ``fn`` must be a
        top-level (picklable) function and ``items`` must be picklable.
        """
        items = list(items)
        if self.n_workers > 1 and len(items) > 1:
            logger.debug(
                "step %s: mapping %d shard(s) over %d worker(s)",
                self.step_id,
                len(items),
                self.n_workers,
            )
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                return list(pool.map(fn, items))
        return [fn(item) for item in items]


# ---------------------------------------------------------------------------
# introspection / schema export
# ---------------------------------------------------------------------------


def get(name: str) -> OpSpec:
    """Look up the :class:`OpSpec` registered under ``name``.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"operation {name!r} not registered; available: {sorted(REGISTRY)}"
        ) from None


def list_ops() -> list[OpSpec]:
    """All registered operations, sorted by name."""
    return [REGISTRY[name] for name in sorted(REGISTRY)]


def describe(name: str) -> dict[str, Any]:
    """JSON-schema description of one operation.

    Returns
    -------
    dict
        ``{"name", "version", "description", "shard", "inputs", "outputs",
        "params"}`` where ``params`` is the pydantic JSON schema of the
        operation's parameter model. JSON-serializable.
    """
    spec = get(name)
    return {
        "name": spec.name,
        "version": spec.version,
        "description": spec.description,
        "shard": spec.shard,
        "inputs": dict(spec.inputs),
        "outputs": dict(spec.outputs),
        "params": spec.params_model.model_json_schema(),
        "provenance": dict(spec.provenance),
    }


def to_mcp() -> list[dict[str, Any]]:
    """Export every operation as an MCP-tool-shaped dict (no server).

    Each entry has ``name`` (dots replaced by underscores), ``description``
    and ``inputSchema`` (a JSON object schema with ``params`` — the op's own
    parameter schema — and ``inputs``, a string map of upstream references).
    The original dotted op name is preserved under ``x-momia3-op``.
    """
    tools: list[dict[str, Any]] = []
    for spec in list_ops():
        tools.append(
            {
                "name": spec.name.replace(".", "_"),
                "description": f"[{spec.name}] {spec.description}".strip(),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "params": spec.params_model.model_json_schema(),
                        "inputs": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "description": (
                                "Named inputs as '<step_id>.<output>' references; "
                                f"expected: {spec.inputs}"
                            ),
                        },
                    },
                    "required": [],
                },
                "x-momia3-op": spec.name,
            }
        )
    return tools


def to_json() -> dict[str, Any]:
    """Full registry dump: ``{"ops": {name: describe(name)}}``."""
    return {"ops": {spec.name: describe(spec.name) for spec in list_ops()}}


# ---------------------------------------------------------------------------
# built-in operations
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    """Import the built-in op modules so their ``@register`` hooks run."""
    from momia3.pipeline import features, io_ops, steps  # noqa: F401


_register_builtins()
