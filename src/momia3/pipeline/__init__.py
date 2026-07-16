"""Pipeline registry, DAG engine and typer CLI (ARCHITECTURE.md §3).

- :mod:`momia3.pipeline.registry` — operation registry (§3.1)
- :mod:`momia3.pipeline.steps` — store-level chain steps + raw-op adapters
- :mod:`momia3.pipeline.io_ops` — TIFF / synthetic-data import ops
- :mod:`momia3.pipeline.engine` — JSON DAG executor (§3.2, §2.5 caching)
- :mod:`momia3.pipeline.cli` — ``momia3`` typer CLI
"""

from momia3.pipeline.engine import (
    PipelineError,
    RunResult,
    StepRecord,
    load_pipeline,
    run,
    validate,
)
from momia3.pipeline.registry import (
    REGISTRY,
    OpSpec,
    StepContext,
    describe,
    get,
    list_ops,
    register,
    to_json,
    to_mcp,
)

__all__ = [
    "REGISTRY",
    "OpSpec",
    "PipelineError",
    "RunResult",
    "StepContext",
    "StepRecord",
    "describe",
    "get",
    "list_ops",
    "load_pipeline",
    "register",
    "run",
    "to_json",
    "to_mcp",
    "validate",
]
