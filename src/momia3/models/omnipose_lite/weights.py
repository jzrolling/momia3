"""Omnipose weight resolution (weights are downloaded separately).

The repository never bundles Omnipose checkpoints. Resolution order:

1. explicit path passed by the caller;
2. MOMIA3_OMNIPOSE_WEIGHTS environment variable;
3. ~/.cellpose/models/bact_phase_omnitorch_0 (Omnipose download cache);
4. ~/.omnipose/models/bact_phase_omnitorch_0.

Download the Omnipose bact_phase model once with the Omnipose package
(python -m omnipose) or the upstream release, then point the environment
variable at it if it lives somewhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL = "bact_phase_omnitorch_0"
ENV_VAR = "MOMIA3_OMNIPOSE_WEIGHTS"

__all__ = ["DEFAULT_MODEL", "ENV_VAR", "resolve_omnipose_weights"]


def resolve_omnipose_weights(
    path: str | Path | None = None, model: str = DEFAULT_MODEL
) -> str:
    """Return a local Omnipose checkpoint path or raise a helpful error."""
    if path is not None:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"Omnipose weights not found: {candidate}")

    candidates: list[Path] = []
    env_path = os.environ.get(ENV_VAR)
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            Path.home() / ".cellpose" / "models" / model,
            Path.home() / ".omnipose" / "models" / model,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "Omnipose weights not found. Download the bact_phase model once with "
        "the Omnipose package (python -m omnipose), or set "
        f"{ENV_VAR}=/path/to/{model}."
    )
