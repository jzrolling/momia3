"""Content-addressed operation cache with provenance (ARCHITECTURE.md §2.5).

Cache keys are ``sha256(op_name + version + params_json + upstream_keys)``:
any parameter change yields a new key, which automatically invalidates
downstream artifacts; re-runs with identical inputs hit the cache at zero
compute cost.

Each artifact lives in ``.momia3_cache/<key>/`` together with a
``provenance.json`` recording ``{op, version, params, upstream, timestamp,
git_hash}``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_KEY_RE = re.compile(r"[0-9a-f]{64}")

__all__ = ["cache_key", "make_provenance", "CacheStore"]


def cache_key(
    op_name: str,
    version: str,
    params: BaseModel,
    upstream: Sequence[str],
) -> str:
    """Compute the content-addressed key of an operation invocation.

    Parameters
    ----------
    op_name : str
        Registered operation name, e.g. ``"segment.threshold"``.
    version : str
        Operation semver; part of the key so version bumps invalidate.
    params : pydantic.BaseModel
        Operation parameter model; serialized with
        :meth:`BaseModel.model_dump` in JSON mode.
    upstream : sequence of str
        Cache keys of upstream artifacts this invocation depends on.
        Order-insensitive (sorted before hashing).

    Returns
    -------
    str
        64-character lowercase hex digest.
    """
    if not isinstance(params, BaseModel):
        raise TypeError(
            f"params must be a pydantic BaseModel, got {type(params).__name__}"
        )
    payload = {
        "op": str(op_name),
        "version": str(version),
        "params": params.model_dump(mode="json"),
        "upstream": sorted(str(k) for k in upstream),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _git_hash() -> str | None:
    """Best-effort ``git rev-parse HEAD`` of the current working directory."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def make_provenance(
    op_name: str,
    version: str,
    params: BaseModel | Mapping[str, Any],
    upstream: Sequence[str],
) -> dict[str, Any]:
    """Build the provenance record stored alongside each artifact.

    Parameters
    ----------
    op_name : str
        Registered operation name.
    version : str
        Operation semver.
    params : pydantic.BaseModel or mapping
        Operation parameters.
    upstream : sequence of str
        Cache keys of upstream artifacts.

    Returns
    -------
    dict
        ``{op, version, params, upstream, timestamp, git_hash}`` —
        JSON-serializable.
    """
    if isinstance(params, BaseModel):
        params_dict: dict[str, Any] = params.model_dump(mode="json")
    else:
        params_dict = dict(params)
    return {
        "op": str(op_name),
        "version": str(version),
        "params": params_dict,
        "upstream": [str(k) for k in upstream],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
    }


class CacheStore:
    """Filesystem artifact store rooted at ``.momia3_cache/``.

    Parameters
    ----------
    root : str or pathlib.Path
        Cache directory (created on first write).
    """

    def __init__(self, root: str | Path = ".momia3_cache") -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    @staticmethod
    def _check_key(key: str) -> str:
        if not _KEY_RE.fullmatch(key):
            raise ValueError(f"malformed cache key (expected sha256 hex): {key!r}")
        return key

    def _entry_dir(self, key: str) -> Path:
        return self.root / self._check_key(key)

    def _artifact_path(self, key: str) -> Path:
        entry = self._entry_dir(key)
        if not entry.is_dir():
            raise KeyError(f"cache miss: {key}")
        artifacts = sorted(p for p in entry.glob("artifact*") if p.is_file())
        if not artifacts:
            raise KeyError(f"cache entry {key} has no artifact")
        return artifacts[0]

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def has(self, key: str) -> bool:
        """Return True if an artifact exists for ``key``."""
        try:
            self._artifact_path(key)
        except KeyError:
            return False
        return True

    def put(
        self,
        key: str,
        data: bytes,
        *,
        suffix: str = "",
        provenance: Mapping[str, Any] | None = None,
    ) -> Path:
        """Store an artifact (plus provenance) under ``key``.

        Parameters
        ----------
        key : str
            Cache key from :func:`cache_key`.
        data : bytes
            Serialized artifact payload (e.g. parquet or npy bytes).
        suffix : str
            Optional filename suffix such as ``".parquet"`` (no slashes).
        provenance : mapping, optional
            Provenance record; see :func:`make_provenance`. Written to
            ``provenance.json`` next to the artifact when given.

        Returns
        -------
        pathlib.Path
            Path of the written artifact file.
        """
        if "/" in suffix or "\\" in suffix:
            raise ValueError(f"suffix must not contain path separators: {suffix!r}")
        entry = self._entry_dir(key)
        entry.mkdir(parents=True, exist_ok=True)
        artifact = entry / f"artifact{suffix}"
        if artifact.exists():
            logger.debug("overwriting existing artifact for %s", key)
        artifact.write_bytes(data)
        if provenance is not None:
            (entry / "provenance.json").write_text(
                json.dumps(dict(provenance), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        logger.info("cached artifact %s (%d bytes) at %s", key, len(data), artifact)
        return artifact

    def get(self, key: str) -> Path:
        """Return the artifact path for ``key``.

        Raises
        ------
        KeyError
            On cache miss.
        """
        return self._artifact_path(key)

    def get_bytes(self, key: str) -> bytes:
        """Return the artifact payload for ``key``.

        Raises
        ------
        KeyError
            On cache miss.
        """
        return self._artifact_path(key).read_bytes()

    def read_provenance(self, key: str) -> dict[str, Any]:
        """Read the ``provenance.json`` of an entry.

        Raises
        ------
        KeyError
            If the entry or its provenance file does not exist.
        """
        prov = self._entry_dir(key) / "provenance.json"
        if not prov.is_file():
            raise KeyError(f"cache entry {key} has no provenance.json")
        return json.loads(prov.read_text(encoding="utf-8"))
