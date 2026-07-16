"""Load the legacy momia2 reference modules without installing its heavy deps.

The momia2 repository at ``/Users/jzrolling/Desktop/Projects/Playground_kimi/momia2``
is a read-only reference. Its package ``__init__`` chain pulls heavy optional
dependencies (``pandas``/``matplotlib`` via ``utils.fish`` and ``core.tracker``,
``cv2`` via ``utils.metrics``, ``yaml`` via ``utils.config``, ``read_roi`` via
``segment._helper``, optional ``tensorflow`` via ``segment.mask``, and PyQt5 via
``classify``). Only three functions are needed as benchmark references:

- ``utils/corrections.py::rolling_ball_bg`` (numba rolling-ball kernel)
- ``utils/midline.py::extract_midline`` (legacy midline extraction)
- ``core/tracker.py::_calculate_overlap_matrix`` (legacy overlap matrix)

Strategy (per benchmark plan): load the real source files with
:func:`importlib.util.spec_from_file_location` under a fake package namespace
(``momia2ref``) so their relative imports resolve, and pre-seed
:data:`sys.modules` with stubs for the heavy relative-import parents
(``momia2ref.utils`` / ``momia2ref.segment`` for ``core/tracker.py``). The
light ``utils`` chain (``linalg`` -> ``generic`` -> ``contour`` ->
``midline`` / ``corrections``) only needs numpy/scipy/skimage/numba and is
loaded from the real files.

Two guards keep the read-only momia2 checkout pristine:

- ``sys.dont_write_bytecode`` is set while loading (no ``__pycache__``
  writes), and
- ``NUMBA_CACHE_DIR`` is redirected to ``benchmarks/.numba_cache`` (the
  momia2 kernels are decorated with ``@jit(cache=True)``).

If this loader fails on a future momia2 revision, the fallback (extracting
the referenced function sources into ``benchmarks/_momia2_ref.py`` with
file:line citations) is documented in ``benchmarks/benchmark_report.md``;
it is currently NOT needed — all three functions load from the real files.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

MOMIA2_ROOT = Path(__file__).resolve().parents[2] / "momia2"

_FAKE_ROOT = "momia2ref"
_CACHE_DIR = Path(__file__).resolve().parent / ".numba_cache"


class Momia2Reference:
    """Namespace of loaded momia2 modules."""

    def __init__(self) -> None:
        self.linalg: types.ModuleType | None = None
        self.generic: types.ModuleType | None = None
        self.contour: types.ModuleType | None = None
        self.midline: types.ModuleType | None = None
        self.corrections: types.ModuleType | None = None
        self.tracker: types.ModuleType | None = None

    # convenience accessors -------------------------------------------------
    def rolling_ball_bg(self):
        return self.corrections.rolling_ball_bg  # type: ignore[union-attr]

    def rolling_ball_cls(self):
        return self.corrections.rolling_ball  # type: ignore[union-attr]

    def extract_midline(self):
        return self.midline.extract_midline  # type: ignore[union-attr]

    def overlap_matrix(self):
        return self.tracker._calculate_overlap_matrix  # type: ignore[union-attr]


def _ensure_package(name: str, path: Path) -> types.ModuleType:
    """Register a bare package module (``__path__`` only) in sys.modules."""
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]  # type: ignore[attr-defined]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


def _load_module(name: str, path: Path) -> types.ModuleType:
    """Execute one momia2 source file under the fake package namespace."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_momia2(root: Path | None = None) -> Momia2Reference:
    """Load the momia2 reference modules; see module docstring for the how."""
    root = Path(root) if root is not None else MOMIA2_ROOT
    if not (root / "utils" / "corrections.py").is_file():
        raise FileNotFoundError(f"momia2 checkout not found at {root}")

    # Guard the read-only checkout: no bytecode writes, numba cache redirected.
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(_CACHE_DIR)
    old_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        _ensure_package(_FAKE_ROOT, root)
        _ensure_package(f"{_FAKE_ROOT}.utils", root / "utils")
        _ensure_package(f"{_FAKE_ROOT}.core", root / "core")

        ref = Momia2Reference()
        # Real light chain: linalg -> generic -> contour -> midline/corrections.
        ref.linalg = _load_module(f"{_FAKE_ROOT}.utils.linalg", root / "utils" / "linalg.py")
        ref.generic = _load_module(f"{_FAKE_ROOT}.utils.generic", root / "utils" / "generic.py")
        ref.contour = _load_module(f"{_FAKE_ROOT}.utils.contour", root / "utils" / "contour.py")
        ref.midline = _load_module(f"{_FAKE_ROOT}.utils.midline", root / "utils" / "midline.py")
        ref.corrections = _load_module(
            f"{_FAKE_ROOT}.utils.corrections", root / "utils" / "corrections.py"
        )

        # core/tracker.py needs `from ..segment import prediction2foreground,
        # prediction2seed` and `from ..utils import *` at module level. Both
        # parents are only used by CellTracker methods (not by
        # _calculate_overlap_matrix), so bare stubs are enough and avoid the
        # segment -> read_roi/tensorflow chain entirely.
        utils_stub = types.ModuleType(f"{_FAKE_ROOT}.utils")
        utils_stub.__path__ = [str(root / "utils")]  # type: ignore[attr-defined]
        sys.modules[f"{_FAKE_ROOT}.utils"] = utils_stub
        segment_stub = types.ModuleType(f"{_FAKE_ROOT}.segment")
        segment_stub.__path__ = [str(root / "segment")]  # type: ignore[attr-defined]
        segment_stub.prediction2foreground = lambda *a, **k: None  # type: ignore[attr-defined]
        segment_stub.prediction2seed = lambda *a, **k: None  # type: ignore[attr-defined]
        sys.modules[f"{_FAKE_ROOT}.segment"] = segment_stub
        ref.tracker = _load_module(f"{_FAKE_ROOT}.core.tracker", root / "core" / "tracker.py")
        return ref
    finally:
        sys.dont_write_bytecode = old_dont_write
