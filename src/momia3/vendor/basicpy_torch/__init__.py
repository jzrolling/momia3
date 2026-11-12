"""Vendored PyTorch BaSiC implementation (BaSiCpy_torch).

Derived from BaSiCPy (MIT), PyTorch adaptation maintained by the lab.
See README.md and LICENSE in this directory for provenance and terms.

Importing this package requires torch; momia3 core and
momia3.ops.preprocess do not import it at module import time.
"""

from .basicpy import BaSiC

__all__ = ["BaSiC", "datasets"]


def __getattr__(name: str):
    """Lazy-load the optional pooch-backed example datasets."""
    if name == "datasets":
        from . import datasets as _datasets
        return _datasets
    raise AttributeError(name)
