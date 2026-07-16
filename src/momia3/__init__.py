"""momia3 — single-bacterium microscopy characterization and lineage tracking.

Phase 2 delivers the data layer and scaffolding:

- :mod:`momia3.keys` — uint64 composite primary keys (ARCHITECTURE.md §2.1)
- :mod:`momia3.schema` — Arrow schemas for cells/edges/tracks tables (§2.3–§2.4)
- :mod:`momia3.storage` — OME-Zarr store, parquet table IO, content cache (§2.2/§2.5)
- :mod:`momia3.backend` — numpy/torch dual backend dispatch (§3.3)
- :mod:`momia3.core` — Field / CellSet / lineage views
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
