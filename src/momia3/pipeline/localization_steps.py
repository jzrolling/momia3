"""Registry adapters for the localization / distribution operators.

Each operator is a thin JSON-boundary wrapper around a pure function in
momia3.ops.localization.  Provenance records the algorithm source and the
date the reference was consulted; see the registry describe() output.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from momia3.ops import localization as loc_ops
from momia3.pipeline.registry import StepContext, register




def _jsonable(obj: Any) -> Any:
    """Recursively convert NumPy containers/scalars for the JSON artifact boundary."""
    from collections.abc import Mapping, Sequence
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return [_jsonable(v) for v in obj]
    return obj

_RETRIEVED = "2026-10-23T23:59:00+08:00"

_ROD_SOURCES = (
    "MicrobeJ (Ducret, Quardokus & Brun, Nat Microbiol 2016); "
    "Oufti (Paintdakhi et al., Mol Microbiol 2016); "
    "BacStalk (Hartmann et al., Mol Microbiol 2020); "
    "ColiCoords (Smit et al. 2019); "
    "BactMAP (van Raaphorst et al. 2020); "
    "momia2 demograph"
)
_RADIAL_SOURCES = (
    "CellProfiler MeasureObjectIntensityDistribution (Stirling et al., "
    "BMC Bioinformatics 2021); Cell Painting compartment features "
    "(Bray et al., Nat Protoc 2016)"
)
_POINT_SOURCES = (
    "Ripley 1976; SR-Tesseler (Levet et al., Nat Methods 2015); "
    "DBSCAN (Ester et al., KDD 1996)"
)
_KERNEL_SOURCES = (
    "Kernel mean embedding (Muandet et al. 2017); MMD (Gretton et al. "
    "2012); optimal transport / sliced Wasserstein (Peyre & Cuturi 2019)"
)
_FOURIER_SOURCES = (
    "Elliptic Fourier descriptors / contour Fourier analysis; "
    "MicrobeJ membrane profiles; periodic boundary signals"
)


def _prov(algorithm: str, sources: str, notes: str = "") -> dict[str, str]:
    return {
        "algorithm": algorithm,
        "sources": sources,
        "retrieved_at": _RETRIEVED,
        "implementation": "momia3 pure Python / NumPy / SciPy",
        "notes": notes,
    }


class RodCoordinatesParams(BaseModel):
    n_points: int = Field(32, ge=4, description="Midline samples used to build coordinates")
    pixel_microns: float = Field(1.0, gt=0, description="Pixel size used to convert width_profile um -> px")


class RodProfileParams(RodCoordinatesParams):
    n_bins: int = Field(64, ge=2, description="Long-axis bins")
    mode: Literal["mean", "max"] = Field("mean", description="Within-bin aggregation")


@register(
    "localization.rod_coordinates",
    params_model=RodCoordinatesParams,
    inputs={"cells": "table", "labels": "array"},
    outputs={"coordinates": "json"},
    shard="frame",
    description="Rod-cell intrinsic coordinates (l, r, pole distances) from actual midline/outline.",
    func=loc_ops.rod_coordinates,
    provenance=_prov("rod-cell intrinsic coordinates", _ROD_SOURCES),
)
def _run_rod_coordinates(inputs: dict[str, Any], params: RodCoordinatesParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"coordinates": _jsonable(loc_ops.rod_coordinates(inputs["cells"], inputs["labels"], params.n_points, params.pixel_microns))}


@register(
    "localization.rod_profile",
    params_model=RodProfileParams,
    inputs={"cells": "table", "image": "array", "labels": "array"},
    outputs={"profiles": "json"},
    shard="frame",
    description="1-D intensity profile along the curved rod long axis.",
    func=loc_ops.rod_profile,
    provenance=_prov("curved-rod longitudinal intensity profile", _ROD_SOURCES),
)
def _run_rod_profile(inputs: dict[str, Any], params: RodProfileParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"profiles": _jsonable(loc_ops.rod_profile(inputs["cells"], inputs["image"], inputs["labels"], params.n_bins, params.mode, params.pixel_microns))}


class RadialParams(BaseModel):
    n_bins: int = Field(8, ge=1, description="Radial bins from cell centroid")


@register(
    "localization.radial_intensity_distribution",
    params_model=RadialParams,
    inputs={"labels": "array", "image": "array"},
    outputs={"distribution": "json"},
    shard="frame",
    description="Per-cell radial intensity distribution normalized by cell radius.",
    func=loc_ops.radial_intensity_distribution,
    provenance=_prov("radial intensity distribution", _RADIAL_SOURCES),
)
def _run_radial(inputs: dict[str, Any], params: RadialParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"distribution": _jsonable(loc_ops.radial_intensity_distribution(inputs["labels"], inputs["image"], params.n_bins))}


class CompartmentParams(BaseModel):
    name: str = Field("compartment", description="Output prefix")


@register(
    "localization.compartment_stats",
    params_model=CompartmentParams,
    inputs={"labels": "array", "image": "array", "compartment": "array"},
    outputs={"stats": "json"},
    shard="frame",
    description="Per-cell intensity/area statistics inside a compartment mask.",
    func=loc_ops.compartment_stats,
    provenance=_prov("compartment intensity statistics", _RADIAL_SOURCES),
)
def _run_compartment(inputs: dict[str, Any], params: CompartmentParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"stats": _jsonable(loc_ops.compartment_stats(inputs["labels"], inputs["image"], inputs["compartment"], params.name))}


