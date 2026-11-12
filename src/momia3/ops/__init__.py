"""Pure-function operators (ARCHITECTURE.md §4) with pydantic v2 parameter models.

This package re-exports every public operator and parameter model from the
seven operator modules so that both spellings work::

    from momia3.ops.segment import threshold_segment
    from momia3.ops import threshold_segment, ThresholdParams

The pipeline registry (:mod:`momia3.pipeline.registry`) registers one entry
per public function of these modules under the systematic name
``"<module>.<function>"`` (e.g. ``"segment.threshold_segment"``), plus a
small set of store-level composite steps (``"segment.threshold"`` …) used by
the canonical pipeline chain.
"""

from momia3.ops import (
    characterize,
    features,
    localization,
    measure,
    midline,
    outline,
    polygon,
    preprocess,
    segment,
    track,
)
from momia3.ops.characterize import *
from momia3.ops.features import *
from momia3.ops.localization import *
from momia3.ops.measure import *
from momia3.ops.midline import *
from momia3.ops.outline import *
from momia3.ops.polygon import *
from momia3.ops.preprocess import *
from momia3.ops.segment import *
from momia3.ops.track import *

__all__ = sorted(
    {
        name
        for module in (preprocess, segment, measure, outline, midline, characterize, features, localization, track, polygon)
        for name in module.__all__
    }
)
