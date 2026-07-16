"""Plain-Python equivalent of ``examples/pipeline_demo.json`` (for debugging).

Runs the canonical chain — synth import -> drift correction -> threshold
segmentation -> cells -> outlines -> midlines -> tracking — through the
pipeline engine API directly, so breakpoints can be placed in the runners
without the CLI in the way.

Usage::

    .venv/bin/python examples/run_demo.py [--workers N] [--no-cache]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from momia3.pipeline import engine

EXAMPLES_DIR = Path(__file__).resolve().parent

PIPELINE: dict = {
    "name": "momia3_demo_py",
    "workspace": str(EXAMPLES_DIR / "demo_workspace"),
    "steps": [
        {
            "id": "import",
            "op": "io.import_synth",
            "params": {
                "n_frames": 12,
                "shape": [256, 256],
                "n_initial": 3,
                "pixel_microns": 0.065,
                "growth_rate_um_per_frame": 0.15,
                "division_length_um": 2.2,
                "drift_px_per_frame": 0.6,
                "noise_sigma": 0.02,
                "seed": 7,
            },
            "inputs": {},
            "outputs": {
                "experiment": "experiment.zarr",
                "truth": "ground_truth.json",
                "truth_labels": "ground_truth_labels.npy",
            },
        },
        {
            "id": "drift",
            "op": "preprocess.drift",
            "params": {"upsample_factor": 10},
            "inputs": {"experiment": "import.experiment"},
            "outputs": {
                "experiment": "experiment_drift.zarr",
                "shifts": "drift_shifts.json",
            },
        },
        {
            "id": "segment",
            "op": "segment.threshold",
            "params": {"method": "composite", "dark_foreground": True, "min_size": 20},
            "inputs": {"experiment": "drift.experiment"},
            "outputs": {"experiment": "experiment_drift.zarr"},
        },
        {
            "id": "cells",
            "op": "measure.cells",
            "params": {"channel": "phase", "include_intensity": True},
            "inputs": {"experiment": "segment.experiment"},
            "outputs": {"cells": "tables/cells_raw.parquet"},
        },
        {
            "id": "outlines",
            "op": "outline.extract",
            "params": {"n_points": 100, "sigma": 1.0},
            "inputs": {"experiment": "segment.experiment", "cells": "cells.cells"},
            "outputs": {"cells": "tables/cells_outlines.parquet"},
        },
        {
            "id": "midlines",
            "op": "midline.extract",
            "params": {"n_points": 32, "method": "zhang"},
            "inputs": {"experiment": "segment.experiment", "cells": "outlines.cells"},
            "outputs": {"cells": "tables/cells.parquet"},
        },
        {
            "id": "track",
            "op": "track.link",
            "params": {"gap_close": 2, "gate_distance": 20.0},
            "inputs": {"experiment": "segment.experiment", "cells": "midlines.cells"},
            "outputs": {
                "edges": "tables/edges.parquet",
                "tracks": "tables/tracks.parquet",
            },
        },
    ],
}


def main() -> None:
    """Parse args and run the demo pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1, help="worker processes")
    parser.add_argument("--no-cache", action="store_true", help="bypass the cache")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    result = engine.run(PIPELINE, n_workers=args.workers, use_cache=not args.no_cache)
    print(f"\npipeline {result.name!r} -> {result.workspace}")
    for record in result.steps:
        print(
            f"  {record.step_id:12s} {record.op:20s} {record.status:8s} "
            f"{record.duration_s:5.2f}s"
        )
    print(f"stats: {result.stats}")
    cells = result.outputs["midlines"]["cells"]
    edges = result.outputs["track"]["edges"]
    tracks = result.outputs["track"]["tracks"]
    print(
        f"cells: {cells.num_rows} rows; edges: {edges.num_rows} rows; "
        f"tracks: {tracks.num_rows} rows"
    )


if __name__ == "__main__":
    main()
