"""Engine validation and end-to-end execution tests (ARCHITECTURE.md §3.2, §2.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from momia3 import schema
from momia3.pipeline import engine
from momia3.storage import tables


def _demo_pipeline(workspace: Path, n_frames: int = 8, shape: tuple[int, int] = (128, 128)) -> dict:
    """Small canonical-chain pipeline (fast; still produces divisions)."""
    return {
        "name": "test_chain",
        "workspace": str(workspace),
        "steps": [
            {
                "id": "import",
                "op": "io.import_synth",
                "params": {
                    "n_frames": n_frames,
                    "shape": list(shape),
                    "n_initial": 3,
                    "pixel_microns": 0.065,
                    "growth_rate_um_per_frame": 0.2,
                    "division_length_um": 2.0,
                    "drift_px_per_frame": 0.5,
                    "noise_sigma": 0.02,
                    "seed": 3,
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
                "params": {},
                "inputs": {"experiment": "import.experiment"},
                "outputs": {"experiment": "experiment_drift.zarr", "shifts": "shifts.json"},
            },
            {
                "id": "segment",
                "op": "segment.threshold",
                "params": {"method": "composite", "min_size": 20},
                "inputs": {"experiment": "drift.experiment"},
                "outputs": {"experiment": "experiment_drift.zarr"},
            },
            {
                "id": "cells",
                "op": "measure.cells",
                "params": {"channel": "phase"},
                "inputs": {"experiment": "segment.experiment"},
                "outputs": {"cells": "tables/cells_raw.parquet"},
            },
            {
                "id": "outlines",
                "op": "outline.extract",
                "params": {},
                "inputs": {"experiment": "segment.experiment", "cells": "cells.cells"},
                "outputs": {"cells": "tables/cells_outlines.parquet"},
            },
            {
                "id": "midlines",
                "op": "midline.extract",
                "params": {},
                "inputs": {"experiment": "segment.experiment", "cells": "outlines.cells"},
                "outputs": {"cells": "tables/cells.parquet"},
            },
            {
                "id": "track",
                "op": "track.link",
                "params": {},
                "inputs": {"experiment": "segment.experiment", "cells": "midlines.cells"},
                "outputs": {
                    "edges": "tables/edges.parquet",
                    "tracks": "tables/tracks.parquet",
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_validate_ok(tmp_path: Path) -> None:
    errors = engine.validate(_demo_pipeline(tmp_path / "ws"))
    assert errors == []


def test_validate_bad_op(tmp_path: Path) -> None:
    pipeline = _demo_pipeline(tmp_path / "ws")
    pipeline["steps"][0]["op"] = "does.not_exist"
    errors = engine.validate(pipeline)
    assert any("unknown op" in e for e in errors)


def test_validate_bad_params(tmp_path: Path) -> None:
    pipeline = _demo_pipeline(tmp_path / "ws")
    pipeline["steps"][2]["params"] = {"method": "not_a_method"}
    errors = engine.validate(pipeline)
    assert any("params failed validation" in e for e in errors)


def test_validate_bad_input_ref(tmp_path: Path) -> None:
    pipeline = _demo_pipeline(tmp_path / "ws")
    pipeline["steps"][1]["inputs"] = {"experiment": "ghost.experiment"}
    errors = engine.validate(pipeline)
    assert any("unknown step" in e for e in errors)


def test_validate_unknown_output_ref(tmp_path: Path) -> None:
    pipeline = _demo_pipeline(tmp_path / "ws")
    pipeline["steps"][1]["inputs"] = {"experiment": "import.nope"}
    errors = engine.validate(pipeline)
    assert any("has no output" in e for e in errors)


def test_validate_cycle(tmp_path: Path) -> None:
    pipeline = _demo_pipeline(tmp_path / "ws")
    # drift <- segment AND segment <- drift
    pipeline["steps"][1]["inputs"] = {"experiment": "segment.experiment"}
    errors = engine.validate(pipeline)
    assert any("cycle" in e for e in errors)


def test_validate_missing_required_input(tmp_path: Path) -> None:
    pipeline = _demo_pipeline(tmp_path / "ws")
    pipeline["steps"][3]["inputs"] = {}
    errors = engine.validate(pipeline)
    assert any("missing required input" in e for e in errors)


def test_validate_unknown_output_name(tmp_path: Path) -> None:
    pipeline = _demo_pipeline(tmp_path / "ws")
    pipeline["steps"][0]["outputs"] = {"bogus": "x.json"}
    errors = engine.validate(pipeline)
    assert any("unknown output" in e for e in errors)


def test_run_invalid_pipeline_raises(tmp_path: Path) -> None:
    pipeline = _demo_pipeline(tmp_path / "ws")
    pipeline["steps"][0]["op"] = "nope.nope"
    with pytest.raises(engine.PipelineError):
        engine.run(pipeline)


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_workers", [1, 2])
def test_end_to_end_and_cache(tmp_path: Path, n_workers: int) -> None:
    """Full canonical chain runs; second identical run is fully cache-hit."""
    workspace = tmp_path / "ws"
    pipeline = _demo_pipeline(workspace)

    result1 = engine.run(pipeline, n_workers=n_workers)
    assert result1.stats["computed"] == 7
    assert result1.stats["cached"] == 0
    assert [r.step_id for r in result1.steps] == [
        "import",
        "drift",
        "segment",
        "cells",
        "outlines",
        "midlines",
        "track",
    ]

    # Artifacts exist.
    cells_path = workspace / "tables" / "cells.parquet"
    edges_path = workspace / "tables" / "edges.parquet"
    tracks_path = workspace / "tables" / "tracks.parquet"
    for path in (cells_path, edges_path, tracks_path):
        assert path.is_file(), path
    assert (workspace / "experiment_drift.zarr").is_dir()
    assert (workspace / "ground_truth.json").is_file()
    assert (workspace / "ground_truth_labels.npy").is_file()

    # Tables validate against the canonical schemas.
    cells = tables.read_cells(cells_path)
    edges = tables.read_edges(edges_path)
    tracks = tables.read_tracks(tracks_path)
    schema.validate_cells(cells.to_arrow(), ("phase",))
    schema.validate_edges(edges.to_arrow())
    schema.validate_tracks(tracks.to_arrow())
    assert cells.height > 0
    assert edges.height > 0
    assert tracks.height >= 2

    # Midline geometry reached the cells table.
    lengths = cells["length_um"].drop_nulls()
    assert len(lengths) > 0
    assert float(lengths.max()) > 1.0  # µm-scale rods

    # The synth ground truth records at least one division; the tracker
    # should recover at least one division edge.
    n_divisions = edges.filter(edges["event"] == "division").height
    assert n_divisions >= 1

    # Per-step provenance files exist.
    for record in result1.steps:
        assert (workspace / "provenance" / f"{record.step_id}.json").is_file()
        prov = json.loads((workspace / "provenance" / f"{record.step_id}.json").read_text())
        assert prov["op"] == record.op
        assert prov["cache_key"] == record.cache_key

    # Second run: everything comes from the cache; artifacts untouched.
    mtimes_before = {p: p.stat().st_mtime_ns for p in (cells_path, edges_path, tracks_path)}
    result2 = engine.run(pipeline, n_workers=n_workers)
    assert result2.stats["computed"] == 0
    assert result2.stats["cached"] == 7
    assert all(r.status == "cached" for r in result2.steps)
    mtimes_after = {p: p.stat().st_mtime_ns for p in (cells_path, edges_path, tracks_path)}
    assert mtimes_before == mtimes_after
    # Cache keys are stable across runs.
    assert [r.cache_key for r in result1.steps] == [r.cache_key for r in result2.steps]


def test_no_cache_recomputes(tmp_path: Path) -> None:
    """use_cache=False bypasses reads and writes of the cache."""
    workspace = tmp_path / "ws"
    pipeline = _demo_pipeline(workspace, n_frames=4, shape=(96, 96))
    engine.run(pipeline)
    result = engine.run(pipeline, use_cache=False)
    assert result.stats["computed"] == 7
    assert result.stats["cached"] == 0


def test_param_change_invalidates_cache(tmp_path: Path) -> None:
    """A params change downstream yields new cache keys for it and its deps."""
    workspace = tmp_path / "ws"
    pipeline = _demo_pipeline(workspace, n_frames=4, shape=(96, 96))
    result1 = engine.run(pipeline)
    pipeline["steps"][2]["params"] = {"method": "isodata"}
    result2 = engine.run(pipeline)
    key1 = {r.step_id: r.cache_key for r in result1.steps}
    key2 = {r.step_id: r.cache_key for r in result2.steps}
    # import/drift unchanged -> cached; segment and downstream recomputed.
    assert key1["import"] == key2["import"]
    assert key1["drift"] == key2["drift"]
    assert key1["segment"] != key2["segment"]
    assert key1["track"] != key2["track"]
    status2 = {r.step_id: r.status for r in result2.steps}
    assert status2["import"] == "cached"
    assert status2["drift"] == "cached"
    assert status2["segment"] == "computed"
    assert status2["track"] == "computed"


def test_run_from_json_file(tmp_path: Path) -> None:
    """Pipelines load from JSON files; relative workspaces anchor to the file."""
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(json.dumps(_demo_pipeline(Path("ws"), n_frames=4, shape=(96, 96))))
    result = engine.run(pipeline_path)
    assert result.workspace == (tmp_path / "ws").resolve()
    assert result.stats["computed"] == 7
