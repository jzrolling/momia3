"""CLI tests via typer.testing.CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from momia3.pipeline.cli import app
from momia3.storage import tables

runner = CliRunner()


def _tiny_pipeline(workspace: Path) -> dict:
    """Smallest end-to-end pipeline: synth -> segment -> cells."""
    return {
        "name": "cli_tiny",
        "workspace": str(workspace),
        "steps": [
            {
                "id": "import",
                "op": "io.import_synth",
                "params": {
                    "n_frames": 4,
                    "shape": [96, 96],
                    "n_initial": 2,
                    "division_length_um": 2.0,
                    "growth_rate_um_per_frame": 0.2,
                    "drift_px_per_frame": 0.3,
                    "seed": 1,
                },
                "inputs": {},
                "outputs": {"experiment": "experiment.zarr"},
            },
            {
                "id": "segment",
                "op": "segment.threshold",
                "params": {"method": "composite", "min_size": 20},
                "inputs": {"experiment": "import.experiment"},
                "outputs": {"experiment": "experiment.zarr"},
            },
            {
                "id": "cells",
                "op": "measure.cells",
                "params": {"channel": "phase"},
                "inputs": {"experiment": "segment.experiment"},
                "outputs": {"cells": "cells.parquet"},
            },
        ],
    }


def test_list_ops() -> None:
    result = runner.invoke(app, ["list-ops"])
    assert result.exit_code == 0, result.output
    assert "segment.threshold" in result.output
    assert "io.import_synth" in result.output
    assert "track.link" in result.output


def test_describe_op() -> None:
    result = runner.invoke(app, ["describe", "segment.threshold"])
    assert result.exit_code == 0, result.output
    description = json.loads(result.output)
    assert description["name"] == "segment.threshold"
    assert description["params"]["type"] == "object"
    assert "method" in description["params"]["properties"]


def test_describe_unknown_op_fails() -> None:
    result = runner.invoke(app, ["describe", "does.not_exist"])
    assert result.exit_code == 1


def test_validate_valid_pipeline(tmp_path: Path) -> None:
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(json.dumps(_tiny_pipeline(tmp_path / "ws")))
    result = runner.invoke(app, ["validate", str(pipeline_path)])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output


def test_validate_invalid_pipeline(tmp_path: Path) -> None:
    pipeline = _tiny_pipeline(tmp_path / "ws")
    pipeline["steps"][0]["op"] = "not.an_op"
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(json.dumps(pipeline))
    result = runner.invoke(app, ["validate", str(pipeline_path)])
    assert result.exit_code == 1
    assert "unknown op" in result.output


def test_run_tiny_pipeline(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(json.dumps(_tiny_pipeline(workspace)))
    result = runner.invoke(app, ["run", str(pipeline_path), "--workers", "1"])
    assert result.exit_code == 0, result.output
    assert "cli_tiny" in result.output
    cells_path = workspace / "cells.parquet"
    assert cells_path.is_file()
    cells = tables.read_cells(cells_path)
    assert cells.height > 0

    # Second run via CLI: fully cache-hit.
    result2 = runner.invoke(app, ["run", str(pipeline_path)])
    assert result2.exit_code == 0, result2.output
    assert "0 computed, 3 cached" in result2.output

    # --no-cache recomputes.
    result3 = runner.invoke(app, ["run", str(pipeline_path), "--no-cache"])
    assert result3.exit_code == 0, result3.output
    assert "3 computed, 0 cached" in result3.output


def test_query_without_duckdb_errors_cleanly(tmp_path: Path) -> None:
    """`query` must exit with a clear message when duckdb is absent (and work when present)."""
    workspace = tmp_path / "ws"
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(json.dumps(_tiny_pipeline(workspace)))
    run_result = runner.invoke(app, ["run", str(pipeline_path)])
    assert run_result.exit_code == 0, run_result.output
    cells_path = workspace / "cells.parquet"

    duckdb = pytest.importorskip  # noqa: F841 - documented below
    try:
        import duckdb  # noqa: F401

        has_duckdb = True
    except ImportError:
        has_duckdb = False

    result = runner.invoke(app, ["query", str(cells_path), "SELECT count(*) AS n FROM t"])
    if has_duckdb:
        assert result.exit_code == 0, result.output
        assert "n" in result.output
    else:
        assert result.exit_code == 2
        assert "duckdb is not installed" in result.output
