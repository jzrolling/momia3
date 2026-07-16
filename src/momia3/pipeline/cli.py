"""``momia3`` command line interface (ARCHITECTURE.md §3.1).

Commands
--------
``momia3 list-ops``
    List every registered operation (name, version, shard, description).
``momia3 describe OP``
    Print the JSON schema of one operation.
``momia3 validate PIPELINE.json``
    Static pipeline validation (dry-run; exit code 1 on errors).
``momia3 run PIPELINE.json [--workers N] [--no-cache] [--workspace DIR]``
    Execute a pipeline.
``momia3 query PARQUET "SQL"``
    Run a DuckDB SQL query against a parquet artifact (the file is exposed
    as table ``t``). DuckDB is an optional dependency; a clean error is
    printed when it is missing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from momia3.pipeline import engine as _engine
from momia3.pipeline import registry as _registry

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="momia3",
    help="momia3 — bacterial timelapse microscopy analysis pipeline.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("list-ops")
def list_ops_cmd() -> None:
    """List all registered operations."""
    for spec in _registry.list_ops():
        shard = spec.shard or "-"
        typer.echo(f"{spec.name:40s} v{spec.version:8s} shard={shard:5s} {spec.description}")
    typer.echo(f"\n{len(_registry.REGISTRY)} operation(s) registered.")


@app.command("describe")
def describe_cmd(op: str = typer.Argument(..., help="Registered operation name.")) -> None:
    """Print the JSON schema of one operation."""
    try:
        description = _registry.describe(op)
    except KeyError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    typer.echo(json.dumps(description, indent=2))


@app.command("validate")
def validate_cmd(
    pipeline: Path = typer.Argument(..., help="Pipeline JSON file.", exists=True),
) -> None:
    """Statically validate a pipeline (no execution)."""
    try:
        data = _engine.load_pipeline(pipeline)
    except _engine.PipelineError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    errors = _engine.validate(data)
    if errors:
        typer.secho(f"{pipeline}: {len(errors)} validation error(s):", fg=typer.colors.RED, err=True)
        for error in errors:
            typer.secho(f"  - {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(
        f"{pipeline}: valid ({len(data['steps'])} step(s)).", fg=typer.colors.GREEN
    )


@app.command("run")
def run_cmd(
    pipeline: Path = typer.Argument(..., help="Pipeline JSON file.", exists=True),
    workers: int = typer.Option(1, "--workers", "-n", min=1, help="Worker processes for sharded steps."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the content cache entirely."),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Artifact root (overrides the pipeline's workspace key)."
    ),
) -> None:
    """Execute a pipeline."""
    try:
        result = _engine.run(
            pipeline, workspace=workspace, n_workers=workers, use_cache=not no_cache
        )
    except _engine.PipelineError as exc:
        typer.secho("pipeline failed:", fg=typer.colors.RED, err=True)
        for error in exc.errors:
            typer.secho(f"  - {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    typer.secho(f"pipeline {result.name!r} finished -> {result.workspace}", fg=typer.colors.GREEN)
    for record in result.steps:
        typer.echo(
            f"  {record.step_id:16s} {record.op:24s} {record.status:8s} "
            f"{record.duration_s:6.2f}s  key={record.cache_key[:12]}"
        )
        for out_name, rel in record.outputs.items():
            typer.echo(f"      {out_name}: {rel}")
    typer.echo(
        f"stats: {result.stats['computed']} computed, {result.stats['cached']} cached"
    )


@app.command("query")
def query_cmd(
    parquet: Path = typer.Argument(..., help="Parquet artifact to query.", exists=True),
    sql: str = typer.Argument(..., help="SQL query; the parquet file is exposed as table 't'."),
) -> None:
    """Run a DuckDB SQL query against a parquet artifact."""
    try:
        import duckdb
    except ImportError:
        typer.secho(
            "duckdb is not installed — `momia3 query` requires the optional "
            "duckdb package (pip install duckdb).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from None
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            "CREATE VIEW t AS SELECT * FROM read_parquet(?)", [str(parquet)]
        )
        arrow = connection.execute(sql).arrow()
    except Exception as exc:
        typer.secho(f"query failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    import polars as pl

    frame = pl.from_arrow(arrow)
    typer.echo(frame)


def main() -> None:
    """Console-script entry point."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    app()


if __name__ == "__main__":
    main()
