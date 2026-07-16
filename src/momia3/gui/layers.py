"""Qt-free builders turning momia3 tables into napari layer data (§8).

Everything here is pure Python/NumPy on plain records — no Qt, no napari
imports — so the layer geometry is unit-testable headless and reusable
outside the GUI.

Coordinate conventions
----------------------
momia3 geometry columns (``outline``, ``midline``, ``centroid``) store
``(x, y)`` pairs in global image coordinates (§2.3); napari layers want
``(t, y, x)`` per vertex. All builders below emit ``(t, y, x)``.

Track identity
--------------
``cells.parquet`` written by ``track.link`` does not backfill
``track_id``; :func:`cells_edges_to_tracks` resolves identities in three
stages, most authoritative first:

1. non-null ``track_id`` column in the cells table (used directly,
   namespaced by ``fov_id``);
2. connected components of ``continue`` edges (union-find), renamed to
   canonical track ids by matching ``tracks.parquet`` on
   ``(fov_id, birth_frame, last_frame, n_frames)``;
3. leftover cells become singleton tracks.

The lineage graph (napari Tracks ``graph`` argument) comes from
``tracks.parquet`` ``parent_track`` when available, else from
``division`` edges.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_SHAPES_CELLS",
    "cells_edges_to_tracks",
    "cells_to_midline_shapes",
    "cells_to_outline_shapes",
    "to_records",
]

MAX_SHAPES_CELLS: int = 5000
"""Default per-layer cell cap; beyond it cells are randomly subsampled."""

_DEFAULT_VALUE_COLUMNS: tuple[str, ...] = (
    "length_um",
    "width_median_um",
    "area",
    "aspect_ratio",
)
"""Cell scalar columns offered as Tracks-layer vertex properties by default."""


# ---------------------------------------------------------------------------
# record conversion
# ---------------------------------------------------------------------------


def to_records(table: Any) -> list[dict[str, Any]]:
    """Normalize a table-like object to a list of plain dicts.

    Accepts polars DataFrames, pyarrow Tables, or iterables of mappings;
    ``None`` yields an empty list.
    """
    if table is None:
        return []
    if hasattr(table, "to_dicts"):  # polars.DataFrame
        return table.to_dicts()
    if hasattr(table, "to_pylist"):  # pyarrow.Table
        return table.to_pylist()
    return [dict(row) for row in table]


def _subsample(
    rows: list[dict[str, Any]], max_cells: int | None, seed: int
) -> tuple[list[dict[str, Any]], bool]:
    """Randomly subsample ``rows`` down to ``max_cells`` (stable order)."""
    if max_cells and len(rows) > max_cells:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(rows), size=int(max_cells), replace=False))
        logger.info("subsampling %d -> %d cells for display", len(rows), max_cells)
        return [rows[int(i)] for i in idx], True
    return rows, False


def _fyx(row: Mapping[str, Any]) -> tuple[int, float, float]:
    """``(frame, y, x)`` of one cell record (NaN coordinates when unknown)."""
    centroid = row.get("centroid") or {}
    return (
        int(row.get("frame", 0)),
        float(centroid.get("y", np.nan)),
        float(centroid.get("x", np.nan)),
    )


def _shape_vertices(ring: Mapping[str, Any], frame: int) -> np.ndarray:
    """``(N, 3)`` vertices ``(t, y, x)`` of one ``{"x": [...], "y": [...]}`` ring."""
    xs = np.asarray(ring["x"], dtype=np.float64)
    ys = np.asarray(ring["y"], dtype=np.float64)
    t = np.full(xs.shape, float(frame), dtype=np.float64)
    return np.column_stack([t, ys, xs])


# ---------------------------------------------------------------------------
# Shapes layers (outlines / midlines)
# ---------------------------------------------------------------------------


def cells_to_outline_shapes(
    cells: Any,
    max_cells: int | None = MAX_SHAPES_CELLS,
    seed: int = 0,
) -> tuple[list[np.ndarray], dict[str, list[int]], bool]:
    """Build one closed polygon per cell for a napari Shapes layer.

    Parameters
    ----------
    cells : table-like
        Cells table with ``cell_key``/``frame``/``outline`` columns.
    max_cells : int or None
        Cap on displayed cells; beyond it a random subsample (``seed``)
        is taken and the third return value is True. None disables.
    seed : int
        Subsample RNG seed (deterministic display).

    Returns
    -------
    shapes : list of (N, 3) numpy.ndarray
        Polygon vertices ``(t, y, x)``; cells with null outlines skipped.
    properties : dict
        ``{"cell_key": [...], "frame": [...]}`` per shape.
    subsampled : bool
        True when ``max_cells`` truncated the display set.
    """
    rows = [r for r in to_records(cells) if r.get("outline")]
    rows, subsampled = _subsample(rows, max_cells, seed)
    shapes: list[np.ndarray] = []
    keys: list[int] = []
    frames: list[int] = []
    for row in rows:
        frame = int(row.get("frame", 0))
        shapes.append(_shape_vertices(row["outline"], frame))
        keys.append(int(row.get("cell_key", 0)))
        frames.append(frame)
    return shapes, {"cell_key": keys, "frame": frames}, subsampled


def cells_to_midline_shapes(
    cells: Any,
    max_cells: int | None = MAX_SHAPES_CELLS,
    seed: int = 0,
) -> tuple[list[np.ndarray], dict[str, list[int]], bool]:
    """Build one open path per midline branch for a napari Shapes layer.

    Branched cells contribute one path per branch (§2.3 ``midline`` is a
    list of branches). Returns ``(shapes, properties, subsampled)`` like
    :func:`cells_to_outline_shapes`.
    """
    rows = [r for r in to_records(cells) if r.get("midline")]
    rows, subsampled = _subsample(rows, max_cells, seed)
    shapes: list[np.ndarray] = []
    keys: list[int] = []
    frames: list[int] = []
    for row in rows:
        frame = int(row.get("frame", 0))
        for branch in row["midline"]:
            shapes.append(_shape_vertices(branch, frame))
            keys.append(int(row.get("cell_key", 0)))
            frames.append(frame)
    return shapes, {"cell_key": keys, "frame": frames}, subsampled


# ---------------------------------------------------------------------------
# Tracks layer (cells + edges + tracks)
# ---------------------------------------------------------------------------


class _UnionFind:
    """Minimal union-find over cell keys (path halving, union by attach)."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        parent = self.parent
        root = parent.setdefault(x, x)
        if root != x:
            # compress iteratively
            while parent[root] != root:
                parent[root] = parent[parent[root]]
                root = parent[root]
            parent[x] = root
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _rows_with_centroid(cells: Any) -> list[dict[str, Any]]:
    return [r for r in to_records(cells) if r.get("centroid")]


def _tracks_from_track_id_column(
    rows: list[dict[str, Any]],
) -> tuple[dict[int, int], dict[int, int]] | None:
    """Stage 1: use a non-null ``track_id`` column directly (per FOV)."""
    if not any(r.get("track_id") is not None for r in rows):
        return None
    pair_ids: dict[tuple[int, int], int] = {}
    for row in rows:
        tid = row.get("track_id")
        if tid is None:
            continue
        pair = (int(row.get("fov_id", 0)), int(tid))
        pair_ids.setdefault(pair, len(pair_ids) + 1)
    track_of_key: dict[int, int] = {}
    graph: dict[int, int] = {}
    for row in rows:
        tid = row.get("track_id")
        if tid is None:
            continue
        fov = int(row.get("fov_id", 0))
        track_of_key[int(row["cell_key"])] = pair_ids[(fov, int(tid))]
        parent = row.get("parent_track")
        if parent is not None and (fov, int(parent)) in pair_ids:
            graph[pair_ids[(fov, int(tid))]] = pair_ids[(fov, int(parent))]
    return track_of_key, graph


def _tracks_from_edges(
    rows: list[dict[str, Any]],
    edges: Any,
    tracks: Any,
) -> tuple[dict[int, int], dict[int, int]]:
    """Stage 2/3: union-find over ``continue`` edges + canonical id match."""
    edge_rows = to_records(edges)
    uf = _UnionFind()
    for edge in edge_rows:
        if str(edge.get("event")) == "continue":
            uf.union(int(edge["parent_key"]), int(edge["child_key"]))

    components: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        key = int(row["cell_key"])
        components.setdefault(uf.find(key), []).append(row)

    track_rows = to_records(tracks)
    remaining = list(track_rows)
    comp_track: dict[int, int] = {}
    used: set[int] = set()
    # Canonical rename: match each component to a tracks.parquet row on
    # (fov_id, birth_frame, last_frame, n_frames).
    for root, members in components.items():
        fov = int(members[0].get("fov_id", 0))
        frames = [int(m.get("frame", 0)) for m in members]
        first, last, count = min(frames), max(frames), len(members)
        match_idx = None
        for i, tr in enumerate(remaining):
            if (
                int(tr.get("fov_id", -1)) == fov
                and int(tr.get("birth_frame", -1)) == first
                and int(tr.get("last_frame", -1)) == last
                and int(tr.get("n_frames", count)) == count
            ):
                match_idx = i
                break
        if match_idx is not None:
            tid = int(remaining.pop(match_idx)["track_id"])
        else:
            tid = -1  # placeholder, assigned below
        comp_track[root] = tid
        if tid > 0:
            used.add(tid)
    next_id = max(used, default=0) + 1
    for root, tid in comp_track.items():
        if tid < 0:
            while next_id in used:
                next_id += 1
            comp_track[root] = next_id
            used.add(next_id)
            next_id += 1

    track_of_key = {
        int(row["cell_key"]): comp_track[uf.find(int(row["cell_key"]))]
        for row in rows
    }

    # Lineage graph: canonical parent_track column, else division edges.
    graph: dict[int, int] = {}
    for tr in track_rows:
        parent = tr.get("parent_track")
        tid = int(tr.get("track_id", 0))
        if parent is not None and tid in used and int(parent) in used:
            graph[tid] = int(parent)
    if not graph:
        for edge in edge_rows:
            if str(edge.get("event")) != "division":
                continue
            parent_key, child_key = int(edge["parent_key"]), int(edge["child_key"])
            if parent_key in track_of_key and child_key in track_of_key:
                parent_tid, child_tid = track_of_key[parent_key], track_of_key[child_key]
                if parent_tid != child_tid:
                    graph[child_tid] = parent_tid
    return track_of_key, graph


def cells_edges_to_tracks(
    cells: Any,
    edges: Any = None,
    tracks: Any = None,
    value_columns: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[int, int]]:
    """Build napari Tracks-layer inputs from cells (+ edges + tracks tables).

    Parameters
    ----------
    cells : table-like
        Cells table with ``cell_key``/``frame``/``centroid`` (and
        optionally ``track_id``/``parent_track``).
    edges : table-like, optional
        Edges table (``parent_key``/``child_key``/``event``); used to
        derive tracks when the cells table has no ``track_id`` column.
    tracks : table-like, optional
        Tracks table (§2.4); supplies canonical track ids and the
        lineage graph.
    value_columns : sequence of str, optional
        Extra per-vertex property columns (for ``color_by``); defaults to
        the scalar candidates present in the table plus any ``*age*``
        column.

    Returns
    -------
    data : numpy.ndarray, shape (N, 4)
        ``[track_id, frame, y, x]`` per vertex, sorted by track then
        frame (napari Tracks layer format).
    properties : dict of str to numpy.ndarray
        Per-vertex properties; always contains ``track_id``.
    graph : dict of int to int
        ``{child_track_id: parent_track_id}`` lineage graph.
    """
    rows = _rows_with_centroid(cells)
    if not rows:
        return (
            np.empty((0, 4), dtype=np.float64),
            {"track_id": np.empty(0, dtype=np.float64)},
            {},
        )

    resolved = _tracks_from_track_id_column(rows)
    if resolved is None:
        resolved = _tracks_from_edges(rows, edges, tracks)
    track_of_key, graph = resolved

    rows.sort(key=lambda r: (track_of_key[int(r["cell_key"])], int(r.get("frame", 0))))
    data = np.array(
        [
            [float(track_of_key[int(r["cell_key"])]), *_fyx(r)]
            for r in rows
        ],
        dtype=np.float64,
    )
    properties: dict[str, np.ndarray] = {"track_id": data[:, 0].copy()}

    if value_columns is None:
        present = set().union(*(r.keys() for r in rows))
        candidates = [c for c in _DEFAULT_VALUE_COLUMNS if c in present]
        candidates += sorted(c for c in present if "age" in c and c not in candidates)
        value_columns = candidates
    for col in value_columns:
        values: list[float] = []
        ok = False
        for row in rows:
            value = row.get(col)
            if value is None:
                values.append(np.nan)
                continue
            try:
                values.append(float(value))
                ok = True
            except (TypeError, ValueError):
                values.append(np.nan)
        if ok:
            properties[col] = np.asarray(values, dtype=np.float64)

    return data, properties, graph
