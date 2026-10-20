"""Cell tracking: LAP linking, gap closing, lineage, refinement (§4.7).

Replaces the heuristic chain of ``momia2/core/tracker.py::silly_link``
with a linear-assignment-problem (LAP) linker plus explicit division
detection:

- :func:`overlap_matrix` — shared-pixel counts between two label frames
  as a sparse COO matrix (single :func:`numpy.bincount` pass).
- :func:`link_frames` — one frame pair: cost matrix (overlap/IoU +
  centroid distance, optionally gated by a constant-velocity Kalman
  prediction + area ratio + Hu-moment similarity) solved with
  :func:`scipy.optimize.linear_sum_assignment`, then division detection
  among the unmatched cells (one mother ↔ two daughters, area
  conservation) and V-snapping flagging.
- :func:`link_timelapse` — full stack → schema-conforming edges and
  tracks tables, with gap closing across ≤ 2 skipped frames.
- :func:`refine_lineage` — graph-surgery correction of spurious
  convergence events (simplified port of the momia2 backtracking
  rectifier; see the function docstring for the simplifications).
- :func:`ctc_metrics` — lightweight SEG-like / TRA-like / division-event
  benchmark scores against ground truth.

All functions are pure and deterministic; tables are returned as
validated polars DataFrames (see :mod:`momia3.schema`).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import polars as pl
import pyarrow as pa
from pydantic import BaseModel, Field
from scipy import ndimage, optimize, sparse
from skimage.measure import regionprops
from skimage.segmentation import watershed

from .. import keys as _keys
from .. import schema as _schema

logger = logging.getLogger(__name__)

__all__ = [
    "LinkParams",
    "TrackParams",
    "RefineParams",
    "overlap_matrix",
    "link_frames",
    "link_timelapse",
    "refine_lineage",
    "ctc_metrics",
]

_BIG = 1e6
"""Forbidden-assignment cost inside the LAP matrix."""

_EPS = 1e-12


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


class LinkParams(BaseModel):
    """Frame-pair linking parameters for :func:`link_frames`."""

    w_overlap: float = Field(1.0, ge=0.0, description="Weight of 1 - overlap fraction")
    w_iou: float = Field(0.5, ge=0.0, description="Weight of 1 - IoU")
    w_distance: float = Field(
        1.0, ge=0.0, description="Weight of centroid distance / gate_distance"
    )
    w_area: float = Field(0.3, ge=0.0, description="Weight of 1 - area ratio")
    w_hu: float = Field(0.2, ge=0.0, description="Weight of Hu-moment dissimilarity")
    gate_distance: float = Field(
        20.0, gt=0, description="Hard centroid-distance gate (px)"
    )
    max_cost: float = Field(
        0.8, gt=0, description="Assignments above this total cost are rejected"
    )
    use_kalman: bool = Field(
        True,
        description="Gate distances on Kalman-predicted centroids (pred_y/pred_x "
        "record fields) instead of raw centroids when available",
    )
    div_min_overlap: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Division: min daughter-area fraction overlapping the mother",
    )
    div_min_area: float = Field(
        5.0, ge=0.0, description="Division: min mother-daughter shared pixels"
    )
    div_area_tolerance: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Division: |A_mother - sum(A_daughters)| / A_mother tolerance",
    )
    vsnap_angle_deg: float = Field(
        20.0,
        ge=0.0,
        le=90.0,
        description="Min daughter/mother major-axis angle change that flags a "
        "division edge as V-snapping (degrees)",
    )


class TrackParams(LinkParams):
    """Timelapse linking parameters (adds gap closing + Kalman noise)."""

    gap_close: int = Field(
        2, ge=0, le=5, description="Max skipped frames bridged by a 'gap' edge"
    )
    kalman_process_noise: float = Field(
        1e-2, gt=0, description="Kalman process noise (px^2)"
    )
    kalman_measurement_noise: float = Field(
        0.5, gt=0, description="Kalman measurement noise (px^2)"
    )


class RefineParams(BaseModel):
    """Lineage-refinement (graph surgery) parameters."""

    n_generations: int = Field(
        4,
        ge=1,
        le=20,
        description="Ancestor window (frames) for the re-merging branch test",
    )
    min_seed_fraction: float = Field(
        0.1,
        gt=0.0,
        le=1.0,
        description="Min fraction of the suspect cell a seed fragment must cover",
    )
    compactness: float = Field(
        0.1, ge=0.0, description="Compactness of the marker watershed"
    )


# ---------------------------------------------------------------------------
# generic table / record helpers
# ---------------------------------------------------------------------------


def _to_pa(table: pa.Table | pl.DataFrame) -> pa.Table:
    """Coerce a polars DataFrame or Arrow table to ``pa.Table``."""
    if isinstance(table, pa.Table):
        return table
    if isinstance(table, pl.DataFrame):
        return table.to_arrow()
    raise TypeError(
        f"expected pyarrow.Table or polars.DataFrame, got {type(table).__name__}"
    )


def _hu_transform(hu: np.ndarray) -> np.ndarray:
    """Sign-preserving log transform of Hu moments (scale compression)."""
    hu = np.asarray(hu, dtype=np.float64)
    return np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def _records(
    cells: pa.Table | pl.DataFrame | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize per-frame cell input to a list of plain record dicts.

    Accepted keys/columns: ``cell_key``, ``label``, ``centroid`` (struct)
    or ``y``/``x``, ``area``, ``hu_0..hu_6`` or ``hu``, ``orientation``
    (radians, optional), ``pred_y``/``pred_x`` (optional Kalman
    prediction), ``touching_edge`` (optional).
    """
    if isinstance(cells, (pa.Table, pl.DataFrame)):
        table = _to_pa(cells)
        rows = table.to_pylist()
    else:
        rows = [dict(r) for r in cells]
    out: list[dict[str, Any]] = []
    for row in rows:
        rec: dict[str, Any] = {
            "cell_key": int(row["cell_key"]),
            "label": int(row["label"]),
        }
        if "centroid" in row and row["centroid"] is not None:
            rec["y"] = float(row["centroid"]["y"])
            rec["x"] = float(row["centroid"]["x"])
        else:
            rec["y"] = float(row.get("y", row.get("centroid_y", np.nan)))
            rec["x"] = float(row.get("x", row.get("centroid_x", np.nan)))
        rec["area"] = float(row.get("area", np.nan))
        hu = row.get("hu")
        if hu is None:
            hu = [row.get(f"hu_{i}") for i in range(7)]
            if any(v is None for v in hu):
                hu = None
        rec["hu"] = None if hu is None else _hu_transform(np.asarray(hu, float))
        orientation = row.get("orientation")
        rec["orientation"] = None if orientation is None else float(orientation)
        rec["touching_edge"] = bool(row.get("touching_edge", False))
        if row.get("pred_y") is not None and row.get("pred_x") is not None:
            rec["pred_y"] = float(row["pred_y"])
            rec["pred_x"] = float(row["pred_x"])
        out.append(rec)
    return out


def _axis_angle_diff(a: float, b: float) -> float:
    """Smallest angle (degrees, [0, 90]) between two headless axes."""
    d = abs((a - b) % np.pi)
    d = min(d, np.pi - d)
    return float(np.degrees(d))


# ---------------------------------------------------------------------------
# overlap matrix
# ---------------------------------------------------------------------------


def overlap_matrix(labels_t: np.ndarray, labels_t1: np.ndarray) -> sparse.coo_matrix:
    """Shared-pixel counts between two instance-label frames.

    A single :func:`numpy.bincount` pass over the flattened pair code
    ``l1 * (max2 + 1) + l2`` (background pairs excluded); no
    ``np.unique(axis=0)``.

    Parameters
    ----------
    labels_t, labels_t1 : numpy.ndarray
        Instance label images of identical shape; 0 is background.

    Returns
    -------
    scipy.sparse.coo_matrix
        Shape ``(max(labels_t) + 1, max(labels_t1) + 1)``; entry
        ``[i, j]`` = pixels labeled ``i`` in ``labels_t`` and ``j`` in
        ``labels_t1``. Row/column 0 is always empty (background excluded).
    """
    l1 = np.asarray(labels_t).ravel()
    l2 = np.asarray(labels_t1).ravel()
    if l1.shape != l2.shape:
        raise ValueError(
            f"label frames must have the same shape, got "
            f"{np.asarray(labels_t).shape} vs {np.asarray(labels_t1).shape}"
        )
    max2 = int(l2.max()) if l2.size else 0
    max1 = int(l1.max()) if l1.size else 0
    fg = (l1 > 0) & (l2 > 0)
    if not fg.any():
        return sparse.coo_matrix((max1 + 1, max2 + 1), dtype=np.int64)
    code = l1[fg].astype(np.int64) * (max2 + 1) + l2[fg].astype(np.int64)
    counts = np.bincount(code, minlength=(max1 + 1) * (max2 + 1))
    nz = np.flatnonzero(counts)
    rows = nz // (max2 + 1)
    cols = nz % (max2 + 1)
    return sparse.coo_matrix(
        (counts[nz], (rows, cols)), shape=(max1 + 1, max2 + 1), dtype=np.int64
    )


def _overlap_dict(overlap: sparse.spmatrix) -> dict[tuple[int, int], int]:
    """Index a sparse overlap matrix as ``{(label_t, label_t1): count}``."""
    coo = sparse.coo_matrix(overlap)
    return {
        (int(r), int(c)): int(v) for r, c, v in zip(coo.row, coo.col, coo.data)
    }


# ---------------------------------------------------------------------------
# frame-pair linking (LAP)
# ---------------------------------------------------------------------------


def _hu_cost(hu1: np.ndarray | None, hu2: np.ndarray | None) -> float:
    """Hu-moment dissimilarity in [0, 1] (1 - cosine)/2; 0 when missing."""
    if hu1 is None or hu2 is None:
        return 0.0
    n1 = float(np.linalg.norm(hu1))
    n2 = float(np.linalg.norm(hu2))
    if n1 <= _EPS or n2 <= _EPS:
        return 0.0
    cos = float(np.dot(hu1, hu2) / (n1 * n2))
    return float(np.clip(1.0 - cos, 0.0, 2.0) / 2.0)


def link_frames(
    cells_t: pa.Table | pl.DataFrame | Sequence[Mapping[str, Any]],
    cells_t1: pa.Table | pl.DataFrame | Sequence[Mapping[str, Any]],
    overlap: sparse.spmatrix,
    params: LinkParams | None = None,
) -> list[dict[str, Any]]:
    """Link one frame pair into lineage edge records.

    Cost per candidate pair ``(i, j)`` is the weighted sum of
    ``1 - parent overlap coverage``, ``1 - IoU``, gated centroid distance
    (measured from the Kalman prediction ``pred_y/pred_x`` when present
    and ``params.use_kalman``), ``1 - area ratio`` and Hu-moment
    dissimilarity. Pairs beyond ``gate_distance`` are forbidden; the
    surviving assignment minimizes total cost via
    :func:`scipy.optimize.linear_sum_assignment`, and pairs with cost
    above ``max_cost`` are rejected.

    Division detection runs on the rejected/unmatched cells: a mother
    with two unmatched children each overlapping her by
    ``div_min_overlap`` of the child area (and ``div_min_area`` pixels)
    and satisfying area conservation within ``div_area_tolerance`` yields
    two ``division`` edges. A division edge gets ``vsnap = True`` when a
    daughter's major-axis orientation differs from the mother's by more
    than ``vsnap_angle_deg`` (orientations taken from the optional
    ``orientation`` record field; NaN disables the flag).

    Parameters
    ----------
    cells_t, cells_t1 : table or sequence of mapping
        Per-frame cell records (see :func:`_records`).
    overlap : scipy.sparse matrix
        Shared-pixel counts as returned by :func:`overlap_matrix`.
    params : LinkParams, optional
        Linking parameters; defaults are used when omitted.

    Returns
    -------
    list of dict
        Edge records ``{parent_key, child_key, event, confidence,
        vsnap}``; ``event`` is one of ``continue``, ``division``,
        ``appear``, ``disappear``. Root/leaf sentinel keys use
        :data:`momia3.keys.ROOT_KEY`.
    """
    params = params or LinkParams()
    parents = _records(cells_t)
    children = _records(cells_t1)
    ov = _overlap_dict(overlap)
    edges: list[dict[str, Any]] = []

    matched_p: set[int] = set()
    matched_c: set[int] = set()
    n1, n2 = len(parents), len(children)
    if n1 and n2:
        cost = np.full((n1, n2), _BIG, dtype=np.float64)
        for i, p in enumerate(parents):
            py = p.get("pred_y") if params.use_kalman else None
            px = p.get("pred_x") if params.use_kalman else None
            if py is None or px is None:
                py, px = p["y"], p["x"]
            for j, c in enumerate(children):
                d = float(np.hypot(py - c["y"], px - c["x"]))
                if d > params.gate_distance:
                    continue
                o = float(ov.get((p["label"], c["label"]), 0))
                a1 = max(p["area"], _EPS)
                a2 = max(c["area"], _EPS)
                frac = 0.0
                iou = 0.0
                if o > 0:
                    # Parent coverage (asymmetric): a mother explains only
                    # ~half of a single daughter, which keeps mother ->
                    # daughter pairs above the rejection gate so division
                    # detection can claim them.
                    frac = o / a1
                    iou = o / (a1 + a2 - o)
                area_ratio = min(a1, a2) / max(a1, a2)
                cost[i, j] = (
                    params.w_overlap * (1.0 - frac)
                    + params.w_iou * (1.0 - iou)
                    + params.w_distance * (d / params.gate_distance)
                    + params.w_area * (1.0 - area_ratio)
                    + params.w_hu * _hu_cost(p["hu"], c["hu"])
                )
        rows, cols = optimize.linear_sum_assignment(cost)
        for i, j in zip(rows.tolist(), cols.tolist()):
            if cost[i, j] >= _BIG or cost[i, j] > params.max_cost:
                continue  # rejected: candidates for division/appear/disappear
            matched_p.add(i)
            matched_c.add(j)
            edges.append(
                {
                    "parent_key": parents[i]["cell_key"],
                    "child_key": children[j]["cell_key"],
                    "event": "continue",
                    "confidence": float(1.0 / (1.0 + cost[i, j])),
                    "vsnap": False,
                }
            )

    # ---- explicit division detection among the unmatched -----------------
    unmatched_p = [i for i in range(n1) if i not in matched_p]
    unmatched_c = [j for j in range(n2) if j not in matched_c]
    for i in sorted(unmatched_p, key=lambda k: parents[k]["cell_key"]):
        p = parents[i]
        candidates = sorted(
            (
                (j, ov.get((p["label"], children[j]["label"]), 0))
                for j in unmatched_c
            ),
            key=lambda item: (-item[1], children[item[0]]["cell_key"]),
        )
        candidates = [
            (j, o)
            for j, o in candidates
            if o >= params.div_min_area
            and o >= params.div_min_overlap * max(children[j]["area"], _EPS)
        ]
        if len(candidates) < 2:
            continue
        (j1, o1), (j2, o2) = candidates[0], candidates[1]
        a_m = max(p["area"], _EPS)
        a_d = children[j1]["area"] + children[j2]["area"]
        rel = abs(a_m - a_d) / a_m
        if rel > params.div_area_tolerance:
            continue
        mother_angle = p["orientation"]
        confidence = float(1.0 - rel)
        for j in (j1, j2):
            vsnap = False
            daughter_angle = children[j]["orientation"]
            if (
                mother_angle is not None
                and daughter_angle is not None
                and np.isfinite(mother_angle)
                and np.isfinite(daughter_angle)
            ):
                vsnap = (
                    _axis_angle_diff(mother_angle, daughter_angle)
                    > params.vsnap_angle_deg
                )
            edges.append(
                {
                    "parent_key": p["cell_key"],
                    "child_key": children[j]["cell_key"],
                    "event": "division",
                    "confidence": confidence,
                    "vsnap": bool(vsnap),
                }
            )
            matched_c.add(j)
        matched_p.add(i)

    for i in sorted(set(range(n1)) - matched_p, key=lambda k: parents[k]["cell_key"]):
        edges.append(
            {
                "parent_key": parents[i]["cell_key"],
                "child_key": _keys.ROOT_KEY,
                "event": "disappear",
                "confidence": 1.0,
                "vsnap": False,
            }
        )
    for j in sorted(set(range(n2)) - matched_c, key=lambda k: children[k]["cell_key"]):
        edges.append(
            {
                "parent_key": _keys.ROOT_KEY,
                "child_key": children[j]["cell_key"],
                "event": "appear",
                "confidence": 1.0,
                "vsnap": False,
            }
        )
    logger.debug(
        "link_frames: %d parents, %d children -> %d edges", n1, n2, len(edges)
    )
    return edges


# ---------------------------------------------------------------------------
# timelapse linking
# ---------------------------------------------------------------------------


class _ConstVelocityKalman:
    """Minimal constant-velocity Kalman filter over the (y, x) centroid."""

    def __init__(self, y: float, x: float, q: float, r: float) -> None:
        self.state = np.array([y, x, 0.0, 0.0], dtype=np.float64)
        self.cov = np.diag([1.0, 1.0, 10.0, 10.0])
        self.q = float(q)
        self.r = float(r)
        self._F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        self._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)

    def predict(self) -> tuple[float, float]:
        """Advance one frame; return the predicted ``(y, x)`` position."""
        self.state = self._F @ self.state
        self.cov = self._F @ self.cov @ self._F.T + self.q * np.eye(4)
        return float(self.state[0]), float(self.state[1])

    def update(self, y: float, x: float) -> None:
        """Incorporate one centroid observation."""
        z = np.array([y, x], dtype=np.float64)
        s = self._H @ self.cov @ self._H.T + self.r * np.eye(2)
        k = self.cov @ self._H.T @ np.linalg.inv(s)
        self.state = self.state + k @ (z - self._H @ self.state)
        self.cov = (np.eye(4) - k @ self._H) @ self.cov

    @property
    def velocity(self) -> tuple[float, float]:
        """Current ``(vy, vx)`` estimate."""
        return float(self.state[2]), float(self.state[3])


def _frame_records_from_labels(
    labels_2d: np.ndarray, fov_id: int, frame: int
) -> list[dict[str, Any]]:
    """Derive linking records (centroid/area/Hu/orientation) from labels."""
    labels = np.asarray(labels_2d)
    h, w = labels.shape
    out: list[dict[str, Any]] = []
    for prop in regionprops(labels):
        minr, minc, maxr, maxc = prop.bbox
        out.append(
            {
                "cell_key": _keys.encode_key(fov_id, frame, int(prop.label)),
                "label": int(prop.label),
                "y": float(prop.centroid[0]),
                "x": float(prop.centroid[1]),
                "area": float(prop.area),
                "hu": _hu_transform(np.asarray(prop.moments_hu, dtype=np.float64)),
                "orientation": float(prop.orientation),
                "touching_edge": bool(
                    minr == 0 or minc == 0 or maxr == h or maxc == w
                ),
            }
        )
    out.sort(key=lambda rec: rec["cell_key"])
    return out


def _excluded_labels(
    cells_table: pa.Table | pl.DataFrame | None,
) -> set[tuple[int, int]]:
    """``(frame, label)`` pairs flagged ``include == False`` in the table."""
    if cells_table is None:
        return set()
    table = _to_pa(cells_table)
    names = set(table.column_names)
    if not {"frame", "label", "include"} <= names:
        return set()
    excluded: set[tuple[int, int]] = set()
    for row in table.select(["frame", "label", "include"]).to_pylist():
        if row["include"] is False:
            excluded.add((int(row["frame"]), int(row["label"])))
    return excluded


def _link_fov(
    labels_stack: np.ndarray, fov_id: int, params: TrackParams
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Link one FOV; return edge records and a key -> record lookup."""
    n_frames = labels_stack.shape[0]
    per_frame = [
        _frame_records_from_labels(labels_stack[t], fov_id, t) for t in range(n_frames)
    ]
    lookup: dict[int, dict[str, Any]] = {
        rec["cell_key"]: rec for frame_recs in per_frame for rec in frame_recs
    }
    filters: dict[int, _ConstVelocityKalman] = {}
    edges: list[dict[str, Any]] = []

    for t in range(n_frames - 1):
        parents = per_frame[t]
        children = per_frame[t + 1]
        if params.use_kalman:
            for rec in parents:
                kf = filters.get(rec["cell_key"])
                if kf is None:
                    kf = _ConstVelocityKalman(
                        rec["y"],
                        rec["x"],
                        params.kalman_process_noise,
                        params.kalman_measurement_noise,
                    )
                    filters[rec["cell_key"]] = kf
                rec["pred_y"], rec["pred_x"] = kf.predict()
        pair_edges = link_frames(
            parents, children, overlap_matrix(labels_stack[t], labels_stack[t + 1]), params
        )
        edges.extend(pair_edges)
        new_filters: dict[int, _ConstVelocityKalman] = {}
        for edge in pair_edges:
            pk, ck = edge["parent_key"], edge["child_key"]
            if pk == _keys.ROOT_KEY:  # appear: seed a fresh filter
                rec = lookup.get(ck)
                if rec is not None and params.use_kalman:
                    new_filters[ck] = _ConstVelocityKalman(
                        rec["y"],
                        rec["x"],
                        params.kalman_process_noise,
                        params.kalman_measurement_noise,
                    )
                continue
            if ck == _keys.ROOT_KEY:  # disappear
                filters.pop(pk, None)
                continue
            kf = filters.pop(pk, None)
            rec = lookup.get(ck)
            if kf is None or rec is None:
                continue
            if edge["event"] == "continue":
                kf.update(rec["y"], rec["x"])
                new_filters[ck] = kf
            elif edge["event"] == "division":
                vy, vx = kf.velocity
                daughter = _ConstVelocityKalman(
                    rec["y"],
                    rec["x"],
                    params.kalman_process_noise,
                    params.kalman_measurement_noise,
                )
                daughter.state[2], daughter.state[3] = vy, vx
                new_filters[ck] = daughter
        filters = new_filters
    return edges, lookup


def _close_gaps(
    edges: list[dict[str, Any]],
    lookup: Mapping[int, dict[str, Any]],
    params: TrackParams,
) -> list[dict[str, Any]]:
    """Bridge appear/disappear pairs spanning ≤ ``gap_close`` frames."""
    if params.gap_close < 1:
        return edges
    disappears = [e for e in edges if e["event"] == "disappear"]
    appears = [e for e in edges if e["event"] == "appear"]
    consumed: set[int] = set()
    replacements: dict[int, dict[str, Any]] = {}
    for edge in sorted(disappears, key=lambda e: e["parent_key"]):
        p = lookup.get(edge["parent_key"])
        if p is None:
            continue
        _, f_p, _ = _keys.decode_key(p["cell_key"])
        best: dict[str, Any] | None = None
        best_score = np.inf
        for cand in appears:
            c_key = cand["child_key"]
            if c_key in consumed:
                continue
            c = lookup.get(c_key)
            if c is None:
                continue
            _, f_c, _ = _keys.decode_key(c_key)
            span = f_c - f_p
            if span < 2 or span > params.gap_close + 1:
                continue
            d = float(np.hypot(p["y"] - c["y"], p["x"] - c["x"]))
            if d > params.gate_distance * span:
                continue
            a1, a2 = max(p["area"], _EPS), max(c["area"], _EPS)
            ratio = min(a1, a2) / max(a1, a2)
            if ratio < 0.5:
                continue
            score = d / (params.gate_distance * span) + (1.0 - ratio)
            if score < best_score:
                best_score = score
                best = cand
        if best is not None:
            consumed.add(best["child_key"])
            replacements[id(edge)] = {
                "parent_key": edge["parent_key"],
                "child_key": best["child_key"],
                "event": "gap",
                "confidence": float(1.0 / (1.0 + best_score)),
                "vsnap": False,
            }
    out: list[dict[str, Any]] = []
    for edge in edges:
        if id(edge) in replacements:
            out.append(replacements[id(edge)])
        elif edge["event"] == "appear" and edge["child_key"] in consumed:
            continue
        else:
            out.append(edge)
    logger.debug("gap closing bridged %d gaps", len(replacements))
    return out


def _build_tables(
    edges: list[dict[str, Any]],
    lookup: Mapping[int, dict[str, Any]],
    fov_id: int,
    n_frames: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Materialize schema-conforming edges + tracks tables from records."""
    edges_sorted = sorted(
        edges, key=lambda e: (e["parent_key"], e["child_key"], e["event"])
    )
    if edges_sorted:
        edge_table = pa.table(
            {
                "parent_key": pa.array(
                    [e["parent_key"] for e in edges_sorted], type=pa.uint64()
                ),
                "child_key": pa.array(
                    [e["child_key"] for e in edges_sorted], type=pa.uint64()
                ),
                "event": pa.array(
                    [e["event"] for e in edges_sorted], type=pa.string()
                ),
                "confidence": pa.array(
                    [e["confidence"] for e in edges_sorted], type=pa.float32()
                ),
                "vsnap": pa.array(
                    [e["vsnap"] for e in edges_sorted], type=pa.bool_()
                ),
            }
        )
        edges_df = pl.from_arrow(_schema.validate_edges(edge_table))
    else:
        edges_df = pl.from_arrow(_schema.empty_edges_table())

    # ---- tracks -----------------------------------------------------------
    track_of: dict[int, int] = {}
    parent_track: dict[int, int] = {}
    next_track = 1
    real_edges = sorted(
        (
            e
            for e in edges_sorted
            if e["parent_key"] != _keys.ROOT_KEY and e["child_key"] != _keys.ROOT_KEY
        ),
        key=lambda e: (
            _keys.decode_key(e["parent_key"])[1],
            e["parent_key"],
            e["child_key"],
        ),
    )
    for edge in real_edges:
        pk, ck = edge["parent_key"], edge["child_key"]
        if pk not in track_of:
            track_of[pk] = next_track
            next_track += 1
        if edge["event"] in ("continue", "gap"):
            track_of[ck] = track_of[pk]
        elif edge["event"] == "division":
            track_of[ck] = next_track
            parent_track[next_track] = track_of[pk]
            next_track += 1
    for key in sorted(lookup):
        if key not in track_of:
            track_of[key] = next_track
            next_track += 1

    members: dict[int, list[int]] = {}
    for key, tid in track_of.items():
        members.setdefault(tid, []).append(key)
    divided_at: dict[int, int] = {}
    for edge in real_edges:
        if edge["event"] == "division":
            tid = track_of[edge["parent_key"]]
            divided_at[tid] = max(
                divided_at.get(tid, -1), _keys.decode_key(edge["parent_key"])[1]
            )

    rows: list[dict[str, Any]] = []
    for tid in sorted(members):
        keys_ = members[tid]
        frames = [_keys.decode_key(k)[1] for k in keys_]
        birth, last = min(frames), max(frames)
        terminal = lookup[max(keys_, key=lambda k: _keys.decode_key(k)[1])]
        if tid in divided_at:
            fate = "divide"
        elif terminal["touching_edge"]:
            fate = "edge"
        elif last < n_frames - 1:
            fate = "disappear"
        else:
            fate = "incomplete"
        rows.append(
            {
                "track_id": tid,
                "fov_id": fov_id,
                "birth_frame": birth,
                "last_frame": last,
                "division_frame": divided_at.get(tid),
                "parent_track": parent_track.get(tid),
                "n_frames": len(set(frames)),
                "fate": fate,
            }
        )
    if rows:
        track_table = pa.table(
            {
                "track_id": pa.array([r["track_id"] for r in rows], type=pa.uint32()),
                "fov_id": pa.array([r["fov_id"] for r in rows], type=pa.uint32()),
                "birth_frame": pa.array(
                    [r["birth_frame"] for r in rows], type=pa.uint32()
                ),
                "last_frame": pa.array(
                    [r["last_frame"] for r in rows], type=pa.uint32()
                ),
                "division_frame": pa.array(
                    [r["division_frame"] for r in rows], type=pa.uint32()
                ),
                "parent_track": pa.array(
                    [r["parent_track"] for r in rows], type=pa.uint32()
                ),
                "n_frames": pa.array([r["n_frames"] for r in rows], type=pa.uint32()),
                "fate": pa.array([r["fate"] for r in rows], type=pa.string()),
            }
        )
        tracks_df = pl.from_arrow(_schema.validate_tracks(track_table))
    else:
        tracks_df = pl.from_arrow(_schema.empty_tracks_table())
    return edges_df, tracks_df


def link_timelapse(
    labels_stack: np.ndarray | Mapping[int, np.ndarray],
    cells_table: pa.Table | pl.DataFrame | None = None,
    params: TrackParams | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Link a full timelapse into edges and tracks tables.

    Per FOV, each frame pair is linked with :func:`link_frames` (Kalman
    predictions maintained across the stack when ``params.use_kalman``),
    then :func:`_close_gaps` bridges disappear/appear pairs spanning at
    most ``params.gap_close`` skipped frames with a ``gap`` edge. Tracks
    are the per-cell lineages: ``continue``/``gap`` edges keep the track,
    ``division`` edges fork two new tracks with ``parent_track`` set.
    ``fate`` precedence is ``divide`` > ``edge`` (terminal cell touches
    the FOV border) > ``disappear`` (lost before the last frame) >
    ``incomplete``.

    Parameters
    ----------
    labels_stack : numpy.ndarray or mapping of int to numpy.ndarray
        ``(t, y, x)`` instance labels for one FOV, or a mapping
        ``fov_id -> (t, y, x)`` for a multi-FOV experiment (linked per
        FOV and concatenated).
    cells_table : pyarrow.Table or polars.DataFrame, optional
        Cells table; rows flagged ``include == False`` are excluded from
        linking. All linking geometry is derived from ``labels_stack``
        itself (schema §2.3: geometry defaults to derivation from the
        labels), so the table is only needed for the include filter.
    params : TrackParams, optional
        Linking / gap-closing / Kalman parameters.

    Returns
    -------
    tuple of polars.DataFrame
        ``(edges, tracks)`` validated against
        :func:`momia3.schema.validate_edges` /
        :func:`momia3.schema.validate_tracks`. The edges table carries an
        extra boolean ``vsnap`` column (passed through validation).
    """
    params = params or TrackParams()
    if isinstance(labels_stack, Mapping):
        stacks = {int(k): np.asarray(v) for k, v in labels_stack.items()}
    else:
        fov_id = 0
        if cells_table is not None:
            table = _to_pa(cells_table)
            if "fov_id" in table.column_names and table.num_rows:
                fov_id = min(int(v) for v in table.column("fov_id").unique().to_pylist())
        stacks = {fov_id: np.asarray(labels_stack)}

    excluded = _excluded_labels(cells_table)
    edges_parts: list[pl.DataFrame] = []
    tracks_parts: list[pl.DataFrame] = []
    for fov_id, stack in sorted(stacks.items()):
        if stack.ndim != 3:
            raise ValueError(
                f"labels_stack for fov {fov_id} must be (t, y, x), got {stack.shape}"
            )
        work = stack
        if excluded:
            work = stack.copy()
            for t in range(work.shape[0]):
                for lab in np.unique(work[t]):
                    if lab != 0 and (t, int(lab)) in excluded:
                        work[t][work[t] == lab] = 0
        edges, lookup = _link_fov(work, fov_id, params)
        edges = _close_gaps(edges, lookup, params)
        edges_df, tracks_df = _build_tables(edges, lookup, fov_id, int(stack.shape[0]))
        edges_parts.append(edges_df)
        tracks_parts.append(tracks_df)

    edges_df = (
        pl.concat(edges_parts)
        if edges_parts
        else pl.from_arrow(_schema.empty_edges_table())
    )
    tracks_df = (
        pl.concat(tracks_parts)
        if tracks_parts
        else pl.from_arrow(_schema.empty_tracks_table())
    )
    logger.info(
        "link_timelapse: %d edges, %d tracks across %d FOV(s)",
        edges_df.height,
        tracks_df.height,
        len(stacks),
    )
    return edges_df, tracks_df


# ---------------------------------------------------------------------------
# lineage refinement (graph surgery)
# ---------------------------------------------------------------------------


def _common_ancestor_within(
    graph: nx.DiGraph, node: int, n_generations: int
) -> bool:
    """True when all mothers of ``node`` share an ancestor ≤ n frames back."""
    mothers = list(graph.predecessors(node))
    if len(mothers) < 2:
        return False
    _, f_node, _ = _keys.decode_key(node)
    common: set[int] | None = None
    for mother in mothers:
        ancestors = nx.ancestors(graph, mother) | {mother}
        close = {
            a for a in ancestors if 0 <= f_node - _keys.decode_key(a)[1] <= n_generations
        }
        common = close if common is None else (common & close)
        if not common:
            return False
    return bool(common)


def refine_lineage(
    labels_stack: np.ndarray,
    edges_table: pa.Table | pl.DataFrame,
    cells_table: pa.Table | pl.DataFrame | None = None,
    params: RefineParams | None = None,
    writeback_labels: bool = False,
) -> pl.DataFrame:
    """Correct spurious convergence events by re-splitting suspect cells.

    A *suspect* is a cell with ≥ 2 mothers whose ancestral branches
    re-merge within ``params.n_generations`` frames (the legacy momia2
    backtracking criterion, evaluated directly on the networkx lineage
    graph). Each suspect label is re-split by a marker-controlled
    watershed on the inverse distance transform of its mask, seeded from
    the previous-frame *ancestor fragments* (the mothers' footprints,
    each required to cover ≥ ``min_seed_fraction`` of the suspect mask —
    the ``hard_split`` fallback of momia2's ``_split_cells``). Edges are
    rewired: each mother connects to the fragment grown from its own
    seed; children of the suspect join the fragment with the largest
    shared-pixel overlap.

    Simplifications versus the legacy ``momia2/core/tracker.py``
    algorithm:

    - split-only surgery (spurious *merges* are detected but only the
      converged cell is split; no merge of over-split siblings),
    - single pass (no iterate-until-convergence loop),
    - seeds come from ancestor label fragments instead of probability
      maps (momia3 linking does not require a pixel classifier),
    - the divergence-count convolution heuristic is replaced by a direct
      common-ancestor test on the lineage graph.

    Parameters
    ----------
    labels_stack : numpy.ndarray
        ``(t, y, x)`` instance labels of one FOV (the segmentation that
        produced ``edges_table``).
    edges_table : pyarrow.Table or polars.DataFrame
        Lineage edges (``parent_key``, ``child_key``, ``event``,
        ``confidence``).
    cells_table : pyarrow.Table or polars.DataFrame, optional
        Reserved for API symmetry; currently unused.
    params : RefineParams, optional
        Refinement parameters.
    writeback_labels : bool, optional
        When ``True``, the split fragments are written back into
        ``labels_stack`` in place (the suspect's pixels are replaced by
        the fresh fragment labels), so the caller's copy of the labels
        matches the returned edges. Off by default to keep the function
        side-effect free.

    Returns
    -------
    polars.DataFrame
        Corrected edges table (schema-conforming, ``vsnap`` column
        preserved when present). Split fragments receive fresh labels
        (frame-local max + 1, + 2, …); the suspect's own edges are
        removed and replaced. With ``writeback_labels=True`` the input
        ``labels_stack`` carries the same fresh fragment labels.
    """
    del cells_table  # reserved for future feature-guided seeding
    params = params or RefineParams()
    table = _to_pa(edges_table)
    names = table.column_names
    rows = table.to_pylist()
    has_vsnap = "vsnap" in names

    real_edges = [
        r
        for r in rows
        if int(r["parent_key"]) != _keys.ROOT_KEY and int(r["child_key"]) != _keys.ROOT_KEY
    ]
    graph = nx.DiGraph()
    graph.add_edges_from(
        (int(r["parent_key"]), int(r["child_key"])) for r in real_edges
    )

    suspects = [
        node
        for node in graph.nodes
        if graph.in_degree(node) >= 2
        and _common_ancestor_within(graph, node, params.n_generations)
    ]
    if not suspects:
        logger.debug("refine_lineage: no suspicious convergence found")
        return pl.from_arrow(_schema.validate_edges(table))

    labels = np.asarray(labels_stack)
    drop: set[tuple[int, int]] = set()  # (parent, child) edges to remove
    additions: list[dict[str, Any]] = []
    for suspect in sorted(suspects):
        fov, frame, lab = _keys.decode_key(suspect)
        if frame < 1 or frame >= labels.shape[0]:
            continue
        mask = labels[frame] == lab
        area = int(mask.sum())
        if area == 0:
            continue
        mothers = sorted(graph.predecessors(suspect))
        prev = labels[frame - 1]
        # Seed markers from the mothers' footprints inside the suspect mask.
        markers = np.zeros(mask.shape, dtype=np.int64)
        mother_of_seed: dict[int, int] = {}
        seed_id = 0
        for mother in mothers:
            _, f_m, l_m = _keys.decode_key(mother)
            if f_m != frame - 1:
                logger.debug(
                    "refine_lineage: mother %d not adjacent to suspect %d; skipped",
                    mother,
                    suspect,
                )
                markers[:] = 0
                break
            fragment = mask & (prev == l_m)
            if fragment.sum() < params.min_seed_fraction * area:
                markers[:] = 0
                break
            seed_id += 1
            markers[fragment] = seed_id
            mother_of_seed[seed_id] = mother
        if seed_id < 2 or not markers.any():
            continue
        distance = ndimage.distance_transform_edt(mask)
        split = watershed(
            -distance, markers=markers, mask=mask, compactness=params.compactness
        )
        base = int(labels[frame].max())
        fragment_key: dict[int, int] = {}
        for seed in range(1, seed_id + 1):
            new_label = base + seed
            fragment_key[seed] = _keys.encode_key(fov, frame, new_label)
        if writeback_labels:
            # Replace the suspect's pixels with the fresh fragments so the
            # caller's labels match the rewired edges (§8 interactive fix).
            labels[frame][mask] = 0
            for seed, frag_key in fragment_key.items():
                labels[frame][mask & (split == seed)] = _keys.decode_key(frag_key)[2]
        # Rewire mothers -> their own fragments.
        for seed, mother in mother_of_seed.items():
            drop.add((mother, suspect))
            additions.append(
                {
                    "parent_key": mother,
                    "child_key": fragment_key[seed],
                    "event": "continue",
                    "confidence": 0.5,
                    "vsnap": False,
                }
            )
        # Rewire children of the suspect to the best-overlapping fragment.
        split_labels = np.where(mask, split, 0)
        children = sorted(graph.successors(suspect))
        assigned: set[int] = set()
        for child in children:
            _, f_c, l_c = _keys.decode_key(child)
            drop.add((suspect, child))
            best_seed, best_overlap = 0, 0
            if f_c == frame + 1:
                nxt = labels[f_c] == l_c
                for seed in range(1, seed_id + 1):
                    o = int(((split_labels == seed) & nxt).sum())
                    if o > best_overlap:
                        best_seed, best_overlap = seed, o
            if best_seed == 0:
                best_seed = 1
            additions.append(
                {
                    "parent_key": fragment_key[best_seed],
                    "child_key": child,
                    "event": "continue",
                    "confidence": 0.5,
                    "vsnap": False,
                }
            )
            assigned.add(best_seed)
        for seed in range(1, seed_id + 1):
            if seed not in assigned and not children:
                additions.append(
                    {
                        "parent_key": fragment_key[seed],
                        "child_key": _keys.ROOT_KEY,
                        "event": "disappear",
                        "confidence": 1.0,
                        "vsnap": False,
                    }
                )
        logger.info(
            "refine_lineage: split suspect cell %d into %d fragments",
            suspect,
            seed_id,
        )

    kept = [
        r
        for r in rows
        if (int(r["parent_key"]), int(r["child_key"])) not in drop
    ]
    out_rows = kept + additions
    out_rows.sort(key=lambda r: (int(r["parent_key"]), int(r["child_key"])))
    data: dict[str, pa.Array] = {
        "parent_key": pa.array([int(r["parent_key"]) for r in out_rows], type=pa.uint64()),
        "child_key": pa.array([int(r["child_key"]) for r in out_rows], type=pa.uint64()),
        "event": pa.array([str(r["event"]) for r in out_rows], type=pa.string()),
        "confidence": pa.array(
            [float(r["confidence"]) for r in out_rows], type=pa.float32()
        ),
    }
    if has_vsnap:
        data["vsnap"] = pa.array(
            [bool(r.get("vsnap", False)) for r in out_rows], type=pa.bool_()
        )
    return pl.from_arrow(_schema.validate_edges(pa.table(data)))


# ---------------------------------------------------------------------------
# benchmark metrics (lightweight CTC variants)
# ---------------------------------------------------------------------------


def _edge_tuples(
    edges: pa.Table | pl.DataFrame | Iterable[Sequence[Any]],
) -> list[tuple[int, int, str]]:
    """Normalize edges (table or ``(parent, child[, event])`` tuples)."""
    if isinstance(edges, (pa.Table, pl.DataFrame)):
        table = _to_pa(edges)
        names = set(table.column_names)
        has_event = "event" in names
        out = []
        for row in table.select(
            [c for c in ("parent_key", "child_key", "event") if c in names]
        ).to_pylist():
            out.append(
                (
                    int(row["parent_key"]),
                    int(row["child_key"]),
                    str(row["event"]) if has_event else "continue",
                )
            )
        return out
    out = []
    for item in edges:
        if len(item) >= 3:
            out.append((int(item[0]), int(item[1]), str(item[2])))
        else:
            out.append((int(item[0]), int(item[1]), "continue"))
    return out


def ctc_metrics(
    pred_edges: pa.Table | pl.DataFrame | Iterable[Sequence[Any]],
    pred_labels: np.ndarray,
    truth_edges: pa.Table | pl.DataFrame | Iterable[Sequence[Any]],
    truth_labels: np.ndarray,
) -> dict[str, Any]:
    """Lightweight Cell-Tracking-Challenge-style benchmark scores.

    These are *variants*, not the reference CTC implementations:

    - ``seg_like`` — mean over ground-truth instances of the best IoU
      against any predicted instance (per frame); the CTC SEG measure
      additionally normalizes by the number of reference objects only
      (no detection matching), which this reproduces, but it skips the
      CTC tooling's pixel-level caching.
    - ``tra_like`` — Jaccard index ``TP / (TP + FP + FN)`` over
      ``(parent_key, child_key)`` sets of continue/division/gap edges
      (appear/disappear excluded); the official TRA is AOGM-weighted
      with operation-specific penalties.
    - division events — precision/recall over the sets of parent keys
      of ``division`` edges.

    Parameters
    ----------
    pred_edges : table or iterable of tuples
        Predicted lineage edges.
    pred_labels : numpy.ndarray
        ``(t, y, x)`` predicted instance labels.
    truth_edges : table or iterable of tuples
        Ground-truth lineage edges.
    truth_labels : numpy.ndarray
        ``(t, y, x)`` ground-truth instance labels.

    Returns
    -------
    dict
        Keys ``seg_like``, ``tra_like``, ``edge_tp/fp/fn``,
        ``division_tp/fp/fn``, ``division_precision``,
        ``division_recall``.
    """
    pred = _edge_tuples(pred_edges)
    truth = _edge_tuples(truth_edges)

    link_events = {"continue", "division", "gap", "merge"}
    pred_pairs = {
        (p, c) for p, c, e in pred if e in link_events and p != _keys.ROOT_KEY
    }
    truth_pairs = {
        (p, c) for p, c, e in truth if e in link_events and p != _keys.ROOT_KEY
    }
    tp = len(pred_pairs & truth_pairs)
    fp = len(pred_pairs - truth_pairs)
    fn = len(truth_pairs - pred_pairs)
    tra_like = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0

    pred_div = {p for p, _, e in pred if e == "division"}
    truth_div = {p for p, _, e in truth if e == "division"}
    div_tp = len(pred_div & truth_div)
    div_fp = len(pred_div - truth_div)
    div_fn = len(truth_div - pred_div)

    pred_stack = np.asarray(pred_labels)
    truth_stack = np.asarray(truth_labels)
    ious: list[float] = []
    for t in range(truth_stack.shape[0]):
        t_lab, p_lab = truth_stack[t], pred_stack[t]
        ov = _overlap_dict(overlap_matrix(t_lab, p_lab))
        truth_areas = np.bincount(t_lab.ravel().astype(np.int64))
        pred_areas = np.bincount(p_lab.ravel().astype(np.int64))
        best: dict[int, float] = {}
        for (lt, lp), o in ov.items():
            union = truth_areas[lt] + pred_areas[lp] - o
            iou = o / union if union > 0 else 0.0
            if iou > best.get(lt, 0.0):
                best[lt] = iou
        for lab in np.unique(t_lab):
            if lab != 0:
                ious.append(best.get(int(lab), 0.0))
    seg_like = float(np.mean(ious)) if ious else 1.0

    return {
        "seg_like": seg_like,
        "tra_like": float(tra_like),
        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
        "division_tp": div_tp,
        "division_fp": div_fp,
        "division_fn": div_fn,
        "division_precision": (
            div_tp / (div_tp + div_fp) if (div_tp + div_fp) else 1.0
        ),
        "division_recall": (div_tp / (div_tp + div_fn) if (div_tp + div_fn) else 1.0),
    }
