"""Midline / pole / width-profile extraction (ARCHITECTURE.md §4.5 / §5.2).

**Pole-first, low-DOF spline algorithm** (replaces the earlier
skeleton-first pipeline, which zigzagged on near-ovoid, chain-forming
cells and grew spur branches at chain-junction crevices):

1. **Pole-first initialization** (biological prior: exactly two poles) —
   the two outline points with maximal geodesic separation, constrained to
   rough alignment with the second-moments major axis of the mask.
2. **Distance-transform geodesic centerline** (replaces the skeleton
   initialization) — pixel-grid Dijkstra (:func:`skimage.graph.
   route_through_array`, 8-connectivity) between the poles with cost
   ``1/(EDT + eps)``: guaranteed inside the mask, naturally centered, no
   staircase topology, no spur branches.
3. **Low-DOF regularized spline fitting** (replaces per-point iterative
   refinement) — a cubic B-spline with few control points
   (``clamp(round(length_px / ctrl_spacing_px), ctrl_min, ctrl_max)``)
   minimizing a perpendicular-midpoint data term (KDTree ray machinery),
   a bending-energy regularizer on second finite differences of the
   control points (``lambda_smooth``) and a centrality pull toward the
   geodesic centerline (``mu_central``). Solved as a small linear
   least-squares problem with 3–5 re-parameterization passes.
4. **Convergence cascade + quality flag** — ``midline_quality`` per cell:
   ``"refined"`` (spline improved on the geodesic residual),
   ``"geodesic"`` (fallback to the smoothed geodesic centerline),
   ``"degenerate"`` (aspect ratio < ``degenerate_ar``: major-axis segment,
   no refinement), ``"failed"`` (any exception or pathological topology:
   empty midline, never garbage).
5. **Septum-aware pole handling** — end tangents are fit on the distal
   ``end_window_frac`` of arc length only; where the width profile dips
   significantly in an end third (septum/chain-junction crevice), the
   geodesic centerline's end direction is trusted over the data-term
   tangent. Poles are extended along the end tangent to the outline.
6. **Width profile & scalars** — unchanged KDTree normal-intersection
   logic, computed on the final curve.

Skeletonization (:func:`skimage.morphology.skeletonize`) is still used,
but ONLY for side-branch *reporting* of branched cells — the main midline
never comes from the skeleton.

Coordinate convention: all output geometry is ``(x, y)`` in *global*
image coordinates. ``width_profile`` and ``length``/``width`` scalars are
in microns (``pixel_microns``).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import networkx as nx
import numpy as np
import polars as pl
import pyarrow as pa
from numba import njit
from scipy import interpolate, ndimage
from scipy.spatial import cKDTree
from skimage import morphology
from skimage.graph import route_through_array

from momia3 import keys as _keys
from momia3 import schema as _schema

logger = logging.getLogger(__name__)

__all__ = ["extract_midlines", "fill_cells_midline"]

MIDLINE_SCALAR_COLUMNS: tuple[str, ...] = (
    "length_um",
    "width_median_um",
    "width_std_um",
    "width_max_um",
    "width_q1_um",
    "width_q3_um",
    "sinuosity",
    "branched",
    "pole1_x",
    "pole1_y",
    "pole2_x",
    "pole2_y",
)
"""Scalar columns written by :func:`fill_cells_midline`."""

MIDLINE_QUALITY_COLUMN = "midline_quality"
"""Per-cell convergence-cascade flag written by :func:`fill_cells_midline`."""

MIDLINE_DIAG_COLUMNS: dict[str, str] = {
    "midline_n_iter": "n_iter",
    "midline_residual": "final_residual",
    "midline_max_curv": "max_curvature",
    "width_cv": "width_cv",
}
"""Diagnostic columns written by :func:`fill_cells_midline`
(arrow column name -> per-label result key)."""

_DEFAULT_PARAMS: dict[str, Any] = {
    # skeleton side-branch detection (branch reporting only)
    "method": "zhang",
    "min_branch_length_um": 0.2,
    # shared
    "n_points": 32,
    "smooth": 1.0,
    "knn": 24,
    "pad": 2,
    # legacy refinement parameters, kept for API compatibility (unused)
    "refine_iterations": 10,
    "refine_tolerance": 0.1,
    "refine_anchor": 3,
    "refine_max_step": 1.5,
    "pole_fit_points": 5,
    # pole-first initialization
    "pole_max_candidates": 400,
    "pole_axis_tol_deg": 45.0,
    # geodesic centerline
    "edt_eps": 0.25,
    "hole_area_frac": 0.02,
    # low-DOF spline fit
    "ctrl_spacing_px": 8.0,
    "ctrl_min": 5,
    "ctrl_max": 9,
    "lambda_smooth": 1.0,
    "mu_central": 0.5,
    "anchor_weight": 10.0,
    "spline_iterations": 5,
    "spline_tolerance": 0.1,
    "spline_max_step": 1.5,
    # septum-aware poles
    "end_window_frac": 0.2,
    "septum_dip_frac": 0.25,
    # degenerate (near-round) cells
    "degenerate_ar": 1.4,
}


# ---------------------------------------------------------------------------
# generic polyline helpers
# ---------------------------------------------------------------------------


def _arc_length(points: np.ndarray) -> float:
    """Total arc length of an open polyline in ``(x, y)`` coords."""
    if len(points) < 2:
        return 0.0
    seg = points[1:] - points[:-1]
    return float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))


def _arc_length_rc(path_rowcol: np.ndarray) -> float:
    """Arc length of a ``(row, col)`` path (isotropic pixels)."""
    if len(path_rowcol) < 2:
        return 0.0
    seg = np.diff(path_rowcol, axis=0)
    return float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))


def _resample_arclength(points: np.ndarray, n: int) -> np.ndarray:
    """Resample an open polyline to ``n`` points uniform in arc length."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.repeat(pts[:1], int(n), axis=0)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return np.repeat(pts[:1], int(n), axis=0)
    targets = np.linspace(0.0, total, int(n))
    x = np.interp(targets, s, pts[:, 0])
    y = np.interp(targets, s, pts[:, 1])
    return np.column_stack([x, y])


def _smooth_open(points: np.ndarray, smooth: float, n: int | None = None) -> np.ndarray:
    """Open B-spline smoothing resampled to ``n`` points (input count if
    None). Falls back to plain arc-length resampling on failure."""
    pts = np.asarray(points, dtype=np.float64)
    m = len(pts) if n is None else int(n)
    if len(pts) < 4:
        return _resample_arclength(pts, m)
    try:
        tck, _ = interpolate.splprep(
            [pts[:, 0], pts[:, 1]], s=float(smooth), per=0, k=3, quiet=1
        )
        uu = np.linspace(0.0, 1.0, m)
        x, y = interpolate.splev(uu, tck, der=0)
        return np.column_stack([x, y])
    except (ValueError, TypeError) as exc:
        logger.debug("open spline fit failed (%s); using linear resample", exc)
        return _resample_arclength(pts, m)


def _normals(points: np.ndarray) -> np.ndarray:
    """Unit normals of an open polyline (central differences)."""
    pts = np.asarray(points, dtype=np.float64)
    tangents = np.gradient(pts, axis=0)
    norm = np.hypot(tangents[:, 0], tangents[:, 1])
    norm[norm == 0] = 1.0
    tangents = tangents / norm[:, None]
    return np.column_stack([-tangents[:, 1], tangents[:, 0]])


def _curvature_stats(points: np.ndarray) -> tuple[float, float]:
    """``(total_curvature, max_curvature)`` of an open polyline.

    Total curvature is the sum of absolute turning angles (radians); max
    curvature is the largest per-step turning angle divided by the mean
    adjacent segment length (rad/px).
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return 0.0, 0.0
    v = np.diff(pts, axis=0)
    seg = np.hypot(v[:, 0], v[:, 1])
    ok = seg > 1e-9
    if ok.sum() < 2:
        return 0.0, 0.0
    ang = np.arctan2(v[ok, 1], v[ok, 0])
    d = np.diff(ang)
    d = (d + np.pi) % (2.0 * np.pi) - np.pi
    seg_ok = seg[ok]
    mean_seg = 0.5 * (seg_ok[:-1] + seg_ok[1:])
    kappa = np.abs(d) / np.maximum(mean_seg, 1e-9)
    return float(np.sum(np.abs(d))), float(np.max(kappa))


# ---------------------------------------------------------------------------
# KDTree-based outline intersection queries
# ---------------------------------------------------------------------------


@njit(cache=True)
def _ray_solve_kernel(
    a: np.ndarray,
    s: np.ndarray,
    rx: np.ndarray,
    ry: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    max_distance: float,
    has_max: bool,
) -> np.ndarray:
    """Nearest valid outline-segment intersection distance per ray.

    Numba port of the vectorized solve formerly inlined in
    :meth:`_OutlineIndex._ray_pass` (ARCHITECTURE.md §3.3: per-cell ROI
    geometry runs numpy/numba on CPU). The arithmetic is written in the
    exact same operation order as the numpy version, so results are
    bit-identical (numba does not enable fast-math/FMA contraction).

    Parameters
    ----------
    a : numpy.ndarray
        ``(M, k, 2)`` segment start points (float64).
    s : numpy.ndarray
        ``(M, k, 2)`` segment vectors (float64).
    rx, ry, dx, dy : numpy.ndarray
        ``(M,)`` ray origins and direction components (float64).
    max_distance : float
        Distance cap (used only when ``has_max``).
    has_max : bool
        Whether the distance cap applies.

    Returns
    -------
    numpy.ndarray
        ``(M,)`` nearest forward intersection distance; ``inf`` when no
        candidate segment crosses the ray.
    """
    m, k, _ = a.shape
    t_out = np.full(m, np.inf, dtype=np.float64)
    for i in range(m):
        t_min = np.inf
        for j in range(k):
            det = -dx[i] * s[i, j, 1] + dy[i] * s[i, j, 0]
            if not (abs(det) > 1e-12):
                continue
            r0 = a[i, j, 0] - rx[i]
            r1 = a[i, j, 1] - ry[i]
            t = (-r0 * s[i, j, 1] + s[i, j, 0] * r1) / det
            u = (dx[i] * r1 - r0 * dy[i]) / det
            if not np.isfinite(t):
                continue
            if not (t > 0.0):
                continue
            if not (u >= -1e-6 and u <= 1.0 + 1e-6):
                continue
            if has_max and not (t <= max_distance):
                continue
            if t < t_min:
                t_min = t
        t_out[i] = t_min
    return t_out


class _OutlineIndex:
    """Closed-outline segment index backed by a cKDTree on midpoints.

    All nearest-point / ray-intersection queries are local (k nearest
    candidate segments), never dense O(n_outline x n_query) matrices.
    """

    def __init__(self, outline_xy: np.ndarray) -> None:
        ring = np.asarray(outline_xy, dtype=np.float64)
        self.v1 = ring
        self.v2 = np.roll(ring, -1, axis=0)
        self.seg = self.v2 - self.v1  # precomputed segment vectors
        self.mid = 0.5 * (self.v1 + self.v2)
        self.tree = cKDTree(self.mid)
        self.n_segments = len(ring)

    def ray_intersections(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        knn: int,
        max_distance: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nearest outline intersection along rays ``p + t * d`` (t > 0).

        Parameters
        ----------
        points : numpy.ndarray
            ``(M, 2)`` ray origins.
        directions : numpy.ndarray
            ``(M, 2)`` ray directions (need not be unit; ``t`` scales
            accordingly).
        knn : int
            Number of candidate segments per query.
        max_distance : float, optional
            If given, intersections beyond this distance are discarded.

        Returns
        -------
        tuple of numpy.ndarray
            ``(t, xy)`` — distances (inf when no local intersection) and
            ``(M, 2)`` intersection points (NaN when missing).

        Notes
        -----
        Rays are answered from the ``knn`` nearest segment midpoints.
        Rays whose true crossing segment ranks beyond ``knn`` (rare;
        happens when many side segments crowd the neighbourhood, e.g.
        pole-ward rays from a cell end) are retried against **all**
        segments in a second pass, so a valid crossing is never missed.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        dirs = np.atleast_2d(np.asarray(directions, dtype=np.float64))
        t_best, xy = self._ray_pass(pts, dirs, knn, max_distance)
        miss = ~np.isfinite(t_best)
        if np.any(miss) and knn < self.n_segments:
            t_full, xy_full = self._ray_pass(
                pts[miss], dirs[miss], self.n_segments, max_distance
            )
            t_best[miss] = t_full
            xy[miss] = xy_full
        return t_best, xy

    def _ray_pass(
        self,
        pts: np.ndarray,
        dirs: np.ndarray,
        knn: int,
        max_distance: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single candidate-limited ray/intersection pass (see above)."""
        k = max(1, min(int(knn), self.n_segments))
        _, idx = self.tree.query(pts, k=k)
        idx = np.asarray(idx)
        if idx.ndim == 1:
            idx = idx[:, None]
        a = self.v1[idx]  # (M, k, 2)
        s = self.seg[idx]  # segment vectors (precomputed in __init__)
        # Numba inner solve (bit-identical to the former numpy block).
        t_best = _ray_solve_kernel(
            np.ascontiguousarray(a),
            np.ascontiguousarray(s),
            np.ascontiguousarray(pts[:, 0]),
            np.ascontiguousarray(pts[:, 1]),
            np.ascontiguousarray(dirs[:, 0]),
            np.ascontiguousarray(dirs[:, 1]),
            float(max_distance) if max_distance is not None else 0.0,
            max_distance is not None,
        )
        with np.errstate(invalid="ignore"):
            xy = pts + t_best[:, None] * dirs
        xy[~np.isfinite(t_best)] = np.nan
        return t_best, xy


# ---------------------------------------------------------------------------
# skeleton graph tracing (side-branch REPORTING only — never the main midline)
# ---------------------------------------------------------------------------


_NEIGHBORS8: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0),
    (1, 1),
)


def _classify_skeleton(skel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify skeleton pixels with ONE convolution.

    Returns
    -------
    tuple of numpy.ndarray
        ``(neighbor_count, endpoints, branchpoints)`` — neighbour count
        per pixel (3x3 ones kernel, center subtracted), boolean endpoint
        mask (count == 1) and boolean branchpoint mask (count > 2).
    """
    skel_i = skel.astype(np.int32)
    conv = ndimage.convolve(
        skel_i, np.ones((3, 3), dtype=np.int32), mode="constant", cval=0
    )
    ncount = conv - skel_i
    endpoints = skel & (ncount == 1)
    branchpoints = skel & (ncount > 2)
    return ncount, endpoints, branchpoints


def _trace_branches(
    skel: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    """Trace the skeleton into graph branches between nodes.

    Nodes are individual endpoint pixels, 8-connected clusters of
    branchpoint pixels, and dead ends reached during tracing. Each
    branch is the ordered pixel path between two nodes (node centroids
    included as first/last points). A skeleton forming a closed loop
    without nodes yields a single branch with ``nodes = (None, None)``.

    Returns
    -------
    branches : list of dict
        ``{"nodes": (a, b), "path": (P, 2) float (row, col)}``.
    node_centers : dict of int to numpy.ndarray
        ``node_id -> (row, col)`` float centroid.
    """
    _, endpoints, branchpoints = _classify_skeleton(skel)

    node_id = np.zeros(skel.shape, dtype=np.int32)
    node_centers: dict[int, np.ndarray] = {}
    nid = 0
    for r, c in np.argwhere(endpoints):
        nid += 1
        node_id[r, c] = nid
        node_centers[nid] = np.array([float(r), float(c)])
    bp_lab, n_bp = ndimage.label(branchpoints, structure=np.ones((3, 3), dtype=np.int32))
    for c in range(1, n_bp + 1):
        nid += 1
        pixels = np.argwhere(bp_lab == c)
        node_id[pixels[:, 0], pixels[:, 1]] = nid
        node_centers[nid] = pixels.mean(axis=0).astype(np.float64)

    # Node pixels of every node in ONE pass (grouped by id, row-major order
    # within a node — identical to per-node ``argwhere(node_id == nid)``).
    node_pixels_map: dict[int, np.ndarray] = {}
    if nid:
        pix_coords = np.argwhere(node_id > 0)
        pix_ids = node_id[node_id > 0]
        order = np.argsort(pix_ids, kind="stable")
        sorted_ids = pix_ids[order]
        sorted_coords = pix_coords[order]
        bounds = np.searchsorted(sorted_ids, np.arange(1, nid + 2))
        for node in range(1, nid + 1):
            node_pixels_map[node] = sorted_coords[bounds[node - 1] : bounds[node]]

    visited = np.zeros_like(skel, dtype=bool)
    branches: list[dict[str, Any]] = []
    recorded_pairs: set[frozenset[int]] = set()

    def skeleton_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for dr, dc in _NEIGHBORS8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1] and skel[rr, cc]:
                out.append((rr, cc))
        return out

    def add_branch(a: int | None, b: int | None, path: list[np.ndarray]) -> None:
        if a is not None and b is not None:
            pair = frozenset((a, b))
            if pair in recorded_pairs:
                return
            recorded_pairs.add(pair)
        if len(path) >= 2:
            branches.append(
                {"nodes": (a, b), "path": np.asarray(path, dtype=np.float64)}
            )

    for start_nid in sorted(node_centers):
        node_pixels = node_pixels_map[start_nid]
        for pix in node_pixels:
            for nb in skeleton_neighbors(int(pix[0]), int(pix[1])):
                nb_node = int(node_id[nb])
                if nb_node == start_nid:
                    continue
                if nb_node > 0:
                    add_branch(
                        start_nid,
                        nb_node,
                        [node_centers[start_nid], node_centers[nb_node]],
                    )
                    continue
                if visited[nb]:
                    continue
                # Walk from this node to the next node (or a dead end).
                path = [node_centers[start_nid]]
                prev = (int(pix[0]), int(pix[1]))
                cur = nb
                end_nid: int | None = None
                while True:
                    path.append(np.array([float(cur[0]), float(cur[1])]))
                    visited[cur] = True
                    nbs = skeleton_neighbors(*cur)
                    hits = sorted({int(node_id[q]) for q in nbs if node_id[q] > 0})
                    hits = [h for h in hits if h != start_nid or len(path) > 2]
                    if hits:
                        end_nid = hits[0]
                        path.append(node_centers[end_nid])
                        break
                    nxt = [q for q in nbs if q != prev and not visited[q]]
                    if not nxt:
                        # Dead end the classifier missed: register a leaf
                        # node so the branch remains graph-addressable.
                        nid += 1
                        node_centers[nid] = np.array([float(cur[0]), float(cur[1])])
                        node_id[cur] = nid
                        end_nid = nid
                        break
                    prev, cur = cur, nxt[0]
                add_branch(start_nid, end_nid, path)

    # Closed loops without any node (no endpoints/branchpoints at all).
    remaining = np.argwhere(skel & ~visited & (node_id == 0))
    if remaining.size and not branches:
        start = (int(remaining[0][0]), int(remaining[0][1]))
        path = [np.array([float(start[0]), float(start[1])])]
        prev = start
        nbs = skeleton_neighbors(*start)
        if nbs:
            cur = nbs[0]
            while cur != start:
                path.append(np.array([float(cur[0]), float(cur[1])]))
                visited[cur] = True
                nxt = [
                    q for q in skeleton_neighbors(*cur) if q != prev and not visited[q]
                ]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
        add_branch(None, None, path)
    return branches, node_centers


def _prune_branches(
    branches: list[dict[str, Any]], min_length_px: float
) -> list[dict[str, Any]]:
    """Iteratively remove short branches ending at a leaf node (spurs).

    Always keeps at least the single longest branch.
    """
    kept = list(branches)
    while len(kept) > 1:
        degree: dict[int, int] = {}
        for b in kept:
            for n in b["nodes"]:
                if n is not None:
                    degree[n] = degree.get(n, 0) + 1
        removed = False
        for i, b in enumerate(kept):
            if _arc_length_rc(b["path"]) >= min_length_px:
                continue
            a, c = b["nodes"]
            leaf_touch = (a is not None and degree.get(a, 0) == 1) or (
                c is not None and degree.get(c, 0) == 1
            )
            if leaf_touch:
                del kept[i]
                removed = True
                break
        if not removed:
            break
    return kept


def _main_branch_path(
    branches: list[dict[str, Any]], node_centers: dict[int, np.ndarray]
) -> tuple[np.ndarray, list[int]]:
    """Longest geodesic path of the skeleton graph (exact diameter over
    leaf pairs; leaf counts are tiny for single cells).

    Returns
    -------
    path : numpy.ndarray
        ``(P, 2)`` ``(row, col)`` polyline of the main branch.
    used : list of int
        Indices of the branches composing the main path.
    """
    loops = [i for i, b in enumerate(branches) if b["nodes"] == (None, None)]
    graph = nx.Graph()
    for i, b in enumerate(branches):
        if i in loops:
            continue
        a, c = b["nodes"]
        graph.add_edge(a, c, weight=_arc_length_rc(b["path"]), branch=i)

    best_path: np.ndarray | None = None
    best_used: list[int] = []
    best_length = -1.0

    if graph.number_of_nodes() > 0:
        leaves = [n for n in graph.nodes if graph.degree(n) <= 1]
        node_path: list[Any] | None = None
        length = -1.0
        if len(leaves) >= 2:
            for src in leaves:
                lengths = nx.single_source_dijkstra_path_length(
                    graph, src, weight="weight"
                )
                for dst in leaves:
                    if dst <= src:
                        continue
                    cand = lengths.get(dst)
                    if cand is not None and cand > length:
                        length = cand
                        node_path = [src, dst]
        if node_path is not None:
            node_path = nx.shortest_path(
                graph, node_path[0], node_path[1], weight="weight"
            )
        if node_path is not None and len(node_path) >= 2:
            polylines: list[np.ndarray] = []
            for u, v in zip(node_path[:-1], node_path[1:]):
                bi = int(graph.edges[u, v]["branch"])
                best_used.append(bi)
                path = branches[bi]["path"]
                center_u = node_centers.get(u)
                if center_u is not None and np.hypot(*(path[0] - center_u)) > np.hypot(
                    *(path[-1] - center_u)
                ):
                    path = path[::-1]
                if polylines:
                    path = path[1:]
                polylines.append(path)
            best_path = np.vstack(polylines)
            best_length = length

    for i in loops:
        length = _arc_length_rc(branches[i]["path"])
        if length > best_length:
            best_length = length
            best_path = branches[i]["path"]
            best_used = [i]

    if best_path is None:
        # Disconnected degenerate graph: take the single longest branch.
        longest = max(range(len(branches)), key=lambda i: _arc_length_rc(branches[i]["path"]))
        best_path = branches[longest]["path"]
        best_used = [longest]
    return best_path, best_used


# ---------------------------------------------------------------------------
# width profile (unchanged KDTree normal-intersection logic)
# ---------------------------------------------------------------------------


def _width_profile(
    midline: np.ndarray, index: _OutlineIndex, knn: int, pixel_microns: float
) -> np.ndarray:
    """Width at each midline point via normal intersections (microns).

    When one side misses locally the found side is doubled; fully missing
    entries are linearly interpolated from neighbours.
    """
    mid = np.asarray(midline, dtype=np.float64)
    normals = _normals(mid)
    # Batched ±normal query (identical to two separate calls).
    n = len(mid)
    d_all, _ = index.ray_intersections(
        np.vstack([mid, mid]), np.vstack([normals, -normals]), knn
    )
    d_pos, d_neg = d_all[:n], d_all[n:]
    pos_ok = np.isfinite(d_pos)
    neg_ok = np.isfinite(d_neg)
    width = np.full(len(mid), np.nan)
    both = pos_ok & neg_ok
    width[both] = d_pos[both] + d_neg[both]
    only = pos_ok ^ neg_ok
    width[only] = np.where(pos_ok[only], 2.0 * d_pos[only], 2.0 * d_neg[only])
    if np.all(np.isnan(width)):
        width[:] = 0.0
    elif np.any(np.isnan(width)):
        good = np.where(~np.isnan(width))[0]
        bad = np.where(np.isnan(width))[0]
        width[bad] = np.interp(bad, good, width[good])
    return width * float(pixel_microns)


# ---------------------------------------------------------------------------
# Step 1 — pole-first initialization
# ---------------------------------------------------------------------------


def _mask_major_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Second-moments major axis of a binary mask.

    Returns
    -------
    centroid_xy : numpy.ndarray
        ``(x, y)`` centroid in crop coordinates.
    axis_xy : numpy.ndarray
        Unit vector along the major axis (sign ambiguous).
    aspect_ratio : float
        ``sqrt(lambda_max / lambda_min)`` of the pixel covariance.
    """
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([ys.astype(np.float64), xs.astype(np.float64)])
    centroid_rc = pts.mean(axis=0)
    cov = np.cov((pts - centroid_rc).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_rc = eigvecs[:, int(np.argmax(eigvals))]
    lam_min = float(np.min(eigvals))
    ar = float(np.sqrt(np.max(eigvals) / lam_min)) if lam_min > 1e-12 else np.inf
    centroid_xy = np.array([centroid_rc[1], centroid_rc[0]])
    axis_xy = np.array([major_rc[1], major_rc[0]])
    norm = float(np.hypot(axis_xy[0], axis_xy[1]))
    if norm > 0:
        axis_xy = axis_xy / norm
    return centroid_xy, axis_xy, ar


def _find_pole_pair(
    ring_xy: np.ndarray,
    axis_xy: np.ndarray,
    max_candidates: int,
    axis_tol_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Outline pair with maximal separation, aligned with the major axis.

    All-pairs euclidean distance of (at most ``max_candidates``) outline
    points; if the farthest pair's direction deviates from the mask major
    axis by more than ``axis_tol_deg`` (modulo pi), the farthest pair
    among those within the tolerance is preferred.
    """
    pts = np.asarray(ring_xy, dtype=np.float64)
    if len(pts) > int(max_candidates):
        step = int(np.ceil(len(pts) / float(max_candidates)))
        pts = pts[::step]
    n = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    iu = np.triu_indices(n, k=1)
    d2 = dist2[iu]
    vecs = pts[iu[1]] - pts[iu[0]]

    cos_tol = float(np.cos(np.deg2rad(axis_tol_deg)))

    def pick(order_ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cand = np.where(order_ok, d2, -1.0)
        k = int(np.argmax(cand))
        return pts[iu[0][k]].copy(), pts[iu[1][k]].copy()

    k_max = int(np.argmax(d2))
    v = vecs[k_max]
    norm = float(np.hypot(v[0], v[1]))
    if norm > 0 and abs(float(np.dot(v / norm, axis_xy))) >= cos_tol:
        return pts[iu[0][k_max]].copy(), pts[iu[1][k_max]].copy()
    # Farthest pair misaligned with the major axis: restrict to aligned pairs.
    norms = np.hypot(vecs[:, 0], vecs[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_dev = np.abs(vecs @ axis_xy) / norms
    ok = (norms > 0) & (cos_dev >= cos_tol)
    if ok.any():
        return pick(ok)
    return pts[iu[0][k_max]].copy(), pts[iu[1][k_max]].copy()


# ---------------------------------------------------------------------------
# Step 2 — distance-transform geodesic centerline
# ---------------------------------------------------------------------------


def _geodesic_centerline(
    crop: np.ndarray, pole1_xy: np.ndarray, pole2_xy: np.ndarray, eps: float
) -> np.ndarray:
    """Shortest path between the poles over the mask pixel grid with cost
    ``1/(EDT + eps)`` (8-connected, geometric weights).

    The path is guaranteed to stay inside the mask (outside pixels carry
    a near-infinite cost) and is naturally centered on the local EDT
    ridge. Outline pole candidates are snapped to the nearest mask pixel
    and then walked uphill on the EDT to the local ridge (greedy
    8-neighbour ascent): on bent cells the max-separation outline pair
    sits around the pole-cap corner, and without the uphill snap the
    geodesic — and every downstream end tangent — would be anchored off
    the tip.

    Returns
    -------
    numpy.ndarray
        ``(P, 2)`` path in ``(x, y)`` crop coordinates.
    """
    dt = ndimage.distance_transform_edt(crop)
    cost = np.where(crop, 1.0 / (dt + float(eps)), 1e6)
    ys, xs = np.nonzero(crop)
    tree = cKDTree(np.column_stack([ys, xs]).astype(np.float64))

    def snap(p_xy: np.ndarray) -> tuple[int, int]:
        _, idx = tree.query([float(p_xy[1]), float(p_xy[0])])
        r, c = int(ys[idx]), int(xs[idx])
        # Greedy EDT uphill walk to the local ridge / plateau.
        for _ in range(64):
            best = (r, c)
            best_dt = dt[r, c]
            for dr, dc in _NEIGHBORS8:
                rr, cc = r + dr, c + dc
                if (
                    0 <= rr < crop.shape[0]
                    and 0 <= cc < crop.shape[1]
                    and crop[rr, cc]
                    and dt[rr, cc] > best_dt + 1e-9
                ):
                    best, best_dt = (rr, cc), dt[rr, cc]
            if best == (r, c):
                break
            r, c = best
        return r, c

    start, end = snap(pole1_xy), snap(pole2_xy)
    if start == end:
        return np.array([[start[1], start[0]], [end[1], end[0]]], dtype=np.float64)
    path, _ = route_through_array(
        cost, start, end, fully_connected=True, geometric=True
    )
    rc = np.asarray(path, dtype=np.float64)
    return np.column_stack([rc[:, 1], rc[:, 0]])


# ---------------------------------------------------------------------------
# Step 3 — low-DOF regularized spline fit
# ---------------------------------------------------------------------------


def _spline_design(u: np.ndarray, n_ctrl: int, degree: int = 3) -> np.ndarray:
    """Dense design matrix of a clamped uniform-knot cubic B-spline.

    ``(M, n_ctrl)`` basis evaluation at parameters ``u`` in ``[0, 1]``;
    clamped knots make the curve pass through the first/last control
    points.
    """
    n_ctrl = int(n_ctrl)
    interior = np.linspace(0.0, 1.0, n_ctrl - degree + 1)[1:-1]
    t = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
    uu = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0 - 1e-9)
    mat = interpolate.BSpline.design_matrix(uu, t, degree, extrapolate=False)
    return np.asarray(mat.todense())


def _second_difference(n_ctrl: int) -> np.ndarray:
    """``(n_ctrl - 2, n_ctrl)`` second finite-difference matrix."""
    d = np.zeros((n_ctrl - 2, n_ctrl), dtype=np.float64)
    for i in range(n_ctrl - 2):
        d[i, i] = 1.0
        d[i, i + 1] = -2.0
        d[i, i + 2] = 1.0
    return d


def _normal_midpoints(
    pts: np.ndarray, index: _OutlineIndex, knn: int, end_anchor: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Perpendicular midpoint targets and validity mask for a polyline.

    For every sample the two normal-ray outline intersections are
    computed (batched KDTree query); the target is their midpoint.
    Samples missing either side, and the first/last ``end_anchor``
    samples (pole-cap region, handled by pole extension), are invalid.
    """
    pts = np.asarray(pts, dtype=np.float64)
    normals = _normals(pts)
    n = len(pts)
    d_all, p_all = index.ray_intersections(
        np.vstack([pts, pts]), np.vstack([normals, -normals]), knn
    )
    d_pos, p_pos = d_all[:n], p_all[:n]
    d_neg, p_neg = d_all[n:], p_all[n:]
    valid = np.isfinite(d_pos) & np.isfinite(d_neg)
    a = max(0, min(int(end_anchor), (n - 1) // 2))
    if a > 0:
        valid[:a] = False
        valid[-a:] = False
    targets = np.where(valid[:, None], 0.5 * (p_pos + p_neg), pts)
    return targets, valid


def _residual_against_targets(
    pts: np.ndarray, targets: np.ndarray, valid: np.ndarray
) -> float:
    """Mean absolute normal-direction deviation from midpoint targets."""
    if not np.any(valid):
        return float("nan")
    normals = _normals(pts)
    dev = np.abs(np.einsum("ij,ij->i", pts - targets, normals))
    return float(np.mean(dev[valid]))


def _fit_lowdof_spline(
    geo_xy: np.ndarray,
    index: _OutlineIndex,
    knn: int,
    params: Mapping[str, Any],
    contains,
) -> dict[str, Any]:
    """Regularized low-DOF cubic B-spline fit to the outline midpoints.

    Unknowns are the ``(n_ctrl, 2)`` control points; the objective is

    - **data term**: at arc-length-uniform samples, the normal-direction
      deviation from the perpendicular outline-midpoint target (linear in
      the control points given a fixed parameterization);
    - **bending energy**: ``lambda_smooth * ||D2 C||^2`` on second finite
      differences of the control points — zigzag is structurally
      impossible with 5–9 control points plus this penalty;
    - **centrality**: ``mu_central``-weighted euclidean pull toward the
      geodesic centerline of Step 2;
    - **end anchors**: the first/last control points are pinned (weight
      ``anchor_weight``) to the geodesic endpoints — the bending energy
      alone under-determines the ends and they would otherwise slide
      outward past the pole caps. Reaching the outline is the job of the
      pole-extension step, not of the spline.

    Solved by :func:`numpy.linalg.lstsq` on a ``(2*n_ctrl,)`` vector;
    the parameterization is refreshed (arc-length resample, normal
    recompute, re-solve) at most ``spline_iterations`` times, stopping
    early when mean control-point movement < ``spline_tolerance`` px.
    Moves that push samples out of the mask are damped by repeated
    halving.
    """
    geo = np.asarray(geo_xy, dtype=np.float64)
    n_sample = len(geo)
    length_px = _arc_length(geo)
    n_ctrl = int(round(length_px / float(params["ctrl_spacing_px"])))
    n_ctrl = max(int(params["ctrl_min"]), min(int(params["ctrl_max"]), n_ctrl))
    n_ctrl = min(n_ctrl, n_sample)

    lam = float(params["lambda_smooth"])
    mu = float(params["mu_central"])
    max_iter = int(params["spline_iterations"])
    tol = float(params["spline_tolerance"])
    max_step = float(params["spline_max_step"])
    anchor_w = float(params.get("anchor_weight", 10.0))

    u = np.linspace(0.0, 1.0, n_sample)
    basis = _spline_design(u, n_ctrl)
    d2 = _second_difference(n_ctrl)

    # Block helper for the stacked [Cx; Cy] unknown vector.
    def block_diag2(m: np.ndarray) -> np.ndarray:
        out = np.zeros((2 * m.shape[0], 2 * m.shape[1]), dtype=np.float64)
        out[: m.shape[0], : m.shape[1]] = m
        out[m.shape[0] :, m.shape[1] :] = m
        return out

    bend_rows = np.sqrt(lam) * block_diag2(d2)
    cent_rows = np.sqrt(mu) * block_diag2(basis)
    cent_rhs = np.concatenate([np.sqrt(mu) * geo[:, 0], np.sqrt(mu) * geo[:, 1]])
    anchor_rows = np.zeros((4, 2 * n_ctrl), dtype=np.float64)
    anchor_rows[0, 0] = anchor_w
    anchor_rows[1, n_ctrl] = anchor_w
    anchor_rows[2, n_ctrl - 1] = anchor_w
    anchor_rows[3, 2 * n_ctrl - 1] = anchor_w
    anchor_rhs = anchor_w * np.array(
        [geo[0, 0], geo[0, 1], geo[-1, 0], geo[-1, 1]]
    )

    # Initialization: plain least-squares fit of the geodesic centerline.
    ctrl, *_ = np.linalg.lstsq(
        np.vstack([block_diag2(basis), bend_rows, anchor_rows]),
        np.concatenate(
            [geo[:, 0], geo[:, 1], np.zeros(2 * d2.shape[0]), anchor_rhs]
        ),
        rcond=None,
    )
    ctrl = ctrl.reshape(2, n_ctrl).T  # (n_ctrl, 2)

    converged = False
    n_iter = 0
    for _ in range(max_iter):
        n_iter += 1
        pts = basis @ ctrl
        targets, valid = _normal_midpoints(pts, index, knn)
        rows = [bend_rows, cent_rows, anchor_rows]
        rhs = [np.zeros(2 * d2.shape[0]), cent_rhs, anchor_rhs]
        if np.any(valid):
            normals = _normals(pts)
            nv = normals[valid]
            av = basis[valid]
            m_data = np.hstack([nv[:, 0:1] * av, nv[:, 1:2] * av])
            b_data = np.einsum("ij,ij->i", nv, targets[valid])
            rows.append(m_data)
            rhs.append(b_data)
        mat = np.vstack(rows)
        vec = np.concatenate(rhs)
        sol, *_ = np.linalg.lstsq(mat, vec, rcond=None)
        new_ctrl = sol.reshape(2, n_ctrl).T

        delta = new_ctrl - ctrl
        step = np.hypot(delta[:, 0], delta[:, 1])
        over = step > max_step
        if np.any(over):
            delta[over] *= (max_step / step[over])[:, None]
        # Damp the move if it pushes samples out of the mask.
        for _halve in range(4):
            cand = ctrl + delta
            cand_pts = basis @ cand
            if contains is None or np.all(contains(cand_pts)):
                ctrl = cand
                break
            delta = 0.5 * delta
        else:
            ctrl = ctrl + delta
        movement = float(np.mean(np.hypot(delta[:, 0], delta[:, 1])))

        # Re-parameterize: arc-length-uniform sample positions.
        dense_u = np.linspace(0.0, 1.0, 4 * n_sample)
        dense = _spline_design(dense_u, n_ctrl) @ ctrl
        seg = np.hypot(np.diff(dense[:, 0]), np.diff(dense[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] > 0:
            u = np.interp(np.linspace(0.0, s[-1], n_sample), s, dense_u)
            basis = _spline_design(u, n_ctrl)
            cent_rows = np.sqrt(mu) * block_diag2(basis)
        if movement <= tol:
            converged = True
            break

    pts = basis @ ctrl
    targets, valid = _normal_midpoints(pts, index, knn)
    residual = _residual_against_targets(pts, targets, valid)
    return {
        "points": pts,
        "ctrl": ctrl,
        "converged": converged,
        "n_iter": n_iter,
        "residual": residual,
    }


# ---------------------------------------------------------------------------
# Step 5 — septum-aware pole handling
# ---------------------------------------------------------------------------


def _end_tangent(points: np.ndarray, end: int, window_frac: float) -> np.ndarray:
    """Outward unit tangent at one end, fit on the distal ``window_frac``
    of arc samples only (junction-proximal asymmetry is ignored)."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    w = max(3, int(round(float(window_frac) * n)))
    w = min(w, n)
    seg = pts[:w] if end == 0 else pts[-w:]
    center = seg.mean(axis=0)
    cov = np.cov((seg - center).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]
    outward = seg[0] - seg[-1] if end == 0 else seg[-1] - seg[0]
    if float(np.dot(v, outward)) < 0:
        v = -v
    norm = float(np.hypot(v[0], v[1]))
    return v / norm if norm > 0 else np.array([1.0, 0.0])


def _end_width_dip(width: np.ndarray, end: int, dip_frac: float) -> bool:
    """True when the width profile has a genuine constriction in an end
    third (septum / chain-junction crevice; detect_septum-style logic).

    The outermost two samples (the pole-cap taper, where width
    monotonically narrows toward the tip) are excluded: a true septum dip
    is an *interior* minimum of the end third, significantly below both
    the middle-body median and the end-face width.
    """
    w = np.asarray(width, dtype=np.float64)
    n = len(w)
    if n < 8:
        return False
    third = max(1, n // 3)
    middle = w[third:-third] if n > 2 * third else w
    med = float(np.median(middle))
    if med <= 0:
        return False
    region = w[2:third] if end == 0 else w[n - third : n - 2]
    if region.size < 2:
        return False
    end_face = float(w[2] if end == 0 else w[n - 3])
    dip = float(np.min(region))
    return dip < (1.0 - float(dip_frac)) * med and dip < 0.9 * end_face


def _extend_poles(
    midline: np.ndarray,
    tangents: tuple[np.ndarray, np.ndarray],
    index: _OutlineIndex,
    knn: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend both midline ends along the given outward end tangents until
    outline intersection (MicrobeJ/Oufti-style pole extension); falls back
    to the midline endpoints when no local intersection is found."""
    mid = np.asarray(midline, dtype=np.float64)
    ends = (mid[0], mid[-1])
    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    slots: list[int] = []
    for slot, (end, direction) in enumerate(zip(ends, tangents)):
        norm = float(np.hypot(direction[0], direction[1]))
        if norm <= 0:
            continue
        origins.append(end)
        directions.append(direction / norm)
        slots.append(slot)
    poles = [end.copy() for end in ends]
    if origins:
        _, hits = index.ray_intersections(
            np.asarray(origins), np.asarray(directions), knn
        )
        for slot, hit in zip(slots, hits):
            if np.isfinite(hit).all():
                poles[slot] = hit
    return poles[0], poles[1]


# ---------------------------------------------------------------------------
# per-label results assembly
# ---------------------------------------------------------------------------


def _width_cv(width_profile: np.ndarray) -> float:
    """Coefficient of variation of the width profile over INTERIOR samples.

    The first/last samples sit on the poles (outline points), where the
    normal-ray width is near zero by construction; including them would
    make the CV measure cap taper rather than body-width variability.
    """
    w = np.asarray(width_profile, dtype=np.float64)
    if w.size > 2:
        w = w[1:-1]
    mean = float(np.mean(w)) if w.size else 0.0
    return float(np.std(w) / mean) if mean > 0 else float("nan")


def _empty_result(params: Mapping[str, Any], quality: str) -> dict[str, Any]:
    """Empty-midline result for failed/pathological cells (never garbage)."""
    nan2 = np.array([np.nan, np.nan])
    return {
        "midline": [np.zeros((0, 2), dtype=np.float64)],
        "width_profile": np.zeros(int(params["n_points"]), dtype=np.float64),
        "pole1": nan2,
        "pole2": nan2,
        "length_um": 0.0,
        "sinuosity": 1.0,
        "branched": 0.0,
        "midline_quality": quality,
        "n_iter": 0,
        "final_residual": float("nan"),
        "max_curvature": float("nan"),
        "width_cv": float("nan"),
    }


def _assemble_result(
    curve_xy: np.ndarray,
    index: _OutlineIndex,
    params: Mapping[str, Any],
    pixel_microns: float,
    quality: str,
    n_iter: int,
    final_residual: float,
    side_branches: list[np.ndarray] | None = None,
    geo_xy: np.ndarray | None = None,
) -> dict[str, Any]:
    """Septum-aware poles + final resample + width profile + diagnostics."""
    knn = int(params["knn"])
    n_points = int(params["n_points"])
    curve = _resample_arclength(np.asarray(curve_xy, dtype=np.float64), n_points)

    # Septum-aware tangent choice: width dip in an end third -> trust the
    # geodesic centerline's end direction over the data-term tangent.
    width_pre = _width_profile(curve, index, knn, pixel_microns)
    tangents = []
    for end in (0, 1):
        src = curve
        if geo_xy is not None and _end_width_dip(
            width_pre, end, float(params["septum_dip_frac"])
        ):
            src = _resample_arclength(np.asarray(geo_xy, dtype=np.float64), n_points)
        tangents.append(_end_tangent(src, end, float(params["end_window_frac"])))
    pole1, pole2 = _extend_poles(curve, (tangents[0], tangents[1]), index, knn)

    extended = np.vstack([pole1[None, :], curve, pole2[None, :]])
    main_resampled = _resample_arclength(extended, n_points)
    width_profile = _width_profile(main_resampled, index, knn, pixel_microns)
    length_px = _arc_length(main_resampled)
    pole_dist = float(np.hypot(*(pole2 - pole1)))
    sinuosity = length_px / pole_dist if pole_dist > 0 else 1.0

    _, max_curv = _curvature_stats(main_resampled)
    width_cv = _width_cv(width_profile)

    branches = [main_resampled, *(side_branches or [])]
    return {
        "midline": branches,
        "width_profile": width_profile,
        "pole1": pole1,
        "pole2": pole2,
        "length_um": length_px * float(pixel_microns),
        "sinuosity": float(sinuosity),
        "branched": 1.0 if side_branches else 0.0,
        "midline_quality": quality,
        "n_iter": int(n_iter),
        "final_residual": float(final_residual),
        "max_curvature": float(max_curv),
        "width_cv": float(width_cv),
    }


def _axis_segment_result(
    crop: np.ndarray,
    offset_xy: np.ndarray,
    centroid_xy: np.ndarray,
    axis_xy: np.ndarray,
    index: _OutlineIndex,
    params: Mapping[str, Any],
    pixel_microns: float,
) -> dict[str, Any]:
    """Degenerate (near-round) cell: major-axis segment clipped to the
    outline, NO geodesic/spline refinement."""
    knn = int(params["knn"])
    centroid_global = centroid_xy + offset_xy
    _, p1 = index.ray_intersections(centroid_global[None, :], axis_xy[None, :], knn)
    _, p2 = index.ray_intersections(centroid_global[None, :], -axis_xy[None, :], knn)
    pole1 = p1[0] if np.isfinite(p1[0]).all() else centroid_global + axis_xy
    pole2 = p2[0] if np.isfinite(p2[0]).all() else centroid_global - axis_xy
    main = _resample_arclength(np.vstack([pole1, pole2]), int(params["n_points"]))
    width = _width_profile(main, index, knn, pixel_microns)
    length_px = _arc_length(main)
    return {
        "midline": [main],
        "width_profile": width,
        "pole1": pole1,
        "pole2": pole2,
        "length_um": length_px * float(pixel_microns),
        "sinuosity": 1.0,
        "branched": 0.0,
        "midline_quality": "degenerate",
        "n_iter": 0,
        "final_residual": 0.0,
        "max_curvature": 0.0,
        "width_cv": _width_cv(width),
    }


def _skeleton_side_branches(
    crop: np.ndarray,
    offset_xy: np.ndarray,
    params: Mapping[str, Any],
    pixel_microns: float,
) -> list[np.ndarray]:
    """Skeleton-based side-branch detection for branch REPORTING only.

    The main midline never comes from the skeleton; the skeleton graph is
    traced solely to enumerate residual arms of branched masks. Any
    failure yields no side branches.
    """
    min_branch_px = max(
        2.0, float(params["min_branch_length_um"]) / float(pixel_microns)
    )
    try:
        skel = morphology.skeletonize(crop, method=str(params["method"]))
        if int(skel.sum()) < 3:
            return []
        branches, node_centers = _trace_branches(skel)
        if not branches:
            return []
        branches = _prune_branches(branches, min_branch_px)
        _, used = _main_branch_path(branches, node_centers)
        side: list[np.ndarray] = []
        for i, b in enumerate(branches):
            if i in used:
                continue
            path_xy = np.column_stack([b["path"][:, 1], b["path"][:, 0]]) + offset_xy
            side.append(_smooth_open(path_xy, float(params["smooth"])))
        return side
    except Exception as exc:  # branch reporting must never fail the cell
        logger.debug("skeleton side-branch detection failed (%s)", exc)
        return []


# ---------------------------------------------------------------------------
# per-label driver
# ---------------------------------------------------------------------------


def _midline_single(
    crop: np.ndarray,
    offset_xy: np.ndarray,
    outline: np.ndarray,
    params: Mapping[str, Any],
    pixel_microns: float,
) -> dict[str, Any] | None:
    """Full pole-first midline pipeline for one cropped instance mask
    (global coords out). See the module docstring for the algorithm."""
    ys, xs = np.nonzero(crop)
    if ys.size < 4:
        return None

    index = _OutlineIndex(np.asarray(outline, dtype=np.float64))
    knn = int(params["knn"])

    try:
        # Pathological topology (holes, e.g. rings/tori): cells never have
        # holes — flag as failed rather than emitting garbage geometry.
        filled = ndimage.binary_fill_holes(crop)
        hole_px = int(filled.sum()) - int(crop.sum())
        if hole_px > max(4.0, float(params["hole_area_frac"]) * float(crop.sum())):
            return _empty_result(params, "failed")

        centroid_xy, axis_xy, aspect_ratio = _mask_major_axis(crop)

        # Degenerate near-round cells: major-axis segment, no refinement.
        if aspect_ratio < float(params["degenerate_ar"]):
            return _axis_segment_result(
                crop, offset_xy, centroid_xy, axis_xy, index, params, pixel_microns
            )

        outline_crop = np.asarray(outline, dtype=np.float64) - offset_xy

        # Step 1 — pole-first initialization.
        pole1_xy, pole2_xy = _find_pole_pair(
            outline_crop,
            axis_xy,
            int(params["pole_max_candidates"]),
            float(params["pole_axis_tol_deg"]),
        )

        # Step 2 — distance-transform geodesic centerline.
        geo_crop = _geodesic_centerline(
            crop, pole1_xy, pole2_xy, float(params["edt_eps"])
        )
        geo_xy = geo_crop + offset_xy

        def _in_mask(points: np.ndarray) -> np.ndarray:
            """True for points whose rounded pixel lies inside the crop."""
            cols = np.round(points[:, 0] - offset_xy[0]).astype(int)
            rows = np.round(points[:, 1] - offset_xy[1]).astype(int)
            ok = (
                (rows >= 0)
                & (rows < crop.shape[0])
                & (cols >= 0)
                & (cols < crop.shape[1])
            )
            out = np.zeros(len(points), dtype=bool)
            out[ok] = crop[rows[ok], cols[ok]]
            return out

        # Light smoothing of the geodesic path; escaped samples (smoothing
        # may cut corners) are snapped back onto the raw path.
        n_sample = max(int(params["n_points"]), int(round(_arc_length(geo_xy))), 8)
        geo_raw = _resample_arclength(geo_xy, n_sample)
        geo_smooth = _smooth_open(geo_xy, float(params["smooth"]), n=n_sample)
        escaped = ~_in_mask(geo_smooth)
        if np.any(escaped):
            geo_smooth[escaped] = geo_raw[escaped]

        # Step 3 — low-DOF regularized spline fit.
        fit = _fit_lowdof_spline(geo_smooth, index, knn, params, _in_mask)

        # Step 4 — convergence cascade. Keep the spline only when it
        # improves on the geodesic centerline's midpoint residual.
        geo_targets, geo_valid = _normal_midpoints(geo_smooth, index, knn)
        geo_residual = _residual_against_targets(geo_smooth, geo_targets, geo_valid)
        if np.isfinite(fit["residual"]) and (
            not np.isfinite(geo_residual) or fit["residual"] <= geo_residual
        ):
            curve = fit["points"]
            quality = "refined"
            final_residual = fit["residual"]
        else:
            curve = geo_smooth
            quality = "geodesic"
            final_residual = geo_residual

        # Branched masks: skeleton-based side-branch REPORTING only.
        side_branches = _skeleton_side_branches(
            crop, offset_xy, params, pixel_microns
        )

        # Steps 5+6 — septum-aware poles, width profile, diagnostics.
        return _assemble_result(
            curve,
            index,
            params,
            pixel_microns,
            quality,
            fit["n_iter"] if quality == "refined" else 0,
            final_residual,
            side_branches,
            geo_smooth,
        )
    except Exception as exc:
        logger.debug("midline pipeline failed (%s); flagging cell as failed", exc)
        return _empty_result(params, "failed")


def extract_midlines(
    labels_2d: np.ndarray,
    outlines: Mapping[int, np.ndarray] | None = None,
    params: Mapping[str, Any] | None = None,
    pixel_microns: float = 0.065,
) -> dict[int, dict[str, Any]]:
    """Extract midlines, poles, width profiles and scalars per label.

    Parameters
    ----------
    labels_2d : numpy.ndarray of int
        2-D instance label image (0 = background).
    outlines : mapping of int to numpy.ndarray, optional
        ``label -> (N, 2)`` closed outline in global ``(x, y)`` coords
        (e.g. from :func:`momia3.ops.outline.extract_outlines`). When
        None, outlines are computed internally with default parameters.
    params : mapping, optional
        ``method`` (``'zhang'`` | ``'lee'``) and ``min_branch_length_um``
        steer the skeleton-based side-branch reporting only; ``n_points``
        (resampled main-branch size), ``smooth`` (spline smoothing),
        ``knn`` (outline segment candidates per query), ``pad`` (bbox
        padding), ``pole_max_candidates``/``pole_axis_tol_deg`` (pole
        search), ``edt_eps``/``hole_area_frac`` (geodesic centerline),
        ``ctrl_spacing_px``/``ctrl_min``/``ctrl_max`` (spline DOF),
        ``lambda_smooth`` (bending-energy weight), ``mu_central``
        (geodesic centrality weight), ``spline_iterations``/
        ``spline_tolerance``/``spline_max_step`` (fit loop),
        ``end_window_frac``/``septum_dip_frac`` (pole handling),
        ``degenerate_ar`` (near-round threshold).
    pixel_microns : float
        Physical pixel size in microns.

    Returns
    -------
    dict of int to dict
        Per label: ``midline`` (list of ``(M, 2)`` branches in global
        ``(x, y)``; branch 0 is the resampled main branch incl. poles),
        ``width_profile`` (``(n_points,)`` microns), ``pole1``/``pole2``
        (``(x, y)``), scalars ``length_um``, ``width_median_um``,
        ``width_std_um``, ``width_max_um``, ``width_q1_um``,
        ``width_q3_um``, ``sinuosity``, ``branched``, the quality flag
        ``midline_quality`` (``"refined"`` | ``"geodesic"`` |
        ``"degenerate"`` | ``"failed"``) and diagnostics ``n_iter``,
        ``final_residual``, ``max_curvature``, ``width_cv``.
    """
    p: dict[str, Any] = {**_DEFAULT_PARAMS, **dict(params or {})}
    labels = np.asarray(labels_2d)
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64)
    if outlines is None:
        from momia3.ops.outline import extract_outlines

        outlines = extract_outlines(labels_2d, None, {"n_points": None})
    results: dict[int, dict[str, Any]] = {}
    pad = int(p["pad"])
    img_h, img_w = labels.shape
    # ONE bounding-box pass for the whole frame (see extract_outlines): all
    # per-label work then crops to ``bbox + pad`` instead of re-scanning the
    # full labels image per label.
    slices = ndimage.find_objects(labels)
    for lab, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        outline = outlines.get(lab)
        if outline is None or len(outline) < 3:
            logger.debug("label %d: no usable outline; skipped", lab)
            continue
        y0 = max(sl[0].start - pad, 0)
        y1 = min(sl[0].stop + pad, img_h)
        x0 = max(sl[1].start - pad, 0)
        x1 = min(sl[1].stop + pad, img_w)
        crop = labels[y0:y1, x0:x1] == lab
        offset_xy = np.array([float(x0), float(y0)])
        result = _midline_single(crop, offset_xy, outline, p, float(pixel_microns))
        if result is None:
            continue
        width = result["width_profile"]
        q1, q3 = np.percentile(width, [25.0, 75.0])
        result.update(
            {
                "width_median_um": float(np.median(width)),
                "width_std_um": float(np.std(width)),
                "width_max_um": float(np.max(width)),
                "width_q1_um": float(q1),
                "width_q3_um": float(q3),
                "pole1_x": float(result["pole1"][0]),
                "pole1_y": float(result["pole1"][1]),
                "pole2_x": float(result["pole2"][0]),
                "pole2_y": float(result["pole2"][1]),
            }
        )
        results[lab] = result
    return results


# ---------------------------------------------------------------------------
# table writer
# ---------------------------------------------------------------------------


def _to_pa(table: pa.Table | pl.DataFrame) -> pa.Table:
    """Coerce a polars DataFrame or Arrow table to ``pa.Table``."""
    if isinstance(table, pa.Table):
        return table
    if isinstance(table, pl.DataFrame):
        return table.to_arrow()
    raise TypeError(f"expected pyarrow.Table or polars.DataFrame, got {type(table).__name__}")


def _back(table: pa.Table, like: pa.Table | pl.DataFrame) -> pa.Table | pl.DataFrame:
    """Return ``table`` in the same container type as ``like``."""
    return pl.from_arrow(table) if isinstance(like, pl.DataFrame) else table


def _upsert_column(table: pa.Table, name: str, arr: pa.Array) -> pa.Table:
    """Replace or append a column, keeping canonical schema metadata."""
    canonical = _schema.cells_schema()
    field = canonical.field(name) if name in canonical.names else pa.field(name, arr.type)
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), field, arr)
    return table.append_column(field, arr)


def fill_cells_midline(
    cells_table: pa.Table | pl.DataFrame,
    results: Mapping[int, Mapping[str, Any]],
) -> pa.Table | pl.DataFrame:
    """Write midline geometry, scalars and diagnostics into a cells table.

    Parameters
    ----------
    cells_table : pyarrow.Table or polars.DataFrame
        Cells table (schema §2.3).
    results : mapping of int to mapping
        Output of :func:`extract_midlines`, keyed by ``label`` or by
        ``cell_key`` (inferred like in
        :func:`momia3.ops.outline.fill_cells_geometry`).

    Returns
    -------
    pyarrow.Table or polars.DataFrame
        Updated table in the same container type as the input, with
        ``midline``, ``width_profile``, ``pole1_x/y``, ``pole2_x/y``,
        ``length_um``, ``width_*_um``, ``sinuosity`` and ``branched``
        columns written, plus the ``midline_quality`` (utf8) flag and
        ``midline_n_iter`` (i32), ``midline_residual`` (f32),
        ``midline_max_curv`` (f32) and ``width_cv`` (f32) diagnostics
        (rows without a result keep nulls). Extra columns beyond the
        canonical schema pass :func:`momia3.schema.validate_cells`.
    """
    table = _to_pa(cells_table)
    by_key = any(int(k) > _keys.LABEL_MAX for k in results)
    row_ids = table.column("cell_key" if by_key else "label").to_pylist()

    midline_vals: list[list[dict[str, list[float]]] | None] = []
    width_vals: list[list[float] | None] = []
    scalar_vals: dict[str, list[float | None]] = {c: [] for c in MIDLINE_SCALAR_COLUMNS}
    quality_vals: list[str | None] = []
    diag_vals: dict[str, list[Any]] = {c: [] for c in MIDLINE_DIAG_COLUMNS}
    for rid in row_ids:
        res = results.get(rid)
        if res is None:
            midline_vals.append(None)
            width_vals.append(None)
            for c in MIDLINE_SCALAR_COLUMNS:
                scalar_vals[c].append(None)
            quality_vals.append(None)
            for c in MIDLINE_DIAG_COLUMNS:
                diag_vals[c].append(None)
            continue
        branches = [
            {
                "x": np.asarray(b, dtype=np.float64)[:, 0].astype(np.float32).tolist(),
                "y": np.asarray(b, dtype=np.float64)[:, 1].astype(np.float32).tolist(),
            }
            for b in res["midline"]
        ]
        midline_vals.append(branches)
        width_vals.append(np.asarray(res["width_profile"], dtype=np.float32).tolist())
        for c in MIDLINE_SCALAR_COLUMNS:
            value = res.get(c)
            scalar_vals[c].append(None if value is None else float(value))
        quality = res.get(MIDLINE_QUALITY_COLUMN)
        quality_vals.append(None if quality is None else str(quality))
        for column, key in MIDLINE_DIAG_COLUMNS.items():
            value = res.get(key)
            diag_vals[column].append(None if value is None else value)

    table = _upsert_column(table, "midline", pa.array(midline_vals, type=_schema.MIDLINE_T))
    table = _upsert_column(
        table, "width_profile", pa.array(width_vals, type=_schema.WIDTH_PROFILE_T)
    )
    for c in MIDLINE_SCALAR_COLUMNS:
        table = _upsert_column(table, c, pa.array(scalar_vals[c], type=pa.float32()))
    table = _upsert_column(
        table, MIDLINE_QUALITY_COLUMN, pa.array(quality_vals, type=pa.string())
    )
    _diag_types = {"midline_n_iter": pa.int32()}
    for c in MIDLINE_DIAG_COLUMNS:
        arrow_type = _diag_types.get(c, pa.float32())
        vals = [
            None if v is None else (int(v) if arrow_type == pa.int32() else float(v))
            for v in diag_vals[c]
        ]
        table = _upsert_column(table, c, pa.array(vals, type=arrow_type))
    return _back(table, cells_table)
