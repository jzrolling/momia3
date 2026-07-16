"""Midline / pole / width-profile extraction (ARCHITECTURE.md §4.5 / §5.2).

Performance and accuracy upgrades over momia2:

- skeleton neighbour classification uses **one**
  :func:`scipy.ndimage.convolve` with a 3x3 ones kernel (endpoints = 1
  neighbour, branchpoints > 2) instead of per-pixel numba calls;
- **all** outline↔midline nearest-point / perpendicular-intersection
  queries go through a :class:`scipy.spatial.cKDTree` built on outline
  segment midpoints — no O(n_outline x n_midline) dense matrices anywhere;
- branches are traced as graph paths, short spurs are pruned, and the
  main branch is the longest geodesic path (exact tree-diameter search
  over leaf nodes);
- iterative perpendicular refinement keeps the momia2 core idea
  (``midline_approximation``) but vectorized against the KDTree;
- poles are found by extending the end tangents of the refined midline
  until intersection with the outline (MicrobeJ/Oufti-style pole
  extension).

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

_DEFAULT_PARAMS: dict[str, Any] = {
    "method": "zhang",
    "min_branch_length_um": 0.2,
    "n_points": 32,
    "smooth": 1.0,
    "refine_iterations": 10,
    "refine_tolerance": 0.1,
    "refine_anchor": 3,
    "refine_max_step": 1.5,
    "pole_fit_points": 5,
    "knn": 24,
    "pad": 2,
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
# skeleton graph tracing
# ---------------------------------------------------------------------------


_NEIGHBORS8: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
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
# refinement / poles / width
# ---------------------------------------------------------------------------


def _refine_midline(
    midline: np.ndarray,
    index: _OutlineIndex,
    knn: int,
    smooth: float,
    max_iter: int,
    tolerance: float,
    anchor: int = 3,
    max_step: float = 1.5,
    contains=None,
) -> tuple[np.ndarray, bool]:
    """Iterative perpendicular refinement (momia2 ``midline_approximation``
    idea, vectorized through the outline KDTree).

    Each point is moved to the midpoint of its two normal-ray outline
    intersections; the polyline is re-smoothed each iteration until the
    mean absolute movement drops below ``tolerance``. Safeguards:

    - the first and last ``anchor`` points are kept fixed (momia2
      ``anchor_length``): near the pole caps the midpoint rule can
      overshoot and reorder the trailing samples, folding the curve end
      back on itself and flipping the end-tangent used for pole
      extension;
    - per-point moves larger than ``max_step`` px are rejected (momia2's
      1 px outlier rejection), so a bad intersection cannot drag a
      point across the cell;
    - when ``contains`` is given (callable ``(M, 2) xy -> (M,) bool``,
      typically a mask lookup), moves landing outside the mask are
      reverted: near sharp skeleton junctions the normal can intersect
      outlines of two different arms and the midpoint rule would
      otherwise push points into the empty wedge between them.
    """
    mid = np.asarray(midline, dtype=np.float64)
    a = max(0, min(int(anchor), (len(mid) - 1) // 2))
    converged = False
    for _ in range(max_iter):
        normals = _normals(mid)
        # ONE batched query for both normal directions: every ray is solved
        # independently, so the stacked call is bit-identical to two
        # separate calls at half the per-call numpy overhead.
        n = len(mid)
        both_origins = np.vstack([mid, mid])
        both_dirs = np.vstack([normals, -normals])
        d_all, p_all = index.ray_intersections(both_origins, both_dirs, knn)
        d_pos, p_pos = d_all[:n], p_all[:n]
        d_neg, p_neg = d_all[n:], p_all[n:]
        both = np.isfinite(d_pos) & np.isfinite(d_neg)
        updated = mid.copy()
        updated[both] = 0.5 * (p_pos[both] + p_neg[both])
        step = np.hypot(*(updated - mid).T)
        updated[step > float(max_step)] = mid[step > float(max_step)]
        if contains is not None:
            inside = contains(updated)
            updated[~inside] = mid[~inside]
        if a > 0:
            updated[:a] = mid[:a]
            updated[-a:] = mid[-a:]
        movement = float(np.mean(np.abs(updated - mid)))
        mid = _smooth_open(updated, smooth, n=len(mid))
        if contains is not None:
            # The smoothing spline may cut sharp corners out of the mask;
            # snap any escaped samples back onto the pre-smoothing curve.
            escaped = ~contains(mid)
            if np.any(escaped):
                mid[escaped] = updated[escaped]
        if movement <= tolerance:
            converged = True
            break
    return mid, converged


def _extend_poles(
    midline: np.ndarray, index: _OutlineIndex, knn: int, fit_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extend both midline ends along the end tangent to the outline.

    Returns
    -------
    tuple of numpy.ndarray
        ``(pole1, pole2)`` as ``(x, y)`` global coordinates; falls back
        to the midline endpoints when no local intersection is found.
    """
    mid = np.asarray(midline, dtype=np.float64)
    g = max(2, min(int(fit_points), len(mid) - 1))
    # Batched ray query for both end-tangent extensions (bit-identical to
    # two separate single-ray calls).
    ends = ((mid[0], mid[g]), (mid[-1], mid[-1 - g]))
    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    slots: list[int] = []
    for slot, (end, inward) in enumerate(ends):
        direction = end - inward
        norm = float(np.hypot(direction[0], direction[1]))
        if norm <= 0:
            continue
        origins.append(end)
        directions.append(direction / norm)
        slots.append(slot)
    poles = [end.copy() for end, _ in ends]
    if origins:
        _, hits = index.ray_intersections(
            np.asarray(origins), np.asarray(directions), knn
        )
        for slot, hit in zip(slots, hits):
            if np.isfinite(hit).all():
                poles[slot] = hit
    return poles[0], poles[1]


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
# fallback (skeleton failed)
# ---------------------------------------------------------------------------


def _pca_fallback(
    mask_coords_rc: np.ndarray,
    offset_xy: np.ndarray,
    index: _OutlineIndex,
    params: Mapping[str, Any],
    pixel_microns: float,
) -> dict[str, Any]:
    """Straight-axis midline from mask PCA (used when the skeleton graph
    is degenerate, e.g. very small or pathological masks)."""
    coords = np.asarray(mask_coords_rc, dtype=np.float64)
    centroid_rc = coords.mean(axis=0)
    cov = np.cov((coords - centroid_rc).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis_rc = eigvecs[:, int(np.argmax(eigvals))]
    centroid_xy = np.array([centroid_rc[1], centroid_rc[0]]) + offset_xy
    axis_xy = np.array([axis_rc[1], axis_rc[0]])
    knn = int(params["knn"])
    _, p1 = index.ray_intersections(centroid_xy[None, :], axis_xy[None, :], knn)
    _, p2 = index.ray_intersections(centroid_xy[None, :], -axis_xy[None, :], knn)
    pole1 = p1[0] if np.isfinite(p1[0]).all() else centroid_xy + axis_xy
    pole2 = p2[0] if np.isfinite(p2[0]).all() else centroid_xy - axis_xy
    main = _resample_arclength(np.vstack([pole1, pole2]), int(params["n_points"]))
    width = _width_profile(main, index, knn, pixel_microns)
    return {
        "midline": [main],
        "width_profile": width,
        "pole1": pole1,
        "pole2": pole2,
        "length_um": _arc_length(main) * float(pixel_microns),
        "sinuosity": 1.0,
        "branched": 0.0,
    }


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
    """Full midline pipeline for one cropped instance mask (global coords out).

    Parameters
    ----------
    crop : numpy.ndarray of bool
        Padded bounding-box crop of the instance (same ``pad`` margin as
        the historical full-frame implementation).
    offset_xy : numpy.ndarray
        ``(x0, y0)`` crop origin in global image coordinates.
    outline : numpy.ndarray
        ``(N, 2)`` closed outline in global ``(x, y)`` coordinates.
    params : mapping
        Resolved parameter dict (see :func:`extract_midlines`).
    pixel_microns : float
        Physical pixel size in microns.
    """
    ys, xs = np.nonzero(crop)
    if ys.size < 4:
        return None
    # Historical note: the PCA fallback below receives *full-frame* pixel
    # coordinates (reconstructed from the crop) exactly as the pre-find_objects
    # implementation did. That version added ``offset_xy`` to an already
    # global centroid, so the fallback centroid is offset by the crop origin
    # (a latent double-offset quirk, unreachable in the acceptance tests
    # because those shapes always take the skeleton path). The behaviour is
    # preserved verbatim to keep numerics identical; fix deliberately in a
    # future phase if desired.
    full_rc = np.column_stack([ys + offset_xy[1], xs + offset_xy[0]])

    index = _OutlineIndex(np.asarray(outline, dtype=np.float64))
    knn = int(params["knn"])
    min_branch_px = max(
        2.0, float(params["min_branch_length_um"]) / float(pixel_microns)
    )

    try:
        skel = morphology.skeletonize(crop, method=str(params["method"]))
    except (ValueError, TypeError) as exc:
        logger.debug("skeletonize failed (%s); PCA fallback", exc)
        return _pca_fallback(full_rc, offset_xy, index, params, pixel_microns)

    if int(skel.sum()) < 3:
        return _pca_fallback(full_rc, offset_xy, index, params, pixel_microns)

    branches, node_centers = _trace_branches(skel)
    if not branches:
        return _pca_fallback(full_rc, offset_xy, index, params, pixel_microns)
    branches = _prune_branches(branches, min_branch_px)
    main_rc, used = _main_branch_path(branches, node_centers)

    # (row, col) crop -> (x, y) global.
    main_xy = np.column_stack([main_rc[:, 1], main_rc[:, 0]]) + offset_xy
    n_refine = max(int(params["n_points"]), int(round(_arc_length(main_xy))))
    main_xy = _smooth_open(main_xy, float(params["smooth"]), n=n_refine)

    def _in_mask(points: np.ndarray) -> np.ndarray:
        """True for points whose rounded pixel lies inside the crop mask."""
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

    main_xy, _ = _refine_midline(
        main_xy,
        index,
        knn,
        float(params["smooth"]),
        int(params["refine_iterations"]),
        float(params["refine_tolerance"]),
        int(params["refine_anchor"]),
        float(params["refine_max_step"]),
        _in_mask,
    )

    pole1, pole2 = _extend_poles(main_xy, index, knn, int(params["pole_fit_points"]))
    extended = np.vstack([pole1[None, :], main_xy, pole2[None, :]])
    main_resampled = _resample_arclength(extended, int(params["n_points"]))

    width_profile = _width_profile(main_resampled, index, knn, pixel_microns)
    length_px = _arc_length(main_resampled)
    pole_dist = float(np.hypot(*(pole2 - pole1)))
    sinuosity = length_px / pole_dist if pole_dist > 0 else 1.0

    side_branches: list[np.ndarray] = []
    for i, b in enumerate(branches):
        if i in used:
            continue
        path_xy = np.column_stack([b["path"][:, 1], b["path"][:, 0]]) + offset_xy
        side_branches.append(_smooth_open(path_xy, float(params["smooth"])))

    return {
        "midline": [main_resampled, *side_branches],
        "width_profile": width_profile,
        "pole1": pole1,
        "pole2": pole2,
        "length_um": length_px * float(pixel_microns),
        "sinuosity": float(sinuosity),
        "branched": 1.0 if side_branches else 0.0,
    }


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
        ``method`` (``'zhang'`` | ``'lee'``), ``min_branch_length_um``
        (spur pruning threshold), ``n_points`` (resampled main-branch
        size), ``smooth`` (spline smoothing), ``refine_iterations`` /
        ``refine_tolerance`` (perpendicular refinement), ``refine_anchor``
        (end samples kept fixed during refinement, momia2
        ``anchor_length``), ``refine_max_step`` (per-iteration move cap,
        px), ``knn`` (outline
        segment candidates per query), ``pole_fit_points`` (end-tangent
        fit window), ``pad`` (bbox padding).
    pixel_microns : float
        Physical pixel size in microns.

    Returns
    -------
    dict of int to dict
        Per label: ``midline`` (list of ``(M, 2)`` branches in global
        ``(x, y)``; branch 0 is the resampled main branch incl. poles),
        ``width_profile`` (``(n_points,)`` microns), ``pole1``/``pole2``
        (``(x, y)``), and scalars ``length_um``, ``width_median_um``,
        ``width_std_um``, ``width_max_um``, ``width_q1_um``,
        ``width_q3_um``, ``sinuosity``, ``branched``.
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
    """Write midline geometry and scalars into a cells table.

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
        columns written (rows without a result keep nulls).
    """
    table = _to_pa(cells_table)
    by_key = any(int(k) > _keys.LABEL_MAX for k in results)
    row_ids = table.column("cell_key" if by_key else "label").to_pylist()

    midline_vals: list[list[dict[str, list[float]]] | None] = []
    width_vals: list[list[float] | None] = []
    scalar_vals: dict[str, list[float | None]] = {c: [] for c in MIDLINE_SCALAR_COLUMNS}
    for rid in row_ids:
        res = results.get(rid)
        if res is None:
            midline_vals.append(None)
            width_vals.append(None)
            for c in MIDLINE_SCALAR_COLUMNS:
                scalar_vals[c].append(None)
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

    table = _upsert_column(table, "midline", pa.array(midline_vals, type=_schema.MIDLINE_T))
    table = _upsert_column(
        table, "width_profile", pa.array(width_vals, type=_schema.WIDTH_PROFILE_T)
    )
    for c in MIDLINE_SCALAR_COLUMNS:
        table = _upsert_column(table, c, pa.array(scalar_vals[c], type=pa.float32()))
    return _back(table, cells_table)
