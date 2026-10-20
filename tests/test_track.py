"""Tests for momia3.ops.track (§4.7)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from momia3 import schema
from momia3.keys import ROOT_KEY, decode_key, encode_key
from momia3.ops import track as tr
from tests.synth import generate_timelapse


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _record(
    frame: int,
    label: int,
    y: float,
    x: float,
    area: float,
    orientation: float = 0.0,
) -> dict:
    """Minimal link_frames record."""
    return {
        "cell_key": encode_key(0, frame, label),
        "label": label,
        "y": y,
        "x": x,
        "area": area,
        "hu": [0.0] * 7,
        "orientation": orientation,
    }


def _timelapse(seed: int = 0, drift: float = 0.0, vsnap: bool = False, n_frames: int = 18):
    """Timelapse with >= 3 divisions, matching the task acceptance setup."""
    return generate_timelapse(
        n_frames=n_frames,
        shape=(256, 256),
        n_initial=4,
        drift_px_per_frame=drift,
        noise_sigma=0.02,
        seed=seed,
        division_length_um=3.0,
        growth_rate_um_per_frame=0.12,
        vsnap=vsnap,
    )


def _truth_edge_set(data: dict) -> set[tuple[int, int]]:
    return {(p, c) for p, c, e in data["edges"] if e in ("continue", "division")}


def _pred_edge_set(edges_df: pl.DataFrame) -> set[tuple[int, int]]:
    return {
        (int(r["parent_key"]), int(r["child_key"]))
        for r in edges_df.to_dicts()
        if r["event"] in ("continue", "division", "gap")
    }


# ---------------------------------------------------------------------------
# overlap_matrix
# ---------------------------------------------------------------------------


def test_overlap_matrix_counts() -> None:
    a = np.zeros((6, 6), dtype=np.uint32)
    b = np.zeros((6, 6), dtype=np.uint32)
    a[1:4, 1:4] = 1
    a[4, 4] = 2
    b[2:5, 2:5] = 1  # overlaps a==1 on (2:4, 2:4) = 4 px
    b[4, 4] = 3  # overlaps a==2 on 1 px
    ov = tr.overlap_matrix(a, b)
    assert ov.shape == (3, 4)
    dense = ov.toarray()
    assert dense[1, 1] == 4
    assert dense[2, 3] == 1
    assert dense.sum() == 5
    assert dense[0].sum() == 0 and dense[:, 0].sum() == 0


def test_overlap_matrix_disjoint_and_empty() -> None:
    a = np.zeros((4, 4), dtype=np.uint32)
    b = np.zeros((4, 4), dtype=np.uint32)
    a[0, 0] = 1
    b[3, 3] = 2
    assert tr.overlap_matrix(a, b).nnz == 0
    assert tr.overlap_matrix(a, a * 0).nnz == 0


# ---------------------------------------------------------------------------
# link_frames
# ---------------------------------------------------------------------------


def test_link_frames_continue() -> None:
    t = np.zeros((60, 60), dtype=np.uint32)
    t1 = np.zeros((60, 60), dtype=np.uint32)
    t[28:33, 10:50] = 1
    t1[28:33, 11:51] = 1  # shifted by 1 px
    parents = [_record(0, 1, 30.0, 29.5, 200.0)]
    children = [_record(1, 1, 30.0, 30.5, 200.0)]
    edges = tr.link_frames(parents, children, tr.overlap_matrix(t, t1))
    assert len(edges) == 1
    (edge,) = edges
    assert edge["event"] == "continue"
    assert edge["parent_key"] == encode_key(0, 0, 1)
    assert edge["child_key"] == encode_key(0, 1, 1)


def test_link_frames_division_and_vsnap() -> None:
    t = np.zeros((60, 70), dtype=np.uint32)
    t1 = np.zeros((60, 70), dtype=np.uint32)
    t[28:33, 10:50] = 1  # mother, area 200
    t1[28:33, 12:29] = 1  # daughter 1, area 85
    t1[28:33, 32:49] = 2  # daughter 2, area 85
    mother = _record(0, 1, 30.0, 29.5, 200.0, orientation=0.0)
    d1 = _record(1, 1, 30.0, 20.0, 85.0, orientation=0.0)
    d2 = _record(1, 2, 30.0, 40.0, 85.0, orientation=0.0)
    edges = tr.link_frames([mother], [d1, d2], tr.overlap_matrix(t, t1))
    div = [e for e in edges if e["event"] == "division"]
    assert len(edges) == 2
    assert {e["child_key"] for e in div} == {encode_key(0, 1, 1), encode_key(0, 1, 2)}
    assert all(e["parent_key"] == encode_key(0, 0, 1) for e in div)
    assert not any(e["vsnap"] for e in div)


def test_link_frames_vsnap_flag() -> None:
    t = np.zeros((60, 70), dtype=np.uint32)
    t1 = np.zeros((60, 70), dtype=np.uint32)
    t[28:33, 10:50] = 1
    t1[28:33, 12:29] = 1
    t1[28:33, 32:49] = 2
    mother = _record(0, 1, 30.0, 29.5, 200.0, orientation=0.0)
    d1 = _record(1, 1, 30.0, 20.0, 85.0, orientation=0.0)
    d2 = _record(1, 2, 30.0, 40.0, 85.0, orientation=np.radians(30.0))
    edges = tr.link_frames([mother], [d1, d2], tr.overlap_matrix(t, t1))
    by_child = {e["child_key"]: e for e in edges}
    assert by_child[encode_key(0, 1, 1)]["vsnap"] is False
    assert by_child[encode_key(0, 1, 2)]["vsnap"] is True


def test_link_frames_appear_disappear() -> None:
    empty = np.zeros((10, 10), dtype=np.uint32)
    ov = tr.overlap_matrix(empty, empty)
    parents = [_record(0, 1, 5.0, 5.0, 100.0), _record(0, 2, 5.0, 50.0, 100.0)]
    children = [_record(1, 1, 5.0, 5.0, 100.0)]
    # No spatial overlap anywhere; gate_distance default 20 px keeps the
    # coincident pair linkable only via distance -> cost too high, so the
    # parent disappears and the child appears.
    edges = tr.link_frames(parents, children, ov)
    events = {(e["parent_key"], e["child_key"]): e["event"] for e in edges}
    assert events[(encode_key(0, 0, 1), ROOT_KEY)] == "disappear"
    assert events[(encode_key(0, 0, 2), ROOT_KEY)] == "disappear"
    assert events[(ROOT_KEY, encode_key(0, 1, 1))] == "appear"


# ---------------------------------------------------------------------------
# link_timelapse (acceptance)
# ---------------------------------------------------------------------------


def test_link_timelapse_recovers_truth_edges() -> None:
    data = _timelapse(seed=0, drift=0.0)
    n_div = len({p for p, _, e in data["edges"] if e == "division"})
    assert n_div >= 3, "fixture must contain >= 3 divisions"

    edges_df, tracks_df = tr.link_timelapse(data["labels"])
    truth = _truth_edge_set(data)
    pred = _pred_edge_set(edges_df)
    recall = len(truth & pred) / len(truth)
    assert recall >= 0.95

    # All truth division events are recovered as division edges.
    div_truth = {p for p, _, e in data["edges"] if e == "division"}
    div_pred = {
        int(r["parent_key"]) for r in edges_df.to_dicts() if r["event"] == "division"
    }
    assert div_truth <= div_pred

    # Dividing tracks get fate='divide'; daughters reference their parent.
    tracks = tracks_df.to_dicts()
    dividing = [r for r in tracks if r["fate"] == "divide"]
    assert len(dividing) == len(div_truth)
    assert all(r["division_frame"] is not None for r in dividing)
    daughters = [r for r in tracks if r["parent_track"] is not None]
    assert len(daughters) == 2 * len(dividing)

    # Output tables pass schema validation (vsnap rides along as extra).
    schema.validate_edges(edges_df.to_arrow())
    schema.validate_tracks(tracks_df.to_arrow())
    assert "vsnap" in edges_df.columns


def test_link_timelapse_with_drift() -> None:
    data = _timelapse(seed=0, drift=1.5)
    edges_df, _ = tr.link_timelapse(data["labels"])
    truth = _truth_edge_set(data)
    pred = _pred_edge_set(edges_df)
    recall = len(truth & pred) / len(truth)
    assert recall >= 0.90


def test_link_timelapse_vsnap_timelapse() -> None:
    data = _timelapse(seed=0, drift=0.0, vsnap=True)
    edges_df, _ = tr.link_timelapse(data["labels"])
    rows = edges_df.to_dicts()
    div = [r for r in rows if r["event"] == "division"]
    flagged = [r for r in div if r["vsnap"]]
    assert len(div) >= 6
    assert len(flagged) >= 2  # kinked daughters are flagged


def test_link_timelapse_is_deterministic() -> None:
    data = _timelapse(seed=1, drift=0.5)
    edges1, tracks1 = tr.link_timelapse(data["labels"])
    edges2, tracks2 = tr.link_timelapse(data["labels"])
    assert edges1.equals(edges2)
    assert tracks1.equals(tracks2)


# ---------------------------------------------------------------------------
# gap closing
# ---------------------------------------------------------------------------


def test_link_timelapse_gap_closing() -> None:
    data = generate_timelapse(
        n_frames=6,
        shape=(128, 128),
        n_initial=2,
        drift_px_per_frame=0.0,
        noise_sigma=0.01,
        seed=5,
        division_length_um=10.0,  # no divisions
        growth_rate_um_per_frame=0.05,
    )
    labels = data["labels"].copy()
    victim = int(labels[0].max())  # a real label present in every frame
    assert victim >= 1
    for f in (2, 3):  # erase the cell for two frames
        labels[f][labels[f] == victim] = 0

    edges_df, tracks_df = tr.link_timelapse(labels)
    gaps = [r for r in edges_df.to_dicts() if r["event"] == "gap"]
    assert len(gaps) == 1
    _, f_parent, _ = decode_key(int(gaps[0]["parent_key"]))
    _, f_child, _ = decode_key(int(gaps[0]["child_key"]))
    assert (f_parent, f_child) == (1, 4)

    # The bridged cell forms one track spanning frames 0-5 (4 present).
    rows = tracks_df.to_dicts()
    victim_tracks = [r for r in rows if r["birth_frame"] == 0 and r["n_frames"] == 4]
    assert len(victim_tracks) == 1
    assert victim_tracks[0]["last_frame"] == 5
    assert victim_tracks[0]["fate"] in ("incomplete", "edge")  # not truncated


def test_link_timelapse_respects_include_filter() -> None:
    data = generate_timelapse(
        n_frames=8,
        shape=(128, 128),
        n_initial=2,
        drift_px_per_frame=0.5,
        noise_sigma=0.02,
        seed=2,
        division_length_um=10.0,
        growth_rate_um_per_frame=0.05,
    )
    labels = data["labels"]
    victim = int(labels[0].max())
    frames, labs, includes = [], [], []
    for t in range(labels.shape[0]):
        for lab in np.unique(labels[t]):
            if lab == 0:
                continue
            frames.append(t)
            labs.append(int(lab))
            includes.append(int(lab) != victim)
    cells = pa.table(
        {
            "fov_id": pa.array([0] * len(frames), type=pa.uint32()),
            "frame": pa.array(frames, type=pa.uint32()),
            "label": pa.array(labs, type=pa.uint32()),
            "include": pa.array(includes, type=pa.bool_()),
        }
    )
    full_edges, full_tracks = tr.link_timelapse(labels)
    filt_edges, filt_tracks = tr.link_timelapse(labels, cells)
    assert filt_edges.height < full_edges.height
    assert filt_tracks.height == full_tracks.height - 1


# ---------------------------------------------------------------------------
# refine_lineage
# ---------------------------------------------------------------------------


def _merge_scenario() -> tuple[np.ndarray, pl.DataFrame, dict[str, int]]:
    """A divides into B and C; B and C spuriously merge into D.

    Frames: 0 -> A (label 1); 1 -> B (label 1), C (label 2);
    2 -> D (label 3) covering both daughters.
    """
    labels = np.zeros((3, 60, 70), dtype=np.uint32)
    labels[0, 28:33, 10:50] = 1
    labels[1, 28:33, 12:29] = 1
    labels[1, 28:33, 32:49] = 2
    labels[2, 28:33, 12:49] = 3
    ka = encode_key(0, 0, 1)
    kb = encode_key(0, 1, 1)
    kc = encode_key(0, 1, 2)
    kd = encode_key(0, 2, 3)
    edges = pa.table(
        {
            "parent_key": pa.array(
                [ROOT_KEY, ka, ka, kb, kc], type=pa.uint64()
            ),
            "child_key": pa.array([ka, kb, kc, kd, kd], type=pa.uint64()),
            "event": pa.array(
                ["appear", "division", "division", "continue", "continue"],
                type=pa.string(),
            ),
            "confidence": pa.array([1.0] * 5, type=pa.float32()),
        }
    )
    return labels, pl.from_arrow(schema.validate_edges(edges)), {
        "a": ka,
        "b": kb,
        "c": kc,
        "d": kd,
    }


def test_refine_lineage_resplits_converged_cell() -> None:
    labels, edges_df, keys = _merge_scenario()
    out = tr.refine_lineage(labels, edges_df)
    schema.validate_edges(out.to_arrow())
    rows = out.to_dicts()
    pairs = {(int(r["parent_key"]), int(r["child_key"])): r["event"] for r in rows}

    # The spurious convergence edges are gone.
    assert (keys["b"], keys["d"]) not in pairs
    assert (keys["c"], keys["d"]) not in pairs
    # The true division is preserved.
    assert pairs[(keys["a"], keys["b"])] == "division"
    assert pairs[(keys["a"], keys["c"])] == "division"
    # Each mother reconnects to its own fresh fragment at frame 2.
    b_children = [c for (p, c), e in pairs.items() if p == keys["b"]]
    c_children = [c for (p, c), e in pairs.items() if p == keys["c"]]
    assert len(b_children) == 1 and len(c_children) == 1
    assert b_children[0] != c_children[0]
    for child in b_children + c_children:
        fov, frame, lab = decode_key(child)
        assert (fov, frame) == (0, 2)
        assert lab > 3  # fresh labels beyond the previous frame max


def test_refine_lineage_noop_on_clean_lineage() -> None:
    data = _timelapse(seed=1, drift=0.0)
    edges_df, _ = tr.link_timelapse(data["labels"])
    out = tr.refine_lineage(data["labels"], edges_df)
    assert out.select(["parent_key", "child_key", "event"]).equals(
        edges_df.select(["parent_key", "child_key", "event"])
    )


def test_refine_lineage_writeback_labels() -> None:
    """writeback_labels=True replaces the suspect pixels with the fragments."""
    labels, edges_df, keys = _merge_scenario()
    d_area = int((labels[2] == 3).sum())
    out = tr.refine_lineage(labels, edges_df, writeback_labels=True)
    rows = out.to_dicts()
    pairs = {(int(r["parent_key"]), int(r["child_key"])): r["event"] for r in rows}
    b_children = [c for (p, c) in pairs if p == keys["b"]]
    c_children = [c for (p, c) in pairs if p == keys["c"]]
    fragment_labels = {
        decode_key(k)[2] for k in b_children + c_children
    }
    assert len(fragment_labels) == 2  # two distinct fresh labels

    # The converged label is gone from the stack; both fragments cover the
    # whole former mask (partitioning the merged blob between the mothers).
    frame2 = labels[2]
    assert (frame2 == 3).sum() == 0  # old D label replaced
    for lab in fragment_labels:
        assert (frame2 == lab).sum() > 0
    assert (frame2 > 3).sum() == d_area


def test_refine_lineage_writeback_default_off() -> None:
    """Without writeback_labels the input stack is left untouched."""
    labels, edges_df, _keys_ = _merge_scenario()
    before = labels.copy()
    tr.refine_lineage(labels, edges_df)
    np.testing.assert_array_equal(labels, before)


# ---------------------------------------------------------------------------
# ctc_metrics
# ---------------------------------------------------------------------------


def test_ctc_metrics_perfect_prediction() -> None:
    data = _timelapse(seed=1, drift=0.0)
    metrics = tr.ctc_metrics(data["edges"], data["labels"], data["edges"], data["labels"])
    assert metrics["seg_like"] == pytest.approx(1.0)
    assert metrics["tra_like"] == pytest.approx(1.0)
    assert metrics["division_precision"] == pytest.approx(1.0)
    assert metrics["division_recall"] == pytest.approx(1.0)
    assert metrics["edge_fp"] == 0 and metrics["edge_fn"] == 0


def test_ctc_metrics_counts() -> None:
    truth_edges = [
        (encode_key(0, 0, 1), encode_key(0, 1, 1), "continue"),
        (encode_key(0, 1, 1), encode_key(0, 2, 1), "division"),
        (encode_key(0, 1, 1), encode_key(0, 2, 2), "division"),
    ]
    pred_edges = [
        (encode_key(0, 0, 1), encode_key(0, 1, 1), "continue"),
        (encode_key(0, 1, 1), encode_key(0, 2, 1), "continue"),  # wrong event, right pair
        (encode_key(0, 5, 1), encode_key(0, 6, 1), "continue"),  # spurious
    ]
    labels = np.zeros((7, 4, 4), dtype=np.uint32)
    labels[:, 1:3, 1:3] = 1  # identical stacks -> seg_like == 1
    metrics = tr.ctc_metrics(pred_edges, labels, truth_edges, labels)
    assert metrics["edge_tp"] == 2
    assert metrics["edge_fp"] == 1
    assert metrics["edge_fn"] == 1
    assert metrics["tra_like"] == pytest.approx(2 / 4)
    assert metrics["division_tp"] == 0
    assert metrics["division_fn"] == 1
    assert metrics["division_recall"] == pytest.approx(0.0)
    assert metrics["seg_like"] == pytest.approx(1.0)


def test_ctc_metrics_against_link_timelapse_output() -> None:
    data = _timelapse(seed=0, drift=0.0)
    edges_df, _ = tr.link_timelapse(data["labels"])
    metrics = tr.ctc_metrics(edges_df, data["labels"], data["edges"], data["labels"])
    assert metrics["seg_like"] == pytest.approx(1.0)
    assert metrics["tra_like"] >= 0.95
    assert metrics["division_recall"] == pytest.approx(1.0)
