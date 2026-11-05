"""Tests for the oracle-merge scorer (curation ceiling)."""
from __future__ import annotations

import numpy as np
import pytest

from ns3_sort_bench._base import SortedResult
from ns3_sort_bench.scorers.oracle_merge import OracleMergeScorer


def _mk(adapter, times_by_unit, sr=20000.0, name="corpus_x"):
    return SortedResult(
        unit_ids=list(times_by_unit),
        spike_times={int(u): np.asarray(t, np.int64) for u, t in times_by_unit.items()},
        sampling_rate=sr,
        adapter=adapter, corpus_item=name,
    )


# ─── over-split: candidate split into N pure fragments → merge recovers 1.0 ──
def test_pure_oversplit_merges_to_perfect_iou():
    """GT cell has 120 spikes.  Candidate split the same 120 spikes into 4
    fragments of 30 each (no FPs anywhere).  The oracle merge must pick
    all 4 fragments and reach IoU = 1.0.
    """
    t = np.arange(10_000, 10_000 + 120 * 500, 500, dtype=np.int64)  # 120 spikes
    frag = np.array_split(t, 4)
    gt  = _mk("ground_truth", {2: t})
    can = _mk("over_split", {10 + i: frag[i] for i in range(4)})

    out = OracleMergeScorer(delta_frames=12)(gt, can)
    assert out.values["oracle_merge_accuracy_mean"]  == pytest.approx(1.0, rel=1e-6)
    assert out.values["oracle_merge_recall_mean"]    == pytest.approx(1.0, rel=1e-6)
    assert out.values["oracle_merge_precision_mean"] == pytest.approx(1.0, rel=1e-6)
    assert out.values["oracle_merge_subset_size_mean"] == 4.0
    assert set(out.details["per_cell_oracle_subset"][2]) == {10, 11, 12, 13}
    assert out.values["cells_ge_80"] == 1.0


# ─── mixed-purity fragments: oracle picks only the subset that improves IoU ──
def test_oracle_rejects_candidates_that_hurt_iou():
    """GT: 100 spikes.  Three candidates:
       A: 50 TP + 0 FP   (pure)
       B: 20 TP + 5 FP   (adds; still raises IoU)
       C:  5 TP + 500 FP (would drop IoU — must NOT be chosen)
    Greedy oracle picks A, then B, and STOPS.
    """
    rng = np.random.default_rng(1)
    delta = 12
    gt_t = np.sort(rng.choice(np.arange(10_000, 500_000), 100, replace=False)).astype(np.int64)

    # Pure A: 50 random GT spikes
    idx_a = rng.choice(100, 50, replace=False)
    a_t   = gt_t[idx_a]

    # B: 20 other GT spikes + 5 nearby noise spikes
    remaining = np.setdiff1d(np.arange(100), idx_a)
    idx_b = rng.choice(remaining, 20, replace=False)
    b_t   = np.sort(np.concatenate([
        gt_t[idx_b],
        np.arange(600_000, 600_000 + 5 * 200, 200, dtype=np.int64),
    ]))

    # C: 5 random GT + 500 noise
    idx_c = rng.choice(np.setdiff1d(remaining, idx_b), 5, replace=False)
    c_t   = np.sort(np.concatenate([
        gt_t[idx_c],
        np.arange(800_000, 800_000 + 500 * 200, 200, dtype=np.int64),
    ]))

    gt  = _mk("ground_truth", {2: gt_t})
    can = _mk("mix", {10: a_t, 11: b_t, 12: c_t})
    out = OracleMergeScorer(delta_frames=delta)(gt, can)

    chosen = set(out.details["per_cell_oracle_subset"][2])
    assert 10 in chosen                       # A is a pure win, must be in
    assert 11 in chosen                       # B still raises IoU
    assert 12 not in chosen                   # C is a trap — must be rejected

    # TP = 50 + 20 = 70; FP = 0 + 5 = 5;  IoU = 70 / (100 + 5) = 0.667
    assert out.details["per_cell_oracle_tp"][2]  == 70
    assert out.details["per_cell_oracle_fp"][2]  == 5
    assert out.values["oracle_merge_accuracy_mean"] == pytest.approx(70 / 105, rel=1e-4)


# ─── identity: raw sort already optimal, subset_size == 1 ────────────────────
def test_identical_results_do_not_need_merging():
    t = np.arange(100_000, 900_000, 1000, dtype=np.int64)
    gt  = _mk("ground_truth", {2: t})
    can = _mk("perfect",      {2: t})
    out = OracleMergeScorer(delta_frames=12)(gt, can)

    assert out.values["oracle_merge_accuracy_mean"] == pytest.approx(1.0)
    assert out.values["oracle_merge_subset_size_mean"] == 1.0
    assert out.details["per_cell_oracle_subset"][2] == [2]


# ─── oracle >= per_cell best single candidate (no scorer regression) ─────────
def test_oracle_never_worse_than_per_cell_best_unit():
    """The oracle is at worst the best single candidate.  This property
    must hold on any over-split input, including the one in the
    regression-guard shape 123/111/28 that motivated this scorer."""
    from ns3_sort_bench.scorers.per_cell import PerCellScorer

    rng = np.random.default_rng(2)
    delta = 12
    sr = 20000.0

    # Simulate the HC1 shape: 1 GT cell with 123 spikes; 28 candidate
    # clusters, together holding ~111 of those spikes, best-single-cand
    # holding ~18 of them.
    gt_t = np.sort(rng.choice(np.arange(100_000, 1_000_000), 123, replace=False)).astype(np.int64)
    rng2 = np.random.default_rng(3)
    cand = {}
    taken = np.zeros(len(gt_t), dtype=bool)
    for cid in range(28):
        tp_n = rng2.integers(1, 8)             # 1-7 TP per candidate (avg ~4)
        fp_n = rng2.integers(10, 60)
        avail = np.flatnonzero(~taken)
        if avail.size == 0:
            tp_idx = np.array([], dtype=np.int64)
        else:
            tp_idx = rng2.choice(avail, min(tp_n, avail.size), replace=False)
            taken[tp_idx] = True
        tp_t = gt_t[tp_idx] + rng2.integers(-delta + 1, delta, len(tp_idx)).astype(np.int64)
        fp_t = rng2.integers(1_500_000, 2_500_000, fp_n).astype(np.int64)
        cand[10 + cid] = np.sort(np.concatenate([tp_t, fp_t]))

    gt  = _mk("ground_truth", {2: gt_t})
    can = _mk("over_split_28", cand)
    pc  = PerCellScorer(delta_frames=delta)(gt, can)
    om  = OracleMergeScorer(delta_frames=delta)(gt, can)

    assert om.values["oracle_merge_accuracy_mean"] >= pc.values["best_unit_accuracy_mean"] - 1e-9
    assert om.values["oracle_merge_recall_mean"]   >= pc.values["best_unit_recall_mean"]   - 1e-9
    # Merging more than 1 candidate is the whole point on an over-split sort
    assert om.values["oracle_merge_subset_size_mean"] > 1.0


# ─── empty / no-match degenerate cases match per_cell's conventions ──────────
def test_empty_candidate_returns_zero():
    gt  = _mk("ground_truth", {2: np.array([1000, 2000, 3000], np.int64)})
    can = _mk("empty",        {})
    out = OracleMergeScorer(delta_frames=12)(gt, can)
    assert out.values["oracle_merge_accuracy_mean"]       == 0.0
    assert out.values["oracle_merge_subset_size_mean"]    == 0.0
    assert out.values["n_ref_cells"]                      == 1.0
    assert out.values["n_cand_units"]                     == 0.0


def test_gt_cell_with_no_overlap_contributes_zero_to_mean():
    """A GT cell with NO candidate spike within +/-delta: subset must be
    empty, IoU = 0, and the cell must STILL count toward n_ref_cells."""
    t_a = np.arange(10_000, 50_000, 500, dtype=np.int64)       # matched
    t_b = np.arange(60_000, 100_000, 500, dtype=np.int64)      # not matched
    can_t = t_a.copy()                                          # nowhere near t_b
    gt  = _mk("ground_truth", {2: t_a, 3: t_b})
    can = _mk("one_match",   {10: can_t})
    out = OracleMergeScorer(delta_frames=12)(gt, can)

    assert out.details["per_cell_oracle_subset"][3]  == []
    assert out.details["per_cell_oracle_tp"][3]      == 0
    assert out.details["per_cell_oracle_accuracy"][3] == 0.0
    # Mean is over BOTH cells
    assert out.values["n_ref_cells"]                 == 2.0
    assert out.values["oracle_merge_recall_mean"]    == pytest.approx(0.5, rel=1e-4)
