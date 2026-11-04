"""Tests for the per-cell best-unit-accuracy scorer.

The paired-recordings regression on PAIRED_CRCNS_HC1 motivated this suite: a
GT cell with 123 spikes vs a candidate single unit matching only 6 of them had
the scorer reporting 1.0 best-unit-accuracy (hiding the 117 FN + N FP) because
``tab.sum(axis={0,1})`` on the match-filtered contingency table used the
matched-subset counts in both legs of the IoU denominator.

What the scorer MUST compute:
    tp        = |GT_cell ∩ best_cand|      from the contingency (temporal
                                             matching within ±delta_frames)
    accuracy  = tp / (|GT_cell| + |best_cand| - tp)
where |GT_cell| and |best_cand| are the TOTAL spike counts of the GT cell
and the best-matching candidate unit — NOT the counts after the match filter.
"""
from __future__ import annotations

import numpy as np
import pytest

from ns3_sort_bench._base import SortedResult
from ns3_sort_bench.scorers.per_cell import PerCellScorer


def _mk(adapter, times_by_unit, sr=20000.0, name="corpus_x"):
    return SortedResult(
        unit_ids=list(times_by_unit),
        spike_times={int(u): np.asarray(t, np.int64) for u, t in times_by_unit.items()},
        sampling_rate=sr,
        adapter=adapter, corpus_item=name,
    )


# ─── regression: 6/123 matched → NOT 1.0 ─────────────────────────────────────
def test_partial_match_single_unit_reports_true_iou():
    """123 GT spikes, 500 candidate spikes, only 6 overlap within ±delta.

    IoU = 6 / (123 + 500 - 6) = 6 / 617 ≈ 0.00972.  The pre-fix scorer
    returned 1.0 because the contingency collapsed to a 1×1 over the matched
    6 spikes; this test guards against that regression.
    """
    rng = np.random.default_rng(0)
    delta = 12
    sr = 20000.0

    # GT: 123 spikes spaced well apart so the matching is unambiguous.
    gt_times = np.sort(rng.choice(np.arange(100_000, 900_000), 123, replace=False)).astype(np.int64)
    gt = _mk("ground_truth", {2: gt_times}, sr=sr)

    # Candidate: 500 spikes elsewhere, plus 6 that collide with 6 random GT spikes
    # (within delta samples).  Keep total disjoint spike count stable at 500.
    coll_idx = rng.choice(len(gt_times), 6, replace=False)
    cand_colliders = gt_times[coll_idx] + rng.integers(-delta + 1, delta, 6).astype(np.int64)
    base_cand = np.sort(rng.choice(np.arange(1_200_000, 2_000_000),
                                    500 - 6, replace=False)).astype(np.int64)
    cand_times = np.sort(np.concatenate([base_cand, cand_colliders]))
    can = _mk("my_sorter", {5: cand_times}, sr=sr)

    scorer = PerCellScorer(delta_frames=delta)
    out = scorer(gt, can)

    tp_true     = 6
    gt_total    = 123
    cand_total  = 500
    iou_true = tp_true / (gt_total + cand_total - tp_true)       # ≈ 0.00972

    assert out.values["best_unit_accuracy_mean"] == pytest.approx(iou_true, rel=1e-4), \
        f"IoU denominator still uses matched-subset counts — got {out.values['best_unit_accuracy_mean']}"
    assert out.values["best_unit_recall_mean"]    == pytest.approx(tp_true / gt_total,   rel=1e-4)
    assert out.values["best_unit_precision_mean"] == pytest.approx(tp_true / cand_total, rel=1e-4)
    assert out.values["n_ref_cells"]  == 1.0
    assert out.values["n_cand_units"] == 1.0                     # only one cand matched

    # Both thresholds should miss (IoU is tiny)
    assert out.values["cells_ge_80"] == 0.0
    assert out.values["cells_ge_50"] == 0.0


# ─── identity: both sides identical → 1.0 ────────────────────────────────────
def test_identical_results_report_perfect_iou():
    t = np.arange(100_000, 900_000, 1000, dtype=np.int64)         # 800 spikes
    gt  = _mk("ground_truth", {2: t})
    can = _mk("perfect",      {2: t})
    out = PerCellScorer(delta_frames=12)(gt, can)

    assert out.values["best_unit_accuracy_mean"]  == pytest.approx(1.0)
    assert out.values["best_unit_recall_mean"]    == pytest.approx(1.0)
    assert out.values["best_unit_precision_mean"] == pytest.approx(1.0)
    assert out.values["cells_ge_80"] == 1.0
    assert out.values["cells_ge_50"] == 1.0


# ─── half-recall, perfect-precision: TP = 50, GT = 100, cand = 50 ────────────
def test_pure_undersplit_candidate():
    """Candidate recovers half the GT cell perfectly (no FP).  IoU = 50/100 = 0.5."""
    gt_t  = np.arange(100_000, 100_000 + 100 * 500, 500, dtype=np.int64)    # 100 spikes
    can_t = gt_t[::2].copy()                                                 # every other one
    gt  = _mk("ground_truth", {2: gt_t})
    can = _mk("recall_half", {5: can_t})

    out = PerCellScorer(delta_frames=12)(gt, can)
    assert out.values["best_unit_accuracy_mean"]  == pytest.approx(50 / (100 + 50 - 50), rel=1e-4)
    assert out.values["best_unit_recall_mean"]    == pytest.approx(50 / 100, rel=1e-4)
    assert out.values["best_unit_precision_mean"] == pytest.approx(50 / 50,  rel=1e-4)


# ─── empty candidate ─────────────────────────────────────────────────────────
def test_empty_candidate_reports_zero():
    gt  = _mk("ground_truth", {2: np.array([1000, 2000, 3000], np.int64)})
    can = _mk("empty",        {})
    out = PerCellScorer(delta_frames=12)(gt, can)
    assert out.values["best_unit_accuracy_mean"]  == 0.0
    assert out.values["best_unit_recall_mean"]    == 0.0
    assert out.values["best_unit_precision_mean"] == 0.0
    assert out.values["n_ref_cells"]  == 1.0
    assert out.values["n_cand_units"] == 0.0


# ─── multi-GT, candidate matches only one ────────────────────────────────────
def test_multiple_gt_cells_one_match_one_miss():
    """Two GT cells; candidate recovers one perfectly, barely touches the other.

    Cell 3's 'misses' are 2 noise spikes placed within ±12 of two of its
    spikes, so tp=2 and its IoU reflects the 78 FN + (cand_total - 2) FP it
    should carry.  An unrelated-noise variant (no match at all) is exercised
    separately in `test_gt_cell_with_zero_matches_contributes_zero`.
    """
    t_a = np.arange(10_000, 50_000, 500, dtype=np.int64)       # 80 spikes
    t_b = np.arange(60_000, 100_000, 500, dtype=np.int64)      # 80 spikes
    gt = _mk("ground_truth", {2: t_a, 3: t_b})
    # Candidate recovers t_a perfectly + 2 spikes exactly at cell 3's first two spikes
    # (within ±12 samples → 2 TP for cell 3).
    coincident = np.array([60_002, 60_505], np.int64)
    can = _mk("partial_sorter", {5: np.concatenate([t_a, coincident])})

    out = PerCellScorer(delta_frames=12)(gt, can)
    cand_total = 80 + 2
    iou_a = 80 / (80 + cand_total - 80)             # 80 / 82
    iou_b =  2 / (80 + cand_total -  2)             #  2 / 160
    mean_iou = (iou_a + iou_b) / 2
    assert out.values["best_unit_accuracy_mean"] == pytest.approx(mean_iou, rel=1e-4)
    assert out.values["n_ref_cells"]  == 2.0
    # One good cell (>=80, >=50) and one bad (<50)
    assert out.values["cells_ge_80"] == 1.0
    assert out.values["cells_ge_50"] == 1.0

    # Details carry per-cell numbers
    assert out.details["per_cell_gt_total"][2]   == 80
    assert out.details["per_cell_gt_total"][3]   == 80
    assert out.details["per_cell_cand_total"][2] == cand_total
    assert out.details["per_cell_tp"][2]         == 80
    assert out.details["per_cell_tp"][3]         == 2


# ─── unmatched GT cells contribute 0 to the mean, not nothing ────────────────
def test_gt_cell_with_zero_matches_contributes_zero():
    """A GT cell with no candidate spike within ±delta of any of its spikes
    MUST contribute IoU = 0 to best_unit_accuracy_mean (and lower n_ref_cells
    DOES NOT absorb it)."""
    t_a = np.arange(10_000, 50_000, 500, dtype=np.int64)       # 80 spikes, matched
    t_b = np.arange(60_000, 100_000, 500, dtype=np.int64)      # 80 spikes, missed
    gt = _mk("ground_truth", {2: t_a, 3: t_b})
    can = _mk("misses_b", {5: t_a.copy()})                      # no spike near t_b

    out = PerCellScorer(delta_frames=12)(gt, can)
    iou_a = 80 / (80 + 80 - 80)                                  # 1.0
    iou_b = 0.0
    assert out.values["best_unit_accuracy_mean"] == pytest.approx((iou_a + iou_b) / 2, rel=1e-4)
    assert out.values["best_unit_recall_mean"]    == pytest.approx(0.5, rel=1e-4)
    assert out.values["n_ref_cells"]  == 2.0                     # both cells counted
    assert out.details["per_cell_tp"][3] == 0
    assert out.details["per_cell_best_unit"][3] is None
    assert out.details["per_cell_accuracy"][3] == 0.0
