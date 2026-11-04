"""Per-cell best-unit accuracy — the SpikeForest paired-recordings convention.

For each ground-truth cell `g`:
    best_unit = argmax_c |g ∩ c|
    tp        = |g ∩ best_unit|                   (spikes of g within
                                                    ±delta_frames of some
                                                    spike of best_unit)
    accuracy  = tp / (|g| + |best_unit| - tp)      intersection-over-union
                                                    (= SpikeForest's "accuracy"
                                                    = 1 / (1/precision + 1/recall - 1))

Two correctness properties the paired-recordings regression required:

1.  `|g|` is the TOTAL number of GT spikes in cell g (the patched cell), NOT
    the number of GT spikes that happened to land near a candidate.  Likewise
    `|best_unit|` is the total candidate spikes.  Reading these from the
    match-filtered contingency silently collapses IoU to 1.0 whenever both GT
    and candidate are single-unit (the entire contingency becomes 1×1 over
    just the matched pairs).  PAIRED_CRCNS_HC1 made this exact mistake: 6/123
    GT spikes matched one candidate fiber and the metric reported 1.0 instead
    of the true ≈ 0.01.

2.  Every GT cell contributes to the mean, including ones with zero overlap
    with any candidate.  The contingency table drops such cells (they have
    no row), so a scorer that iterates its rows silently excludes the worst
    cases from the mean.  Iterate over `ref.unit_ids` instead; an unmatched
    cell has tp = 0 → IoU = recall = precision = 0 and lowers the mean the
    way it should.

Aggregates:
    best_unit_accuracy_mean  mean IoU over GT cells
    best_unit_recall_mean    mean tp / |g|        (how much of each GT is recovered)
    best_unit_precision_mean mean tp / |best_unit| (how pure the best cand is)
    cells_ge_<thr*100>       count of GT cells whose IoU meets each threshold
    n_ref_cells, n_cand_units
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .._base import ScoreResult, Scorer, SortedResult, _register, scorer_registry
from .pairwise import _assemble_label_array, _contingency


class PerCellScorer(Scorer):
    """Best-unit-accuracy per ground-truth cell (SpikeForest convention)."""
    name = "per_cell"

    def __init__(self, delta_frames: int = 12, thresholds=(0.8, 0.5)):
        self.delta_frames = int(delta_frames)
        self.thresholds = tuple(float(t) for t in thresholds)

    def __call__(self, reference: SortedResult, candidate: SortedResult,
                 **_) -> ScoreResult:
        ref = reference.dropped_reserved()
        can = candidate.dropped_reserved()

        # TRUE per-unit totals — NOT the row/col sums of the match-filtered
        # contingency.  These are the denominators for recall (|GT|) and
        # precision (|cand|), including the unmatched FN / FP populations
        # the contingency filter dropped.
        n_cell_true: Mapping[int, int] = {
            int(u): int(np.asarray(ref.spike_times[u]).size)
            for u in ref.unit_ids
        }
        n_unit_true: Mapping[int, int] = {
            int(u): int(np.asarray(can.spike_times[u]).size)
            for u in can.unit_ids
        }
        ref_cells = [int(u) for u in ref.unit_ids]
        n_ref_cells = len(ref_cells)

        # Degenerate cases: no GT cells, or no candidate spikes anywhere.
        if n_ref_cells == 0 or sum(n_unit_true.values()) == 0:
            values = {
                "best_unit_accuracy_mean":  0.0,
                "best_unit_recall_mean":    0.0,
                "best_unit_precision_mean": 0.0,
                "n_ref_cells":              float(n_ref_cells),
                "n_cand_units":             float(len(can.unit_ids)),
            }
            for t in self.thresholds:
                values[f"cells_ge_{int(round(t*100)):02d}"] = 0.0
            return ScoreResult(
                scorer=self.name, corpus_item=candidate.corpus_item,
                reference_adapter=reference.adapter,
                candidate_adapter=candidate.adapter,
                values=values,
            )

        # Match-filtered contingency: rows are GT cells WITH at least one match,
        # columns are cand units WITH at least one match.  Unmatched GT cells
        # are added back below with tp = 0.
        ref_t, ref_lab = _assemble_label_array(ref)
        _, can_lab = _assemble_label_array(can, delta_frames=self.delta_frames,
                                            index_space=ref_t)
        tab, u_ref, u_can = _contingency(ref_lab, can_lab)

        # Resolve best unit + TP for every GT cell (not just the matched rows).
        row_of = {int(u): i for i, u in enumerate(u_ref)}
        best_unit_by_cell = {}
        tp_by_cell        = {}
        for g in ref_cells:
            i = row_of.get(g, -1)
            if i < 0 or tab.shape[1] == 0:
                # No candidate spike landed near any of this cell's spikes.
                best_unit_by_cell[g] = None
                tp_by_cell[g]        = 0
            else:
                j = int(tab[i].argmax())
                best_unit_by_cell[g] = int(u_can[j])
                tp_by_cell[g]        = int(tab[i, j])

        # Build per-cell IoU/recall/precision arrays in ref_cells order.
        tp         = np.array([tp_by_cell[g]           for g in ref_cells], np.int64)
        gt_total   = np.array([n_cell_true[g]          for g in ref_cells], np.int64)
        cand_total = np.array(
            [n_unit_true.get(int(best_unit_by_cell[g]), 0) if best_unit_by_cell[g] is not None else 0
             for g in ref_cells],
            dtype=np.int64,
        )
        denom = gt_total + cand_total - tp
        acc    = np.where(denom     > 0, tp / np.maximum(denom,     1), 0.0)
        recall = np.where(gt_total  > 0, tp / np.maximum(gt_total,  1), 0.0)
        prec   = np.where(cand_total > 0, tp / np.maximum(cand_total, 1), 0.0)

        values = {
            "best_unit_accuracy_mean":  float(np.mean(acc))    if acc.size    else 0.0,
            "best_unit_recall_mean":    float(np.mean(recall)) if recall.size else 0.0,
            "best_unit_precision_mean": float(np.mean(prec))   if prec.size   else 0.0,
            "n_ref_cells":              float(n_ref_cells),
            "n_cand_units":             float(len(u_can)),
        }
        for thr in self.thresholds:
            key = f"cells_ge_{int(round(thr*100)):02d}"
            values[key] = float((acc >= thr).sum())

        details = dict(
            per_cell_accuracy    ={g: float(acc[i])    for i, g in enumerate(ref_cells)},
            per_cell_recall      ={g: float(recall[i]) for i, g in enumerate(ref_cells)},
            per_cell_precision   ={g: float(prec[i])   for i, g in enumerate(ref_cells)},
            per_cell_best_unit   ={g: (int(best_unit_by_cell[g])
                                       if best_unit_by_cell[g] is not None else None)
                                    for g in ref_cells},
            per_cell_tp          ={g: int(tp[i])         for i, g in enumerate(ref_cells)},
            per_cell_gt_total    ={g: int(gt_total[i])   for i, g in enumerate(ref_cells)},
            per_cell_cand_total  ={g: int(cand_total[i]) for i, g in enumerate(ref_cells)},
        )
        return ScoreResult(scorer=self.name, corpus_item=candidate.corpus_item,
                           reference_adapter=reference.adapter,
                           candidate_adapter=candidate.adapter,
                           values=values, details=details)


_register(scorer_registry, PerCellScorer(), "per_cell")
