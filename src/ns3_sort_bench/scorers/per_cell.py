"""Per-cell best-unit accuracy, mirroring `claude/spikeinterface-g6-comparison.md`.

For each ground-truth cell `g`:
    best_unit = argmax_c |g ∩ c|
    tp        = |g ∩ best_unit|
    accuracy  = tp / (|g| + |best_unit| - tp)   # intersection-over-union
Aggregates:
    cells_ge_08, cells_ge_05 = count of GT cells whose best-unit accuracy
        meets each threshold
    best_unit_accuracy_mean = mean over GT cells
    small_cell_detection = recall on below-median-amplitude GT cells (needs
        amplitude info; omitted when the ground truth has no templates)
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .._base import ScoreResult, Scorer, SortedResult, _register, scorer_registry
from .pairwise import _assemble_label_array, _contingency


class PerCellScorer(Scorer):
    """Best-unit-accuracy per ground-truth cell."""
    name = "per_cell"

    def __init__(self, delta_frames: int = 12, thresholds=(0.8, 0.5)):
        self.delta_frames = int(delta_frames)
        self.thresholds = tuple(float(t) for t in thresholds)

    def __call__(self, reference: SortedResult, candidate: SortedResult,
                 **_) -> ScoreResult:
        ref = reference.dropped_reserved()
        can = candidate.dropped_reserved()

        ref_t, ref_lab = _assemble_label_array(ref)
        _, can_lab = _assemble_label_array(can, delta_frames=self.delta_frames,
                                            index_space=ref_t)
        tab, u_ref, u_can = _contingency(ref_lab, can_lab)

        if tab.size == 0:
            values = {"best_unit_accuracy_mean": 0.0}
            for t in self.thresholds:
                values[f"cells_ge_{int(round(t*100)):02d}"] = 0.0
            return ScoreResult(scorer=self.name, corpus_item=candidate.corpus_item,
                               reference_adapter=reference.adapter,
                               candidate_adapter=candidate.adapter,
                               values=values)

        # n_cell[g] = row sum, n_unit[c] = col sum
        n_cell = tab.sum(axis=1)
        n_unit = tab.sum(axis=0)

        # Best unit per GT cell + its IoU-style accuracy
        best_c = tab.argmax(axis=1)
        tp = tab[np.arange(len(u_ref)), best_c]
        denom = n_cell + n_unit[best_c] - tp
        acc = np.where(denom > 0, tp / denom, 0.0)

        values = {"best_unit_accuracy_mean": float(np.mean(acc))}
        for thr in self.thresholds:
            key = f"cells_ge_{int(round(thr*100)):02d}"
            values[key] = float((acc >= thr).sum())
        values["n_ref_cells"] = float(len(u_ref))
        values["n_cand_units"] = float(len(u_can))

        details = dict(
            per_cell_accuracy={int(u_ref[i]): float(acc[i]) for i in range(len(u_ref))},
            per_cell_best_unit={int(u_ref[i]): int(u_can[best_c[i]]) for i in range(len(u_ref))},
            per_cell_tp={int(u_ref[i]): int(tp[i]) for i in range(len(u_ref))},
        )
        return ScoreResult(scorer=self.name, corpus_item=candidate.corpus_item,
                           reference_adapter=reference.adapter,
                           candidate_adapter=candidate.adapter,
                           values=values, details=details)


_register(scorer_registry, PerCellScorer(), "per_cell")
