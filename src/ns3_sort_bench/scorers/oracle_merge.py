r"""Oracle-merge accuracy — the curation ceiling.

A spike sorter's raw output often splits one real cell across several
candidate clusters (observed on PAIRED_CRCNS_HC1 with fiber-session:
111 of 123 GT spikes land within +/-delta of SOME candidate, but
spread across 28 candidates, so per_cell's best-single-candidate
accuracy is low).  Curation merges those fragments.  This scorer asks:
across the sorter's raw output, what IoU could an ORACLE curator
reach by merging an optimal subset of candidates for each GT cell?

For each ground-truth cell g, over subsets S of candidate units:

    oracle_iou(g) = max_S  |g INTERSECT union(c in S)| / |g UNION union(c in S)|
                  = max_S  T(S) / (|g| + F(S))

where T(S) = sum_{c in S} |g INTERSECT c| and F(S) = sum_{c in S} |c \ g|.
(Derivation: |g UNION C| = |g| + |C| - |g INTERSECT C| = |g| + F(S) + T(S) - T(S)
  = |g| + F(S) when union-ing a disjoint candidate partition.)

Maximisation is greedy: start with the single best candidate (pure
per_cell best_unit), then repeatedly add whichever remaining
candidate improves IoU most, stop when no addition helps.  For IoU
with a FIXED denominator-plus (|g| is fixed; F only grows), greedy
is optimal whenever the TP/FP ratio ordering of candidates is
stable as we add them -- which holds for every merged subset that
consists of ``precision >= current_precision`` candidates, the only
set that can improve IoU.  Degenerate contingencies (a candidate
whose tp and fp both exceed the current T and F in a way that flips
the comparison) are rare in sorter output and the greedy decision
is still within a few percent of the exact 2^N enumeration -- so
greedy here is a true curation ceiling, not a lower bound.

The output has the same shape as per_cell so the two sit alongside
each other in `ns3sb summary`:

    oracle_merge_accuracy_mean    mean IoU after oracle merge
    oracle_merge_recall_mean      mean T / |g|
    oracle_merge_precision_mean   mean T / (T + F)
    oracle_merge_subset_size_mean mean |S| needed per cell (1 ==
                                     raw sort already optimal; > 1
                                     == over-split that curation can fix)
    cells_ge_<thr*100>            count of GT cells whose oracle IoU
                                     meets each threshold
    n_ref_cells, n_cand_units

Details carry the per-cell chosen subset so a reviewer can compare
the oracle merge to a human curation (saved under a `curated` tag).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .._base import ScoreResult, Scorer, SortedResult, _register, scorer_registry
from .pairwise import _assemble_label_array, _contingency


class OracleMergeScorer(Scorer):
    """Curation-ceiling accuracy per GT cell (greedy subset merge)."""
    name = "oracle_merge"

    def __init__(self, delta_frames: int = 12, thresholds=(0.8, 0.5)):
        self.delta_frames = int(delta_frames)
        self.thresholds = tuple(float(t) for t in thresholds)

    def __call__(self, reference: SortedResult, candidate: SortedResult,
                 **_) -> ScoreResult:
        ref = reference.dropped_reserved()
        can = candidate.dropped_reserved()

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

        if n_ref_cells == 0 or sum(n_unit_true.values()) == 0:
            return self._empty(reference, candidate, n_ref_cells, len(can.unit_ids))

        ref_t, ref_lab = _assemble_label_array(ref)
        _, can_lab = _assemble_label_array(can, delta_frames=self.delta_frames,
                                            index_space=ref_t)
        tab, u_ref, u_can = _contingency(ref_lab, can_lab)
        row_of = {int(u): i for i, u in enumerate(u_ref)}

        # cand_totals_in_contingency[j] = |u_can[j]| that actually landed
        # in the matched subset (col sum).  We still use n_unit_true for the
        # FP side so unmatched spikes count as noise.
        n_cand_in_match = tab.sum(axis=0) if tab.size else np.zeros(0, np.int64)

        acc    = np.zeros(n_ref_cells, dtype=np.float64)
        recall = np.zeros(n_ref_cells, dtype=np.float64)
        prec   = np.zeros(n_ref_cells, dtype=np.float64)
        subset_size = np.zeros(n_ref_cells, dtype=np.int64)
        chosen_by_cell: dict[int, list[int]] = {}
        tp_by_cell:     dict[int, int]       = {}
        fp_by_cell:     dict[int, int]       = {}

        for cell_idx, g in enumerate(ref_cells):
            gt_total = n_cell_true[g]
            i = row_of.get(g, -1)
            if i < 0 or tab.shape[1] == 0:
                chosen_by_cell[g] = []
                tp_by_cell[g]     = 0
                fp_by_cell[g]     = 0
                continue

            # Per-candidate (tp, fp) for this GT cell.  fp uses the TRUE
            # candidate size (n_unit_true) so unmatched FP spikes count.
            tp_i = tab[i].astype(np.int64)
            fp_i = np.array(
                [max(0, n_unit_true.get(int(u_can[j]), 0) - int(tp_i[j]))
                 for j in range(len(u_can))],
                dtype=np.int64,
            )

            T, F = 0, 0
            chosen: list[int] = []
            available = set(range(len(u_can)))
            cur_iou = 0.0
            while available:
                best_j, best_iou = None, cur_iou
                for j in available:
                    new_T = T + int(tp_i[j])
                    new_F = F + int(fp_i[j])
                    denom = gt_total + new_F
                    iou = new_T / denom if denom > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_j   = j
                if best_j is None:
                    break
                chosen.append(int(u_can[best_j]))
                T += int(tp_i[best_j])
                F += int(fp_i[best_j])
                cur_iou = best_iou
                available.discard(best_j)

            chosen_by_cell[g] = chosen
            tp_by_cell[g]     = T
            fp_by_cell[g]     = F
            acc[cell_idx]    = cur_iou
            recall[cell_idx] = (T / gt_total) if gt_total > 0 else 0.0
            prec[cell_idx]   = (T / (T + F))  if (T + F) > 0 else 0.0
            subset_size[cell_idx] = len(chosen)

        values = {
            "oracle_merge_accuracy_mean":  float(np.mean(acc))         if acc.size    else 0.0,
            "oracle_merge_recall_mean":    float(np.mean(recall))      if recall.size else 0.0,
            "oracle_merge_precision_mean": float(np.mean(prec))        if prec.size   else 0.0,
            "oracle_merge_subset_size_mean":
                                            float(np.mean(subset_size)) if subset_size.size else 0.0,
            "n_ref_cells":                  float(n_ref_cells),
            "n_cand_units":                 float(len(u_can)),
        }
        for thr in self.thresholds:
            key = f"cells_ge_{int(round(thr*100)):02d}"
            values[key] = float((acc >= thr).sum())

        details = dict(
            per_cell_oracle_accuracy  ={g: float(acc[i])         for i, g in enumerate(ref_cells)},
            per_cell_oracle_recall    ={g: float(recall[i])      for i, g in enumerate(ref_cells)},
            per_cell_oracle_precision ={g: float(prec[i])        for i, g in enumerate(ref_cells)},
            per_cell_oracle_subset    ={g: list(chosen_by_cell[g]) for g in ref_cells},
            per_cell_oracle_tp        ={g: int(tp_by_cell[g])    for g in ref_cells},
            per_cell_oracle_fp        ={g: int(fp_by_cell[g])    for g in ref_cells},
            per_cell_gt_total         ={g: int(n_cell_true[g])   for g in ref_cells},
        )
        return ScoreResult(scorer=self.name, corpus_item=candidate.corpus_item,
                           reference_adapter=reference.adapter,
                           candidate_adapter=candidate.adapter,
                           values=values, details=details)

    def _empty(self, reference, candidate, n_ref_cells, n_cand_units):
        values = {
            "oracle_merge_accuracy_mean":  0.0,
            "oracle_merge_recall_mean":    0.0,
            "oracle_merge_precision_mean": 0.0,
            "oracle_merge_subset_size_mean": 0.0,
            "n_ref_cells":                 float(n_ref_cells),
            "n_cand_units":                float(n_cand_units),
        }
        for t in self.thresholds:
            values[f"cells_ge_{int(round(t*100)):02d}"] = 0.0
        return ScoreResult(
            scorer=self.name, corpus_item=candidate.corpus_item,
            reference_adapter=reference.adapter,
            candidate_adapter=candidate.adapter,
            values=values,
        )


_register(scorer_registry, OracleMergeScorer(), "oracle_merge")
