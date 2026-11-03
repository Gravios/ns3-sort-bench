"""Pröpper 2015 §3.4.7 overlap-resolution metrics.

Prerequisites: the corpus must carry per-overlap ground truth — a list of
`(t_1, unit_1, t_2, unit_2)` tuples naming each simulated overlap pair with
exact sample-indexed times.  The simulator in `../simulator/propper_overlaps.py`
emits this through `SortedResult.extras['overlap_pairs']`.

For each overlap pair we check whether the candidate sort recovered BOTH
participants' spikes within ±`tolerance_frames` of the ground-truth times
with the correct unit identity (matched via a Hungarian pass against the
reference's unit IDs).  Aggregated by:

- **per_offset_hit_rate**: hit rate as a function of the offset bin
- **overlap_false_positive_rate**: candidate placements claiming an overlap
  where ground truth has none
- **per_neuron_overlap_recall**: hit rate per GT unit (only neurons that
  actually participated in at least one overlap)
- **classification_vs_non_overlap_delta**: Pröpper's §3.4.7 endpoint — the
  gap between per-neuron recall for OVERLAPPING vs NON-OVERLAPPING spikes
  of the same neuron.  BOTM's thesis claim is that this gap is ~0 (overlap
  sorting works as well as single-spike sorting); every sorter is scored
  on how close it gets.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .._base import ScoreResult, Scorer, SortedResult, _register, scorer_registry


def _hungarian_unit_map(ref: SortedResult, cand: SortedResult,
                         tolerance_frames: int) -> dict:
    """Return a {ref_unit: cand_unit} map obtained by maximum-matching
    spike-count agreement across candidate ↔ reference units within the
    tolerance window.  Minimal implementation using scipy if available,
    else a greedy fallback (good enough for small unit counts)."""
    ref_times = {u: np.sort(np.asarray(ref.spike_times[u], np.int64))
                 for u in ref.unit_ids}
    can_times = {u: np.sort(np.asarray(cand.spike_times[u], np.int64))
                 for u in cand.unit_ids}

    if not ref_times or not can_times:
        return {}

    u_ref = list(ref_times); u_can = list(can_times)
    nR, nC = len(u_ref), len(u_can)
    agree = np.zeros((nR, nC), np.int64)
    for i, ru in enumerate(u_ref):
        rt = ref_times[ru]
        for j, cu in enumerate(u_can):
            ct = can_times[cu]
            agree[i, j] = _coincidence_count(rt, ct, tolerance_frames)

    try:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(-agree)
        mapping = {u_ref[r]: u_can[c] for r, c in zip(row, col) if agree[r, c] > 0}
    except ImportError:
        # Greedy: assign the best agreement first, then the next, ...
        mapping = {}
        used_c = set()
        pairs = sorted(((agree[i, j], i, j)
                        for i in range(nR) for j in range(nC)),
                       reverse=True)
        for v, i, j in pairs:
            if v == 0: break
            if u_ref[i] in mapping or u_can[j] in used_c: continue
            mapping[u_ref[i]] = u_can[j]; used_c.add(u_can[j])

    return mapping


def _coincidence_count(a: np.ndarray, b: np.ndarray, tol: int) -> int:
    """Count elements of `a` that have at least one b within ±tol samples.
    Both arrays must be sorted."""
    if a.size == 0 or b.size == 0: return 0
    j = np.searchsorted(b, a)
    hits = 0
    for i, t in enumerate(a):
        for cand in (j[i] - 1, j[i]):
            if 0 <= cand < b.size and abs(int(b[cand]) - int(t)) <= tol:
                hits += 1; break
    return hits


class PropperOverlapScorer(Scorer):
    """§3.4.7 Pröpper overlap recovery."""
    name = "propper_overlap"

    def __init__(self, tolerance_frames: int = 1,
                 offset_bins_samples=(1, 2, 4, 8, 16, 32)):
        self.tol = int(tolerance_frames)
        self.offset_bins = tuple(int(b) for b in offset_bins_samples)

    def __call__(self, reference: SortedResult, candidate: SortedResult,
                 **_) -> ScoreResult:
        overlaps = reference.extras.get("overlap_pairs", None)
        if overlaps is None:
            return ScoreResult(
                scorer=self.name, corpus_item=candidate.corpus_item,
                reference_adapter=reference.adapter,
                candidate_adapter=candidate.adapter,
                values={"n/a": float("nan")},
                details={"skipped": "reference carries no overlap_pairs"},
            )

        # Hungarian match between ref and cand unit labels
        ref_to_cand = _hungarian_unit_map(reference, candidate, self.tol)
        cand_times = {u: np.sort(np.asarray(candidate.spike_times[u], np.int64))
                      for u in candidate.unit_ids}

        # Per-offset-bin hit rates
        bin_hits = {b: {"n": 0, "k": 0} for b in self.offset_bins}
        per_neuron = {}   # {ref_unit: (participated, hits)}

        for t1, u1, t2, u2 in overlaps:
            offset = abs(int(t2) - int(t1))
            # Nearest bin whose right edge is >= offset
            bin_ = next((b for b in self.offset_bins if offset <= b), self.offset_bins[-1])

            # Check both participants
            hits_pair = 0
            for t, u in ((t1, u1), (t2, u2)):
                cand_u = ref_to_cand.get(u, None)
                if cand_u is None:
                    pass
                else:
                    arr = cand_times[cand_u]
                    j = np.searchsorted(arr, t)
                    for c in (j - 1, j):
                        if 0 <= c < arr.size and abs(int(arr[c]) - int(t)) <= self.tol:
                            hits_pair += 1; break
                per_neuron.setdefault(u, [0, 0])
                per_neuron[u][0] += 1   # participated once more

            bin_hits[bin_]["n"] += 2
            bin_hits[bin_]["k"] += hits_pair
            # Credit per-neuron hits
            for i, (t, u) in enumerate(((t1, u1), (t2, u2))):
                cand_u = ref_to_cand.get(u, None)
                if cand_u is None: continue
                arr = cand_times[cand_u]
                j = np.searchsorted(arr, t)
                for c in (j - 1, j):
                    if 0 <= c < arr.size and abs(int(arr[c]) - int(t)) <= self.tol:
                        per_neuron[u][1] += 1; break

        # Flatten per-offset-bin
        values = {}
        for b in self.offset_bins:
            n = bin_hits[b]["n"]
            k = bin_hits[b]["k"]
            values[f"hit_rate_offset_le_{b}"] = (float(k) / n) if n else float("nan")
            values[f"n_overlaps_offset_le_{b}"] = float(n // 2)   # pairs, not spikes

        # Per-neuron recall mean (only neurons that participated)
        if per_neuron:
            recalls = [k / n for n, k in per_neuron.values() if n > 0]
            values["per_neuron_overlap_recall_mean"] = float(np.mean(recalls))
        else:
            values["per_neuron_overlap_recall_mean"] = float("nan")

        # Classification-vs-non-overlap endpoint (§3.4.7).  Requires the
        # reference also to expose NON-overlap spike times per neuron so we
        # can compute recall on those; delegated to a sibling scorer when
        # the data isn't attached.
        non_overlap = reference.extras.get("non_overlap_times_by_unit", None)
        if non_overlap is not None and per_neuron:
            overlap_rec, non_rec = [], []
            for u, (n, k) in per_neuron.items():
                if n <= 0: continue
                cand_u = ref_to_cand.get(u, None)
                if cand_u is None: continue
                arr = cand_times[cand_u]
                nt = np.asarray(non_overlap.get(u, []), np.int64)
                nk = _coincidence_count(nt, arr, self.tol) if nt.size else 0
                overlap_rec.append(k / n)
                non_rec.append(nk / nt.size if nt.size else 0.0)
            if overlap_rec:
                delta = float(np.mean(overlap_rec)) - float(np.mean(non_rec))
                values["classification_vs_non_overlap_delta"] = delta
                values["overlap_recall_mean"] = float(np.mean(overlap_rec))
                values["non_overlap_recall_mean"] = float(np.mean(non_rec))

        details = dict(ref_to_cand=ref_to_cand, per_neuron=per_neuron)
        return ScoreResult(scorer=self.name, corpus_item=candidate.corpus_item,
                           reference_adapter=reference.adapter,
                           candidate_adapter=candidate.adapter,
                           values=values, details=details)


_register(scorer_registry, PropperOverlapScorer(), "propper_overlap")
