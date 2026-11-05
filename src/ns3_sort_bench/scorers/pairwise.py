"""Pairwise clustering metrics.  Mirrors `claude/fiber-kit-audit-cba.md`:

- **ARI** (adjusted Rand index) on GT>1 spikes
- **Purity** = Σ_c max_g |c ∩ g| / N (weighted dominant-parent per machine cluster)
- **Completeness** = Σ_g max_c |c ∩ g| / N (weighted largest-machine-cluster
  per GT parent)
- **Directed move audit** — on edits (optional, only emitted when the
  reference and the candidate share the detection, i.e. identical spike
  sample sets).  A moved spike is `fixed` if its new label's dominant GT
  parent matches its own and the old didn't; `broke` for the reverse.

Spike alignment: by default spikes with the same sample index are considered
the same spike (shared-detection runs).  For independent-detection runs,
pass `delta_frames` so near-coincident spikes count as the same event; this
degrades to the SpikeInterface comparison semantics.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .._base import ScoreResult, Scorer, SortedResult, _register, scorer_registry


def _assemble_label_array(sr: SortedResult, delta_frames: int = 0,
                          index_space: Optional[np.ndarray] = None):
    """Return parallel arrays `(times, labels)` of length n_total_spikes
    (not n_unique_times), with `labels[i]` = unit id of the i-th spike.
    When `index_space` is given, labels are projected onto its grid by
    nearest-match within `delta_frames` — unmatched slots get label -1.

    Projection is a greedy nearest-neighbour BIPARTITE match: each slot
    in `index_space` claims its nearest unused candidate within tolerance,
    and each candidate matches at most one slot.  The walk direction is
    slot-driven (not candidate-time-driven) so that on dense candidate
    streams against sparse reference grids (e.g. 5 000 sort spikes vs
    100 juxta-GT spikes) the match counts the real nearest-neighbour
    pairs instead of being dominated by whichever candidate happened to
    arrive first in time order.  Previously the direction was reversed,
    which collapsed n_matched_spikes / n_cand_units on sorts that
    over-cluster noise — see the hc1 curated case in the Nov-5 log.
    """
    # Flatten into (time, unit)
    times, labels = [], []
    for u in sr.unit_ids:
        t = np.asarray(sr.spike_times[u], np.int64)
        times.append(t); labels.append(np.full(t.size, int(u), np.int64))
    times = np.concatenate(times) if times else np.array([], np.int64)
    labels = np.concatenate(labels) if labels else np.array([], np.int64)
    order = np.argsort(times, kind="stable")
    times, labels = times[order], labels[order]

    if index_space is None:
        return times, labels

    # Project onto index_space by greedy nearest-neighbour bipartite match.
    ix = np.full(len(index_space), -1, np.int64)
    if times.size == 0 or len(index_space) == 0:
        return index_space, ix

    index_space = np.asarray(index_space, np.int64)
    taken = np.zeros(times.size, dtype=bool)
    # For each slot in index_space, find its insertion point within the
    # (sorted) candidate times — the two candidates at j-1 and j span
    # the slot.  GT spikes are much sparser than the detection grid in
    # every realistic use (|index_space| << |times|), so the near-
    # neighbour check is always among those two.
    j = np.searchsorted(times, index_space)
    for i, tref in enumerate(index_space):
        best_k = -1; best_d = delta_frames + 1
        # Walk a few neighbours in each direction in case the closer
        # one is already taken; stop as soon as we see a candidate
        # strictly outside tolerance (times is sorted, so there's no
        # point continuing).
        # Left side:
        k = j[i] - 1
        while 0 <= k < times.size:
            d = int(tref) - int(times[k])
            if d > delta_frames:
                break
            if not taken[k] and d < best_d:
                best_k = k; best_d = d
                # The closest-unused-on-the-left is the best candidate
                # on this side; no earlier left-neighbour can be closer.
                break
            k -= 1
        # Right side:
        k = j[i]
        while 0 <= k < times.size:
            d = int(times[k]) - int(tref)
            if d > delta_frames:
                break
            if not taken[k] and d < best_d:
                best_k = k; best_d = d
                break
            k += 1
        if best_k >= 0:
            ix[i] = int(labels[best_k])
            taken[best_k] = True
    return index_space, ix


def _contingency(ref_lab: np.ndarray, cand_lab: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the contingency table C[g, c] = |{i : ref=g, cand=c}|.  Returns
    (table, unique_ref, unique_cand).  Ignores slots where either label is
    -1 (unmatched)."""
    mask = (ref_lab >= 0) & (cand_lab >= 0)
    r, c = ref_lab[mask], cand_lab[mask]
    if r.size == 0:
        return np.zeros((0, 0), np.int64), np.array([], np.int64), np.array([], np.int64)
    u_ref, r_ix = np.unique(r, return_inverse=True)
    u_can, c_ix = np.unique(c, return_inverse=True)
    tab = np.zeros((u_ref.size, u_can.size), np.int64)
    np.add.at(tab, (r_ix, c_ix), 1)
    return tab, u_ref, u_can


def _ari(tab: np.ndarray) -> float:
    """Adjusted Rand index from a contingency table.  Standard formula
    (Hubert & Arabie 1985)."""
    if tab.size == 0:
        return 0.0
    n = int(tab.sum())
    if n <= 1:
        return 0.0
    a = tab.sum(axis=1); b = tab.sum(axis=0)
    sum_comb_c = (tab * (tab - 1) // 2).sum()
    sum_comb_a = (a * (a - 1) // 2).sum()
    sum_comb_b = (b * (b - 1) // 2).sum()
    total_comb = n * (n - 1) // 2
    expected = (sum_comb_a * sum_comb_b) / total_comb if total_comb else 0.0
    max_ = (sum_comb_a + sum_comb_b) / 2
    denom = max_ - expected
    if denom == 0:
        return 1.0 if sum_comb_c == expected else 0.0
    return float((sum_comb_c - expected) / denom)


def _purity(tab: np.ndarray) -> float:
    """Weighted dominant-parent fraction, Σ_c max_g C[g, c] / N."""
    if tab.size == 0: return 0.0
    n = int(tab.sum())
    return float(tab.max(axis=0).sum() / n) if n else 0.0


def _completeness(tab: np.ndarray) -> float:
    """Weighted largest-machine-cluster per GT parent, Σ_g max_c C[g, c] / N."""
    if tab.size == 0: return 0.0
    n = int(tab.sum())
    return float(tab.max(axis=1).sum() / n) if n else 0.0


class PairwiseScorer(Scorer):
    """ARI / purity / completeness on shared or near-shared detections."""
    name = "pairwise"

    def __init__(self, delta_frames: int = 0, drop_reserved: bool = True,
                 include_directed_move_audit: bool = False):
        self.delta_frames = int(delta_frames)
        self.drop_reserved = bool(drop_reserved)
        self.include_dma = bool(include_directed_move_audit)

    def __call__(self, reference: SortedResult, candidate: SortedResult,
                 **_) -> ScoreResult:
        ref = reference.dropped_reserved() if self.drop_reserved else reference
        can = candidate.dropped_reserved() if self.drop_reserved else candidate

        # Align candidate onto the ref's spike-time grid.  If delta_frames
        # == 0 this is a pure index-join (shared-detection run); otherwise
        # it's a near-match within tolerance (independent-detection run).
        ref_t, ref_lab = _assemble_label_array(ref)
        _, can_lab = _assemble_label_array(can, delta_frames=self.delta_frames,
                                            index_space=ref_t)

        tab, u_ref, u_can = _contingency(ref_lab, can_lab)
        # Total non-reserved candidate spike count, pre-near-match-projection.
        # Useful as a sanity check that the detector / sorter actually
        # produced events at all — a sort that writes 0 spikes (botm_ref
        # when its SIC-B loop can't bootstrap, or a misconfigured sorter)
        # has `n_cand_spikes = 0`, which stands out next to a working
        # sort's thousand-plus count.  Unlike `n_matched_spikes` (which
        # only counts GT-overlapping spikes), this is independent of GT
        # density and reports what the sort itself did.
        n_cand_spikes = int(sum(int(np.asarray(can.spike_times[u]).size)
                                 for u in can.unit_ids))
        values = dict(
            ari=_ari(tab),
            purity=_purity(tab),
            completeness=_completeness(tab),
            n_ref_units=float(u_ref.size),
            n_cand_units=float(u_can.size),
            n_matched_spikes=float(int(tab.sum())),
            n_cand_spikes=float(n_cand_spikes),
            n_ref_spikes=float(ref_lab.size),
        )

        details = dict(
            contingency_shape=tuple(tab.shape),
            delta_frames=self.delta_frames,
        )

        if self.include_dma and self.delta_frames == 0 and len(ref_t) == len(_assemble_label_array(can)[0]):
            # Directed move audit only valid when sorts share detection
            # exactly (same spike times in the same order).  Compute
            # fixed/broke counts: a spike's directed move is
            # fixed  if cand's dominant-GT-parent on its new label matches its own GT
            #                AND ref's dominant-GT-parent on its old label did NOT.
            fixed, broke = _directed_move_audit(ref_lab, can_lab, tab, u_ref, u_can)
            values["dma_fixed"] = float(fixed)
            values["dma_broke"] = float(broke)
            values["dma_ratio"] = float(fixed / broke) if broke else float("inf")

        return ScoreResult(
            scorer=self.name, corpus_item=candidate.corpus_item,
            reference_adapter=reference.adapter,
            candidate_adapter=candidate.adapter,
            values=values, details=details,
        )


def _directed_move_audit(ref_lab, cand_lab, tab, u_ref, u_can):
    """Count fixed vs broke moves.  Returns (fixed, broke).  Interprets ref
    as the ground truth; `cand` is the sorter's output; a 'move' is any
    spike whose candidate label differs from its ref label, scored by
    whether the new label is 'cleaner' w.r.t. the GT parent.

    Dominant GT parent of ref label g = argmax_g' C[g', g]; same for cand
    labels against the GT.  Here ref IS the GT so a ref label's dominant
    parent is itself; cand's dominant parent is argmax over the column.
    """
    if tab.size == 0 or u_ref.size == 0 or u_can.size == 0:
        return 0, 0
    # dominant_gt_of_cand[c] = u_ref[argmax tab[:, c]]
    arg_c = tab.argmax(axis=0)
    dominant_gt_of_cand = u_ref[arg_c]
    # Build lookup tables
    ref_to_dom = {int(g): int(g) for g in u_ref}
    cand_to_dom = {int(c): int(dominant_gt_of_cand[i]) for i, c in enumerate(u_can)}

    fixed = broke = 0
    for r, c in zip(ref_lab, cand_lab):
        if r < 0 or c < 0:
            continue
        if int(r) == int(c):
            continue   # not moved
        old_dom = ref_to_dom.get(int(r), -1)
        new_dom = cand_to_dom.get(int(c), -1)
        own_gt = int(r)
        if new_dom == own_gt and old_dom != own_gt:
            fixed += 1
        elif new_dom != own_gt and old_dom == own_gt:
            broke += 1
    return fixed, broke


# Default "pairwise" uses a 12-sample near-match tolerance (~0.6 ms at
# 20 kHz, ~0.4 ms at 32 kHz).  Independent-detection sorts (the common
# case — GT juxta vs extracellular sort) have 1-5-sample timing offsets
# between the two detection modes, so delta=0 (exact index match) sees
# essentially no co-occurrences and collapses n_matched_spikes to zero
# for any realistic sort against juxta GT.  The 12-sample tolerance
# matches what per_cell uses by default, so the two scorers' contingency
# tables agree.  An explicit "pairwise_exact" registration is kept for
# the shared-detection scoring path (fiber_kit with
# shared_detection=True, propper_sim, hybrid corpora) where the sort
# is built on top of the GT spike indices and exact-match is correct.
_register(scorer_registry, PairwiseScorer(delta_frames=12), "pairwise")
_register(scorer_registry, PairwiseScorer(delta_frames=0),  "pairwise_exact")
# Back-compat: the old name still resolves to the near-match scorer
# (same instance a caller gets via "pairwise") so any plan that lists
# "pairwise_delta12" by name keeps working.
_register(scorer_registry, PairwiseScorer(delta_frames=12), "pairwise_delta12")
