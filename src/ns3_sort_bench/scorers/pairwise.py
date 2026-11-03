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
    Used to align independent detections onto a common axis."""
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

    # Project onto index_space by nearest-time within delta_frames
    ix = np.full(len(index_space), -1, np.int64)
    if times.size == 0 or len(index_space) == 0:
        return index_space, ix

    j = np.searchsorted(index_space, times)
    for k, t in enumerate(times):
        # Check j[k] and j[k]-1 for closest within tolerance
        best = -1; best_d = delta_frames + 1
        for cand in (j[k] - 1, j[k]):
            if 0 <= cand < len(index_space):
                d = abs(int(index_space[cand]) - int(t))
                if d <= delta_frames and d < best_d:
                    best = cand; best_d = d
        if best >= 0 and ix[best] == -1:
            # First-come-first-matched; a strict ground-truth alignment uses
            # the Hungarian match offered by SpikeInterface, which is more
            # expensive and only matters when candidate spike densities are
            # much higher than ref.
            ix[best] = int(labels[k])
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
        values = dict(
            ari=_ari(tab),
            purity=_purity(tab),
            completeness=_completeness(tab),
            n_ref_units=float(u_ref.size),
            n_cand_units=float(u_can.size),
            n_matched_spikes=float(int(tab.sum())),
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


_register(scorer_registry, PairwiseScorer(), "pairwise")
_register(scorer_registry, PairwiseScorer(delta_frames=12),
          "pairwise_delta12")   # ~0.4 ms @ 32 kHz
