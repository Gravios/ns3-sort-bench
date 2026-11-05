"""Regression tests for PairwiseScorer's nearest-neighbour bipartite
match.  Before patch 0029, `_assemble_label_array` used a first-come-
first-matched walk: for each candidate spike in time order, it tried
to claim its nearest GT slot; later candidates that happened to be
closer to that GT lost out.  On a dense noise-dominated sort against
sparse GT (hc1 curated: 5551 candidates vs 123 GT) the walk order
let noise clusters lock out the real cell from most slots, collapsing
`n_matched_spikes` and `n_cand_units` to almost nothing even though
the real cell was fully present.  per_cell's best-unit metrics
disagreed with pairwise's `n_match` by two orders of magnitude.

The fix is a slot-driven greedy nearest-neighbour bipartite match:
for each GT slot find its nearest unused candidate within tolerance.
This agrees with per_cell's math and with the inspect-clu confusion
matrix.
"""
from __future__ import annotations

import numpy as np

from ns3_sort_bench._base import SortedResult
from ns3_sort_bench.scorers.pairwise import PairwiseScorer
from ns3_sort_bench.scorers.per_cell import PerCellScorer


def _sr(unit_to_times, *, adapter="x"):
    return SortedResult(
        unit_ids=sorted(unit_to_times),
        spike_times={int(u): np.asarray(t, np.int64)
                     for u, t in unit_to_times.items()},
        sampling_rate=20000.0,
        adapter=adapter, corpus_item="test",
    )


def test_pairwise_matches_closer_candidate_not_first_in_time_order():
    """Two candidates land within tolerance of each GT spike; the
    closer one must win regardless of its position in the sorted
    candidate time list.

    Without the fix: the EARLIER candidate (noise cluster 5 at
    GT-5 samples) claimed the slot, leaving the CLOSER real-cell
    candidate (cluster 7 at GT+1) unmatched.  Then n_cand=1 and the
    "winning" unit was the noise cluster — exactly the wrong cell.

    With the fix: each GT slot picks its nearest unused candidate
    bipartitely, so cluster 7 wins every slot (distance 1 < 5).
    """
    gt = _sr({3: [100, 500, 900]}, adapter="gt")
    # Noise cluster 5 arrives in time order BEFORE the real cell 7 at
    # each slot, but is farther away.
    can = _sr({
        5: [95,  495, 895],   # noise, distance 5 — earlier in time
        7: [101, 501, 901],   # real cell, distance 1 — later in time
    }, adapter="sorter")
    sr = PairwiseScorer(delta_frames=12)(gt, can)
    assert sr.values["n_matched_spikes"] == 3.0
    # The real cell's cluster claimed every slot; only one surviving
    # candidate unit in the contingency.
    assert sr.values["n_cand_units"] == 1.0


def test_pairwise_n_match_agrees_with_per_cell_best_unit_count():
    """The number of matched spikes pairwise reports as
    `n_matched_spikes` must equal the number of GT spikes paired with
    SOME candidate; per_cell's `best_unit_recall_mean` × n_GT must
    equal `n_matched_spikes` when there is one GT cell.  Before the
    fix these disagreed by 25× on hc1 curated."""
    np.random.seed(0)
    gt_t = np.linspace(100, 60_000, 123, dtype=np.int64)
    real_ix = np.sort(np.random.choice(123, 85, replace=False))
    real_t = gt_t[real_ix] + 1

    # Noise: 5000 random spikes FAR from GT.
    noise_t = np.random.randint(0, 65_000, 10_000).astype(np.int64)
    far = np.min(np.abs(noise_t[:, None] - gt_t[None, :]), axis=1) > 12
    noise_t = noise_t[far][:5000]

    gt = _sr({3: gt_t}, adapter="gt")
    can = _sr({7: real_t, 5: noise_t}, adapter="sorter")

    pw = PairwiseScorer(delta_frames=12)(gt, can)
    pc = PerCellScorer(delta_frames=12)(gt, can)

    n_match = int(pw.values["n_matched_spikes"])
    tp_from_pc = int(round(pc.values["best_unit_recall_mean"] * 123))

    assert n_match == tp_from_pc == 85, (
        f"pairwise n_match={n_match}, per_cell implied tp={tp_from_pc}; "
        "should agree at 85 (the 85 real-cell spikes near GT)."
    )


def test_registered_pairwise_uses_near_match_tolerance():
    """The `pairwise` scorer in the registry MUST default to
    delta_frames=12 (~0.6 ms at 20 kHz) — on an independent-detection
    sort against juxta GT, exact-index matching (delta=0) collapses
    n_matched_spikes to coincidental matches (4 of 123 on hc1) and
    makes the pairwise metrics useless.

    If someone adds a new registration without delta_frames, this test
    catches the regression.  `pairwise_exact` is the explicit opt-in
    for shared-detection scoring and must retain delta=0.
    """
    from ns3_sort_bench._base import scorer_registry
    from ns3_sort_bench import scorers as _  # noqa — populate registry

    assert scorer_registry["pairwise"].delta_frames == 12
    assert scorer_registry["pairwise_exact"].delta_frames == 0
    # Back-compat alias keeps working for plans that named it.
    assert scorer_registry["pairwise_delta12"].delta_frames == 12


def test_pairwise_match_drops_reserved_before_match():
    """Candidates in Klusters-reserved clusters (0/1) must be dropped
    BEFORE the nearest-neighbour match runs — otherwise a reserved
    spike at GT±1 could steal the match from a real cluster at GT±3,
    and the real cluster would then not contribute to n_match.
    """
    gt = _sr({3: [100]}, adapter="gt")
    can = _sr({
        1: [99],       # reserved noise at distance 1 — must be dropped
        7: [103],      # real cell at distance 3 — the real match
    }, adapter="sorter")

    sr = PairwiseScorer(delta_frames=12, drop_reserved=True)(gt, can)
    assert sr.values["n_matched_spikes"] == 1.0
    assert sr.values["n_cand_units"] == 1.0
