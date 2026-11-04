"""`_sorting_true_to_sorted_result` shifts GT unit ids by +2 so the Klusters
convention (0 = artefact, 1 = noise, real units at >= 2) does NOT swallow
paired-recording GT (which uses unit 1 for the single patched cell).

Regression for the hc1_fk_smoke run that reported all-zero metrics because
every GT unit was dropped by `dropped_reserved()`.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pytest

from ns3_sort_bench._base import SortedResult
from ns3_sort_bench.corpus.spikeforest import (
    _sorting_true_to_sorted_result,
    _GT_UNIT_ID_OFFSET,
)


class _FakeSorting:
    """Duck-typed SpikeInterface BaseSorting — unit ids + spike trains."""
    def __init__(self, trains):
        self._tr = {int(u): np.asarray(t, dtype=np.int64)
                     for u, t in trains.items()}

    def get_unit_ids(self) -> Iterable[int]:
        return list(self._tr)

    def get_unit_spike_train(self, u, segment_index=0):
        return self._tr[int(u)]


# ─── offset ────────────────────────────────────────────────────────────────
def test_default_offset_is_two():
    """Documents the default; everywhere the spec says `+2`, this is the
    number.  A change here needs the equivalent test in dropped_reserved()."""
    assert _GT_UNIT_ID_OFFSET == 2


def test_paired_gt_unit_1_becomes_unit_3():
    """Classic paired-recording case: one patched reference cell at unit 1."""
    sorting = _FakeSorting({1: [100, 200, 300, 400]})
    sr = _sorting_true_to_sorted_result(
        sorting, corpus_item="paired_crcns_hc1_d533101",
        sampling_rate=20000.0,
    )
    assert sr.unit_ids == [3]
    assert sr.spike_times[3].tolist() == [100, 200, 300, 400]
    assert sr.extras["unit_id_offset"] == 2
    assert sr.extras["unit_id_sf_original"] == {3: 1}


def test_gt_survives_dropped_reserved():
    """`SortedResult.dropped_reserved()` must keep every GT unit — this is
    the exact failure mode from the hc1_fk_smoke run."""
    sorting = _FakeSorting({0: [10], 1: [20, 30], 2: [40, 50, 60]})
    sr = _sorting_true_to_sorted_result(
        sorting, corpus_item="x", sampling_rate=1.0,
    )
    # Pre-offset: units 0, 1, 2 — all would be/survive reserved filter differently.
    # Post-offset: units 2, 3, 4 — all above the reserved threshold.
    assert sorted(sr.unit_ids) == [2, 3, 4]
    dropped = sr.dropped_reserved()
    assert sorted(dropped.unit_ids) == [2, 3, 4]
    # Total spike count preserved
    assert dropped.n_spikes() == 1 + 2 + 3


def test_multi_unit_synthetic_gt_preserves_order():
    """Synthetic sets (SYNTH_MEAREC_*) carry many GT units.  All get +2."""
    sorting = _FakeSorting({i: [10 * i, 10 * i + 1] for i in range(10)})
    sr = _sorting_true_to_sorted_result(
        sorting, corpus_item="synth", sampling_rate=1.0,
    )
    assert sr.unit_ids == list(range(2, 12))
    # Each unit's train is intact and still paired with the original id
    for shifted in sr.unit_ids:
        original = sr.extras["unit_id_sf_original"][shifted]
        assert original == shifted - 2
        assert sr.spike_times[shifted].tolist() == [10 * original,
                                                     10 * original + 1]


def test_custom_offset_passes_through():
    sorting = _FakeSorting({1: [5, 6, 7]})
    sr = _sorting_true_to_sorted_result(
        sorting, corpus_item="x", sampling_rate=1.0,
        unit_id_offset=100,
    )
    assert sr.unit_ids == [101]
    assert sr.extras["unit_id_offset"] == 100


def test_empty_gt_is_still_valid():
    sr = _sorting_true_to_sorted_result(
        _FakeSorting({}), corpus_item="x", sampling_rate=1.0,
    )
    assert sr.unit_ids == []
    assert sr.n_spikes() == 0
    assert sr.extras["unit_id_offset"] == 2
