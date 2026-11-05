"""SpikeInterface unit IDs must be shifted past Klusters' reserved IDs.

Scorers drop cluster IDs 0 (artefact) and 1 (noise/MUA).  SI sorters number
units from 0 (kilosort4, dartsort, tridesclous2, spykingcircus2) or 1
(mountainsort5); passing those IDs through raw discarded a sorter's first
one or two real units before scoring — dartsort's GT-matching unit on
d11222.002 (spikes −8 samples from GT, inside tolerance, n_match 16).
"""
from __future__ import annotations

import numpy as np


class _FakeSorting:
    def __init__(self, units):
        self._u = units                      # {unit_id: spike_train}

    def get_unit_ids(self):
        return np.array(list(self._u), dtype=object)

    def get_unit_spike_train(self, u):
        return np.asarray(self._u[u])


def test_zero_based_ids_are_shifted_past_reserved():
    from ns3_sort_bench._base import si_sorting_to_spike_times
    st, id_map = si_sorting_to_spike_times(_FakeSorting({
        0: [10, 20], 1: [30], 2: [40, 50, 60],
    }))
    assert sorted(st) == [2, 3, 4]
    assert min(st) >= 2
    assert id_map == {2: 0, 3: 1, 4: 2}
    assert list(st[2]) == [10, 20]           # unit 0 survives as ID 2


def test_one_based_ids_and_string_ids_also_safe():
    from ns3_sort_bench._base import si_sorting_to_spike_times
    st, _ = si_sorting_to_spike_times(_FakeSorting({1: [5], 2: [6]}))
    assert sorted(st) == [2, 3]              # MS5's unit 1 no longer dropped
    st, id_map = si_sorting_to_spike_times(_FakeSorting({"a": [1], "b": [2]}))
    assert sorted(st) == [2, 3] and id_map[2] == "a"


def test_shifted_units_survive_dropped_reserved_and_score():
    """End-to-end at the scorer: the GT-matching unit was SI unit 0."""
    from ns3_sort_bench._base import SortedResult, si_sorting_to_spike_times
    from ns3_sort_bench.scorers.pairwise import PairwiseScorer
    gt_t = np.arange(1000, 50_000, 1000)
    st, _ = si_sorting_to_spike_times(_FakeSorting({
        0: gt_t - 8,                         # the real cell, −8 samples
        1: np.arange(500, 50_000, 997),      # unrelated unit
    }))
    can = SortedResult(unit_ids=sorted(st), spike_times=st,
                       sampling_rate=20000.0, adapter="dartsort",
                       corpus_item="x")
    gt = SortedResult(unit_ids=[2], spike_times={2: gt_t},
                      sampling_rate=20000.0, adapter="gt", corpus_item="x")
    sr = PairwiseScorer(delta_frames=12)(gt, can)
    assert sr.values["n_matched_spikes"] == gt_t.size


def test_si_wrapped_contracts_declare_offset_2():
    from ns3_sort_bench.adapters import si_sorters, dartsort, slay
    for mod in (si_sorters, dartsort, slay):
        src = open(mod.__file__).read()
        assert "unit_id_offset=0" not in src, mod.__name__
