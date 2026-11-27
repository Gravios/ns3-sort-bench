"""The slay adapter's post-pass, without Kilosort 4: SLAy on a generated
recording whose units were split at random, and the merge bookkeeping."""
import numpy as np
import pytest

from ns3_sort_bench.adapters.slay import apply_merge_groups, slay_merge_groups


def test_apply_merge_groups_merges_into_lowest_id():
    st = {2: np.array([5, 1]), 3: np.array([4]), 7: np.array([9]), 9: np.array([2])}
    out = apply_merge_groups(st, [[7, 2], [9, 99]])
    assert sorted(out) == [2, 3, 9]
    assert out[2].tolist() == [1, 5, 9]
    assert out[9].tolist() == [2]


def test_slay_merge_groups_finds_planted_splits(tmp_path):
    pytest.importorskip("slay")
    pytest.importorskip("torch")
    from spikeinterface.core.generate import generate_ground_truth_recording
    recording, gt = generate_ground_truth_recording(
        durations=[300.0], sampling_frequency=30000.0, num_channels=32, num_units=10,
        seed=7, generate_sorting_kwargs=dict(firing_rates=8.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="on_the_fly"),
    )
    rng = np.random.default_rng(0)
    spike_times, pairs, next_id = {}, [], 100
    for u in gt.unit_ids:
        t = gt.get_unit_spike_train(u)
        if int(u) in (0, 1):
            half = rng.random(t.size) < 0.5
            spike_times[next_id], spike_times[next_id + 1] = t[half], t[~half]
            pairs.append({next_id, next_id + 1})
            next_id += 2
        else:
            spike_times[int(u) + 2] = t
    groups = slay_merge_groups(
        recording, spike_times, work_dir=str(tmp_path),
        similarity_type="l2", merge_parameters={"k1": 0.25, "k2": 1, "merge_threshold": 0.5},
    )
    found = [set(g) for g in groups]
    assert all(any(p <= f for f in found) for p in pairs), groups
    assert all(any(f <= p for p in pairs) for f in found), groups
    merged = apply_merge_groups(spike_times, groups)
    assert len(merged) == len(spike_times) - 2
