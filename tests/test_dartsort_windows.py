"""DARTsort window lengths stay consistent at any sampling rate.

The helpers below copy DARTsort 0.5.31's rounding (util/internal_config.py
WaveformConfig and templates/realignment.py trim_templates_to_shift) so the
test runs without dartsort or torch installed.
"""
import math

import pytest

from ns3_sort_bench.adapters.dartsort import _WINDOW_DEFAULTS, snap_windows_to_samples


def _m2s(ms, fs):
    return math.floor(float(ms) * float(fs) / 1000.0 + 0.5)


def _length(before, after, fs):
    n = _m2s(before + after, fs)
    return 2 * (n // 2) + 1


def _realign_trimmed_length(before, after, shift, fs):
    padded = _length(before + shift, after + shift, fs)
    max_shift = _m2s(before + shift, fs) - _m2s(before, fs)
    return padded - 2 * max_shift


def _svd_slice_length(before, after, pad, fs):
    my_len, my_trough = _length(before, after, fs), _m2s(before, fs)
    o_len, o_trough = _length(before + pad, after + pad, fs), _m2s(before + pad, fs)
    start = o_trough - my_trough
    end = (o_len - o_trough) - (my_len - my_trough)
    return o_len - end - start


def _consistent(kw, fs):
    b, a, al = kw["ms_before"], kw["ms_after"], kw["alignment_ms"]
    base = _length(b, a, fs)
    return (_realign_trimmed_length(b, a, al, fs) == base
            and _svd_slice_length(b, a, al / 2, fs) == base)


def test_upstream_defaults_break_at_32552():
    # Negative control: the lab rate that crashed with 131 vs 133.
    kw = dict(_WINDOW_DEFAULTS)
    assert _length(kw["ms_before"], kw["ms_after"], 32552) == 131
    assert _realign_trimmed_length(kw["ms_before"], kw["ms_after"],
                                   kw["alignment_ms"], 32552) == 133


@pytest.mark.parametrize("fs", [20000, 24414.0625, 25000, 30000, 32552, 40000])
def test_snapped_windows_consistent(fs):
    kw = snap_windows_to_samples({}, fs)
    assert _consistent(kw, fs)
    assert _length(kw["ms_before"], kw["ms_after"], fs) % 2 == 1


def test_snapped_windows_consistent_all_rates():
    bad = [fs for fs in range(16000, 50001, 7)
           if not _consistent(snap_windows_to_samples({}, fs), fs)]
    assert bad == []


def test_snapping_keeps_30k_defaults_and_user_values():
    kw = snap_windows_to_samples({"peak_sign": "neg"}, 30000)
    assert kw["peak_sign"] == "neg"
    assert _m2s(kw["ms_before"], 30000) == 42
    assert _length(kw["ms_before"], kw["ms_after"], 30000) == 121
    kw = snap_windows_to_samples({"ms_before": 1.0, "alignment_ms": 1.0}, 32552)
    assert _m2s(kw["ms_before"], 32552) == 33
    assert _m2s(kw["alignment_ms"], 32552) % 2 == 0
