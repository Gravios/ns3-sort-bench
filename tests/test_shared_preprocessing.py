"""Shared per-item preprocessing for the Klusters export.

`prepare_shared_session` writes one bandpassed .fil/.dat/.yaml/.probe per
corpus item under `<run>/shared_sessions/<item>/`, before any sorter runs;
every `klusters_<sorter>/<item>/` links from it.  These tests cover the
preprocessing and linking (no ndm_* binaries needed).
"""
from __future__ import annotations

import os

import numpy as np

from ns3_sort_bench._base import Corpus, SortedResult


class _Rec:
    def __init__(self, n_samp=4000, n_ch=4, sr=20000.0):
        self._n, self._c, self._sr = n_samp, n_ch, sr
        rng = np.random.default_rng(0)
        self._x = (rng.standard_normal((n_samp, n_ch)) * 50).astype(np.float32)

    def get_num_samples(self, _seg=0): return self._n
    def get_num_channels(self): return self._c
    def get_sampling_frequency(self): return self._sr
    def get_traces(self, start_frame=0, end_frame=None, **_):
        return self._x[start_frame:end_frame]
    def get_annotation(self, k, *a, **kw): return None
    def get_channel_ids(self): return list(range(self._c))
    def get_channel_locations(self):
        return np.array([[0.0, 20.0 * i] for i in range(self._c)])


def _corpus(name="PAIRED:item:1"):
    gt = SortedResult(unit_ids=[2], spike_times={2: np.array([100, 900])},
                      sampling_rate=20000.0, adapter="ground_truth",
                      corpus_item=name)
    return Corpus(name=name, recording=_Rec(), ground_truth=gt, kind="test")


def test_prepare_shared_session_writes_complete_fil_outside_klusters_ns(tmp_path):
    from ns3_sort_bench.klusters_export import prepare_shared_session
    base = prepare_shared_session(str(tmp_path), _corpus())
    assert os.path.isabs(base)
    assert f"{os.sep}shared_sessions{os.sep}" in base
    assert f"{os.sep}klusters_" not in base   # never seen as a sorter dir
    assert os.path.getsize(base + ".fil") == 4000 * 4 * 2
    assert os.path.samefile(base + ".fil", base + ".dat")
    assert os.path.exists(base + ".yaml")


def test_truncated_fil_from_interrupted_run_is_rebuilt(tmp_path):
    """The hc1 failure mode: a stale/partial .fil was silently reused
    because the old idempotence check only tested existence."""
    from ns3_sort_bench.klusters_export import (
        prepare_shared_session, shared_session_base,
    )
    c = _corpus()
    base = shared_session_base(str(tmp_path), c)
    os.makedirs(os.path.dirname(base))
    for ext in (".fil", ".dat", ".yaml"):
        open(base + ext, "wb").write(b"\x00" * 10)   # truncated junk
    prepare_shared_session(str(tmp_path), c)
    assert os.path.getsize(base + ".fil") == 4000 * 4 * 2


def test_relink_replaces_stale_target_but_keeps_same_file(tmp_path):
    from ns3_sort_bench.klusters_export import _relink
    src = tmp_path / "src.fil"; src.write_bytes(b"good" * 8)
    dst = tmp_path / "dst.fil"; dst.write_bytes(b"stale")
    _relink(str(src), str(dst))
    assert os.path.samefile(src, dst)
    _relink(str(src), str(dst))                  # already same — no-op
    assert os.path.samefile(src, dst)


def test_fil_is_complete_checks_exact_size(tmp_path):
    from ns3_sort_bench.klusters_export import _fil_is_complete
    f = tmp_path / "x.fil"
    f.write_bytes(b"\x00" * (10 * 4 * 2))
    assert _fil_is_complete(str(f), 10, 4)
    assert not _fil_is_complete(str(f), 11, 4)
    assert not _fil_is_complete(str(tmp_path / "missing.fil"), 10, 4)


def test_score_curated_does_not_treat_shared_dir_as_a_sorter(tmp_path):
    from ns3_sort_bench.klusters_export import prepare_shared_session
    from ns3_sort_bench.cli.commands.score_curated_cmd import _iter_session_dirs
    prepare_shared_session(str(tmp_path), _corpus())
    assert list(_iter_session_dirs(str(tmp_path))) == []
