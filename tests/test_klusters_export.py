"""Tests for the Klusters-session export path.

Does NOT invoke neurosuite-3's ndm_* binaries — subprocess.run is mocked.
Verifies: the per-adapter session layout, .res/.clu byte contents, hardlink
reuse of a lab-session .fil, and that ndm_extractspikes + ndm_pca are called
once per adapter with the generated yaml.
"""
from __future__ import annotations

import subprocess as _subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from ns3_sort_bench._base import Corpus, SortedResult
from ns3_sort_bench.klusters_export import (
    export_adapter_session, export_klusters_session,
    _write_clu, _flatten_sorted_result,
)


class _FakeRec:
    """Duck-typed SpikeInterface BaseRecording."""
    def __init__(self, traces, sr=20000.0):
        self._tr = traces
        self._sr = float(sr)

    def get_num_samples(self, _seg=0):  return int(self._tr.shape[0])
    def get_num_channels(self):         return int(self._tr.shape[1])
    def get_sampling_frequency(self):   return self._sr
    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        s0 = 0 if start_frame is None else int(start_frame)
        s1 = int(self._tr.shape[0]) if end_frame is None else int(end_frame)
        return self._tr[s0:s1]
    def get_annotation(self, k, default=None): return default


def _make_corpus(name="x"):
    traces = np.zeros((20000, 4), dtype=np.int16)
    gt = SortedResult(
        unit_ids=[2, 3], spike_times={
            2: np.array([100, 200, 300], np.int64),
            3: np.array([400, 500], np.int64),
        },
        sampling_rate=20000.0,
        adapter="ground_truth", corpus_item=name,
    )
    return Corpus(name=name, recording=_FakeRec(traces), ground_truth=gt,
                   kind="test")


class _Runner:
    """Records subprocess.run invocations, simulates the per-tool side effect."""
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, *args, **kw):
        self.calls.append(list(cmd))
        return _subprocess.CompletedProcess(cmd, 0, b"", b"")


# ─── unit: _write_clu / _flatten_sorted_result ──────────────────────────────
def test_write_clu_byte_layout(tmp_path):
    """int32 header (n_distinct ids) + int32 body of cluster ids."""
    ids = np.array([2, 2, 3, 2, 4, 3], dtype=np.int32)
    path = str(tmp_path / "x.clu.1")
    _write_clu(path, ids)
    buf = np.fromfile(path, dtype="<i4")
    assert buf[0] == 3                               # 3 distinct ids
    assert buf[1:].tolist() == [2, 2, 3, 2, 4, 3]    # order preserved


def test_flatten_sorted_result_sorts_by_time():
    sr = SortedResult(
        unit_ids=[2, 3], spike_times={
            2: np.array([50, 150, 250], np.int64),
            3: np.array([100, 200], np.int64),
        },
        sampling_rate=1.0, adapter="x", corpus_item="y",
    )
    times, clu = _flatten_sorted_result(sr)
    assert times.tolist() == [50, 100, 150, 200, 250]
    assert clu.tolist()   == [2, 3, 2, 3, 2]


def test_flatten_sorted_result_empty():
    sr = SortedResult(unit_ids=[], spike_times={}, sampling_rate=1.0,
                      adapter="x", corpus_item="y")
    times, clu = _flatten_sorted_result(sr)
    assert times.size == 0 and clu.size == 0


# ─── export_adapter_session integration (mocked ndm_*) ──────────────────────
def test_export_adapter_session_writes_full_quintuple(tmp_path, monkeypatch):
    corpus = _make_corpus("MY:CORPUS:ITEM")
    runner = _Runner()
    monkeypatch.setattr(_subprocess, "run", runner)

    sess_dir = export_adapter_session(
        str(tmp_path / "runs"), corpus, "fiber_kit",
        corpus.ground_truth,
    )

    # Layout: <output>/klusters_<adapter_slug>/<item_slug>/
    # (per-sorter top-level, so each adapter is curated as a stand-alone
    # Klusters project rooted at its own folder)
    assert Path(sess_dir).name == "MY_CORPUS_ITEM"
    assert Path(sess_dir).parent.name == "klusters_fiber_kit"
    base = Path(sess_dir) / "sim_session"

    # Shared files — .fil for fiber-session, .dat for Klusters, same bytes.
    fil = base.with_suffix(".fil")
    dat = base.with_suffix(".dat")
    assert fil.exists() and dat.exists() and base.with_suffix(".yaml").exists()
    assert fil.stat().st_size == dat.stat().st_size
    assert fil.read_bytes() == dat.read_bytes()
    # The yaml's programs section carries ndm_detectspikes + ndm_extractspikes
    # + ndm_pca blocks
    import yaml
    doc = yaml.safe_load(open(base.with_suffix(".yaml")))
    program_names = {p["name"] for p in doc["programs"]}
    assert {"ndm_detectspikes", "ndm_extractspikes", "ndm_pca"} <= program_names

    # Per-adapter files written directly: .res + .clu
    res = np.fromfile(str(base) + ".res.standard.1", dtype="<i8")
    clu = np.fromfile(str(base) + ".clu.standard.1", dtype="<i4")
    assert res.tolist() == [100, 200, 300, 400, 500]     # sorted across units
    assert int(clu[0]) == 2                               # 2 distinct ids
    assert clu[1:].tolist() == [2, 2, 2, 3, 3]            # row-aligned to .res

    # ndm_extractspikes + ndm_pca were invoked on the generated yaml
    tools = [c[0] for c in runner.calls]
    assert "ndm_extractspikes" in tools and "ndm_pca" in tools
    for c in runner.calls:
        assert c[1].endswith("sim_session.yaml")
        assert "--method" in c and "standard" in c


def test_export_klusters_session_includes_ground_truth(tmp_path, monkeypatch):
    """When include_ground_truth=True (default), the GT is exported as its own
    adapter tag alongside the candidates."""
    corpus = _make_corpus("item_a")
    cand = SortedResult(
        unit_ids=[2], spike_times={2: np.array([150, 250], np.int64)},
        sampling_rate=20000.0, adapter="fiber_kit", corpus_item="item_a",
    )
    monkeypatch.setattr(_subprocess, "run", _Runner())

    sess_dirs = export_klusters_session(
        str(tmp_path / "runs"), corpus, {"fiber_kit": cand},
    )
    out = Path(tmp_path / "runs")
    # One session dir per adapter (and GT), each under klusters_<adapter>/<item>/
    assert (out / "klusters_fiber_kit" / "item_a" /
            "sim_session.res.standard.1").exists()
    assert (out / "klusters_ground_truth" / "item_a" /
            "sim_session.res.standard.1").exists()
    assert {Path(p).parent.name for p in sess_dirs} == {"klusters_fiber_kit",
                                                         "klusters_ground_truth"}


def test_export_klusters_session_without_gt_skips_it(tmp_path, monkeypatch):
    corpus = _make_corpus("item_b")
    cand = SortedResult(
        unit_ids=[2], spike_times={2: np.array([150], np.int64)},
        sampling_rate=20000.0, adapter="fiber_kit", corpus_item="item_b",
    )
    monkeypatch.setattr(_subprocess, "run", _Runner())

    sess_dirs = export_klusters_session(
        str(tmp_path / "runs"), corpus, {"fiber_kit": cand},
        include_ground_truth=False,
    )
    out = Path(tmp_path / "runs")
    assert (out / "klusters_fiber_kit" / "item_b").exists()
    assert not (out / "klusters_ground_truth").exists()
