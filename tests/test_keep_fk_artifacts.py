"""`ns3sb set keep-fk-artifacts` — plan round-trip, CLI parsing, harness
threading of `fk_artifacts_dir` to the adapter, slugify, and the
authoritative-corpus_item override in BenchmarkRun.
"""
from __future__ import annotations

import io
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import pytest

from ns3_sort_bench.cli.plan import Plan, plan_path
from ns3_sort_bench.adapters.fiber_kit import _slugify


# ─── slugify ────────────────────────────────────────────────────────────────
def test_slugify_strips_colons_and_slashes():
    assert _slugify("PAIRED_CRCNS_HC1:paired_crcns:d11221_d11221.001") == \
           "PAIRED_CRCNS_HC1_paired_crcns_d11221_d11221.001"
    assert _slugify("foo/bar.yaml") == "foo_bar.yaml"


def test_slugify_collapses_runs_of_separators():
    assert _slugify("a::b//c") == "a_b_c"


def test_slugify_handles_edge_cases():
    assert _slugify("") == "sort"
    assert _slugify(":::") == "sort"
    assert _slugify("plain-name_1.2+3") == "plain-name_1.2+3"


# ─── plan round-trip ────────────────────────────────────────────────────────
def test_plan_roundtrip_keep_fk_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("NS3_SORT_BENCH_PLAN", str(tmp_path / "plan.yaml"))
    assert Plan.load().keep_fk_artifacts is False
    p = Plan.load()
    p.keep_fk_artifacts = True
    p.save()
    assert Plan.load().keep_fk_artifacts is True
    # pretty() includes the field
    assert "keep_fk_artifacts: True" in Plan.load().pretty()


# ─── CLI set keep-fk-artifacts ──────────────────────────────────────────────
@pytest.mark.parametrize("truthy", ["true", "TRUE", "True", "1", "yes", "on"])
def test_cli_set_keep_fk_artifacts_truthy(tmp_path, monkeypatch, truthy):
    monkeypatch.setenv("NS3_SORT_BENCH_PLAN", str(tmp_path / "plan.yaml"))
    from ns3_sort_bench.cli.main import main
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["set", "keep-fk-artifacts", truthy])
    assert rc == 0
    assert Plan.load().keep_fk_artifacts is True


@pytest.mark.parametrize("falsy", ["false", "0", "no", "off"])
def test_cli_set_keep_fk_artifacts_falsy(tmp_path, monkeypatch, falsy):
    monkeypatch.setenv("NS3_SORT_BENCH_PLAN", str(tmp_path / "plan.yaml"))
    from ns3_sort_bench.cli.main import main
    # Start with True
    p = Plan.load(); p.keep_fk_artifacts = True; p.save()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["set", "keep-fk-artifacts", falsy])
    assert rc == 0
    assert Plan.load().keep_fk_artifacts is False


def test_cli_set_keep_fk_artifacts_rejects_garbage(tmp_path, monkeypatch):
    monkeypatch.setenv("NS3_SORT_BENCH_PLAN", str(tmp_path / "plan.yaml"))
    from ns3_sort_bench.cli.main import main
    err = io.StringIO()
    with redirect_stderr(err):
        rc = main(["set", "keep-fk-artifacts", "maybe"])
    assert rc == 1
    assert "true|false" in err.getvalue()


# ─── harness: adapter_kwargs + authoritative corpus_item ────────────────────
class _FakeAdapter:
    name = "fake"
    version = "0.0"

    def __init__(self):
        self.calls = []
        # Minimal duck-typed contract that validate() would skip
        self.contract = None

    def available(self):
        return True, ""

    def sort(self, recording, *, ns3_yaml=None, **kwargs):
        from ns3_sort_bench._base import SortedResult
        self.calls.append(kwargs)
        return SortedResult(
            unit_ids=[2], spike_times={2: np.asarray([10, 20, 30], np.int64)},
            sampling_rate=20000.0,
            adapter=self.name,
            corpus_item="None",   # intentional trap: pandas would NaN this
            adapter_version=self.version,
        )


def _make_corpus(name: str, sr: float = 20000.0):
    from ns3_sort_bench._base import Corpus, SortedResult

    class _R:
        def get_sampling_frequency(self): return sr
        def get_num_samples(self, _seg=0): return 1000
        def get_num_channels(self): return 1
        def get_annotation(self, k, copy=True): return None

    gt = SortedResult(
        unit_ids=[2], spike_times={2: np.asarray([10, 15, 20], np.int64)},
        sampling_rate=sr,
        adapter="ground_truth", corpus_item=name,
    )
    return Corpus(name=name, recording=_R(), ground_truth=gt, kind="test")


def test_benchmarkrun_overrides_corpus_item_to_corpus_name(tmp_path):
    """The adapter sets corpus_item='None' (a pandas NA token); the harness
    must override it with corpus.name before scoring + CSV."""
    from ns3_sort_bench._base import BenchmarkRun
    from ns3_sort_bench.scorers.pairwise import PairwiseScorer
    adapter = _FakeAdapter()
    bench = BenchmarkRun(output_dir=str(tmp_path), verbose=False)
    bench.add_corpus(_make_corpus("MY:CORPUS:ITEM"))
    bench.add_adapter(adapter)
    bench.add_scorer(PairwiseScorer())
    df = bench.go()
    # The adapter was called with corpus_name='MY:CORPUS:ITEM'
    assert adapter.calls[-1]["corpus_name"] == "MY:CORPUS:ITEM"
    # And the written scores.csv has the corpus_item set correctly
    assert "MY:CORPUS:ITEM" in df["corpus_item"].astype(str).tolist()


def test_benchmarkrun_forwards_adapter_kwargs(tmp_path):
    """BenchmarkRun.adapter_kwargs must reach every adapter.sort() call."""
    from ns3_sort_bench._base import BenchmarkRun
    from ns3_sort_bench.scorers.pairwise import PairwiseScorer
    adapter = _FakeAdapter()
    bench = BenchmarkRun(
        output_dir=str(tmp_path), verbose=False,
        adapter_kwargs={"fk_artifacts_dir": "/scratch/fk"},
    )
    bench.add_corpus(_make_corpus("X"))
    bench.add_adapter(adapter)
    bench.add_scorer(PairwiseScorer())
    bench.go()
    assert adapter.calls[-1]["fk_artifacts_dir"] == "/scratch/fk"
    assert adapter.calls[-1]["corpus_name"] == "X"


def test_run_cmd_sets_fk_artifacts_dir_from_plan(tmp_path, monkeypatch):
    """`ns3sb run` with keep_fk_artifacts=True should configure
    BenchmarkRun(adapter_kwargs=dict(fk_artifacts_dir=<output>/fiber_kit))."""
    monkeypatch.setenv("NS3_SORT_BENCH_PLAN", str(tmp_path / "plan.yaml"))
    plan = Plan.load()
    plan.corpora = ["propper_sim_tiny"]
    plan.adapters = ["botm_ref"]   # cheap, always available
    plan.scorers = ["per_cell"]
    plan.output_dir = str(tmp_path / "run")
    plan.keep_fk_artifacts = True
    plan.save()
    # Spy on BenchmarkRun construction
    captured = {}

    import ns3_sort_bench._base as _base
    orig_ctor = _base.BenchmarkRun

    def _spy(*args, **kw):
        captured.update(kw)
        return orig_ctor(*args, **kw)
    monkeypatch.setattr(_base, "BenchmarkRun", _spy)
    # run_cmd imports BenchmarkRun from the package root; patch that import
    import ns3_sort_bench
    monkeypatch.setattr(ns3_sort_bench, "BenchmarkRun", _spy)

    from ns3_sort_bench.cli.main import main
    # Don't actually run — just resolve enough to build BenchmarkRun.  Pass
    # --dry-run?  No — dry-run returns before constructing.  Instead rely on
    # the run producing a scores.csv (botm_ref on propper_sim_tiny is fast).
    try:
        main(["run"])
    except SystemExit:
        pass
    assert captured.get("adapter_kwargs", {}).get("fk_artifacts_dir", "").endswith(
        "/fiber_kit"
    )
