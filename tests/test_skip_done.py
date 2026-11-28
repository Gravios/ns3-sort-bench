"""BenchmarkRun(skip_done=True) resumes a run instead of re-sorting.

Fake adapters and a fake scorer, so the test needs no SpikeInterface.
"""
import json
import os

import numpy as np
import pandas as pd

from ns3_sort_bench._base import (
    BenchmarkRun, Corpus, Scorer, ScoreResult, SortedResult, SorterAdapter,
)


def _result(adapter, item="item", extras=None):
    return SortedResult(unit_ids=[2], spike_times={2: np.arange(5)},
                        sampling_rate=1000.0, adapter=adapter,
                        corpus_item=item, extras=extras or {})


class FakeAdapter(SorterAdapter):
    def __init__(self, name, fail=False, stages=False):
        self.name, self.fail, self.stages, self.calls = name, fail, stages, 0

    @property
    def contract(self):
        return None

    @property
    def version(self):
        return "0"

    def available(self):
        return True, ""

    def sort(self, recording, *, ns3_yaml=None, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        extras = {}
        if self.stages:
            extras["stage_results"] = {"s1": _result(f"{self.name}@s1")}
        return _result(self.name, extras=extras)


class FakeScorer(Scorer):
    name = "fake"

    def __call__(self, reference, candidate, **kwargs):
        return ScoreResult(scorer="fake", corpus_item=candidate.corpus_item,
                           reference_adapter="gt",
                           candidate_adapter=candidate.adapter,
                           values={"n": float(candidate.n_spikes())})


def _bench(out, adapters, skip_done):
    b = BenchmarkRun(output_dir=str(out), verbose=False, skip_done=skip_done)
    b.add_corpus(Corpus(name="item", recording=None,
                        ground_truth=_result("gt")))
    for a in adapters:
        b.add_adapter(a)
    b.add_scorer(FakeScorer())
    return b


def _candidates(out):
    df = pd.read_csv(os.path.join(out, "scores.csv"))
    return sorted(df["candidate_adapter"])


def test_skip_done_reruns_only_failed_and_new(tmp_path):
    a, s = FakeAdapter("a", stages=True), FakeAdapter("s", fail=True)
    _bench(tmp_path, [a, s], skip_done=False).go()
    assert _candidates(tmp_path) == ["a", "a@s1"]

    a2, s2, n2 = FakeAdapter("a", stages=True), FakeAdapter("s"), FakeAdapter("n")
    _bench(tmp_path, [a2, s2, n2], skip_done=True).go()
    assert (a2.calls, s2.calls, n2.calls) == (0, 1, 1)
    # a's rows (stage included) carried over once, not duplicated
    assert _candidates(tmp_path) == ["a", "a@s1", "n", "s"]
    with open(tmp_path / "sort_summaries.json") as f:
        summ = json.load(f)
    assert sorted(x["adapter"] for x in summ) == ["a", "a@s1", "n", "s"]
    assert not any("error" in x for x in summ)


def test_skip_done_keeps_adapters_left_out_of_this_run(tmp_path):
    _bench(tmp_path, [FakeAdapter("a"), FakeAdapter("b")], skip_done=False).go()
    b2 = FakeAdapter("b")
    _bench(tmp_path, [b2], skip_done=True).go()
    assert b2.calls == 0
    assert _candidates(tmp_path) == ["a", "b"]


def test_without_skip_done_everything_resorts(tmp_path):
    _bench(tmp_path, [FakeAdapter("a")], skip_done=False).go()
    a2 = FakeAdapter("a")
    _bench(tmp_path, [a2], skip_done=False).go()
    assert a2.calls == 1
    assert _candidates(tmp_path) == ["a"]


def test_checkpoint_survives_a_crash_mid_run(tmp_path):
    class Killer(FakeAdapter):
        def sort(self, *a, **k):
            raise KeyboardInterrupt  # not caught by go(), like a kill

    try:
        _bench(tmp_path, [FakeAdapter("a"), Killer("k")], skip_done=False).go()
    except KeyboardInterrupt:
        pass
    assert _candidates(tmp_path) == ["a"]
    a2 = FakeAdapter("a")
    _bench(tmp_path, [a2], skip_done=True).go()
    assert a2.calls == 0


def test_skip_done_on_empty_dir(tmp_path):
    a = FakeAdapter("a")
    _bench(tmp_path / "new", [a], skip_done=True).go()
    assert a.calls == 1


# ─── Klusters export, per sorter, and its interplay with --skip-done ──────
def _fake_export(monkeypatch):
    import ns3_sort_bench.klusters_export as ke
    calls = []

    def export(output_dir, corpus, candidates, *, include_ground_truth=True, **kw):
        names = list(candidates) + (["ground_truth"] if include_ground_truth else [])
        calls.append(names)
        out = []
        for n in names:
            base = os.path.join(output_dir, f"klusters_{ke._slugify(n)}",
                                ke._slugify(corpus.name), "sim_session")
            os.makedirs(os.path.dirname(base), exist_ok=True)
            open(ke._klusters_path(base, "clu", 1, method=ke._NS3_METHOD), "w").close()
            out.append(os.path.dirname(base))
        return out

    monkeypatch.setattr(ke, "export_klusters_session", export)
    monkeypatch.setattr(ke, "prepare_shared_session", lambda *a, **k: None)
    return calls


def _bench_k(out, adapters, skip_done):
    b = _bench(out, adapters, skip_done)
    b.export_klusters = True
    return b


def test_export_runs_after_each_sorter(tmp_path, monkeypatch):
    calls = _fake_export(monkeypatch)
    _bench_k(tmp_path, [FakeAdapter("a", stages=True), FakeAdapter("b"),
                        FakeAdapter("s", fail=True)], skip_done=False).go()
    # one export per successful sorter, stages with their parent, GT once
    assert calls == [["a", "a@s1", "ground_truth"], ["b"]]


def test_export_survives_a_crash_mid_run(tmp_path, monkeypatch):
    from ns3_sort_bench.klusters_export import exported_session_exists
    _fake_export(monkeypatch)

    class Killer(FakeAdapter):
        def sort(self, *a, **k):
            raise KeyboardInterrupt

    try:
        _bench_k(tmp_path, [FakeAdapter("a"), Killer("k")], skip_done=False).go()
    except KeyboardInterrupt:
        pass
    assert exported_session_exists(str(tmp_path), "item", "a")


def test_skip_done_resorts_pairs_never_exported(tmp_path, monkeypatch):
    calls = _fake_export(monkeypatch)
    # first run without export: sorted and scored, no Klusters sessions
    _bench(tmp_path, [FakeAdapter("a"), FakeAdapter("b")], skip_done=False).go()
    # resume with export on: both are sorted again, since a sort is not
    # saved and only a fresh one can be exported
    a2, b2 = FakeAdapter("a"), FakeAdapter("b")
    _bench_k(tmp_path, [a2, b2], skip_done=True).go()
    assert (a2.calls, b2.calls) == (1, 1)
    assert calls == [["a", "ground_truth"], ["b"]]
    assert _candidates(tmp_path) == ["a", "b"]
    # and once exported, they are skipped
    a3, b3 = FakeAdapter("a"), FakeAdapter("b")
    _bench_k(tmp_path, [a3, b3], skip_done=True).go()
    assert (a3.calls, b3.calls) == (0, 0)
    assert _candidates(tmp_path) == ["a", "b"]
