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


def test_skip_done_exports_saved_sorts_instead_of_resorting(tmp_path, monkeypatch):
    calls = _fake_export(monkeypatch)
    # first run without export: sorted, scored and saved, no Klusters sessions
    _bench(tmp_path, [FakeAdapter("a", stages=True), FakeAdapter("b")],
           skip_done=False).go()
    from ns3_sort_bench.sort_store import sort_path
    assert os.path.exists(sort_path(str(tmp_path), "item", "a@s1"))
    # resume with export on: nothing re-sorted, the saved sorts exported
    a2, b2 = FakeAdapter("a", stages=True), FakeAdapter("b")
    _bench_k(tmp_path, [a2, b2], skip_done=True).go()
    assert (a2.calls, b2.calls) == (0, 0)
    assert calls == [["a", "a@s1", "ground_truth"], ["b"]]
    assert _candidates(tmp_path) == ["a", "a@s1", "b"]
    # and once exported, nothing more is exported either
    _bench_k(tmp_path, [FakeAdapter("a"), FakeAdapter("b")], skip_done=True).go()
    assert len(calls) == 2


def test_skip_done_resorts_when_no_saved_sort(tmp_path, monkeypatch):
    import shutil
    calls = _fake_export(monkeypatch)
    _bench(tmp_path, [FakeAdapter("a")], skip_done=False).go()
    shutil.rmtree(tmp_path / "sorts")       # a run from before sorts were saved
    a2 = FakeAdapter("a")
    _bench_k(tmp_path, [a2], skip_done=True).go()
    assert a2.calls == 1
    assert calls == [["a", "ground_truth"]]


def test_saved_sort_round_trip(tmp_path):
    from ns3_sort_bench.sort_store import load_sort, save_sort
    r = SortedResult(unit_ids=[5, 2, 9], spike_times={2: np.array([30, 10]),
                     5: np.array([7]), 9: np.array([], np.int64)},
                     sampling_rate=32552.0, adapter="x@fc_polish",
                     corpus_item="lab_g8", adapter_version="1.2")
    save_sort(str(tmp_path), r)
    back = load_sort(str(tmp_path), "lab_g8", "x@fc_polish")
    assert list(back.unit_ids) == [5, 2, 9]
    assert back.spike_times[2].tolist() == [30, 10]
    assert back.spike_times[5].tolist() == [7]
    assert back.spike_times[9].tolist() == []
    assert (back.sampling_rate, back.adapter, back.adapter_version) == \
        (32552.0, "x@fc_polish", "1.2")
    assert load_sort(str(tmp_path), "lab_g8", "missing") is None
