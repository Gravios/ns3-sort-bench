"""`ns3sb rescore`: saved sorts scored against the reference as it is now."""
import argparse
import json
import os

import numpy as np
import pandas as pd

from ns3_sort_bench._base import Corpus, SortedResult
from ns3_sort_bench.cli.commands.rescore_cmd import cmd_rescore
from ns3_sort_bench.sort_store import read_corpora, record_corpora

from test_skip_done import FakeAdapter, FakeScorer, _bench


def _gt(n):
    return SortedResult(unit_ids=[1], spike_times={1: np.arange(n)},
                        sampling_rate=1000.0, adapter="gt", corpus_item="item")


def _args(run_dir, corpus="", scorers=""):
    return argparse.Namespace(run_dir=str(run_dir), corpus=corpus, scorers=scorers)


def _patch_registries(monkeypatch, ref_n):
    import ns3_sort_bench as nsb
    import ns3_sort_bench.runners.sorter_grid as sg
    monkeypatch.setitem(nsb.scorer_registry, "fake", FakeScorer())
    monkeypatch.setattr(sg, "_resolve_corpus",
                        lambda key: Corpus(name="item", recording=None,
                                           ground_truth=_gt(ref_n[0])))


def test_rescore_uses_the_reference_on_disk_now(tmp_path, monkeypatch):
    class RefScorer(FakeScorer):
        def __call__(self, reference, candidate, **kw):
            r = super().__call__(reference, candidate)
            r.values["ref_spikes"] = float(reference.n_spikes())
            return r

    b = _bench(tmp_path, [FakeAdapter("a", stages=True)], skip_done=False)
    b.scorers = [RefScorer()]
    b.go()
    record_corpora(str(tmp_path), {"item": "lab:x.yaml:8:v:t"})
    # an unrelated item's rows must survive the rescore
    df = pd.read_csv(tmp_path / "scores.csv")
    extra = df.iloc[:1].copy()
    extra["corpus_item"] = "other"
    pd.concat([df, extra]).to_csv(tmp_path / "scores.csv", index=False)

    ref_n = [42]                     # the curated reference has changed
    _patch_registries(monkeypatch, ref_n)
    import ns3_sort_bench as nsb
    monkeypatch.setitem(nsb.scorer_registry, "fake", RefScorer())
    assert cmd_rescore(_args(tmp_path)) == 0

    df = pd.read_csv(tmp_path / "scores.csv")
    mine = df[df["corpus_item"] == "item"]
    assert sorted(mine["candidate_adapter"].unique()) == ["a", "a@s1"]
    assert set(mine[mine["metric"] == "ref_spikes"]["value"]) == {42.0}
    assert (df["corpus_item"] == "other").sum() == 1
    assert os.path.exists(tmp_path / "scores.prev.csv")
    prov = json.load(open(tmp_path / "rescore.json"))
    assert prov["items"][0]["ref_spikes"] == 42
    assert prov["items"][0]["corpus_key"] == "lab:x.yaml:8:v:t"


def test_rescore_needs_a_corpus_key(tmp_path, monkeypatch):
    _bench(tmp_path, [FakeAdapter("a")], skip_done=False).go()
    _patch_registries(monkeypatch, [5])
    assert cmd_rescore(_args(tmp_path)) == 1           # no corpora.json
    assert cmd_rescore(_args(tmp_path, corpus="k")) == 0
    assert read_corpora(str(tmp_path)) == {"item": "k"}


def test_run_records_corpus_keys(tmp_path):
    record_corpora(str(tmp_path), {"a": "k1"})
    record_corpora(str(tmp_path), {"b": "k2"})
    assert read_corpora(str(tmp_path)) == {"a": "k1", "b": "k2"}
