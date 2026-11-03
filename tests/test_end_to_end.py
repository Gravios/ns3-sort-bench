"""End-to-end smoke tests on the Pröpper simulator corpus."""
import tempfile
import pytest

import ns3_sort_bench.adapters  # noqa: F401
import ns3_sort_bench.scorers   # noqa: F401
from ns3_sort_bench import (
    BenchmarkRun, adapter_registry, scorer_registry,
)
from ns3_sort_bench.simulator import make_propper_overlap_corpus


def test_registries_populated():
    assert "botm_ref" in adapter_registry
    assert "pairwise" in scorer_registry
    assert "per_cell" in scorer_registry
    assert "propper_overlap" in scorer_registry


def test_simulator_produces_corpus_with_ns3_yaml():
    c = make_propper_overlap_corpus(
        n_samples=16000, n_channels=4, sampling_rate=16000.0,
        n_units=3, rate_hz=15.0, overlap_fraction=0.2, seed=0,
    )
    assert c.recording.get_num_channels() == 4
    assert c.ns3_yaml is not None
    assert c.ns3_yaml["acquisitionSystem"]["nChannels"] == 4


def test_botm_ref_sorts_and_scorers_run():
    c = make_propper_overlap_corpus(
        n_samples=16000, n_channels=4, sampling_rate=16000.0,
        n_units=3, rate_hz=15.0, overlap_fraction=0.2, seed=1,
    )
    botm = adapter_registry["botm_ref"]
    result = botm.sort(c.recording, n_units_hint=3)
    assert result.n_units() >= 1
    for s_name in ("pairwise", "per_cell", "propper_overlap"):
        sr = scorer_registry[s_name](c.ground_truth, result)
        assert sr.values, f"{s_name} returned no values"


def test_benchmark_run_validates_contracts():
    """Prove that contract validation runs before each sort — a corpus
    missing required yaml fields produces a skip, not a crash."""
    c = make_propper_overlap_corpus(
        n_samples=12000, n_channels=3, sampling_rate=12000.0,
        n_units=2, rate_hz=20.0, overlap_fraction=0.1, seed=3,
        name="good_corpus",
    )

    # BOTM reference requires acquisitionSystem.{samplingRate,nChannels} —
    # strip them to force a contract violation
    broken_yaml = {"acquisitionSystem": {"nBits": 16}}
    bad_corpus = type(c)(
        name="broken_corpus", recording=c.recording,
        ground_truth=c.ground_truth, description="missing fields",
        kind="simulated", ns3_yaml=broken_yaml, extras={},
    )

    with tempfile.TemporaryDirectory() as td:
        bench = BenchmarkRun(output_dir=td, verbose=False)
        bench.add_corpus(c)         # should PASS contract check
        bench.add_corpus(bad_corpus) # should FAIL contract check for BOTM
        bench.add_adapter(adapter_registry["botm_ref"])
        bench.add_scorer(scorer_registry["pairwise"])
        df = bench.go()

    # The good corpus should produce rows; the broken one should not
    good_rows = df[df["corpus_item"] == "good_corpus"] if len(df) else df
    bad_rows = df[df["corpus_item"] == "broken_corpus"] if len(df) else df
    assert len(good_rows) >= 1, f"good corpus got no scores: {df}"
    assert len(bad_rows) == 0, f"broken corpus got {len(bad_rows)} scores despite contract violation"
