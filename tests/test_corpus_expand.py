"""Tests for `ns3_sort_bench.runners.sorter_grid.expand_corpus_key`.

The function sits between plan keys and `_resolve_corpus`: a bare paired-set
name (e.g. ``paired_english``) fans out to every item SpikeForest reports in
that study_set; every other key passes through unchanged.
"""
from __future__ import annotations

import pytest


def test_non_paired_keys_pass_through_unchanged():
    """Everything that is not a bare paired-set prefix stays as a single
    element in the returned list — so the caller's expand→resolve loop is
    a no-op for propper_sim, lab:, spikeforest:, and explicit
    paired_<lab>:<item> keys.
    """
    from ns3_sort_bench.runners.sorter_grid import expand_corpus_key

    for key in (
        "propper_sim",
        "propper_sim_tiny",
        "propper_sim_medium",
        "lab:/data/foo.yaml",
        "lab:/data/foo.yaml:7:stderiv_C5_D34:exp_final",
        "spikeforest:PAIRED_X:recX",
        "paired_english:item_42",
        "paired_crcns_hc1:d533101",
        "nonsense_that_is_not_a_prefix",
    ):
        assert expand_corpus_key(key) == [key], f"{key!r} didn't pass through"


def test_bare_paired_set_name_expands_to_every_item(monkeypatch):
    """A bare `paired_english` with no `:<item>` tail must expand to a
    `paired_english:<study_name>` key for each recording SpikeForest lists
    in the PAIRED_ENGLISH study set.  Order mirrors SpikeForest's listing.
    """
    from ns3_sort_bench.runners import sorter_grid
    from ns3_sort_bench.corpus import spikeforest as sf_mod

    def _fake_list(study_set=None, study_name=None, uri=None):
        assert study_set == "PAIRED_ENGLISH"
        return [
            {"study_set_name": "PAIRED_ENGLISH",
             "study_name":     "paired_english_m1_s1",
             "recording_name": "m1_s1",
             "sampling_frequency": 30000.0,
             "num_channels": 32, "duration_sec": 600.0, "num_true_units": 1},
            {"study_set_name": "PAIRED_ENGLISH",
             "study_name":     "paired_english_m1_s2",
             "recording_name": "m1_s2",
             "sampling_frequency": 30000.0,
             "num_channels": 32, "duration_sec": 600.0, "num_true_units": 1},
            {"study_set_name": "PAIRED_ENGLISH",
             "study_name":     "paired_english_m2_s1",
             "recording_name": "m2_s1",
             "sampling_frequency": 30000.0,
             "num_channels": 32, "duration_sec": 600.0, "num_true_units": 1},
        ]
    # Monkey-patch where the function is LOOKED UP (the local import in
    # expand_corpus_key), not just the definition site.  The function
    # does a from-import inside, so the attribute on the module is what
    # gets resolved.
    monkeypatch.setattr(sf_mod, "list_recordings", _fake_list)

    got = sorter_grid.expand_corpus_key("paired_english")
    assert got == [
        "paired_english:paired_english_m1_s1",
        "paired_english:paired_english_m1_s2",
        "paired_english:paired_english_m2_s1",
    ]


def test_bare_paired_set_empty_listing_raises(monkeypatch):
    """When SpikeForest returns no rows for the study set (unauthenticated,
    bad kachery credentials, or an index miss), expand must raise with a
    message that points the user at the manual listing — not pass an
    empty list through to `_resolve_corpus`, which would silently sort
    nothing.
    """
    from ns3_sort_bench.runners import sorter_grid
    from ns3_sort_bench.corpus import spikeforest as sf_mod

    monkeypatch.setattr(sf_mod, "list_recordings",
                         lambda **_: [])
    with pytest.raises(RuntimeError, match="PAIRED_ENGLISH"):
        sorter_grid.expand_corpus_key("paired_english")


def test_every_paired_prefix_in_loader_is_also_in_study_set_map():
    """`_PAIRED_PREFIX_TO_LOADER` and `_PAIRED_PREFIX_TO_STUDY_SET` are
    kept in lock-step: a prefix that has a loader but no study_set
    mapping would silently skip bare-name expansion while explicit
    `prefix:<item>` keeps working, which is a confusing split.
    """
    from ns3_sort_bench.runners.sorter_grid import (
        _PAIRED_PREFIX_TO_LOADER, _PAIRED_PREFIX_TO_STUDY_SET,
    )
    assert set(_PAIRED_PREFIX_TO_LOADER) == set(_PAIRED_PREFIX_TO_STUDY_SET)
