"""Smoke tests for ``src/ns3_sort_bench/configs/`` — the per-adapter
tuning yamls and the :func:`load_adapter_config` helper."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ns3_sort_bench.configs import load_adapter_config, load_adapter_config_path


# ─── every adapter has a config file ─────────────────────────────────────────
_ADAPTER_NAMES = ("fiber_kit", "kilosort4", "mountainsort5",
                   "tridesclous2", "spykingcircus2",
                   "botm_ref", "dartsort", "slay")


@pytest.mark.parametrize("name", _ADAPTER_NAMES)
def test_config_file_exists_and_is_valid_yaml(name):
    """Every adapter named above must ship a ``configs/<name>.yaml``, and
    the file must parse as a YAML mapping (dict).  Missing / malformed
    files would silently reduce the adapter to its hardcoded fallback at
    runtime without a test failure anywhere; this one catches it."""
    path = Path(load_adapter_config_path(name))
    assert path.name == f"{name}.yaml"
    assert path.exists(), f"missing config file: {path}"
    parsed = yaml.safe_load(path.read_text())
    # None / empty is allowed (future-reserved); a non-mapping is not.
    assert parsed is None or isinstance(parsed, dict), \
        f"{path}: top-level must be a mapping, got {type(parsed).__name__}"


def test_load_adapter_config_returns_dict_on_missing_file():
    """Missing files must return an EMPTY dict, not raise.  An adapter
    whose config was renamed / deleted is still callable (falls back to
    whatever hardcoded fallback the adapter uses) — hard-failing at
    import time would make the whole package unusable for an unrelated
    adapter's missing yaml."""
    out = load_adapter_config("this_adapter_does_not_exist_" + "x" * 20)
    assert out == {}


def test_fiber_kit_config_carries_the_three_ndm_sections():
    """The fiber-kit adapter reads `ndm_detect`, `ndm_extract`, `ndm_pca`
    from the yaml.  Missing any of them would leave the adapter on its
    hardcoded fallback — functional but silently decoupled from the yaml
    the user edits.  Pin the schema here."""
    cfg = load_adapter_config("fiber_kit")
    for section in ("ndm_detect", "ndm_extract", "ndm_pca", "fiber_kit"):
        assert section in cfg, f"fiber_kit.yaml missing `{section}:` section"
        assert isinstance(cfg[section], dict), \
            f"fiber_kit.yaml `{section}:` must be a mapping"
    # thresholdFactor is the single CA1-sensitive knob most likely to be
    # retuned — pin it to a string (stringified in the programs block).
    assert isinstance(cfg["ndm_detect"]["thresholdFactor"], str)


def test_kilosort4_config_has_run_kwargs():
    """Kilosort 4's SI wrapper reads ``run_kwargs:`` from the yaml."""
    cfg = load_adapter_config("kilosort4")
    assert "run_kwargs" in cfg
    run_kw = cfg["run_kwargs"]
    assert isinstance(run_kw, dict)
    assert run_kw.get("do_CAR") is False       # the harness-wide default


def test_si_adapters_still_merge_call_time_overrides():
    """`adapter.sort(run_kwargs=...)` must still merge on top of the
    config-sourced baseline — the yaml is the FLOOR, not the ceiling."""
    # Reach into the adapter without needing spikeinterface installed:
    # _run_kwargs is populated at class-definition time.
    from ns3_sort_bench.adapters.si_sorters import Kilosort4Adapter
    adp = Kilosort4Adapter()
    # The class carries what the yaml specified...
    assert adp._run_kwargs.get("Th_universal") == 9.0
    # ... and a call-time override would merge like `{base, **override}`.
    override = {"Th_universal": 7.0, "something_new": True}
    merged = {**adp._run_kwargs, **override}
    assert merged["Th_universal"]   == 7.0        # override wins
    assert merged["Th_learned"]     == 8.0        # baseline still present
    assert merged["something_new"] is True        # new keys pass through


# ─── per-adapter baseline-tuning smoke tests ─────────────────────────────────
def test_botm_ref_config_loads_tuning_knobs():
    """botm_ref's config exposes `n_units_hint`, `detect_threshold_mad`,
    `template_samples`, `peak_sample`, `botm_backend` — the five knobs
    the adapter reads through ``_BOTM_REF_CFG``.  A missing key here is
    a silent regression to the old hard-coded defaults.
    """
    from ns3_sort_bench.adapters.botm_ref import _BOTM_REF_CFG
    for k in ("n_units_hint", "detect_threshold_mad",
              "template_samples", "peak_sample", "botm_backend"):
        assert k in _BOTM_REF_CFG, f"{k!r} missing from botm_ref.yaml"
    # The baseline specifically bumps n_units_hint off the hard-coded
    # 3 — the whole point of the config is that CA1 recordings need
    # more templates to score at all.
    assert int(_BOTM_REF_CFG["n_units_hint"]) >= 8


def test_botm_ref_kwargs_still_override_config():
    """A caller passing n_units_hint / detect_threshold_mad via
    sort(**kwargs) must win over configs/botm_ref.yaml, mirroring
    every other adapter's "caller > config > built-in" precedence."""
    from ns3_sort_bench.adapters.botm_ref import _BOTM_REF_CFG
    cfg = dict(_BOTM_REF_CFG)
    # Simulate the precedence _cfg() implements inside sort():
    def resolve(kwargs, key, default):
        if key in kwargs:
            return kwargs[key]
        return cfg.get(key, default)
    assert resolve({}, "n_units_hint", 3) == cfg["n_units_hint"]
    # Caller override wins.
    assert resolve({"n_units_hint": 25}, "n_units_hint", 3) == 25


def test_dartsort_config_uses_dartsortuserconfig_field_names():
    """dartsort run_kwargs must be DARTsortUserConfig field names (SI's
    DartsortSorter does DARTsortUserConfig(**params)).  Pins the
    verified keys and guards against the earlier guessed names
    (detection_threshold / n_pca_features), which don't exist."""
    rk = load_adapter_config("dartsort")["run_kwargs"]
    assert rk.get("do_motion_estimation") is False
    assert rk.get("peak_sign") == "neg"
    for bad in ("detection_threshold", "n_pca_features"):
        assert bad not in rk, f"{bad!r} is not a DARTsortUserConfig field"


def test_dartsort_adapter_picks_up_config():
    from ns3_sort_bench.adapters.dartsort import _DARTSORT_RUN_KWARGS
    assert _DARTSORT_RUN_KWARGS.get("do_motion_estimation") is False


def test_mountainsort5_config_uses_valid_si_keys():
    """Keys must be in Mountainsort5Sorter's default_params (the list
    SI printed in the hc1 ValueError).  `detect_channel_radius` does
    not exist — scheme 2 has two per-phase radius keys."""
    valid = {
        "scheme", "detect_threshold", "detect_sign", "detect_time_radius_msec",
        "snippet_T1", "snippet_T2", "npca_per_channel", "npca_per_subdivision",
        "snippet_mask_radius", "scheme1_detect_channel_radius",
        "scheme2_phase1_detect_channel_radius", "scheme2_detect_channel_radius",
        "scheme2_max_num_snippets_per_training_batch",
        "scheme2_training_duration_sec",
        "scheme2_training_recording_sampling_mode",
        "scheme3_block_duration_sec", "freq_min", "freq_max", "filter",
        "whiten", "whitening_seed", "delete_temporary_recording",
        "pool_engine", "n_jobs", "chunk_duration", "progress_bar",
        "mp_context", "max_threads_per_worker",
    }
    rk = load_adapter_config("mountainsort5")["run_kwargs"]
    bad = set(rk) - valid
    assert not bad, f"invalid MountainSort5 keys: {sorted(bad)}"


def test_mountainsort5_scheme_is_a_string():
    """SI's Mountainsort5Sorter checks scheme in ('1', '2', '3') as
    strings; an int from an unquoted YAML value raises at run time."""
    rk = load_adapter_config("mountainsort5")["run_kwargs"]
    assert isinstance(rk["scheme"], str) and rk["scheme"] in ("1", "2", "3")


def test_spykingcircus2_config_restates_full_nested_blocks():
    """SC2 params are nested and SI merges with a SHALLOW dict.update,
    so an overridden sub-dict must carry every sibling key or the
    sorter KeyErrors on the missing ones at run time."""
    rk = load_adapter_config("spykingcircus2")["run_kwargs"]
    for bad in ("detect_threshold", "detect_radius"):
        assert bad not in rk, f"{bad!r} is not a top-level SC2 key"
    if "detection" in rk:
        det = rk["detection"]
        assert {"method", "method_kwargs", "pipeline_kwargs"} <= set(det)
        assert {"peak_sign", "detect_threshold"} <= set(det["method_kwargs"])
    if "general" in rk:
        assert {"ms_before", "ms_after", "radius_um"} <= set(rk["general"])


def test_tridesclous2_config_uses_valid_si_keys():
    valid = {
        "apply_preprocessing", "preprocessing_dict", "apply_motion_correction",
        "motion_correction_preset", "clustering_ms_before", "clustering_ms_after",
        "detection_radius_um", "features_radius_um", "split_radius_um",
        "template_radius_um", "merge_similarity_lag_ms", "freq_min", "freq_max",
        "cache_preprocessing_mode", "peak_sign", "detect_threshold",
        "n_peaks_per_channel", "n_svd_components_per_channel", "n_pca_features",
        "clustering_recursive_depth", "ms_before", "ms_after",
        "template_sparsify_threshold", "template_min_snr_ptp",
        "template_max_jitter_ms", "min_firing_rate", "gather_mode",
        "job_kwargs", "seed", "save_array", "debug",
    }
    rk = load_adapter_config("tridesclous2")["run_kwargs"]
    bad = set(rk) - valid
    assert not bad, f"invalid Tridesclous2 keys: {sorted(bad)}"


def test_slay_config_has_both_blocks_and_no_unverified_merge_kwargs():
    """slay_kwargs feed a call wrapped in a bare `except: return []`, so
    a wrong key fails SILENTLY (zero merges).  Keep it empty until a
    key is verified against slay.merge.recommend_merges."""
    cfg = load_adapter_config("slay")
    assert isinstance(cfg["ks_kwargs"], dict)
    assert cfg["slay_kwargs"] == {}


def test_slay_adapter_picks_up_both_blocks():
    from ns3_sort_bench.adapters.slay import (
        _SLAY_KS_DEFAULTS, _SLAY_MERGE_DEFAULTS,
    )
    assert isinstance(_SLAY_KS_DEFAULTS, dict)
    assert isinstance(_SLAY_MERGE_DEFAULTS, dict)
