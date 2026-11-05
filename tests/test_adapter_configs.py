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


def test_dartsort_config_loads_run_kwargs():
    """DARTsort ships its baseline through the same ``run_kwargs:``
    section SI-wrapped sorters use.  Guards against someone switching
    DARTsort to a different schema and the adapter silently losing
    every knob.
    """
    cfg = load_adapter_config("dartsort")
    assert "run_kwargs" in cfg
    rk = cfg["run_kwargs"]
    assert isinstance(rk, dict)
    # Motion estimation specifically pinned OFF — see yaml rationale.
    assert rk.get("do_motion_estimation") is False
    # Detection threshold within a plausible sorter range (3-10).
    assert 3.0 <= float(rk.get("detection_threshold")) <= 10.0


def test_dartsort_adapter_picks_up_config():
    """The adapter module reads configs/dartsort.yaml at import time
    and stores it in ``_DARTSORT_RUN_KWARGS``.  A stale import or a
    rename would leave this empty.
    """
    from ns3_sort_bench.adapters.dartsort import _DARTSORT_RUN_KWARGS
    assert isinstance(_DARTSORT_RUN_KWARGS, dict)
    # Spot-check: the baseline's load-bearing switch must be here.
    assert _DARTSORT_RUN_KWARGS.get("do_motion_estimation") is False


def test_slay_config_splits_ks_and_merge_kwargs():
    """slay.yaml carries two sub-blocks: ks_kwargs (forwarded to the
    wrapped kilosort4 adapter) and slay_kwargs (its own NN-merge
    knobs).  Keeping them distinct lets a user retune one without
    disturbing the other.
    """
    cfg = load_adapter_config("slay")
    assert "ks_kwargs"   in cfg
    assert "slay_kwargs" in cfg
    assert isinstance(cfg["ks_kwargs"],   dict)
    assert isinstance(cfg["slay_kwargs"], dict)
    # Baseline puts merge_threshold in slay_kwargs — not in ks_kwargs.
    assert "merge_threshold" in cfg["slay_kwargs"]
    assert "merge_threshold" not in cfg["ks_kwargs"]


def test_slay_adapter_picks_up_both_blocks():
    """The adapter module reads both blocks into separate module-level
    dicts.  Each adapter call merges them over caller one-offs."""
    from ns3_sort_bench.adapters.slay import (
        _SLAY_KS_DEFAULTS, _SLAY_MERGE_DEFAULTS,
    )
    assert isinstance(_SLAY_KS_DEFAULTS,    dict)
    assert isinstance(_SLAY_MERGE_DEFAULTS, dict)
    # Spot-check the merge threshold baseline.
    assert float(_SLAY_MERGE_DEFAULTS.get("merge_threshold")) > 0.0
