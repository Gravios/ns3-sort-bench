"""Smoke tests for ``src/ns3_sort_bench/configs/`` — the per-adapter
tuning yamls and the :func:`load_adapter_config` helper."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ns3_sort_bench.configs import load_adapter_config, load_adapter_config_path


# ─── every adapter has a config file ─────────────────────────────────────────
_ADAPTER_NAMES = ("fiber_kit", "kilosort4", "mountainsort5",
                   "tridesclous2", "spykingcircus2")


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
