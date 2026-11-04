"""Per-adapter tuning defaults, one YAML per adapter.

Each adapter in ``src/ns3_sort_bench/adapters/`` reads its baseline knobs
from ``configs/<adapter_name>.yaml`` via :func:`load_adapter_config` at
import time, instead of carrying a hardcoded dict.  Keeping the tuning
out of Python source means:

    * `git diff src/ns3_sort_bench/configs/` shows every tuning change
      as a single, reviewable line;
    * one file per adapter — no interleaving of knobs across sorters;
    * the user can override our baseline without editing Python
      (``adapter.sort(run_kwargs={...})`` or, for fiber-kit, their own
      ``fiber-kit.yaml``); our yamls are the DEFAULTS, not the only word.

Layout:
    configs/
        __init__.py                this module
        fiber_kit.yaml             ndm_detect/extract/pca defaults + FK_*
        kilosort4.yaml             run_kwargs for the SI wrapper
        mountainsort5.yaml         (empty — reserved for future tuning)
        tridesclous2.yaml          (empty — reserved for future tuning)
        spykingcircus2.yaml        (empty — reserved for future tuning)

Shape expectations (documented per yaml, enforced by adapters):

    SI-wrapped sorters (``_SISorterBase``):
        run_kwargs: {<kw>: <value>, ...}

    fiber-kit:
        ndm_detect:   {<param>: <value>, ...}
        ndm_extract:  {<param>: <value>, ...}
        ndm_pca:      {<param>: <value>, ...}
        fiber_kit:    {FK_*: <value>, ...}        baseline merged with
                                                   the user's fiber-kit.yaml
"""
from __future__ import annotations

import os
from importlib import resources
from typing import Any, Mapping

try:
    import yaml as _yaml
except ImportError:                                     # pragma: no cover
    _yaml = None                                        # loader errors deferred


_CACHE: dict[str, Mapping[str, Any]] = {}


def load_adapter_config(name: str) -> Mapping[str, Any]:
    """Return ``configs/<name>.yaml`` as a dict, or ``{}`` if the file is
    absent / empty / unreadable.  Results are cached per name.

    A missing file is NOT an error — an adapter without baseline tuning
    (mountainsort5 today) still works; the empty yaml just documents that
    the knob surface exists and is intentionally untuned.
    """
    if name in _CACHE:
        return _CACHE[name]
    if _yaml is None:                                   # pragma: no cover
        _CACHE[name] = {}
        return _CACHE[name]
    try:
        with resources.files(__package__).joinpath(f"{name}.yaml").open("r") as f:
            data = _yaml.safe_load(f) or {}
    except (FileNotFoundError, OSError, _yaml.YAMLError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    _CACHE[name] = data
    return data


def load_adapter_config_path(name: str) -> str:
    """Absolute path to ``configs/<name>.yaml`` on the installed package.

    Used when a step needs the file path itself (e.g. fiber-session reads
    its yaml via ``FK_CONFIG``); importlib.resources returns a file-like
    object for a wheel-installed package, but for a source/editable install
    it resolves to a real path on disk.
    """
    return str(resources.files(__package__).joinpath(f"{name}.yaml"))
