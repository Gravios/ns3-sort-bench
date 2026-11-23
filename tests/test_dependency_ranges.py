"""The core dependencies must let the harness share an environment with
fiber-kit and neurosuite-3's Python helpers.

An exact pin on numpy / scipy / pandas makes `pip install ns3-sort-bench`
replace the versions already installed, which breaks whatever else in
that environment was built against them (numba's numpy ceiling, the
pandas < 3 of hdmf and streamlit).  The exact, reproducible set lives in
requirements.lock instead.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _core():
    with open(_ROOT / "pyproject.toml", "rb") as fh:
        deps = tomllib.load(fh)["project"]["dependencies"]
    out = {}
    for d in deps:
        name = re.match(r"[A-Za-z0-9_.\-]+", d).group(0).lower()
        out[name] = d[len(name):].strip()
    return out


def test_core_dependencies_are_ranges():
    core = _core()
    exact = {n: s for n, s in core.items() if "==" in s}
    assert not exact, f"exact pins in the core dependencies: {exact}"
    # An upper bound is allowed only where a release changes the scores.
    capped = {n for n, s in core.items() if "<" in s}
    assert capped <= {"spikeinterface"}, capped


def test_lockfile_satisfies_the_ranges():
    """requirements.lock is one resolution of the ranges, so every core
    package's locked version must lie inside its range."""
    from packaging.specifiers import SpecifierSet
    locked = {}
    for line in (_ROOT / "requirements.lock").read_text().splitlines():
        m = re.match(r"([A-Za-z0-9_.\-]+)==(\S+)$", line.strip())
        if m:
            locked[m.group(1).lower()] = m.group(2)
    for name, spec in _core().items():
        assert name in locked, f"{name} is not in requirements.lock"
        assert locked[name] in SpecifierSet(spec), (name, spec, locked[name])


def test_shared_floors_equal_fiber_kits():
    """numpy, scipy and pyyaml are declared exactly as fiber-kit declares
    them, so the two packages can never ask for disjoint versions."""
    from importlib import metadata
    try:
        reqs = metadata.requires("fiber-kit") or []
    except metadata.PackageNotFoundError:
        pytest.skip("fiber-kit is not installed as a distribution")
    fk = {}
    for r in reqs:
        if ";" in r:                     # extras
            continue
        name = re.match(r"[A-Za-z0-9_.\-]+", r).group(0).lower()
        fk[name] = r[len(name):].strip().replace(" ", "")
    core = _core()
    for name in ("numpy", "scipy", "pyyaml"):
        assert core[name].replace(" ", "") == fk[name], (name, core[name], fk[name])
