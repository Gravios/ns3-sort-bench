"""`_resolve_sort_pair` resolution — staged `.res` vs. shared `.res`.

Per neurosuite-3's canonical naming (doc/ndmanager-plugins/formats/naming.md):
  * `.clu` is MethodSpecific — strict `<base>.clu.<variant>.<group>[.<tag>]`.
  * `.res` is Shared — one per group; a reextract stage writes a staged copy
    at `<base>.res.<variant>.<group>.<tag>`, otherwise the shared
    `<base>.res.<variant>.<group>` (no tag) is the one that pairs with the
    tagged `.clu`.

Resolution: try staged first, fall back to shared.  Mirrors fiber-kit's own
`read_res_at` / `read_res` split.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ns3_sort_bench.ns3.extractors import _resolve_sort_pair


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_shared_res_pairs_with_tagged_clu(tmp_path):
    """fiber-session default: untagged shared `.res` + tagged `.clu`.
    This is the exact case that broke on the paired-recordings smoke run."""
    base = str(tmp_path / "sim_session")
    _touch(Path(f"{base}.res.standard.1"))
    _touch(Path(f"{base}.clu.standard.1.fiber_session"))
    res, clu = _resolve_sort_pair(base, elec=1,
                                   variant="standard", tag="fiber_session")
    assert res == f"{base}.res.standard.1"
    assert clu == f"{base}.clu.standard.1.fiber_session"


def test_staged_res_wins_when_both_exist(tmp_path):
    """A stage that reextracted writes `<base>.res.<v>.<g>.<tag>` next to the
    shared copy; resolution must prefer the staged one so the .res count
    matches the .clu count."""
    base = str(tmp_path / "sim_session")
    _touch(Path(f"{base}.res.standard.1"))
    _touch(Path(f"{base}.res.standard.1.realigned"))
    _touch(Path(f"{base}.clu.standard.1.realigned"))
    res, clu = _resolve_sort_pair(base, elec=1,
                                   variant="standard", tag="realigned")
    assert res == f"{base}.res.standard.1.realigned"
    assert clu == f"{base}.clu.standard.1.realigned"


def test_missing_clu_raises(tmp_path):
    base = str(tmp_path / "sim_session")
    _touch(Path(f"{base}.res.standard.1"))
    with pytest.raises(FileNotFoundError, match="clu"):
        _resolve_sort_pair(base, elec=1, variant="standard",
                            tag="fiber_session")


def test_missing_both_res_copies_raises_with_both_paths(tmp_path):
    base = str(tmp_path / "sim_session")
    _touch(Path(f"{base}.clu.standard.1.fiber_session"))
    with pytest.raises(FileNotFoundError) as ei:
        _resolve_sort_pair(base, elec=1, variant="standard",
                            tag="fiber_session")
    msg = str(ei.value)
    # The error message names both candidate paths, so the user knows which
    # staged/shared file is missing.
    assert "standard.1.fiber_session" in msg
    assert "standard.1" in msg
    assert "staged" in msg and "shared" in msg


def test_empty_tag_treats_shared_as_strict(tmp_path):
    """No stage tag — just `.res.<v>.<g>` + `.clu.<v>.<g>`; no fallback needed."""
    base = str(tmp_path / "sim_session")
    _touch(Path(f"{base}.res.standard.5"))
    _touch(Path(f"{base}.clu.standard.5"))
    res, clu = _resolve_sort_pair(base, elec=5, variant="standard", tag="")
    assert res == f"{base}.res.standard.5"
    assert clu == f"{base}.clu.standard.5"
