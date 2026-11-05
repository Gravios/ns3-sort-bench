"""A reused --keep-fk-artifacts folder is emptied before each fiber_kit sort.

fiber-pipeline consumes stage outputs already on disk, so leftovers from an
earlier run into the same --out silently changed the next run's result.
"""
from __future__ import annotations

import pytest

from ns3_sort_bench.adapters.fiber_kit import _fresh_artifacts_dir


def test_leftovers_are_removed(tmp_path, capsys):
    d = tmp_path / "fiber_kit" / "item"
    d.mkdir(parents=True)
    (d / "sim_session.clu.standard.1.fiber_session").write_text("4\n")
    (d / "sub").mkdir()
    (d / "sub" / "x").write_text("x")
    out = _fresh_artifacts_dir(str(tmp_path / "fiber_kit"), "item")
    assert out == str(d)
    assert d.is_dir() and list(d.iterdir()) == []
    assert "clearing artifacts" in capsys.readouterr().err


def test_new_folder_is_created_quietly(tmp_path, capsys):
    out = _fresh_artifacts_dir(str(tmp_path / "fk"), "item")
    assert (tmp_path / "fk" / "item").is_dir()
    assert out.endswith("item")
    assert capsys.readouterr().err == ""


def test_siblings_are_left_alone(tmp_path):
    root = tmp_path / "fk"
    (root / "other").mkdir(parents=True)
    (root / "other" / "keep").write_text("k")
    (root / "item").mkdir()
    (root / "item" / "old").write_text("o")
    _fresh_artifacts_dir(str(root), "item")
    assert (root / "other" / "keep").exists()


@pytest.mark.parametrize("slug", ["", ".", ".."])
def test_unsafe_slugs_refused(tmp_path, slug):
    with pytest.raises(ValueError):
        _fresh_artifacts_dir(str(tmp_path), slug)
