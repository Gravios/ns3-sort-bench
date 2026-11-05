"""Which tuning a fiber_kit sort merged is recorded, and implicit pickup is loud.

Two runs of the same command from different folders sorted differently when
one folder held a ./fiber-kit.yaml; nothing in the run said so.
"""
from __future__ import annotations

import os

from ns3_sort_bench.adapters.fiber_kit import (
    _FiberKitBase, _resolve_user_fk_config, fk_config_digest,
)


def test_no_user_tuning(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("FK_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    assert _resolve_user_fk_config() == (None, "none")
    path = _FiberKitBase._write_fk_config(str(tmp_path))
    head = open(path).read().split("\n", 6)[:6]
    assert any("no user tuning" in l for l in head)
    assert not any("user tuning from" in l for l in head)
    assert capsys.readouterr().err == ""


def test_cwd_pickup_is_reported(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("FK_CONFIG", raising=False)
    (tmp_path / "fiber-kit.yaml").write_text("FK_CHUNKS: 7\n")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"; out.mkdir()
    path = _FiberKitBase._write_fk_config(str(out))
    assert "current folder" in capsys.readouterr().err
    assert "current folder" in open(path).read()
    src, origin = _resolve_user_fk_config(quiet=True)
    assert src == str(tmp_path / "fiber-kit.yaml")
    assert capsys.readouterr().err == ""          # quiet really is quiet


def test_named_missing_file_reported_and_not_fallen_through(tmp_path,
                                                           monkeypatch, capsys):
    (tmp_path / "fiber-kit.yaml").write_text("FK_CHUNKS: 7\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FK_CONFIG", str(tmp_path / "nope.yaml"))
    src, origin = _resolve_user_fk_config()
    assert src is None and "not found" in origin
    assert "not found" in capsys.readouterr().err


def test_env_and_explicit_order(tmp_path, monkeypatch):
    a = tmp_path / "a.yaml"; a.write_text("FK_CHUNKS: 1\n")
    b = tmp_path / "b.yaml"; b.write_text("FK_CHUNKS: 2\n")
    monkeypatch.setenv("FK_CONFIG", str(a))
    assert _resolve_user_fk_config() == (str(a), "$FK_CONFIG")
    assert _resolve_user_fk_config(str(b)) == (str(b), "fk_config_path")


def test_digest_ignores_comments_and_tracks_values(tmp_path):
    p1 = tmp_path / "1.yaml"; p1.write_text("# from /x\nFK_CHUNKS: 7\n")
    p2 = tmp_path / "2.yaml"; p2.write_text("# from /elsewhere\nFK_CHUNKS: 7\n")
    p3 = tmp_path / "3.yaml"; p3.write_text("# from /x\nFK_CHUNKS: 8\n")
    assert fk_config_digest(str(p1)) == fk_config_digest(str(p2))
    assert fk_config_digest(str(p1)) != fk_config_digest(str(p3))


def test_same_tuning_same_digest_across_folders(tmp_path, monkeypatch):
    monkeypatch.delenv("FK_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    d1 = tmp_path / "r1"; d1.mkdir()
    d2 = tmp_path / "r2"; d2.mkdir()
    a = _FiberKitBase._write_fk_config(str(d1))
    b = _FiberKitBase._write_fk_config(str(d2))
    assert fk_config_digest(a) == fk_config_digest(b)
