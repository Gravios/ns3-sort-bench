"""Tests for ``ns3sb score-curated`` — reads curated .clu files from
the per-sorter klusters_<sorter>/ layout, scores against GT, and
measures curation effort."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pytest

from ns3_sort_bench.cli.commands.score_curated_cmd import run as _run


_CLU_DTYPE = np.dtype("<i4")
_RES_DTYPE = np.dtype("<i8")


def _write_res(path, times):
    np.ascontiguousarray(np.asarray(times, _RES_DTYPE)).tofile(path)


def _write_clu(path, ids):
    ids = np.ascontiguousarray(np.asarray(ids, _CLU_DTYPE))
    n_distinct = int(np.unique(ids).size)
    with open(path, "wb") as f:
        np.array([n_distinct], _CLU_DTYPE).tofile(f)
        ids.tofile(f)


def _make_session_dir(root: Path, sorter: str, item: str,
                       res: np.ndarray, clu_raw: np.ndarray,
                       raw_tag: str,
                       clu_curated: np.ndarray | None = None,
                       cur_log: list[dict] | None = None):
    d = root / f"klusters_{sorter}" / item
    d.mkdir(parents=True)
    (d / "sim_session.yaml").write_text("acquisitionSystem:\n  samplingRate: 20000\n")
    _write_res(d / "sim_session.res.standard.1", res)
    _write_clu(d / f"sim_session.clu.standard.1.{raw_tag}", clu_raw)
    if clu_curated is not None:
        _write_clu(d / "sim_session.clu.standard.1.curated", clu_curated)
    if cur_log is not None:
        import json
        with open(d / "sim_session.cur.standard.1.curated", "w") as f:
            for row in cur_log:
                f.write(json.dumps(row) + "\n")
    return d


def _args(output_dir, **overrides):
    ns = argparse.Namespace(
        output_dir=str(output_dir),
        curated_tag="curated",
        require_curated=False,
        **overrides,
    )
    return ns


# ─── end-to-end: raw + curated both scored, effort recorded ─────────────────
def test_score_curated_writes_both_stages_and_effort(tmp_path):
    """One sorter + one item + raw + curated + cur log: produces one
    (raw, item) row and one (curated, item) row per scorer-metric, plus
    one curation-effort row carrying both diff and log numbers.
    """
    # 100 spikes.  GT has one cell with all 100.  Raw sorter over-splits
    # into 3 fragments; curated merges them back into one cluster.
    res = np.arange(1000, 1000 + 100 * 100, 100, dtype=np.int64)        # 100 spikes
    gt_clu     = np.full(100, 2, dtype=np.int32)                        # all GT
    raw_clu    = np.tile([2, 3, 4], 100)[:100].astype(np.int32)         # 3 fragments
    curated    = np.full(100, 5, dtype=np.int32)                        # merged

    item = "it1"
    _make_session_dir(tmp_path, "ground_truth", item,
                       res, gt_clu, raw_tag="ground_truth")
    _make_session_dir(tmp_path, "fiber_kit_session", item,
                       res, raw_clu, raw_tag="fiber_session",
                       clu_curated=curated,
                       cur_log=[
                           # 2 GROUP actions (merges) merging the 3 fragments
                           {"action_idx": 1, "action": "GROUP", "status": "good"},
                           {"action_idx": 2, "action": "GROUP", "status": "good"},
                       ])

    rc = _run(_args(tmp_path))
    assert rc == 0

    # curated_scores.csv has BOTH stages
    sc_path = tmp_path / "curated_scores.csv"
    assert sc_path.exists()
    rows = list(csv.DictReader(sc_path.open()))
    stages = {r["stage"] for r in rows}
    assert stages == {"raw", "curated"}

    # The curated per_cell IoU is strictly better than the raw's.
    def pc_iou(stage):
        for r in rows:
            if (r["scorer"], r["metric"], r["stage"]) == \
               ("per_cell", "best_unit_accuracy_mean", stage):
                return float(r["value"])
        pytest.fail(f"no per_cell row for stage={stage}")

    raw_iou  = pc_iou("raw")
    cur_iou  = pc_iou("curated")
    assert cur_iou > raw_iou
    assert cur_iou == pytest.approx(1.0, rel=1e-6)   # curated collapses to GT

    # curation_effort.csv carries both diff and log columns
    ef_path = tmp_path / "curation_effort.csv"
    rows_ef = list(csv.DictReader(ef_path.open()))
    assert len(rows_ef) == 1
    ef = rows_ef[0]
    assert ef["sorter"]            == "fiber_kit_session"
    assert ef["corpus_item"]       == item
    assert int(ef["n_total"])      == 100
    assert int(ef["n_reassignments"]) == 100           # every spike re-labelled
    assert int(ef["n_merges"])     >= 1                # 3 → 1
    assert int(ef["n_actions_good"]) == 2              # from the .cur log
    assert float(ef["complexity_score"]) == pytest.approx(2.0, rel=1e-6)


# ─── raw-only session has no curated rows, no effort row ────────────────────
def test_score_curated_skips_sessions_without_curated_clu(tmp_path):
    res = np.arange(1000, 1000 + 20 * 100, 100, dtype=np.int64)
    _make_session_dir(tmp_path, "ground_truth", "x",
                       res, np.full(20, 2, np.int32), raw_tag="ground_truth")
    _make_session_dir(tmp_path, "fiber_kit_session", "x",
                       res, np.full(20, 3, np.int32), raw_tag="fiber_session")
    # no curated .clu

    rc = _run(_args(tmp_path))
    assert rc == 0
    rows = list(csv.DictReader((tmp_path / "curated_scores.csv").open()))
    # Still produced raw rows (useful cross-check vs GT) but zero curated rows
    assert {r["stage"] for r in rows} == {"raw"}
    assert not (tmp_path / "curation_effort.csv").exists() or \
           list(csv.DictReader((tmp_path / "curation_effort.csv").open())) == []


# ─── inter-sorter comparison: two sorters, same item, both curated ──────────
def test_score_curated_handles_multiple_sorters_on_one_item(tmp_path):
    """Two sorters both curated on the same item — the output tables must
    carry both sorter identities so a pivot can compare curation effort
    and achieved parity across them."""
    res = np.arange(1000, 1000 + 60 * 100, 100, dtype=np.int64)
    gt  = np.full(60, 2, np.int32)
    _make_session_dir(tmp_path, "ground_truth", "it", res, gt,
                       raw_tag="ground_truth")

    # Sorter A: moderate over-split (3 fragments → merged)
    raw_a = np.tile([2, 3, 4], 60)[:60].astype(np.int32)
    _make_session_dir(tmp_path, "fiber_kit_session", "it",
                       res, raw_a, raw_tag="fiber_session",
                       clu_curated=np.full(60, 5, np.int32),
                       cur_log=[{"action_idx": 1, "action": "GROUP", "status": "good"},
                                {"action_idx": 2, "action": "GROUP", "status": "good"}])

    # Sorter B: already clean, no curation needed — curated == raw
    raw_b = np.full(60, 7, np.int32)
    _make_session_dir(tmp_path, "mountainsort5", "it",
                       res, raw_b, raw_tag="mountainsort5",
                       clu_curated=raw_b.copy(),
                       cur_log=[])

    rc = _run(_args(tmp_path))
    assert rc == 0
    rows_ef = list(csv.DictReader((tmp_path / "curation_effort.csv").open()))
    by_sorter = {r["sorter"]: r for r in rows_ef}
    assert set(by_sorter) == {"fiber_kit_session", "mountainsort5"}
    # fiber_kit_session needed work; mountainsort5 didn't
    assert int(by_sorter["fiber_kit_session"]["n_reassignments"]) == 60
    assert int(by_sorter["mountainsort5"]["n_reassignments"])     == 0
    assert float(by_sorter["fiber_kit_session"].get("complexity_score", "0")) > \
           float(by_sorter["mountainsort5"].get("complexity_score", "0"))


# ─── no GT: scorer can't run against this item ──────────────────────────────
def test_score_curated_warns_on_missing_gt(tmp_path, capsys):
    res = np.arange(1000, 1000 + 20 * 100, 100, dtype=np.int64)
    _make_session_dir(tmp_path, "fiber_kit_session", "lonely",
                       res, np.full(20, 3, np.int32), raw_tag="fiber_session",
                       clu_curated=np.full(20, 4, np.int32))
    # no ground_truth/ dir

    rc = _run(_args(tmp_path))
    out = capsys.readouterr().out
    assert rc == 0
    assert "no GT for item" in out
    # No scores were produced (nothing to compare against)
    sc_path = tmp_path / "curated_scores.csv"
    assert not sc_path.exists() or \
           not list(csv.DictReader(sc_path.open()))
