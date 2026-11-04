"""Curation-effort diff + Klusters .cur-log parse.

Hermetic: all inputs are fabricated byte-exact into tmp_path; no neurosuite-3
or fiber-kit installation needed.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import pytest

from ns3_sort_bench.curation import (
    curate_diff,
    diff_clus,
    parse_cur_log,
    _res_path_for_clu,
    _cur_path_for_clu,
    _swap_type_token,
    _ACTION_WEIGHT,
)


# ─── path derivation ────────────────────────────────────────────────────────
def test_swap_type_token_handles_stage_suffix():
    assert _swap_type_token("/x/base.clu.standard.1.final", "res") == \
           "/x/base.res.standard.1.final"
    assert _swap_type_token("/x/base.clu.stderiv_C5_D34.5.linked", "cur") == \
           "/x/base.cur.stderiv_C5_D34.5.linked"


def test_swap_type_token_not_a_clu_path():
    with pytest.raises(ValueError, match="not a .clu path"):
        _swap_type_token("/x/base.fet.standard.1", "res")


def test_res_path_prefers_staged_over_shared(tmp_path):
    """A stage-tagged .res (reextract/realign) wins over the shared one."""
    base = tmp_path / "s"
    (base.with_suffix(".clu.standard.1.realigned"))  # path only
    shared = Path(str(base) + ".res.standard.1"); shared.write_bytes(b"")
    staged = Path(str(base) + ".res.standard.1.realigned"); staged.write_bytes(b"")
    clu_path = str(base) + ".clu.standard.1.realigned"
    assert _res_path_for_clu(clu_path) == str(staged)
    staged.unlink()
    assert _res_path_for_clu(clu_path) == str(shared)


def test_cur_path_mirrors_clu_suffix():
    assert _cur_path_for_clu("/x/s.clu.standard.5.manual") == \
           "/x/s.cur.standard.5.manual"


# ─── log parsing ────────────────────────────────────────────────────────────
def _write_cur(path: Path, actions):
    """actions: list of (action_idx, action_type, status, rows_per_action)
    where rows_per_action is how many before/after rows to emit (simulating
    the real log's multiple-row-per-action layout)."""
    with path.open("w") as f:
        for aidx, atype, status, n_rows in actions:
            for i in range(n_rows):
                f.write(json.dumps({
                    "event": "before" if i == 0 else "after",
                    "action": atype,
                    "action_idx": aidx,
                    "status": status,
                }) + "\n")


def test_parse_cur_log_dedupes_by_action_idx(tmp_path):
    log = tmp_path / "s.cur.standard.1"
    _write_cur(log, [
        (1, "GROUP",     "good", 3),
        (2, "SPLIT",     "good", 2),
        (3, "RECLUSTER", "good", 4),
        (4, "GROUP",     "bad",  3),   # undo
    ])
    s = parse_cur_log(str(log))
    assert s.n_actions_good == 3
    assert s.n_actions_bad == 1
    assert s.actions_by_type_good == {"GROUP": 1, "SPLIT": 1, "RECLUSTER": 1}
    assert s.actions_by_type_bad  == {"GROUP": 1}
    # 1 × GROUP + 1 × SPLIT + 1 × RECLUSTER = 1 + 2 + 3 = 6
    assert s.complexity_score == 1.0 + 2.0 + 3.0


def test_parse_cur_log_weights_are_tunable(tmp_path):
    log = tmp_path / "s.cur.standard.1"
    _write_cur(log, [(1, "GROUP", "good", 2), (2, "SPLIT", "good", 2)])
    s = parse_cur_log(str(log), weights={"GROUP": 10.0, "SPLIT": 100.0})
    assert s.complexity_score == 110.0


def test_parse_cur_log_tolerates_malformed_lines(tmp_path):
    log = tmp_path / "s.cur.standard.1"
    log.write_text(
        '{"event":"before","action":"GROUP","action_idx":1,"status":"good"}\n'
        'not valid json\n'
        '{"event":"after","action":"GROUP","action_idx":1,"status":"good"}\n'
        '{"event":"ACTION_DETAIL","action":"GROUP","action_idx":1,"status":"good"}\n'
    )
    s = parse_cur_log(str(log))
    assert s.n_actions_good == 1


def test_parse_cur_log_known_action_types_covered():
    """Every ActionType that curationlogger.cpp emits must have a weight."""
    expected = {
        "GROUP", "SPLIT", "SPLIT_N", "RECLUSTER", "REALIGN", "NUDGE",
        "DELETE_NOISE", "DELETE_ARTEFACT",
        "DELETE_REGION_NOISE", "DELETE_REGION_ARTEFACT",
        "MOVE_SPIKES", "RENUMBER_PARTIAL", "WATERSHED", "SORT_ALL",
    }
    assert expected <= set(_ACTION_WEIGHT)


# ─── diff over two .clu arrays ──────────────────────────────────────────────
def test_diff_clus_identical_zero_reassignments():
    a = np.array([2, 2, 3, 3, 4, 4, 2], dtype=np.int64)
    d = diff_clus(a, a)
    assert d.n_reassignments == 0
    assert d.fraction_touched == 0.0
    assert d.n_merges == 0 and d.n_splits == 0
    assert d.n_units_before == d.n_units_after == 3


def test_diff_clus_pure_merge():
    """Before has units 2, 3; curator merged 3 into 2 → after has only 2."""
    before = np.array([2, 2, 3, 3, 2], dtype=np.int64)
    after  = np.array([2, 2, 2, 2, 2], dtype=np.int64)
    d = diff_clus(before, after)
    assert d.n_reassignments == 2
    assert d.n_units_before == 2 and d.n_units_after == 1
    assert d.n_merges == 1 and d.n_splits == 0


def test_diff_clus_pure_split():
    """Before has one unit 2; curator split it into 2 + 3."""
    before = np.array([2, 2, 2, 2, 2], dtype=np.int64)
    after  = np.array([2, 2, 3, 3, 2], dtype=np.int64)
    d = diff_clus(before, after)
    assert d.n_reassignments == 2
    assert d.n_units_before == 1 and d.n_units_after == 2
    assert d.n_splits == 1 and d.n_merges == 0


def test_diff_clus_delete_to_noise_counts_as_reassignment():
    """exclude_reserved drops noise/artefact from unit counts but counts the
    spikes as reassigned (the curator moved them)."""
    before = np.array([2, 2, 3, 3, 4], dtype=np.int64)
    after  = np.array([2, 2, 3, 3, 1], dtype=np.int64)   # unit 4 → noise
    d = diff_clus(before, after)
    assert d.n_reassignments == 1
    assert d.n_units_before == 3 and d.n_units_after == 2


def test_diff_clus_mismatched_sizes_raises():
    with pytest.raises(ValueError, match="same .res is required"):
        diff_clus(np.array([2, 3]), np.array([2, 3, 4]))


# ─── end-to-end curate_diff (binary files + log) ────────────────────────────
def _write_res(path: Path, times):
    np.asarray(times, dtype="<i8").tofile(path)


def _write_clu(path: Path, ids):
    ids = np.asarray(ids, dtype="<i4")
    n_distinct = int(np.unique(ids).size) if ids.size else 0
    with path.open("wb") as f:
        np.array([n_distinct], dtype="<i4").tofile(f)
        ids.tofile(f)


def test_curate_diff_end_to_end_with_log(tmp_path):
    base = tmp_path / "s"
    _write_res(Path(str(base) + ".res.standard.1"), [10, 20, 30, 40, 50])
    before_ids = [2, 2, 3, 3, 2]
    after_ids  = [2, 2, 2, 2, 2]
    _write_clu(Path(str(base) + ".clu.standard.1"), before_ids)
    after_path = Path(str(base) + ".clu.standard.1.curated")
    _write_clu(after_path, after_ids)
    _write_cur(Path(str(base) + ".cur.standard.1.curated"), [
        (1, "GROUP", "good", 2),
    ])
    report = curate_diff(
        str(base) + ".clu.standard.1",
        str(after_path),
    )
    assert report.diff.n_reassignments == 2
    assert report.diff.n_merges == 1
    assert report.log is not None
    assert report.log.n_actions_good == 1
    assert report.log.complexity_score == _ACTION_WEIGHT["GROUP"]


def test_curate_diff_missing_cur_log_is_fine(tmp_path):
    """A `.cur` log is optional — diff metrics still land."""
    base = tmp_path / "s"
    _write_res(Path(str(base) + ".res.standard.1"), [10, 20])
    _write_clu(Path(str(base) + ".clu.standard.1"),         [2, 3])
    _write_clu(Path(str(base) + ".clu.standard.1.curated"), [2, 2])
    report = curate_diff(
        str(base) + ".clu.standard.1",
        str(base) + ".clu.standard.1.curated",
    )
    assert report.log is None
    assert report.diff.n_reassignments == 1
    # to_rows still carries the diff rows
    sources = {r["source"] for r in report.to_rows()}
    assert sources == {"diff"}


def test_curate_diff_missing_res_raises(tmp_path):
    base = tmp_path / "s"
    _write_clu(Path(str(base) + ".clu.standard.1"),         [2, 3])
    _write_clu(Path(str(base) + ".clu.standard.1.curated"), [2, 2])
    with pytest.raises(FileNotFoundError, match="no .res"):
        curate_diff(
            str(base) + ".clu.standard.1",
            str(base) + ".clu.standard.1.curated",
        )


# ─── CLI ────────────────────────────────────────────────────────────────────
def test_cli_curate_diff_text_output(tmp_path, capsys):
    base = tmp_path / "s"
    _write_res(Path(str(base) + ".res.standard.1"), [10, 20, 30])
    _write_clu(Path(str(base) + ".clu.standard.1"),         [2, 3, 3])
    _write_clu(Path(str(base) + ".clu.standard.1.curated"), [2, 2, 2])
    _write_cur(Path(str(base) + ".cur.standard.1.curated"), [
        (1, "GROUP", "good", 2),
    ])
    from ns3_sort_bench.cli.main import main
    rc = main(["curate-diff",
                str(base) + ".clu.standard.1",
                str(base) + ".clu.standard.1.curated"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "n_reassignments" in out
    assert "n_actions_good" in out
    assert "complexity_score" in out
    assert "GROUP" in out


def test_cli_curate_diff_csv_format(tmp_path, capsys):
    base = tmp_path / "s"
    _write_res(Path(str(base) + ".res.standard.1"), [10, 20])
    _write_clu(Path(str(base) + ".clu.standard.1"),         [2, 3])
    _write_clu(Path(str(base) + ".clu.standard.1.curated"), [2, 2])
    from ns3_sort_bench.cli.main import main
    rc = main(["curate-diff",
                str(base) + ".clu.standard.1",
                str(base) + ".clu.standard.1.curated",
                "--format", "csv"])
    out = capsys.readouterr().out
    assert rc == 0
    header = out.splitlines()[0]
    assert "before_clu,after_clu,source,metric,value" == header


def test_cli_curate_diff_missing_file(tmp_path, capsys):
    from ns3_sort_bench.cli.main import main
    rc = main(["curate-diff", "/does/not/exist.clu", "/also/nope.clu"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no such file" in err
