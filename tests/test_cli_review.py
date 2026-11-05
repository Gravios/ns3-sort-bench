"""Smoke tests for the lab-session corpus key and the summary/compare commands.

These are hermetic:
  - we don't actually load a session yaml (we check only that the key parses
    into the right loader kwargs by monkeypatching the loader);
  - we write a tiny synthetic scores.csv and verify summary + compare on it.
"""
from __future__ import annotations

import io
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pandas as pd
import pytest


# ─── lab-session key parsing ─────────────────────────────────────────────────
def test_lab_key_defaults(monkeypatch):
    """`lab:/path/to.yaml` → group=7, variant=stderiv_C5_D34, tag=exp_final."""
    from ns3_sort_bench.runners import sorter_grid

    captured = {}

    def _fake(path, *, electrode_group, variant, tag):
        captured.update(dict(path=path, group=electrode_group,
                             variant=variant, tag=tag))
        return "ok"

    monkeypatch.setattr("ns3_sort_bench.corpus.lab_sessions.load", _fake)
    out = sorter_grid._resolve_corpus("lab:/data/g7_session.yaml")
    assert out == "ok"
    assert captured == dict(
        path="/data/g7_session.yaml", group=7,
        variant="stderiv_C5_D34", tag="exp_final",
    )


def test_lab_key_all_fields(monkeypatch):
    from ns3_sort_bench.runners import sorter_grid

    captured = {}

    def _fake(path, *, electrode_group, variant, tag):
        captured.update(dict(path=path, group=electrode_group,
                             variant=variant, tag=tag))

    monkeypatch.setattr("ns3_sort_bench.corpus.lab_sessions.load", _fake)
    sorter_grid._resolve_corpus(
        "lab:/data/g6_0316.yaml:6:stderiv_C5_D34:exp_final"
    )
    assert captured == dict(
        path="/data/g6_0316.yaml", group=6,
        variant="stderiv_C5_D34", tag="exp_final",
    )


def test_lab_key_windows_path(monkeypatch):
    """`lab:C:/data/x.yaml:7:...` tolerates a Windows drive colon."""
    from ns3_sort_bench.runners import sorter_grid

    captured = {}

    def _fake(path, *, electrode_group, variant, tag):
        captured["path"] = path
        captured["group"] = electrode_group

    monkeypatch.setattr("ns3_sort_bench.corpus.lab_sessions.load", _fake)
    sorter_grid._resolve_corpus("lab:C:/data/x.yaml:7")
    assert captured["path"] == "C:/data/x.yaml"
    assert captured["group"] == 7


def test_unknown_key_message_lists_lab():
    """Error message should mention `lab:` so the user discovers it."""
    from ns3_sort_bench.runners import sorter_grid

    with pytest.raises(KeyError) as ei:
        sorter_grid._resolve_corpus("not_a_known_prefix:x")
    assert "lab:<yaml>" in str(ei.value)


# ─── summary / compare on a tiny synthetic scores.csv ───────────────────────
def _write_scores(path: Path, rows):
    df = pd.DataFrame(rows, columns=[
        "scorer", "corpus_item", "reference_adapter",
        "candidate_adapter", "metric", "value",
    ])
    df.to_csv(path, index=False)


def test_summary_prints_pivot(tmp_path):
    """summary on one run → pivot has every (corpus, adapter, metric).
    Uses --format text to inspect the raw pivot shape.
    """
    rundir = tmp_path / "run_a"
    rundir.mkdir()
    _write_scores(rundir / "scores.csv", [
        ("per_cell", "sim1", "gt", "fiber_kit", "accuracy", 0.9),
        ("per_cell", "sim1", "gt", "botm_ref",   "accuracy", 0.7),
        ("per_cell", "sim1", "gt", "fiber_kit", "f1",       0.88),
        ("per_cell", "sim1", "gt", "botm_ref",   "f1",       0.72),
    ])
    from ns3_sort_bench.cli.main import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["summary", str(rundir), "--format", "text"])
    out = buf.getvalue()
    assert rc == 0
    assert "fiber_kit" in out and "botm_ref" in out
    assert "accuracy" in out and "f1" in out
    assert "0.9000" in out and "0.7200" in out


def test_summary_org_table_layout(tmp_path):
    """Default --format org: one org-style table per corpus, pipe-separated
    header, `|---+---|` separator, short metric names, right-aligned nums.
    """
    rundir = tmp_path / "org_run"
    rundir.mkdir()
    _write_scores(rundir / "scores.csv", [
        # Default-column-set metrics + a count column
        ("per_cell", "sim1", "gt", "fiber_kit_session", "best_unit_accuracy_mean", 0.049),
        ("per_cell", "sim1", "gt", "fiber_kit_session", "best_unit_recall_mean",   0.146),
        ("per_cell", "sim1", "gt", "fiber_kit_session", "n_cand_units",            28.0),
        ("per_cell", "sim1", "gt", "kilosort4",         "best_unit_accuracy_mean", 0.051),
        ("per_cell", "sim1", "gt", "kilosort4",         "best_unit_recall_mean",   0.659),
        ("per_cell", "sim1", "gt", "kilosort4",         "n_cand_units",            2.0),
    ])
    from ns3_sort_bench.cli.main import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["summary", str(rundir)])          # default --format org
    out = buf.getvalue()
    assert rc == 0

    # One section header per corpus
    assert "# sim1" in out

    # Pipe-separated header + org separator (uses `+` at column joins)
    assert "| adapter" in out
    assert "|-" in out and "+-" in out

    # Short metric aliases replace the long scores.csv names
    assert "| iou" in out or "iou " in out
    assert "recall" in out
    assert "n_cand" in out
    # Full long names must NOT appear in the org output
    assert "best_unit_accuracy_mean" not in out

    # Count columns render as ints (no "28.0000")
    assert "28" in out
    assert "28.0000" not in out
    # Floats keep the --decimals width
    assert "0.0490" in out and "0.6590" in out


def test_summary_org_multi_corpus_prints_two_tables(tmp_path):
    rundir = tmp_path / "org_multi"
    rundir.mkdir()
    _write_scores(rundir / "scores.csv", [
        ("per_cell", "simA", "gt", "fiber_kit_session", "best_unit_accuracy_mean", 0.05),
        ("per_cell", "simB", "gt", "fiber_kit_session", "best_unit_accuracy_mean", 0.12),
    ])
    from ns3_sort_bench.cli.main import main
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["summary", str(rundir)])
    out = buf.getvalue()
    assert rc == 0
    # Two separate tables with their own section headers, in sorted order
    assert out.index("# simA") < out.index("# simB")
    # Each corpus has its own separator row
    assert out.count("|-") >= 2


def test_summary_org_separates_pipeline_stages_from_primaries(tmp_path):
    """Pipeline stage adapters (named `<primary>@<tag>`) must render in a
    `## <primary> stages` sub-table beneath the main one — the primary
    sorters' scoreboard stays uncluttered, and the stages sit next to
    their primary with the final-chain `final` row for a continuous
    read.
    """
    rundir = tmp_path / "stages_run"
    rundir.mkdir()
    # Rows in plan order so the stage sub-table renders in plan order
    # (we pivot with sort=False to honour this).
    _write_scores(rundir / "scores.csv", [
        ("per_cell", "sim1", "gt", "botm_ref",   "best_unit_accuracy_mean", 0.00),
        ("per_cell", "sim1", "gt", "kilosort4",  "best_unit_accuracy_mean", 0.05),
        ("per_cell", "sim1", "gt", "fiber_kit",  "best_unit_accuracy_mean", 0.048),
        ("per_cell", "sim1", "gt", "fiber_kit@fiber_session",
                                                "best_unit_accuracy_mean", 0.044),
        ("per_cell", "sim1", "gt", "fiber_kit@fs_clean",
                                                "best_unit_accuracy_mean", 0.046),
        ("per_cell", "sim1", "gt", "fiber_kit@exp_xcm",
                                                "best_unit_accuracy_mean", 0.047),
    ])
    from ns3_sort_bench.cli.main import main
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["summary", str(rundir)])
    out = buf.getvalue()
    assert rc == 0

    # Main table + sub-table have their own heading levels.
    assert "# sim1" in out
    assert "## fiber_kit stages" in out
    # Main table lists only primaries (one row each; the `@` rows go
    # below).
    main_block = out.split("## fiber_kit stages")[0]
    stages_block = out.split("## fiber_kit stages")[1]
    assert "| fiber_kit "  in main_block
    assert "fiber_session" not in main_block     # stages left to sub-table
    assert "fs_clean"      not in main_block
    assert "exp_xcm"       not in main_block
    assert "fiber_kit@"    not in main_block     # "@" format kept out

    # Sub-table carries the three stages in plan order + the `final` row.
    i_fs    = stages_block.index("fiber_session")
    i_clean = stages_block.index("fs_clean")
    i_xcm   = stages_block.index("exp_xcm")
    i_final = stages_block.index("final")
    assert i_fs < i_clean < i_xcm < i_final
    # The sub-table's first column header is `stage`, not `adapter`.
    assert "| stage " in stages_block
    # And its separator is org-style.
    assert "+-" in stages_block.split("| stage")[1]


def test_summary_merges_curated_scores_csv_as_stage_rows(tmp_path):
    """When `curated_scores.csv` lives beside `scores.csv`, `ns3sb summary`
    must fold its `stage=curated` rows into the main pivot as
    `<sorter>@curated` candidate adapters — they then render under the
    stage sub-table AFTER the `final` row, so the raw-to-curated delta
    is visible in the same block the pipeline stages already use.
    """
    rundir = tmp_path / "curated_run"
    rundir.mkdir()
    _write_scores(rundir / "scores.csv", [
        ("per_cell", "sim1", "gt", "fiber_kit", "best_unit_accuracy_mean", 0.048),
        ("per_cell", "sim1", "gt", "fiber_kit", "best_unit_recall_mean",   0.691),
        ("per_cell", "sim1", "gt", "kilosort4", "best_unit_accuracy_mean", 0.051),
        ("per_cell", "sim1", "gt", "kilosort4", "best_unit_recall_mean",   0.659),
    ])
    # curated_scores.csv has the shape score-curated writes.
    import pandas as pd
    pd.DataFrame([
        ("fiber_kit", "sim1", "raw",     "per_cell",
         "best_unit_accuracy_mean", 0.048),
        ("fiber_kit", "sim1", "curated", "per_cell",
         "best_unit_accuracy_mean", 0.42),
        ("fiber_kit", "sim1", "curated", "per_cell",
         "best_unit_recall_mean",   0.85),
    ], columns=["sorter", "corpus_item", "stage",
                "scorer", "metric", "value"]
    ).to_csv(rundir / "curated_scores.csv", index=False)

    from ns3_sort_bench.cli.main import main
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["summary", str(rundir)])
    out = buf.getvalue()
    assert rc == 0

    # The curated row must NOT live in the primary scoreboard block —
    # it belongs under fiber_kit's stage sub-table.
    main_block = out.split("## fiber_kit stages")[0]
    stages_block = out.split("## fiber_kit stages")[1]
    assert "curated" not in main_block
    # And it must appear AFTER the final row — curation happens on top
    # of the final sort, so rendering it earlier would mis-order the
    # workflow.
    assert stages_block.index("final") < stages_block.index("curated")
    # The curated's own numbers made it in (sanity check).
    assert "0.4200" in stages_block
    assert "0.8500" in stages_block
    # The raw-stage row from curated_scores.csv is DROPPED (score-curated
    # re-scores the sorter's own .clu against GT and emits stage=raw,
    # which would duplicate the primary scores.csv row).  0.048 shows
    # up exactly twice — once in the main table for fiber_kit and once
    # as the `final` row in fiber_kit's stage block (that row is the
    # SAME primary) — NOT a third time from a raw-stage merge.
    assert out.count("0.0480") == 2
    # Nothing in the stages block carries the explicit "raw" label either.
    assert "| raw" not in stages_block


def test_summary_without_curated_scores_csv_is_unchanged(tmp_path):
    """Safety guard: when curated_scores.csv is absent, the summary is
    identical to the pre-patch behaviour — the merge is strictly
    additive.
    """
    rundir = tmp_path / "no_curated"
    rundir.mkdir()
    _write_scores(rundir / "scores.csv", [
        ("per_cell", "sim1", "gt", "fiber_kit", "best_unit_accuracy_mean", 0.048),
        ("per_cell", "sim1", "gt", "kilosort4", "best_unit_accuracy_mean", 0.051),
    ])
    from ns3_sort_bench.cli.main import main
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["summary", str(rundir)])
    out = buf.getvalue()
    assert rc == 0
    # No stage sub-section anywhere.
    assert "## " not in out
    # No stray "@" adapter names in the main table.
    assert "@" not in out


def test_summary_org_no_stages_section_when_no_stage_rows(tmp_path):
    """Adapters with no `@<tag>` rows must not produce an empty
    `## <primary> stages` block.
    """
    rundir = tmp_path / "no_stages"
    rundir.mkdir()
    _write_scores(rundir / "scores.csv", [
        ("per_cell", "sim1", "gt", "fiber_kit",  "best_unit_accuracy_mean", 0.048),
        ("per_cell", "sim1", "gt", "kilosort4",  "best_unit_accuracy_mean", 0.051),
    ])
    from ns3_sort_bench.cli.main import main
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["summary", str(rundir)])
    out = buf.getvalue()
    assert rc == 0
    assert "## " not in out, "no stage sub-sections should appear"


def test_compare_delta_and_only_regressed(tmp_path):
    """compare: delta = B - A; --only-regressed drops delta>=0 rows."""
    a_dir = tmp_path / "A"
    b_dir = tmp_path / "B"
    a_dir.mkdir(); b_dir.mkdir()
    _write_scores(a_dir / "scores.csv", [
        ("per_cell", "sim1", "gt", "fiber_kit", "accuracy", 0.80),
        ("per_cell", "sim1", "gt", "botm_ref",   "accuracy", 0.70),
    ])
    _write_scores(b_dir / "scores.csv", [
        ("per_cell", "sim1", "gt", "fiber_kit", "accuracy", 0.85),  # +0.05
        ("per_cell", "sim1", "gt", "botm_ref",   "accuracy", 0.60),  # -0.10
    ])
    from ns3_sort_bench.cli.main import main

    # Full compare: both rows present, biggest |delta| first
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["compare", str(a_dir), str(b_dir)])
    assert rc == 0
    out = buf.getvalue()
    assert "+0.0500" in out and "-0.1000" in out
    # Biggest absolute delta on top → botm_ref before fiber_kit
    assert out.index("botm_ref") < out.index("fiber_kit")

    # Only-regressed → fiber_kit row (positive delta) is gone
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["compare", str(a_dir), str(b_dir), "--only-regressed"])
    assert rc == 0
    out = buf.getvalue()
    assert "botm_ref" in out
    assert "fiber_kit" not in out


def test_compare_handles_missing_sides(tmp_path):
    """A row present only in one run keeps NaN on the other side as `—`."""
    a_dir = tmp_path / "A"
    b_dir = tmp_path / "B"
    a_dir.mkdir(); b_dir.mkdir()
    _write_scores(a_dir / "scores.csv", [
        ("per_cell", "sim1", "gt", "fiber_kit", "accuracy", 0.9),
    ])
    _write_scores(b_dir / "scores.csv", [
        ("per_cell", "sim1", "gt", "fiber_kit", "f1",       0.8),
    ])
    from ns3_sort_bench.cli.main import main
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["compare", str(a_dir), str(b_dir)])
    assert rc == 0
    out = buf.getvalue()
    assert "—" in out   # at least one side missing
    assert "accuracy" in out and "f1" in out


def test_summary_missing_dir(tmp_path):
    from ns3_sort_bench.cli.main import main
    err = io.StringIO()
    with redirect_stderr(err):
        rc = main(["summary", str(tmp_path / "nope")])
    assert rc == 1
    assert "no scores.csv" in err.getvalue()
