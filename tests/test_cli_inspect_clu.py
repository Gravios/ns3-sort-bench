"""Tests for `ns3sb inspect-clu` — the diagnostic that surfaces raw
per-cluster spike counts and confusion matrices for a klusters session.
"""
from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import numpy as np


def _mkdir(path):
    os.makedirs(path, exist_ok=True)


def _write_res(path, t):
    np.asarray(t, "<i8").tofile(path)


def _write_clu(path, c):
    arr = np.asarray(c, "<i4")
    with open(path, "wb") as f:
        np.array([int(np.unique(arr).size)], "<i4").tofile(f)
        arr.tofile(f)


def _build_run_dir(tmp_path):
    """Build a tiny klusters run dir with a sorter whose curated .clu
    has moved spikes into reserved IDs (1), mimicking Klusters' usual
    'renumber on save' behaviour."""
    sorter = tmp_path / "klusters_fiber_kit" / "item1"
    gt     = tmp_path / "klusters_ground_truth" / "item1"
    _mkdir(sorter); _mkdir(gt)

    # 10 sorter spikes, 3 clusters (2,3,4) — all non-reserved.
    _write_res(sorter / "sim_session.res.standard.1",
               [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    _write_clu(sorter / "sim_session.clu.standard.1",
               [2, 2, 3, 3, 4, 4, 2, 3, 4, 2])
    # Curated: Klusters renumbered to cluster ids (1, 2) — 1 is MUA bin,
    # dropped from scoring.
    _write_clu(sorter / "sim_session.clu.standard.1.curated",
               [2, 2, 1, 1, 1, 1, 2, 1, 1, 2])
    # GT: 3 spikes.
    _write_res(gt / "sim_session.res.standard.1", [102, 405, 902])
    _write_clu(gt / "sim_session.clu.standard.1", [2, 2, 2])
    return tmp_path


def _run_inspect(tmp_path):
    from ns3_sort_bench.cli.main import main
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["inspect-clu", str(tmp_path),
                    "--item", "item1", "--sorter", "fiber_kit"])
    return rc, buf.getvalue()


def test_inspect_clu_reports_counts_and_flags_reserved(tmp_path):
    """The command prints a per-cluster count table for the raw sort,
    the curated .clu, and the GT — and flags cluster IDs < 2 as
    RESERVED (score-curated drops them from scoring).
    """
    _build_run_dir(tmp_path)
    rc, out = _run_inspect(tmp_path)
    assert rc == 0, out

    # Section headers for each source.
    assert "raw fiber_kit"  in out
    assert "curated fiber_kit" in out
    assert "ground_truth" in out
    # The curator's cluster 1 (reserved) MUST be flagged.
    assert "RESERVED" in out
    # And the summary line reports "N would be dropped".
    assert "would be dropped by score-curated's reserved filter" in out


def test_inspect_clu_surfaces_confusion_matrix_vs_gt(tmp_path):
    """The raw × GT and curated × GT confusion matrices are printed as
    tables with row sums and column sums.  The curated × GT table must
    include the reserved row (annotated), because dropping it silently
    is the exact bug the command exists to diagnose.
    """
    _build_run_dir(tmp_path)
    rc, out = _run_inspect(tmp_path)
    assert rc == 0, out

    # Both confusion matrices are rendered.
    assert "raw fiber_kit × ground_truth"    in out
    assert "curated fiber_kit × ground_truth" in out
    # Curated × GT table has a row for the reserved cluster 1, labelled.
    curated_block = out.split("curated fiber_kit × ground_truth")[1]
    assert "RESERVED" in curated_block
    # And the raw × curated self-join is there.
    assert "raw fiber_kit × curated fiber_kit" in out


def test_inspect_clu_errors_on_missing_sorter(tmp_path):
    """Pointing at a sorter that has no klusters_<sorter>/<item>/ dir
    must error clearly rather than crashing."""
    _build_run_dir(tmp_path)
    from ns3_sort_bench.cli.main import main
    import io
    from contextlib import redirect_stderr
    err = io.StringIO()
    with redirect_stderr(err):
        rc = main(["inspect-clu", str(tmp_path),
                    "--item", "item1",
                    "--sorter", "no_such_sorter"])
    assert rc == 1
    assert "missing session dir" in err.getvalue()
