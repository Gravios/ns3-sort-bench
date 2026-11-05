"""``ns3sb fk-sweep``: settings, tuning files, resume, and the table."""
from __future__ import annotations

import csv
import io
from contextlib import redirect_stdout, redirect_stderr

import yaml

from ns3_sort_bench.cli.commands import fk_sweep_cmd as fs


def test_parse_set_reads_yaml_values_and_keeps_bool_words():
    assert fs.parse_set("FK_X=0.15,3, off,") == ("FK_X", [0.15, 3, "off", ""])


def test_one_at_a_time_and_grid():
    sweep = {"FK_A": [1, 2], "FK_B": ["x"]}
    one = fs.settings(sweep, grid=False)
    assert [l for l, _ in one] == ["baseline", "A=1", "A=2", "B=x"]
    grid = fs.settings(sweep, grid=True)
    assert [o for _, o in grid] == [{}, {"FK_A": 1, "FK_B": "x"},
                                    {"FK_A": 2, "FK_B": "x"}]


def test_tuning_text_merges_base_under_setting():
    t = yaml.safe_load(fs.tuning_text({"FK_A": 1, "FK_C": "off"}, {"FK_A": 9}))
    assert t == {"FK_A": 9, "FK_C": "off"}


def _fake_scores(path, n_fs, rec_fs, n_fin, rec_fin):
    rows = []
    for adapter, n, rec in (("fiber_kit@fiber_session", n_fs, rec_fs),
                            ("fiber_kit", n_fin, rec_fin)):
        for metric, v in (("n_cand_units", n), ("best_unit_recall_mean", rec),
                          ("best_unit_accuracy_mean", rec / 2),
                          ("best_unit_precision_mean", 0.5)):
            rows.append(dict(scorer="per_cell", corpus_item="c",
                             reference_adapter="gt", candidate_adapter=adapter,
                             metric=metric, value=v))
    rows.append(dict(scorer="pairwise", corpus_item="c", reference_adapter="gt",
                     candidate_adapter="fiber_kit", metric="n_cand_units",
                     value=999))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def test_run_tabulates_and_resumes(tmp_path, monkeypatch):
    monkeypatch.delenv("FK_CONFIG", raising=False)
    calls = []

    def fake_run(corpus, run_dir, tuning_path):
        calls.append(run_dir)
        tuning = yaml.safe_load(open(tuning_path)) or {}
        n = 4 if tuning.get("FK_SESSION_DIPSPLIT") == 0 else 16
        _fake_scores(f"{run_dir}/scores.csv", n, 0.6, n - 2, 0.7)
        return 0
    monkeypatch.setattr(fs, "_run_one", fake_run)

    from ns3_sort_bench.cli.main import main
    argv = ["fk-sweep", "--corpus", "c", "--out", str(tmp_path / "sw"),
            "--set", "FK_SESSION_DIPSPLIT=0"]
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        assert main(argv) == 0
    assert len(calls) == 2
    text = out.getvalue()
    assert "SESSION_DIPSPLIT=0" in text and "baseline" in text
    rows = list(csv.reader(open(tmp_path / "sw" / "sweep.csv")))
    by = {r[0]: r for r in rows[1:]}
    assert by["SESSION_DIPSPLIT=0"][1] == "4"     # fiber_session n
    assert by["baseline"][1] == "16"
    assert by["baseline"][5] == "14"              # final n (pairwise row ignored)

    # Second invocation: nothing re-runs.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        assert main(argv) == 0
    assert len(calls) == 2
    # Changed base tuning → the tuning file differs → runs again.
    base = tmp_path / "base.yaml"; base.write_text("FK_CHUNKS: 7\n")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        assert main(argv + ["--base-config", str(base)]) == 0
    assert len(calls) == 4


def test_failed_run_reported(tmp_path, monkeypatch):
    monkeypatch.delenv("FK_CONFIG", raising=False)
    monkeypatch.setattr(fs, "_run_one", lambda *a: 1)
    from ns3_sort_bench.cli.main import main
    err = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(err):
        rc = main(["fk-sweep", "--corpus", "c", "--out", str(tmp_path / "s"),
                   "--set", "FK_A=1"])
    assert rc == 1
    assert "run.log" in err.getvalue()
