"""fiber_kit on a lab session: the group is sorted in a scratch copy of its
files, in the session's own method, and read back from there; nothing is
written beside the lab session."""
from __future__ import annotations

import os

import numpy as np
import pytest
import yaml

from ns3_sort_bench.adapters import fiber_kit as fk

G, NS, NC, N, SR = 8, 4, 2, 50, 20000.0


def _make_session(d, groups=(7, 8)):
    """A lab session folder: yaml (probe named relative to it), .fil/.dat, and
    per-group files in three waveform spaces, plus things a sort must never see."""
    os.makedirs(d / "probes", exist_ok=True)
    (d / "probes" / "buz.probe").write_text("probe\n")
    chans = {7: [0, 1], 8: [2, 3]}
    y = {"acquisitionSystem": {"samplingRate": SR, "nChannels": 4, "nBits": 16},
         "probeFile": "probes/buz.probe",
         "spikeDetection": {"channelGroups": [
             {"channels": chans[g], "nSamples": NS, "peakSampleIndex": 2, "nFeatures": 3}
             for g in sorted(chans)]}}
    # group g is the g-th channel group in the session; pad so index g-1 exists
    y["spikeDetection"]["channelGroups"] = (
        [{"channels": [0], "nSamples": NS, "peakSampleIndex": 2, "nFeatures": 3}] * 6
        + y["spikeDetection"]["channelGroups"])
    b = str(d / "rat")
    with open(b + ".yaml", "w") as f:
        yaml.safe_dump(y, f)
    np.zeros(1000 * 4, "<i2").tofile(b + ".fil")
    os.link(b + ".fil", b + ".dat")
    for g in groups:
        for v in ("standard", "stderiv_C5", "stderiv_C5_U4"):
            np.zeros((N, NS, NC), "<i2").tofile(f"{b}.spk.{v}.{g}")
        np.arange(N, dtype="<i8").tofile(f"{b}.res.stderiv_S3.{g}")
        np.zeros(10, "<i8").tofile(f"{b}.fet.stderiv_C5_D34.{g}")
        np.zeros(10, "<f8").tofile(f"{b}.pca.standard.{g}")
        # earlier sorts / curations / collision tables: never copied
        np.r_[2, np.ones(N)].astype("<i4").tofile(f"{b}.clu.stderiv_C5.{g}.cur_pl")
        np.r_[2, np.ones(N)].astype("<i4").tofile(f"{b}.clu.stderiv_C5.{g}.fc_session")
        open(f"{b}.eap.{g}", "wb").write(b"EAP\x01")
    return b


def _listing(d):
    return sorted(os.path.relpath(os.path.join(r, f), d) for r, _, fs in os.walk(d) for f in fs)


def test_stage_lab_session_copies_the_group_and_links_the_recording(tmp_path):
    lab = tmp_path / "lab"
    b = _make_session(lab)
    before = {p: os.path.getmtime(lab / p) for p in _listing(lab)}
    base, placed = fk.stage_lab_session(b, G, str(tmp_path / "work"))
    work = tmp_path / "work"
    assert base == str(work / "rat")
    assert sorted(placed) == sorted([
        "rat.yaml", "rat.fil", "rat.dat", "probes/buz.probe",
        "rat.spk.standard.8", "rat.spk.stderiv_C5.8",
        "rat.res.stderiv_S3.8", "rat.fet.stderiv_C5_D34.8", "rat.pca.standard.8"])
    # recording and probe linked, group files copied
    for name in ("rat.fil", "rat.dat", "probes/buz.probe"):
        assert os.path.islink(work / name)
    for name in ("rat.yaml", "rat.spk.stderiv_C5.8", "rat.res.stderiv_S3.8"):
        assert os.path.isfile(work / name) and not os.path.islink(work / name)
    # never: another group, a stage-tagged file, a super-sampled space, collision tables
    assert not any(n.endswith(".7") or ".cur_pl" in n or "fc_session" in n or "_U4" in n
                   or ".eap." in n for n in placed)
    # the lab folder is as it was
    assert {p: os.path.getmtime(lab / p) for p in _listing(lab)} == before



@pytest.mark.parametrize("spks, expected", [
    (["standard", "stderiv_C5", "stderiv_C5_U4"], "stderiv_C5"),
    (["standard", "stderiv"], "stderiv"),
    (["standard"], "standard"),
    (["standard", "stderiv_C5", "stderiv_C6"], ValueError),
    (["stderiv_C5_D34"], ValueError),
])
def test_lab_session_method_reads_the_spk(tmp_path, spks, expected):
    b = str(tmp_path / "s")
    for v in spks:
        open(f"{b}.spk.{v}.{G}", "wb").close()
    if expected is ValueError:
        with pytest.raises(ValueError, match="fk_method="):
            fk.lab_session_method(b, G)
    else:
        assert fk.lab_session_method(b, G) == expected


def test_lab_sort_runs_in_the_scratch_copy_and_reads_back_there(tmp_path, monkeypatch):
    """The adapter, with fiber-pipeline replaced by a stand-in that writes the
    plan's outputs: the group, the method and the session it is given, and
    the readback, all come from the scratch copy."""
    from spikeinterface.core import NumpyRecording
    lab = tmp_path / "lab"
    b = _make_session(lab)
    before = _listing(lab)

    calls = []

    def fake_run_stage(self, stage, sess_base, group, *, env, extra_args, method):
        calls.append(dict(stage=stage, sess_base=sess_base, group=group, method=method,
                          fk_config=env.get("FK_CONFIG")))
        clu = np.r_[3, np.where(np.arange(N) % 2, 2, 3), ][: N + 1].astype("<i4")
        for tag in ("fc_session", "fc_polish", "fc_clean", "fc_xcm", "fc_unmix"):
            clu.tofile(f"{sess_base}.clu.{method}.{group}.{tag}")
        # no .res of its own: the readback pairs the .clu with the session's
        # shared .res.stderiv_S3.<g>, copied into the scratch session

    monkeypatch.setattr(fk._FiberKitBase, "_run_stage", fake_run_stage)
    monkeypatch.setattr(fk._Full, "_available", True, raising=False)
    rec = NumpyRecording([np.zeros((1000, 2), np.float32)], sampling_frequency=SR)
    rec.annotate(session_base=b, electrode_group=G)
    keep = tmp_path / "keep"
    res = fk._Full().sort(rec, fk_artifacts_dir=str(keep), corpus_name="rat_g8")

    assert len(calls) == 1
    c = calls[0]
    assert c["group"] == G and c["method"] == "stderiv_C5"
    assert os.path.dirname(c["sess_base"]) == str(keep / "rat_g8")
    assert c["fk_config"] and os.path.dirname(c["fk_config"]) == str(keep / "rat_g8")
    assert res.extras["variant"] == "stderiv_C5" and res.extras["tag"] == "fc_unmix"
    assert res.extras["lab_session"] == b
    assert sorted(res.unit_ids) == [2, 3] and res.n_spikes() == N
    assert set(res.extras["stage_results"]) == {"fc_session", "fc_polish", "fc_clean", "fc_xcm"}
    # nothing was written beside the lab session
    assert _listing(lab) == before


def test_relative_artifacts_dir_still_runs_the_embedded_plan(tmp_path, monkeypatch):
    """--out runs/x --keep-fk-artifacts gives a RELATIVE fk_artifacts_dir.
    fiber-pipeline changes into the session folder before it opens
    $FK_CONFIG, so a relative path there found no config and the runner fell
    back to its built-in default chain (fiber-session -> refine -> ...)."""
    import shutil
    if shutil.which("fiber-pipeline") is None:
        pytest.skip("fiber-pipeline not on PATH")
    from spikeinterface.core import NumpyRecording
    lab = tmp_path / "lab"
    b = _make_session(lab)
    monkeypatch.chdir(tmp_path)
    rec = NumpyRecording([np.zeros((1000, 2), np.float32)], sampling_frequency=SR)
    rec.annotate(session_base=b, electrode_group=G)
    monkeypatch.setattr(fk._Full, "_available", True, raising=False)
    with pytest.raises(FileNotFoundError):      # a dry run writes no fc_unmix to read back
        fk._Full().sort(rec, fk_artifacts_dir="runs/x/fiber_kit", corpus_name="rat_g8",
                        extra_args=["--dry-run"])
    log = (tmp_path / "runs/x/fiber_kit/rat_g8/rat.fiber-pipeline.log").read_text()
    assert "running the primary plan embedded in /" in log
    assert "fiber-flag-artifacts" in log and "--clu-stage fc_session" in log
    assert "fiber-refine" not in log
