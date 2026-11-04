"""Smoke tests for the fiber-kit adapter's Klusters-convention detect-and-write
step.  Does NOT invoke `fiber-session`; validates only the inputs that step
needs to see on disk.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ns3_sort_bench.adapters.fiber_kit import (
    detect_and_write_klusters,
    _bandpass,
    _detect_peak_indices,
    _extract_snippets,
    _write_res,
    _write_spk,
    _write_fet,
    _compute_features,
    _klusters_path,
    _NS_SAMPLES,
    _NS_PEAK,
    _NS3_METHOD,
    _RES_DTYPE,
    _SPK_DTYPE,
    _FET_DTYPE,
    _FET_HDR_DTYPE,
)


class _FakeRec:
    """Minimal duck-typed SpikeInterface BaseRecording."""
    def __init__(self, traces: np.ndarray, sr: float):
        self._tr = traces
        self._sr = float(sr)

    def get_num_samples(self, _seg=0):  return int(self._tr.shape[0])
    def get_num_channels(self):         return int(self._tr.shape[1])
    def get_sampling_frequency(self):   return self._sr
    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        s0 = 0 if start_frame is None else int(start_frame)
        s1 = int(self._tr.shape[0]) if end_frame is None else int(end_frame)
        out = self._tr[s0:s1]
        if channel_indices is not None:
            out = out[:, np.asarray(channel_indices, dtype=np.int64)]
        return out


def _synth_tetrode(n_samp: int = 20000 * 5, n_ch: int = 4, sr: float = 20000.0,
                   spike_hz: float = 50.0, peak_ampl_uv: float = 400.0,
                   noise_sigma_uv: float = 10.0, seed: int = 0,
                   spike_width_ms: float = 0.4,
                   refract_ms: float = 5.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (traces, true_peak_samples, sr): negative-going spikes injected at
    a Poisson rate into Gaussian white noise on every channel.  All channels
    see the same spike (shared on-tetrode unit).  The waveform is narrow
    (0.4 ms default) so it sits inside the 300-6000 Hz band naturally — a
    wider kernel induces filter ringing that would register as side-peaks."""
    rng = np.random.default_rng(seed)
    traces = (rng.standard_normal((n_samp, n_ch)) * noise_sigma_uv).astype(np.float32)

    # Poisson spike times with a strict refractory gap so adjacent spikes
    # don't fall inside the bandpass filter's ring-down from a predecessor.
    refract = int(sr * refract_ms * 1e-3)
    peaks = []
    t = refract
    while t < n_samp - refract:
        step = max(int(rng.exponential(sr / max(spike_hz, 1.0))), refract)
        t += step
        if t < n_samp - refract:
            peaks.append(t)
    peaks = np.asarray(peaks, dtype=np.int64)

    # Narrow negative-going Gaussian spike, half-width ~0.2 ms → ~2500 Hz lobe
    sw = max(int(sr * spike_width_ms * 1e-3), 1)
    tt = np.arange(-sw, sw + 1)
    kernel = (-peak_ampl_uv * np.exp(-(tt ** 2) / (2 * (sw / 2) ** 2))).astype(np.float32)
    for p in peaks:
        s0, s1 = p - sw, p + sw + 1
        if 0 <= s0 and s1 <= n_samp:
            traces[s0:s1, :] += kernel[:, None]
    return traces, peaks, sr


# ─── bandpass ───────────────────────────────────────────────────────────────
def test_bandpass_shape_and_finiteness():
    tr, _, sr = _synth_tetrode(n_samp=20000)
    out = _bandpass(tr, sr)
    assert out.shape == tr.shape
    assert np.isfinite(out).all()
    assert out.dtype == np.float64


# ─── detection ──────────────────────────────────────────────────────────────
def test_detect_recovers_most_true_spikes():
    tr, truth, sr = _synth_tetrode(n_samp=20000 * 5, spike_hz=30.0, seed=1)
    tr_f = _bandpass(tr, sr)
    peaks = _detect_peak_indices(tr_f, sr, threshold_sigma=5.0)
    # Match each detected peak to the nearest truth within ±1 ms
    tol = int(sr * 1e-3)
    truth_set = truth.astype(np.int64)
    matched = 0
    for p in peaks:
        if np.any(np.abs(truth_set - int(p)) <= tol):
            matched += 1
    # Recall: most true spikes recovered.  Precision: most detections are real.
    # Tolerant bars — this is a smoke test, not a precision benchmark.
    assert len(peaks) >= 0.7 * len(truth), (len(peaks), len(truth))
    assert matched >= 0.7 * len(peaks), (matched, len(peaks))


def test_detect_zero_on_pure_noise():
    rng = np.random.default_rng(0)
    tr = (rng.standard_normal((20000 * 2, 4)) * 10.0).astype(np.float32)
    tr_f = _bandpass(tr, 20000.0)
    peaks = _detect_peak_indices(tr_f, 20000.0, threshold_sigma=8.0)
    # 8σ on pure Gaussian noise → expected ~0 events for a few seconds
    assert len(peaks) < 20, len(peaks)


# ─── snippet extractor ──────────────────────────────────────────────────────
def test_extract_snippets_shape_and_boundary_drop():
    """With n_samples=32 and peak_sample=16, a peak at sample s is kept iff
    s >= 16  and  s + 16 <= n_total.  For n_total=1000: 16 <= s <= 984."""
    rng = np.random.default_rng(0)
    tr = rng.standard_normal((1000, 4)).astype(np.float64)
    peaks = np.array([0, 1, 15, 16, 500, 984, 985, 999], dtype=np.int64)
    kept, snips = _extract_snippets(tr, peaks)
    # Boundary rejects:
    for p in (0, 1, 15, 985, 999):
        assert p not in kept.tolist()
    # Boundary keeps:
    for p in (16, 500, 984):
        assert p in kept.tolist()
    assert snips.shape == (kept.size, _NS_SAMPLES, 4)
    assert snips.dtype == np.int16


# ─── byte-level format contracts (must match fiber_kit/neuro_io.py) ─────────
def test_dtypes_match_fiber_kit():
    """_RES_DTYPE and _SPK_DTYPE are the fiber-kit on-disk contract."""
    assert _RES_DTYPE == np.dtype("<i8")
    assert _SPK_DTYPE == np.dtype("<i2")


def test_write_res_is_binary_int64(tmp_path):
    """Produces exactly N*8 bytes binary, round-trippable through
    np.fromfile(dtype='<i8') — the exact call fiber-kit's read_res_file makes."""
    samples = np.array([5, 10_000_000, 42, 7], dtype=np.int64)
    path = str(tmp_path / "x.res.1")
    _write_res(path, samples)
    size = Path(path).stat().st_size
    assert size == samples.size * 8, (size, samples.size * 8)
    roundtrip = np.fromfile(path, dtype="<i8")
    assert roundtrip.tolist() == sorted(samples.tolist())


def test_write_spk_round_trips_through_fiber_kit_reshape(tmp_path):
    """fiber-kit's open_spk_file does:
        np.memmap(path, dtype='<i2').reshape(n, nsamp, nchan)
    so writing a C-order (n, nsamp, nchan) int16 array must round-trip
    through exactly that call."""
    rng = np.random.default_rng(0)
    snips = (rng.integers(-1000, 1000,
                           size=(7, _NS_SAMPLES, 4))).astype(np.int16)
    path = str(tmp_path / "x.spk.1")
    _write_spk(path, snips)
    assert Path(path).stat().st_size == 7 * _NS_SAMPLES * 4 * 2
    flat = np.fromfile(path, dtype="<i2")
    out = flat.reshape(7, _NS_SAMPLES, 4)
    assert np.array_equal(out, snips)


# ─── canonical ns3 naming contract ──────────────────────────────────────────
def test_klusters_path_follows_ns3_naming():
    """<base>.<type>.<method>.<group> per doc/ndmanager-plugins/formats/naming.md."""
    assert _klusters_path("/tmp/x", "res", 1) == "/tmp/x.res.standard.1"
    assert _klusters_path("/tmp/x", "spk", 7) == "/tmp/x.spk.standard.7"
    assert _klusters_path("/tmp/x", "res", 1, method="stderiv") == \
           "/tmp/x.res.stderiv.1"
    assert _NS3_METHOD == "standard"


# ─── ndm_* delegation ────────────────────────────────────────────────────────
# `detect_and_write_klusters` now shells out to neurosuite-3's `ndm_detectspikes`,
# `ndm_extractspikes`, `ndm_pca`.  CI can't install ns3, so these tests mock
# subprocess.run and assert the commands are invoked with the right arguments.
import subprocess as _subprocess

from ns3_sort_bench.adapters.fiber_kit import (
    detect_and_write_klusters, _programs_yaml_section,
)


class _RecordingRuns:
    """subprocess.run replacement that records invocations and simulates
    the side effect each `ndm_*` tool would have (writing its output file).
    """
    def __init__(self, base, group, n_spikes=50, n_samples=32, n_ch=4):
        self.calls = []
        self.base = base
        self.group = group
        self.n_spikes = n_spikes
        self.n_samples = n_samples
        self.n_ch = n_ch

    def __call__(self, cmd, *args, **kw):
        self.calls.append(list(cmd))
        name = cmd[0]
        g = self.group
        if name == "ndm_detectspikes":
            Path(f"{self.base}.res.standard.{g}").write_bytes(
                np.arange(self.n_spikes, dtype="<i8").tobytes()
            )
        elif name == "ndm_extractspikes":
            Path(f"{self.base}.spk.standard.{g}").write_bytes(
                b"\x00" * (self.n_spikes * self.n_samples * self.n_ch * 2)
            )
        elif name == "ndm_pca":
            Path(f"{self.base}.fet.standard.{g}").write_bytes(
                b"\x00" * (4 + self.n_spikes * 4 * 8)
            )
            Path(f"{self.base}.pca.standard.{g}").write_bytes(b"\x00" * 36)
        rc = _subprocess.CompletedProcess(cmd, 0, b"", b"")
        return rc


def _make_base_with_yaml(tmp_path) -> str:
    """Write the minimal <base>.yaml a mocked ndm_detectspikes would see."""
    base = str(tmp_path / "session")
    Path(base + ".yaml").write_text("acquisitionSystem: {}\n")
    Path(base + ".fil").write_bytes(b"")
    return base


def test_detect_and_write_klusters_runs_the_three_ndm_tools(tmp_path, monkeypatch):
    base = _make_base_with_yaml(tmp_path)
    runner = _RecordingRuns(base, group=1, n_spikes=42)
    monkeypatch.setattr(_subprocess, "run", runner)

    n = detect_and_write_klusters(
        _FakeRec(np.zeros((1000, 4), dtype=np.int16), 20000.0),
        base=base, group=1,
    )

    # Three ndm_* subprocesses were invoked, in pipeline order.
    assert [c[0] for c in runner.calls] == [
        "ndm_detectspikes", "ndm_extractspikes", "ndm_pca",
    ]
    for c in runner.calls:
        assert c[1] == base + ".yaml"
        assert "--method" in c and "standard" in c
        assert "--force" in c

    # The .res byte count determines the returned spike count.
    assert n == 42


def test_shared_detection_skips_detectspikes(tmp_path, monkeypatch):
    """shared_res=<GT samples> writes the .res directly and skips
    ndm_detectspikes; the rest of the chain still runs."""
    base = _make_base_with_yaml(tmp_path)
    gt = np.array([100, 200, 300, 400, 500], dtype=np.int64)
    runner = _RecordingRuns(base, group=2, n_spikes=gt.size)
    monkeypatch.setattr(_subprocess, "run", runner)

    n = detect_and_write_klusters(
        _FakeRec(np.zeros((10000, 4), dtype=np.int16), 20000.0),
        base=base, group=2, shared_res=gt,
    )

    assert [c[0] for c in runner.calls] == ["ndm_extractspikes", "ndm_pca"]
    assert n == gt.size
    # .res holds exactly the GT samples, sorted, binary int64 LE
    on_disk = np.fromfile(f"{base}.res.standard.2", dtype="<i8")
    assert on_disk.tolist() == sorted(gt.tolist())


def test_detect_and_write_klusters_raises_on_ndm_failure(tmp_path, monkeypatch):
    base = _make_base_with_yaml(tmp_path)

    def _failing_run(cmd, *args, **kw):
        if cmd[0] == "ndm_extractspikes":
            Path(kw.get("stdout").name if hasattr(kw.get("stdout"), "name")
                 else "/dev/null")
            return _subprocess.CompletedProcess(cmd, 2, b"", b"")
        # detectspikes succeeds, writes a .res
        Path(f"{base}.res.standard.1").write_bytes(
            np.arange(5, dtype="<i8").tobytes()
        )
        return _subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(_subprocess, "run", _failing_run)
    with pytest.raises(RuntimeError, match="ndm_extractspikes failed"):
        detect_and_write_klusters(
            _FakeRec(np.zeros((1000, 4), dtype=np.int16), 20000.0),
            base=base, group=1,
        )


# ─── programs-yaml section helper ───────────────────────────────────────────
def test_programs_yaml_section_minimum_set():
    """The yaml's `programs:` block must carry all three ndm_* steps and
    their mandatory parameters so `ndm_detectspikes <base>.yaml` works
    without further configuration."""
    programs = _programs_yaml_section()
    steps = {p["name"]: p for p in programs}
    assert set(steps) == {"ndm_detectspikes", "ndm_extractspikes", "ndm_pca"}
    detect_params = {p["name"] for p in steps["ndm_detectspikes"]["parameters"]}
    # Every ndm_detectspikes mandatory parameter is present
    assert {"thresholdFactor", "refractoryPeriod", "peakSearchLength",
            "start", "duration", "method"} <= detect_params


def test_programs_yaml_section_overrides_merge():
    programs = _programs_yaml_section(overrides={
        "ndm_detectspikes": {"thresholdFactor": "0.5", "start": "2"},
    })
    detect = next(p for p in programs if p["name"] == "ndm_detectspikes")
    params = {p["name"]: p["value"] for p in detect["parameters"]}
    assert params["thresholdFactor"] == "0.5"
    assert params["start"] == "2"
    # Non-overridden mandatory parameters survive
    assert params["method"] == "standard"


# ─── .fet writer / feature computation ──────────────────────────────────────
def test_compute_features_shape_and_dtype():
    """`_compute_features` returns (n_spikes, n_pcs * n_ch + 1) int64."""
    rng = np.random.default_rng(0)
    snippets = rng.integers(-1000, 1000,
                             size=(50, _NS_SAMPLES, 4)).astype(np.int16)
    peaks = np.arange(50, dtype=np.int64) * 100
    feats = _compute_features(snippets, peaks, n_pcs_per_channel=3)
    assert feats.shape == (50, 3 * 4 + 1)
    assert feats.dtype == _FET_DTYPE
    # Last column is the time feature, scaled to int
    assert (feats[:, -1] == (peaks * 10_000).astype(_FET_DTYPE)).all()


def test_compute_features_empty_input():
    empty_snips = np.zeros((0, _NS_SAMPLES, 4), dtype=np.int16)
    empty_peaks = np.zeros(0, dtype=np.int64)
    feats = _compute_features(empty_snips, empty_peaks, n_pcs_per_channel=3)
    assert feats.shape == (0, 3 * 4 + 1)


def test_write_fet_byte_layout(tmp_path):
    """fiber-kit's `read_fet_file`: int32 header(nFeatures) + int64 body."""
    feats = np.array([[1, 2, 3, 100],
                      [4, 5, 6, 200]], dtype=_FET_DTYPE)
    path = str(tmp_path / "x.fet.standard.1")
    _write_fet(path, feats)
    # Expected bytes = 4 (header) + 2*4*8 = 68
    assert Path(path).stat().st_size == 4 + feats.size * 8
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype="<i4", count=1)
        body = np.fromfile(f, dtype="<i8")
    assert int(hdr[0]) == 4
    assert body.reshape(2, 4).tolist() == feats.tolist()


# (.fet.standard.<g> is now produced by ndm_pca, exercised by
# test_detect_and_write_klusters_runs_the_three_ndm_tools above.  The zero-
# spike path is similarly ndm_detectspikes' responsibility now.)


# ─── materialise .fil + .dat for Klusters ───────────────────────────────────
def test_materialise_recording_writes_dat_hardlinked_to_fil(tmp_path):
    """Klusters' TracesProvider opens <base>.dat (hardcoded, not .fil).
    `_materialise_recording` must leave both next to the yaml with
    identical bytes."""
    from ns3_sort_bench.adapters.fiber_kit import _FiberKitBase
    adapter = _FiberKitBase.__new__(_FiberKitBase)
    rec = _FakeRec(np.zeros((2000, 4), dtype=np.int16), 20000.0)
    base = adapter._materialise_recording(rec, str(tmp_path))
    fil = Path(base + ".fil")
    dat = Path(base + ".dat")
    yaml_p = Path(base + ".yaml")
    assert fil.exists() and dat.exists() and yaml_p.exists()
    # Same bytes — hardlink shares inode; even if we fell back to a copy
    # the content must still match.
    assert fil.stat().st_size == dat.stat().st_size
    assert fil.read_bytes() == dat.read_bytes()
    # And the file size divides evenly by (n_channels × 2 bytes) — the exact
    # arithmetic Klusters' TracesProvider uses to compute recording length.
    assert fil.stat().st_size % (4 * 2) == 0


def test_materialise_recording_bandpasses_before_writing_fil(tmp_path):
    """`.fil` must be the 300-6000 Hz bandpass of the input, NOT the raw
    SpikeInterface traces.  Matches ndm_hipass → ndm_detectspikes flow:
    the detector reads whatever is in .fil and assumes it is already
    hipassed, so the adapter has to bake that in.

    Regression for the paired-recordings cluster collapse: without this
    filter, ndm_detectspikes threshold-crosses on wideband including LFP,
    inflates detection count, and drowns fiber-session's clustering.
    """
    from ns3_sort_bench.adapters.fiber_kit import _FiberKitBase
    adapter = _FiberKitBase.__new__(_FiberKitBase)
    sr = 20000.0
    n_samp = 20_000                                  # 1 s
    rng = np.random.default_rng(0)
    # Pure low-frequency sine (10 Hz) deep in the stop band — a correct
    # 300 Hz hipass should knock it to near-zero.
    t = np.arange(n_samp) / sr
    lf = (3000 * np.sin(2 * np.pi * 10 * t)).astype(np.int16)
    traces = np.tile(lf[:, None], (1, 4))
    rec = _FakeRec(traces, sr)
    base = adapter._materialise_recording(rec, str(tmp_path))

    filt = np.fromfile(base + ".fil", dtype=np.int16).reshape(-1, 4)
    assert filt.shape == (n_samp, 4)
    # After a 300 Hz hipass, a 10 Hz sine of amplitude 3000 collapses to
    # a handful of counts (tolerance for filter transient edges).  Raw
    # (no filter) would stay near +/- 3000.
    assert np.max(np.abs(filt)) < 500, \
        f".fil peak = {np.max(np.abs(filt))}; bandpass did not run"


def test_materialise_recording_programs_block_follows_sample_rate(tmp_path):
    """refractoryPeriod and peakSearchLength in the programs block must
    track the session's sampling rate — a 10-sample refractory at 20 kHz
    (= 0.5 ms, Python-port default) and 16 samples at 32 kHz.
    """
    import yaml
    from ns3_sort_bench.adapters.fiber_kit import _FiberKitBase, _NS_SAMPLES
    adapter = _FiberKitBase.__new__(_FiberKitBase)

    def _read_detect_params(sr):
        d = tmp_path / f"sr{int(sr)}"
        d.mkdir()
        rec = _FakeRec(np.zeros((int(sr * 2), 4), dtype=np.int16), float(sr))
        base = adapter._materialise_recording(rec, str(d))
        doc = yaml.safe_load(Path(base + ".yaml").read_text())
        progs = {p["name"]: {kv["name"]: kv["value"] for kv in p["parameters"]}
                 for p in doc["programs"]}
        return progs["ndm_detectspikes"]

    p20 = _read_detect_params(20_000.0)
    assert p20["refractoryPeriod"] == str(max(1, int(round(0.5e-3 * 20_000))))
    assert p20["peakSearchLength"] == str(_NS_SAMPLES)

    p32 = _read_detect_params(32_000.0)
    assert p32["refractoryPeriod"] == str(max(1, int(round(0.5e-3 * 32_000))))

    # The yaml-level baseline also stays sane (thresholdFactor, start,
    # method are NOT rewritten and keep the configs/fiber_kit.yaml value).
    assert p20["thresholdFactor"] == "1.25"
    assert p20["method"]          == "standard"


def test_fiber_kit_config_sets_pca_to_python_port_shape():
    """`configs/fiber_kit.yaml` must disable varimax + extras for ndm_pca so
    the feature space matches the previous Python port's _compute_features
    (plain per-channel PCA, 3 components, no peak-value extras, no rotation).
    fiber-session's clustering has been tuned against that shape.
    """
    from ns3_sort_bench.configs import load_adapter_config
    cfg = load_adapter_config("fiber_kit")
    pca = cfg["ndm_pca"]
    assert pca["extra"]   == "false", \
        "ndm_pca must not add peak-value extras (Python port had none)"
    assert pca["varimax"] == "false", \
        "ndm_pca must not apply varimax (Python port used plain PCA)"


# ─── fiber-kit.yaml setup ───────────────────────────────────────────────────
def test_write_fk_config_pins_realign_off(tmp_path):
    """The auto-written fiber-kit.yaml must set FK_SESSION_REALIGN=off, with
    `off` quoted so YAML 1.1 doesn't coerce it to a boolean."""
    import os
    # Scrub $FK_CONFIG so the merge sees only the baseline (not whatever
    # the test environment has pointed at).
    orig = os.environ.pop("FK_CONFIG", None)
    try:
        from ns3_sort_bench.adapters.fiber_kit import _FiberKitBase
        # cwd a clean dir so ./fiber-kit.yaml doesn't exist either
        cwd_before = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            path = _FiberKitBase._write_fk_config(str(tmp_path))
        finally:
            os.chdir(cwd_before)
    finally:
        if orig is not None:
            os.environ["FK_CONFIG"] = orig
    assert Path(path).name == "fiber-kit.yaml"
    body = Path(path).read_text()
    assert 'FK_SESSION_REALIGN: "off"' in body
    # Quoted — a bare `off` would YAML-coerce to False
    assert "FK_SESSION_REALIGN: off\n" not in body
    # Round-trip under PyYAML to confirm the string survives
    import yaml
    parsed = yaml.safe_load(body)
    assert parsed["FK_SESSION_REALIGN"] == "off"


def test_write_fk_config_preserves_user_tuning(tmp_path):
    """When the user has their own ./fiber-kit.yaml (or passes
    fk_config_path), those keys MUST win in the merged per-session yaml.

    Regression for the 349966b setup: before this fix, our minimal baseline
    was the only thing written and FK_CONFIG pointed at it, so a user
    ./fiber-kit.yaml holding FK_XCM_*, FK_CHUNKS, ... was silently
    discarded — likely cause of fiber-session's 33 → 1 cluster collapse
    on PAIRED_CRCNS_HC1.
    """
    import os, yaml
    user_cfg = tmp_path / "user_tuning.yaml"
    user_cfg.write_text(
        "FK_XCM_WINDOW: 42\n"
        "FK_CHUNKS: 8\n"
        "FK_SESSION_REALIGN: \"on\"\n"      # the user overrides baseline
    )
    out_dir = tmp_path / "session"
    out_dir.mkdir()

    orig = os.environ.pop("FK_CONFIG", None)
    try:
        from ns3_sort_bench.adapters.fiber_kit import _FiberKitBase
        path = _FiberKitBase._write_fk_config(str(out_dir),
                                                 user_cfg_path=str(user_cfg))
    finally:
        if orig is not None:
            os.environ["FK_CONFIG"] = orig

    parsed = yaml.safe_load(Path(path).read_text())
    assert parsed["FK_XCM_WINDOW"] == 42
    assert parsed["FK_CHUNKS"]     == 8
    assert parsed["FK_SESSION_REALIGN"] == "on"       # user's value wins


def test_write_fk_config_falls_back_to_cwd_fiber_kit_yaml(tmp_path, monkeypatch):
    """When no explicit `fk_config_path` is given and `$FK_CONFIG` is unset,
    the merge pulls from ``./fiber-kit.yaml`` in CWD (matching fiber-kit's
    own `load_global_config` search order)."""
    import os, yaml
    cwd_yaml = tmp_path / "fiber-kit.yaml"
    cwd_yaml.write_text("FK_RDR_MIN_AMP: 3.14\n")
    monkeypatch.delenv("FK_CONFIG", raising=False)
    monkeypatch.chdir(str(tmp_path))
    out_dir = tmp_path / "session"
    out_dir.mkdir()

    from ns3_sort_bench.adapters.fiber_kit import _FiberKitBase
    path = _FiberKitBase._write_fk_config(str(out_dir))
    parsed = yaml.safe_load(Path(path).read_text())
    assert parsed["FK_RDR_MIN_AMP"] == 3.14
    # Baseline still present for keys the user didn't set
    assert parsed["FK_SESSION_REALIGN"] == "off"


def test_write_fk_config_tolerates_broken_user_yaml(tmp_path):
    """A malformed user yaml must NOT break the sort — fall back to the
    baseline alone and leave a breadcrumb in the written file."""
    import os
    user_cfg = tmp_path / "bad.yaml"
    user_cfg.write_text("this: is: not: yaml\n")          # invalid
    out_dir = tmp_path / "session"
    out_dir.mkdir()

    orig = os.environ.pop("FK_CONFIG", None)
    try:
        from ns3_sort_bench.adapters.fiber_kit import _FiberKitBase
        path = _FiberKitBase._write_fk_config(str(out_dir),
                                                 user_cfg_path=str(user_cfg))
    finally:
        if orig is not None:
            os.environ["FK_CONFIG"] = orig

    body = Path(path).read_text()
    assert 'FK_SESSION_REALIGN: "off"' in body            # baseline survived
    assert "WARNING" in body                               # breadcrumb for user
