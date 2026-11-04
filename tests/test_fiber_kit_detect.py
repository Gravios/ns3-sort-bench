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


# ─── end-to-end .res + .spk writer ──────────────────────────────────────────
def test_detect_and_write_klusters_files_and_shapes(tmp_path):
    tr, truth, sr = _synth_tetrode(n_samp=20000 * 3, spike_hz=40.0, seed=2)
    rec = _FakeRec(tr, sr)
    base = str(tmp_path / "session")
    n_detected = detect_and_write_klusters(rec, base=base, group=1)

    # Canonical paths — NOT the legacy untagged `.res.1` / `.spk.1`.
    res_path = Path(f"{base}.res.standard.1")
    spk_path = Path(f"{base}.spk.standard.1")
    assert res_path.exists() and spk_path.exists()
    # And the legacy untagged names must NOT be written (they would mask the
    # canonical ones under fiber-kit's Shared-type resolver).
    assert not Path(f"{base}.res.1").exists()
    assert not Path(f"{base}.spk.1").exists()

    # .res: binary int64 LE, ascending, 0-indexed, exact byte count
    assert res_path.stat().st_size == n_detected * 8
    samples = np.fromfile(res_path, dtype="<i8")
    assert samples.size == n_detected
    assert np.all(np.diff(samples) >= 0)
    assert np.all((samples >= 0) & (samples < tr.shape[0]))

    # .spk: int16 LE, size exactly nspikes * 32 * n_ch * 2 bytes
    expected_bytes = n_detected * _NS_SAMPLES * tr.shape[1] * 2
    assert spk_path.stat().st_size == expected_bytes


def test_res_and_spk_counts_agree(tmp_path):
    """fiber-session's `check_spk_count` asserts the two writers agree.
    This is the exact regression that broke the paired-recordings sweep."""
    tr, _, sr = _synth_tetrode(n_samp=20000 * 4, spike_hz=35.0, seed=4)
    rec = _FakeRec(tr, sr)
    base = str(tmp_path / "session")
    n = detect_and_write_klusters(rec, base=base, group=3)
    n_from_res = Path(f"{base}.res.standard.3").stat().st_size // 8
    n_from_spk = Path(f"{base}.spk.standard.3").stat().st_size // (_NS_SAMPLES * tr.shape[1] * 2)
    assert n_from_res == n_from_spk == n


def test_shared_detection_uses_ground_truth_res(tmp_path):
    """shared_res bypasses the MAD detector; .res equals the input indices
    (minus any truncated-at-boundary peaks)."""
    tr, truth, sr = _synth_tetrode(n_samp=20000 * 2, spike_hz=20.0, seed=3)
    # Keep only peaks safely inside boundaries
    safe = truth[(truth >= _NS_PEAK) &
                 (truth + (_NS_SAMPLES - _NS_PEAK) <= tr.shape[0])]
    rec = _FakeRec(tr, sr)
    base = str(tmp_path / "session")
    n_written = detect_and_write_klusters(rec, base=base, group=2,
                                           shared_res=safe)
    on_disk = np.fromfile(f"{base}.res.standard.2", dtype="<i8").tolist()
    assert on_disk == sorted(safe.tolist())
    assert n_written == safe.size


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


def test_detect_and_write_klusters_emits_fet_too(tmp_path):
    """.fet.standard.<g> must land alongside .res + .spk."""
    tr, _, sr = _synth_tetrode(n_samp=20000 * 3, spike_hz=40.0, seed=7)
    rec = _FakeRec(tr, sr)
    base = str(tmp_path / "session")
    n_detected = detect_and_write_klusters(rec, base=base, group=1,
                                            n_pcs_per_channel=3)
    fet_path = Path(f"{base}.fet.standard.1")
    assert fet_path.exists()
    # Header byte count + body byte count
    expected_body_bytes = n_detected * (3 * tr.shape[1] + 1) * 8
    assert fet_path.stat().st_size == 4 + expected_body_bytes


def test_detect_and_write_zero_spikes_returns_zero(tmp_path):
    rng = np.random.default_rng(0)
    tr = (rng.standard_normal((20000 * 2, 4)) * 10.0).astype(np.float32)
    rec = _FakeRec(tr, 20000.0)
    base = str(tmp_path / "session")
    n = detect_and_write_klusters(rec, base=base, group=1,
                                   threshold_sigma=15.0)
    assert n < 20
    # All three files still land — downstream readers rely on the triple
    # being present and self-consistent, even when empty.
    assert Path(f"{base}.res.standard.1").exists()
    assert Path(f"{base}.spk.standard.1").exists()
    assert Path(f"{base}.fet.standard.1").exists()


# ─── fiber-kit.yaml setup ───────────────────────────────────────────────────
def test_write_fk_config_pins_realign_off(tmp_path):
    """The auto-written fiber-kit.yaml must set FK_SESSION_REALIGN=off, with
    `off` quoted so YAML 1.1 doesn't coerce it to a boolean."""
    from ns3_sort_bench.adapters.fiber_kit import _FiberKitBase
    path = _FiberKitBase._write_fk_config(str(tmp_path))
    assert Path(path).name == "fiber-kit.yaml"
    body = Path(path).read_text()
    assert 'FK_SESSION_REALIGN: "off"' in body
    # Quoted — a bare `off` would YAML-coerce to False
    assert "FK_SESSION_REALIGN: off\n" not in body
    # Round-trip under PyYAML to confirm the string survives
    import yaml
    parsed = yaml.safe_load(body)
    assert parsed["FK_SESSION_REALIGN"] == "off"
