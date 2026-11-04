"""fiber-kit adapter — wraps the fiber-session chain behind SorterAdapter.

Three modes registered: fiber_kit (full chain), fiber_kit_no_deconv,
fiber_kit_session.  All share the same Contract shape; only the
`_mode` + env override differ.

For a lab session, `fiber-session` reads a pre-existing `.res.<group>`
(and `.spk.<group>`) produced by `ndm_extractspikes`.  For a SpikeInterface
BaseRecording that comes in from a corpus loader (SpikeForest paired data,
the Propper simulator, etc.), nothing has produced those files yet, so the
adapter must detect spikes and write the Klusters-convention
`.res.<group>` + `.spk.<group>` ahead of the fiber-session chain.

Shared-detection mode: pass `shared_detection=True` as a sort kwarg and the
adapter writes the ground_truth_res sample indices as the `.res.<group>`
instead of detecting.  Useful for paired recordings — it isolates fiber-kit's
*clustering* quality from its detection quality.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Optional

import numpy as np

from .._base import (
    SortedResult, SorterAdapter, _register, adapter_registry, measure_runtime,
)
from ..contracts import (
    Contract, RecordingReqs, OutputSchema, Capabilities, ResourceEstimates,
)


# ─── Klusters-convention spike detection ─────────────────────────────────────
# Written here (rather than in a shared module) because this is the only
# caller.  Self-contained: only depends on scipy, which is a core dep.

_NS_SAMPLES = 32          # Klusters snippet length
_NS_PEAK    = 16          # offset of peak sample within the snippet
_DEFAULT_THRESHOLD_SIGMA = 5.0
_DEFAULT_DEAD_TIME_MS    = 0.5
_BAND_HZ = (300.0, 6000.0)

# Byte-level fiber-kit file formats (from fiber_kit/neuro_io.py):
#   .res.<g>   RES_DTYPE  = np.dtype("<i8")  binary int64 LE, no header
#   .spk.<g>   SPK_DTYPE  = np.dtype("<i2")  binary int16 LE, no header,
#                                             C-order (n_spikes, nsamp, nchan)
# fiber-session's `main()` reads only .res + .spk at entry; .fet / .clu / .nst
# are outputs or optional.
_RES_DTYPE = np.dtype("<i8")
_SPK_DTYPE = np.dtype("<i2")


def _bandpass(traces: np.ndarray, sr: float,
              low: float = _BAND_HZ[0], high: float = _BAND_HZ[1]) -> np.ndarray:
    """4th-order Butterworth bandpass along axis 0.  Returns float64."""
    from scipy.signal import iirfilter, sosfiltfilt
    nyq = 0.5 * float(sr)
    high_eff = min(float(high), 0.99 * nyq)
    sos = iirfilter(N=4, Wn=[float(low), high_eff], btype="bandpass",
                    ftype="butter", fs=float(sr), output="sos")
    return sosfiltfilt(sos, np.asarray(traces, dtype=np.float64), axis=0)


def _detect_peak_indices(traces_filt: np.ndarray, sr: float,
                          threshold_sigma: float = _DEFAULT_THRESHOLD_SIGMA,
                          dead_time_ms: float = _DEFAULT_DEAD_TIME_MS
                          ) -> np.ndarray:
    """MAD-threshold peak detector on the negative-going cross-channel envelope.
    Returns a 1-D int64 array of sample indices, sorted ascending."""
    from scipy.signal import find_peaks
    neg = -traces_filt                               # positive peaks ↔ negative spikes
    env = neg.max(axis=1)                             # cross-channel envelope
    # Per-channel noise, then take the median across channels for a robust floor.
    mad_per_ch = np.median(np.abs(traces_filt - np.median(traces_filt, axis=0)), axis=0)
    noise_sigma = 1.4826 * float(np.median(mad_per_ch))
    threshold = float(threshold_sigma) * max(noise_sigma, 1e-6)
    dead = max(int(round(1e-3 * dead_time_ms * sr)), 1)
    idx, _ = find_peaks(env, height=threshold, distance=dead)
    return np.asarray(idx, dtype=np.int64)


def _extract_snippets(traces_filt: np.ndarray, peaks: np.ndarray,
                       n_samples: int = _NS_SAMPLES,
                       peak_sample: int = _NS_PEAK) -> np.ndarray:
    """Slice `(n_spikes, n_samples, n_ch)` int16 snippets around each peak.
    Peaks too close to either end are dropped and their indices are returned
    too, so the caller can match .res to .spk 1-to-1."""
    n_total, n_ch = traces_filt.shape
    before = peak_sample
    after = n_samples - peak_sample
    ok = (peaks >= before) & (peaks + after <= n_total)
    kept = peaks[ok]
    out = np.empty((kept.size, n_samples, n_ch), dtype=_SPK_DTYPE)
    for i, p in enumerate(kept):
        s0 = int(p) - before
        s1 = s0 + n_samples
        out[i] = np.clip(np.rint(traces_filt[s0:s1]), -32768, 32767).astype(_SPK_DTYPE)
    return kept, out


def _write_res(path: str, peak_samples: np.ndarray) -> None:
    """Binary little-endian int64, ascending, no header.

    This is what fiber-kit's `read_res_file` expects:
        return np.fromfile(path, dtype=RES_DTYPE).astype(np.int64)
    An ASCII "<int>\\n" per line format decodes as random int64s whose byte
    count divides into a near-correct spike count by coincidence (~8 bytes
    per line ≈ 8 bytes per int64) and fails fiber-session's `.res`/`.spk`
    cross-check at a slightly-off number — observed as "5,942 vs 6,053
    spikes -- per-spike files out of step" before this patch.
    """
    arr = np.sort(np.asarray(peak_samples, dtype=_RES_DTYPE).ravel())
    with open(path, "wb") as f:
        arr.tofile(f)


def _write_spk(path: str, snippets_i16: np.ndarray) -> None:
    """Binary little-endian int16, C-order (n_spikes, n_samples, n_ch), no header.

    fiber-kit's `open_spk_file` memmaps as flat int16 and reshapes to
    (n, nsamp, nchan) — so the byte layout must put sample-0 across all
    channels first, then sample-1, etc., within each spike.  C-order of
    the (n_spikes, n_samples, n_ch) array gives exactly that layout.
    """
    arr = np.ascontiguousarray(snippets_i16, dtype=_SPK_DTYPE)
    with open(path, "wb") as f:
        arr.tofile(f)


def detect_and_write_klusters(recording, base: str, group: int,
                                *, threshold_sigma: float = _DEFAULT_THRESHOLD_SIGMA,
                                dead_time_ms: float = _DEFAULT_DEAD_TIME_MS,
                                n_samples: int = _NS_SAMPLES,
                                peak_sample: int = _NS_PEAK,
                                shared_res: Optional[np.ndarray] = None,
                                ) -> int:
    """Produce `<base>.res.<group>` + `<base>.spk.<group>` from a
    SpikeInterface BaseRecording.  Returns the number of spikes written.

    When `shared_res` is provided (a 1-D int array of sample indices), those
    are used as the detections instead of running the MAD detector.
    Snippets are still extracted from the filtered trace so fiber-session
    has `.spk` to work with.
    """
    # Load traces (bytes are small for paired tetrode items; chunk later if needed).
    n_samp = int(recording.get_num_samples(0))
    traces = recording.get_traces(start_frame=0, end_frame=n_samp)
    sr = float(recording.get_sampling_frequency())
    traces_f = _bandpass(traces, sr)
    if shared_res is None:
        peaks = _detect_peak_indices(traces_f, sr, threshold_sigma, dead_time_ms)
    else:
        peaks = np.asarray(shared_res, dtype=_RES_DTYPE)
    kept, snippets = _extract_snippets(traces_f, peaks,
                                        n_samples=n_samples,
                                        peak_sample=peak_sample)
    _write_res(f"{base}.res.{group}", kept)
    _write_spk(f"{base}.spk.{group}", snippets)
    return int(kept.size)


def _contract_for(mode: str, with_deconv: bool) -> Contract:
    return Contract(
        adapter_name=mode,
        upstream_url="https://github.com/Gravios/fiber-kit",
        upstream_version_pin=">= 0.57.6",
        yaml_fields_required=(
            "acquisitionSystem.samplingRate",
            "acquisitionSystem.nChannels",
            "acquisitionSystem.nBits",
            "spikeDetection.channelGroups",
            "spikeDetection.channelGroups[0].channels",
        ),
        yaml_fields_optional=(
            "spikeDetection.channelGroups[0].nSamples",
            "spikeDetection.channelGroups[0].peakSampleIndex",
            "spikeDetection.channelGroups[0].nFeatures",
            "fieldPotentials.lfpSamplingRate",
        ),
        recording=RecordingReqs(
            min_channels=1, min_duration_s=30.0,
            dtype=("int16", "float32"), probe_geometry_required=False,
            signal_space="raw",       # fiber-session reads .fil directly
        ),
        output=OutputSchema(
            unit_id_offset=2, includes_reserved=False, templates_emitted=False,
            per_group_or_whole="per_group",
        ),
        capabilities=Capabilities(
            drift_aware=True,
            overlap_resolution="omp" if with_deconv else "none",
            requires_gpu=False, requires_torch=False,
            supports_stderiv_input=True, supports_shared_detection=True,
        ),
        resources=ResourceEstimates(
            min_ram_mb=4096, min_cpu_cores=4,
            estimated_sec_per_minute_of_recording=30.0,
        ),
        known_limitations=(
            "Session yaml must be ADJACENT to the .fil / .dat (same basename).",
            "Deconv ON requires fiber-kit >= 0.57.6 for the FK_SESSION_DECONV env flag.",
            "Variant/tag names must match the stage output you want "
            "(default: stderiv_C5_D34 / exp_final).",
            "fiber-session needs a pre-existing .res.<group> + .spk.<group>; "
            "the adapter detects and writes them for a from-scratch SpikeInterface "
            "recording.  Override with shared_detection=True to reuse the GT's "
            "spike samples as the detections.",
        ),
        notes="Subject of the benchmark; three modes registered as fiber_kit, "
              "fiber_kit_no_deconv, fiber_kit_session.",
    )


class _FiberKitBase(SorterAdapter):
    _mode: str = "fiber_kit"
    _env_overrides: dict = {}
    _contract_obj: Optional[Contract] = None

    def __init__(self):
        try:
            import fiber_kit
            self._available = True
            self._version = getattr(fiber_kit, "__version__", "unknown")
        except ImportError:
            self._available = False; self._version = ""

    @property
    def contract(self): return self._contract_obj

    @property
    def version(self): return self._version

    def available(self):
        if not self._available:
            return False, "fiber_kit not installed (`pip install ns3-sort-bench[fiber-kit]`)"
        return True, ""

    _MODE_STAGES = {
        "fiber_kit": ["fiber-session", "fiber-realign", "fiber-refine",
                      "fiber-peel", "fiber-intrachunk", "fiber-link"],
        "fiber_kit_no_deconv": ["fiber-session", "fiber-realign", "fiber-refine",
                                 "fiber-peel", "fiber-intrachunk", "fiber-link"],
        "fiber_kit_session": ["fiber-session"],
    }

    def sort(self, recording, *, ns3_yaml=None, ground_truth_res=None,
             **kwargs) -> SortedResult:
        if not self._available:
            raise EnvironmentError("fiber_kit not available")

        from ..ns3.extractors import NS3SortingExtractor
        from spikeinterface.core import BaseRecording
        if not isinstance(recording, BaseRecording):
            raise TypeError("fiber_kit adapter needs a SpikeInterface BaseRecording")

        env = {**os.environ, **self._env_overrides, **kwargs.get("env", {})}
        group = int(kwargs.get("electrode_group", 1))

        sess_base = recording.get_annotation("session_base", None)
        tmp_dir = None
        if sess_base is None:
            tmp_dir = tempfile.mkdtemp(prefix="ns3sb_fk_")
            sess_base = self._materialise_recording(recording, tmp_dir)
            # fiber-session expects a Klusters-convention .res + .spk alongside
            # the .fil / .yaml we just wrote.  Detect and emit them now,
            # or (if the caller asked for shared detection with the GT)
            # project ground_truth_res sample indices as the .res.
            shared_res = None
            if kwargs.get("shared_detection", False):
                if ground_truth_res is None:
                    raise ValueError(
                        "fiber_kit adapter: shared_detection=True requires "
                        "ground_truth_res (the GT's spike sample indices)."
                    )
                shared_res = np.asarray(ground_truth_res, dtype=np.int64)
            n_detected = detect_and_write_klusters(
                recording, sess_base, group,
                threshold_sigma=float(kwargs.get("detect_threshold_sigma",
                                                  _DEFAULT_THRESHOLD_SIGMA)),
                dead_time_ms=float(kwargs.get("detect_dead_time_ms",
                                               _DEFAULT_DEAD_TIME_MS)),
                shared_res=shared_res,
            )
            if n_detected == 0:
                raise RuntimeError(
                    f"fiber_kit adapter: detection produced 0 spikes for "
                    f"group {group} of {sess_base}.  Try lowering "
                    "detect_threshold_sigma (default 5.0)."
                )

        with measure_runtime() as rt:
            for stage in self._MODE_STAGES[self._mode]:
                self._run_stage(stage, sess_base, group, env=env,
                                 extra_args=kwargs.get("extra_args", []))

        variant = kwargs.get("variant", "stderiv_C5_D34")
        tag = kwargs.get("tag", "exp_final")
        sort_ex = NS3SortingExtractor(
            sess_base + ".yaml", electrode_group=group,
            variant=variant, tag=tag,
        )
        spike_times = {int(u): np.asarray(sort_ex.get_unit_spike_train(u), np.int64)
                       for u in sort_ex.get_unit_ids()}

        extras = dict(mode=self._mode, electrode_group=group,
                       variant=variant, tag=tag, session_base=sess_base)
        if tmp_dir and not kwargs.get("preserve_outputs", False):
            try: shutil.rmtree(tmp_dir)
            except OSError: pass
        elif tmp_dir:
            extras["output_dir"] = tmp_dir

        return SortedResult(
            unit_ids=sorted(spike_times), spike_times=spike_times,
            sampling_rate=recording.get_sampling_frequency(),
            adapter=self._mode,
            corpus_item=str(recording.get_annotation("name", sess_base)),
            adapter_version=self._version,
            channels=list(recording.get_channel_ids()),
            wall_time_s=rt["wall_time_s"], peak_rss_mb=rt["peak_rss_mb"],
            extras=extras,
        )

    def _materialise_recording(self, recording, tmp_dir: str) -> str:
        import yaml
        n_samp = recording.get_num_samples(0)
        n_ch = recording.get_num_channels()
        sr = recording.get_sampling_frequency()
        base = os.path.join(tmp_dir, "sim_session")
        traces = recording.get_traces(start_frame=0, end_frame=n_samp)
        traces_i16 = np.clip(np.rint(traces), -32768, 32767).astype(np.int16)
        traces_i16.tofile(base + ".fil")
        yml = {
            "parameters": {"version": "1.0", "creator": "ns3-sort-bench"},
            "acquisitionSystem": dict(nBits=16, nChannels=int(n_ch),
                                       voltageRange=20, amplification=1000,
                                       offset=0, samplingRate=float(sr)),
            "spikeDetection": {"channelGroups": [dict(
                channels=list(range(n_ch)),
                nSamples=32, peakSampleIndex=16, nFeatures=3,
            )]},
        }
        with open(base + ".yaml", "w") as f:
            yaml.safe_dump(yml, f, sort_keys=False)
        return base

    def _run_stage(self, stage, sess_base, group, *, env, extra_args):
        cmd = [stage, sess_base, str(group), *list(extra_args)]
        log_path = f"{sess_base}.{stage}.log"
        with open(log_path, "w") as logf:
            r = subprocess.run(cmd, env=env, stdout=logf,
                                stderr=subprocess.STDOUT, check=False)
        if r.returncode != 0:
            tail = open(log_path).read()[-2000:]
            raise RuntimeError(f"{stage} failed (exit {r.returncode}); log tail:\n{tail}")


class _Full(_FiberKitBase):
    name = "fiber_kit"; _mode = "fiber_kit"
    _env_overrides = {"FK_SESSION_DECONV": "1"}
    _contract_obj = _contract_for("fiber_kit", with_deconv=True)


class _NoDeconv(_FiberKitBase):
    name = "fiber_kit_no_deconv"; _mode = "fiber_kit_no_deconv"
    _env_overrides = {"FK_SESSION_DECONV": "0"}
    _contract_obj = _contract_for("fiber_kit_no_deconv", with_deconv=False)


class _SessionOnly(_FiberKitBase):
    name = "fiber_kit_session"; _mode = "fiber_kit_session"
    _contract_obj = _contract_for("fiber_kit_session", with_deconv=False)


_register(adapter_registry, _Full(), "fiber_kit")
_register(adapter_registry, _NoDeconv(), "fiber_kit_no_deconv")
_register(adapter_registry, _SessionOnly(), "fiber_kit_session")
