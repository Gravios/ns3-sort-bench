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

# Canonical neurosuite-3 file naming (doc/ndmanager-plugins/formats/naming.md):
#   <base>.<type>.<method>.<group>[.<suffix>]
# where <method> is a family token from {standard, sdiff, stderiv} optionally
# followed by _S<n> / _C<n> / _D<lag><dims>.  Our detector reads the raw .fil
# bandpass-filtered to 300-6000 Hz and writes .spk.standard.<group> — the
# "standard" family means raw waveform values (no spatial or temporal
# derivative).  Legacy untagged forms (.res.<N> / .spk.<N>) are still accepted
# by fiber-kit's Shared-type resolver as a fallback, but writing them tags
# every downstream resolve as a legacy-fallback and locks the pipeline out of
# the stderiv family.  See
# doc/ndmanager-plugins/formats/{naming,res,spk}.md in neurosuite-3.
_NS3_METHOD = "standard"

# Byte-level fiber-kit file formats (from fiber_kit/neuro_io.py):
#   .res.<method>.<g>   RES_DTYPE  = np.dtype("<i8")  binary int64 LE, no header,
#                                                     ascending sample indices
#   .spk.<method>.<g>   SPK_DTYPE  = np.dtype("<i2")  binary int16 LE, no header,
#                                                     sample-major, C-order
#                                                     (n_spikes, nsamp, nchan)
#   .fet.<method>.<g>   FET_HDR_DTYPE = np.dtype("<i4") × 1 (n_features header)
#                        FET_DTYPE     = np.dtype("<i8") × (n_spikes × n_features)
#                                        row-major; last column is the spike sample index
# fiber-session's `main()` reads .res + .spk at entry; fiber-kit's later stages
# (realign / refine / peel / intrachunk / link) and Klusters open the .fet, so
# writing it alongside .res / .spk keeps every downstream consumer happy.
_RES_DTYPE     = np.dtype("<i8")
_SPK_DTYPE     = np.dtype("<i2")
_FET_HDR_DTYPE = np.dtype("<i4")
_FET_DTYPE     = np.dtype("<i8")
# PCA features are floats; Klusters stores them as int64 scaled by this
# factor so modest precision survives the round-trip.  Matches the
# traditional ndm_pca scaling; exact value is immaterial for a benchmark.
_FET_INT_SCALE = 10_000.0


def _klusters_path(base: str, type_: str, group: int, method: str = _NS3_METHOD) -> str:
    """Canonical ns3 path: `<base>.<type>.<method>.<group>`."""
    return f"{base}.{type_}.{method}.{group}"


_SLUG_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-+")


def _slugify(name: str) -> str:
    """Filesystem-safe form of a corpus-item name.  Keeps `[A-Za-z0-9._+-]`,
    replaces every other character (including `:` and `/`) with `_`.
    Collapses adjacent underscores and trims leading/trailing ones.
    """
    out = "".join(c if c in _SLUG_SAFE else "_" for c in name)
    # collapse runs
    parts = [p for p in out.split("_") if p]
    slug = "_".join(parts) or "sort"
    return slug[:200]   # keep room for the .fiber-session.log etc. names


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


def _compute_features(snippets: np.ndarray, peaks: np.ndarray,
                       n_pcs_per_channel: int = 3) -> np.ndarray:
    """PCA features for Klusters-style .fet files.

    Returns an (n_spikes, n_pcs_per_channel × n_channels + 1) int64 matrix.
    The last column is the spike sample index (time feature) so KK's
    `useFeatures` default of masking that one off works as expected.

    Per-channel PCA: for each channel, SVD the centred (n_spikes, n_samples)
    waveform matrix and keep the top `n_pcs_per_channel` components.  Values
    are scaled by `_FET_INT_SCALE` and cast to int64 — Klusters' on-disk
    convention preserves modest fractional precision through the int round-trip.
    """
    n_spikes, n_samples, n_ch = snippets.shape
    if n_spikes == 0:
        return np.zeros((0, n_pcs_per_channel * n_ch + 1), dtype=_FET_DTYPE)
    cols = []
    for ch in range(n_ch):
        X = snippets[:, :, ch].astype(np.float64)
        X -= X.mean(axis=0, keepdims=True)
        # Rank is min(n_spikes, n_samples); if the caller asked for more PCs
        # than available, pad with zeros so the output shape is deterministic.
        k = min(int(n_pcs_per_channel), X.shape[1], X.shape[0])
        if k == 0:
            cols.append(np.zeros((n_spikes, n_pcs_per_channel), dtype=np.float64))
            continue
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        pc = U[:, :k] * S[:k]
        if k < n_pcs_per_channel:
            pc = np.hstack([pc, np.zeros((n_spikes, n_pcs_per_channel - k))])
        cols.append(pc)
    cols.append(peaks.astype(np.float64).reshape(-1, 1))     # time column
    features = np.hstack(cols)
    scaled = np.rint(features * _FET_INT_SCALE).astype(_FET_DTYPE)
    return scaled


def _write_fet(path: str, features: np.ndarray) -> None:
    """Binary int32 header (nFeatures per row) + int64 body, row-major.

    Matches fiber-kit's `write_fet_file` / Klusters' `readFetBinary` / ndm_pca
    byte layout — the format the full fiber-kit pipeline and the KK recluster
    step both expect.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}")
    n_features = int(features.shape[1])
    body = np.ascontiguousarray(features, dtype=_FET_DTYPE)
    with open(path, "wb") as f:
        np.array([n_features], dtype=_FET_HDR_DTYPE).tofile(f)
        body.tofile(f)


# ─── ndm_* delegation ────────────────────────────────────────────────────────
# Rather than reimplement detection + extraction + PCA in Python, the adapter
# shells out to neurosuite-3's `ndm_detectspikes`, `ndm_extractspikes` and
# `ndm_pca`.  Reasons:
#   - these are the reference implementations every other ns3 tool was built
#     against (byte-exact outputs, correct stderiv transform, varimax PCA);
#   - maintaining a parallel Python port is a liability — any drift between it
#     and the C++/bash sources would silently bias benchmark numbers;
#   - neurosuite-3 is already installed in every environment this harness
#     runs in (the adapter's Contract documents the hard dep).

_NDM_DETECT_DEFAULTS = {
    "thresholdFactor":  "0.8",     # × 4σ ≈ 3.2σ, the standard CA1 setting
    "refractoryPeriod": "16",      # samples
    "peakSearchLength": "16",      # samples
    "start":            "0",       # noise-window start (s)
    "duration":         "10",      # noise-window length (s)
    "method":           "standard",
}

_NDM_EXTRACT_DEFAULTS = {"method": "standard"}

_NDM_PCA_DEFAULTS = {"method": "standard"}


def _programs_yaml_section(overrides: Optional[Mapping[str, Mapping[str, str]]] = None
                            ) -> list:
    """Build the `programs:` list for an auto-generated session yaml.

    `ndm_detectspikes` / `ndm_extractspikes` / `ndm_pca` each read their
    parameters from a `programs[].name == "ndm_<step>"` block of the session
    yaml (doc/ndmanager-plugins/parameters.md).  This produces the minimal
    set with sensible CA1 defaults; `overrides` merges per-step overrides
    keyed by step name.
    """
    ov = dict(overrides or {})
    def _block(step, defaults):
        params = {**defaults, **ov.get(step, {})}
        return {
            "name": step,
            "parameters": [
                {"name": k, "value": v, "status": "Optional"}
                for k, v in params.items()
            ],
        }
    return [
        _block("ndm_detectspikes", _NDM_DETECT_DEFAULTS),
        _block("ndm_extractspikes", _NDM_EXTRACT_DEFAULTS),
        _block("ndm_pca",           _NDM_PCA_DEFAULTS),
    ]


def _run_ndm(cmd_name: str, session_yaml: str, *,
             extra: Sequence[str] = (), env: Optional[Mapping[str, str]] = None,
             log_path: Optional[str] = None) -> None:
    """Invoke a single `ndm_*` tool as a subprocess.

    Captured stdout + stderr land in `log_path` when given; otherwise they
    ride straight through to the parent's output.  Raises RuntimeError with
    the log tail on a non-zero exit.
    """
    cmd = [cmd_name, session_yaml, "--force", *list(extra)]
    env_full = {**os.environ, **(env or {})}
    if log_path:
        with open(log_path, "wb") as logf:
            r = subprocess.run(cmd, env=env_full,
                                 stdout=logf, stderr=subprocess.STDOUT,
                                 check=False)
    else:
        r = subprocess.run(cmd, env=env_full, check=False)
    if r.returncode != 0:
        tail = ""
        if log_path and os.path.exists(log_path):
            try:
                tail = open(log_path).read()[-2000:]
            except OSError:
                pass
        raise RuntimeError(
            f"{cmd_name} failed (exit {r.returncode}); log tail:\n{tail}"
        )


def detect_and_write_klusters(recording, base: str, group: int,
                                *, method: str = _NS3_METHOD,
                                shared_res: Optional[np.ndarray] = None,
                                env: Optional[Mapping[str, str]] = None,
                                log_dir: Optional[str] = None,
                                ) -> int:
    """Produce the Klusters-convention per-group files from a SpikeInterface
    BaseRecording, delegating to neurosuite-3's `ndm_*` binaries:

        <base>.res.<method>.<group>   ← ndm_detectspikes (or shared_res, below)
        <base>.spk.<method>.<group>   ← ndm_extractspikes
        <base>.fet.<method>.<group>   ← ndm_pca
        <base>.pca.<method>.<group>   ← ndm_pca (PCAE basis)

    Assumes the shared `.fil` + `<base>.yaml` already exist next to `base`
    (the adapter's `_materialise_recording` writes both before calling this).

    When `shared_res` is provided (a 1-D int array of sample indices), those
    are used as the detections instead of running `ndm_detectspikes`.
    `ndm_extractspikes` and `ndm_pca` then run against that `.res` as normal.

    Returns the number of spikes in the resulting `.res`.  Raises RuntimeError
    with the step log tail when any ndm_* invocation exits non-zero.
    """
    session_yaml = base + ".yaml"
    res_path = _klusters_path(base, "res", group, method=method)

    if shared_res is not None:
        # Project the GT's sample indices as the shared .res; skip detection.
        arr = np.sort(np.asarray(shared_res, dtype=_RES_DTYPE).ravel())
        with open(res_path, "wb") as f:
            arr.tofile(f)
    else:
        _run_ndm("ndm_detectspikes", session_yaml,
                  extra=["--method", method], env=env,
                  log_path=os.path.join(log_dir, "ndm_detectspikes.log")
                           if log_dir else None)

    _run_ndm("ndm_extractspikes", session_yaml,
              extra=["--method", method], env=env,
              log_path=os.path.join(log_dir, "ndm_extractspikes.log")
                       if log_dir else None)
    _run_ndm("ndm_pca", session_yaml,
              extra=["--method", method], env=env,
              log_path=os.path.join(log_dir, "ndm_pca.log")
                       if log_dir else None)

    n = int(os.path.getsize(res_path) // _RES_DTYPE.itemsize)
    return n


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
            "Requires neurosuite-3's ndm_detectspikes, ndm_extractspikes and "
            "ndm_pca on PATH.  For a from-scratch SpikeInterface recording "
            "(SpikeForest paired items, Propper simulator, ...) the adapter "
            "writes <base>.fil + <base>.yaml (with a `programs:` block "
            "carrying CA1-sensible defaults) and shells out to the three "
            "ndm_* tools in sequence to produce .res.standard.<g>, "
            ".spk.standard.<g>, .fet.standard.<g> and .pca.standard.<g> under "
            "ns3's canonical naming (doc/ndmanager-plugins/formats/naming.md).  "
            "--method standard is injected into every pipeline stage so "
            "resolves stay in the raw-waveform family; the readback expects "
            "variant=`standard` and tag=`fiber_session` (session-only mode) "
            "or `fiber_session_linked` (full chain).",
            "fiber-kit.yaml setup: pass fk_config_path=<path> or set "
            "$FK_CONFIG to point at your own tuning yaml.  When neither is set, "
            "the adapter writes a minimal `fiber-kit.yaml` next to the session "
            "in the tmp dir pinning FK_SESSION_REALIGN=off (the end-of-run "
            "realign would otherwise ask for .spk.stderiv.<g> and refuse to "
            "operate on our raw waveforms).  Precedence: explicit flag > env "
            "FK_* > this file > session yaml > built-in default.",
            "shared_detection=True reuses the GT's spike samples as the .res "
            "for paired-recordings benchmarks — isolates fiber-kit's clustering "
            "quality from its detection quality.  Requires ground_truth_res.",
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

    # Default `.clu.<variant>.<group>.<tag>` tag written by the LAST stage of
    # each mode's chain.  fiber-session writes `fiber_session`; fiber-link
    # suffixes `_linked`; so the end-of-chain product is one of these.
    _MODE_FINAL_TAG = {
        "fiber_kit":            "fiber_session_linked",
        "fiber_kit_no_deconv":  "fiber_session_linked",
        "fiber_kit_session":    "fiber_session",
    }

    @staticmethod
    def _write_fk_config(tmp_dir: str) -> str:
        """Write a minimal `fiber-kit.yaml` into the session's tmp dir and
        return its path.  Overrides that matter for a from-scratch
        SpikeInterface recording (no stderiv variant available):

          * FK_SESSION_REALIGN: "off"   — the end-of-run realign refuses to
                                           work on raw waveforms when its token
                                           asks for stderiv; we have no
                                           .spk.stderiv.<g>, so disable it.

        The quoted form is deliberate: YAML 1.1 parses bare `off` as a boolean,
        which fiber-kit's config loader then rejects as the wrong type.
        """
        path = os.path.join(tmp_dir, "fiber-kit.yaml")
        with open(path, "w") as f:
            f.write(
                "# Written by ns3_sort_bench.adapters.fiber_kit when it "
                "materialised\n"
                "# this session from a SpikeInterface BaseRecording.  Keep the\n"
                "# quoted `off` strings — YAML 1.1 reads bare off/on as bools.\n"
                "FK_SESSION_REALIGN: \"off\"\n"
            )
        return path

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
        materialised = False
        keep_artifacts = False
        if sess_base is None:
            materialised = True
            fk_artifacts_dir = kwargs.get("fk_artifacts_dir")
            if fk_artifacts_dir:
                # Persistent mode: place the session dir under the caller's
                # artifacts root, named by the corpus item (slugified).  The
                # dir is NOT deleted after the sort finishes — its full
                # contents survive for post-hoc inspection.
                slug = _slugify(str(kwargs.get("corpus_name") or "sort"))
                tmp_dir = os.path.join(fk_artifacts_dir, slug)
                os.makedirs(tmp_dir, exist_ok=True)
                keep_artifacts = True
            else:
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
                shared_res = np.asarray(ground_truth_res, dtype=_RES_DTYPE)
            n_detected = detect_and_write_klusters(
                recording, sess_base, group,
                method=_NS3_METHOD,
                shared_res=shared_res,
                env=kwargs.get("ndm_env"),
                log_dir=tmp_dir,
            )
            if n_detected == 0:
                raise RuntimeError(
                    f"fiber_kit adapter: detection produced 0 spikes for "
                    f"group {group} of {sess_base}.  Lower "
                    "`thresholdFactor` in the session yaml's "
                    "`programs:` → ndm_detectspikes block."
                )

            # fiber-kit.yaml setup — explicit path wins; otherwise we write a
            # minimal one next to the session and point FK_CONFIG at it.  This
            # pins FK_SESSION_REALIGN=off (no .spk.stderiv.<g> exists) and makes
            # the tuning reproducible across workstations.
            fk_config = kwargs.get("fk_config_path") or env.get("FK_CONFIG")
            if not fk_config:
                fk_config = self._write_fk_config(tmp_dir)
            env["FK_CONFIG"] = fk_config

        # Inject --method standard on auto-materialised runs so every stage
        # resolves our raw .spk.standard.<g> / .res.standard.<g> files directly
        # (no stderiv fallback).  Lab-session usage keeps whatever --method is
        # baked into the extra_args / env.
        stage_args = list(kwargs.get("extra_args", []))
        if materialised:
            stage_args = ["--method", _NS3_METHOD, *stage_args]

        with measure_runtime() as rt:
            for stage in self._MODE_STAGES[self._mode]:
                self._run_stage(stage, sess_base, group, env=env,
                                 extra_args=stage_args)

        # Readback defaults.  For a lab-session sort the user's full pipeline
        # emits `stderiv_C5_D34` / `exp_final` at the end.  For an auto-
        # materialised sort we only produced the standard-family chain, so the
        # end-of-chain tag is mode-dependent (see _MODE_FINAL_TAG).
        if materialised:
            default_variant = _NS3_METHOD
            default_tag = self._MODE_FINAL_TAG[self._mode]
        else:
            default_variant = "stderiv_C5_D34"
            default_tag = "exp_final"
        variant = kwargs.get("variant", default_variant)
        tag = kwargs.get("tag", default_tag)
        sort_ex = NS3SortingExtractor(
            sess_base + ".yaml", electrode_group=group,
            variant=variant, tag=tag,
        )
        spike_times = {int(u): np.asarray(sort_ex.get_unit_spike_train(u), np.int64)
                       for u in sort_ex.get_unit_ids()}

        extras = dict(mode=self._mode, electrode_group=group,
                       variant=variant, tag=tag, session_base=sess_base)
        # Cleanup policy, in order of precedence:
        #   keep_artifacts (set when the harness passed fk_artifacts_dir)  → always keep
        #   preserve_outputs=True kwarg                                     → always keep
        #   otherwise, sort-success                                         → delete tmp
        if tmp_dir:
            if keep_artifacts or kwargs.get("preserve_outputs", False):
                extras["output_dir"] = tmp_dir
            else:
                try: shutil.rmtree(tmp_dir)
                except OSError: pass

        return SortedResult(
            unit_ids=sorted(spike_times), spike_times=spike_times,
            sampling_rate=recording.get_sampling_frequency(),
            adapter=self._mode,
            # Placeholder; BenchmarkRun.go overrides with the authoritative
            # `corpus.name`.  Any value works, but keep sess_base so direct
            # (harness-bypassing) calls to adapter.sort() produce a useful
            # debuggable identifier.
            corpus_item=sess_base,
            adapter_version=self._version,
            channels=list(recording.get_channel_ids()),
            wall_time_s=rt["wall_time_s"], peak_rss_mb=rt["peak_rss_mb"],
            extras=extras,
        )

    def _materialise_recording(self, recording, tmp_dir: str) -> str:
        """Write `<base>.fil` + `<base>.yaml` for a from-scratch SpikeInterface
        recording.  The yaml carries both the session description that every
        ns3 reader needs (`acquisitionSystem`, `spikeDetection.channelGroups`)
        and a `programs:` block with sensible CA1 defaults for ndm_detectspikes,
        ndm_extractspikes and ndm_pca — so a plain
        `ndm_detectspikes <base>.yaml` works without further configuration.
        """
        import yaml
        n_samp = recording.get_num_samples(0)
        n_ch = recording.get_num_channels()
        sr = recording.get_sampling_frequency()
        base = os.path.join(tmp_dir, "sim_session")
        traces = recording.get_traces(start_frame=0, end_frame=n_samp)
        traces_i16 = np.clip(np.rint(traces), -32768, 32767).astype(np.int16)
        traces_i16.tofile(base + ".fil")
        # If the recording is shorter than the default 10-s noise window, clip
        # the window to the recording duration so ndm_detectspikes doesn't
        # error on `start + duration > samples / sr`.
        noise_duration = min(10.0, float(n_samp) / float(sr) * 0.5)
        yml = {
            "parameters": {"version": "1.0", "creator": "ns3-sort-bench"},
            "acquisitionSystem": dict(nBits=16, nChannels=int(n_ch),
                                       voltageRange=20, amplification=1000,
                                       offset=0, samplingRate=float(sr)),
            "spikeDetection": {"channelGroups": [dict(
                channels=list(range(n_ch)),
                nSamples=_NS_SAMPLES, peakSampleIndex=_NS_PEAK, nFeatures=3,
            )]},
            # ndm_detectspikes / ndm_extractspikes / ndm_pca read their
            # parameters here.  Overrides are per-adapter / per-call via
            # the sort() kwargs `ndm_*_overrides` dict.
            "programs": _programs_yaml_section(overrides={
                "ndm_detectspikes": {"duration": str(noise_duration)},
            }),
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
