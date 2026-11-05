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
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .._base import (
    SortedResult, SorterAdapter, _register, adapter_registry, measure_runtime,
)
from ..configs import load_adapter_config
from ..contracts import (
    Contract, RecordingReqs, OutputSchema, Capabilities, ResourceEstimates,
)


# ─── Baseline tuning loaded from ns3_sort_bench/configs/fiber_kit.yaml ──────
# One place for the ndm_detect/extract/pca params and the FK_* baseline.
# A missing key falls back to the hard-coded fallback below (keeps the
# adapter working even if the file is deleted or empty).
_FK_CFG = load_adapter_config("fiber_kit")


def _cfg_section(name: str, fallback: Mapping[str, Any]) -> dict:
    """Return the dict value at ``_FK_CFG[name]`` when it's a mapping;
    otherwise the `fallback` dict verbatim.  Keeps the adapter resilient
    to a truncated/missing config while still preferring on-disk tuning."""
    v = _FK_CFG.get(name)
    return dict(v) if isinstance(v, Mapping) else dict(fallback)


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


def _hardlink_or_copy(src: str, dst: str) -> None:
    """Hardlink `src` → `dst` (zero disk cost); fall back to a copy on cross-fs
    or any other OSError.  No-op when `dst` already exists.
    """
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


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


def _fresh_artifacts_dir(root: str, slug: str) -> str:
    """Return ``<root>/<slug>``, empty, for one fiber_kit sort.

    The folder is reused across runs into the same ``--out``.  fiber-pipeline
    deliberately consumes outputs already on disk (``_pf_on_disk``, the
    method resolver's family match, realign's in-place commits), so a
    leftover ``.clu``/``.res``/``.spk`` from an earlier run — older code or
    other settings — fed silently into the next one (the second hc1_full
    run: 29 fiber_session clusters, recall 0.15, against 16 / 0.21 from a
    fresh folder).  Clearing it makes ``--keep-fk-artifacts`` keep the
    artifacts of *this* run only.
    """
    if slug in ("", ".", ".."):
        raise ValueError(f"fiber_kit adapter: unsafe artifacts slug {slug!r}")
    path = os.path.join(root, slug)
    if os.path.isdir(path) and os.listdir(path):
        print(f"ns3sb fiber_kit: clearing artifacts from an earlier run in "
              f"{path}", file=sys.stderr)
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


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

# Sourced from src/ns3_sort_bench/configs/fiber_kit.yaml (ndm_detect /
# ndm_extract / ndm_pca sections).  The in-code fallback below is used
# only when the yaml is missing or truncated — the yaml is authoritative.
_NDM_DETECT_DEFAULTS = _cfg_section("ndm_detect", {
    "thresholdFactor":  "0.8",     # × 4σ ≈ 3.2σ, the standard CA1 setting
    "refractoryPeriod": "16",      # samples
    "peakSearchLength": "16",      # samples
    "start":            "0",       # noise-window start (s)
    "duration":         "10",      # noise-window length (s)
    "method":           "standard",
})

_NDM_EXTRACT_DEFAULTS = _cfg_section("ndm_extract", {"method": "standard"})

_NDM_PCA_DEFAULTS = _cfg_section("ndm_pca", {"method": "standard"})

# Baseline FK_* overrides written into the per-session fiber-kit.yaml.
# When the user has their own `./fiber-kit.yaml` (or passes `fk_config_path`),
# their keys are merged in and win — this baseline fills the gaps.
#
# Minimal baseline (legacy per-stage modes: fiber_kit_no_deconv,
# fiber_kit_session).  Just the `fiber_kit:` structured section from
# configs/fiber_kit.yaml, holding FK_SESSION_REALIGN=off.
_FK_SESSION_BASELINE = _cfg_section("fiber_kit", {"FK_SESSION_REALIGN": "off"})


def _fk_pipeline_cfg_baseline() -> dict:
    """Full baseline for fiber_kit mode — dispatched through the
    `fiber-pipeline` driver.  Returns the ENTIRE top-level of
    configs/fiber_kit.yaml (every flat FK_* key plus the embedded
    `pipeline:` block) MINUS the structured ns3sb sections (ndm_detect /
    ndm_extract / ndm_pca / fiber_kit) that fiber-pipeline would warn on
    and skip.

    The `fiber_kit:` structured section's own keys (e.g.
    FK_SESSION_REALIGN) are promoted to flat FK_* keys in the baseline so
    their load-bearing defaults survive the ns3sb → fiber-pipeline hop.
    """
    _NS3_ONLY = {"ndm_detect", "ndm_extract", "ndm_pca", "fiber_kit"}
    base: dict = {k: v for k, v in _FK_CFG.items() if k not in _NS3_ONLY}
    # Promote the structured `fiber_kit:` keys back to flat FK_* keys
    # (they live at the top of configs/fiber_kit.yaml only because that
    # section carries the ns3sb-side adapter schema).
    fk_section = _FK_CFG.get("fiber_kit")
    if isinstance(fk_section, Mapping):
        for k, v in fk_section.items():
            base.setdefault(k, v)
    return base


_FK_PIPELINE_BASELINE = _fk_pipeline_cfg_baseline()

# Scoreable intermediate .clu tags produced along the fiber_kit mode's
# embedded 9-step plan (configs/fiber_kit.yaml → `pipeline:`):
#     fiber_session   — fiber-session's initial over-clustering
#     fs_clean        — fiber-consensus (quarantine-only) pass on session
#     exp_xcm         — fiber-anchor-link output
#     (exp_final      — fiber-consensus review; this is the PRIMARY
#                       readback, not re-emitted as a stage result)
# Order mirrors the plan's step order, which is also roughly
# precision-increasing — useful for a reader skimming the scored rows.
_FIBER_KIT_SCOREABLE_STAGES = ("fiber_session", "fs_clean", "exp_xcm")


def _channel_count_fk_overrides(n_channels: int) -> dict:
    """Return the FK_* overrides keyed on the session's channel count.

    The fiber-kit tuning shipped in ``configs/fiber_kit.yaml`` is the
    user's own, measured on an 8-channel octrode (sirotaA-jg-000005,
    group 6 / 7).  That geometry makes group-delay / warp features
    near-degenerate — the yaml compensates by relaxing warp thresholds
    and leaning on depth; on a 4-channel probe, the trade flips.

    The override map below mirrors the guidance in the yaml's leading
    comment table (CHANNEL-COUNT-SENSITIVE knobs), written when the yaml
    was ported in patch 0015.  Four- or fewer-channel groups (tetrodes
    and smaller) get the right-column tetrode values; everything else
    takes the yaml defaults.

    Returns {} outside that regime so the FK_PIPELINE baseline wins.
    """
    if n_channels <= 4:
        return {
            # LOOSE position gates — fewer channels means less z-axis
            # resolution; the 35 µm / 1.5 gates designed around 20-µm
            # octrode pitch become false negatives on tetrodes.
            "FK_INTRA_DEPTH_GATE": 60,
            "FK_LINK_POS_THR":     2.5,
            # TIGHTER warp gates — on a tetrode, warp (group delay) is
            # actually discriminative (not near-degenerate like on 8 ch),
            # so the gate can require more coherence.
            "FK_LINK_WARP_THR":    0.5,
            "FK_BBLINK_WARP_THR":  0.7,
            # ALINK primary-channel fraction — higher cutoff restricts
            # comparison to the top 2-3 channels on a 4-channel probe,
            # which is where a tetrode's amplitude profile actually sits.
            "FK_ALINK_PRIM_FRAC":  0.25,
            # FK_INTRA_BACKBONE_WARP is intentionally NOT overridden —
            # the yaml comment table marks it "unchanged".
        }
    return {}


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


# ─── probe-file writer for fiber-cpos / fiber-intrachunk ─────────────────────
# fiber-intrachunk runs the fiber-cpos localiser on-the-fly when no
# precomputed .cpos.<method>.<g>.<stage>.clusters.npz exists; the localiser
# reads probe geometry from a NeuroSuite .probe YAML:
#
#     probeFile:
#       sites:
#         geometry:
#           - [x, y]       # µm
#           - [x, y]
#           ...
#
# fiber-kit (fiber_localize.load_geometry) raises
#   "no probeFile.sites.geometry"
# when the key is missing, and fiber-intrachunk raises
#   "no fiber-cpos table and no probe geometry to compute one"
# when the probe path itself is absent from the session yaml.  On a from-
# scratch SpikeInterface recording the probe file doesn't exist on disk
# — we write it from `recording.get_channel_locations()`, which the
# SpikeForest paired loaders and NS3RecordingExtractor both populate (the
# mountainsort5 log confirms a 4-channel linear 0/25/50/75-µm layout for
# PAIRED_CRCNS_HC1).
def _write_probe_file(path: str, locations_xy: np.ndarray) -> None:
    """Write a minimal NeuroSuite .probe YAML with the channel geometry.

    `locations_xy` is (n_channels, 2) in µm.  fiber-kit's load_geometry()
    reads doc["probeFile"]["sites"]["geometry"] and nothing else here.
    """
    import yaml
    xy = np.asarray(locations_xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"locations_xy must be (n, 2); got {xy.shape}")
    doc = {
        "probeFile": {
            "sites": {
                "geometry": [[float(x), float(y)] for x, y in xy],
                "count_per_shank": int(xy.shape[0]),
            },
            "label": "ns3-sort-bench-materialised",
        },
    }
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def _channel_locations_or_default(recording, n_ch: int) -> np.ndarray:
    """Read `recording.get_channel_locations()` if available; fall back to
    a linear 25-µm-spaced tetrode layout (0, 25, ..., 25·(n-1)) so the
    probe file is always writable.  The default matches mountainsort5's
    fallback on PAIRED_CRCNS_HC1 (same 0/25/50/75-µm shape observed in
    its log) and is good enough for fiber-cpos to localise within one
    group — the sorter's own geometry handling uses the SI recording
    directly, this file only feeds the fiber-kit localiser.
    """
    try:
        locs = recording.get_channel_locations()
    except (AttributeError, Exception):               # noqa: BLE001
        locs = None
    locs = np.asarray(locs) if locs is not None else None
    if (locs is None or locs.ndim != 2 or locs.shape[0] != n_ch
            or locs.shape[1] < 2):
        y = 25.0 * np.arange(n_ch, dtype=float)
        return np.column_stack([np.zeros(n_ch, float), y])
    return np.ascontiguousarray(locs[:, :2].astype(float))


# ─── per-stage I/O wiring for the full fiber-kit chain ──────────────────────
# Full chain on a materialised SpikeInterface recording, cross-checked
# against each stage's argparse + input-resolution code:
#
#   stage            reads                           writes
#   ---------------  ------------------------------  -------------------------------
#   fiber-session    .res.<m>.<g>, .spk.<m>.<g>      .clu.<m>.<g>.fiber_session
#   fiber-realign    --clu <path>                    .clu.<m>.<g>              (canonical)
#                                                    updates .res / .spk / .fet
#   fiber-refine     --in-clu <path>                 .clu.<m>.<g>.refine
#   fiber-cpos       .clu.<m>.<g>.refine, .spk.<m>.<g>, probe   .cpos.<m>.<g>.refine.clusters.npz
#   fiber-peel       .clu.<m>.<g>.refine             .clu.<m>.<g>.refine (in-place)
#   fiber-intrachunk .clu.<m>.<g>.refine, .cpos|probe .clu.<m>.<g>.refine_intrachunk
#   fiber-link       .clu.<m>.<g>.refine,
#                    .cpos.<m>.<g>.refine.clusters.npz  (REQUIRED, no on-the-fly fallback)
#                                                   .clu.<m>.<g>.refine_linked
#
# Argparse surfaces for the method token (same token, different flag per
# stage; cross-checked against src/fiber_kit/fiber_<stage>.py):
#
#   fiber-session      --method <m>
#   fiber-realign      --variant <m>
#   fiber-refine       --out-method <m>
#   fiber-cpos         --spk-method, --clu-method, --out-method
#   fiber-peel         --cpos-method, --clu-method
#   fiber-intrachunk   --cpos-method, --clu-method
#   fiber-link         --cpos-method, --clu-method
#
# Two things the defaults get wrong on a materialised recording and we
# correct here:
#
#   1. fiber-refine's default --in-clu lookup is ``prefer_canonical()`` =
#      ``["", "stderiv", "D"]`` — it NEVER tries the "standard" variant,
#      so ``.clu.standard.<g>`` (what fiber-realign leaves) is invisible
#      to it and it silently starts a fresh fine sort.  We pass --in-clu
#      explicitly.
#   2. fiber-link requires a precomputed .cpos .clusters.npz (no on-the-
#      fly localizer fallback — unlike fiber-intrachunk).  So the chain
#      MUST include fiber-cpos between fiber-refine and fiber-link.
#
# The chain's final readback tag with this wiring is ``refine_linked``
# (fiber-link's ``--out-stage`` default is ``<clu-stage>_linked`` and
# clu-stage inherits cpos-stage=refine) — NOT the historical
# ``fiber_session_linked`` which a previous iteration tried to force.
#
# The `--clu` (fiber-realign) and `--in-clu` (fiber-refine) paths are
# session-specific, so they are added in `sort()` after the method flags.
def _stage_method_flags(method: str) -> dict[str, list[str]]:
    """Return {stage: [flag, value, ...]} for the method token injection
    across the full fiber-kit chain.  See the module-level table above
    for the complete input/output contract of each stage."""
    return {
        "fiber-session":    ["--method", method],
        "fiber-realign":    ["--variant", method],
        "fiber-refine":     ["--out-method", method],
        "fiber-cpos":       ["--spk-method",  method,
                              "--clu-method",  method,
                              "--out-method",  method,
                              "--clu-stage",   "refine",
                              "--out-stage",   "refine"],
        # fiber-peel / fiber-intrachunk / fiber-link read cpos-stage=refine
        # by default (fiber-refine's output), which is what we want now
        # that fiber-cpos has written .cpos.<m>.<g>.refine for them.
        "fiber-peel":       ["--cpos-method", method, "--clu-method", method],
        "fiber-intrachunk": ["--cpos-method", method, "--clu-method", method],
        "fiber-link":       ["--cpos-method", method, "--clu-method", method],
    }


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

    # Chain order matters: fiber-cpos sits BETWEEN fiber-refine and
    # fiber-peel because (a) it reads the refine-tagged .clu/.spk, and
    # (b) fiber-link requires the .cpos.<m>.<g>.refine.clusters.npz file
    # fiber-cpos produces (fiber-link has NO on-the-fly localizer
    # fallback, unlike fiber-intrachunk which does).
    _MODE_STAGES = {
        # `fiber_kit` mode runs the user's preferred 9-step plan embedded as
        # `pipeline:` in configs/fiber_kit.yaml — fiber-flag-artifacts →
        # session → realign → consensus(clean) → cpos → anchor-link → realign
        # → consensus(review) → rdr.  We don't orchestrate those stages
        # ourselves; the `fiber-pipeline` bash driver (installed with
        # fiber-kit) reads the embedded plan from $FK_CONFIG and dispatches
        # each step with the right flags.  The lone "fiber-pipeline"
        # sentinel below tells _run_stage to use that driver instead of the
        # per-stage invocation that fiber_kit_no_deconv still uses.
        "fiber_kit": ["fiber-pipeline"],
        # fiber_kit_no_deconv stays on the legacy per-stage chain: the
        # user explicitly asked to keep it as is.
        "fiber_kit_no_deconv": ["fiber-session", "fiber-realign", "fiber-refine",
                                 "fiber-cpos",
                                 "fiber-peel", "fiber-intrachunk", "fiber-link"],
        "fiber_kit_session": ["fiber-session"],
    }

    # Default `.clu.<variant>.<group>.<tag>` tag written by the LAST stage of
    # each mode's chain.
    #   fiber-session mode: fiber-session writes tag `fiber_session`.
    #   fiber_kit (pipeline): the last step in the embedded plan whose
    #                       `out:` is set is fiber-consensus(review),
    #                       which writes `exp_final` — the full-chain
    #                       output the harness scores by default.
    #   fiber_kit_no_deconv: legacy per-stage chain; fiber-link's default
    #                       --out-stage is "<clu-stage>_linked"; with
    #                       cpos-stage=refine (fiber-cpos's output), that
    #                       resolves to "refine_linked".
    _MODE_FINAL_TAG = {
        "fiber_kit":            "exp_final",
        "fiber_kit_no_deconv":  "refine_linked",
        "fiber_kit_session":    "fiber_session",
    }

    # Modes that dispatch through the `fiber-pipeline` bash driver (which
    # reads configs/fiber_kit.yaml's embedded `pipeline:` block + every
    # flat FK_* tuning knob) need the FULL baseline, not just the minimal
    # `fiber_kit:` section.  Every other mode runs the legacy per-stage
    # chain and consumes FK_* env vars at stage boundaries, where the
    # minimal baseline is enough.
    _USES_FIBER_PIPELINE: bool = False

    @classmethod
    def _write_fk_config(cls, tmp_dir: str,
                          user_cfg_path: Optional[str] = None,
                          auto_overrides: Optional[Mapping[str, Any]] = None,
                          ) -> str:
        """Write a per-session `fiber-kit.yaml` into `tmp_dir` and return
        its path.  The file is the MERGE of THREE sources:

          1. ``configs/fiber_kit.yaml`` baseline — either
             ``_FK_SESSION_BASELINE`` (legacy per-stage modes) or
             ``_FK_PIPELINE_BASELINE`` (fiber_kit mode, i.e. dispatch via
             fiber-pipeline — then the whole FK_* flat map + the embedded
             `pipeline:` block carry over so the driver sees the complete
             tuning + plan).  Load-bearing default in both:
             ``FK_SESSION_REALIGN: "off"``, because the end-of-run realign
             refuses to work on raw waveforms when its token asks for
             stderiv — we have no ``.spk.stderiv.<g>``.
          2. `auto_overrides` — per-session adjustments ns3sb computes
             from the recording itself (today: the
             channel-count-sensitive warp / position / primary-channel
             knobs; see ``_channel_count_fk_overrides``).  Merged above
             the baseline, below the user, so an explicit user value
             still wins.
          3. The user's own tuning, from `user_cfg_path` when given,
             otherwise from `$FK_CONFIG`, otherwise from
             ``./fiber-kit.yaml`` in CWD (the same search order
             fiber-kit's ``load_global_config`` uses before we override
             ``FK_CONFIG``).

        User keys WIN on conflict — our baseline fills gaps.  The quoted
        ``"off"`` is deliberate: YAML 1.1 reads bare ``off/on`` as booleans
        and fiber-kit's config loader then rejects them as the wrong type.

        Prior behaviour (patch 349966b) wrote ONLY the baseline, which,
        combined with the FK_CONFIG override, silently discarded any
        ``./fiber-kit.yaml`` tuning the user had — a regression this
        merge undoes.
        """
        import yaml as _yaml
        baseline = (_FK_PIPELINE_BASELINE if cls._USES_FIBER_PIPELINE
                    else _FK_SESSION_BASELINE)
        merged: dict = dict(baseline)
        if auto_overrides:
            merged.update(auto_overrides)       # auto > baseline

        user_path = (user_cfg_path
                      or os.environ.get("FK_CONFIG")
                      or os.path.join(os.getcwd(), "fiber-kit.yaml"))
        if user_path and os.path.isfile(user_path):
            try:
                with open(user_path) as f:
                    loaded = _yaml.safe_load(f) or {}
                if isinstance(loaded, dict):
                    merged.update(loaded)       # user wins on key conflict
            except (OSError, _yaml.YAMLError):
                # A broken user yaml should NOT kill the sort; fall back
                # to the baseline alone and leave a breadcrumb in the
                # on-disk file so the user can spot the ignore.
                merged.setdefault("# WARNING",
                                   f"ignored unreadable {user_path!r}")

        # Flat FK_* map: emit by hand so string values that LOOK like
        # YAML 1.1 booleans (off/on/yes/no/true/false) stay quoted while
        # keys remain bare — PyYAML's `default_style='"'` would quote both.
        _YAML11_BOOLS = {"off", "on", "yes", "no", "true", "false",
                          "y", "n", "t", "f"}

        def _emit(v: Any) -> str:
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, (int, float)):
                return repr(v)
            s = str(v)
            needs_quotes = (
                s == "" or s.strip() != s
                or s.lower() in _YAML11_BOOLS
                or any(c in s for c in ":#{}[]&*!|>'\"%@`,")
            )
            return f'"{s}"' if needs_quotes else s

        path = os.path.join(tmp_dir, "fiber-kit.yaml")
        with open(path, "w") as f:
            f.write(
                "# Written by ns3_sort_bench.adapters.fiber_kit when it "
                "materialised\n"
                "# this session from a SpikeInterface BaseRecording.\n"
                "# Merge of configs/fiber_kit.yaml `fiber_kit:` baseline +\n"
                f"# user tuning from {user_path!r}.  User keys WIN.\n"
                "# Keep quoted `off/on` strings — YAML 1.1 reads bare "
                "off/on as bools.\n"
            )
            for k, v in merged.items():
                if str(k).startswith("#"):
                    f.write(f"{k}: {v}\n")
                    continue
                # fiber-pipeline's embedded `pipeline:` block is a list of
                # dicts; our flat scalar emitter can't represent it, and
                # fiber-pipeline SPECIALLY-cases this one non-FK_ key in
                # its config reader.  Fall back to yaml.safe_dump for
                # non-scalar values (today: just `pipeline:`).
                if isinstance(v, (list, dict)):
                    f.write(_yaml.safe_dump({k: v}, sort_keys=False,
                                             default_flow_style=None))
                    continue
                f.write(f"{k}: {_emit(v)}\n")
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
                # contents survive for post-hoc inspection — but it IS
                # emptied before the sort starts (see _fresh_artifacts_dir).
                slug = _slugify(str(kwargs.get("corpus_name") or "sort"))
                tmp_dir = _fresh_artifacts_dir(fk_artifacts_dir, slug)
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

            # fiber-kit.yaml setup — always write a per-session merge of
            # the baseline (configs/fiber_kit.yaml `fiber_kit:` section) and
            # the user's own ./fiber-kit.yaml (or $FK_CONFIG, or an
            # explicit fk_config_path), with USER KEYS WINNING on conflict.
            # Pins FK_SESSION_REALIGN=off (no .spk.stderiv.<g> exists) and
            # keeps the user's CA1 tuning (FK_XCM_*, FK_CHUNKS, ...) intact.
            # On top of that, ship any channel-count-sensitive auto
            # overrides (DEPTH_GATE / LINK_POS_THR / WARP thresholds /
            # ALINK_PRIM_FRAC retuned for ≤4-channel probes) — these sit
            # between the baseline and the user's tuning, so a user who
            # has set them explicitly still wins.
            # Point FK_CONFIG at the merged file so every stage sees it.
            auto_overrides = _channel_count_fk_overrides(
                int(recording.get_num_channels())
            )
            fk_config = self._write_fk_config(
                tmp_dir,
                user_cfg_path=kwargs.get("fk_config_path"),
                auto_overrides=auto_overrides,
            )
            env["FK_CONFIG"] = fk_config

        # Inject the method token on auto-materialised runs so every stage
        # resolves our raw .spk.standard.<g> / .res.standard.<g> files
        # directly instead of falling back to a stderiv file that doesn't
        # exist.  `_STAGE_METHOD_FLAGS` (module-level) names the stage-
        # specific argparse flag each stage accepts for the method token;
        # a single ["--method", _NS3_METHOD] at every stage breaks
        # fiber-realign ("unrecognized arguments: --method standard" —
        # observed on the hc1_full run).
        #
        # Lab-session usage keeps whatever --method is baked into
        # extra_args / env (materialised is False — no injection).
        base_args = list(kwargs.get("extra_args", []))
        per_stage_args = {
            stage: list(base_args) for stage in self._MODE_STAGES[self._mode]
        }
        if materialised:
            for stage, flags in _stage_method_flags(_NS3_METHOD).items():
                if stage in per_stage_args:
                    per_stage_args[stage] = flags + per_stage_args[stage]
            # Session-specific input paths — two stages' input-clu lookup
            # defaults don't match the method-tagged names we produce:
            #
            #   fiber-realign  default --clu = legacy UNTAGGED
            #                  <base>.clu.<group>; we point it at
            #                  fiber-session's <...>.clu.standard.<g>.fiber_session
            #   fiber-refine   default --in-clu lookup is
            #                  prefer_canonical() = ["", "stderiv", "D"]
            #                  which doesn't include "standard".  Without
            #                  --in-clu, fiber-refine silently starts a
            #                  fresh fine sort instead of reading
            #                  fiber-realign's output (<...>.clu.standard.<g>).
            session_clu = _klusters_path(sess_base, "clu", group,
                                            method=_NS3_METHOD)
            fs_clu       = f"{session_clu}.fiber_session"
            realign_out  = session_clu                         # canonical, no tag
            if "fiber-realign" in per_stage_args:
                per_stage_args["fiber-realign"] = [
                    "--clu", fs_clu, *per_stage_args["fiber-realign"]
                ]
            if "fiber-refine" in per_stage_args:
                per_stage_args["fiber-refine"] = [
                    "--in-clu", realign_out,
                    *per_stage_args["fiber-refine"],
                ]

        with measure_runtime() as rt:
            for stage in self._MODE_STAGES[self._mode]:
                self._run_stage(stage, sess_base, group, env=env,
                                 extra_args=per_stage_args[stage])

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

        # Multi-stage readback.  For fiber_kit mode (fiber-pipeline-dispatched)
        # the embedded 9-step plan writes three scoreable intermediate .clu
        # outputs along the way — fiber_session (initial over-cluster),
        # fs_clean (after fiber-consensus quarantine), exp_xcm (after
        # fiber-anchor-link).  The user explicitly asked to score each of
        # them against the ground truth; we read them back here and attach
        # the per-stage SortedResult via extras["stage_results"] so
        # BenchmarkRun.go can score them like any other adapter output.
        # The exp_final stage is NOT re-emitted: that's already the
        # primary readback above, so re-scoring would just duplicate the
        # fiber_kit row.
        stage_results: Dict[str, "SortedResult"] = {}
        if self._mode == "fiber_kit" and materialised:
            for stage_tag in _FIBER_KIT_SCOREABLE_STAGES:
                stage_result = self._readback_stage_result(
                    sess_base, group, variant, stage_tag, recording,
                )
                if stage_result is not None:
                    stage_results[stage_tag] = stage_result

        extras = dict(mode=self._mode, electrode_group=group,
                       variant=variant, tag=tag, session_base=sess_base)
        if stage_results:
            extras["stage_results"] = stage_results
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
        recording.  The ``.fil`` is **bandpass-filtered 300–6000 Hz** before
        being written — ``ndm_detectspikes`` reads whatever is in ``.fil``
        and assumes it is already high-passed (the pipeline's split is
        ``ndm_hipass -> .fil -> ndm_detectspikes``; our SpikeInterface
        source has no equivalent step, so we bake the filter in here).

        The session yaml carries the usual ns3 description
        (``acquisitionSystem``, ``spikeDetection.channelGroups``) plus a
        ``programs:`` block whose ndm_detectspikes parameters are rewritten
        at this call to reflect the session's actual sampling rate:

          * ``refractoryPeriod``  = round(0.5 ms × sr)  — the Python port's
                                                          _DEFAULT_DEAD_TIME_MS
          * ``peakSearchLength``  = nSamples (snippet width)
          * ``duration``          = min(yaml default, half the recording)

        The baseline values in ``configs/fiber_kit.yaml`` are sensible for
        20 kHz; these overrides keep them correct across sampling rates.
        """
        import yaml
        n_samp = recording.get_num_samples(0)
        n_ch = recording.get_num_channels()
        sr = float(recording.get_sampling_frequency())
        base = os.path.join(tmp_dir, "sim_session")

        # Bandpass 300–6000 Hz BEFORE writing .fil.  Matches the Python
        # port's internal _bandpass(traces, sr, 300, 6000) that produced
        # the signal fiber-session's clustering was tuned against.  Without
        # this, ndm_detectspikes threshold-crosses on wideband including
        # LFP sweeps, which inflates detection counts and (on CA1 paired
        # recordings) collapses fiber-session's output to 1 cluster.
        traces = recording.get_traces(start_frame=0, end_frame=n_samp)
        traces_filt = _bandpass(np.asarray(traces, np.float64), sr)
        traces_i16 = np.clip(np.rint(traces_filt), -32768, 32767).astype(np.int16)
        traces_i16.tofile(base + ".fil")
        # fiber-session reads `.fil`; Klusters' `TracesProvider` is hardcoded
        # to read `<base>.dat` (src/klusters/src/klustersdoc.cpp line 745).
        # Hardlink the same bytes so both open the session without a
        # duplicate copy.  A fallback copy on OSError handles cross-fs tmp.
        _hardlink_or_copy(base + ".fil", base + ".dat")

        # Session-rate-aware overrides for the detect block.  Clamp the
        # noise window to <= half the recording so ndm_detectspikes
        # doesn't error on short sessions.
        noise_duration = min(
            float(_NDM_DETECT_DEFAULTS.get("duration", "10")),
            float(n_samp) / sr * 0.5,
        )
        refractory_samples  = max(1, int(round(0.5e-3 * sr)))      # 0.5 ms
        peak_search_samples = int(_NS_SAMPLES)                     # snippet width

        # Write a minimal .probe YAML next to the session.  fiber-intrachunk
        # needs the probe to on-the-fly-localise clusters when no
        # fiber-cpos table exists — observed failure on hc1_full:
        #   "[intrachunk] no fiber-cpos table and no probe geometry to
        #    compute one: run fiber-cpos first, or name a probe in
        #    <session>.yaml / pass --probe"
        # The session yaml's `probe:` key points at this file.
        probe_path = base + ".probe"
        _write_probe_file(probe_path,
                            _channel_locations_or_default(recording, n_ch))

        yml = {
            "parameters": {"version": "1.0", "creator": "ns3-sort-bench"},
            "acquisitionSystem": dict(nBits=16, nChannels=int(n_ch),
                                       voltageRange=20, amplification=1000,
                                       offset=0, samplingRate=sr),
            # Named probe — fiber-kit's cfg.get("probe") returns this path,
            # which fiber_localize.load_geometry then opens.  Keep it a
            # single string (fiber-kit also accepts a list for multi-shank
            # sessions; one shank → one path.)
            "probe": probe_path,
            "spikeDetection": {"channelGroups": [dict(
                channels=list(range(n_ch)),
                nSamples=_NS_SAMPLES, peakSampleIndex=_NS_PEAK, nFeatures=3,
            )]},
            # ndm_detectspikes / ndm_extractspikes / ndm_pca read their
            # parameters here.  The baseline comes from
            # configs/fiber_kit.yaml; we override only the knobs that must
            # follow the session (rate-dependent samples + short-session
            # noise window).
            "programs": _programs_yaml_section(overrides={
                "ndm_detectspikes": {
                    "refractoryPeriod":  str(refractory_samples),
                    "peakSearchLength":  str(peak_search_samples),
                    "duration":          f"{noise_duration:g}",
                },
            }),
        }
        with open(base + ".yaml", "w") as f:
            yaml.safe_dump(yml, f, sort_keys=False)
        return base

    def _run_stage(self, stage, sess_base, group, *, env, extra_args):
        if stage == "fiber-pipeline":
            # fiber-pipeline <elec> is the bash driver: takes the group as
            # its ONE positional arg, resolves the session from FK_DIR +
            # FK_SESS in env, and reads the embedded `pipeline:` block
            # from $FK_CONFIG — we already point FK_CONFIG at the merged
            # yaml in _materialise_recording.  --method tells every stage
            # which .spk/.clu variant to compose; since we auto-materialise
            # a `standard` .spk only, that's the tag we ship.  (If no
            # --method is given, fiber-pipeline falls back to inspecting
            # the .spk files on disk, which would resolve the same way,
            # but we pass it explicitly so the dispatch is self-describing
            # in the log file.)
            sess_dir, sess_name = os.path.split(sess_base)
            env = dict(env)
            env.setdefault("FK_DIR",  sess_dir or ".")
            env.setdefault("FK_SESS", sess_name)
            cmd = ["fiber-pipeline", str(group),
                   "--method", _NS3_METHOD, *list(extra_args)]
            log_path = f"{sess_base}.fiber-pipeline.log"
        else:
            cmd = [stage, sess_base, str(group), *list(extra_args)]
            log_path = f"{sess_base}.{stage}.log"
        with open(log_path, "w") as logf:
            r = subprocess.run(cmd, env=env, stdout=logf,
                                stderr=subprocess.STDOUT, check=False)
        if r.returncode != 0:
            tail = open(log_path).read()[-2000:]
            raise RuntimeError(f"{stage} failed (exit {r.returncode}); log tail:\n{tail}")

    def _readback_stage_result(self, sess_base: str, group: int,
                                variant: str, stage_tag: str,
                                recording) -> Optional["SortedResult"]:
        """Read back the `.clu.<variant>.<group>.<stage_tag>` file an
        intermediate plan step wrote, return it as a SortedResult tagged
        with adapter=`<mode>@<stage_tag>`, or None if the file isn't
        present.

        Absent files are not an error — fiber-pipeline's steps are
        conditional on earlier ones producing a .clu with their own `in:`
        tag, and the user may run a plan that skips some of them.  The
        caller just omits the missing stage from stage_results.
        """
        from ..ns3.extractors import NS3SortingExtractor
        clu_path = _klusters_path(sess_base, "clu", group) + f".{stage_tag}"
        if not os.path.isfile(clu_path):
            return None
        try:
            sort_ex = NS3SortingExtractor(
                sess_base + ".yaml", electrode_group=group,
                variant=variant, tag=stage_tag,
            )
            st = {int(u): np.asarray(sort_ex.get_unit_spike_train(u),
                                       np.int64)
                  for u in sort_ex.get_unit_ids()}
        except Exception:
            # A malformed intermediate .clu shouldn't kill the whole sort —
            # the primary result already returned fine.  Drop this stage.
            return None
        return SortedResult(
            unit_ids=sorted(st), spike_times=st,
            sampling_rate=recording.get_sampling_frequency(),
            adapter=f"{self._mode}@{stage_tag}",
            corpus_item=sess_base,          # overridden by BenchmarkRun.go
            adapter_version=self._version,
            channels=list(recording.get_channel_ids()),
            # Intermediate .clu files don't have their own wall-time —
            # they were written as side effects of the single
            # fiber-pipeline invocation.  Leave at 0 so a reader doesn't
            # misread the stage row as a separately-timed run.
            wall_time_s=0.0, peak_rss_mb=0.0,
            extras=dict(mode=self._mode, stage_tag=stage_tag,
                         electrode_group=group, variant=variant,
                         session_base=sess_base),
        )


class _Full(_FiberKitBase):
    name = "fiber_kit"; _mode = "fiber_kit"
    _env_overrides = {"FK_SESSION_DECONV": "1"}
    _contract_obj = _contract_for("fiber_kit", with_deconv=True)
    # Dispatch via the fiber-pipeline bash driver → _write_fk_config
    # ships the full baseline (pipeline: + every flat FK_*).
    _USES_FIBER_PIPELINE = True


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
