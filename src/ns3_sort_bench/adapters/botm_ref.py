"""Pröpper 2015 §3.4 BOTM reference — in-tree, no third-party deps.

The zero-extension baseline every fiber_deconv claim is measured against.
Implements the thesis exactly: static block-Toeplitz noise cov, integer
templates, no amplitude prior, SIC-B subtraction, uniform firing-rate prior,
hard refractory.

Does NOT implement template initialisation or drift tracking — the thesis
expects an initial sort as input.  For the harness we bootstrap with a
negative-threshold detector + KMeans on PCA features so BOTM can run end-
to-end on synthetic corpora.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from .._base import (
    SortedResult, SorterAdapter, _register, adapter_registry, measure_runtime,
)
from ..contracts import (
    Contract, RecordingReqs, OutputSchema, Capabilities, ResourceEstimates,
)


CONTRACT = Contract(
    adapter_name="botm_ref",
    upstream_url="https://doi.org/10.14279/depositonce-4891",   # Pröpper 2015 TU Berlin
    upstream_version_pin="in-tree/0.1",
    yaml_fields_required=(
        "acquisitionSystem.samplingRate",
        "acquisitionSystem.nChannels",
    ),
    yaml_fields_optional=(
        "spikeDetection.channelGroups",
        "acquisitionSystem.nBits",
        "acquisitionSystem.voltageRange",
    ),
    recording=RecordingReqs(
        min_channels=1, max_channels=64,
        min_duration_s=0.1,
        dtype=("int16", "float32", "float64"),
        probe_geometry_required=False,
        signal_space="raw",
    ),
    output=OutputSchema(
        unit_id_offset=2,         # offset so 0/1 stay reserved
        includes_reserved=False,
        templates_emitted=True,
        per_group_or_whole="per_group",
    ),
    capabilities=Capabilities(
        drift_aware=False,          # thesis assumes templates constant
        overlap_resolution="sic",   # SIC-B
        requires_gpu=False,
        requires_torch=False,
        supports_stderiv_input=True,
        supports_shared_detection=True,
        reports_templates=True,
    ),
    resources=ResourceEstimates(
        min_ram_mb=256, min_cpu_cores=1,
        gpu_vram_mb=None,
        estimated_sec_per_minute_of_recording=20.0,
    ),
    known_limitations=(
        "No drift adaptation (thesis §3.4.1 assumption 1).",
        "Templates must be stationary over the whole segment.",
        "Bootstrap detector is naive negative-threshold + KMeans — not meant"
        " for curation, only to let BOTM demonstrate its sorting step.",
        "Static block-Toeplitz noise cov; mild non-stationarity hurts.",
    ),
    notes=(
        "The MEASUREMENT BASELINE for every fiber_deconv extension claim.  "
        "If an extension doesn't beat this reference on measurable metrics, "
        "the extension doesn't justify its complexity."
    ),
)


class BOTMReferenceAdapter(SorterAdapter):
    name = "botm_ref"
    _version = "pröpper-2015/0.1"

    @property
    def contract(self) -> Contract: return CONTRACT

    @property
    def version(self) -> str: return self._version

    def available(self): return True, ""

    def sort(self, recording, *, ns3_yaml=None,
             ground_truth_res=None, **kwargs) -> SortedResult:
        from spikeinterface.core import BaseRecording
        if not isinstance(recording, BaseRecording):
            raise TypeError("botm_ref needs a SpikeInterface BaseRecording")

        with measure_runtime() as rt:
            traces = recording.get_traces(start_frame=0,
                                           end_frame=recording.get_num_samples(0))
            traces = np.asarray(traces, dtype=np.float64)
            n_samp, n_ch = traces.shape
            sr = recording.get_sampling_frequency()

            tpl_samples = int(kwargs.get("template_samples", 32))
            peak = int(kwargs.get("peak_sample", 16))
            n_units_hint = int(kwargs.get("n_units_hint", 3))

            templates = kwargs.get("initial_templates", None)
            if templates is None:
                templates, init_times, init_labels = _bootstrap_templates(
                    traces, tpl_samples=tpl_samples, peak=peak,
                    n_units_hint=n_units_hint,
                )
            else:
                templates = np.asarray(templates, np.float64)

            C = _estimate_noise_cov(traces, templates.shape[1], peak,
                                     spike_times=init_times if 'init_times' in dir() else None,
                                     n_windows=200)
            f, c, _ = _botm_filters(templates, C)
            placements = _botm_sic_b(
                traces, templates, f, c,
                tpl_samples=templates.shape[1], peak=peak,
                refrac=int(0.5e-3 * sr),
            )

        unit_ids = sorted(set(p[1] for p in placements))
        unit_ids_offset = [u + 2 for u in unit_ids]
        spike_times = {u + 2: np.array(
            [p[0] for p in placements if (p[1] + 2) == (u + 2)], np.int64)
            for u in unit_ids}

        return SortedResult(
            unit_ids=unit_ids_offset, spike_times=spike_times,
            sampling_rate=sr, adapter=self.name,
            corpus_item=str(recording.get_annotation("name", "botm_ref_run")),
            adapter_version=self._version,
            wall_time_s=rt["wall_time_s"], peak_rss_mb=rt["peak_rss_mb"],
            extras=dict(sic_variant="B",
                        n_initial_units=int(templates.shape[0])),
        )


# ─── helpers (thesis-form, plain numpy) ──────────────────────────────────────
def _bootstrap_templates(traces, *, tpl_samples=32, peak=16, n_units_hint=3):
    n_samp, n_ch = traces.shape
    mad = np.median(np.abs(traces - np.median(traces, axis=0)), axis=0) * 1.4826
    peaks = []
    ch = int(np.argmax(np.ptp(traces, axis=0)))
    thr = -4.0 * mad[ch]
    for i in range(peak, n_samp - tpl_samples + peak):
        if traces[i, ch] < thr and traces[i, ch] <= traces[i - 1, ch] \
                                 and traces[i, ch] <= traces[i + 1, ch]:
            if not peaks or i - peaks[-1] > tpl_samples:
                peaks.append(i)
    peaks = np.asarray(peaks, np.int64)
    if peaks.size == 0:
        return (np.zeros((1, tpl_samples, n_ch), np.float64),
                np.array([], np.int64), np.array([], np.int64))

    snips = np.stack([traces[p - peak:p - peak + tpl_samples, :] for p in peaks])
    flat = snips.reshape(len(snips), -1)
    mean = flat.mean(axis=0)
    _, _, Vt = np.linalg.svd(flat - mean, full_matrices=False)
    feats = (flat - mean) @ Vt[:min(6, Vt.shape[0])].T
    labels = _kmeans(feats, k=n_units_hint, seed=0)
    templates = np.stack([snips[labels == u].mean(axis=0) if (labels == u).any()
                          else np.zeros_like(snips[0])
                          for u in range(n_units_hint)])
    return templates, peaks, labels


def _kmeans(x, k, *, seed=0, iters=20):
    rng = np.random.default_rng(seed)
    n = len(x)
    if k >= n: return np.arange(n) % k
    centres = x[rng.choice(n, k, replace=False)]
    for _ in range(iters):
        d = np.linalg.norm(x[:, None, :] - centres[None, :, :], axis=2)
        lab = d.argmin(axis=1)
        new = np.stack([x[lab == i].mean(axis=0) if (lab == i).any() else centres[i]
                        for i in range(k)])
        if np.allclose(new, centres): break
        centres = new
    return lab


def _estimate_noise_cov(traces, tpl_samples, peak, *, spike_times=None,
                         n_windows=200, seed=1):
    rng = np.random.default_rng(seed)
    n_samp, n_ch = traces.shape
    guard = 2 * tpl_samples
    forbidden = np.zeros(n_samp, bool)
    if spike_times is not None:
        for t in spike_times:
            forbidden[max(0, int(t) - guard):min(n_samp, int(t) + guard)] = True

    cands = np.where(~forbidden)[0]
    cands = cands[(cands >= 0) & (cands < n_samp - tpl_samples)]
    starts = cands if cands.size < n_windows else rng.choice(cands, n_windows, replace=False)
    wins = np.stack([traces[s:s + tpl_samples] for s in starts], axis=0)
    flat = wins.reshape(len(wins), -1)
    C = np.cov(flat.T)
    tr = np.trace(C) / C.shape[0]
    C += 1e-3 * tr * np.eye(C.shape[0])
    return C


def _botm_filters(templates, C):
    T, n_sam, n_ch = templates.shape
    tpl_flat = templates.reshape(T, -1)
    f_flat = np.linalg.solve(C, tpl_flat.T).T
    c = -0.5 * np.einsum("ij,ij->i", tpl_flat, f_flat)
    return f_flat.reshape(T, n_sam, n_ch), c, 0.0


def _botm_sic_b(traces, templates, f, c, *, tpl_samples, peak, refrac):
    n_samp, n_ch = traces.shape
    T = templates.shape[0]
    f_flat = f.reshape(T, -1)
    r = traces.copy()
    placements = []
    noise_est = _estimate_noise_threshold(r, templates, f_flat, tpl_samples, c)
    for _ in range(500):
        best_t, best_u, best_v = -1, -1, -np.inf
        for u in range(T):
            d_u = _discriminant_series(r, f[u], tpl_samples, c[u])
            for (p_t, p_u) in placements:
                if p_u == u:
                    lo = max(0, p_t - refrac); hi = min(len(d_u), p_t + refrac + 1)
                    d_u[lo:hi] = -np.inf
            idx = int(np.argmax(d_u))
            v = float(d_u[idx])
            if v > best_v:
                best_v = v; best_t = idx; best_u = u
        if best_v < noise_est: break
        lo = max(0, best_t - peak); hi = lo + tpl_samples
        if hi > n_samp: continue
        r[lo:hi, :] -= templates[best_u]
        placements.append((best_t, best_u))
    return placements


def _discriminant_series(residual, f_u, tpl_samples, c_u):
    n_samp, n_ch = residual.shape
    out_len = n_samp - tpl_samples + 1
    if out_len <= 0: return np.full(0, -np.inf)
    acc = np.zeros(out_len)
    for ch in range(n_ch):
        acc += np.correlate(residual[:, ch], f_u[:, ch], mode="valid")
    acc += c_u
    return acc


def _estimate_noise_threshold(traces, templates, f_flat, tpl_samples, c):
    rng = np.random.default_rng(2)
    n_samp = traces.shape[0]
    starts = rng.integers(0, max(1, n_samp - tpl_samples), size=100)
    vals = []
    for s in starts:
        w = traces[s:s + tpl_samples].ravel()
        for u in range(len(templates)):
            vals.append(float(w @ f_flat[u]) + c[u])
    return float(np.mean(vals) + 4.0 * np.std(vals))


_register(adapter_registry, BOTMReferenceAdapter(), "botm_ref")
