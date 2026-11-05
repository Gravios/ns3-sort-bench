"""Pröpper 2015 §3.4 BOTM reference — in-tree, no third-party deps.

The zero-extension baseline every fiber_deconv claim is measured against.
Implements the thesis exactly: static block-Toeplitz noise cov, integer
templates, no amplitude prior, SIC-B subtraction, uniform firing-rate prior,
hard refractory.

Does NOT implement template initialisation or drift tracking — the thesis
expects an initial sort as input.  For the harness we bootstrap with a
negative-threshold detector + KMeans on PCA features so BOTM can run end-
to-end on synthetic corpora.

Backends
--------
The SIC-B hot loop recomputes a (T templates × out_len) discriminant matrix
per iteration (up to 500 iters) — on long recordings this dominates the
runtime (user-reported 270 s on a 300 s HC1 recording, 20 kHz × 4 ch).  Two
backends compute that matrix:

    numpy     per-channel np.correlate, outer loop over templates.
              Ground truth; zero dependencies.
    torch     one batched conv1d call over all templates and channels at
              once; runs on CUDA when a GPU is visible, else CPU-BLAS
              (which already has OpenMP parallelism underneath).

Autodetect picks torch+CUDA when available, torch+CPU when only torch is
available, else numpy.  Override via ``adapter.sort(botm_backend=...)`` or
env ``NS3SB_BOTM_BACKEND=cuda|torch|numpy``.  The numpy backend is bit-
exact within floating-point tolerance to the pre-refactor thesis-form
implementation; the torch backend is within rtol=1e-4 of it (its own
accumulation order).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional

import numpy as np

from .._base import (
    SortedResult, SorterAdapter, _register, adapter_registry, measure_runtime,
)
from ..configs import load_adapter_config
from ..contracts import (
    Contract, RecordingReqs, OutputSchema, Capabilities, ResourceEstimates,
)


def _config_defaults() -> dict:
    """Pull the kwarg defaults out of ``configs/botm_ref.yaml``.

    The file is flat scalars at the top level (n_units_hint,
    detect_threshold_mad, template_samples, peak_sample, botm_backend).
    Missing keys fall back to the hard-coded defaults in sort() /
    _bootstrap_templates — this is a strict overlay.
    """
    cfg = load_adapter_config("botm_ref")
    if not isinstance(cfg, Mapping):
        return {}
    allowed = {"n_units_hint", "detect_threshold_mad",
                "template_samples", "peak_sample", "botm_backend"}
    return {k: v for k, v in cfg.items() if k in allowed}


_BOTM_REF_CFG = _config_defaults()


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
        requires_gpu=False,         # GPU opportunistic, not required
        requires_torch=False,       # torch is optional; numpy backend works
        supports_stderiv_input=True,
        supports_shared_detection=True,
        reports_templates=True,
    ),
    resources=ResourceEstimates(
        min_ram_mb=256, min_cpu_cores=1,
        gpu_vram_mb=512,            # modest; one conv1d over (T, n_ch, L)
        estimated_sec_per_minute_of_recording=20.0,
    ),
    known_limitations=(
        "No drift adaptation (thesis §3.4.1 assumption 1).",
        "Templates must be stationary over the whole segment.",
        "Bootstrap detector is naive negative-threshold + KMeans — not meant"
        " for curation, only to let BOTM demonstrate its sorting step.",
        "Static block-Toeplitz noise cov; mild non-stationarity hurts.",
        "Discriminant hot path uses torch.conv1d (CUDA when available, else"
        " CPU-BLAS) when torch is installed, falling back to pure numpy.  "
        "Override via adapter.sort(botm_backend='numpy'|'torch'|'cuda') or"
        " $NS3SB_BOTM_BACKEND.  Results across backends match within"
        " float32 accumulation tolerance (rtol ≈ 1e-4 vs the numpy ground"
        " truth).",
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

        # Resolve each knob in precedence order: caller kwarg >
        # configs/botm_ref.yaml > hard-coded default.  `.get(k, cfg_v)`
        # puts the caller first and the config next.  Env still wins
        # for the backend only (kept for scripted sweeps).
        def _cfg(key, default):
            if key in kwargs:
                return kwargs[key]
            return _BOTM_REF_CFG.get(key, default)

        backend_pref = _cfg("botm_backend",
                              os.environ.get("NS3SB_BOTM_BACKEND") or "auto")
        if os.environ.get("NS3SB_BOTM_BACKEND"):
            backend_pref = os.environ["NS3SB_BOTM_BACKEND"]
        discriminant_fn, backend_used = _pick_discriminant_backend(backend_pref)

        with measure_runtime() as rt:
            traces = recording.get_traces(start_frame=0,
                                           end_frame=recording.get_num_samples(0))
            traces = np.asarray(traces, dtype=np.float64)
            n_samp, n_ch = traces.shape
            sr = recording.get_sampling_frequency()

            tpl_samples = int(_cfg("template_samples", 32))
            peak        = int(_cfg("peak_sample", 16))
            n_units_hint = int(_cfg("n_units_hint", 3))
            detect_thr_mad = float(_cfg("detect_threshold_mad", -4.0))

            init_times = None
            templates = kwargs.get("initial_templates", None)
            if templates is None:
                templates, init_times, _init_labels = _bootstrap_templates(
                    traces, tpl_samples=tpl_samples, peak=peak,
                    n_units_hint=n_units_hint,
                    detect_threshold_mad=detect_thr_mad,
                )
            else:
                templates = np.asarray(templates, np.float64)

            C = _estimate_noise_cov(traces, templates.shape[1], peak,
                                     spike_times=init_times,
                                     n_windows=200)
            f, c, _ = _botm_filters(templates, C)
            placements = _botm_sic_b(
                traces, templates, f, c,
                tpl_samples=templates.shape[1], peak=peak,
                refrac=int(0.5e-3 * sr),
                discriminant_fn=discriminant_fn,
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
                        n_initial_units=int(templates.shape[0]),
                        backend=backend_used),
        )


# ─── helpers (thesis-form, plain numpy) ──────────────────────────────────────
def _bootstrap_templates(traces, *, tpl_samples=32, peak=16, n_units_hint=3,
                         detect_threshold_mad: float = -4.0):
    """Negative-threshold detector + KMeans on PCA features → initial
    templates for SIC-B.  `detect_threshold_mad` is a multiple of the
    per-channel MAD (median absolute deviation × 1.4826) — expected
    negative, since BOTM detects on negative-going spikes.  Looser
    (closer to 0, e.g. -3.0) picks up more candidates; tighter (more
    negative, e.g. -5.0) keeps only the bigger spikes.
    """
    n_samp, n_ch = traces.shape
    mad = np.median(np.abs(traces - np.median(traces, axis=0)), axis=0) * 1.4826
    peaks = []
    ch = int(np.argmax(np.ptp(traces, axis=0)))
    thr = float(detect_threshold_mad) * mad[ch]
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


def _botm_sic_b(traces, templates, f, c, *, tpl_samples, peak, refrac,
                 discriminant_fn=None):
    """Batched SIC-B.  Picks the global-best (template, time) each iteration
    across ALL templates in one argmax over (T, out_len), subtracts the
    template at that time, and repeats.

    `discriminant_fn(r, f, c, tpl_samples) -> (T, out_len) ndarray` is the
    backend-specific hot kernel; defaults to the numpy backend.
    """
    if discriminant_fn is None:
        discriminant_fn = _discriminant_all_numpy
    n_samp, n_ch = traces.shape
    T = templates.shape[0]
    f_flat = f.reshape(T, -1)
    r = traces.copy()
    placements: list[tuple[int, int]] = []
    noise_est = _estimate_noise_threshold(r, templates, f_flat, tpl_samples, c)
    out_len = n_samp - tpl_samples + 1
    if out_len <= 0:
        return placements
    for _ in range(500):
        D = discriminant_fn(r, f, c, tpl_samples)         # (T, out_len)
        # Mask each already-placed spike's refractory window on its own
        # template.  Vectorised over placements for one template at a time
        # (list is short — 500 max).
        for p_t, p_u in placements:
            lo = max(0, p_t - refrac)
            hi = min(out_len, p_t + refrac + 1)
            D[p_u, lo:hi] = -np.inf
        best_flat = int(np.argmax(D))
        best_u, best_t = divmod(best_flat, out_len)
        best_v = float(D[best_u, best_t])
        if best_v < noise_est:
            break
        lo = max(0, best_t - peak)
        hi = lo + tpl_samples
        if hi > n_samp:
            continue
        r[lo:hi, :] -= templates[best_u]
        placements.append((int(best_t), int(best_u)))
    return placements


# ─── discriminant backends ───────────────────────────────────────────────────
def _discriminant_all_numpy(residual, f, c, tpl_samples):
    """CPU-numpy fallback.  Returns (T, out_len).  Mirrors the original
    inner double-loop byte-for-byte modulo accumulation order."""
    n_samp, n_ch = residual.shape
    T = f.shape[0]
    out_len = n_samp - tpl_samples + 1
    if out_len <= 0:
        return np.full((T, 0), -np.inf)
    D = np.zeros((T, out_len))
    for u in range(T):
        acc = np.zeros(out_len)
        for ch in range(n_ch):
            acc += np.correlate(residual[:, ch], f[u, :, ch], mode="valid")
        D[u] = acc + c[u]
    return D


def _discriminant_all_torch(residual, f, c, tpl_samples, *, device):
    """Single batched conv1d: one call computes the discriminant for every
    template × every time at once.  `conv1d` is cross-correlation applied to
    a flipped kernel — fiber-kit and scipy use the same trick — so the
    output equals `sum_ch correlate(r[:, ch], f[u, :, ch], 'valid')`.

    Layout notes:
      residual: (n_samp, n_ch) → torch (1, n_ch, n_samp)
      f:        (T, tpl_samples, n_ch) → torch (T, n_ch, tpl_samples) flipped
      conv1d(1, n_ch, n_samp) × kernel(T, n_ch, K) → (1, T, out_len)
    """
    import torch                                           # local to keep optional
    n_samp, n_ch = residual.shape
    T = f.shape[0]
    out_len = n_samp - tpl_samples + 1
    if out_len <= 0:
        return np.full((T, 0), -np.inf)
    R = torch.from_numpy(np.ascontiguousarray(residual.T, np.float32))
    R = R.to(device=device)[None]                          # (1, n_ch, n_samp)
    # f: (T, tpl_samples, n_ch) → (T, n_ch, tpl_samples); flip last dim for
    # cross-correlation.
    F = torch.from_numpy(np.ascontiguousarray(
        f.transpose(0, 2, 1), np.float32)).to(device=device)
    F = torch.flip(F, dims=(-1,))
    with torch.inference_mode():
        out = torch.nn.functional.conv1d(R, F).squeeze(0)  # (T, out_len)
        out = out + torch.from_numpy(np.asarray(c, np.float32)) \
                        .to(device=device)[:, None]
    return out.detach().to("cpu", dtype=torch.float64).numpy()


def _pick_discriminant_backend(pref: str) -> tuple[Callable, str]:
    """Resolve the preferred backend into (discriminant_fn, name).

    pref ∈ {"auto", "cuda", "torch", "torch-cpu", "numpy"}.

      auto       cuda if torch+gpu, else torch-cpu if torch, else numpy
      cuda       require torch + CUDA; raise if unavailable
      torch      same as cuda but allow CPU-only torch
      torch-cpu  force torch on CPU even if a GPU is present
      numpy      the pure-numpy fallback
    """
    pref = (pref or "auto").lower()
    want_torch = pref in ("auto", "cuda", "torch", "torch-cpu")
    if want_torch:
        try:
            import torch                                   # noqa: F401
        except ImportError:
            if pref in ("cuda", "torch", "torch-cpu"):
                raise RuntimeError(
                    f"botm_backend={pref!r} but torch is not installed"
                )
            return _discriminant_all_numpy, "numpy"
        # torch is importable — now pick device
        if pref == "torch-cpu":
            device = "cpu"
        elif pref == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "botm_backend='cuda' but torch.cuda.is_available() is False"
                )
            device = "cuda"
        else:                                              # auto or torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        fn = lambda r, f, c, k, _d=device: _discriminant_all_torch(
            r, f, c, k, device=_d
        )
        return fn, f"torch:{device}"
    return _discriminant_all_numpy, "numpy"


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
