"""Pröpper 2015 §3.4.6-style overlap simulator.

Takes a set of clean per-unit templates + a target recording length + a
desired overlap-pair distribution and emits:

- a synthetic recording: templates placed at ground-truth sample indices
  plus Gaussian noise with configurable covariance (default: AR(1) per
  channel + spatial correlation estimated from the templates themselves)
- a `Corpus` wrapping the recording + a `SortedResult` ground truth that
  carries, in `extras["overlap_pairs"]`, the list of
  `(t_1, unit_1, t_2, unit_2)` tuples the Pröpper scorer needs.

The emitted recording wraps a plain in-memory numpy array behind a
SpikeInterface `NumpyRecording`, so it is drop-in compatible with every
SI sorter the harness runs.

For a quick-start the function also supports a `synthetic_templates` mode
that generates triangular/biphasic templates from scratch (one shape per
unit, varying amplitude), useful when real templates aren't available —
Pröpper's §3.4.6 Figure 3.6 uses the same simplification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# NumpyRecording import is deferred so this module imports in a bare env
def _numpy_recording(traces: np.ndarray, sampling_rate: float):
    from spikeinterface.core import NumpyRecording
    rec = NumpyRecording([traces], sampling_frequency=sampling_rate)
    # Dummy probe so SI's drift-aware sorters have geometry to look at
    n_ch = traces.shape[1]
    locs = np.column_stack([np.zeros(n_ch, np.float32),
                            20.0 * np.arange(n_ch, dtype=np.float32)])
    if hasattr(rec, "set_dummy_probe_from_locations"):
        rec.set_dummy_probe_from_locations(locs)
    else:
        rec.set_channel_locations(locs)
    return rec


# ─── template generation ─────────────────────────────────────────────────────
def _synth_triangular_template(n_samples: int, n_channels: int, *,
                                peak_sample: int, peak_amp: float,
                                decay_channels: float = 2.0,
                                seed: int = 0) -> np.ndarray:
    """Toy biphasic template on `n_channels` channels, peak on the middle
    channel, spatial decay with σ = `decay_channels`.  Returns
    (n_samples, n_channels) float64 — matches `.spk` row shape."""
    t = np.arange(n_samples)
    # Biphasic: triangular trough at peak_sample, half-amp bump 1 ms later
    shape = np.zeros(n_samples, np.float64)
    half = max(1, n_samples // 16)
    shape[max(0, peak_sample - half):peak_sample + 1] = \
        -np.linspace(0, 1, half + 1)[:peak_sample + 1 - max(0, peak_sample - half)]
    shape[peak_sample:peak_sample + 2 * half] = \
        np.concatenate([
            -np.linspace(1, 0, half + 1)[:min(half + 1, 2 * half)],
            np.linspace(0, 0.5, half)[:max(0, 2 * half - half - 1)],
        ])[:min(2 * half, n_samples - peak_sample)]
    shape *= peak_amp

    ch_center = (n_channels - 1) / 2.0
    ch = np.arange(n_channels)
    spatial = np.exp(-((ch - ch_center) ** 2) / (2 * decay_channels ** 2))
    rng = np.random.default_rng(seed)
    spatial = spatial * (1.0 + 0.05 * rng.normal(size=n_channels))   # mild per-channel jitter
    return shape[:, None] * spatial[None, :]


def _ar1_noise(n_samples: int, n_channels: int, *, sigma: float = 20.0,
               phi: float = 0.5, spatial_corr: float = 0.3,
               seed: int = 1) -> np.ndarray:
    """Colored Gaussian noise: AR(1) per channel + spatial correlation."""
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=sigma, size=(n_samples, n_channels)).astype(np.float64)
    # Spatial correlation: smooth channels by a tridiagonal weight.
    if n_channels > 1:
        kernel = np.array([spatial_corr, 1.0 - 2 * spatial_corr, spatial_corr])
        from scipy.signal import convolve
        try:
            e = np.stack([convolve(e[:, c], kernel, mode="same")
                          for c in range(n_channels)], axis=1)
        except ImportError:
            # Trivial fallback — un-correlate across channels
            pass
    # AR(1) in time
    for t in range(1, n_samples):
        e[t] += phi * e[t - 1]
    return e


# ─── placement + ground truth ────────────────────────────────────────────────
@dataclass(frozen=True)
class _PropperConfig:
    n_samples: int
    n_channels: int
    sampling_rate: float
    n_units: int
    rate_hz: float             # mean firing rate per unit
    overlap_fraction: float    # fraction of spikes that participate in an overlap
    offset_distribution: str   # "uniform_small" | "pröpper_fig3_7"
    max_offset_samples: int
    peak_sample: int
    template_samples: int
    noise_sigma: float
    seed: int


def _gen_poisson_spikes(cfg: _PropperConfig, unit_idx: int) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + 100 * unit_idx)
    mean_isi = cfg.sampling_rate / cfg.rate_hz
    isi = rng.exponential(mean_isi, size=int(2 * cfg.n_samples / mean_isi))
    t = np.cumsum(isi).astype(np.int64)
    t = t[(t >= cfg.template_samples) & (t < cfg.n_samples - cfg.template_samples)]
    return t


def _choose_overlap_pairs(all_times, cfg: _PropperConfig):
    """Pick which spikes become overlap participants and at which offsets.
    Returns (overlap_set, overlap_pairs) where `overlap_set` is a set of
    (unit, t) tuples that got moved, and overlap_pairs is the ground-truth
    annotation list the scorer consumes."""
    rng = np.random.default_rng(cfg.seed + 7)
    pool = [(u, int(t)) for u, times in enumerate(all_times) for t in times]
    rng.shuffle(pool)
    n_want = int(cfg.overlap_fraction * len(pool) / 2) * 2
    pairs = []
    moved = set()
    i = 0
    while len(pairs) < n_want // 2 and i + 1 < len(pool):
        u1, t1 = pool[i]
        u2, t2 = pool[i + 1]
        if u1 == u2:
            i += 1
            continue
        # Choose offset
        if cfg.offset_distribution == "uniform_small":
            offset = rng.integers(-cfg.max_offset_samples, cfg.max_offset_samples + 1)
        else:
            # §3.4.7 Fig 3.7 shape: heavier-tailed, mode at zero
            offset = int(rng.normal(0, cfg.max_offset_samples / 2))
            offset = int(np.clip(offset, -cfg.max_offset_samples, cfg.max_offset_samples))

        new_t2 = int(t1 + offset)
        if abs(offset) < 1 or new_t2 < cfg.template_samples or new_t2 >= cfg.n_samples - cfg.template_samples:
            i += 2; continue
        pairs.append((t1, u1, new_t2, u2))
        moved.add((u1, t1)); moved.add((u2, t2))
        # Note: we EDIT the second time; the caller is responsible for
        # removing the original (u2, t2) and inserting (u2, new_t2).
        i += 2
    return moved, pairs


def make_propper_overlap_corpus(
    *, n_samples: int = 5 * 32552,
    n_channels: int = 4,
    sampling_rate: float = 32552.0,
    n_units: int = 3,
    rate_hz: float = 10.0,
    overlap_fraction: float = 0.2,
    offset_distribution: str = "pröpper_fig3_7",
    max_offset_samples: int = 32,
    peak_sample: int = 16,
    template_samples: int = 32,
    noise_sigma: float = 15.0,
    seed: int = 0,
    name: str = "propper_sim",
    description: str = "Pröpper §3.4.6 overlap simulator",
):
    """Build a synthetic recording + ground-truth Corpus with controlled
    overlap-pair annotations.

    Default settings reproduce something close to §3.4.6 Figure 3.6: 3
    units at ~10 Hz on 4 channels for 5 seconds, 20 % of spikes made
    overlapping, offsets drawn from a Fig 3.7-style distribution.

    Returns a `Corpus` import-compatible with `BenchmarkRun.add_corpus`.
    """
    from .._base import Corpus, SortedResult

    cfg = _PropperConfig(
        n_samples=int(n_samples), n_channels=int(n_channels),
        sampling_rate=float(sampling_rate), n_units=int(n_units),
        rate_hz=float(rate_hz), overlap_fraction=float(overlap_fraction),
        offset_distribution=offset_distribution,
        max_offset_samples=int(max_offset_samples), peak_sample=int(peak_sample),
        template_samples=int(template_samples), noise_sigma=float(noise_sigma),
        seed=int(seed),
    )

    # 1. Templates (one per unit, amplitudes 60/100/140 ADU)
    amps = np.linspace(60, 140, cfg.n_units)
    templates = [
        _synth_triangular_template(
            cfg.template_samples, cfg.n_channels,
            peak_sample=cfg.peak_sample, peak_amp=a, seed=cfg.seed + i,
        )
        for i, a in enumerate(amps)
    ]

    # 2. Spike trains (homogeneous Poisson per unit)
    all_times = [_gen_poisson_spikes(cfg, u) for u in range(cfg.n_units)]

    # 3. Choose overlap pairs; move the second participant of each pair
    moved, pairs = _choose_overlap_pairs(all_times, cfg)
    # Apply the moves: remove originals of the right-hand participants,
    # insert at new times.  Left-hand participants are untouched (they're
    # the anchor).
    adjusted = [list(t) for t in all_times]
    for t1, u1, new_t2, u2 in pairs:
        # Find the original t2 we moved — recorded in `moved`.  To do that
        # we need the original time; we logged (u2, t2) but not the pair
        # structure linking them to (t1, u1, new_t2, u2).  Reconstruct by
        # looking up any member of `moved` whose unit is u2 and whose time
        # is close enough to the new_t2 - offset distance.  Simpler: pop
        # (u2, t2) with lowest index that we haven't popped yet.
        pass
    # Simpler re-do: build spike lists from `moved` state directly.
    adjusted = [[t for t in times if (u, t) not in moved]
                for u, times in enumerate(all_times)]
    for t1, u1, new_t2, u2 in pairs:
        adjusted[u2].append(new_t2)
    adjusted = [np.sort(np.asarray(t, np.int64)) for t in adjusted]

    # 4. Build the recording: zeros + each spike's template, + noise
    traces = _ar1_noise(cfg.n_samples, cfg.n_channels, sigma=cfg.noise_sigma,
                        seed=cfg.seed + 1)
    for u, times in enumerate(adjusted):
        tpl = templates[u]
        for t in times:
            lo = int(t) - cfg.peak_sample
            hi = lo + cfg.template_samples
            if lo < 0 or hi > cfg.n_samples: continue
            traces[lo:hi, :] += tpl

    traces = traces.astype(np.float32)
    recording = _numpy_recording(traces, cfg.sampling_rate)
    # Annotate the recording with the corpus name so SortedResult.corpus_item
    # carries it (the harness's BenchmarkRun passes recording through to the
    # adapter, which reads this annotation when constructing the result).
    recording.annotate(name=name)

    # 5. Build the ground-truth SortedResult (unit IDs offset by 2 to leave
    # 0/1 free as reserved IDs by Klusters convention)
    unit_ids = [u + 2 for u in range(cfg.n_units)]
    spike_times = {u + 2: adjusted[u] for u in range(cfg.n_units)}

    # Non-overlap times (per unit) for the Pröpper classification endpoint
    non_overlap = {}
    overlap_times_by_unit = {u + 2: set() for u in range(cfg.n_units)}
    for t1, u1, new_t2, u2 in pairs:
        overlap_times_by_unit[u1 + 2].add(int(t1))
        overlap_times_by_unit[u2 + 2].add(int(new_t2))
    for u in range(cfg.n_units):
        s = overlap_times_by_unit[u + 2]
        non_overlap[u + 2] = np.array([t for t in adjusted[u] if int(t) not in s], np.int64)

    gt = SortedResult(
        unit_ids=unit_ids,
        spike_times=spike_times,
        sampling_rate=cfg.sampling_rate,
        adapter="ground_truth", corpus_item=name,
        adapter_version="propper_sim/0.1",
        templates=np.stack(templates).astype(np.float32),
        channels=list(range(cfg.n_channels)),
        extras=dict(
            overlap_pairs=[(int(t1), int(u1 + 2), int(t2), int(u2 + 2))
                           for t1, u1, t2, u2 in pairs],
            non_overlap_times_by_unit=non_overlap,
            n_overlap_pairs=len(pairs),
            offset_distribution=cfg.offset_distribution,
            max_offset_samples=cfg.max_offset_samples,
            seed=cfg.seed,
        ),
    )

    # Minimal ns3-shaped yaml so adapter contracts validate on the
    # simulator corpus too.
    ns3_yaml = {
        "acquisitionSystem": dict(
            samplingRate=cfg.sampling_rate, nChannels=cfg.n_channels,
            nBits=16, voltageRange=20, amplification=1000,
        ),
        "spikeDetection": {"channelGroups": [dict(
            channels=list(range(cfg.n_channels)),
            nSamples=cfg.template_samples, peakSampleIndex=cfg.peak_sample,
            nFeatures=3,
        )]},
    }

    return Corpus(
        name=name,
        recording=recording,
        ground_truth=gt,
        description=description,
        kind="simulated",
        ns3_yaml=ns3_yaml,
        extras=dict(n_units=cfg.n_units, rate_hz=cfg.rate_hz,
                    noise_sigma=cfg.noise_sigma,
                    overlap_fraction=cfg.overlap_fraction),
    )
