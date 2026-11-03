"""Lattice BOTM adapter — fiber_deconv's primitives wrapped around the
variant-pursue function in `../variants/fiber_deconv_lattice.py`.

Exposes five independent ablation axes:
    overlap_rule     ∈ {'omp', 'sic-a', 'sic-b', 'sic-c'}  (4)
    sub_sample_bank  ∈ {'integer', 'R4'}                    (2)
    amplitude_prior  ∈ {'none', 'gaussian'}                 (2)
    noise_model      ∈ {'spatial', 'var4'}                  (2)
    refit_rule       : derived from overlap_rule (OMP ↔ joint refit;
                                                   SIC-* ↔ no joint refit)

Full lattice: 4 × 2 × 2 × 2 = 32 configurations.  Each is registered
under the key `lattice_<rule>_<R>_<prior>_<noise>`, e.g.
`lattice_omp_R4_gaussian_var4`.  The adapter registry gains all 32 at
import time; the runner picks whichever subset the user asks for.

Only runs on a single-group recording (one spike-detection group's worth
of channels).  For multi-group benchmarking the lattice runner iterates
groups and recombines results.
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from .._base import (
    SortedResult, SorterAdapter, _register, adapter_registry, measure_runtime,
)
from ..contracts import (
    Contract, RecordingReqs, OutputSchema, Capabilities, ResourceEstimates,
)
from ..variants.fiber_deconv_lattice import pursue_variant, PURSUE_RULES


def _lattice_contract(name: str, overlap_rule: str) -> Contract:
    """One Contract per lattice config.  Pins the five axis values into
    the capabilities + notes so a dashboard reader can tell at a glance
    which configuration a row came from."""
    return Contract(
        adapter_name=name,
        upstream_url="https://github.com/Gravios/fiber-kit",
        upstream_version_pin=">= 0.57.6",
        yaml_fields_required=(
            "acquisitionSystem.samplingRate",
            "acquisitionSystem.nChannels",
            "spikeDetection.channelGroups",
        ),
        yaml_fields_optional=(
            "spikeDetection.channelGroups[0].channels",
        ),
        recording=RecordingReqs(
            min_channels=1, min_duration_s=1.0,
            dtype=("int16", "float32", "float64"),
            probe_geometry_required=False,
            signal_space="raw",
        ),
        output=OutputSchema(
            unit_id_offset=2, templates_emitted=False,
            per_group_or_whole="per_group",
        ),
        capabilities=Capabilities(
            drift_aware=False,
            overlap_resolution=("omp" if overlap_rule == "omp" else "sic"),
            requires_gpu=False, supports_shared_detection=True,
            supports_stderiv_input=True,
        ),
        resources=ResourceEstimates(
            min_ram_mb=1024, min_cpu_cores=1,
            estimated_sec_per_minute_of_recording=30.0,
        ),
        known_limitations=(
            "Requires ground-truth .res/.labels to anchor template estimation "
            "(normal lattice runs get this from the corpus).",
        ),
        notes=f"Lattice variant: {name}",
    )


# ─── axis values ─────────────────────────────────────────────────────────────
RULES = PURSUE_RULES                           # 4
BANKS = ("integer", "R4")                      # 2 — R=1 vs R=4
PRIORS = ("none", "gaussian")                  # 2 — amp_sd_floor huge vs real
NOISES = ("spatial", "var4")                   # 2 — order=0 vs 4


def _bank_R(name: str) -> int:
    return 1 if name == "integer" else 4


def _prior_amp_sd_floor(name: str) -> float:
    # "none" ⇒ huge amp_sd ⇒ λ ≈ 0 ⇒ gain reduces to BOTM's 2ac − a²n
    return 1e6 if name == "none" else 0.08


def _noise_order(name: str) -> int:
    return 0 if name == "spatial" else 4


# ─── adapter ─────────────────────────────────────────────────────────────────
class LatticeBOTMAdapter(SorterAdapter):
    """One lattice configuration.  Instantiated 32× at import time, each
    with its own `name` and axis settings."""

    requires_gpu = False
    upstream_url = "https://github.com/Gravios/fiber-kit"

    def __init__(self, rule: str, bank_kind: str, prior: str, noise: str):
        self._rule = rule
        self._bank_kind = bank_kind
        self._prior = prior
        self._noise = noise
        self.name = f"lattice_{rule}_{bank_kind}_{prior}_{noise}"
        self._contract_obj = _lattice_contract(self.name, rule)
        self._available, self._reason, self._version = self._detect()

    @property
    def contract(self):
        return self._contract_obj

    def _detect(self):
        try:
            import fiber_kit
            v = getattr(fiber_kit, "__version__", "unknown")
            return True, "", f"lattice/{v}"
        except ImportError as e:
            return False, f"fiber_kit not installed ({e})", ""

    @property
    def version(self) -> str:
        return self._version

    def available(self):
        return self._available, self._reason

    def sort(self, recording, *, ns3_yaml=None,
             ground_truth_res=None, **kwargs) -> SortedResult:
        if not self._available:
            raise EnvironmentError(f"{self.name}: {self._reason}")

        # Import here so a bare install can still import this module.
        from fiber_kit.fiber_deconv import (
            fit_noise_model, build_bank, whiten, null_threshold,
            unit_templates, segments, read_transformed, memmap_reader,
            amplitude_spread,
        )
        from spikeinterface.core import BaseRecording
        if not isinstance(recording, BaseRecording):
            raise TypeError("lattice_botm adapter needs a SpikeInterface BaseRecording")

        # 1. Materialise the recording as a (n_samples, n_channels) float array
        #    the fiber_deconv primitives can read via a simple `read_raw`
        #    callback.  This is wasteful for long recordings; the lattice is
        #    meant for the Pröpper simulator and short real-data snippets, so
        #    the extra memory is bounded.
        n_samp = recording.get_num_samples(0)
        traces = np.asarray(recording.get_traces(start_frame=0, end_frame=n_samp),
                            np.float64)

        def read_raw(a, b):
            a = max(0, int(a)); b = min(n_samp, int(b))
            if b <= a:
                return np.zeros((0, traces.shape[1]), np.float64)
            return traces[a:b]

        with measure_runtime() as rt:
            # 2. Need the ground-truth spike times to anchor template
            # estimation and the noise-cov fit.  The lattice adapter requires
            # the caller to pass them via `ground_truth_res` + `ground_truth_labels`;
            # falls back to a simple threshold detector if neither is given.
            res_abs = kwargs.get("ground_truth_res", ground_truth_res)
            labels = kwargs.get("ground_truth_labels", None)
            if res_abs is None or labels is None:
                res_abs, labels = _bootstrap_detection(traces)

            res_abs = np.asarray(res_abs, np.int64)
            labels = np.asarray(labels, np.int64)

            # 3. Noise model + whitening transform (axis: noise_model)
            noise_order = _noise_order(self._noise)
            nm = fit_noise_model(
                read_raw, s0=0, s1=n_samp, spike_abs=res_abs,
                order=noise_order, n_seg=200, seg_len=min(192, n_samp // 4),
            )

            # 4. Templates: unit means on the median-aligned snippets
            nsamp_tpl = int(kwargs.get("template_samples", 32))
            peak = int(kwargs.get("peak_sample", 16))
            templates = unit_templates(
                read_raw, res_abs, labels, nsamp=nsamp_tpl, peak=peak,
                min_n=max(5, len(res_abs) // 100),
            )

            # 5. Atom bank (axes: sub_sample_bank, amplitude_prior)
            amp_sd = None
            if self._prior == "gaussian":
                # Use the measured per-unit amplitude spread
                try:
                    bank_tmp = build_bank(templates, nm, nsamp_tpl, peak,
                                          R=_bank_R(self._bank_kind))
                    amp_sd = amplitude_spread(
                        read_raw, res_abs, labels, bank_tmp, nm,
                    )
                except Exception:
                    amp_sd = None   # bank will use its own floor
            bank = build_bank(
                templates, nm, nsamp_tpl, peak,
                R=_bank_R(self._bank_kind),
                amp_sd=amp_sd,
                amp_sd_floor=_prior_amp_sd_floor(self._prior),
            )

            # 6. Null-threshold calibration
            thr, _ = null_threshold(
                read_raw, s0=0, s1=n_samp, spike_abs=res_abs,
                bank=bank, nm=nm, n=200,
            )

            # 7. Run the variant pursuit over detection-anchored segments
            refrac = int(kwargs.get("refrac_ms", 1.0) *
                         recording.get_sampling_frequency() / 1000.0)
            La = bank.atoms.shape[1]
            segs, order = segments(res_abs, nsamp_tpl, peak, pad=0,
                                    lead=bank.lead, La=La)

            placed: Dict[int, list] = {}   # unit → list of absolute sample indices
            for seg_start, seg_end, _, _ in segs:
                x = read_transformed(read_raw, seg_start, seg_end + bank.order)
                y = whiten(x, nm)[bank.order:]
                plcs, _ = pursue_variant(
                    y, bank, thr, refrac, refit_rule=self._rule,
                )
                # Translate atom indices → unit ids; sample offsets → absolute
                for atom_i, s_rel, _, _ in plcs:
                    unit_idx = int(bank.unit[atom_i])
                    t_abs = seg_start + int(s_rel) + peak
                    placed.setdefault(unit_idx, []).append(t_abs)

        unit_ids = sorted(placed)
        unit_ids = [u + 2 for u in unit_ids]   # offset so 0/1 stay reserved
        spike_times = {u + 2: np.array(sorted(set(placed[u])), np.int64)
                       for u in sorted(placed)}

        return SortedResult(
            unit_ids=unit_ids, spike_times=spike_times,
            sampling_rate=recording.get_sampling_frequency(),
            adapter=self.name,
            corpus_item=str(recording.get_annotation("name", f"{self.name}_run")),
            adapter_version=self._version,
            channels=list(recording.get_channel_ids()),
            wall_time_s=rt["wall_time_s"], peak_rss_mb=rt["peak_rss_mb"],
            extras=dict(
                overlap_rule=self._rule, sub_sample_bank=self._bank_kind,
                amplitude_prior=self._prior, noise_model=self._noise,
                noise_order=_noise_order(self._noise),
                bank_R=_bank_R(self._bank_kind),
                amp_sd_floor=_prior_amp_sd_floor(self._prior),
            ),
        )


def _bootstrap_detection(traces: np.ndarray) -> tuple:
    """Fallback detection when no ground-truth res/labels are passed.
    Simple 4×MAD negative threshold on the max-ptp channel, with naive
    KMeans clustering in a small PCA space to assign unit labels.  Only
    used by the adapter when it has to run blind; the normal lattice path
    passes ground truth from the corpus."""
    n_samp, n_ch = traces.shape
    ch = int(np.argmax(np.ptp(traces, axis=0)))
    mad = np.median(np.abs(traces[:, ch] - np.median(traces[:, ch]))) * 1.4826
    thr = -4.0 * mad
    peaks = []
    for i in range(1, n_samp - 1):
        if traces[i, ch] < thr and traces[i, ch] <= traces[i - 1, ch] \
                                 and traces[i, ch] <= traces[i + 1, ch]:
            if not peaks or i - peaks[-1] > 20:
                peaks.append(i)
    peaks = np.asarray(peaks, np.int64)
    if peaks.size == 0:
        return np.array([], np.int64), np.array([], np.int64)

    # KMeans k=3 on 6-D PCA of snippets around the peaks
    half = 16
    peaks = peaks[(peaks >= half) & (peaks < n_samp - half)]
    snips = np.stack([traces[p - half:p + half].ravel() for p in peaks])
    mean = snips.mean(axis=0)
    _, _, Vt = np.linalg.svd(snips - mean, full_matrices=False)
    feats = (snips - mean) @ Vt[:6].T
    labels = _kmeans(feats, k=3, seed=0)
    return peaks, labels


def _kmeans(x, k, *, seed=0, iters=20):
    rng = np.random.default_rng(seed)
    n = len(x)
    if k >= n:
        return np.arange(n) % k
    centres = x[rng.choice(n, k, replace=False)]
    for _ in range(iters):
        d = np.linalg.norm(x[:, None, :] - centres[None, :, :], axis=2)
        lab = d.argmin(axis=1)
        new = np.stack([x[lab == i].mean(axis=0) if (lab == i).any() else centres[i]
                        for i in range(k)])
        if np.allclose(new, centres): break
        centres = new
    return lab


# ─── register all 32 configurations ──────────────────────────────────────────
_LATTICE_INSTANCES = []
for rule in RULES:
    for bank_kind in BANKS:
        for prior in PRIORS:
            for noise in NOISES:
                inst = LatticeBOTMAdapter(rule, bank_kind, prior, noise)
                _LATTICE_INSTANCES.append(inst)
                try:
                    _register(adapter_registry, inst, inst.name)
                except KeyError:
                    pass   # re-import under pytest, etc.


def enumerate_lattice():
    """Return the list of registered lattice adapter names."""
    return [a.name for a in _LATTICE_INSTANCES]
