"""SLAy adapter (Koukuntla et al. 2025).  Merge post-pass on Kilosort 4 output.

Kilosort 4 sorts the recording; its units go into a SpikeInterface
SortingAnalyzer on the (band-passed) recording, and
``slay.compute_slay_merges`` returns groups of units to merge, which are
applied to the Kilosort 4 spike trains.

Uses the Python 3.11-3.13 fork (github.com/Gravios/SLAy): upstream pins
Python 3.11 only and does not import on SpikeInterface 0.105.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Mapping

import numpy as np

from .._base import (
    SortedResult, SorterAdapter, _register, adapter_registry, measure_runtime,
)
from ..configs import load_adapter_config
from ..contracts import (
    Contract, RecordingReqs, OutputSchema, Capabilities, ResourceEstimates,
)


def _slay_config_defaults() -> "tuple[dict, dict, dict]":
    """Pull ks_kwargs, slay_kwargs and analyzer defaults out of
    configs/slay.yaml.

    Shape is ``{ks_kwargs: {...}, slay_kwargs: {...}, analyzer: {...}}``.
    Missing or non-mapping sections → empty dicts.  Caller's own
    sort(ks_kwargs=..., slay_kwargs=..., analyzer=...) merges on top.
    """
    cfg = load_adapter_config("slay")
    if not isinstance(cfg, Mapping):
        return {}, {}, {}
    out = []
    for key in ("ks_kwargs", "slay_kwargs", "analyzer"):
        v = cfg.get(key, {})
        out.append(dict(v) if isinstance(v, Mapping) else {})
    return tuple(out)


_SLAY_KS_DEFAULTS, _SLAY_MERGE_DEFAULTS, _SLAY_ANALYZER_DEFAULTS = _slay_config_defaults()


CONTRACT = Contract(
    adapter_name="slay",
    upstream_url="https://github.com/Gravios/SLAy",
    upstream_version_pin="git+main (fork of saikoukunt/SLAy for Python 3.11-3.13)",
    yaml_fields_required=(
        "acquisitionSystem.samplingRate",
        "acquisitionSystem.nChannels",
    ),
    yaml_fields_optional=("spikeDetection.channelGroups",),
    recording=RecordingReqs(
        min_channels=1, min_duration_s=30.0,
        dtype=("int16", "float32"), probe_geometry_required=True,
        signal_space="raw",
    ),
    # unit_id_offset=2 is inherited from the kilosort4 adapter's IDs.
    output=OutputSchema(unit_id_offset=2, templates_emitted=False,
                         per_group_or_whole="whole_session"),
    capabilities=Capabilities(
        drift_aware=False,                   # inherited from upstream KS4
        overlap_resolution="none",           # merge-step only
        requires_gpu=False, requires_torch=True,   # GPU only speeds the autoencoder
    ),
    resources=ResourceEstimates(
        min_ram_mb=4096, min_cpu_cores=4, gpu_vram_mb=4096,
        estimated_sec_per_minute_of_recording=2.0,  # post-pass only
    ),
    known_limitations=(
        "Depends on Kilosort 4 adapter being available in the same registry.",
        "Upstream API is pre-1.0; signature changes may break the adapter.",
        "The autoencoder is trained afresh for every sort (in the sort's "
        "temporary folder); its random initialisation makes repeated runs differ.",
        "Automatic merge parameters need units of 1000 spikes; without one SLAy "
        "uses its defaults (k1 0.25, k2 1, threshold 0.5).",
    ),
    notes=(
        "Direct conceptual competitor to fiber-xcorr-merge and the planned "
        "_unit_margin LLR plumbing (see "
        "claude/botm-vs-fiber-deconv-audit-2026-11-02.md §5.1)."
    ),
)


class SLAyAdapter(SorterAdapter):
    name = "slay"

    def __init__(self):
        self._available, self._reason, self._version = self._detect()

    def _detect(self):
        try:
            import slay
            from importlib.metadata import version as _dist_version
            v = _dist_version("slay")
        except ImportError as e:
            return False, (f"slay not installed (`pip install -e '.[slay]'` in the "
                           f"ns3-sort-bench checkout): {e}"), ""
        if not hasattr(slay, "compute_slay_merges"):
            return False, "installed slay has no compute_slay_merges", v
        ks = adapter_registry.get("kilosort4")
        if ks is None:
            return False, "requires kilosort4 adapter", v
        ok, reason = ks.available()
        if not ok:
            return False, f"kilosort4 adapter unavailable ({reason})", v
        return True, "", v

    @property
    def contract(self) -> Contract: return CONTRACT

    @property
    def version(self) -> str: return self._version

    def available(self): return self._available, self._reason

    def sort(self, recording, *, ns3_yaml=None, ground_truth_res=None,
             **kwargs) -> SortedResult:
        if not self._available:
            raise EnvironmentError(f"slay: {self._reason}")

        # configs/slay.yaml defaults, caller one-offs on top.  ks_kwargs go
        # to the wrapped kilosort4 adapter (its own config fills the rest),
        # slay_kwargs to slay.compute_slay_merges, analyzer to the
        # SortingAnalyzer built for it.
        ks_kw = {**_SLAY_KS_DEFAULTS, **kwargs.get("ks_kwargs", {})}
        slay_kw = {**_SLAY_MERGE_DEFAULTS, **kwargs.get("slay_kwargs", {})}
        an_kw = {**_SLAY_ANALYZER_DEFAULTS, **kwargs.get("analyzer", {})}
        with measure_runtime() as rt:
            ks = adapter_registry["kilosort4"]
            ks_result = ks.sort(recording, **ks_kw)
            with tempfile.TemporaryDirectory(prefix="slay_") as td:
                groups = slay_merge_groups(recording, ks_result.spike_times,
                                           work_dir=td, analyzer=an_kw, **slay_kw)
            merged = apply_merge_groups(ks_result.spike_times, groups)

        return SortedResult(
            unit_ids=sorted(merged), spike_times=merged,
            sampling_rate=recording.get_sampling_frequency(),
            adapter=self.name,
            corpus_item=str(recording.get_annotation("name", "slay_run")),
            adapter_version=self._version,
            channels=list(recording.get_channel_ids()),
            wall_time_s=rt["wall_time_s"], peak_rss_mb=rt["peak_rss_mb"],
            extras=dict(ks_version=ks_result.adapter_version,
                        ks_n_units=ks_result.n_units(),
                        n_merge_groups=len(groups),
                        n_units_merged=sum(len(g) for g in groups)),
        )


def slay_merge_groups(recording, spike_times: Mapping[int, np.ndarray], *,
                      work_dir: str, analyzer: Mapping[str, Any] | None = None,
                      **slay_kwargs) -> List[List[int]]:
    """Groups of unit ids (keys of `spike_times`) that SLAy would merge.

    Builds a sparse in-memory SortingAnalyzer of `spike_times` on
    `recording` -- band-passed 300-6000 Hz first unless SpikeInterface
    marks it filtered -- with the extensions SLAy reads, and calls
    `slay.compute_slay_merges`.  Errors propagate: a failed post-pass is a
    failed sort, not a silent zero-merge result.

    `analyzer` keys: max_spikes_per_unit (500), correlogram_window_ms (100),
    correlogram_bin_ms (1.0), freq_min / freq_max (300 / 6000).
    `slay_kwargs` go to compute_slay_merges; model_path defaults to a file
    in `work_dir`, and autoencoder_params' num_channels to 8 or the
    smallest unit sparsity, whichever is less.
    """
    import spikeinterface.core as sc
    from slay import compute_slay_merges

    an = {"max_spikes_per_unit": 500, "correlogram_window_ms": 100.0,
          "correlogram_bin_ms": 1.0, "freq_min": 300.0, "freq_max": 6000.0,
          **(analyzer or {})}
    units = {int(u): np.asarray(t, np.int64) for u, t in spike_times.items()
             if len(t) > 0}
    if len(units) < 2:
        return []
    if not recording.is_filtered():
        from spikeinterface.preprocessing import bandpass_filter
        recording = bandpass_filter(recording, freq_min=an["freq_min"],
                                    freq_max=an["freq_max"])
    sorting = sc.NumpySorting.from_unit_dict(
        [units], sampling_frequency=recording.get_sampling_frequency())
    analyzer_obj = sc.create_sorting_analyzer(sorting, recording, format="memory",
                                              sparse=True)
    analyzer_obj.compute("random_spikes", method="uniform",
                         max_spikes_per_unit=an["max_spikes_per_unit"])
    analyzer_obj.compute("waveforms")
    analyzer_obj.compute("templates")
    analyzer_obj.compute("correlograms", window_ms=an["correlogram_window_ms"],
                         bin_ms=an["correlogram_bin_ms"])
    analyzer_obj.compute("noise_levels")

    kw = dict(slay_kwargs)
    kw.setdefault("model_path", os.path.join(work_dir, "slay_autoencoder.pt"))
    ae = dict(kw.get("autoencoder_params") or {})
    if "num_channels" not in ae:
        sparsity = analyzer_obj.sparsity
        narrowest = (min(len(c) for c in sparsity.unit_id_to_channel_ids.values())
                  if sparsity is not None else recording.get_num_channels())
        ae["num_channels"] = max(1, min(8, narrowest))
    kw["autoencoder_params"] = ae

    merges, _, _ = compute_slay_merges(analyzer_obj, **kw)
    return [[int(u) for u in group] for group in merges]


def apply_merge_groups(spike_times: Mapping[int, np.ndarray],
                       groups: List[List[int]]) -> Dict[int, np.ndarray]:
    """Merge each group into its lowest unit id; spike trains concatenated
    and sorted.  Units in no group are kept as they are."""
    target = {}
    for group in groups:
        members = [u for u in group if u in spike_times]
        for u in members:
            target[u] = min(members)
    merged: Dict[int, List[np.ndarray]] = {}
    for u, t in spike_times.items():
        merged.setdefault(target.get(u, u), []).append(np.asarray(t, np.int64))
    return {u: np.sort(np.concatenate(ts)) for u, ts in merged.items()}


_register(adapter_registry, SLAyAdapter(), "slay")
