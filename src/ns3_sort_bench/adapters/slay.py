"""SLAy adapter (Janelia 2025).  Post-pass on Kilosort 4 output."""
from __future__ import annotations

import numpy as np

from .._base import (
    SortedResult, SorterAdapter, _register, adapter_registry, measure_runtime,
)
from ..contracts import (
    Contract, RecordingReqs, OutputSchema, Capabilities, ResourceEstimates,
)


CONTRACT = Contract(
    adapter_name="slay",
    upstream_url="https://github.com/JaneliaSciComp/slay",
    upstream_version_pin="git+main",
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
    output=OutputSchema(unit_id_offset=0, templates_emitted=False,
                         per_group_or_whole="whole_session"),
    capabilities=Capabilities(
        drift_aware=False,                   # inherited from upstream KS4
        overlap_resolution="none",           # merge-step only
        requires_gpu=True, requires_torch=True,
    ),
    resources=ResourceEstimates(
        min_ram_mb=4096, min_cpu_cores=4, gpu_vram_mb=4096,
        estimated_sec_per_minute_of_recording=2.0,  # post-pass only
    ),
    known_limitations=(
        "Depends on Kilosort 4 adapter being available in the same registry.",
        "Upstream API is pre-1.0; signature changes may break the adapter.",
        "CNN autoencoder weights are learned per installation; results depend "
        "on which weights were cached when the first run completed.",
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
            v = getattr(slay, "__version__", "unknown")
        except ImportError as e:
            return False, f"slay not installed (`pip install ns3-sort-bench[slay]`): {e}", ""
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

        with measure_runtime() as rt:
            ks = adapter_registry["kilosort4"]
            ks_result = ks.sort(recording, **kwargs.get("ks_kwargs", {}))
            merges = _slay_merges(recording, ks_result,
                                   **kwargs.get("slay_kwargs", {}))
            merged = _apply_merges(ks_result.spike_times, merges)

        return SortedResult(
            unit_ids=sorted(merged), spike_times=merged,
            sampling_rate=recording.get_sampling_frequency(),
            adapter=self.name,
            corpus_item=str(recording.get_annotation("name", "slay_run")),
            adapter_version=self._version,
            channels=list(recording.get_channel_ids()),
            wall_time_s=rt["wall_time_s"], peak_rss_mb=rt["peak_rss_mb"],
            extras=dict(ks_version=ks_result.adapter_version,
                        ks_n_units=ks_result.n_units(), n_merges=len(merges)),
        )


def _slay_merges(recording, ks_result, **kwargs):
    try:
        from slay import merge
    except ImportError:
        return []
    try:
        return list(merge.recommend_merges(
            recording=recording, spike_times=ks_result.spike_times,
            sampling_rate=ks_result.sampling_rate, **kwargs,
        ))
    except Exception:
        return []


def _apply_merges(spike_times, merges):
    parent = {u: u for u in spike_times}
    def find(u):
        while parent[u] != u: parent[u] = parent[parent[u]]; u = parent[u]
        return u
    for a, b in merges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    merged = {}
    for u, t in spike_times.items():
        r = find(u)
        merged[r] = np.sort(np.concatenate([merged.get(r, np.array([], np.int64)), t]))
    return merged


_register(adapter_registry, SLAyAdapter(), "slay")
