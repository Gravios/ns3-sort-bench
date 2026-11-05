"""DARTsort adapter (Boussard, Varol, Paninski 2023+)."""
from __future__ import annotations

import tempfile
from typing import Mapping
import numpy as np

from .._base import (
    SortedResult, si_sorting_to_spike_times, SorterAdapter, _register, adapter_registry, measure_runtime,
)
from ..configs import load_adapter_config
from ..contracts import (
    Contract, RecordingReqs, OutputSchema, Capabilities, ResourceEstimates,
)


def _run_kwargs_from_config() -> dict:
    """``run_kwargs:`` section of ``configs/dartsort.yaml`` as a dict.

    Empty when the file is absent or the section isn't a mapping.
    Same contract as si_sorters._run_kwargs_from_config — the caller's
    sort(run_kwargs={...}) merges on top, so one-off tuning still wins.
    """
    cfg = load_adapter_config("dartsort")
    rk = cfg.get("run_kwargs", {}) if isinstance(cfg, Mapping) else {}
    return dict(rk) if isinstance(rk, Mapping) else {}


_DARTSORT_RUN_KWARGS = _run_kwargs_from_config()


CONTRACT = Contract(
    adapter_name="dartsort",
    upstream_url="https://github.com/cwindolf/dartsort",
    upstream_version_pin="0.5.31",
    yaml_fields_required=(
        "acquisitionSystem.samplingRate",
        "acquisitionSystem.nChannels",
    ),
    yaml_fields_optional=(
        "anatomicalDescription.channelGroups",
        "acquisitionSystem.nBits",
    ),
    recording=RecordingReqs(
        min_channels=8, min_duration_s=30.0,          # DARTsort needs enough drift signal
        dtype=("int16", "float32"), probe_geometry_required=True,
        signal_space="raw",
    ),
    output=OutputSchema(unit_id_offset=2, templates_emitted=True,
                         per_group_or_whole="whole_session"),
    capabilities=Capabilities(
        drift_aware=True, overlap_resolution="iterative_subtract",
        requires_gpu=True, requires_torch=True, reports_templates=True,
    ),
    resources=ResourceEstimates(
        min_ram_mb=8192, min_cpu_cores=4, gpu_vram_mb=12288,
        estimated_sec_per_minute_of_recording=15.0,
    ),
    known_limitations=(
        "Depth-registration assumes 1-D vertical probe geometry "
        "(Neuropixels-shaped); misbehaves on tetrode / octrode layouts.",
        "The published 'hybrid drift' corpus is not released — regenerate "
        "from an IBL recording + dartsort.util.hybrid.inject_units().",
    ),
    notes=(
        "Closest conceptual neighbour to fiber-kit: per-chunk sort + linking + "
        "drift-aware templates; different concrete choices on each axis."
    ),
)


class DARTsortAdapter(SorterAdapter):
    name = "dartsort"

    def __init__(self):
        self._available, self._reason, self._version = self._detect()

    def _detect(self):
        try:
            import dartsort
            return True, "", getattr(dartsort, "__version__", "unknown")
        except ImportError as e:
            return False, f"dartsort not installed (`pip install ns3-sort-bench[dartsort]`): {e}", ""

    @property
    def contract(self) -> Contract: return CONTRACT

    @property
    def version(self) -> str: return self._version

    def available(self): return self._available, self._reason

    def sort(self, recording, *, ns3_yaml=None, ground_truth_res=None,
             **kwargs) -> SortedResult:
        if not self._available:
            raise EnvironmentError(f"dartsort: {self._reason}")

        with tempfile.TemporaryDirectory(prefix="dartsort_") as td:
            with measure_runtime() as rt:
                sorting = _run_dartsort(recording, td, **kwargs)
            # Shift past Klusters' reserved IDs 0/1 (see
            # _base.si_sorting_to_spike_times) — raw SI IDs start at 0/1.
            spike_times, id_map = si_sorting_to_spike_times(sorting)

        return SortedResult(
            unit_ids=sorted(spike_times), spike_times=spike_times,
            sampling_rate=recording.get_sampling_frequency(),
            adapter=self.name,
            corpus_item=str(recording.get_annotation("name", "dartsort_run")),
            adapter_version=self._version,
            channels=list(recording.get_channel_ids()),
            wall_time_s=rt["wall_time_s"], peak_rss_mb=rt["peak_rss_mb"],
            extras=dict(folder=td, si_unit_id_map=id_map),
        )


def _run_dartsort(recording, folder, **kwargs):
    # Merge configs/dartsort.yaml defaults with the caller's one-off
    # run_kwargs; the caller WINS on key conflict, matching every
    # other adapter's config-overlay semantics.
    run_kw = {**_DARTSORT_RUN_KWARGS, **kwargs.get("run_kwargs", {})}
    try:
        from spikeinterface.sorters import run_sorter
        return run_sorter(sorter_name="dartsort", recording=recording,
                           folder=folder, remove_existing_folder=True,
                           verbose=False, **run_kw)
    except (ValueError, ImportError):
        import dartsort
        # Direct-call path has its own kwarg shape (direct_kwargs); the
        # config overlay only fires on the SI path, which is the
        # production codepath.  Callers of the fallback can still
        # pass direct_kwargs for one-off tuning.
        return dartsort.main.run_dartsort(
            recording=recording, output_directory=folder,
            **kwargs.get("direct_kwargs", {}),
        )


_register(adapter_registry, DARTsortAdapter(), "dartsort")
