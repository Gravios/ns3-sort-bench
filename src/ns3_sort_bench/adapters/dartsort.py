"""DARTsort adapter (Boussard, Varol, Paninski 2023+)."""
from __future__ import annotations

import math
import tempfile
from typing import Mapping
import numpy as np

from .._base import (
    SortedResult, si_sorting_to_spike_times, SorterAdapter, _register, adapter_registry, measure_runtime,
)
from ..configs import load_adapter_config
from ..parallel import cuda_available, resolve_n_jobs, si_job_kwargs
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


# DARTsortUserConfig defaults (0.5.31) for the window fields below; used
# only when dartsort itself can't be imported to read them.
_WINDOW_DEFAULTS = {
    "ms_before": 1.4,
    "ms_after": 2.6 + 0.1 / 3,
    "feature_ms_before": 0.75,
    "feature_ms_after": 1.25,
    "alignment_ms": 1.5,
}


def _window_defaults() -> dict:
    try:
        from dartsort.config import DARTsortUserConfig
        fields = DARTsortUserConfig.model_fields
        return {k: float(fields[k].default) if k in fields else v
                for k, v in _WINDOW_DEFAULTS.items()}
    except Exception:
        return dict(_WINDOW_DEFAULTS)


def snap_windows_to_samples(run_kw: Mapping, fs: float) -> dict:
    """Return *run_kw* with DARTsort's ms windows set to whole samples.

    DARTsort rounds every ms value to samples on its own
    (``floor(ms*fs/1000 + 0.5)``), separately for the trough offset, the
    total length and each padded variant.  At rates where 1.4 ms is not
    a whole number of samples those roundings disagree: at 32552 Hz the
    base window is 131 samples but the realignment pad (alignment_ms
    each side, trimmed by twice the trough shift) leaves 133, and
    TemplateReduction then fails copying a 133-long TSVD mean into a
    131-long buffer.  With every window an exact sample count the
    roundings agree at any rate.  ``alignment_ms`` is snapped to an
    even count because DARTsort halves it for the SVD alignment pad.
    The total of ms_before + ms_after is kept odd, as DARTsort wants.
    """
    out = dict(run_kw)
    base = _window_defaults()
    base.update({k: float(v) for k, v in run_kw.items() if k in base})

    def n(ms):
        return math.floor(ms * fs / 1000.0 + 0.5)

    def ms(samples):
        return 1000.0 * samples / fs

    for before, after in (("ms_before", "ms_after"),
                          ("feature_ms_before", "feature_ms_after")):
        nb = max(1, n(base[before]))
        total = n(base[before] + base[after])
        total = 2 * (total // 2) + 1  # DARTsort's odd rounding
        na = max(1, total - nb)
        out[before], out[after] = ms(nb), ms(na)
    na = max(2, 2 * round(base["alignment_ms"] * fs / 2000.0))
    out["alignment_ms"] = ms(na)
    return out


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
            return False, f"dartsort not installed (`pip install -e '.[dartsort]'` in the ns3-sort-bench checkout): {e}", ""

    @property
    def contract(self) -> Contract: return CONTRACT

    @property
    def version(self) -> str: return self._version

    def available(self): return self._available, self._reason

    def sort(self, recording, *, ns3_yaml=None, ground_truth_res=None,
             **kwargs) -> SortedResult:
        if not self._available:
            raise EnvironmentError(f"dartsort: {self._reason}")

        # SI's job kwargs cover the wrapper's own chunked work; DARTsort's
        # workers are its n_jobs_cpu / n_jobs_gpu (configs/dartsort.yaml).
        with tempfile.TemporaryDirectory(prefix="dartsort_") as td, \
                si_job_kwargs(kwargs.get("n_jobs")):
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


# DARTsort's GPU workers when it finds a GPU and the yaml sets none.  On
# one device they are threads sharing it (DARTsort only uses processes
# for multi-GPU), so a few keep the card busy between host-side steps.
_DEFAULT_GPU_JOBS = 4


def dartsort_jobs(run_kw: Mapping, n_jobs) -> dict:
    """Fill DARTsort's n_jobs_cpu / n_jobs_gpu unless the yaml set them.

    Upstream defaults are 0 for both: everything on the main thread.
    n_jobs_cpu takes the bench's --n-jobs; n_jobs_gpu is
    _DEFAULT_GPU_JOBS when CUDA is there.  DARTsort uses only the one
    that matches its device.
    """
    out = dict(run_kw)
    out.setdefault("n_jobs_cpu", resolve_n_jobs(n_jobs))
    if cuda_available():
        out.setdefault("n_jobs_gpu", _DEFAULT_GPU_JOBS)
    return out


def _run_dartsort(recording, folder, **kwargs):
    # Merge configs/dartsort.yaml defaults with the caller's one-off
    # run_kwargs; the caller WINS on key conflict, matching every
    # other adapter's config-overlay semantics.
    run_kw = {**_DARTSORT_RUN_KWARGS, **kwargs.get("run_kwargs", {})}
    run_kw = snap_windows_to_samples(run_kw, recording.get_sampling_frequency())
    run_kw = dartsort_jobs(run_kw, kwargs.get("n_jobs"))
    try:
        from spikeinterface.sorters import run_sorter
        from .si_sorters import sortable_recording
        # Two subfolders: run_sorter empties its own folder first.
        import os
        return run_sorter(sorter_name="dartsort",
                           recording=sortable_recording(
                               recording, os.path.join(folder, "recording")),
                           folder=os.path.join(folder, "sorter"),
                           remove_existing_folder=True,
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
