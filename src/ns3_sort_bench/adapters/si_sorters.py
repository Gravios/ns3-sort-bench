"""SpikeInterface-wrapped sorters: MountainSort 5, Tridesclous 2,
SpykingCircus 2, Kilosort 4.  Each declares its own Contract so the
runner can pre-filter corpora.

Per-adapter run_kwargs live in ``src/ns3_sort_bench/configs/<name>.yaml``
(shape: ``run_kwargs: {<kw>: <value>, ...}``) and are loaded at import
time via :func:`configs.load_adapter_config`.  Call-time overrides via
``adapter.sort(run_kwargs={...})`` merge on top.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Mapping, Optional

import numpy as np

from .._base import (
    SortedResult, si_sorting_to_spike_times, SorterAdapter, _register, adapter_registry, measure_runtime,
)
from ..configs import load_adapter_config
from ..parallel import cuda_available, si_job_kwargs
from ..contracts import (
    Contract, RecordingReqs, OutputSchema, Capabilities, ResourceEstimates,
)


def sortable_recording(recording, folder: str):
    """The recording itself when SpikeInterface can hand it to a sorter,
    else a binary copy of it saved under `folder`.

    `run_sorter` dumps the recording to json or pickle before it sorts and
    refuses one that is neither -- an in-memory NumpyRecording, which is
    what the Propper simulator and other generated corpora give.  Saving
    a copy costs a write of the traces; a recording read from files
    (SpikeForest, a lab session) is passed through untouched.
    """
    if (recording.check_serializability("json")
            or recording.check_serializability("pickle")):
        return recording
    # n_jobs=1: an in-memory recording is pickled into every worker of a
    # process pool, so N jobs would hold N copies of the traces.
    return recording.save(folder=folder, format="binary", verbose=False,
                          n_jobs=1)


def _run_kwargs_from_config(name: str) -> dict:
    """``run_kwargs:`` section of ``configs/<name>.yaml`` as a dict; ``{}``
    when the file is absent, the section is missing, or the section is
    not a mapping.  Called once per adapter at module import."""
    cfg = load_adapter_config(name)
    rk = cfg.get("run_kwargs", {})
    return dict(rk) if isinstance(rk, Mapping) else {}


# ─── common ns3-yaml fields SI-wrapped sorters consume ───────────────────────
_COMMON_YAML_REQ = (
    "acquisitionSystem.samplingRate",
    "acquisitionSystem.nChannels",
    "acquisitionSystem.nBits",
)
_COMMON_YAML_OPT = (
    "acquisitionSystem.voltageRange",
    "acquisitionSystem.amplification",
    "spikeDetection.channelGroups",
)


class _SISorterBase(SorterAdapter):
    """Shared machinery for SI-wrapped sorters."""
    _si_name: str = ""
    _package_name: str = ""
    # A sorter built into SpikeInterface has no package of its own: it is
    # available when SpikeInterface's own check (`is_installed()`) passes,
    # and its version is SpikeInterface's.
    _si_internal: bool = False
    _run_kwargs: dict = {}
    _contract: Optional[Contract] = None

    def __init__(self):
        self._available, self._reason, self._version = self._detect()

    def _detect(self):
        try:
            import spikeinterface
        except ImportError:
            return False, "spikeinterface not installed", ""
        if self._si_internal:
            from spikeinterface.sorters import sorter_dict
            if sorter_dict[self._si_name].is_installed():
                return True, "", f"spikeinterface-{spikeinterface.__version__}"
            return (False,
                    f"SpikeInterface's {self._si_name} is missing a dependency: "
                    f"numba or hdbscan (`pip install -e '.[{self.name}]'` in the "
                    f"ns3-sort-bench checkout)",
                    "")
        try:
            if self._package_name:
                import importlib
                m = importlib.import_module(self._package_name)
                return True, "", getattr(m, "__version__", "unknown")
            return True, "", "builtin"
        except ImportError:
            return (False,
                    f"{self._package_name} not installed "
                    f"(`pip install -e '.[{self.name}]'` in the ns3-sort-bench checkout)",
                    "")

    @property
    def contract(self) -> Contract:
        return self._contract

    @property
    def version(self) -> str: return self._version

    def available(self): return self._available, self._reason

    def _device_kwargs(self, run_kw: dict) -> dict:
        """Hook for a sorter whose GPU use has to be asked for."""
        return run_kw

    def sort(self, recording, *, ns3_yaml=None, ground_truth_res=None,
             **kwargs) -> SortedResult:
        if not self._available:
            raise EnvironmentError(f"{self.name}: {self._reason}")
        from spikeinterface.sorters import run_sorter
        run_kw = {**self._run_kwargs, **kwargs.get("run_kwargs", {})}
        run_kw = self._device_kwargs(run_kw)

        with tempfile.TemporaryDirectory(prefix=f"{self.name}_") as td, \
                si_job_kwargs(kwargs.get("n_jobs")) as n_jobs:
            with measure_runtime() as rt:
                # Two subfolders: run_sorter empties its own folder first.
                sorting = run_sorter(
                    sorter_name=self._si_name,
                    recording=sortable_recording(
                        recording, os.path.join(td, "recording")),
                    folder=os.path.join(td, "sorter"),
                    remove_existing_folder=True, verbose=False,
                    **run_kw,
                )
            # Shift past Klusters' reserved IDs 0/1 (see
            # _base.si_sorting_to_spike_times) — raw SI IDs start at 0/1.
            spike_times, id_map = si_sorting_to_spike_times(sorting)

        return SortedResult(
            unit_ids=sorted(spike_times), spike_times=spike_times,
            sampling_rate=recording.get_sampling_frequency(),
            adapter=self.name,
            corpus_item=str(recording.get_annotation("name", f"{self.name}_run")),
            adapter_version=self._version,
            channels=list(recording.get_channel_ids()),
            wall_time_s=rt["wall_time_s"], peak_rss_mb=rt["peak_rss_mb"],
            extras=dict(si_name=self._si_name, run_kwargs=run_kw,
                        si_unit_id_map=id_map, n_jobs=n_jobs),
        )


# ─── MountainSort 5 ──────────────────────────────────────────────────────────
class MountainSort5Adapter(_SISorterBase):
    name = "mountainsort5"
    _si_name = "mountainsort5"
    _package_name = "mountainsort5"
    _run_kwargs = _run_kwargs_from_config("mountainsort5")
    _contract = Contract(
        adapter_name="mountainsort5",
        upstream_url="https://github.com/flatironinstitute/mountainsort5",
        upstream_version_pin="0.5.9",
        yaml_fields_required=_COMMON_YAML_REQ,
        yaml_fields_optional=_COMMON_YAML_OPT,
        recording=RecordingReqs(
            min_channels=1, max_channels=None, min_duration_s=1.0,
            dtype=("int16", "float32"), probe_geometry_required=True,
            signal_space="filtered",
        ),
        output=OutputSchema(unit_id_offset=2, templates_emitted=False,
                             per_group_or_whole="per_group"),
        capabilities=Capabilities(
            drift_aware=False, overlap_resolution="none",
            requires_gpu=False, supports_shared_detection=False,
        ),
        resources=ResourceEstimates(
            min_ram_mb=1024, min_cpu_cores=2,
            estimated_sec_per_minute_of_recording=15.0,
        ),
        known_limitations=(
            "Default is bandpass 300-6000 + whitening; feed pre-filtered signal to skip.",
            "Weak on below-median-amplitude cells (0.59 detection on g6 reference).",
        ),
    )


# ─── Tridesclous 2 ───────────────────────────────────────────────────────────
class Tridesclous2Adapter(_SISorterBase):
    name = "tridesclous2"
    _si_name = "tridesclous2"
    _package_name = "tridesclous"
    _run_kwargs = _run_kwargs_from_config("tridesclous2")
    _contract = Contract(
        adapter_name="tridesclous2",
        upstream_url="https://github.com/tridesclous/tridesclous",
        upstream_version_pin="1.6.9",
        yaml_fields_required=_COMMON_YAML_REQ,
        yaml_fields_optional=_COMMON_YAML_OPT,
        recording=RecordingReqs(
            min_channels=1, min_duration_s=1.0,
            dtype=("int16", "float32"), probe_geometry_required=True,
            signal_space="raw",
        ),
        output=OutputSchema(unit_id_offset=2, per_group_or_whole="per_group"),
        capabilities=Capabilities(
            drift_aware=False, overlap_resolution="none", requires_gpu=False,
        ),
        resources=ResourceEstimates(
            min_ram_mb=1024, min_cpu_cores=2,
            estimated_sec_per_minute_of_recording=20.0,
        ),
    )


# ─── SpykingCircus 2 ─────────────────────────────────────────────────────────
class SpykingCircus2Adapter(_SISorterBase):
    name = "spykingcircus2"
    _si_name = "spykingcircus2"
    # Built into SpikeInterface (sorters/internal/spyking_circus2.py); its
    # check wants scipy, numba and hdbscan.  The `circus` package probed
    # here before is SpyKING CIRCUS 1, which SpikeInterface 0.105 does not
    # use for this sorter.
    _si_internal = True
    _run_kwargs = _run_kwargs_from_config("spykingcircus2")

    def _device_kwargs(self, run_kw: dict) -> dict:
        """Template matching (circus-omp, the bulk of the sort) on CUDA.

        SpikeInterface's default engine is numpy on the CPU.  With a GPU
        and no `matching:` in configs/spykingcircus2.yaml, the torch
        engine runs the convolutions on cuda.  Every job worker opens
        its own CUDA context, and an earlier sorter in this process
        (Kilosort 4, DARTsort) has usually initialised CUDA already,
        which a forked worker can't inherit, so the workers are spawned.
        Note the SpikeInterface 0.105 wrapper reads `matching_kwargs`,
        not the `method_kwargs` its own default dict shows.
        """
        if "matching" in run_kw or not cuda_available():
            return run_kw
        out = dict(run_kw)
        out["matching"] = {
            "method": "circus-omp",
            "matching_kwargs": {"engine": "torch", "torch_device": "cuda"},
            "pipeline_kwargs": {},
        }
        job = dict(out.get("job_kwargs") or {})
        job.setdefault("mp_context", "spawn")
        out["job_kwargs"] = job
        return out
    _contract = Contract(
        adapter_name="spykingcircus2",
        upstream_url="https://github.com/SpikeInterface/spikeinterface",
        upstream_version_pin="spikeinterface >=0.105.4,<0.106 (built in)",
        yaml_fields_required=_COMMON_YAML_REQ,
        yaml_fields_optional=_COMMON_YAML_OPT,
        recording=RecordingReqs(
            min_channels=1, min_duration_s=5.0,
            dtype=("int16", "float32"), probe_geometry_required=True,
            signal_space="filtered",
        ),
        output=OutputSchema(unit_id_offset=2, templates_emitted=True,
                             per_group_or_whole="per_group"),
        capabilities=Capabilities(
            drift_aware=False, overlap_resolution="iterative_subtract",
            requires_gpu=False, reports_templates=True,
        ),
        resources=ResourceEstimates(
            min_ram_mb=2048, min_cpu_cores=4,
            estimated_sec_per_minute_of_recording=25.0,
        ),
        known_limitations=(
            "Motion correction can misfire on short recordings; off by default.",
        ),
    )


# ─── Kilosort 4 ──────────────────────────────────────────────────────────────
class Kilosort4Adapter(_SISorterBase):
    name = "kilosort4"
    _si_name = "kilosort4"
    _package_name = "kilosort"
    _run_kwargs = _run_kwargs_from_config("kilosort4")
    _contract = Contract(
        adapter_name="kilosort4",
        upstream_url="https://github.com/MouseLand/Kilosort",
        upstream_version_pin="4.1.7",
        yaml_fields_required=_COMMON_YAML_REQ,
        yaml_fields_optional=_COMMON_YAML_OPT + ("anatomicalDescription.channelGroups",),
        recording=RecordingReqs(
            min_channels=1, min_duration_s=10.0,
            dtype=("int16", "float32"), probe_geometry_required=True,
            signal_space="raw",
        ),
        output=OutputSchema(unit_id_offset=2, templates_emitted=True,
                             per_group_or_whole="whole_session"),
        capabilities=Capabilities(
            drift_aware=True, overlap_resolution="iterative_subtract",
            requires_gpu=True, requires_torch=True,
            reports_templates=True,
        ),
        resources=ResourceEstimates(
            min_ram_mb=4096, min_cpu_cores=4, gpu_vram_mb=8192,
            estimated_sec_per_minute_of_recording=5.0,
        ),
        known_limitations=(
            "do_CAR=True subtracts spikes on sub-10-channel geometries "
            "(see spikeinterface-g6-comparison.md); disabled by default here.",
            "Drift tracking assumes 1-D displacement — misfires on tetrodes without "
            "common depth axis.",
        ),
    )


for cls, key in [
    (MountainSort5Adapter, "mountainsort5"),
    (Tridesclous2Adapter, "tridesclous2"),
    (SpykingCircus2Adapter, "spykingcircus2"),
    (Kilosort4Adapter, "kilosort4"),
]:
    _register(adapter_registry, cls(), key)
