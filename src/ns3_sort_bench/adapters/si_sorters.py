"""SpikeInterface-wrapped sorters: MountainSort 5, Tridesclous 2,
SpykingCircus 2, Kilosort 4.  Each declares its own Contract so the
runner can pre-filter corpora.

Per-adapter run_kwargs live in ``src/ns3_sort_bench/configs/<name>.yaml``
(shape: ``run_kwargs: {<kw>: <value>, ...}``) and are loaded at import
time via :func:`configs.load_adapter_config`.  Call-time overrides via
``adapter.sort(run_kwargs={...})`` merge on top.
"""
from __future__ import annotations

import tempfile
from typing import Any, Mapping, Optional

import numpy as np

from .._base import (
    SortedResult, SorterAdapter, _register, adapter_registry, measure_runtime,
)
from ..configs import load_adapter_config
from ..contracts import (
    Contract, RecordingReqs, OutputSchema, Capabilities, ResourceEstimates,
)


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
    _run_kwargs: dict = {}
    _contract: Optional[Contract] = None

    def __init__(self):
        self._available, self._reason, self._version = self._detect()

    def _detect(self):
        try:
            import spikeinterface  # noqa: F401
        except ImportError:
            return False, "spikeinterface not installed", ""
        try:
            if self._package_name:
                import importlib
                m = importlib.import_module(self._package_name)
                return True, "", getattr(m, "__version__", "unknown")
            return True, "", "builtin"
        except ImportError:
            return (False,
                    f"{self._package_name} not installed "
                    f"(`pip install ns3-sort-bench[{self.name}]`)",
                    "")

    @property
    def contract(self) -> Contract:
        return self._contract

    @property
    def version(self) -> str: return self._version

    def available(self): return self._available, self._reason

    def sort(self, recording, *, ns3_yaml=None, ground_truth_res=None,
             **kwargs) -> SortedResult:
        if not self._available:
            raise EnvironmentError(f"{self.name}: {self._reason}")
        from spikeinterface.sorters import run_sorter
        run_kw = {**self._run_kwargs, **kwargs.get("run_kwargs", {})}

        with tempfile.TemporaryDirectory(prefix=f"{self.name}_") as td:
            with measure_runtime() as rt:
                sorting = run_sorter(
                    sorter_name=self._si_name, recording=recording,
                    folder=td, remove_existing_folder=True, verbose=False,
                    **run_kw,
                )
            spike_times = {int(u): np.asarray(sorting.get_unit_spike_train(u), np.int64)
                           for u in sorting.get_unit_ids()}

        return SortedResult(
            unit_ids=sorted(spike_times), spike_times=spike_times,
            sampling_rate=recording.get_sampling_frequency(),
            adapter=self.name,
            corpus_item=str(recording.get_annotation("name", f"{self.name}_run")),
            adapter_version=self._version,
            channels=list(recording.get_channel_ids()),
            wall_time_s=rt["wall_time_s"], peak_rss_mb=rt["peak_rss_mb"],
            extras=dict(si_name=self._si_name, run_kwargs=run_kw),
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
        output=OutputSchema(unit_id_offset=0, templates_emitted=False,
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
        output=OutputSchema(unit_id_offset=0, per_group_or_whole="per_group"),
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
    _package_name = "circus"
    _run_kwargs = _run_kwargs_from_config("spykingcircus2")
    _contract = Contract(
        adapter_name="spykingcircus2",
        upstream_url="https://github.com/spyking-circus/spyking-circus",
        upstream_version_pin="docs/sorter-installs.md",   # non-PyPI install
        yaml_fields_required=_COMMON_YAML_REQ,
        yaml_fields_optional=_COMMON_YAML_OPT,
        recording=RecordingReqs(
            min_channels=1, min_duration_s=5.0,
            dtype=("int16", "float32"), probe_geometry_required=True,
            signal_space="filtered",
        ),
        output=OutputSchema(unit_id_offset=0, templates_emitted=True,
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
        output=OutputSchema(unit_id_offset=0, templates_emitted=True,
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
