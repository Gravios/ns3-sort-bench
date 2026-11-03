"""fiber-kit adapter — wraps the fiber-session chain behind SorterAdapter.

Three modes registered: fiber_kit (full chain), fiber_kit_no_deconv,
fiber_kit_session.  All share the same Contract shape; only the
`_mode` + env override differ."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Optional

import numpy as np

from .._base import (
    SortedResult, SorterAdapter, _register, adapter_registry, measure_runtime,
)
from ..contracts import (
    Contract, RecordingReqs, OutputSchema, Capabilities, ResourceEstimates,
)


def _contract_for(mode: str, with_deconv: bool) -> Contract:
    return Contract(
        adapter_name=mode,
        upstream_url="https://github.com/Gravios/fiber-kit",
        upstream_version_pin=">= 0.57.6",
        yaml_fields_required=(
            "acquisitionSystem.samplingRate",
            "acquisitionSystem.nChannels",
            "acquisitionSystem.nBits",
            "spikeDetection.channelGroups",
            "spikeDetection.channelGroups[0].channels",
        ),
        yaml_fields_optional=(
            "spikeDetection.channelGroups[0].nSamples",
            "spikeDetection.channelGroups[0].peakSampleIndex",
            "spikeDetection.channelGroups[0].nFeatures",
            "fieldPotentials.lfpSamplingRate",
        ),
        recording=RecordingReqs(
            min_channels=1, min_duration_s=30.0,
            dtype=("int16", "float32"), probe_geometry_required=False,
            signal_space="raw",       # fiber-session reads .fil directly
        ),
        output=OutputSchema(
            unit_id_offset=2, includes_reserved=False, templates_emitted=False,
            per_group_or_whole="per_group",
        ),
        capabilities=Capabilities(
            drift_aware=True,
            overlap_resolution="omp" if with_deconv else "none",
            requires_gpu=False, requires_torch=False,
            supports_stderiv_input=True, supports_shared_detection=True,
        ),
        resources=ResourceEstimates(
            min_ram_mb=4096, min_cpu_cores=4,
            estimated_sec_per_minute_of_recording=30.0,
        ),
        known_limitations=(
            "Session yaml must be ADJACENT to the .fil / .dat (same basename).",
            "Deconv ON requires fiber-kit >= 0.57.6 for the FK_SESSION_DECONV env flag.",
            "Variant/tag names must match the stage output you want "
            "(default: stderiv_C5_D34 / exp_final).",
        ),
        notes="Subject of the benchmark; three modes registered as fiber_kit, "
              "fiber_kit_no_deconv, fiber_kit_session.",
    )


class _FiberKitBase(SorterAdapter):
    _mode: str = "fiber_kit"
    _env_overrides: dict = {}
    _contract_obj: Optional[Contract] = None

    def __init__(self):
        try:
            import fiber_kit
            self._available = True
            self._version = getattr(fiber_kit, "__version__", "unknown")
        except ImportError:
            self._available = False; self._version = ""

    @property
    def contract(self): return self._contract_obj

    @property
    def version(self): return self._version

    def available(self):
        if not self._available:
            return False, "fiber_kit not installed (`pip install ns3-sort-bench[fiber-kit]`)"
        return True, ""

    _MODE_STAGES = {
        "fiber_kit": ["fiber-session", "fiber-realign", "fiber-refine",
                      "fiber-peel", "fiber-intrachunk", "fiber-link"],
        "fiber_kit_no_deconv": ["fiber-session", "fiber-realign", "fiber-refine",
                                 "fiber-peel", "fiber-intrachunk", "fiber-link"],
        "fiber_kit_session": ["fiber-session"],
    }

    def sort(self, recording, *, ns3_yaml=None, ground_truth_res=None,
             **kwargs) -> SortedResult:
        if not self._available:
            raise EnvironmentError("fiber_kit not available")

        from ..ns3.extractors import NS3SortingExtractor
        from spikeinterface.core import BaseRecording
        if not isinstance(recording, BaseRecording):
            raise TypeError("fiber_kit adapter needs a SpikeInterface BaseRecording")

        sess_base = recording.get_annotation("session_base", None)
        tmp_dir = None
        if sess_base is None:
            tmp_dir = tempfile.mkdtemp(prefix="ns3sb_fk_")
            sess_base = self._materialise_recording(recording, tmp_dir)

        env = {**os.environ, **self._env_overrides, **kwargs.get("env", {})}
        group = int(kwargs.get("electrode_group", 1))

        with measure_runtime() as rt:
            for stage in self._MODE_STAGES[self._mode]:
                self._run_stage(stage, sess_base, group, env=env,
                                 extra_args=kwargs.get("extra_args", []))

        variant = kwargs.get("variant", "stderiv_C5_D34")
        tag = kwargs.get("tag", "exp_final")
        sort_ex = NS3SortingExtractor(
            sess_base + ".yaml", electrode_group=group,
            variant=variant, tag=tag,
        )
        spike_times = {int(u): np.asarray(sort_ex.get_unit_spike_train(u), np.int64)
                       for u in sort_ex.get_unit_ids()}

        extras = dict(mode=self._mode, electrode_group=group,
                       variant=variant, tag=tag, session_base=sess_base)
        if tmp_dir and not kwargs.get("preserve_outputs", False):
            try: shutil.rmtree(tmp_dir)
            except OSError: pass
        elif tmp_dir:
            extras["output_dir"] = tmp_dir

        return SortedResult(
            unit_ids=sorted(spike_times), spike_times=spike_times,
            sampling_rate=recording.get_sampling_frequency(),
            adapter=self._mode,
            corpus_item=str(recording.get_annotation("name", sess_base)),
            adapter_version=self._version,
            channels=list(recording.get_channel_ids()),
            wall_time_s=rt["wall_time_s"], peak_rss_mb=rt["peak_rss_mb"],
            extras=extras,
        )

    def _materialise_recording(self, recording, tmp_dir: str) -> str:
        import yaml
        n_samp = recording.get_num_samples(0)
        n_ch = recording.get_num_channels()
        sr = recording.get_sampling_frequency()
        base = os.path.join(tmp_dir, "sim_session")
        traces = recording.get_traces(start_frame=0, end_frame=n_samp)
        traces_i16 = np.clip(np.rint(traces), -32768, 32767).astype(np.int16)
        traces_i16.tofile(base + ".fil")
        yml = {
            "parameters": {"version": "1.0", "creator": "ns3-sort-bench"},
            "acquisitionSystem": dict(nBits=16, nChannels=int(n_ch),
                                       voltageRange=20, amplification=1000,
                                       offset=0, samplingRate=float(sr)),
            "spikeDetection": {"channelGroups": [dict(
                channels=list(range(n_ch)),
                nSamples=32, peakSampleIndex=16, nFeatures=3,
            )]},
        }
        with open(base + ".yaml", "w") as f:
            yaml.safe_dump(yml, f, sort_keys=False)
        return base

    def _run_stage(self, stage, sess_base, group, *, env, extra_args):
        cmd = [stage, sess_base, str(group), *list(extra_args)]
        log_path = f"{sess_base}.{stage}.log"
        with open(log_path, "w") as logf:
            r = subprocess.run(cmd, env=env, stdout=logf,
                                stderr=subprocess.STDOUT, check=False)
        if r.returncode != 0:
            tail = open(log_path).read()[-2000:]
            raise RuntimeError(f"{stage} failed (exit {r.returncode}); log tail:\n{tail}")


class _Full(_FiberKitBase):
    name = "fiber_kit"; _mode = "fiber_kit"
    _env_overrides = {"FK_SESSION_DECONV": "1"}
    _contract_obj = _contract_for("fiber_kit", with_deconv=True)


class _NoDeconv(_FiberKitBase):
    name = "fiber_kit_no_deconv"; _mode = "fiber_kit_no_deconv"
    _env_overrides = {"FK_SESSION_DECONV": "0"}
    _contract_obj = _contract_for("fiber_kit_no_deconv", with_deconv=False)


class _SessionOnly(_FiberKitBase):
    name = "fiber_kit_session"; _mode = "fiber_kit_session"
    _contract_obj = _contract_for("fiber_kit_session", with_deconv=False)


_register(adapter_registry, _Full(), "fiber_kit")
_register(adapter_registry, _NoDeconv(), "fiber_kit_no_deconv")
_register(adapter_registry, _SessionOnly(), "fiber_kit_session")
