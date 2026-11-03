"""Contracts — formal declarations of how each adapter interfaces with the
neurosuite-3 session YAML.

Every `SorterAdapter` in this harness declares a `Contract` instance that
pins down:

1. **YAML fields it reads** — the exact dotted paths into the ns3 session
   yaml it depends on, split into required and optional.  The harness
   validates each session yaml against this before invoking the adapter;
   missing required fields fail fast with an actionable error.
2. **Recording preconditions** — min channels, dtype, min duration, whether
   probe geometry must be set, whether the recording must have been
   whitened / filtered / etc.
3. **Output schema** — unit-id convention (does the adapter emit reserved
   IDs 0 and 1 as artefact/noise?  does it start from 2?  does it collide
   with the Klusters reserved range?), whether templates are returned,
   whether per-group vs whole-session sorting.
4. **Capabilities** — drift-aware, overlap-resolution semantics, GPU
   requirements, supports fiber-kit's stderiv-transformed input.
5. **Resource estimates** — min RAM, min CPU cores, GPU VRAM, so the
   runner can skip adapters it can't afford on the current host.
6. **Known limitations** — free-text, surfaced in the dashboard so results
   carry their caveats.
7. **Upstream version pin** — the exact `pyproject.toml` version pin the
   adapter was tested against.

The `validate()` method on a Contract takes the yaml + the recording and
returns `(ok, list_of_reasons)`.  The harness calls it before every sort,
so a session that doesn't satisfy an adapter's preconditions produces a
clear skip message instead of a mid-sort traceback.

Each adapter's Contract is a module-level constant.  The full catalog is
documented in `CONTRACTS.md` at the project root — regenerate that
document with `ns3-sort-bench --dump-contracts > CONTRACTS.md`.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# ─── sub-records ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RecordingReqs:
    """Preconditions on the recording the adapter sees.

    `signal_space` controls whether the adapter expects raw, filtered, or
    stderiv-transformed signal — the harness forwards this to the recording
    factory so a single yaml can be served to multiple adapters at their
    preferred space.
    """
    min_channels: int = 1
    max_channels: Optional[int] = None
    min_duration_s: float = 0.1
    dtype: Tuple[str, ...] = ("int16", "float32", "float64")
    probe_geometry_required: bool = False
    signal_space: str = "raw"   # "raw" | "filtered" | "stderiv" | "whitened"
    sampling_rate_min_hz: float = 1_000.0
    sampling_rate_max_hz: float = 50_000.0


@dataclass(frozen=True)
class OutputSchema:
    """What the SortedResult produced by this adapter looks like.

    `unit_id_offset`: non-zero means the adapter remaps its native IDs so
    they start at `unit_id_offset`.  Default 0 is "whatever upstream
    emits"; the Klusters convention is 2 (reserving 0 for artefacts and 1
    for noise).

    `includes_reserved`: True if the adapter emits unit IDs < 2 (artefact /
    noise clusters that the Klusters-convention scorers filter out).

    `templates_emitted`: True if the adapter populates `SortedResult.templates`.

    `per_group_or_whole`: "per_group" for adapters that run on one
    electrode-group's worth of channels at a time (the lab convention);
    "whole_session" for adapters that take the entire multi-group recording
    at once (DARTsort / Kilosort 4 on Neuropixels).
    """
    unit_id_offset: int = 0
    includes_reserved: bool = False
    templates_emitted: bool = False
    per_group_or_whole: str = "per_group"   # "per_group" | "whole_session"


@dataclass(frozen=True)
class Capabilities:
    """What the adapter is DESIGNED to do, independent of what it does well.

    Used by the runner to pre-filter corpora: a "drift_aware=False"
    adapter run against a known-drifting session gets flagged in the
    dashboard as "run outside its design envelope".
    """
    drift_aware: bool = False
    overlap_resolution: str = "none"   # "none" | "iterative_subtract" | "sic" | "omp"
    requires_gpu: bool = False
    requires_torch: bool = False
    supports_stderiv_input: bool = False
    supports_shared_detection: bool = False   # can accept ground-truth .res as detection
    reports_templates: bool = False
    streaming: bool = False   # true if the adapter processes the recording in chunks


@dataclass(frozen=True)
class ResourceEstimates:
    """Rough estimates per-call.  Not enforced; surfaced in the dashboard
    so a 2-core container can filter out the 32-GB adapters."""
    min_ram_mb: int = 512
    min_cpu_cores: int = 1
    gpu_vram_mb: Optional[int] = None   # None = no GPU
    estimated_sec_per_minute_of_recording: float = 10.0


# ─── the contract ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Contract:
    """One adapter's formal interface declaration.

    See the module docstring for the full semantics; see
    `CONTRACTS.md` for the per-adapter catalog.
    """
    adapter_name: str
    upstream_url: str
    upstream_version_pin: str              # exact `pyproject.toml` pin
    yaml_fields_required: Tuple[str, ...]  # dotted paths into ns3 yaml
    yaml_fields_optional: Tuple[str, ...]
    recording: RecordingReqs = field(default_factory=RecordingReqs)
    output: OutputSchema = field(default_factory=OutputSchema)
    capabilities: Capabilities = field(default_factory=Capabilities)
    resources: ResourceEstimates = field(default_factory=ResourceEstimates)
    known_limitations: Tuple[str, ...] = ()
    notes: str = ""

    # ─── validators ──────────────────────────────────────────────────────
    def validate_yaml(self, yaml_dict: Mapping[str, Any]) -> Tuple[bool, List[str]]:
        """Check the yaml dict against `yaml_fields_required`.  Returns
        (ok, reasons).  Dotted paths use `.` as the separator; list indices
        use `[i]` suffix on the preceding key, e.g.
            `spikeDetection.channelGroups[0].channels`.
        """
        reasons = []
        for path in self.yaml_fields_required:
            try:
                _dotted_get(yaml_dict, path)
            except (KeyError, IndexError, TypeError) as e:
                reasons.append(f"required yaml field missing: {path} ({e})")
        return (not reasons), reasons

    def validate_recording(self, recording) -> Tuple[bool, List[str]]:
        """Check a SpikeInterface recording against `self.recording`.
        Returns (ok, reasons).  Recording must be a SI BaseRecording; the
        checks are duck-typed so a stub without SI installed still runs."""
        reasons = []
        r = self.recording
        try:
            n_ch = int(recording.get_num_channels())
            sr = float(recording.get_sampling_frequency())
            n_samp = int(recording.get_num_samples(0))
            dur = n_samp / sr
        except AttributeError as e:
            return False, [f"not a SpikeInterface recording: {e}"]

        if n_ch < r.min_channels:
            reasons.append(f"recording has {n_ch} channels, min {r.min_channels}")
        if r.max_channels is not None and n_ch > r.max_channels:
            reasons.append(f"recording has {n_ch} channels, max {r.max_channels}")
        if dur < r.min_duration_s:
            reasons.append(f"recording is {dur:.1f}s, min {r.min_duration_s}s")
        if sr < r.sampling_rate_min_hz or sr > r.sampling_rate_max_hz:
            reasons.append(
                f"sampling rate {sr:.0f} Hz outside "
                f"[{r.sampling_rate_min_hz:.0f}, {r.sampling_rate_max_hz:.0f}]"
            )
        if r.probe_geometry_required:
            try:
                locs = recording.get_channel_locations()
                if locs is None or len(locs) == 0:
                    reasons.append("probe geometry required but recording has none")
            except Exception:
                reasons.append("probe geometry required but recording.get_channel_locations() failed")
        return (not reasons), reasons

    def validate(self, yaml_dict: Mapping[str, Any], recording) -> Tuple[bool, List[str]]:
        """Combined yaml + recording validation.  Convenience wrapper the
        harness runs before each sort."""
        y_ok, y_reasons = self.validate_yaml(yaml_dict)
        r_ok, r_reasons = self.validate_recording(recording)
        return (y_ok and r_ok), (y_reasons + r_reasons)

    # ─── documentation ───────────────────────────────────────────────────
    def to_markdown(self) -> str:
        """Render this contract as a markdown section (for CONTRACTS.md)."""
        lines = [f"### `{self.adapter_name}`", "",
                 f"- Upstream: <{self.upstream_url}>",
                 f"- Version pin: `{self.upstream_version_pin}`",
                 f"- Signal space: `{self.recording.signal_space}`",
                 f"- Output: `{self.output.per_group_or_whole}` sorting, "
                 f"unit IDs {'start at ' + str(self.output.unit_id_offset) if self.output.unit_id_offset else 'as-emitted'}, "
                 f"reserved IDs {'INCLUDED' if self.output.includes_reserved else 'excluded'}, "
                 f"templates {'YES' if self.output.templates_emitted else 'NO'}", ""]

        if self.yaml_fields_required:
            lines.append("**Required ns3-yaml fields:**")
            for p in self.yaml_fields_required:
                lines.append(f"- `{p}`")
            lines.append("")
        if self.yaml_fields_optional:
            lines.append("**Optional ns3-yaml fields:**")
            for p in self.yaml_fields_optional:
                lines.append(f"- `{p}`")
            lines.append("")

        c = self.capabilities
        caps = []
        if c.drift_aware: caps.append("drift-aware")
        if c.overlap_resolution != "none":
            caps.append(f"overlap resolution: {c.overlap_resolution}")
        if c.requires_gpu: caps.append("requires GPU")
        if c.requires_torch: caps.append("requires torch")
        if c.supports_stderiv_input: caps.append("accepts stderiv-transformed input")
        if c.supports_shared_detection: caps.append("supports shared-detection runs")
        if c.streaming: caps.append("streaming")
        if caps:
            lines.append("**Capabilities:** " + "; ".join(caps))
            lines.append("")

        r = self.resources
        lines.append(f"**Resources (per call):** ≥ {r.min_ram_mb} MB RAM, "
                     f"≥ {r.min_cpu_cores} cores"
                     + (f", {r.gpu_vram_mb} MB GPU" if r.gpu_vram_mb else "")
                     + f", ~{r.estimated_sec_per_minute_of_recording:.0f}s per minute of recording")
        lines.append("")

        rec = self.recording
        lines.append(f"**Recording preconditions:** {rec.min_channels}+ channels, "
                     f"≥ {rec.min_duration_s:.1f}s, dtype ∈ {rec.dtype}"
                     + (", probe geometry required" if rec.probe_geometry_required else ""))
        lines.append("")

        if self.known_limitations:
            lines.append("**Known limitations:**")
            for lim in self.known_limitations:
                lines.append(f"- {lim}")
            lines.append("")

        if self.notes:
            lines.append(f"**Notes:** {self.notes}")
            lines.append("")

        return "\n".join(lines)


# ─── dotted-path helper ──────────────────────────────────────────────────────
_INDEX_RE = re.compile(r"\[(\d+)\]$")


def _dotted_get(d: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted path with optional [i] indices.  Supports lists
    whose entries are indexed by number: `spikeDetection.channelGroups[0]`.

    Raises KeyError / IndexError / TypeError on miss, as appropriate.
    """
    cur: Any = d
    for part in path.split("."):
        m = _INDEX_RE.search(part)
        if m:
            key = part[:m.start()]
            idx = int(m.group(1))
            cur = cur[key][idx]
        else:
            cur = cur[part]
    return cur
