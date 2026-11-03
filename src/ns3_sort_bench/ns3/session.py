"""NS3 session yaml loader.  Single source of truth for the parsed-yaml
shape every adapter contract validates against."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import yaml


@dataclass(frozen=True)
class NS3Session:
    """Resolved view of a neurosuite-3 session yaml."""
    base: str                       # <dir>/<stem>, no extension
    sampling_rate: float            # Hz, raw samples
    n_channels_total: int           # total channels in .fil / .dat
    n_bits: int                     # 16 or 32
    voltage_range_mV: float
    amplification: float
    spike_groups: Tuple[Tuple[int, ...], ...]   # per group: tuple of int channel ids
    n_samples_per_spike: Tuple[int, ...]        # per group
    peak_sample: Tuple[int, ...]                # per group
    n_features: Tuple[int, ...]                 # per group
    raw_dict: Mapping[str, Any] = None          # populated at load time


def load_session_yaml(path: str | os.PathLike) -> NS3Session:
    """Parse an ns3 session yaml.  Accepts a bare stem — probes `.yaml`
    then `.yml` next to it.  Raises on `.xml` with a pointer to the
    legacy NeuroscopeRecordingExtractor."""
    p = Path(path)
    if p.suffix.lower() not in {".yaml", ".yml"}:
        for ext in (".yaml", ".yml"):
            if p.with_suffix(ext).exists():
                p = p.with_suffix(ext); break
        else:
            raise FileNotFoundError(
                f"{path!s}: no .yaml / .yml.  XML sessions use "
                "spikeinterface.extractors.NeuroscopeRecordingExtractor instead."
            )
    with open(p) as f:
        y = yaml.safe_load(f)
    base = str(p.with_suffix(""))

    acq = y["acquisitionSystem"]
    sd = y.get("spikeDetection", {}).get("channelGroups", []) or []

    def _channels(g):
        chs = g["channels"]
        return tuple(int(c["id"]) if isinstance(c, dict) else int(c) for c in chs)

    return NS3Session(
        base=base,
        sampling_rate=float(acq["samplingRate"]),
        n_channels_total=int(acq["nChannels"]),
        n_bits=int(acq.get("nBits", 16)),
        voltage_range_mV=float(acq.get("voltageRange", 20)),
        amplification=float(acq.get("amplification", 1)),
        spike_groups=tuple(_channels(g) for g in sd),
        n_samples_per_spike=tuple(int(g.get("nSamples", 0)) for g in sd),
        peak_sample=tuple(int(g.get("peakSampleIndex", 0)) for g in sd),
        n_features=tuple(int(g.get("nFeatures", 0)) for g in sd),
        raw_dict=y,
    )


def load_session_dict(path: str | os.PathLike) -> Dict[str, Any]:
    """Just the parsed yaml dict.  For adapter.contract.validate_yaml()."""
    return load_session_yaml(path).raw_dict
