"""Persistent per-user plan state for ns3sb.

A `Plan` is just the four lists that drive BenchmarkRun — corpora,
adapters, scorers, output_dir — plus a couple of common knobs
(`ns3_yaml` path for Contract validation, `delta_frames` for the
pairwise scorer).  It lives at

    $XDG_CONFIG_HOME/ns3-sort-bench/plan.yaml
      (default: ~/.config/ns3-sort-bench/plan.yaml)

so `ns3sb load`, `ns3sb set`, and `ns3sb run` compose across
invocations without the user re-typing everything.  The file is plain
YAML; feel free to hand-edit it or delete it to start over
(`ns3sb reset` does the latter).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import yaml  # pyyaml is a core dep of ns3-sort-bench
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


_ENV_PATH = "NS3_SORT_BENCH_PLAN"


def _config_dir() -> Path:
    """Resolve the config dir following XDG rules, overridden by env."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "ns3-sort-bench"


def plan_path() -> Path:
    """The resolved plan.yaml path.  Honor $NS3_SORT_BENCH_PLAN first."""
    env = os.environ.get(_ENV_PATH)
    if env:
        return Path(env).expanduser()
    return _config_dir() / "plan.yaml"


# Fields the Plan supports.  Keep this list in sync with Plan.
_PLAN_FIELDS = (
    "corpora", "adapters", "scorers",
    "output_dir", "ns3_yaml", "delta_frames",
    "keep_fk_artifacts",
)


@dataclass
class Plan:
    """The active benchmark plan."""
    corpora:  List[str]           = field(default_factory=list)
    adapters: List[str]           = field(default_factory=list)
    scorers:  List[str]           = field(default_factory=list)
    output_dir:    str            = "runs/latest"
    ns3_yaml:      Optional[str]  = None
    delta_frames:  Optional[int]  = None   # for the pairwise scorer
    # When true, the fiber-kit adapter places its per-sort session dir
    # (.fil, .yaml, .res.standard.<g>, .spk.standard.<g>, fiber-kit.yaml,
    # every stage log, every .clu/.clc/.clp/.deconv it writes) under
    # <output_dir>/fiber_kit/<item_slug>/ and never deletes it.  Default
    # false — sorts go to a tmp dir that is cleaned up on success.
    keep_fk_artifacts: bool       = False

    # ─── load / save ────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Plan":
        p = path or plan_path()
        if not p.exists():
            return cls()
        if yaml is None:
            raise RuntimeError("pyyaml is required to read the plan file")
        with p.open("r") as fh:
            data = yaml.safe_load(fh) or {}
        # Tolerate extra keys written by a future ns3sb version
        kwargs = {k: data[k] for k in _PLAN_FIELDS if k in data}
        return cls(**kwargs)

    def save(self, path: Optional[Path] = None) -> Path:
        p = path or plan_path()
        if yaml is None:
            raise RuntimeError("pyyaml is required to write the plan file")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False)
        return p

    @classmethod
    def reset(cls, path: Optional[Path] = None) -> Path:
        p = path or plan_path()
        if p.exists():
            p.unlink()
        return p

    # ─── mutations ──────────────────────────────────────────────────────────
    def add_corpus(self, name: str) -> bool:
        if name in self.corpora:
            return False
        self.corpora.append(name)
        return True

    def drop_corpus(self, name: str) -> bool:
        if name not in self.corpora:
            return False
        self.corpora.remove(name)
        return True

    def set_adapters(self, names: List[str]) -> None:
        self.adapters = list(names)

    def set_scorers(self, names: List[str]) -> None:
        self.scorers = list(names)

    # ─── display ────────────────────────────────────────────────────────────
    def is_empty(self) -> bool:
        return not (self.corpora or self.adapters or self.scorers)

    def pretty(self) -> str:
        lines = [f"plan file : {plan_path()}"]
        lines.append("corpora   : " + (", ".join(self.corpora) if self.corpora else "(none)"))
        lines.append("adapters  : " + (", ".join(self.adapters) if self.adapters else "(auto: all available)"))
        lines.append("scorers   : " + (", ".join(self.scorers) if self.scorers else "pairwise, per_cell, propper_overlap"))
        lines.append(f"output_dir: {self.output_dir}")
        if self.ns3_yaml:
            lines.append(f"ns3_yaml  : {self.ns3_yaml}")
        if self.delta_frames is not None:
            lines.append(f"delta_frames: {self.delta_frames}")
        lines.append(f"keep_fk_artifacts: {self.keep_fk_artifacts}")
        return "\n".join(lines)
