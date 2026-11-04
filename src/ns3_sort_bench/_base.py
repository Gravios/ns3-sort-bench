"""Base types for ns3-sort-bench.

Public:
    SortedResult  — frozen record of one sorter's output
    SorterAdapter — ABC every sorter adapter subclasses; carries a Contract
    Scorer        — ABC every metric subclasses
    ScoreResult   — Scorer output
    Corpus        — recording + ground truth
    BenchmarkRun  — orchestrator (corpus × adapter × scorer grid)
    measure_runtime — contextmanager for wall-time + peak-RSS
    adapter_registry, scorer_registry — populated by self-registering modules
"""
from __future__ import annotations

import json
import os
import resource
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .contracts import Contract


adapter_registry: Dict[str, "SorterAdapter"] = {}
scorer_registry: Dict[str, "Scorer"] = {}


def _register(registry, obj, name=None):
    key = name or getattr(obj, "name", obj.__class__.__name__)
    if key in registry:
        raise KeyError(f"duplicate registration: {key} (existing: {registry[key]})")
    registry[key] = obj
    return obj


# ─── SortedResult ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SortedResult:
    """One sorter's output on one corpus item.  Frozen; pass by reference.

    See `ns3_sort_bench.contracts.OutputSchema` for the fields' semantics —
    the adapter's Contract declares what shape THIS SortedResult will have.
    """
    unit_ids: Sequence[int]
    spike_times: Mapping[int, np.ndarray]
    sampling_rate: float
    adapter: str
    corpus_item: str
    adapter_version: str = ""
    templates: Optional[np.ndarray] = None
    channels: Optional[Sequence[int]] = None
    wall_time_s: float = 0.0
    peak_rss_mb: float = 0.0
    extras: Mapping[str, Any] = field(default_factory=dict)

    def n_units(self) -> int: return len(self.unit_ids)
    def n_spikes(self) -> int: return int(sum(len(t) for t in self.spike_times.values()))

    def dropped_reserved(self) -> "SortedResult":
        """Drop Klusters-convention IDs 0 (artefact) and 1 (noise)."""
        keep = [u for u in self.unit_ids if int(u) >= 2]
        return SortedResult(
            unit_ids=keep,
            spike_times={u: self.spike_times[u] for u in keep if u in self.spike_times},
            sampling_rate=self.sampling_rate,
            adapter=self.adapter, corpus_item=self.corpus_item,
            adapter_version=self.adapter_version, templates=self.templates,
            channels=self.channels, wall_time_s=self.wall_time_s,
            peak_rss_mb=self.peak_rss_mb, extras=dict(self.extras),
        )

    def to_meta(self) -> dict:
        return dict(
            adapter=self.adapter, corpus_item=self.corpus_item,
            adapter_version=self.adapter_version,
            n_units=self.n_units(), n_spikes=self.n_spikes(),
            wall_time_s=round(self.wall_time_s, 3),
            peak_rss_mb=round(self.peak_rss_mb, 1),
            extras=dict(self.extras),
        )


# ─── SorterAdapter ───────────────────────────────────────────────────────────
class SorterAdapter(ABC):
    """One method per sorter.  Each subclass:

    1. Sets `name` (unique registry key)
    2. Declares a module-level `CONTRACT` of type Contract
    3. Implements `.sort(recording, **kwargs) -> SortedResult`
    4. Implements `.available() -> (bool, str)` — reports dependency state

    The `contract` property returns the module's CONTRACT; the harness
    uses it to pre-validate (yaml, recording) before each sort.
    """
    name: str = ""

    @property
    @abstractmethod
    def contract(self) -> Contract:
        """The adapter's formal interface declaration.  Must be the same
        Contract instance across calls (used for identity in dashboards)."""
        ...

    @property
    def version(self) -> str:
        """Upstream package version string (what adapter.sort will invoke)."""
        return "unknown"

    @abstractmethod
    def sort(self, recording, *, ns3_yaml: Optional[Mapping[str, Any]] = None,
             ground_truth_res: Optional[np.ndarray] = None,
             **kwargs) -> SortedResult:
        """Run the sorter.  `ns3_yaml`, when given, is the parsed session
        yaml; the harness also hands it to validate_yaml BEFORE this call."""
        ...

    def available(self) -> tuple[bool, str]:
        """Default: available.  Override to probe for optional deps / GPU."""
        return True, ""


# ─── Scorer ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ScoreResult:
    scorer: str
    corpus_item: str
    reference_adapter: str
    candidate_adapter: str
    values: Mapping[str, float]
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_rows(self) -> List[dict]:
        base = dict(scorer=self.scorer, corpus_item=self.corpus_item,
                    reference_adapter=self.reference_adapter,
                    candidate_adapter=self.candidate_adapter)
        return [{**base, "metric": k, "value": float(v)}
                for k, v in self.values.items()]


class Scorer(ABC):
    name: str = ""

    @abstractmethod
    def __call__(self, reference: SortedResult, candidate: SortedResult,
                 **kwargs) -> ScoreResult:
        ...


# ─── Corpus ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Corpus:
    """One corpus item: recording + ground truth + (optionally) the ns3
    yaml dict the harness uses to validate adapter contracts against."""
    name: str
    recording: Any
    ground_truth: SortedResult
    description: str = ""
    kind: str = "lab"
    ns3_yaml: Optional[Mapping[str, Any]] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


# ─── runtime measurement ────────────────────────────────────────────────────
@contextmanager
def measure_runtime():
    out: dict = {}
    t0 = time.perf_counter()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        yield out
    finally:
        out["wall_time_s"] = time.perf_counter() - t0
        rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: ru_maxrss is kilobytes; macOS: bytes
        rss_mb = (max(rss0, rss1) / 1024.0) if rss1 > 1e5 \
                  else (max(rss0, rss1) / (1024.0 * 1024.0))
        out["peak_rss_mb"] = rss_mb


def _as_dict(obj):
    """Shallow dataclass → dict; preserves nested frozen dataclasses."""
    return {k: getattr(obj, k) for k in obj.__dataclass_fields__}


# ─── BenchmarkRun ───────────────────────────────────────────────────────────
@dataclass
class BenchmarkRun:
    """Orchestrates (corpus × adapter × scorer) grids.  Each adapter's
    Contract is validated against (ns3_yaml, recording) before the sort;
    violations produce a skip with reasons.

    `adapter_kwargs` is forwarded to every `adapter.sort()` call.  Adapters
    ignore keys they don't know.  Current contract:

        fk_artifacts_dir : str  — the fiber-kit family places its per-sort
                                   session dir at
                                   `<fk_artifacts_dir>/<corpus_item_slug>/`
                                   and does not delete it afterwards.
                                   Wired from `ns3sb set keep-fk-artifacts`.
    """
    output_dir: str = "runs"
    corpora: List[Corpus] = field(default_factory=list)
    adapters: List[SorterAdapter] = field(default_factory=list)
    scorers: List[Scorer] = field(default_factory=list)
    verbose: bool = True
    adapter_kwargs: Mapping[str, Any] = field(default_factory=dict)
    # When true, after every corpus item is fully sorted, write a
    # stand-alone Klusters-openable session for each adapter (plus the
    # ground truth) at `<output_dir>/klusters/<item>/<adapter>/` via
    # `ns3_sort_bench.klusters_export.export_klusters_session`.  Requires
    # neurosuite-3's `ndm_extractspikes` and `ndm_pca` on PATH.
    export_klusters: bool = False

    def add_corpus(self, c: Corpus) -> "BenchmarkRun":
        self.corpora.append(c); return self

    def add_adapter(self, a: SorterAdapter) -> "BenchmarkRun":
        self.adapters.append(a); return self

    def add_scorer(self, s: Scorer) -> "BenchmarkRun":
        self.scorers.append(s); return self

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[ns3-sort-bench] {msg}", flush=True)

    def go(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("ns3-sort-bench requires pandas")

        os.makedirs(self.output_dir, exist_ok=True)
        rows: List[dict] = []
        sort_summaries: List[dict] = []

        for corpus in self.corpora:
            self._log(f"corpus: {corpus.name} ({corpus.kind})")
            ref = corpus.ground_truth
            # Successful candidates for THIS corpus, keyed by adapter name;
            # consumed by the Klusters export at the end of the inner loop.
            candidates_for_corpus: Dict[str, SortedResult] = {}

            for adapter in self.adapters:
                ok, reason = adapter.available()
                if not ok:
                    self._log(f"  skipping {adapter.name}: {reason}")
                    continue

                # Contract validation
                if corpus.ns3_yaml is not None:
                    contract_ok, contract_reasons = adapter.contract.validate(
                        corpus.ns3_yaml, corpus.recording,
                    )
                    if not contract_ok:
                        self._log(f"  skipping {adapter.name}: contract: "
                                  + "; ".join(contract_reasons))
                        sort_summaries.append(dict(
                            corpus_item=corpus.name, adapter=adapter.name,
                            error="contract-violation", reasons=contract_reasons,
                        ))
                        continue

                self._log(f"  running {adapter.name} v{adapter.version}...")
                try:
                    with measure_runtime() as rt:
                        candidate = adapter.sort(
                            corpus.recording, ns3_yaml=corpus.ns3_yaml,
                            corpus_name=corpus.name,
                            **dict(self.adapter_kwargs),
                        )
                    # The corpus is the authoritative source of the item name
                    # (it drives the task list on the ref side too), so
                    # override whatever the adapter set.  Fixes a class of bugs
                    # where an adapter falls back to None/"None"/a tmp path —
                    # pandas's default `na_values` contains "None", so a row
                    # written with `corpus_item="None"` read back as NaN and
                    # silently vanished from pivots.
                    candidate = SortedResult(**{
                        **_as_dict(candidate),
                        "corpus_item": corpus.name,
                        "wall_time_s": rt["wall_time_s"],
                        "peak_rss_mb": rt["peak_rss_mb"],
                    })
                except Exception as e:
                    self._log(f"    FAILED: {type(e).__name__}: {e}")
                    sort_summaries.append(dict(
                        corpus_item=corpus.name, adapter=adapter.name,
                        error=f"{type(e).__name__}: {e}",
                        traceback=traceback.format_exc(limit=8),
                    ))
                    continue

                sort_summaries.append(candidate.to_meta())
                self._log(f"    {candidate.n_units()} units, "
                          f"{candidate.n_spikes():,} spikes, "
                          f"{candidate.wall_time_s:.1f}s, "
                          f"{candidate.peak_rss_mb:.0f} MB")
                candidates_for_corpus[adapter.name] = candidate

                for scorer in self.scorers:
                    try:
                        rows.extend(scorer(ref, candidate).to_rows())
                    except Exception as e:
                        self._log(f"    scorer {scorer.name} FAILED: {type(e).__name__}: {e}")

            # End of adapter loop for THIS corpus.  Write per-adapter Klusters
            # sessions (via neurosuite-3's ndm_* binaries) when requested.
            if self.export_klusters and candidates_for_corpus:
                try:
                    from .klusters_export import export_klusters_session
                    item_dir = export_klusters_session(
                        self.output_dir, corpus, candidates_for_corpus,
                    )
                    self._log(f"  klusters sessions written under {item_dir}")
                except Exception as e:
                    self._log(f"  klusters export FAILED: {type(e).__name__}: {e}")

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.output_dir, "scores.csv"), index=False)
        with open(os.path.join(self.output_dir, "sort_summaries.json"), "w") as f:
            json.dump(sort_summaries, f, indent=2, default=str)
        self._log(f"wrote {os.path.join(self.output_dir, 'scores.csv')} ({len(rows)} rows)")
        return df
