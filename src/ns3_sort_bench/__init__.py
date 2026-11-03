"""ns3-sort-bench — cross-sorter benchmark harness for neurosuite-3 / fiber-kit sessions.

Public API:

    from ns3_sort_bench import (
        SortedResult, SorterAdapter, Scorer, ScoreResult, Contract,
        BenchmarkRun, Corpus,
        adapter_registry, scorer_registry,
    )
    import ns3_sort_bench.adapters   # populate adapter_registry
    import ns3_sort_bench.scorers    # populate scorer_registry

    # Then build a BenchmarkRun — see README.md for the usage recipes.

Each adapter declares a formal `Contract` describing its ns3-yaml interface,
preconditions, output schema, and capabilities (see `CONTRACTS.md`).  The
harness validates every (yaml, recording) ↔ contract pair before invoking
the sorter so misconfigurations surface as skip reasons, not mid-sort tracebacks.
"""
from ._base import (
    SortedResult, SorterAdapter, Scorer, ScoreResult,
    BenchmarkRun, Corpus, adapter_registry, scorer_registry,
    measure_runtime,
)
from .contracts import (
    Contract, RecordingReqs, OutputSchema, Capabilities, ResourceEstimates,
)

__all__ = [
    "SortedResult", "SorterAdapter", "Scorer", "ScoreResult",
    "BenchmarkRun", "Corpus", "adapter_registry", "scorer_registry",
    "measure_runtime",
    "Contract", "RecordingReqs", "OutputSchema", "Capabilities", "ResourceEstimates",
]

__version__ = "0.1.0"
