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
import os as _os

# OpenBLAS builds cap their thread count (64 or 128).  On a machine with
# more logical CPUs than that it warns, and once several libraries call
# BLAS from their own threads at once (Kilosort 4 after SpikeInterface
# workers) it aborts: "too many memory regions", then a segfault.  It
# reads the variable when numpy loads, so set it before the first
# import below; a value the user exported wins.
if "OPENBLAS_NUM_THREADS" not in _os.environ:
    _os.environ["OPENBLAS_NUM_THREADS"] = str(
        min(32, max(1, len(_os.sched_getaffinity(0))
                    if hasattr(_os, "sched_getaffinity") else
                    (_os.cpu_count() or 1))))

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
