"""`sorter_grid` — every available adapter × every registered scorer on one
or more corpora.  The default entry point for a benchmark run.

Usage (programmatic):

    from sort_benchmark import BenchmarkRun, adapter_registry, scorer_registry
    from sort_benchmark.runners.sorter_grid import default_run

    df = default_run(["propper_sim_small"])
    print(df.pivot_table("value", ["corpus_item", "candidate_adapter"], "metric"))

Usage (CLI):

    python -m sort_benchmark.runners.sorter_grid \\
        --corpus propper_sim_small \\
        --adapters fiber_kit,fiber_kit_no_deconv,botm_ref \\
        --out docs/bench/_runs/2026-11-02

Writes `scores.csv` and `sort_summaries.json` under `--out`.
"""
from __future__ import annotations

import argparse
import sys

from .. import BenchmarkRun, adapter_registry, scorer_registry
from .. import adapters as _  # noqa: F401 — populate registries
from .. import scorers as _s  # noqa: F401


def default_run(corpus_names, adapters=None, scorers=None,
                output_dir="sort_benchmark_runs/default"):
    """Build + run a sensible default grid.  `corpus_names` is a list of
    corpus-loader keys recognised by `_resolve_corpus`."""
    bench = BenchmarkRun(output_dir=output_dir)

    for name in corpus_names:
        bench.add_corpus(_resolve_corpus(name))

    # Default adapter set: everything available
    for a_name in (adapters or sorted(adapter_registry)):
        a = adapter_registry.get(a_name)
        if a is not None:
            bench.add_adapter(a)

    for s_name in (scorers or ("pairwise", "per_cell", "propper_overlap")):
        s = scorer_registry.get(s_name)
        if s is not None:
            bench.add_scorer(s)

    return bench.go()


def _resolve_corpus(name: str):
    """Hard-coded corpus registry.  Keeps the CLI self-contained; extend as
    corpora are added (SpikeForest loader, DARTsort hybrid loader, lab
    session loader, ...)."""
    if name.startswith("propper_sim"):
        from ..simulator.propper_overlaps import make_propper_overlap_corpus
        # Allow tiny / small / medium presets via suffix
        if name.endswith("_tiny"):
            return make_propper_overlap_corpus(
                n_samples=1 * 32552, n_units=2, rate_hz=15.0, name=name,
            )
        elif name.endswith("_medium"):
            return make_propper_overlap_corpus(
                n_samples=30 * 32552, n_units=5, rate_hz=10.0,
                overlap_fraction=0.25, name=name,
            )
        else:
            return make_propper_overlap_corpus(name=name)
    raise KeyError(f"unknown corpus key: {name} "
                   "(add a loader for it in sort_benchmark.runners.sorter_grid._resolve_corpus)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--corpus", required=True,
                    help="comma-separated corpus names")
    ap.add_argument("--adapters", default="",
                    help="comma-separated adapter names (default: all available)")
    ap.add_argument("--scorers", default="pairwise,per_cell,propper_overlap",
                    help="comma-separated scorer names")
    ap.add_argument("--out", default="sort_benchmark_runs/default",
                    help="output directory")
    args = ap.parse_args(argv)

    corpora = [c.strip() for c in args.corpus.split(",") if c.strip()]
    adapters = [a.strip() for a in args.adapters.split(",") if a.strip()] or None
    scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]

    df = default_run(corpora, adapters=adapters, scorers=scorers,
                     output_dir=args.out)
    print(df.head())
    return 0


if __name__ == "__main__":
    sys.exit(main())
