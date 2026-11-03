"""`slay_vs_fiberlink` — focused merge-step head-to-head.

Three legs:
    (A) Kilosort 4 + SLAy post-pass
    (B) fiber-kit full chain (anchor-link + xcorr-merge, cosine-based)
    (C) fiber-kit full chain WITH _unit_margin LLR routed to the merge
        decision sites (fiber-xcorr-merge, fiber-anchor-link's calibrated
        null, fiber-intrachunk)

Leg (C) is the "plumb _unit_margin to merge decisions" roadmap item.  Until
that lands, the runner only has (A) and (B); it still produces a comparable
dashboard row and the user can see the gap SLAy opens vs today's fiber-kit
linker.

Metrics focus:
    - oversplit units correctly merged (vs ground-truth parent)
    - correct-merge precision (merges that don't pair different cells)
    - under-merge rate (two cells left fused)
    - the overall PairwiseScorer ARI / purity / completeness delta
"""
from __future__ import annotations

from .. import BenchmarkRun, adapter_registry, scorer_registry
from .sorter_grid import _resolve_corpus


def run(corpus_names, output_dir="sort_benchmark_runs/slay_vs_fiberlink"):
    bench = BenchmarkRun(output_dir=output_dir)

    for name in corpus_names:
        bench.add_corpus(_resolve_corpus(name))

    for a_name in ("kilosort4", "slay", "fiber_kit"):
        a = adapter_registry.get(a_name)
        if a is not None:
            bench.add_adapter(a)

    for s_name in ("pairwise", "per_cell"):
        bench.add_scorer(scorer_registry[s_name])

    return bench.go()


if __name__ == "__main__":
    import sys
    df = run(sys.argv[1:] or ["propper_sim_small"])
    print(df.head())
