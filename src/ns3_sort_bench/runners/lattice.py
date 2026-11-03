"""`lattice` — fiber_deconv ablation lattice.

Enumerates every combination of the five independent ablation axes:

    overlap_rule     ∈ {'omp', 'sic-a', 'sic-b', 'sic-c'}  (4)
    sub_sample_bank  ∈ {'integer', 'R4'}                    (2)
    amplitude_prior  ∈ {'none', 'gaussian'}                 (2)
    noise_model      ∈ {'spatial', 'var4'}                  (2)

4 × 2 × 2 × 2 = 32 configurations.  Each is an adapter registered by
`../adapters/lattice_botm.py` as `lattice_<rule>_<R>_<prior>_<noise>`.

The baselines to compare against:
    `botm_ref`              — literal thesis BOTM (zero extensions)
    `lattice_sic-b_integer_none_var4`  — closest to the thesis;
                                          only the VAR(4) noise differs
    `lattice_omp_R4_gaussian_var4`     — closest to production fiber_deconv

Usage (CLI):
    python -m sort_benchmark.runners.lattice \\
        --corpus propper_sim_medium \\
        --out _runs/lattice-2026-11-02

Usage (programmatic):
    from sort_benchmark.runners.lattice import lattice_run, select_lattice
    df = lattice_run(["propper_sim_medium"],
                      configs=select_lattice(rules=("sic-b", "omp"),
                                              banks=("integer", "R4")))
"""
from __future__ import annotations

import argparse
import itertools
import sys
from typing import Iterable, Optional, Sequence

from .. import BenchmarkRun, adapter_registry, scorer_registry
from ..adapters.lattice_botm import RULES, BANKS, PRIORS, NOISES


def select_lattice(*, rules: Optional[Sequence[str]] = None,
                    banks: Optional[Sequence[str]] = None,
                    priors: Optional[Sequence[str]] = None,
                    noises: Optional[Sequence[str]] = None):
    """Return the subset of lattice adapter names matching the filters.
    None on any axis means 'all values'."""
    rr = rules or RULES; bb = banks or BANKS
    pp = priors or PRIORS; nn = noises or NOISES
    return [f"lattice_{r}_{b}_{p}_{n}"
            for r, b, p, n in itertools.product(rr, bb, pp, nn)]


def lattice_run(corpus_names, configs=None, *,
                include_baselines=True,
                output_dir="sort_benchmark_runs/lattice"):
    """Run the (sub)lattice on `corpus_names`.

    Parameters
    ----------
    corpus_names : iterable of corpus keys (see
        runners.sorter_grid._resolve_corpus)
    configs : optional subset of lattice adapter names.  None = full 32.
    include_baselines : also run `botm_ref` and `fiber_kit` for comparison
    output_dir : run directory for scores.csv + sort_summaries.json
    """
    from .sorter_grid import _resolve_corpus
    from .. import adapters as _  # noqa: F401 — populate registry

    bench = BenchmarkRun(output_dir=output_dir)
    for name in corpus_names:
        bench.add_corpus(_resolve_corpus(name))

    cfg_names = list(configs) if configs is not None else select_lattice()
    for n in cfg_names:
        a = adapter_registry.get(n)
        if a is None:
            continue
        bench.add_adapter(a)

    if include_baselines:
        for base_name in ("botm_ref", "fiber_kit"):
            a = adapter_registry.get(base_name)
            if a is not None:
                ok, _ = a.available()
                if ok:
                    bench.add_adapter(a)

    for s_name in ("pairwise", "per_cell", "propper_overlap"):
        bench.add_scorer(scorer_registry[s_name])

    return bench.go()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--corpus", required=True,
                    help="comma-separated corpus names")
    ap.add_argument("--rules", default="",
                    help="comma-separated subset of {omp,sic-a,sic-b,sic-c}")
    ap.add_argument("--banks", default="",
                    help="comma-separated subset of {integer,R4}")
    ap.add_argument("--priors", default="",
                    help="comma-separated subset of {none,gaussian}")
    ap.add_argument("--noises", default="",
                    help="comma-separated subset of {spatial,var4}")
    ap.add_argument("--no-baselines", action="store_true",
                    help="skip botm_ref + fiber_kit baselines")
    ap.add_argument("--out", default="sort_benchmark_runs/lattice",
                    help="output directory")
    args = ap.parse_args(argv)

    def _split(s):
        return [x.strip() for x in s.split(",") if x.strip()] or None

    configs = select_lattice(
        rules=_split(args.rules),
        banks=_split(args.banks),
        priors=_split(args.priors),
        noises=_split(args.noises),
    )
    print(f"[lattice] running {len(configs)} configuration(s)")
    df = lattice_run(
        [c.strip() for c in args.corpus.split(",") if c.strip()],
        configs=configs,
        include_baselines=not args.no_baselines,
        output_dir=args.out,
    )
    print(df.head())
    return 0


if __name__ == "__main__":
    sys.exit(main())
