"""`ns3sb run` — execute the active plan, or a one-shot override.

Precedence (left wins):
    CLI flags  >  active plan  >  defaults
"""
from __future__ import annotations

import sys

from .._util import eprint, import_registries, split_csv
from ..plan import Plan


def cmd_run(args) -> int:
    import_registries()
    from ... import BenchmarkRun, adapter_registry, scorer_registry
    from ...runners.sorter_grid import _resolve_corpus

    plan = Plan.load()

    # ─ resolve corpora ─
    corpus_keys = split_csv(args.corpus) or plan.corpora
    if not corpus_keys:
        eprint("no corpora given.  Load one with `ns3sb load <key>` or pass --corpus.")
        return 1

    # ─ resolve adapters ─
    adapter_names = split_csv(args.adapters) or plan.adapters
    if not adapter_names:
        # auto: every registered adapter that reports itself available
        adapter_names = [n for n, a in sorted(adapter_registry.items())
                         if a.available()[0]]
    missing_adapters = [n for n in adapter_names if n not in adapter_registry]
    if missing_adapters:
        eprint(f"unknown adapter(s): {missing_adapters}")
        eprint(f"  registered: {sorted(adapter_registry)}")
        return 1

    # ─ resolve scorers ─
    scorer_names = split_csv(args.scorers) or plan.scorers or \
                   [n for n in ("pairwise", "per_cell", "propper_overlap")
                    if n in scorer_registry]
    missing_scorers = [n for n in scorer_names if n not in scorer_registry]
    if missing_scorers:
        eprint(f"unknown scorer(s): {missing_scorers}")
        eprint(f"  registered: {sorted(scorer_registry)}")
        return 1

    output_dir = args.out or plan.output_dir

    if args.dry_run:
        print(f"output_dir : {output_dir}")
        print(f"corpora    : {corpus_keys}")
        print(f"adapters   : {adapter_names}")
        print(f"scorers    : {scorer_names}")
        if plan.ns3_yaml:    print(f"ns3_yaml   : {plan.ns3_yaml}")
        if plan.delta_frames: print(f"delta_frames: {plan.delta_frames}")
        return 0

    # ─ build + run ─
    adapter_kwargs = {}
    if plan.keep_fk_artifacts:
        # Every fiber-kit mode honours `fk_artifacts_dir`.  The adapter
        # places its session dir under `<fk_artifacts_dir>/<corpus_slug>/`
        # and never deletes it, so every .res / .spk / .clu / stage log
        # survives the sort for post-hoc inspection by Klusters / fiber-view.
        adapter_kwargs["fk_artifacts_dir"] = f"{output_dir}/fiber_kit"

    bench = BenchmarkRun(output_dir=output_dir, verbose=not args.quiet,
                          adapter_kwargs=adapter_kwargs)

    for key in corpus_keys:
        try:
            c = _resolve_corpus(key)
        except KeyError as e:
            eprint(str(e))
            return 1
        bench.add_corpus(c)

    for name in adapter_names:
        bench.add_adapter(adapter_registry[name])
    for name in scorer_names:
        bench.add_scorer(scorer_registry[name])

    df = bench.go()
    if not args.quiet:
        print(f"\nwrote {output_dir}/scores.csv  ({len(df)} rows)")
        if plan.keep_fk_artifacts:
            print(f"fiber-kit artifacts kept under {output_dir}/fiber_kit/")
    return 0


def register(subparsers) -> None:
    ap = subparsers.add_parser("run",
                               help="run the active plan (or one-shot via flags)")
    ap.add_argument("--corpus", default="",
                    help="comma-separated corpus keys; overrides the plan's corpora")
    ap.add_argument("--adapters", default="",
                    help="comma-separated adapter names (default: all available)")
    ap.add_argument("--scorers", default="",
                    help="comma-separated scorer names "
                         "(default: pairwise,per_cell,propper_overlap)")
    ap.add_argument("--out", default="",
                    help="output directory (default: plan.output_dir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved plan and exit without running")
    ap.add_argument("--quiet", action="store_true",
                    help="silence per-step progress messages")
    ap.set_defaults(fn=cmd_run)
