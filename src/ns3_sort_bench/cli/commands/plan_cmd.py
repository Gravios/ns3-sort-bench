"""`ns3sb load / unload / set / show / reset` — manage the active plan."""
from __future__ import annotations

from .._util import eprint, split_csv
from ..plan import Plan, plan_path


# ─── load / unload ───────────────────────────────────────────────────────────
def cmd_load(args) -> int:
    plan = Plan.load()
    for key in args.corpus:
        for c in split_csv(key):
            if plan.add_corpus(c):
                print(f"loaded  {c}")
            else:
                eprint(f"(skip)  {c} already in plan")
    plan.save()
    print(f"\nplan: {plan_path()}")
    return 0


def cmd_unload(args) -> int:
    plan = Plan.load()
    for key in args.corpus:
        for c in split_csv(key):
            if plan.drop_corpus(c):
                print(f"unload  {c}")
            else:
                eprint(f"(skip)  {c} not in plan")
    plan.save()
    return 0


# ─── set ────────────────────────────────────────────────────────────────────
def cmd_set_adapters(args) -> int:
    plan = Plan.load()
    plan.set_adapters(split_csv(args.value))
    plan.save()
    print(f"adapters = {plan.adapters or '(auto)'}")
    return 0


def cmd_set_scorers(args) -> int:
    plan = Plan.load()
    plan.set_scorers(split_csv(args.value))
    plan.save()
    print(f"scorers = {plan.scorers or '(defaults: pairwise, per_cell, propper_overlap)'}")
    return 0


def cmd_set_output(args) -> int:
    plan = Plan.load()
    plan.output_dir = args.value
    plan.save()
    print(f"output_dir = {plan.output_dir}")
    return 0


def cmd_set_yaml(args) -> int:
    plan = Plan.load()
    plan.ns3_yaml = args.value or None
    plan.save()
    print(f"ns3_yaml = {plan.ns3_yaml}")
    return 0


def cmd_set_delta(args) -> int:
    plan = Plan.load()
    try:
        plan.delta_frames = int(args.value)
    except ValueError:
        eprint(f"delta-frames must be an integer, got {args.value!r}")
        return 1
    plan.save()
    print(f"delta_frames = {plan.delta_frames}")
    return 0


# ─── show / reset ───────────────────────────────────────────────────────────
def cmd_show(args) -> int:
    plan = Plan.load()
    print(plan.pretty())
    if plan.is_empty():
        print("\n(empty — try `ns3sb load paired_crcns_hc1:<item>`)")
    return 0


def cmd_reset(args) -> int:
    path = Plan.reset()
    print(f"removed {path}" if path else "(no plan file)")
    return 0


# ─── argparse wiring ─────────────────────────────────────────────────────────
def register(subparsers) -> None:
    ld = subparsers.add_parser("load", help="add corpus key(s) to the active plan")
    ld.add_argument("corpus", nargs="+",
                    help="one or more corpus keys, e.g. "
                         "paired_crcns_hc1:d533101 propper_sim_tiny")
    ld.set_defaults(fn=cmd_load)

    un = subparsers.add_parser("unload", help="remove corpus key(s) from the active plan")
    un.add_argument("corpus", nargs="+")
    un.set_defaults(fn=cmd_unload)

    st = subparsers.add_parser("set", help="set a plan option (adapters, scorers, output, ...)")
    st_sub = st.add_subparsers(dest="set_what", required=True)

    for key, fn, help_ in (
        ("adapters",     cmd_set_adapters,
         "comma-separated adapter names (empty → auto: all available)"),
        ("scorers",      cmd_set_scorers,
         "comma-separated scorer names (empty → pairwise,per_cell,propper_overlap)"),
        ("output",       cmd_set_output,
         "output directory"),
        ("yaml",         cmd_set_yaml,
         "path to the ns3 session yaml for Contract validation (empty → clear)"),
        ("delta-frames", cmd_set_delta,
         "pairwise-scorer spike-match tolerance in samples (int)"),
    ):
        sp = st_sub.add_parser(key, help=help_)
        sp.add_argument("value", help=help_)
        sp.set_defaults(fn=fn)

    subparsers.add_parser("show", help="print the active plan"
                          ).set_defaults(fn=cmd_show)
    subparsers.add_parser("reset", help="delete the plan file"
                          ).set_defaults(fn=cmd_reset)
