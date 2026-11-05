"""ns3sb — argparse glue.  Each verb lives in its own commands/*_cmd.py."""
from __future__ import annotations

import argparse
import sys

from .commands import (
    contract_cmd,
    curate_cmd,
    inspect_cmd,
    list_cmd,
    misc_cmd,
    plan_cmd,
    review_cmd,
    run_cmd,
    score_curated_cmd,
    inspect_clu_cmd,
    fk_sweep_cmd,
)
from .plan import plan_path


_EPILOG = """\
examples:
  # discovery
  ns3sb list adapters --available-only
  ns3sb list study-sets
  ns3sb list datasets --study-set PAIRED_CRCNS_HC1

  # build a plan, then run it
  ns3sb load paired_crcns_hc1:d533101 propper_sim_tiny
  ns3sb set adapters botm_ref,fiber_kit,mountainsort5,kilosort4
  ns3sb set scorers pairwise,per_cell
  ns3sb set output runs/2026-11-03
  ns3sb show
  ns3sb run

  # one-shot
  ns3sb run --corpus paired_kampff:<item> --adapters fiber_kit,botm_ref

  # inspect / document
  ns3sb inspect paired_crcns_hc1:d533101
  ns3sb contract fiber_kit --format markdown
  ns3sb doctor

  # review past runs
  ns3sb summary runs/baseline
  ns3sb compare runs/baseline runs/candidate --only-regressed
  ns3sb compare runs/baseline runs/candidate --metric accuracy,f1

  # measure Klusters curation effort between a sorter output and a curated one
  ns3sb curate-diff fiber_kit.clu.standard.1 fiber_kit.clu.standard.1.curated
  ns3sb curate-diff fiber_kit.clu.standard.1 fiber_kit.clu.standard.1.curated --format csv

  # your own data (CA1 lab session + curated GT)
  ns3sb load lab:/data/g7_session.yaml
  ns3sb load lab:/data/g6_0316.yaml:6:stderiv_C5_D34:exp_final
"""


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ns3sb",
        description="ns3-sort-bench — cross-sorter benchmark harness.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="store_true",
                    help="print version and exit")
    ap.add_argument("--plan-path", action="store_true",
                    help="print the active plan file path and exit")
    sub = ap.add_subparsers(dest="cmd", required=False, metavar="<command>")

    # Discovery
    list_cmd.register(sub)
    # Plan management
    plan_cmd.register(sub)
    # Inspection
    inspect_cmd.register(sub)
    contract_cmd.register(sub)
    # Running
    run_cmd.register(sub)
    fk_sweep_cmd.register(sub)
    # Review of previous runs
    review_cmd.register(sub)
    # Curation-effort diff (Klusters .clu + .cur comparison)
    curate_cmd.register(sub)
    score_curated_cmd.register(sub)
    inspect_clu_cmd.register(sub)
    misc_cmd.register_lattice(sub)
    misc_cmd.register_render(sub)
    misc_cmd.register_doctor(sub)

    return ap


def main(argv=None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)

    if args.version:
        try:
            from .. import __version__
        except ImportError:
            __version__ = "unknown"
        print(f"ns3-sort-bench {__version__}")
        return 0

    if args.plan_path:
        print(plan_path())
        return 0

    if not getattr(args, "cmd", None):
        ap.print_help()
        return 0

    try:
        return args.fn(args) or 0
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
