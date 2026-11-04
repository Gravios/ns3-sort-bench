"""`ns3sb curate-diff` — measure Klusters curation effort from a
(before.clu, after.clu) pair.

Combines:

  * a spike-level diff over the two `.clu` files (requires the same `.res`)
  * a `.cur` log parse when the Klusters JSON-Lines log sits next to the
    curated `.clu` (`.cur.<method>.<group>[.<tag>]`)

Prints a two-column report (metric / value), or CSV with `--format csv`.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from .._util import eprint, table
from ...curation import curate_diff, CurationReport


def _format_text(report: CurationReport, decimals: int = 4) -> str:
    buf = io.StringIO()
    buf.write(f"before.clu : {report.before_clu}\n")
    buf.write(f"after.clu  : {report.after_clu}\n")
    buf.write(f".cur log   : {report.cur_log or '(none found)'}\n\n")
    rows = []
    for r in report.to_rows():
        v = r["value"]
        rows.append((r["source"], r["metric"],
                      f"{v:.{decimals}f}" if isinstance(v, float) and v % 1 else f"{v:g}"))
    buf.write(table(rows, headers=("source", "metric", "value")))
    buf.write("\n")
    return buf.getvalue()


def _format_csv(report: CurationReport) -> str:
    import pandas as pd
    df = pd.DataFrame(report.to_rows())
    df.insert(0, "before_clu", report.before_clu)
    df.insert(1, "after_clu", report.after_clu)
    return df.to_csv(index=False)


def cmd_curate_diff(args) -> int:
    before = args.before
    after = args.after
    for p, label in ((before, "before.clu"), (after, "after.clu")):
        if not Path(p).exists():
            eprint(f"no such file: {label}={p}")
            return 1
    try:
        report = curate_diff(
            before, after,
            res_path=args.res or None,
            cur_log_path=args.cur_log or None,
        )
    except FileNotFoundError as e:
        eprint(str(e))
        return 2
    except ValueError as e:
        eprint(str(e))
        return 3

    if args.format == "csv":
        print(_format_csv(report), end="")
    else:
        print(_format_text(report, decimals=args.decimals))
    return 0


def register(subparsers) -> None:
    ap = subparsers.add_parser(
        "curate-diff",
        help="compare pre/post Klusters curation: diff + .cur-log complexity",
    )
    ap.add_argument("before", help="pre-curation .clu.<method>.<group>[.<tag>]")
    ap.add_argument("after",  help="post-curation .clu.<method>.<group>[.<tag>]")
    ap.add_argument("--res", default="",
                    help="override the paired .res (default: derived from after.clu)")
    ap.add_argument("--cur-log", default="",
                    help="override the .cur log (default: derived from after.clu)")
    ap.add_argument("--decimals", type=int, default=4)
    ap.add_argument("--format", choices=("text", "csv"), default="text")
    ap.set_defaults(fn=cmd_curate_diff)
