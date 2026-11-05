"""`ns3sb summary` and `ns3sb compare` — read scored runs and print a view.

Both commands operate on the `scores.csv` written by `BenchmarkRun.go`.
Its rows are `(scorer, corpus_item, reference_adapter, candidate_adapter,
metric, value)`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from .._util import eprint, split_csv, table


# ─── org-style renderer ─────────────────────────────────────────────────────
# One table per corpus, pipe-separated with an org-mode separator row:
#
#     | adapter            |    iou | recall |   prec | n_cand | merge_n |
#     |--------------------+--------+--------+--------+--------+---------|
#     | fiber_kit_session  | 0.0490 | 0.1463 | 0.0687 |     28 |       1 |
#
# Numeric columns right-aligned, strings left.  Readable in any terminal
# and pastes verbatim into an Org-mode document or a Markdown-with-pipes
# renderer; a reviewer can wrap it in `#+NAME: results` for citation.

# Preferred column-name aliases — short enough to keep columns narrow
# without losing what they measure.  Not an exhaustive renaming; metrics
# not listed here keep their scores.csv name.
_METRIC_ALIASES = {
    "best_unit_accuracy_mean":    "iou",
    "best_unit_recall_mean":      "recall",
    "best_unit_precision_mean":   "prec",
    "cells_ge_50":                ">=0.5",
    "cells_ge_80":                ">=0.8",
    "oracle_merge_accuracy_mean":    "o_iou",
    "oracle_merge_recall_mean":      "o_recall",
    "oracle_merge_precision_mean":   "o_prec",
    "oracle_merge_subset_size_mean": "merge_n",
    "n_matched_spikes":           "n_match",
    "n_cand_units":               "n_cand",
    "n_ref_units":                "n_ref",
    "n_ref_cells":                "n_cells",
    "n_ref_spikes":               "n_sp",
}

# Default column order when --metric is not given — compact, top-to-bottom
# "what did the sort do, how well did it hit, how many units, could a
# curator fix it, what did the matching look like".
_DEFAULT_COLS = (
    "best_unit_accuracy_mean",
    "best_unit_recall_mean",
    "best_unit_precision_mean",
    "cells_ge_80",
    "cells_ge_50",
    "n_cand_units",
    "oracle_merge_accuracy_mean",
    "oracle_merge_subset_size_mean",
    "ari",
    "purity",
    "completeness",
    "n_matched_spikes",
)


def _short(name: str) -> str:
    """Terminal-friendly name for a metric column."""
    return _METRIC_ALIASES.get(name, name)


def _is_count_metric(name: str) -> bool:
    """Count-like metrics render as integers (no decimal zeros)."""
    return (name.startswith("n_") or
            name.endswith("_units") or name.endswith("_cells") or
            name.endswith("_spikes") or
            name.startswith("cells_ge_") or
            name == "oracle_merge_subset_size_mean")


def _fmt_cell(v, decimals: int, as_int: bool) -> str:
    """One table cell — "—" for missing, int for count columns, else float."""
    try:
        import math
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
    except Exception:
        pass
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if as_int:
        return f"{int(round(f))}"
    return f"{f:.{decimals}f}"


def _org_table(headers: Sequence[str], rows: Sequence[Sequence[str]],
                 numeric_mask: Sequence[bool]) -> str:
    """Emacs org-mode / GitHub-pipe table.

    `numeric_mask[i]` true → column i right-aligned; false → left.  The
    separator row uses `+` at column joins (org convention) so Emacs
    recognises it as a table-with-rules; also renders as a valid Markdown
    table (which uses `|` everywhere) in surfaces that don't understand
    org.
    """
    cols = list(headers)
    cells = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in cols]
    for row in cells:
        for i, c in enumerate(row):
            if len(c) > widths[i]:
                widths[i] = len(c)

    def _fmt_row(cells_in_row):
        parts = []
        for i, c in enumerate(cells_in_row):
            w = widths[i]
            parts.append(f" {c:>{w}} " if numeric_mask[i] else f" {c:<{w}} ")
        return "|" + "|".join(parts) + "|"

    sep = "|" + "+".join("-" * (w + 2) for w in widths) + "|"
    out = [_fmt_row(cols), sep]
    out += [_fmt_row(r) for r in cells]
    return "\n".join(out)


def _load_scores(run_dir: Path):
    """Return the scores DataFrame or raise FileNotFoundError.

    Reads with `keep_default_na=False` and a narrow `na_values` set so a
    row whose `corpus_item` happens to be the literal string "None" (or an
    adapter name of "NA", etc.) is NOT silently coerced to NaN — pandas's
    default `na_values` list is surprisingly wide and takes out legitimate
    identifier strings.  Empty cells are still NaN.
    """
    import pandas as pd
    p = Path(run_dir) / "scores.csv"
    if not p.exists():
        raise FileNotFoundError(f"no scores.csv in {run_dir}")
    df = pd.read_csv(p, keep_default_na=False, na_values=[""])
    return df


# ─── summary ────────────────────────────────────────────────────────────────
def cmd_summary(args) -> int:
    """Print a corpus × adapter × metric pivot for one run directory."""
    try:
        df = _load_scores(Path(args.run_dir))
    except FileNotFoundError as e:
        eprint(str(e))
        return 1
    except Exception as e:
        eprint(f"{type(e).__name__}: {e}")
        return 2

    # Optional narrowing
    metrics = split_csv(args.metric)
    adapters = split_csv(args.adapter)
    corpora = split_csv(args.corpus)
    if metrics:   df = df[df["metric"].isin(metrics)]
    if adapters:  df = df[df["candidate_adapter"].isin(adapters)]
    if corpora:   df = df[df["corpus_item"].isin(corpora)]
    if df.empty:
        eprint("(no rows after filtering)")
        return 1

    piv = df.pivot_table(
        values="value",
        index=["corpus_item", "candidate_adapter"],
        columns="metric",
        aggfunc="first",
    )
    # Column order: when the user named metrics, keep their order; else
    # take the compact _DEFAULT_COLS set (dropping any that don't exist in
    # the pivot) and append the rest at the end so nothing is silently lost.
    if metrics:
        cols = [c for c in metrics if c in piv.columns]
    else:
        cols = [c for c in _DEFAULT_COLS if c in piv.columns] + \
               [c for c in piv.columns if c not in _DEFAULT_COLS]
    piv = piv[cols]

    fmt_decimals = int(args.decimals)
    fmt = f"%.{fmt_decimals}f"

    if args.format == "csv":
        print(piv.to_csv(float_format=fmt))
        return 0

    if args.format == "org":
        _print_org(piv, decimals=fmt_decimals)
    else:
        # Pretty text (pandas default) — fixed-decimal so --decimals controls.
        with _pandas_display():
            print(piv.to_string(float_format=lambda v: fmt % v if v == v else "—"))

    print(f"\n{len(df)} rows across {df['corpus_item'].nunique()} corpus "
          f"item(s), {df['candidate_adapter'].nunique()} adapter(s), "
          f"{df['metric'].nunique()} metric(s)")
    return 0


def _print_org(piv, *, decimals: int) -> None:
    """Emit one org-style table per corpus_item, newline-separated.

    `piv` is a pandas DataFrame indexed by (corpus_item, candidate_adapter)
    with one column per metric.  The rendered column names come from
    `_METRIC_ALIASES`; count-like columns render as integers.
    """
    import math

    metrics = list(piv.columns)
    numeric_mask = [True] * (1 + len(metrics))                 # all cols right-aligned
    numeric_mask[0] = False                                     # adapter name stays left
    headers = ["adapter"] + [_short(m) for m in metrics]

    corpora = piv.index.get_level_values("corpus_item").unique()
    for ci, corpus in enumerate(corpora):
        if ci:
            print()
        print(f"# {corpus}")
        sub = piv.xs(corpus, level="corpus_item")
        rows = []
        for adapter, row in sub.iterrows():
            cells = [str(adapter)]
            for m in metrics:
                v = row.get(m)
                cells.append(_fmt_cell(v, decimals, _is_count_metric(m)))
            rows.append(cells)
        print(_org_table(headers, rows, numeric_mask))


# ─── compare ────────────────────────────────────────────────────────────────
def cmd_compare(args) -> int:
    """Diff two run directories metric-by-metric.

    Join on (corpus_item, candidate_adapter, scorer, metric); report
    delta = value_b - value_a.  Rows present in only one run are kept with
    the missing side shown as NaN.
    """
    try:
        a = _load_scores(Path(args.run_a))
        b = _load_scores(Path(args.run_b))
    except FileNotFoundError as e:
        eprint(str(e))
        return 1
    import pandas as pd

    key = ["corpus_item", "candidate_adapter", "scorer", "metric"]
    merged = pd.merge(
        a[key + ["value"]].rename(columns={"value": "A"}),
        b[key + ["value"]].rename(columns={"value": "B"}),
        on=key, how="outer",
    )
    merged["delta"] = merged["B"] - merged["A"]

    # Optional narrowing
    metrics = split_csv(args.metric)
    adapters = split_csv(args.adapter)
    corpora = split_csv(args.corpus)
    if metrics:  merged = merged[merged["metric"].isin(metrics)]
    if adapters: merged = merged[merged["candidate_adapter"].isin(adapters)]
    if corpora:  merged = merged[merged["corpus_item"].isin(corpora)]
    if merged.empty:
        eprint("(no rows after filtering)")
        return 1

    # Sort by |delta| descending so the biggest movers are on top
    merged = merged.assign(abs_delta=merged["delta"].abs()
                                     .fillna(float("inf"))) \
                   .sort_values("abs_delta", ascending=False) \
                   .drop(columns="abs_delta")

    if args.only_regressed:
        merged = merged[merged["delta"] < 0]
    elif args.only_improved:
        merged = merged[merged["delta"] > 0]

    print(f"A = {args.run_a}")
    print(f"B = {args.run_b}")
    print()
    rows = []
    for _, r in merged.iterrows():
        rows.append((
            r["corpus_item"], r["candidate_adapter"], r["scorer"], r["metric"],
            _fmt_val(r["A"], args.decimals),
            _fmt_val(r["B"], args.decimals),
            _fmt_delta(r["delta"], args.decimals),
        ))
    print(table(rows, headers=("corpus", "adapter", "scorer", "metric",
                               "A", "B", "Δ (B-A)")))
    print(f"\n{len(rows)} metric row(s)")
    return 0


# ─── small formatting helpers ────────────────────────────────────────────────
def _fmt_val(v, decimals: int) -> str:
    try:
        import math
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
    except Exception:
        pass
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_delta(d, decimals: int) -> str:
    try:
        import math
        if d is None or (isinstance(d, float) and math.isnan(d)):
            return "—"
        if d > 0:
            return f"+{d:.{decimals}f}"
        return f"{d:.{decimals}f}"
    except Exception:
        return str(d)


class _pandas_display:
    """Context manager: wide text pandas output without truncation."""
    _KEYS = ("display.max_rows", "display.max_columns",
             "display.width", "display.max_colwidth")

    def __enter__(self):
        import pandas as pd
        self.old = {k: pd.get_option(k) for k in self._KEYS}
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 60)
        return self

    def __exit__(self, *exc):
        import pandas as pd
        for k, v in self.old.items():
            pd.set_option(k, v)


# ─── argparse wiring ─────────────────────────────────────────────────────────
def register(subparsers) -> None:
    s = subparsers.add_parser("summary",
                              help="print a corpus × adapter × metric pivot for one run")
    s.add_argument("run_dir", help="directory holding scores.csv")
    s.add_argument("--metric", default="", help="comma-separated metric filter")
    s.add_argument("--adapter", default="", help="comma-separated adapter filter")
    s.add_argument("--corpus", default="", help="comma-separated corpus filter")
    s.add_argument("--decimals", type=int, default=4)
    s.add_argument("--format", choices=("org", "text", "csv"), default="org",
                    help="org (default): one org-mode table per corpus item, "
                         "short metric names, right-aligned numerics.  text: "
                         "legacy pandas pivot (wide, long column names).  csv: "
                         "raw pivot for piping.")
    s.set_defaults(fn=cmd_summary)

    c = subparsers.add_parser("compare",
                              help="diff two run directories metric-by-metric (B - A)")
    c.add_argument("run_a", help="baseline run directory (holds scores.csv)")
    c.add_argument("run_b", help="candidate run directory (holds scores.csv)")
    c.add_argument("--metric", default="", help="comma-separated metric filter")
    c.add_argument("--adapter", default="", help="comma-separated adapter filter")
    c.add_argument("--corpus", default="", help="comma-separated corpus filter")
    c.add_argument("--decimals", type=int, default=4)
    c.add_argument("--only-regressed", action="store_true",
                   help="only rows where B is worse than A (delta < 0)")
    c.add_argument("--only-improved", action="store_true",
                   help="only rows where B is better than A (delta > 0)")
    c.set_defaults(fn=cmd_compare)
