"""`ns3sb summary` and `ns3sb compare` — read scored runs and print a view.

Both commands operate on the `scores.csv` written by `BenchmarkRun.go`.
Its rows are `(scorer, corpus_item, reference_adapter, candidate_adapter,
metric, value)`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .._util import eprint, split_csv, table


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
    # Choose a canonical column order: F1-style metrics first if present
    _preferred = ("accuracy", "precision", "recall", "f1",
                  "ari", "purity", "completeness",
                  "n_matched_spikes", "n_ref_units", "n_cand_units")
    cols = [c for c in _preferred if c in piv.columns] + \
           [c for c in piv.columns if c not in _preferred]
    piv = piv[cols]

    fmt = f"%.{args.decimals}f"
    if args.format == "csv":
        print(piv.to_csv(float_format=fmt))
        return 0
    # Pretty text (default) — fixed-decimal so `--decimals` controls output
    with _pandas_display():
        print(piv.to_string(float_format=lambda v: fmt % v if v == v else "—"))
    print(f"\n{len(df)} rows across {df['corpus_item'].nunique()} corpus "
          f"item(s), {df['candidate_adapter'].nunique()} adapter(s), "
          f"{df['metric'].nunique()} metric(s)")
    return 0


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
    s.add_argument("--format", choices=("text", "csv"), default="text")
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
