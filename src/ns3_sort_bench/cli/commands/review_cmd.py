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

    Reads scores.csv with `keep_default_na=False` and a narrow
    `na_values` set so a row whose `corpus_item` happens to be the
    literal string "None" (or an adapter name of "NA", etc.) is NOT
    silently coerced to NaN — pandas's default `na_values` list is
    surprisingly wide and takes out legitimate identifier strings.
    Empty cells are still NaN.

    When ``curated_scores.csv`` also lives in the run directory (written
    by ``ns3sb score-curated``), its ``stage=curated`` rows are appended
    to the returned frame under candidate adapter
    ``<sorter>@curated`` — so they render in the "## <sorter> stages"
    sub-table ns3sb summary already emits (patch 0019), right alongside
    the pipeline-stage rows.  The ``stage=raw`` rows are dropped: they
    are a re-scoring of the same .clu already in scores.csv and would
    duplicate the primary.
    """
    import pandas as pd
    p = Path(run_dir) / "scores.csv"
    if not p.exists():
        raise FileNotFoundError(f"no scores.csv in {run_dir}")
    df = pd.read_csv(p, keep_default_na=False, na_values=[""])

    curated = Path(run_dir) / "curated_scores.csv"
    if curated.exists():
        cdf = pd.read_csv(curated, keep_default_na=False, na_values=[""])
        # curated_scores.csv shape:
        #     sorter, corpus_item, stage, scorer, metric, value
        # Normalise into scores.csv shape:
        #     reference_adapter, candidate_adapter, scorer, corpus_item,
        #     metric, value.
        needed = {"sorter", "corpus_item", "stage",
                  "scorer", "metric", "value"}
        if needed.issubset(cdf.columns):
            cdf = cdf[cdf["stage"] == "curated"].copy()
            cdf["candidate_adapter"] = cdf["sorter"] + "@curated"
            cdf["reference_adapter"] = "ground_truth"
            # score-curated iterates the on-disk klusters_<sorter>/<item>/
            # directory tree, where <item> is the SLUGIFIED corpus_item
            # (`:` → `_`, etc.).  scores.csv holds the ORIGINAL
            # corpus_item.  Reverse-map here so the curated row shares a
            # corpus_item value with its primary — otherwise summary
            # renders two separate blocks ("primary not shown") for the
            # same item.
            slug_to_original = {_slugify_corpus_item(ci): ci
                                 for ci in df["corpus_item"].astype(str).unique()}
            cdf["corpus_item"] = cdf["corpus_item"].astype(str).map(
                lambda s: slug_to_original.get(s, s)
            )
            cdf = cdf[["reference_adapter", "candidate_adapter",
                        "scorer", "corpus_item", "metric", "value"]]
            # Concat; pandas tolerates the column superset (scores.csv
            # may carry extra columns the curated run didn't produce).
            df = pd.concat([df, cdf], ignore_index=True)
    return df


# Characters _slugify keeps (mirrors adapters.fiber_kit._SLUG_SAFE); every
# other character becomes `_` and consecutive underscores collapse.  This
# is the SAME algorithm klusters_export uses when it builds its
# klusters_<sorter>/<slug>/ directory names, so the summary-side reverse
# map stays in sync without a cross-module import.
_SLUG_KEEP = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  "0123456789._+-")


def _slugify_corpus_item(name: str) -> str:
    out = "".join(c if c in _SLUG_KEEP else "_" for c in name)
    parts = [p for p in out.split("_") if p]
    slug = "_".join(parts) or "sort"
    return slug[:200]


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
        sort=False,       # keep scores.csv row order so pipeline stages
                          # render in plan order (fiber_session →
                          # fs_clean → exp_xcm), not alphabetical.
    )
    # Pivot with sort=False keeps candidate_adapter order, but the
    # corpus_item level may still interleave poorly if a corpus has
    # non-contiguous rows.  Re-sort the first level alone, by first-row
    # order.  (A categorical index with the discovered order keeps that
    # order stable across subsequent operations.)
    corpus_order = list(dict.fromkeys(df["corpus_item"].tolist()))
    piv = piv.reindex(corpus_order, level="corpus_item")
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


def _split_stage_adapters(adapter_names: Sequence[str]
                          ) -> "tuple[list[str], dict[str, list[str]]]":
    """Partition a list of candidate_adapter names into primaries and
    per-primary stage rows.

    A stage row's name is ``"<primary>@<stage_tag>"`` (today: emitted by
    the fiber_kit adapter — fiber_kit@fiber_session, fiber_kit@fs_clean,
    fiber_kit@exp_xcm).  Primaries keep their input order; per-primary
    stage lists also keep theirs, so a caller that sorted the input by
    plan discovery order gets plan order out.

    Returns (primaries, {primary: [stage_full_names, ...]}).  The second
    map is keyed by primary name and lists the FULL stage adapter names
    (not bare stage tags) so a pivot lookup still hits.
    """
    primaries: list[str] = []
    stages_by_primary: dict[str, list[str]] = {}
    for name in adapter_names:
        s = str(name)
        if "@" in s:
            primary, _ = s.split("@", 1)
            stages_by_primary.setdefault(primary, []).append(s)
        else:
            primaries.append(s)
    return primaries, stages_by_primary


def _print_org(piv, *, decimals: int) -> None:
    """Emit one org-style table per corpus_item, newline-separated.

    `piv` is a pandas DataFrame indexed by (corpus_item, candidate_adapter)
    with one column per metric.  The rendered column names come from
    `_METRIC_ALIASES`; count-like columns render as integers.

    Pipeline stages (adapter names of the form ``<primary>@<tag>``) are
    pulled out of the main table and printed as separate sub-tables
    beneath it so the primary sorters' scoreboard stays uncluttered.
    """
    import math

    metrics = list(piv.columns)
    numeric_mask = [True] * (1 + len(metrics))                 # all cols right-aligned
    numeric_mask[0] = False                                     # adapter name stays left
    headers_main   = ["adapter"] + [_short(m) for m in metrics]
    headers_stages = ["stage"]   + [_short(m) for m in metrics]

    corpora = piv.index.get_level_values("corpus_item").unique()
    for ci, corpus in enumerate(corpora):
        if ci:
            print()
        print(f"# {corpus}")
        sub = piv.xs(corpus, level="corpus_item")
        # Partition in the pivot's own adapter order (which pandas took
        # from the scores.csv rows, i.e. BenchmarkRun.go's write order —
        # plan order for the stage rows).
        primaries, stages_by_primary = _split_stage_adapters(sub.index.tolist())

        # Main table — primaries only.
        rows = []
        for adapter in primaries:
            row = sub.loc[adapter]
            cells = [adapter]
            for m in metrics:
                cells.append(_fmt_cell(row.get(m), decimals, _is_count_metric(m)))
            rows.append(cells)
        print(_org_table(headers_main, rows, numeric_mask))

        # Per-primary stage sub-tables.  Only primaries that actually have
        # stage rows get one; if neither the primary nor its stages got
        # filtered out, we include a trailing `<primary>@exp_final`-equivalent
        # row — the primary row itself — so the reader sees the whole chain
        # (fiber_session → fs_clean → exp_xcm → <primary>) in one block.
        for primary in primaries:
            stage_names = stages_by_primary.get(primary)
            if not stage_names:
                continue
            print()
            print(f"## {primary} stages")
            # Split pipeline-intermediate stages (fiber_session, fs_clean,
            # exp_xcm, ...) from any post-hoc stages (today: `@curated`
            # from `ns3sb score-curated`).  Pipeline stages render in
            # arrival order, then `final` (the primary itself, which IS
            # the end-of-chain sort), then any curated-style rows AFTER
            # final — those are human-added on top of the sort.
            pipeline_stages = [s for s in stage_names
                                if not s.endswith("@curated")]
            curated_stages = [s for s in stage_names
                                if s.endswith("@curated")]
            stage_rows = []
            for full_name in pipeline_stages:
                row = sub.loc[full_name]
                _, stage_tag = full_name.split("@", 1)
                cells = [stage_tag]
                for m in metrics:
                    cells.append(_fmt_cell(row.get(m), decimals, _is_count_metric(m)))
                stage_rows.append(cells)
            # End-of-chain row: the primary itself, labelled with its
            # mode-final tag (today: fiber_kit → exp_final).  The label
            # "final" is clear enough given the section heading.
            row = sub.loc[primary]
            cells = ["final"]
            for m in metrics:
                cells.append(_fmt_cell(row.get(m), decimals, _is_count_metric(m)))
            stage_rows.append(cells)
            # Curated row(s) AFTER final — curation happens on top of the
            # final sort, so rendering them earlier would read as if the
            # curator ran mid-pipeline.
            for full_name in curated_stages:
                row = sub.loc[full_name]
                _, stage_tag = full_name.split("@", 1)
                cells = [stage_tag]
                for m in metrics:
                    cells.append(_fmt_cell(row.get(m), decimals, _is_count_metric(m)))
                stage_rows.append(cells)
            print(_org_table(headers_stages, stage_rows, numeric_mask))

        # Orphan stage groups: stage rows whose primary isn't in this
        # slice (e.g. --adapter fiber_kit@exp_xcm alone).  Still render
        # them so the filter doesn't silently drop rows.
        for primary, stage_names in stages_by_primary.items():
            if primary in primaries or not stage_names:
                continue
            print()
            print(f"## {primary} stages (primary not shown)")
            stage_rows = []
            for full_name in stage_names:
                row = sub.loc[full_name]
                _, stage_tag = full_name.split("@", 1)
                cells = [stage_tag]
                for m in metrics:
                    cells.append(_fmt_cell(row.get(m), decimals, _is_count_metric(m)))
                stage_rows.append(cells)
            print(_org_table(headers_stages, stage_rows, numeric_mask))


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
