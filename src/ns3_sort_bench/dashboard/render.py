"""Markdown dashboard renderer.

Reads `scores.csv` (long format: corpus_item, candidate_adapter, metric,
value, ...) and `sort_summaries.json` (per-adapter runtime + unit counts)
from a run directory and writes `sort_benchmark.md` with:

- one pivot table per scorer (corpus × adapter × metric)
- a resources section (wall time, peak RSS) across sorters
- a CI gate: lines where the current value regresses more than 1σ of the
  seed spread from the baseline JSON get flagged

Baseline JSON (`ci/baseline.json`) carries the last known-good value +
seed spread per (corpus, adapter, metric).  See `ci/README.md`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


def render_markdown(run_dir: str, out_path: Optional[str] = None,
                     baseline_path: Optional[str] = None) -> str:
    """Build the dashboard markdown.  Writes to `out_path` if given;
    returns the markdown string either way."""
    import pandas as pd

    run_dir = Path(run_dir)
    scores_csv = run_dir / "scores.csv"
    summaries_json = run_dir / "sort_summaries.json"

    if not scores_csv.exists():
        return f"# sort_benchmark dashboard\n\nNo scores.csv in {run_dir}.\n"

    df = pd.read_csv(scores_csv)
    with open(summaries_json) as f:
        summaries = json.load(f)
    baseline = {}
    if baseline_path and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)

    lines = ["# sort_benchmark dashboard", "",
             f"*Rendered from `{scores_csv}` with {len(df)} score rows "
             f"and {len(summaries)} sort summaries.*", ""]

    # ─── per-scorer pivot tables ────────────────────────────────────────
    for scorer, sub in df.groupby("scorer"):
        lines.append(f"## {scorer}")
        lines.append("")
        pivot = sub.pivot_table(index=["corpus_item", "metric"],
                                 columns="candidate_adapter",
                                 values="value",
                                 aggfunc="mean")
        lines.append(pivot.to_markdown(floatfmt=".3f"))
        lines.append("")

    # ─── resources ──────────────────────────────────────────────────────
    lines.append("## Resources (wall time, peak RSS)")
    lines.append("")
    res_rows = []
    for s in summaries:
        if "error" in s: continue
        res_rows.append({
            "corpus_item": s["corpus_item"],
            "adapter": s["adapter"],
            "n_units": s["n_units"],
            "n_spikes": s["n_spikes"],
            "wall_time_s": s["wall_time_s"],
            "peak_rss_mb": s["peak_rss_mb"],
        })
    if res_rows:
        res = pd.DataFrame(res_rows)
        lines.append(res.to_markdown(index=False, floatfmt=".1f"))
    else:
        lines.append("_no successful sorts_")
    lines.append("")

    # ─── failed sorts ────────────────────────────────────────────────────
    failed = [s for s in summaries if "error" in s]
    if failed:
        lines.append("## Failed sorts")
        lines.append("")
        for s in failed:
            lines.append(f"- **{s['adapter']}** on `{s['corpus_item']}`: {s['error']}")
        lines.append("")

    # ─── CI gate ─────────────────────────────────────────────────────────
    if baseline:
        lines.append("## CI gate")
        lines.append("")
        regressions = _check_regressions(df, baseline)
        if not regressions:
            lines.append("No regressions beyond ±1σ of baseline seed spread.")
        else:
            lines.append("| corpus | adapter | metric | baseline | current | Δ (σ) |")
            lines.append("|---|---|---|---|---|---|")
            for r in regressions:
                lines.append(f"| {r['corpus_item']} | {r['adapter']} | "
                             f"{r['metric']} | {r['baseline']:.3f} | "
                             f"{r['current']:.3f} | {r['delta_sigma']:+.2f} |")

    md = "\n".join(lines) + "\n"
    if out_path:
        with open(out_path, "w") as f:
            f.write(md)
    return md


def _check_regressions(df, baseline):
    """Flag rows where the current value is >1σ below baseline (for
    higher-is-better metrics) or >1σ above (for lower-is-better like RSS).
    Default direction: higher-is-better; override with
    `baseline[key]["direction"] = "min"`."""
    out = []
    for _, r in df.iterrows():
        key = f"{r['corpus_item']}::{r['candidate_adapter']}::{r['metric']}"
        base = baseline.get(key)
        if not base: continue
        bv = base["value"]; bs = max(base.get("sigma", 0.0), 1e-6)
        direction = base.get("direction", "max")
        delta = r["value"] - bv
        delta_sigma = delta / bs
        if direction == "max" and delta_sigma < -1.0:
            out.append(dict(corpus_item=r["corpus_item"],
                            adapter=r["candidate_adapter"],
                            metric=r["metric"],
                            baseline=bv, current=r["value"],
                            delta_sigma=delta_sigma))
        elif direction == "min" and delta_sigma > 1.0:
            out.append(dict(corpus_item=r["corpus_item"],
                            adapter=r["candidate_adapter"],
                            metric=r["metric"],
                            baseline=bv, current=r["value"],
                            delta_sigma=delta_sigma))
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m sort_benchmark.dashboard.render <run_dir> [out.md] [baseline.json]")
        sys.exit(2)
    run_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    base = sys.argv[3] if len(sys.argv) > 3 else None
    md = render_markdown(run_dir, out_path=out, baseline_path=base)
    if not out:
        print(md)
