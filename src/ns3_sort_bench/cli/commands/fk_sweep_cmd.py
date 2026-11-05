"""``ns3sb fk-sweep`` — run fiber_kit over a set of FK_* values and tabulate.

Each setting is one ``ns3sb run --adapters fiber_kit --scorers per_cell``
in its own folder under ``--out``, with ``$FK_CONFIG`` pointing at a tuning
file written for it: the base tuning (``--base-config``, default
``$FK_CONFIG``) plus that setting's keys.  Runs go in fresh temporary
folders (no ``--keep-fk-artifacts``), so no run can read another's files.

Two ways to combine ``--set`` values:

  one at a time (default)   a baseline run, then each value of each key
                            alone, everything else at the base tuning;
  ``--grid``                every combination of the values.

With no ``--set``, a one-at-a-time sweep of the fiber-session knobs most
likely to move its cluster count (``DEFAULT_SESSION_SWEEP``).

A finished setting whose tuning file is unchanged is not run again, so an
interrupted sweep resumes; ``--rerun`` forces every run.

Invocation::

    ns3sb fk-sweep --corpus paired_crcns_hc1:d11221_d11221.001 --out DIR
                   [--set FK_KEY=v1,v2 ...] [--grid] [--base-config Y]
                   [--rerun] [--dry-run]

Prints, per setting, cluster count / IoU / recall / precision at
fiber_session and at the final stage, and writes ``DIR/sweep.csv``.
"""
from __future__ import annotations

import csv
import itertools
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .._util import eprint, table


# Values around the shipped configs/fiber_kit.yaml baseline; the comment on
# each is the baseline value.
DEFAULT_SESSION_SWEEP: Dict[str, List[Any]] = {
    "FK_SESSION_RESPLIT_RESIDUAL_THR": [0.08, 0.25],   # 0.15
    "FK_SESSION_RESPLIT_PASSES":       [0],            # 3
    "FK_SESSION_RKK_PENALTY_MIX":      [0.5, 1.0],     # 0.0 (AIC)
    "FK_SESSION_DIPSPLIT":             [0],            # 1
    "FK_SESSION_KEEP_SMALL":           [0],            # 1
    "FK_SESSION_INCLUSION_K":          [2.5],          # 3.14
    "FK_SESSION_DECONV":               [0],            # 1
}

_STAGES = (("fiber_session", "fiber_kit@fiber_session"),
           ("final", "fiber_kit"))
_METRICS = (("n", "n_cand_units"),
            ("iou", "best_unit_accuracy_mean"),
            ("recall", "best_unit_recall_mean"),
            ("prec", "best_unit_precision_mean"))


def parse_set(spec: str) -> Tuple[str, List[Any]]:
    """``KEY=v1,v2`` → ``(KEY, [v1, v2])``, each value read as YAML
    (``0.15`` → float, ``off`` stays a string, ``""`` → empty)."""
    import yaml
    if "=" not in spec:
        raise ValueError(f"--set wants KEY=v1,v2,...; got {spec!r}")
    key, vals = spec.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"--set has an empty key: {spec!r}")
    out: List[Any] = []
    for v in vals.split(","):
        v = v.strip()
        parsed = yaml.safe_load(v) if v else ""
        # YAML 1.1 reads off/on/yes/no as bools; fiber-kit wants strings.
        out.append(v if isinstance(parsed, bool) else parsed)
    return key, out


def settings(sweep: Dict[str, List[Any]], grid: bool
             ) -> List[Tuple[str, Dict[str, Any]]]:
    """``[(label, {key: value})]`` — baseline first, then the sweep."""
    out: List[Tuple[str, Dict[str, Any]]] = [("baseline", {})]
    if grid:
        keys = list(sweep)
        for combo in itertools.product(*(sweep[k] for k in keys)):
            over = dict(zip(keys, combo))
            out.append((_label(over), over))
    else:
        for k, vals in sweep.items():
            for v in vals:
                out.append((_label({k: v}), {k: v}))
    return out


def _label(over: Dict[str, Any]) -> str:
    parts = []
    for k, v in over.items():
        short = k[3:] if k.startswith("FK_") else k
        parts.append(f"{short}={'empty' if v == '' else v}")
    return "+".join(parts)


def tuning_text(base: Dict[str, Any], over: Dict[str, Any]) -> str:
    import yaml
    merged = {**base, **over}
    return yaml.safe_dump(merged, sort_keys=True, default_flow_style=None)


def read_stage_metrics(scores_csv: str) -> Dict[str, Dict[str, float]]:
    """``{stage: {metric: value}}`` from one run's per_cell rows."""
    want = {adapter: stage for stage, adapter in _STAGES}
    col = {src: dst for dst, src in _METRICS}
    out: Dict[str, Dict[str, float]] = {}
    with open(scores_csv) as f:
        for row in csv.DictReader(f):
            if row.get("scorer") != "per_cell":
                continue
            stage = want.get(row.get("candidate_adapter", ""))
            name = col.get(row.get("metric", ""))
            if stage and name:
                out.setdefault(stage, {})[name] = float(row["value"])
    return out


def _run_one(corpus: str, run_dir: str, tuning_path: str) -> int:
    env = {**os.environ, "FK_CONFIG": tuning_path}
    cmd = [sys.executable, "-c",
           "import sys; from ns3_sort_bench.cli import main; "
           "sys.exit(main(sys.argv[1:]))", "run",
           "--corpus", corpus, "--adapters", "fiber_kit",
           "--scorers", "per_cell", "--out", run_dir,
           "--no-export-klusters", "--no-keep-fk-artifacts", "--quiet"]
    with open(os.path.join(run_dir, "run.log"), "w") as log:
        return subprocess.run(cmd, env=env, stdout=log,
                              stderr=subprocess.STDOUT).returncode


def run(args) -> int:
    import yaml
    sweep: Dict[str, List[Any]] = {}
    for spec in args.set or []:
        try:
            k, vals = parse_set(spec)
        except ValueError as e:
            eprint(f"fk-sweep: {e}")
            return 2
        sweep.setdefault(k, []).extend(vals)
    if not sweep:
        sweep = {k: list(v) for k, v in DEFAULT_SESSION_SWEEP.items()}

    base_path = args.base_config or os.environ.get("FK_CONFIG") or ""
    base: Dict[str, Any] = {}
    if base_path:
        if not os.path.isfile(base_path):
            eprint(f"fk-sweep: base config {base_path!r} not found")
            return 2
        with open(base_path) as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            eprint(f"fk-sweep: {base_path!r} is not a mapping")
            return 2
        base = loaded

    plan = settings(sweep, args.grid)
    if args.dry_run:
        print(f"{len(plan)} runs on {args.corpus}"
              + (f", base tuning {base_path}" if base_path else ""))
        for label, over in plan:
            print(f"  {label}: {over or '(base tuning)'}")
        return 0

    os.makedirs(args.out, exist_ok=True)
    results: List[Tuple[str, Dict[str, Dict[str, float]], str]] = []
    for i, (label, over) in enumerate(plan, 1):
        run_dir = os.path.join(args.out, label)
        os.makedirs(run_dir, exist_ok=True)
        tuning_path = os.path.join(run_dir, "fk-tuning.yaml")
        text = tuning_text(base, over)
        scores = os.path.join(run_dir, "scores.csv")
        done = (os.path.isfile(scores) and os.path.isfile(tuning_path)
                and open(tuning_path).read() == text)
        if done and not args.rerun:
            eprint(f"[{i}/{len(plan)}] {label}: already run, reusing")
        else:
            with open(tuning_path, "w") as f:
                f.write(text)
            if os.path.isfile(scores):
                os.remove(scores)
            eprint(f"[{i}/{len(plan)}] {label}: running")
            rc = _run_one(args.corpus, run_dir, tuning_path)
            if rc != 0 or not os.path.isfile(scores):
                eprint(f"  failed (exit {rc}); see {run_dir}/run.log")
                results.append((label, {}, "failed"))
                continue
        results.append((label, read_stage_metrics(scores), ""))

    headers = ["setting"]
    for stage, _ in _STAGES:
        headers += [f"{stage} {m}" for m, _ in _METRICS]
    rows = []
    for label, m, note in results:
        row: List[Any] = [label]
        for stage, _ in _STAGES:
            for name, _ in _METRICS:
                v = m.get(stage, {}).get(name)
                row.append("" if v is None else
                           (f"{int(v)}" if name == "n" else f"{v:.3f}"))
        if note:
            row[1] = note
        rows.append(row)
    print(table(rows, headers=headers))

    out_csv = os.path.join(args.out, "sweep.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    eprint(f"wrote {out_csv}")
    return 0 if all(not n for _, _, n in results) else 1


def register(sub) -> None:
    ap = sub.add_parser(
        "fk-sweep",
        help="run fiber_kit over a set of FK_* values and tabulate "
             "fiber_session / final per_cell scores",
    )
    ap.add_argument("--corpus", required=True,
                    help="one corpus item, as for `ns3sb run --corpus`")
    ap.add_argument("--out", required=True,
                    help="sweep folder; one sub-folder per setting")
    ap.add_argument("--set", action="append", metavar="FK_KEY=v1,v2",
                    help="values to try for one key (repeatable); default: "
                         "a one-at-a-time sweep of fiber-session knobs")
    ap.add_argument("--grid", action="store_true",
                    help="run every combination instead of one key at a time")
    ap.add_argument("--base-config", default="",
                    help="tuning yaml every setting starts from "
                         "(default: $FK_CONFIG; none = shipped baseline)")
    ap.add_argument("--rerun", action="store_true",
                    help="run every setting again, even finished ones")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the runs and exit")
    ap.set_defaults(fn=run)
