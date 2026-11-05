"""``ns3sb score-curated`` — score human-curated Klusters sessions.

Walks an ``<output_dir>`` with the per-sorter layout written by
``export_klusters_session``:

    <output_dir>/klusters_<sorter>/<item>/sim_session.yaml
    <output_dir>/klusters_<sorter>/<item>/sim_session.clu.standard.1.<raw_tag>
    <output_dir>/klusters_<sorter>/<item>/sim_session.clu.standard.1.curated
    <output_dir>/klusters_<sorter>/<item>/sim_session.cur.standard.1.curated  (optional)

For every (sorter, item) pair that carries a curated ``.clu``:

    1. Load the raw and curated clusterings as SortedResults.
    2. Load the GT from ``klusters_ground_truth/<item>/`` (its raw-sort
       .clu is the reference).
    3. Score both raw and curated against the GT with every scorer the
       harness has (per_cell / oracle_merge / pairwise / ...).
    4. Measure curation effort via :func:`curation.curate_diff` — spike
       reassignment diff + ``.cur`` log complexity when the log is there.

Writes two artefacts under ``<output_dir>``:

    curated_scores.csv            long format (same shape as scores.csv),
                                   one row per (sorter, item, stage, metric)
                                   where stage ∈ {raw, curated}
    curation_effort.csv           one row per (sorter, item) with the
                                   curate-diff numbers — spike reassignments,
                                   n_merges / n_splits, complexity_score

The (sorter, item, stage) axis means a later pivot can read:

    * ``raw`` row    — the sorter's accuracy before any human touched it
    * ``curated`` row — the ceiling the curator REACHED (vs oracle_merge's
                        theoretical ceiling)
    * side-by-side across sorters — inter-sorter curation comparison
      (which sorter's output curated to higher parity, and with how much
      effort).
"""
from __future__ import annotations

import csv
import glob
import os
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from ..._base import SortedResult, scorer_registry
from ... import scorers as _scorers_pkg   # noqa: F401 — populates scorer_registry
from ...curation import curate_diff, _read_clu, _read_res


_METHOD_TOKEN   = "standard"   # mirrors klusters_export._NS3_METHOD
_GROUP          = 1
_RAW_TAG_FOR_SORTER = {
    # fiber-kit variants write the .clu whose tag is their mode's
    # end-of-chain tag — kept in sync with adapters/fiber_kit.py's
    # _MODE_FINAL_TAG.  Post-patch-0016, fiber_kit dispatches through
    # the fiber-pipeline driver and writes exp_final; fiber_kit_no_deconv
    # still runs the legacy per-stage chain and writes refine_linked;
    # fiber_kit_session stops at fiber_session.
    "fiber_kit":            "exp_final",
    "fiber_kit_no_deconv":  "refine_linked",
    "fiber_kit_session":    "fiber_session",
    # Per-stage dirs that patch 0018 writes (klusters_fiber_kit_<stage>/):
    # each carries the single stage's .clu, named by the stage tag itself.
    "fiber_kit_fiber_session":  "fiber_session",
    "fiber_kit_fs_clean":       "fs_clean",
    "fiber_kit_exp_xcm":        "exp_xcm",
    # SI-wrapped sorters: klusters_export writes the .clu under the adapter
    # slug (no fiber-style suffix); the raw tag MATCHES the slug.  We fall
    # back to a scan when the sorter isn't listed here.
}


def _clu_path(base: str, tag: str) -> str:
    return f"{base}.clu.{_METHOD_TOKEN}.{_GROUP}.{tag}"


def _untagged_clu_path(base: str) -> str:
    """Canonical ns3 .clu path (no stage tag), the one klusters_export
    actually writes today: ``<base>.clu.<method>.<group>``.  The curated
    .clu is tag-suffixed (``.curated``); the raw sort-export is not."""
    return f"{base}.clu.{_METHOD_TOKEN}.{_GROUP}"


def _res_path(base: str) -> str:
    return f"{base}.res.{_METHOD_TOKEN}.{_GROUP}"


def _find_raw_clu(sess_dir: str, sorter: str) -> Optional[str]:
    """Return the sorter's RAW .clu path, or None if nothing matches.

    klusters_export writes the canonical un-tagged .clu
    (``<base>.clu.<method>.<group>``) when it exports a sort to disk.
    Older designs / future-us may also emit a stage-tagged variant
    (``<base>.clu.<method>.<group>.<tag>``) — we accept both.  Order of
    preference:
      1. The un-tagged canonical path — what klusters_export writes.
      2. The sorter's documented stage tag from ``_RAW_TAG_FOR_SORTER``.
      3. A glob scan over every tag in the dir (skipping curated,
         autosave, and the curated-log .cur siblings' tails).

    Returns None only if all three come up empty.
    """
    base = os.path.join(sess_dir, "sim_session")
    # (1) Untagged canonical — the only thing klusters_export writes.
    untagged = _untagged_clu_path(base)
    if os.path.exists(untagged):
        return untagged
    # (2) Documented stage tag (useful once klusters_export learns to
    # tag the end-of-chain write with its mode's final tag — see
    # patch 0022's note).
    tag = _RAW_TAG_FOR_SORTER.get(sorter, sorter)
    direct = _clu_path(base, tag)
    if os.path.exists(direct):
        return direct
    # (3) Fallback scan — pick any .clu.<method>.<group>.* that is not
    # the curated one and is not an atomic / stale sibling.
    pattern = f"{base}.clu.{_METHOD_TOKEN}.{_GROUP}.*"
    for p in sorted(glob.glob(pattern)):
        tail = p.rsplit(".", 1)[-1]
        if tail in ("curated", "autosave", "rollback"):
            continue
        return p
    return None


def _load_sorted_result(clu_path: str, res_path: str, *,
                          adapter: str, corpus_item: str,
                          sampling_rate: float = 20_000.0,
                          exclude_reserved: bool = True) -> SortedResult:
    """Build a SortedResult from a byte-level (.res, .clu) pair.

    Mirrors NS3SortingExtractor's exclude_reserved=True default so a
    curator's "delete to noise" / "move to MUA" drops out of the
    candidate unit list as it would in a sort read-back.
    """
    res = _read_res(res_path)
    clu = _read_clu(clu_path, n_spikes=res.size)

    kept = np.unique(clu)
    if exclude_reserved:
        kept = kept[kept >= 2]

    times_by_unit: dict[int, np.ndarray] = {}
    for uid in kept:
        t = np.sort(res[clu == uid])
        times_by_unit[int(uid)] = t

    return SortedResult(
        unit_ids=sorted(times_by_unit),
        spike_times=times_by_unit,
        sampling_rate=float(sampling_rate),
        adapter=adapter, corpus_item=corpus_item,
    )


def _iter_session_dirs(output_dir: str):
    """Yield (sorter, item, sess_dir) for every klusters_<sorter>/<item>/
    directory that holds a sim_session.yaml.
    """
    for sorter_dir in sorted(glob.glob(os.path.join(output_dir, "klusters_*"))):
        if not os.path.isdir(sorter_dir):
            continue
        sorter = os.path.basename(sorter_dir).removeprefix("klusters_")
        for item_dir in sorted(os.listdir(sorter_dir)):
            full = os.path.join(sorter_dir, item_dir)
            if not os.path.isfile(os.path.join(full, "sim_session.yaml")):
                continue
            yield sorter, item_dir, full


def _score_against_gt(candidate: SortedResult, gt: SortedResult,
                        *, scorer_names: Optional[list[str]] = None) -> list[dict]:
    """Run every registered scorer against (gt, candidate); return long rows."""
    names = scorer_names or list(scorer_registry.keys())
    rows: list[dict] = []
    for name in names:
        try:
            s = scorer_registry[name]
        except KeyError:
            continue
        try:
            res = s(gt, candidate)
        except Exception as e:                      # pragma: no cover
            rows.append({
                "scorer": name, "metric": "ERROR",
                "value": f"{type(e).__name__}: {e}",
            })
            continue
        for metric, v in res.values.items():
            rows.append({"scorer": name, "metric": metric, "value": float(v)})
    return rows


def run(args) -> int:
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(f"error: {output_dir}: not a directory"); return 1
    curated_tag = args.curated_tag

    # Collect GT SortedResults keyed by item slug — one pass, since a
    # given item may appear under many sorters.
    gt_by_item: dict[str, SortedResult] = {}
    for sorter, item, sess_dir in _iter_session_dirs(output_dir):
        if sorter != "ground_truth":
            continue
        base = os.path.join(sess_dir, "sim_session")
        gt_clu = _find_raw_clu(sess_dir, "ground_truth")
        if gt_clu is None:
            # GT may have been exported under the explicit "ground_truth" tag.
            cand = _clu_path(base, "ground_truth")
            gt_clu = cand if os.path.exists(cand) else None
        if gt_clu is None:
            print(f"warn: no GT .clu under {sess_dir}; skipping {item}")
            continue
        gt_res = _res_path(base)
        if not os.path.exists(gt_res):
            print(f"warn: no GT .res under {sess_dir}; skipping {item}")
            continue
        gt_by_item[item] = _load_sorted_result(
            gt_clu, gt_res, adapter="ground_truth", corpus_item=item,
        )

    # Score each (sorter, item) raw + curated against GT.
    scores_rows: list[dict] = []
    effort_rows: list[dict] = []
    seen = 0
    for sorter, item, sess_dir in _iter_session_dirs(output_dir):
        if sorter == "ground_truth":
            continue
        if item not in gt_by_item:
            print(f"warn: no GT for item {item!r}; skipping sorter={sorter}")
            continue

        base = os.path.join(sess_dir, "sim_session")
        raw_clu = _find_raw_clu(sess_dir, sorter)
        cur_clu = _clu_path(base, curated_tag)
        res_p   = _res_path(base)
        # Frozen .res snapshot written by klusters_export at sort time.
        # Klusters' realign operations rewrite the live .res in place
        # with new per-spike sample peaks; the raw-stage re-scoring must
        # read the sorter's ORIGINAL times (what the sort produced),
        # not the post-curation times.  Old exports pre-dating the
        # snapshot don't have .res.raw — in that case we fall back to
        # the live .res and warn once, since a rewrite there would
        # silently bias the raw-stage numbers.
        raw_res_p = f"{res_p}.raw"
        if not os.path.exists(raw_res_p):
            print(f"warn: {sess_dir}: no .res.raw snapshot; raw-stage "
                   "scoring uses the live .res and may read Klusters-"
                   "realigned samples.  Re-export this session to get "
                   "a frozen snapshot.")
            raw_res_p = res_p

        if raw_clu is None or not os.path.exists(res_p):
            print(f"warn: {sess_dir}: missing raw .clu or .res; skipping")
            continue

        gt = gt_by_item[item]

        # Stage: raw — read against the FROZEN .res snapshot so Klusters'
        # realign (if the user ran it during curation) doesn't shift the
        # raw sort's apparent spike times under the sort it originally
        # emitted.
        raw_sr = _load_sorted_result(raw_clu, raw_res_p,
                                        adapter=sorter, corpus_item=item)
        for r in _score_against_gt(raw_sr, gt):
            scores_rows.append({
                "sorter": sorter, "corpus_item": item, "stage": "raw",
                **r,
            })

        # Stage: curated (if the user actually curated it)
        if os.path.exists(cur_clu):
            seen += 1
            cur_sr = _load_sorted_result(cur_clu, res_p,
                                            adapter=f"{sorter}_curated",
                                            corpus_item=item)
            for r in _score_against_gt(cur_sr, gt):
                scores_rows.append({
                    "sorter": sorter, "corpus_item": item, "stage": "curated",
                    **r,
                })
            # Curation effort
            try:
                report = curate_diff(raw_clu, cur_clu, res_path=res_p)
            except Exception as e:                  # pragma: no cover
                print(f"warn: curate_diff failed on {cur_clu}: {e}")
            else:
                row = {
                    "sorter":            sorter,
                    "corpus_item":       item,
                    "n_total":           report.diff.n_total,
                    "n_units_before":    report.diff.n_units_before,
                    "n_units_after":     report.diff.n_units_after,
                    "n_reassignments":   report.diff.n_reassignments,
                    "fraction_touched":  report.diff.fraction_touched,
                    "n_merges":          report.diff.n_merges,
                    "n_splits":          report.diff.n_splits,
                    "purity_before":     report.diff.purity_before,
                    "purity_after":      report.diff.purity_after,
                }
                if report.log is not None:
                    row["n_actions_good"]   = report.log.n_actions_good
                    row["n_actions_bad"]    = report.log.n_actions_bad
                    row["complexity_score"] = report.log.complexity_score
                effort_rows.append(row)
        else:
            if args.require_curated:
                print(f"warn: no curated .clu at {cur_clu}")

    # Write artefacts
    scores_path = os.path.join(output_dir, "curated_scores.csv")
    effort_path = os.path.join(output_dir, "curation_effort.csv")

    if scores_rows:
        keys = list(scores_rows[0].keys())
        with open(scores_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(scores_rows)
    if effort_rows:
        # Collect union of keys (log-based columns may be absent for some rows)
        keys: list[str] = []
        for row in effort_rows:
            for k in row:
                if k not in keys: keys.append(k)
        with open(effort_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in effort_rows:
                w.writerow({k: r.get(k, "") for k in keys})

    print(f"wrote {scores_path}  ({len(scores_rows)} rows)")
    print(f"wrote {effort_path}  ({len(effort_rows)} rows)")
    print(f"scored {seen} curated session(s) across "
          f"{len({r['sorter'] for r in effort_rows})} sorter(s)")
    return 0


def register(subparsers) -> None:
    ap = subparsers.add_parser(
        "score-curated",
        help="score curated .clu outputs against the GT + "
             "report curation effort",
    )
    ap.add_argument("output_dir",
                    help="run directory containing klusters_<sorter>/ dirs")
    ap.add_argument("--curated-tag", default="curated",
                    help="stage tag Klusters wrote the curated .clu under "
                         "(default: curated)")
    ap.add_argument("--require-curated", action="store_true",
                    help="warn when a (sorter, item) has no curated .clu "
                         "(default: silently skip)")
    # Harness dispatcher reads `args.fn`; match the convention every
    # other command uses.  Historical `func=run` here never fired —
    # `main` raised AttributeError on `args.fn`.
    ap.set_defaults(fn=run)
