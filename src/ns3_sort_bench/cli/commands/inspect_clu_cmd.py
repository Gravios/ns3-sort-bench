"""``ns3sb inspect-clu`` — surface the raw truth of a klusters session.

For one ``klusters_<sorter>/<item>/`` directory, prints:

    * per-source cluster-id → spike-count breakdown (raw sort, curated
      if present, ground truth if the klusters_ground_truth/<item>/
      dir is there);
    * cluster IDs that would be DROPPED by score-curated's reserved
      filter (Klusters-convention IDs 0 and 1);
    * pairwise contingency tables (confusion matrices) between every
      combination of the three — raw vs GT, curated vs GT, raw vs
      curated — with spike-time near-matching under the harness-default
      ±12 samples (0.6 ms at 20 kHz).

Use this when score-curated's rolled-up metrics (iou / recall / prec /
n_match) disagree in a way that makes no sense — it almost always
turns out to be a reserved-ID cluster hiding the real cell, or a
Klusters realign shifting spike times under the sort.

Invocation::

    ns3sb inspect-clu <run_dir> --item <item> --sorter <sorter>
                                [--group <g>] [--method <m>]
                                [--delta-frames <n>]
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from .._util import eprint
from ...curation import _read_clu, _read_res


_METHOD_DEFAULT = "standard"
_GROUP_DEFAULT = 1


def _clu_path(sess_dir: str, method: str, group: int,
               tag: Optional[str] = None) -> str:
    base = os.path.join(sess_dir, "sim_session")
    core = f"{base}.clu.{method}.{group}"
    return f"{core}.{tag}" if tag else core


def _res_path(sess_dir: str, method: str, group: int, *,
                prefer_raw_snapshot: bool = True) -> Optional[str]:
    base = os.path.join(sess_dir, "sim_session")
    live = f"{base}.res.{method}.{group}"
    raw  = f"{live}.raw"
    if prefer_raw_snapshot and os.path.isfile(raw):
        return raw
    if os.path.isfile(live):
        return live
    return None


def _counts_table(ids: np.ndarray) -> list[tuple[int, int, bool]]:
    """(cluster_id, count, reserved?) rows sorted by id ascending."""
    out: list[tuple[int, int, bool]] = []
    for u in sorted(np.unique(ids).tolist()):
        out.append((int(u), int((ids == u).sum()), int(u) < 2))
    return out


def _print_counts_block(label: str, ids: np.ndarray) -> None:
    if ids.size == 0:
        print(f"\n{label}: (empty)")
        return
    rows = _counts_table(ids)
    total = ids.size
    dropped = sum(c for _, c, r in rows if r)
    kept_clusters = [c for _, c, r in rows if not r]
    print(f"\n{label}: {total} spikes in {len(rows)} cluster(s); "
          f"{dropped} would be dropped by score-curated's reserved filter "
          f"(IDs 0 & 1).")
    print("  cluster_id  spikes  note")
    for cid, cnt, reserved in rows:
        note = "RESERVED — DROPPED from scoring" if reserved else ""
        print(f"  {cid:>10d}  {cnt:>6d}  {note}")
    if kept_clusters:
        print(f"  kept for scoring: {len(kept_clusters)} cluster(s), "
              f"{sum(kept_clusters)} spikes.")


def _align_near(a_times: np.ndarray, a_lab: np.ndarray,
                 b_times: np.ndarray, b_lab: np.ndarray,
                 delta: int) -> tuple[np.ndarray, np.ndarray]:
    """For each spike in `a`, find the nearest spike in `b` within
    `delta` samples; return paired labels (unmatched a-side slots get
    b-label = -1).  First-come-first-matched, same semantics as the
    scorers.
    """
    a_times = np.asarray(a_times, np.int64)
    b_times = np.asarray(b_times, np.int64)
    a_lab = np.asarray(a_lab, np.int64)
    b_lab = np.asarray(b_lab, np.int64)

    if a_times.size == 0 or b_times.size == 0:
        return a_lab, np.full(a_times.size, -1, np.int64)

    # Walk b; claim nearest a slot.
    order = np.argsort(b_times, kind="stable")
    b_times_s = b_times[order]; b_lab_s = b_lab[order]
    a_order = np.argsort(a_times, kind="stable")
    a_times_s = a_times[a_order]
    # Map a-slot indices back to the caller's order so the return array
    # is aligned to the ORIGINAL a_times argument.
    inverse_a = np.empty_like(a_order); inverse_a[a_order] = np.arange(a_order.size)

    paired_b = np.full(a_times_s.size, -1, np.int64)
    taken    = np.zeros(a_times_s.size, dtype=bool)
    j = np.searchsorted(a_times_s, b_times_s)
    for k, t in enumerate(b_times_s):
        best = -1; best_d = delta + 1
        for cand in (j[k] - 1, j[k]):
            if 0 <= cand < a_times_s.size and not taken[cand]:
                d = abs(int(a_times_s[cand]) - int(t))
                if d <= delta and d < best_d:
                    best = cand; best_d = d
        if best >= 0:
            paired_b[best] = int(b_lab_s[k])
            taken[best] = True

    return a_lab, paired_b[inverse_a]


def _contingency(ra: np.ndarray, rb: np.ndarray
                  ) -> tuple[np.ndarray, list[int], list[int]]:
    mask = (ra >= 0) & (rb >= 0)
    ra, rb = ra[mask], rb[mask]
    if ra.size == 0:
        return np.zeros((0, 0), np.int64), [], []
    u_a = sorted(set(ra.tolist()))
    u_b = sorted(set(rb.tolist()))
    idx_a = {v: i for i, v in enumerate(u_a)}
    idx_b = {v: i for i, v in enumerate(u_b)}
    tab = np.zeros((len(u_a), len(u_b)), np.int64)
    for x, y in zip(ra, rb):
        tab[idx_a[int(x)], idx_b[int(y)]] += 1
    return tab, u_a, u_b


def _print_contingency(label_a: str, label_b: str,
                         tab: np.ndarray,
                         u_a: list[int], u_b: list[int]) -> None:
    if tab.size == 0:
        print(f"\n{label_a} × {label_b}: (no overlap within tolerance)")
        return
    print(f"\n{label_a} × {label_b}  (near-matched spikes, "
          f"confusion matrix: rows={label_a}, cols={label_b})")
    header = "".join(f"{v:>8d}" for v in u_b)
    print(f"  {'':>10s} {header}  row_sum  note")
    for i, va in enumerate(u_a):
        row = tab[i]
        rowsum = int(row.sum())
        note = "RESERVED" if va < 2 else ""
        cells = "".join(f"{int(c):>8d}" for c in row)
        print(f"  {va:>10d}{cells}  {rowsum:>7d}  {note}")
    col_sum = "".join(f"{int(c):>8d}" for c in tab.sum(axis=0))
    print(f"  {'col_sum':>10s}{col_sum}  {int(tab.sum()):>7d}")
    print("  RESERVED rows (cluster id < 2) are dropped from score-curated's "
          "scoring; a large count there is the usual cause of unexpected "
          "low recall / n_match in the summary.")


def _load_from_sess(sess_dir: str, method: str, group: int,
                      curated_tag: str
                      ) -> tuple[Optional[np.ndarray],
                                 Optional[np.ndarray],
                                 Optional[np.ndarray]]:
    """Return (res_times, raw_clu_ids, curated_clu_ids) for one session
    dir; any of them may be None when the file is missing."""
    res_p = _res_path(sess_dir, method, group)
    res = _read_res(res_p) if res_p else None

    raw_clu_p = _clu_path(sess_dir, method, group, tag=None)
    raw_ids = _read_clu(raw_clu_p, n_spikes=res.size) if \
        (res is not None and os.path.isfile(raw_clu_p)) else None

    cur_clu_p = _clu_path(sess_dir, method, group, tag=curated_tag)
    cur_ids = _read_clu(cur_clu_p, n_spikes=res.size) if \
        (res is not None and os.path.isfile(cur_clu_p)) else None

    return res, raw_ids, cur_ids


def run(args) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        eprint(f"{run_dir}: not a directory"); return 1

    method = args.method
    group  = args.group
    delta  = args.delta_frames
    curated_tag = args.curated_tag

    item_slug = args.item
    # The slug form stored on disk (score-curated's dir names).
    # We don't try to reverse-slugify: the user passes the on-disk slug
    # directly (same they'd see in `ls klusters_<sorter>/`).

    sorter_dir = run_dir / f"klusters_{args.sorter}" / item_slug
    gt_dir     = run_dir / "klusters_ground_truth"  / item_slug

    if not sorter_dir.is_dir():
        eprint(f"missing session dir: {sorter_dir}"); return 1

    print(f"# inspect-clu  sorter={args.sorter}  item={item_slug}  "
          f"method={method}  group={group}  delta_frames={delta}")

    # Load each side.
    sorter_res, sorter_raw, sorter_cur = _load_from_sess(
        str(sorter_dir), method, group, curated_tag)
    gt_res = gt_raw = None
    if gt_dir.is_dir():
        gt_res, gt_raw, _gt_cur = _load_from_sess(
            str(gt_dir), method, group, curated_tag)

    print(f"\n{sorter_dir}:  "
          f"{sorter_res.size if sorter_res is not None else '—'} spikes "
          f"in the sorter's .res")
    if gt_res is not None:
        print(f"{gt_dir}:  "
              f"{gt_res.size} spikes in the GT .res")

    # Per-source cluster-id → count tables.
    if sorter_raw is not None:
        _print_counts_block(f"raw {args.sorter}", sorter_raw)
    else:
        print(f"\nraw {args.sorter}: (no .clu or .res)")
    if sorter_cur is not None:
        _print_counts_block(f"curated {args.sorter} (tag: {curated_tag})",
                              sorter_cur)
    else:
        print(f"\ncurated {args.sorter}: (no .clu.*.{curated_tag} found)")
    if gt_raw is not None:
        _print_counts_block("ground_truth", gt_raw)
    else:
        print("\nground_truth: (no GT dir or .clu for this item)")

    # Contingencies.  Each one does a first-come-first-matched join on
    # spike times within ±delta samples, matching the scorers' semantics.
    if (sorter_raw is not None and gt_raw is not None
            and sorter_res is not None and gt_res is not None):
        a_lab, b_lab = _align_near(sorter_res, sorter_raw,
                                     gt_res, gt_raw, delta)
        tab, u_a, u_b = _contingency(a_lab, b_lab)
        _print_contingency(f"raw {args.sorter}", "ground_truth",
                            tab, u_a, u_b)

    if (sorter_cur is not None and gt_raw is not None
            and sorter_res is not None and gt_res is not None):
        a_lab, b_lab = _align_near(sorter_res, sorter_cur,
                                     gt_res, gt_raw, delta)
        tab, u_a, u_b = _contingency(a_lab, b_lab)
        _print_contingency(f"curated {args.sorter}", "ground_truth",
                            tab, u_a, u_b)

    if sorter_cur is not None and sorter_raw is not None:
        # Same .res on both sides — index-by-index join.
        tab, u_a, u_b = _contingency(sorter_raw, sorter_cur)
        _print_contingency(f"raw {args.sorter}",
                            f"curated {args.sorter}",
                            tab, u_a, u_b)
    return 0


def register(subparsers) -> None:
    ap = subparsers.add_parser(
        "inspect-clu",
        help="raw .clu / .res truth for one klusters session: spike "
             "counts per cluster + confusion matrices against the GT "
             "and curated .clu",
    )
    ap.add_argument("run_dir", help="run directory containing "
                                         "klusters_<sorter>/ dirs")
    ap.add_argument("--item",  required=True,
                    help="item slug (the directory name under "
                         "klusters_<sorter>/)")
    ap.add_argument("--sorter", required=True,
                    help="sorter name (e.g. fiber_kit)")
    ap.add_argument("--group",  type=int, default=_GROUP_DEFAULT)
    ap.add_argument("--method", default=_METHOD_DEFAULT)
    ap.add_argument("--curated-tag", default="curated",
                    help="stage tag for the curated .clu (default: curated)")
    ap.add_argument("--delta-frames", type=int, default=12,
                    help="near-match tolerance in samples (default: 12, "
                         "~0.6 ms at 20 kHz; matches per_cell + pairwise)")
    ap.set_defaults(fn=run)
