"""Curation-effort scoring.

Given a pre-curation sort (`.res` + `.clu`) and a post-curation sort
(`.res` + `.clu`, same session), measure how much human work went into the
curation.

Two metric sets, produced side by side:

**Diff-based** — always computable from the two `.clu` files alone:

  n_units_before     / n_units_after
  n_spikes_before    / n_spikes_after
  n_reassignments    — spikes whose cluster_id differs (0 when counts mismatch)
  fraction_touched   — n_reassignments / n_spikes_before
  n_merges           — rows of the (before × after) contingency that collapse
                       into one column (per-row-argmax column duplicates)
  n_splits           — columns that come from multiple rows
  purity_shift       — Σ_c max_row C[row, c] / N_after − Σ_r max_col C[r, c] / N_before

**Log-based** — present when the Klusters `.cur.<method>.<group>[.<tag>]`
log sits next to the `.clu`.  The log is JSON Lines; each curator action
emits one "before" row per source cluster + one "after" row per result
cluster + zero or more "ACTION_DETAIL" rows, every row carrying the same
`action_idx` and a `status` of `"good"` or `"bad"` (undone).

  n_actions_good     — number of distinct `action_idx`s marked good
  n_actions_bad      — undo-flipped-to-bad counterpart
  n_<ACTION_TYPE>    — per-type good-action counts (GROUP, SPLIT, SPLIT_N,
                       RECLUSTER, REALIGN, NUDGE, WATERSHED, DELETE_*,
                       MOVE_SPIKES, SORT_ALL, RENUMBER_PARTIAL)
  complexity_score   — weighted sum of good actions (weights in
                       `_ACTION_WEIGHT` below)

The weights are a rough curator-time proxy: a GROUP (merge) is a 1-click
affair, a SPLIT costs a few-second feature-space decision (weight 2), a
RECLUSTER / REALIGN / WATERSHED kicks off an algorithmic re-pass that the
curator must then review (weight 3).  Tweak the weights per lab convention
by overriding the dict.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np


# ─── byte-level readers ─────────────────────────────────────────────────────
_RES_DTYPE = np.dtype("<i8")
_CLU_DTYPE = np.dtype("<i4")


def _read_res(path: str) -> np.ndarray:
    """Binary little-endian int64, no header.  Matches fiber-kit's RES_DTYPE."""
    return np.fromfile(path, dtype=_RES_DTYPE).astype(np.int64)


def _read_clu(path: str, *, n_spikes: Optional[int] = None) -> np.ndarray:
    """Binary little-endian int32: header (n_distinct) + ids.  Matches fiber-kit's
    `read_clu_file`.  Enforces n_spikes match when given."""
    raw = np.fromfile(path, dtype=_CLU_DTYPE)
    if raw.size == 0:
        return np.array([], dtype=np.int64)
    ids = raw[1:].astype(np.int64)
    if n_spikes is not None and ids.size != n_spikes:
        raise ValueError(
            f"{path}: {ids.size} cluster ids for {n_spikes} spikes — "
            "stale sort, or .res / .clu mismatch?"
        )
    return ids


# ─── session-path derivation ────────────────────────────────────────────────
def _swap_type_token(clu_path: str, new_type: str) -> str:
    """`<base>.clu.<v>.<g>[.<suffix>]` → `<base>.<new_type>.<v>.<g>[.<suffix>]`.

    Splits on the LAST `.clu.` token in the path so a session named
    `something.clu.old` still parses.
    """
    idx = clu_path.rfind(".clu.")
    if idx < 0:
        raise ValueError(f"not a .clu path: {clu_path}")
    return clu_path[:idx] + f".{new_type}." + clu_path[idx + len(".clu."):]


def _res_path_for_clu(clu_path: str) -> str:
    """`.clu.<v>.<g>[.<suffix>]` → sibling `.res.<v>.<g>`.

    Per ns3's naming contract (doc/ndmanager-plugins/formats/naming.md), `.res`
    is a Shared artifact — no stage suffix.  A curator who reextracts (realign /
    de-collide) writes a staged `.res.<v>.<g>.<suffix>`; try that first, fall
    back to the shared one.
    """
    staged = _swap_type_token(clu_path, "res")
    if os.path.exists(staged):
        return staged
    # Drop the trailing suffix if any: "<base>.res.<v>.<g>.<suffix>" → "<base>.res.<v>.<g>"
    parts = staged.split(".")
    # Find ".res.<v>.<g>" prefix: res token is somewhere in the middle
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "res" and i + 2 < len(parts):
            shared = ".".join(parts[: i + 3])
            if os.path.exists(shared):
                return shared
    return staged   # path returned for the error message


def _cur_path_for_clu(clu_path: str) -> str:
    """`.clu.<v>.<g>[.<suffix>]` → `.cur.<v>.<g>[.<suffix>]` (same stage)."""
    return _swap_type_token(clu_path, "cur")


# ─── log parsing ────────────────────────────────────────────────────────────
_ACTION_WEIGHT: Mapping[str, float] = {
    # merges / deletes / nudges: one-click decisions
    "GROUP":                 1.0,
    "DELETE_NOISE":          1.0,
    "DELETE_ARTEFACT":       1.0,
    "DELETE_REGION_NOISE":   1.0,
    "DELETE_REGION_ARTEFACT":1.0,
    "MOVE_SPIKES":           1.0,
    "NUDGE":                 1.0,
    "RENUMBER_PARTIAL":      0.5,       # mostly automatic
    # splits: curator must choose the cut in feature space
    "SPLIT":                 2.0,
    "SPLIT_N":               2.0,
    # algorithmic re-passes that still need a human review afterwards
    "RECLUSTER":             3.0,
    "REALIGN":               3.0,
    "WATERSHED":             3.0,
    "SORT_ALL":              4.0,
}


@dataclass
class CurationLogSummary:
    """Parsed `.cur` log, collapsed to per-action status + type."""
    n_actions_good: int                       = 0
    n_actions_bad: int                        = 0
    actions_by_type_good: Mapping[str, int]   = field(default_factory=dict)
    actions_by_type_bad: Mapping[str, int]    = field(default_factory=dict)
    complexity_score: float                   = 0.0


def parse_cur_log(path: str,
                   weights: Mapping[str, float] = _ACTION_WEIGHT
                   ) -> CurationLogSummary:
    """Collapse a Klusters JSON-Lines `.cur` log into per-action counts.

    The log emits multiple rows per action (one before-row per source
    cluster + one after-row per result cluster + optional ACTION_DETAILs,
    all sharing `action_idx`).  We dedupe by `(action_idx, status)` so the
    returned count is curator-action count, not row count.
    """
    actions: dict[int, Tuple[str, str]] = {}   # action_idx → (type, status)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = row.get("action_idx")
            if not isinstance(idx, int):
                continue
            atype = str(row.get("action", "UNKNOWN"))
            status = str(row.get("status", "good"))
            # Later rows for the same action may flip its status
            # (good ↔ bad on undo / redo); keep the latest.
            actions[idx] = (atype, status)

    n_good = 0
    n_bad = 0
    by_good: dict[str, int] = {}
    by_bad: dict[str, int] = {}
    score = 0.0
    for atype, status in actions.values():
        if status == "good":
            n_good += 1
            by_good[atype] = by_good.get(atype, 0) + 1
            score += float(weights.get(atype, 1.0))
        else:
            n_bad += 1
            by_bad[atype] = by_bad.get(atype, 0) + 1
    return CurationLogSummary(
        n_actions_good=n_good,
        n_actions_bad=n_bad,
        actions_by_type_good=by_good,
        actions_by_type_bad=by_bad,
        complexity_score=score,
    )


# ─── diff over two .clu arrays ──────────────────────────────────────────────
@dataclass
class ClusteringDiff:
    """Spike-level diff between two cluster-id arrays on the SAME .res."""
    n_total: int
    n_units_before: int
    n_units_after: int
    n_reassignments: int
    n_merges: int        # rows whose argmax column is shared with another row
    n_splits: int        # columns whose argmax row is shared with another column
    purity_before: float
    purity_after: float

    @property
    def fraction_touched(self) -> float:
        return float(self.n_reassignments) / self.n_total if self.n_total else 0.0


def diff_clus(before: np.ndarray, after: np.ndarray,
               *, exclude_reserved: bool = True) -> ClusteringDiff:
    """Compare two parallel cluster-id arrays.

    `exclude_reserved=True` (default) drops Klusters ids 0 (artefact) and 1
    (noise) from the row/column counts but STILL counts their reassignments
    — a delete-to-noise is a legitimate curator action.
    """
    before = np.asarray(before, dtype=np.int64)
    after = np.asarray(after, dtype=np.int64)
    if before.size != after.size:
        raise ValueError(
            f"before has {before.size} spikes, after has {after.size}; "
            "same .res is required"
        )
    n_total = int(before.size)
    n_reassignments = int(np.sum(before != after))

    def _units(arr: np.ndarray) -> np.ndarray:
        u = np.unique(arr)
        if exclude_reserved:
            u = u[u >= 2]
        return u

    u_before = _units(before)
    u_after = _units(after)
    if n_total == 0 or u_before.size == 0 or u_after.size == 0:
        return ClusteringDiff(
            n_total=n_total,
            n_units_before=int(u_before.size),
            n_units_after=int(u_after.size),
            n_reassignments=n_reassignments,
            n_merges=0, n_splits=0,
            purity_before=0.0, purity_after=0.0,
        )

    # Contingency table on the non-reserved units; map ids to dense indices
    ib = np.searchsorted(u_before, before)
    ia = np.searchsorted(u_after, after)
    mask = np.isin(before, u_before) & np.isin(after, u_after)
    ib = ib[mask]
    ia = ia[mask]
    tab = np.zeros((u_before.size, u_after.size), dtype=np.int64)
    np.add.at(tab, (ib, ia), 1)

    # n_merges = number of before-rows whose argmax column has >= 2 rows
    # mapping onto it (two rows both prefer column c → merged into c).
    if tab.size > 0:
        col_of_row = tab.argmax(axis=1)
        _, counts = np.unique(col_of_row, return_counts=True)
        n_merges = int(np.sum(np.maximum(counts - 1, 0)))

        row_of_col = tab.argmax(axis=0)
        _, counts = np.unique(row_of_col, return_counts=True)
        n_splits = int(np.sum(np.maximum(counts - 1, 0)))
    else:
        n_merges = n_splits = 0

    n_in_table = int(tab.sum())
    purity_before = float(tab.max(axis=1).sum()) / n_in_table if n_in_table else 0.0
    purity_after  = float(tab.max(axis=0).sum()) / n_in_table if n_in_table else 0.0

    return ClusteringDiff(
        n_total=n_total,
        n_units_before=int(u_before.size),
        n_units_after=int(u_after.size),
        n_reassignments=n_reassignments,
        n_merges=n_merges,
        n_splits=n_splits,
        purity_before=purity_before,
        purity_after=purity_after,
    )


# ─── top-level entry point ─────────────────────────────────────────────────
@dataclass
class CurationReport:
    """Everything `curate-diff` produces for one (before.clu, after.clu) pair."""
    before_clu: str
    after_clu: str
    cur_log: Optional[str]
    diff: ClusteringDiff
    log: Optional[CurationLogSummary]

    def to_rows(self) -> list[dict]:
        rows: list[dict] = []
        d = self.diff
        for metric, value in [
            ("n_total",          d.n_total),
            ("n_units_before",   d.n_units_before),
            ("n_units_after",    d.n_units_after),
            ("n_reassignments",  d.n_reassignments),
            ("fraction_touched", d.fraction_touched),
            ("n_merges",         d.n_merges),
            ("n_splits",         d.n_splits),
            ("purity_before",    d.purity_before),
            ("purity_after",     d.purity_after),
        ]:
            rows.append({"source": "diff", "metric": metric, "value": float(value)})
        if self.log is not None:
            lg = self.log
            rows.append({"source": "log", "metric": "n_actions_good",
                          "value": float(lg.n_actions_good)})
            rows.append({"source": "log", "metric": "n_actions_bad",
                          "value": float(lg.n_actions_bad)})
            rows.append({"source": "log", "metric": "complexity_score",
                          "value": float(lg.complexity_score)})
            for atype, n in sorted(lg.actions_by_type_good.items()):
                rows.append({"source": "log", "metric": f"n_{atype}_good",
                              "value": float(n)})
        return rows


def curate_diff(before_clu: str, after_clu: str,
                 *, res_path: Optional[str] = None,
                 cur_log_path: Optional[str] = None,
                 weights: Mapping[str, float] = _ACTION_WEIGHT
                 ) -> CurationReport:
    """Measure curator effort from a (before.clu, after.clu) pair.

    `res_path` defaults to the shared `.res` paired with `after_clu`;
    `cur_log_path` defaults to the matching `.cur.<v>.<g>[.<tag>]` next
    to `after_clu`.  Both are optional overrides.
    """
    if res_path is None:
        res_path = _res_path_for_clu(after_clu)
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"no .res for .clu pair: {res_path}")

    n_spikes = int(os.path.getsize(res_path) // _RES_DTYPE.itemsize)
    before = _read_clu(before_clu, n_spikes=n_spikes)
    after = _read_clu(after_clu, n_spikes=n_spikes)
    diff = diff_clus(before, after)

    log: Optional[CurationLogSummary] = None
    if cur_log_path is None:
        candidate = _cur_path_for_clu(after_clu)
        if os.path.exists(candidate):
            cur_log_path = candidate
    if cur_log_path and os.path.exists(cur_log_path):
        log = parse_cur_log(cur_log_path, weights=weights)

    return CurationReport(
        before_clu=before_clu,
        after_clu=after_clu,
        cur_log=cur_log_path,
        diff=diff,
        log=log,
    )
