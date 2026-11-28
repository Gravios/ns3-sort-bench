"""Every finished sort saved to disk, so a resumed run can still use it.

``<output_dir>/sorts/<item>/<adapter>.npz`` holds one SortedResult's
spike times and labels plus its bookkeeping (no templates, no extras).
``ns3sb run --skip-done`` reloads them for pairs it doesn't sort again,
and ``ns3sb rescore`` scores them against the reference as it is now,
so a Klusters export asked for later, or a moving curation, needs no
re-sort.  ``corpora.json`` maps each item to the corpus key it came
from, so the reference can be resolved again.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

import numpy as np


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def sort_path(output_dir: str, corpus_item: str, adapter: str) -> str:
    return os.path.join(output_dir, "sorts", _slug(corpus_item),
                        _slug(adapter) + ".npz")


def save_sort(output_dir: str, result) -> str:
    """Write *result* atomically (tmp file + rename); returns the path."""
    path = sort_path(output_dir, result.corpus_item, result.adapter)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    units = [int(u) for u in result.unit_ids]
    times = [np.asarray(result.spike_times.get(u, ()), np.int64) for u in units]
    labels = [np.full(len(t), u, np.int64) for u, t in zip(units, times)]
    tmp = path + ".tmp.npz"
    np.savez(
        tmp,
        unit_ids=np.asarray(units, np.int64),
        times=np.concatenate(times) if times else np.zeros(0, np.int64),
        labels=np.concatenate(labels) if labels else np.zeros(0, np.int64),
        sampling_rate=float(result.sampling_rate),
        adapter=str(result.adapter), corpus_item=str(result.corpus_item),
        adapter_version=str(result.adapter_version),
        wall_time_s=float(result.wall_time_s),
        peak_rss_mb=float(result.peak_rss_mb),
    )
    os.replace(tmp, path)
    return path


def load_sort(output_dir: str, corpus_item: str, adapter: str) -> Optional[object]:
    """The saved SortedResult, or None when there is none (or it's unreadable)."""
    return load_sort_file(sort_path(output_dir, corpus_item, adapter))


def saved_sorts(output_dir: str, corpus_item: str) -> List[object]:
    """Every readable saved sort of *corpus_item*, in file-name order."""
    d = os.path.dirname(sort_path(output_dir, corpus_item, "x"))
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz") and not f.endswith(".tmp.npz"):
            r = load_sort_file(os.path.join(d, f))
            if r is not None:
                out.append(r)
    return out


def corpora_path(output_dir: str) -> str:
    return os.path.join(output_dir, "corpora.json")


def record_corpora(output_dir: str, keys: Dict[str, str]) -> None:
    """Merge {corpus_item: corpus key} into <output_dir>/corpora.json, so
    `ns3sb rescore` can resolve each item's reference again."""
    path = corpora_path(output_dir)
    known = read_corpora(output_dir)
    known.update(keys)
    os.makedirs(output_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(known, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def read_corpora(output_dir: str) -> Dict[str, str]:
    try:
        with open(corpora_path(output_dir)) as f:
            d = json.load(f)
        return {str(k): str(v) for k, v in d.items()} if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def load_sort_file(path: str) -> Optional[object]:
    """A saved SortedResult read from *path*, or None if it can't be read."""
    from ._base import SortedResult
    try:
        with np.load(path, allow_pickle=False) as z:
            units = [int(u) for u in z["unit_ids"]]
            times, labels = z["times"], z["labels"]
            order = np.argsort(labels, kind="stable")
            times, labels = times[order], labels[order]
            spike_times = {u: times[np.searchsorted(labels, u, "left"):
                                    np.searchsorted(labels, u, "right")]
                           for u in units}
            return SortedResult(
                unit_ids=units, spike_times=spike_times,
                sampling_rate=float(z["sampling_rate"]),
                adapter=str(z["adapter"]), corpus_item=str(z["corpus_item"]),
                adapter_version=str(z["adapter_version"]),
                wall_time_s=float(z["wall_time_s"]),
                peak_rss_mb=float(z["peak_rss_mb"]),
            )
    except (OSError, KeyError, ValueError):
        return None
