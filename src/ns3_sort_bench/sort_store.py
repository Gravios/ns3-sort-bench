"""Every finished sort saved to disk, so a resumed run can still use it.

``<output_dir>/sorts/<item>/<adapter>.npz`` holds one SortedResult's
spike times and labels plus its bookkeeping (no templates, no extras).
``ns3sb run --skip-done`` reloads them for pairs it doesn't sort again,
so a Klusters export asked for later, or a rescore, needs no re-sort.
"""
from __future__ import annotations

import os
import re
from typing import Optional

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
    from ._base import SortedResult
    path = sort_path(output_dir, corpus_item, adapter)
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
