"""Convenience wrappers for the SpikeForest PAIRED_* study sets.

These are the ground-truth datasets with a simultaneously-recorded reference
spike train — intracellular patch, juxtacellular, or (for PAIRED_MONOTRODE) a
sub-selection down to a single extracellular channel.  Every sorter sees the
extracellular trace; the scorer compares its unit timings against the
reference cell's known spikes.

Shortcuts (each returns `Corpus`):

    load_paired_crcns_hc1(item_name)     Buzsaki rat hippocampus tetrode + patch
    load_paired_kampff(item_name)        Kampff mouse cortex probe + patch
    load_paired_english(item_name)       English awake mouse + juxtacellular
    load_paired_boyden(item_name)        Boyden mouse cortex + patch
    load_paired_mea64c_yger(item_name)   Yger mouse retina MEA + patch
    load_paired_monotrode(item_name)     1-ch sub-selected paired recordings

Discover item names with

    python -m ns3_sort_bench.corpus.spikeforest list --study-set PAIRED_CRCNS_HC1

Each item name is a `study_name` string the SpikeForest index assigns, e.g.
`paired_crcns_hc1_d533101` or `paired_kampff_2014_11_25_Pair_3_0`.

Scoring recipe for paired data
------------------------------
Pass the loaded Corpus to BenchmarkRun with the PairwiseScorer at
`delta_frames=12` (≈0.4 ms at 30 kHz) and the PerCellScorer; because the
ground truth has typically one unit, PerCellScorer gives you the best-unit
accuracy per intracellular cell — the usual paired-recordings metric.
"""
from __future__ import annotations

from typing import Optional

from .._base import Corpus
from . import spikeforest as _sf


def _load_paired(study_set: str, item_name: str,
                 *, uri: Optional[str] = None,
                 ns3_yaml: Optional[dict] = None) -> Corpus:
    """Shared body: resolve the item within a study set, then load."""
    # SpikeForest's `study_name` IS the item identifier for the paired sets
    # (each paired recording is its own study with a single recording).  Allow
    # the caller to pass either the full study_name or just the trailing id.
    all_items = _sf.list_recordings(study_set=study_set, uri=uri)
    if not all_items:
        raise RuntimeError(
            f"No recordings found for study_set={study_set!r}.  "
            "Check kachery-cloud auth (first run) and `python -m "
            "ns3_sort_bench.corpus.spikeforest list` output."
        )
    matches = [r for r in all_items
               if r["study_name"] == item_name
               or r["recording_name"] == item_name
               or r["study_name"].endswith("_" + item_name)]
    if not matches:
        names = [r["study_name"] for r in all_items]
        raise KeyError(
            f"item_name={item_name!r} not found in {study_set}.  "
            f"Available ({len(names)}): {names[:5]}..."
        )
    hit = matches[0]
    return _sf.load(
        study_name=hit["study_name"],
        recording_name=hit["recording_name"],
        uri=uri, ns3_yaml=ns3_yaml, corpus_kind="paired",
    )


def load_paired_crcns_hc1(item_name: str, *, ns3_yaml: Optional[dict] = None,
                           uri: Optional[str] = None) -> Corpus:
    """Buzsaki lab rat hippocampal tetrode + intracellular reference.

    Anatomically matched to CA1 extracellular recordings.  Tetrodes or
    4-6 ch silicon shanks; the ground-truth unit is one patched cell.

    Example:
        corpus = load_paired_crcns_hc1("d533101")
    """
    return _load_paired("PAIRED_CRCNS_HC1", item_name,
                        ns3_yaml=ns3_yaml, uri=uri)


def load_paired_kampff(item_name: str, *, ns3_yaml: Optional[dict] = None,
                        uri: Optional[str] = None) -> Corpus:
    """Kampff lab mouse cortex 32-ch probe + whole-cell patch reference.

    Best-SNR paired set; often used as the headline paired-sorting benchmark.
    Example:
        corpus = load_paired_kampff("2014_11_25_Pair_3_0")
    """
    return _load_paired("PAIRED_KAMPFF", item_name,
                        ns3_yaml=ns3_yaml, uri=uri)


def load_paired_english(item_name: str, *, ns3_yaml: Optional[dict] = None,
                         uri: Optional[str] = None) -> Corpus:
    """English lab awake mouse recordings + juxtacellular reference."""
    return _load_paired("PAIRED_ENGLISH", item_name,
                        ns3_yaml=ns3_yaml, uri=uri)


def load_paired_boyden(item_name: str, *, ns3_yaml: Optional[dict] = None,
                        uri: Optional[str] = None) -> Corpus:
    """Boyden lab mouse cortex 32-ch probe + whole-cell patch reference."""
    return _load_paired("PAIRED_BOYDEN", item_name,
                        ns3_yaml=ns3_yaml, uri=uri)


def load_paired_mea64c_yger(item_name: str, *, ns3_yaml: Optional[dict] = None,
                             uri: Optional[str] = None) -> Corpus:
    """Yger lab mouse retina 64-of-252 MEA + patch reference."""
    return _load_paired("PAIRED_MEA64C_YGER", item_name,
                        ns3_yaml=ns3_yaml, uri=uri)


def load_paired_monotrode(item_name: str, *, ns3_yaml: Optional[dict] = None,
                           uri: Optional[str] = None) -> Corpus:
    """1-channel sub-selection from the four paired sets above."""
    return _load_paired("PAIRED_MONOTRODE", item_name,
                        ns3_yaml=ns3_yaml, uri=uri)
