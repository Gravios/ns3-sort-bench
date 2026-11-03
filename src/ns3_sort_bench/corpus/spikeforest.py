"""SpikeForest corpus loader (Flatiron Institute, Magland et al. 2020).

Pulls ground-truth-labelled recordings via SpikeInterface's extractors.
Preferred items for this harness (per
`claude/botm-vs-fiber-deconv-audit-2026-11-02.md`):

    PAIRED_CRCNS_HC1       — Buzsaki lab tetrode paired recordings
                             (closest to the lab's octrode geometry)
    SYNTH_MEAREC_TETRODE   — MEArec simulated tetrode, full ground truth

Status: SCAFFOLD — the SpikeInterface SpikeForest wrapper has moved
across releases (0.100: `spikeinterface.extractors.spikeforest`; later
rolled into separate datalad / dandi loaders).  Code path exists but is
tagged with a `NotImplementedError` message pointing at the current
upstream recipe.  Fill in once the lab environment pins a specific SI
version.
"""
from __future__ import annotations

from .._base import Corpus


def load(dataset: str, item_name: str) -> Corpus:
    """Load one SpikeForest item.

    Parameters
    ----------
    dataset : one of 'PAIRED_CRCNS_HC1', 'SYNTH_MEAREC_TETRODE', ...
    item_name : the item identifier within the dataset

    Returns
    -------
    Corpus with the recording and the paired/synthetic ground truth
    """
    raise NotImplementedError(
        "SpikeForest loader stub.  To implement:\n"
        "  1. `pip install spikeinterface-extractors[spikeforest]` (or follow\n"
        "     https://spikeinterface.readthedocs.io/en/stable/modules/extractors.html\n"
        "     for the current recipe — the API has moved across releases)\n"
        "  2. Use `si.read_<format>()` for the recording and the paired-ground-truth\n"
        "     reader for the labels\n"
        "  3. Wrap into the harness's SortedResult + Corpus\n"
        f"Requested: dataset={dataset!r} item={item_name!r}"
    )
