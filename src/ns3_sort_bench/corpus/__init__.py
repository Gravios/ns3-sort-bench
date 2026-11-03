"""Corpus loaders.

Each corpus module exports a `load()` function returning a `Corpus`.  The
runner's `_resolve_corpus` dispatches by name; add a new corpus by writing
a `<name>.py` here with a `load()` function and wiring it into
`runners/sorter_grid.py::_resolve_corpus`.
"""
from .lab_sessions import load as load_lab_session
from .spikeforest import load as load_spikeforest
from .spikeforest import list_recordings as list_spikeforest
from .spikeforest import iter_study_set as iter_spikeforest_study_set
from .dartsort_hybrid import load as load_dartsort_hybrid
from .paired import (
    load_paired_crcns_hc1,
    load_paired_kampff,
    load_paired_english,
    load_paired_boyden,
    load_paired_mea64c_yger,
    load_paired_monotrode,
)

__all__ = [
    "load_lab_session",
    "load_spikeforest",
    "list_spikeforest",
    "iter_spikeforest_study_set",
    "load_dartsort_hybrid",
    "load_paired_crcns_hc1",
    "load_paired_kampff",
    "load_paired_english",
    "load_paired_boyden",
    "load_paired_mea64c_yger",
    "load_paired_monotrode",
]
