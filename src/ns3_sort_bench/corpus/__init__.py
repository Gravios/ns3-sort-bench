"""Corpus loaders.

Each corpus module exports a `load()` function returning a `Corpus`.  The
runner's `_resolve_corpus` dispatches by name; add a new corpus by writing
a `<name>.py` here with a `load()` function and wiring it into
`runners/sorter_grid.py::_resolve_corpus`.
"""
from .lab_sessions import load as load_lab_session
from .spikeforest import load as load_spikeforest
from .dartsort_hybrid import load as load_dartsort_hybrid

__all__ = ["load_lab_session", "load_spikeforest", "load_dartsort_hybrid"]
