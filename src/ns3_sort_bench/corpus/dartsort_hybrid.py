"""DARTsort hybrid-recording corpus (Boussard et al. 2023 §Fig 5).

The DARTsort authors describe but do NOT release their three hybrid
recordings: templates from earlier IBL sorting runs injected into raw IBL
recording snippets.  This loader REGENERATES one, matching the paper's
methodology, from a cached IBL recording + a cached DARTsort sort.

Scaffold.  Full implementation needs:
    1. an IBL recording pulled through ONE API
       (`pip install ONE-api`; see
       https://int-brain-lab.github.io/iblenv/notebooks_external/data_release_brainwidemap.html)
    2. a DARTsort output on that recording (`fiber_kit.spikeinterface`
       extractors + the DARTsort adapter)
    3. template injection: pick N random templates, pick N random injection
       times (respecting template ground-truth refractory), add template
       waveforms into the raw signal.  DARTsort's `dartsort.util.hybrid`
       module provides a reference implementation.
"""
from __future__ import annotations

from .._base import Corpus


def load(ibl_recording_eid: str, *, n_injected_units: int = 10,
         seed: int = 0) -> Corpus:
    raise NotImplementedError(
        "DARTsort hybrid corpus loader stub.  To implement:\n"
        "  1. Pull the IBL recording via ONE API:\n"
        "       from one.api import ONE; one = ONE(); ...\n"
        "  2. Sort it with DARTsort (via this harness's dartsort adapter)\n"
        "  3. Use DARTsort's own `dartsort.util.hybrid.inject_units()` to\n"
        "     inject n_injected_units templates at ground-truth times\n"
        "  4. Return a Corpus whose ground_truth carries the injected\n"
        "     spike times (not DARTsort's full sort)\n"
        f"Requested: eid={ibl_recording_eid!r} n_injected={n_injected_units} seed={seed}"
    )
