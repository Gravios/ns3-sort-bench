"""Load Sirota reference sessions (g5 / g6 / g7 of jg05-20120316 and
sirotaA-jg-000005-20120316) through the NS3 SpikeInterface extractors.

Depends on fiber_kit (for the NS3 extractors + variant/tag resolution).
"""
from __future__ import annotations

import os

from .._base import Corpus, SortedResult


def load(session_yaml: str, *, electrode_group: int = 7,
         variant: str = "stderiv_C5_D34", tag: str = "exp_final",
         name: str | None = None) -> Corpus:
    """Load one lab session + its curated ground truth as a Corpus.

    Parameters
    ----------
    session_yaml : path to the lab's session yaml.
    electrode_group : 1-indexed group (lab convention).  Defaults to g7
        which is the pyramidal-focus reference for the g7 work in
        `claude/g7-*` docs.
    variant / tag : the ground-truth sort to use as reference.  Default
        matches the lab's `exp_final` curation in the stderiv_C5_D34 space.
    name : short identifier for the dashboard; derived from the session +
        group when omitted.
    """
    # ns3-sort-bench ships its own extractors — fiber_kit is only needed
    # for variant/tag-aware path resolution (optional).
    from ..ns3.extractors import NS3RecordingExtractor, NS3SortingExtractor
    from ..ns3.session import load_session_yaml

    sess = load_session_yaml(session_yaml)
    chans = sess.spike_groups[electrode_group - 1]
    rec = NS3RecordingExtractor(session_yaml, channels=chans, raw="fil",
                                 name=f"ns3-g{electrode_group}")
    gt_sort = NS3SortingExtractor(session_yaml, electrode_group=electrode_group,
                                   variant=variant, tag=tag)

    # Turn the SI sorting into a SortedResult the scorer wants
    import numpy as np
    spike_times = {int(u): np.asarray(gt_sort.get_unit_spike_train(u), np.int64)
                   for u in gt_sort.get_unit_ids()}
    gt = SortedResult(
        unit_ids=sorted(spike_times), spike_times=spike_times,
        sampling_rate=sess.sampling_rate,
        adapter="ground_truth", corpus_item=session_yaml,
        adapter_version=f"{variant}/{tag}",
        channels=list(chans),
        extras=dict(variant=variant, tag=tag, electrode_group=electrode_group),
    )

    cname = name or f"{os.path.splitext(os.path.basename(session_yaml))[0]}_g{electrode_group}"
    rec.annotate(name=cname)

    # Load the parsed yaml dict so adapter contracts can validate against it.
    from ..ns3.session import load_session_dict
    yaml_dict = load_session_dict(session_yaml)

    return Corpus(
        name=cname, recording=rec, ground_truth=gt,
        description=f"lab session {cname} ({variant}/{tag})",
        kind="lab", ns3_yaml=yaml_dict,
        extras=dict(session_yaml=session_yaml, variant=variant, tag=tag,
                    electrode_group=electrode_group),
    )
