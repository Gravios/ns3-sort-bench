"""`ns3sb` — the ns3-sort-bench command-line interface.

Verb-noun layout:

    ns3sb list       {adapters,scorers,contracts,corpora,study-sets,datasets,runs}
    ns3sb load       <corpus> [...]           # add corpus to the active plan
    ns3sb unload     <corpus>                 # drop corpus from the active plan
    ns3sb set        {adapters,scorers,output,yaml,delta-frames}  <value>
    ns3sb show                                # print the active plan
    ns3sb reset                               # clear the active plan
    ns3sb inspect    <corpus>                 # load a corpus and summarise it
    ns3sb contract   <adapter>                # print one adapter's Contract
    ns3sb run        [--corpus ...] [--adapters ...]  # one-shot or from plan
    ns3sb lattice    [...]                    # delegate to runners.lattice
    ns3sb render     <run_dir>                # render HTML dashboard for a run
    ns3sb doctor                              # environment / dependency check

The "active plan" lives at `${XDG_CONFIG_HOME:-~/.config}/ns3-sort-bench/plan.yaml`
and composes across invocations, so you can set up a benchmark in a few
small commands and then run it with a bare `ns3sb run`.

Backward-compat: the old `ns3-sort-bench` entry point still works and
delegates to this same `main`.
"""
from __future__ import annotations

from .main import main

__all__ = ["main"]
