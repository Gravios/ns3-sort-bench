"""`ns3sb list` — enumerate adapters, scorers, contracts, corpora,
SpikeForest study-sets / datasets, and recent runs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .._util import eprint, import_registries, table


# ─── list adapters ───────────────────────────────────────────────────────────
def list_adapters(args) -> int:
    import_registries()
    from ... import adapter_registry
    rows = []
    for name in sorted(adapter_registry):
        a = adapter_registry[name]
        ok, reason = a.available()
        state = "available" if ok else f"disabled: {reason}"
        if args.available_only and not ok:
            continue
        rows.append((name, a.version, state))
    print(table(rows, headers=("adapter", "version", "state")))
    print(f"\n{len(rows)} / {len(adapter_registry)} adapter(s)")
    return 0


# ─── list scorers ────────────────────────────────────────────────────────────
def list_scorers(args) -> int:
    import_registries()
    from ... import scorer_registry
    rows = []
    for name in sorted(scorer_registry):
        s = scorer_registry[name]
        rows.append((name, getattr(s, "__class__", type(s)).__name__))
    print(table(rows, headers=("scorer", "class")))
    print(f"\n{len(rows)} scorer(s)")
    return 0


# ─── list contracts ──────────────────────────────────────────────────────────
def list_contracts(args) -> int:
    import_registries()
    from ... import adapter_registry
    names = [args.adapter] if args.adapter else sorted(adapter_registry)
    missing = [n for n in names if n not in adapter_registry]
    if missing:
        eprint(f"unknown adapter(s): {missing}")
        return 1
    for name in names:
        a = adapter_registry[name]
        c = a.contract
        print(f"\n{name}")
        if c is None:
            print("  (no contract declared)")
            continue
        print(f"  required_yaml:  {list(c.yaml_fields_required)}")
        print(f"  optional_yaml:  {list(c.yaml_fields_optional)}")
        print(f"  signal_space:   {c.recording.signal_space}")
        print(f"  min_channels:   {c.recording.min_channels}")
        print(f"  drift_aware:    {c.capabilities.drift_aware}")
        print(f"  overlap_mode:   {c.capabilities.overlap_resolution}")
        print(f"  gpu_required:   {c.resources.gpu_required}")
        print(f"  gpu_vram_mb:    {c.resources.gpu_vram_mb}")
    return 0


# ─── list corpora (the resolver's known prefixes) ────────────────────────────
def list_corpora(args) -> int:
    """Enumerate corpus-key prefixes the sorter_grid runner understands."""
    print("Known corpus-key prefixes (pass to --corpus / `ns3sb load`):\n")
    print("  propper_sim                 Propper 2015 overlap simulator (default size)")
    print("  propper_sim_tiny            same, 1 s / 2 units")
    print("  propper_sim_medium          same, 30 s / 5 units / 25 % overlap")
    print()
    print("  spikeforest:<study>:<rec>   any SpikeForest item by full id")
    print()
    print("  lab:<yaml>[:<group>[:<variant>:<tag>]]")
    print("                              your own lab session + curated GT")
    print("                              defaults: group=7, variant=stderiv_C5_D34, tag=exp_final")
    print()
    print("  paired_crcns_hc1:<item>     Buzsaki rat hippocampus (tetrode + patch)")
    print("  paired_kampff:<item>        Kampff mouse cortex (32 ch + patch)")
    print("  paired_english:<item>       English awake mouse (juxtacellular)")
    print("  paired_boyden:<item>        Boyden mouse cortex (patch)")
    print("  paired_mea64c_yger:<item>   Yger mouse retina (MEA + patch)")
    print("  paired_monotrode:<item>     subset above, single channel")
    print()
    print("For paired recordings, discover <item> names with:")
    print("    ns3sb list datasets --study-set PAIRED_CRCNS_HC1")
    return 0


# ─── list study-sets (SpikeForest) ───────────────────────────────────────────
def list_study_sets(args) -> int:
    try:
        from ...corpus.spikeforest import PAIRED_STUDY_SETS, SYNTH_STUDY_SETS
    except ImportError as e:
        eprint("SpikeForest study-set metadata is not available.")
        eprint(f"  underlying error: {e}")
        eprint("  Likely cause: the paired-recordings loader "
               "(src/ns3_sort_bench/corpus/spikeforest.py) is still a stub on this branch.")
        return 1
    print("Paired (ground truth = simultaneous intracellular/juxtacellular cell):\n")
    rows = [(k, v["species"], v["region"], v["gt"]) for k, v in PAIRED_STUDY_SETS.items()]
    print(table(rows, headers=("study_set", "species", "region", "ground_truth")))
    print("\nSynthetic (ground truth = known injected spikes):\n")
    for s in sorted(SYNTH_STUDY_SETS):
        print(f"  {s}")
    return 0


# ─── list datasets (items in a SpikeForest study-set) ────────────────────────
def list_datasets(args) -> int:
    try:
        from ...corpus.spikeforest import list_recordings
    except ImportError as e:
        eprint(f"could not import ns3_sort_bench.corpus.spikeforest: {e}")
        return 1
    try:
        rows = list_recordings(study_set=args.study_set,
                               study_name=args.study_name)
    except ImportError as e:
        eprint(f"upstream dep missing: {e}")
        eprint("  pip install 'ns3-sort-bench[spikeforest]'")
        return 2
    if not rows:
        eprint(f"(no recordings for study_set={args.study_set!r}, study_name={args.study_name!r})")
        return 1
    cols = ("study_set_name", "study_name", "recording_name",
            "sampling_frequency", "num_channels", "duration_sec", "num_true_units")
    print(table([[r[c] for c in cols] for r in rows], headers=cols))
    print(f"\n{len(rows)} recording(s)")
    return 0


# ─── list runs (results under an output dir) ─────────────────────────────────
def list_runs(args) -> int:
    root = Path(args.dir)
    if not root.exists():
        eprint(f"no such directory: {root}")
        return 1
    rows = []
    for p in sorted(root.rglob("scores.csv")):
        try:
            size_kb = p.stat().st_size / 1024.0
            mtime = p.stat().st_mtime
        except OSError:
            continue
        rel = p.parent.relative_to(root)
        rows.append((str(rel) or ".",
                     f"{size_kb:7.1f} KB",
                     time_ago(mtime)))
    if not rows:
        eprint(f"no scores.csv under {root}")
        return 1
    print(table(rows, headers=("run_dir", "scores.csv", "age")))
    print(f"\n{len(rows)} run(s)")
    return 0


def time_ago(ts: float) -> str:
    import time
    delta = time.time() - ts
    for unit, s in (("d", 86400), ("h", 3600), ("m", 60)):
        if delta >= s:
            return f"{int(delta // s)}{unit} ago"
    return f"{int(delta)}s ago"


# ─── argparse wiring ─────────────────────────────────────────────────────────
def register(subparsers) -> None:
    ap = subparsers.add_parser("list", help="list adapters / scorers / corpora / datasets / runs")
    sub = ap.add_subparsers(dest="list_what", required=True)

    a = sub.add_parser("adapters", help="installed sorter adapters")
    a.add_argument("--available-only", action="store_true")
    a.set_defaults(fn=list_adapters)

    sub.add_parser("scorers", help="registered scorers").set_defaults(fn=list_scorers)

    c = sub.add_parser("contracts", help="each adapter's declared Contract")
    c.add_argument("adapter", nargs="?", default=None,
                   help="one adapter name, or omit for all")
    c.set_defaults(fn=list_contracts)

    sub.add_parser("corpora", help="corpus-key prefixes the runner understands"
                   ).set_defaults(fn=list_corpora)

    sub.add_parser("study-sets", help="SpikeForest paired / synthetic study-sets"
                   ).set_defaults(fn=list_study_sets)

    d = sub.add_parser("datasets", help="SpikeForest items in a study-set")
    d.add_argument("--study-set", required=True,
                   help="e.g. PAIRED_CRCNS_HC1, PAIRED_KAMPFF, SYNTH_MEAREC_TETRODE")
    d.add_argument("--study-name", default=None,
                   help="narrow to one study_name within the study-set")
    d.set_defaults(fn=list_datasets)

    r = sub.add_parser("runs", help="scored benchmark runs under a directory")
    r.add_argument("dir", nargs="?", default="runs")
    r.set_defaults(fn=list_runs)
