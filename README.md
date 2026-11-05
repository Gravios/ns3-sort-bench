<h1><picture>
  <source media="(max-width: 600px)" srcset="docs/images/banner-narrow.png">
  <img src="docs/images/banner.png" width="100%" alt="ns3-sort-bench">
</picture></h1>

Cross-sorter benchmark harness for **neurosuite-3 / fiber-kit** sessions.
Compares spike-sorting algorithms on a unified metric schema and shared
corpora, with per-adapter *contracts* describing exactly how each sorter
interfaces with the ns3 session yaml.

Target sorters:

- **BOTM reference** — literal Pröpper 2015 §3.4 (in-tree, no deps)
- **fiber-kit** — the subject of the comparison; three modes
- **Kilosort 4** — Pachitariu 2024
- **DARTsort** — Boussard, Varol, Paninski 2023+
- **SLAy** — Janelia 2025 (merge-step post-pass on Kilosort 4)
- **MountainSort 5**, **Tridesclous 2**, **SpykingCircus 2**
- **32 lattice-ablation configs** of fiber_deconv (OMP / SIC-A / SIC-B /
  SIC-C × integer / R=4 bank × amp-prior on/off × VAR(4) / spatial-only)

## Install

```sh
pip install ns3-sort-bench                         # core (BOTM reference + simulator)
pip install ns3-sort-bench[fiber-kit]              # + the fiber-kit chain + lattice
pip install ns3-sort-bench[kilosort4]              # + Kilosort 4 (GPU recommended)
pip install ns3-sort-bench[dartsort]               # + DARTsort (GPU required)
pip install ns3-sort-bench[slay]                   # + SLAy (post-pass on KS4)
pip install ns3-sort-bench[mountainsort5,tridesclous2]  # + SI-wrapped sorters
pip install ns3-sort-bench[spikeforest]            # + SpikeForest corpus
pip install ns3-sort-bench[ibl]                    # + IBL ONE API corpus
pip install ns3-sort-bench[all-cpu]                # everything CPU-only
```

## Pinned versions (2026-11-02)

Core deps are pinned to the exact versions the harness was validated
against.  Bumping a pin requires re-running the smoke suite — see
`pyproject.toml` for the rationale.

| Package | Version |
|---|---|
| numpy | 2.5.4 |
| pandas | 3.1.0 |
| pyyaml | 6.0.3 |
| scipy | 1.18.1 |
| spikeinterface | 0.105.4 |
| probeinterface | 0.4.2 |
| neo | 0.14.6 |
| tqdm | 4.70.1 |

Optional sorter extras pin their upstream to its latest known-good
release (Kilosort 4.1.7, MountainSort 5.0.5.9, Tridesclous 1.6.9,
DARTsort 0.5.31, SLAy from git+main).

## Contracts

Every adapter declares a formal `Contract` naming:

1. which ns3-yaml fields it reads (required and optional, as dotted paths)
2. what preconditions it imposes on the recording (channel count, dtype,
   duration, probe geometry, signal space)
3. what shape its `SortedResult` has (unit-id convention, templates
   emitted, per-group vs whole-session)
4. its capabilities (drift-aware, overlap-resolution semantics, GPU
   requirement, torch requirement, supports stderiv-transformed input)
5. its resource estimates (RAM, CPU cores, GPU VRAM, seconds per minute
   of recording)
6. known limitations
7. the exact `pyproject.toml` upstream pin it was tested against

The harness validates `(ns3_yaml, recording) ↔ contract` before every
sort; violations produce a logged skip, not a mid-sort traceback.

Browse all contracts:

```sh
ns3-sort-bench list-contracts        # one-line summary each
ns3-sort-bench dump-contracts > CONTRACTS.md    # full markdown catalog
```

See `CONTRACTS.md` for the current catalog.  It is REGENERATED from the
adapter source files; edit `src/ns3_sort_bench/adapters/*.py` and re-dump.

## Quick start

```python
from ns3_sort_bench import BenchmarkRun, adapter_registry, scorer_registry
import ns3_sort_bench.adapters, ns3_sort_bench.scorers  # populate registries
from ns3_sort_bench.simulator import make_propper_overlap_corpus

bench = BenchmarkRun(output_dir="runs/first")
bench.add_corpus(make_propper_overlap_corpus(name="sim_tiny"))
bench.add_adapter(adapter_registry["botm_ref"])
for s in ("pairwise", "per_cell", "propper_overlap"):
    bench.add_scorer(scorer_registry[s])

df = bench.go()
print(df.pivot_table("value", ["corpus_item", "metric"], "candidate_adapter"))
```

CLI:

```sh
ns3-sort-bench list-adapters
ns3-sort-bench run --corpus propper_sim_tiny --out runs/first
ns3-sort-bench lattice --corpus propper_sim_medium \
    --rules omp,sic-b --banks integer --priors none \
    --out runs/lattice-sicb-vs-omp
ns3-bench-render runs/first runs/first/dashboard.md
```

## Writing a new adapter

Three steps:

1. Create `src/ns3_sort_bench/adapters/<name>.py`
2. Declare `CONTRACT = Contract(...)` as a module-level constant
3. Subclass `SorterAdapter`:
   - `name = "<name>"`
   - `@property` `contract` returns `CONTRACT`
   - implement `sort(recording, *, ns3_yaml=None, **kwargs) -> SortedResult`
   - implement `available() -> (bool, str)` — probe for upstream package
   - register with `_register(adapter_registry, YourAdapter(), "<name>")`

See `adapters/botm_ref.py` for a reference implementation with no
third-party dependencies.

## Writing a new scorer

Subclass `Scorer`, implement `__call__(reference, candidate, **kwargs)`
returning a `ScoreResult`.  Register with `_register(scorer_registry, ...)`.

## Related

- [neurosuite-3](https://github.com/Gravios/neurosuite-3) — the parent toolchain
- [fiber-kit](https://github.com/Gravios/fiber-kit) — the sorter under study
- Pröpper, R. (2015), "New tools for electrophysiological data analysis
  and their application to a working memory study," TU Berlin dissertation
- Boussard et al. (2023), "DARTsort: A modular drift tracking spike sorter"
- Pachitariu et al. (2024), "Kilosort 4"
- Janelia (2025), "SLAy-ing oversplitting errors in high-density ephys"

## License

MIT.
