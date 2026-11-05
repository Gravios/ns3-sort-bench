<h1><picture>
  <source media="(max-width: 600px)" srcset="images/doc-paired-recordings-banner-narrow.png">
  <img src="images/doc-paired-recordings-banner.png" width="100%" alt="Running on simultaneous whole-cell / paired ground-truth">
</picture></h1>

The SpikeForest corpus includes six **paired recording** study sets where
every extracellular trace comes with a simultaneously-recorded intracellular
or juxtacellular reference cell.  Those paired spike trains are the gold
standard for benchmarking a sorter: every extracellular unit the sorter
reports is compared against the known, un-ambiguous intracellular spikes of
the reference cell.

| Study set | Species | Region | Ground truth | Ch | Recs |
|-----------|---------|--------|--------------|----|------|
| `PAIRED_CRCNS_HC1` | rat | hippocampus | intracellular | 4-6 | 93 |
| `PAIRED_KAMPFF` | mouse | cortex | whole-cell patch | 32 | 15 |
| `PAIRED_ENGLISH` | awake mouse | cortex/hc/thalamus | juxtacellular | 4-32 | 29 |
| `PAIRED_BOYDEN` | mouse | cortex | whole-cell patch | 32 | 19 |
| `PAIRED_MEA64C_YGER` | mouse | retina | whole-cell patch | 64 | 18 |
| `PAIRED_MONOTRODE` | various | various | subset of the above | 1 | 100 |

For `neurosuite-3` / `fiber-kit` work on rat CA1, `PAIRED_CRCNS_HC1` is the
direct anatomical match (same Buzsaki-lab tetrode geometry).  For headline
numbers that the field reports, `PAIRED_KAMPFF` is the convention.

## Install

```sh
pip install 'ns3-sort-bench[spikeforest,all]'
# first-time kachery-cloud auth (follow the prompt):
kachery-cloud-init
```

## List what's available

```sh
# All CRCNS HC1 (Buzsaki rat hippocampus) items
python -m ns3_sort_bench.corpus.spikeforest list --study-set PAIRED_CRCNS_HC1

# All Kampff mouse cortex items
python -m ns3_sort_bench.corpus.spikeforest list --study-set PAIRED_KAMPFF
```

Columns: `study_set_name study_name recording_name sampling_frequency num_channels duration_sec num_true_units`.

The last column — `num_true_units` — is 1 for paired recordings (the one
patched / juxtacellular cell).

## Minimal run (programmatic)

```python
from ns3_sort_bench import BenchmarkRun, adapter_registry, scorer_registry
from ns3_sort_bench.corpus.paired import load_paired_crcns_hc1

bench = BenchmarkRun(output_dir="runs/hc1_d533101")
bench.add_corpus(load_paired_crcns_hc1("d533101"))

# Pick your adapters — these are the ones installed by [all] + [fiber-kit]
for a in ("botm_ref", "mountainsort5", "tridesclous2",
          "kilosort4", "dartsort", "fiber_kit"):
    if a in adapter_registry:
        bench.add_adapter(adapter_registry[a])

# For paired data the per-cell scorer is the one that matters: it gives you
# the best-matching unit's accuracy against the single patched reference cell.
bench.add_scorer(scorer_registry["pairwise"])
bench.add_scorer(scorer_registry["per_cell"])

df = bench.go()
print(df.pivot_table("value", ["corpus_item", "candidate_adapter"], "metric"))
```

Output lands in `runs/hc1_d533101/scores.csv` and `sort_summaries.json`.

## Minimal run (CLI)

```sh
ns3-sort-bench \
    --corpus paired_crcns_hc1:d533101 \
    --adapters botm_ref,mountainsort5,tridesclous2,kilosort4,dartsort,fiber_kit \
    --scorers pairwise,per_cell \
    --out runs/hc1_d533101
```

The `paired_<lab>:<item>` prefix is resolved by
`runners.sorter_grid._resolve_corpus`; any prefix the harness knows about is
documented in the error message when you give it an unknown one.

## Sweep an entire study set

```python
from ns3_sort_bench import BenchmarkRun, adapter_registry, scorer_registry
from ns3_sort_bench.corpus.spikeforest import iter_study_set

bench = BenchmarkRun(output_dir="runs/kampff_sweep")
for c in iter_study_set("PAIRED_KAMPFF"):
    bench.add_corpus(c)

for a in ("botm_ref", "fiber_kit", "kilosort4"):
    bench.add_adapter(adapter_registry[a])
bench.add_scorer(scorer_registry["per_cell"])
bench.go()
```

## What "accuracy" means for paired data

For paired recordings SpikeForest's convention — which the `per_cell` scorer
mirrors — is:

1. Find the sorter's unit whose spike times line up best with the single
   reference cell's intracellular spikes (within ±0.4 ms by default:
   `delta_frames=12` at 30 kHz).
2. Report that unit's precision, recall, and F1 (= accuracy) against the
   reference train.

The scorer's `delta_frames` argument is tunable per corpus; raise it to 24
(≈0.8 ms) for lower sampling rates or if your pipeline's spike-time
convention differs by more than half a sample from SpikeForest's.

## First-time kachery notes

- `pip install 'ns3-sort-bench[spikeforest]'` pulls `kachery_cloud` as a dep.
- Run `kachery-cloud-init` once and follow the console link to associate
  this workstation with a free kachery-cloud account.
- Each paired recording is tens of MB; the full index is ~1 MB.
- Cache lives under `~/.kachery-cloud/`; set `KACHERY_CLOUD_DIR` to redirect.
- Not every SpikeForest recording was migrated to the current kachery
  backing store; the index call will say if a specific item is unavailable.
