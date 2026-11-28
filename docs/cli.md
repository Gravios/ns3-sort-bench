<h1><picture>
  <source media="(max-width: 600px)" srcset="images/doc-cli-banner-narrow.png">
  <img src="images/doc-cli-banner.png" width="100%" alt="ns3sb — command-line reference">
</picture></h1>

Primary entry point for ns3-sort-bench.  Short, muscle-memory-friendly,
verb-noun subcommands.  The old `ns3-sort-bench` command still works
and forwards to the same `main()`.

## Verbs at a glance

| verb         | what it does |
|--------------|--------------|
| `list`       | enumerate adapters, scorers, contracts, corpora, SpikeForest study-sets / datasets, past runs |
| `load`       | add corpus key(s) to the active plan |
| `unload`     | drop corpus key(s) from the plan |
| `set`        | configure adapters / scorers / output / yaml / delta-frames |
| `show`       | print the active plan |
| `reset`      | delete the plan file |
| `inspect`    | load one corpus and summarise it (channels, duration, GT) |
| `contract`   | print one adapter's Contract |
| `run`        | execute the plan (or a one-shot via flags) |
| `fk-sweep`   | run fiber_kit over a set of FK_* values; table of fc_session / final scores |
| `summary`    | corpus × adapter × metric pivot for one scored run |
| `compare`    | diff two scored runs (B − A); regressions and improvements separately |
| `lattice`    | fiber-kit lattice ablation (passes through to `runners.lattice`) |
| `render`     | render a run directory as HTML dashboard |
| `doctor`     | environment / adapter readiness check |

Global flags: `--version`, `--plan-path`.

## The active plan

`load` / `set` persist to

    ${XDG_CONFIG_HOME:-~/.config}/ns3-sort-bench/plan.yaml

overridable with `NS3_SORT_BENCH_PLAN=/path/to/plan.yaml`.  Fields:

```yaml
corpora:    [paired_crcns_hc1:d533101, propper_sim_tiny]
adapters:   [botm_ref, fiber_kit, mountainsort5, kilosort4]
scorers:    [pairwise, per_cell]
output_dir: runs/2026-11-03
ns3_yaml:   null         # path, or null
delta_frames: null       # int or null (pairwise match tolerance)
```

Hand-edit it, delete it, or clear with `ns3sb reset`.

## Discovery

```sh
# What adapters are installed, and which are available right now?
ns3sb list adapters
ns3sb list adapters --available-only

# What scorers are registered?
ns3sb list scorers

# What are the known corpus-key prefixes?
ns3sb list corpora

# What SpikeForest study-sets exist (paired / synthetic)?
ns3sb list study-sets

# What's actually in PAIRED_CRCNS_HC1?
ns3sb list datasets --study-set PAIRED_CRCNS_HC1

# What runs have I done under runs/?
ns3sb list runs runs/
```

## Build a plan

```sh
ns3sb load paired_crcns_hc1:d533101 propper_sim_tiny
ns3sb set adapters botm_ref,fiber_kit,mountainsort5,kilosort4,dartsort
ns3sb set scorers pairwise,per_cell
ns3sb set output runs/2026-11-03

# verify before launching
ns3sb show
```

## Run

```sh
# use the plan
ns3sb run

# dry-run: print the resolved plan, don't execute
ns3sb run --dry-run

# one-shot (flags override the plan but don't mutate it)
ns3sb run --corpus paired_kampff:<item> --adapters fiber_kit,botm_ref

# quiet mode (no per-step progress)
ns3sb run --quiet

# resume: re-run into the same --out, skipping what already sorted
ns3sb run --out runs/0312-g8 --adapters fiber_kit,kilosort4,dartsort --skip-done
```

`scores.csv` and `sort_summaries.json` are rewritten after every adapter, so
a crash or a kill keeps what finished.  With `--skip-done`, an
(item, adapter) pair that sorted without error in `--out` is not sorted again:
its earlier scores (fiber_kit's `@fc_*` stages included) are kept.  Failed and
new pairs are sorted.  Results for adapters left out of this run's
`--adapters` stay in the files too.  A kept score is the one computed back
then, so after a scorer fix re-run without `--skip-done`.

`--export-klusters` (off unless given, or set in the plan) writes each
sorter's Klusters session to `<out>/klusters_<sorter>/<item>/` as soon as that
sorter finishes, fiber_kit's stages to `klusters_fiber_kit_fc_*`, and the
reference to `klusters_ground_truth`.  Sorts are not saved otherwise, so with
both flags a finished pair that has no Klusters session yet is sorted again
to make one.

Flag precedence: CLI flags > plan > defaults.  CLI flags never mutate
the plan — add to the plan with `load` / `set` for persistence.

## Sweep fiber-kit settings

```sh
# list the runs first: the default sweeps fiber-session knobs one at a time
ns3sb fk-sweep --corpus paired_crcns_hc1:d11221_d11221.001 \
               --out runs/fk_sweep --dry-run

# your own keys and values; --grid runs every combination instead
ns3sb fk-sweep --corpus paired_crcns_hc1:d11221_d11221.001 --out runs/fk_sweep \
               --set FK_SESSION_RKK_PENALTY_MIX=0,0.5,1 --set FK_SESSION_DIPSPLIT=0,1 --grid
```

Each setting runs `ns3sb run --adapters fiber_kit --scorers per_cell` in
`<out>/<setting>/`, in a fresh temporary folder, with `$FK_CONFIG` set to
`<out>/<setting>/fk-tuning.yaml`: the base tuning (`--base-config`, default
`$FK_CONFIG`) plus that setting.  A first `baseline` run uses the base
tuning alone.  The table (also `<out>/sweep.csv`) gives cluster count, IoU,
recall and precision at fc_session (fiber-session's sort) and at the final stage.  Finished
settings are reused when you run the same sweep again; `--rerun` forces them.

## Inspection

```sh
# summarise a corpus without running anything
ns3sb inspect paired_crcns_hc1:d533101

# dump one adapter's Contract (text, markdown, or YAML)
ns3sb contract fiber_kit
ns3sb contract fiber_kit --format markdown
ns3sb contract fiber_kit --format yaml

# all adapters' Contracts at once
ns3sb list contracts
```

## Environment check

```sh
ns3sb doctor
```

Reports Python / core deps / optional sorter packages / CUDA / per-adapter
import readiness.  Exit code is 0 whether or not some adapters are
unavailable — it's an informational report, not a gate.

## Review past runs

```sh
# A pivot table for one run (corpus × adapter rows, metric columns)
ns3sb summary runs/2026-11-03
ns3sb summary runs/2026-11-03 --metric accuracy,f1,precision,recall
ns3sb summary runs/2026-11-03 --format csv > review.csv

# Diff two runs: value_b - value_a, biggest absolute movers first
ns3sb compare runs/baseline runs/candidate
ns3sb compare runs/baseline runs/candidate --only-regressed
ns3sb compare runs/baseline runs/candidate --only-improved
ns3sb compare runs/baseline runs/candidate --metric accuracy --adapter fiber_kit
```

`compare` joins on `(corpus_item, candidate_adapter, scorer, metric)`.
Rows present in only one run keep that side's value with the other as `—`.

## Your own data (lab sessions)

```sh
# Default group / variant / tag (g7 pyramidal, stderiv_C5_D34, exp_final)
ns3sb load lab:/data/g7_session.yaml
# All four spelled out
ns3sb load lab:/data/g6_0316.yaml:6:stderiv_C5_D34:exp_final
```

The curated sort named by `<variant>:<tag>` serves as the reference
SortedResult: `<base>.clu.<variant>.<group>.<tag>`, with the stage's own
`.res` if it has one, else the group's shared `.res` under whatever token it
was detected with (e.g. `.res.stderiv_S3.<group>`).  Every registered adapter
is run on the group's channels and scored against it.

Nothing is written beside the session.  The SpikeInterface sorters read the
group's channels into memory; the fiber_kit adapters sort a scratch copy
(`<out>/fiber_kit/<item>/` with `--keep-fk-artifacts`, else a temporary folder
removed after the sort):

- copied: the session yaml and, for the group, every untagged
  `.res` / `.spk` / `.fet` / `.pca` / `.nst` of each waveform space (no
  super-sampled `_U` copies, no stage-tagged files, so no earlier `fc_*` stage
  or curation is read or overwritten);
- symlinked: the `.fil` / `.dat`, and the probe files the yaml names relative
  to its folder.

The plan runs in the session's own method, read off its `.spk` the way
fiber-pipeline does (`.spk.stderiv.<g>`, else the one `stderiv_*` token), and
the readback is that method's `fc_unmix`.  The bundled tuning is the
standard-method one, so give the session's own:

```sh
ns3sb run --corpus lab:/data/0312/sirotaA-jg-000005-20120312.yaml:8:stderiv_C5:cur_pl \
          --adapters fiber_kit --fk-config /data/0312/fiber-kit-exp-0312.yaml \
          --keep-fk-artifacts --out runs/0312-g8
# --fk-method stderiv_C5   when the session has more than one stderiv .spk
```

A `--fk-config` with a `pipeline:` of its own replaces the bundled plan; if
that plan does not end in `fc_unmix`, the readback will not find it.

## Dashboard

```sh
ns3sb render runs/2026-11-03
```

Reads `scores.csv` and `sort_summaries.json`, writes an HTML dashboard
alongside them.

## Lattice ablation

```sh
ns3sb lattice --help
ns3sb lattice --corpus propper_sim_medium --out runs/lattice/2026-11-03
```

All flags after `lattice` are forwarded to `runners.lattice.main`
untouched.
