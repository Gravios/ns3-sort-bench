# `ns3sb` — command-line reference

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
```

Flag precedence: CLI flags > plan > defaults.  CLI flags never mutate
the plan — add to the plan with `load` / `set` for persistence.

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
