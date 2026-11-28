<h1><picture>
  <source media="(max-width: 600px)" srcset="docs/images/doc-install-banner-narrow.png">
  <img src="docs/images/doc-install-banner.png" width="100%" alt="ns3-sort-bench — installation">
</picture></h1>

## TL;DR

ns3-sort-bench is not on PyPI: run these in a clone of the repository.
Keep the quotes round an extra; zsh reads `[...]` as a glob otherwise.

```sh
# Core only (BOTM reference + simulator + harness; ~10 packages, no GPU)
pip install -e .

# CPU-only sorter bundle (no torch, no CUDA wheels; ~400 MB)
pip install -e '.[all-cpu]'

# Everything, SLAy included (~150 packages, ~3 GB with CUDA 13 wheels)
pip install -e '.[all]'

# Developing the harness itself — [all] + ruff; same packages as [all]
# plus the linter.
pip install -e '.[dev]'

# Reproducible — install the exact resolved environment the smoke tests pass against
pip install -r requirements.lock
pip install -e .
```

## What `[all]` includes

Resolved against **Python 3.13.18, linux-x86_64, 2026-11-02** (see
`requirements.lock` for the full 148-package manifest).  The table
shows the versions in that lockfile.  `pyproject.toml` itself declares
ranges for the core (numpy, scipy and pyyaml with fiber-kit's floors), so
`pip install -e .` into an environment that already has
fiber-kit keeps the numpy / scipy / pandas installed there; only the
sorters and SpikeInterface's minor series are held fixed.  Use the
lockfile in a fresh environment when the numbers must be reproducible.

| Group | Packages |
|---|---|
| Core harness | numpy 2.5.4, pandas 3.1.0, pyyaml 6.0.3, scipy 1.18.1, spikeinterface 0.105.4, probeinterface 0.4.2, neo 0.14.6, tqdm 4.70.1 |
| Sorters | kilosort 4.1.7, dartsort 0.5.31, mountainsort5 0.5.9, tridesclous 1.6.9 |
| GPU stack (torch + faiss pull) | torch 2.15.0, triton 3.9.0, faiss-cpu 1.15.1, nvidia-cublas / cudnn / cufft / cusolver / cusparse / nccl / nvtx (CUDA 13 wheels) |
| Corpora | spikeforest 0.12.9 (+ kachery 2.0.3, sortingview 0.14.2), one-api 3.6.2 (IBL) |
| fiber-kit | `git+https://github.com/Gravios/fiber-kit.git@<SHA>` (lockfile pins exact commit) |
| Acceleration | numba 0.68.1 (SpikeInterface's spike-matching comparison fast path) |
| Test runner | pytest 9.2.0, pytest-cov 7.1.0 |

Total installed size with CUDA wheels: **~3 GB**.  Install `[all-cpu]`
on GPU-less workstations to skip kilosort + dartsort + the ~2.5 GB of
CUDA (and still get pytest for the self-check).

## Run the test suite after installing

The `[test]` extra ships pytest + pytest-cov, and is pulled in by both
`[all]` and `[all-cpu]`, so a full install can validate itself without
a second `pip install` step:

```sh
pytest tests/ -q                                    # 20-ish hermetic cases
pytest tests/ --cov=ns3_sort_bench --cov-report=term-missing  # with coverage
```

For a core-only install (`pip install -e .`), add the test
extra explicitly:

```sh
pip install -e '.[test]'
```

`[dev]` is the kitchen-sink developer install: it references `[all]`
(which already includes `[test]`) and adds ruff on top, so one
command gives you the full sorter stack, corpora loaders, test
runner, and linter:

```sh
pip install -e '.[dev]'
```

Pick `[test]` instead when you want the test runner but only a chosen
subset of sorter extras — e.g. `pip install -e '.[fiber-kit,test]'`.

## SLAy

SLAy comes from the [Gravios/SLAy](https://github.com/Gravios/SLAy) fork,
which runs on Python 3.11-3.13 and SpikeInterface 0.105 (upstream pins
Python 3.11 and does not import on 0.105).  It installs into the same
environment as everything else and is part of `[all]`; the extra brings
Kilosort 4, which SLAy post-processes:

```sh
pip install -e '.[slay]'
```

`requirements.lock` pins the fork at `aec305d` (the eight Python 3.13 / fix
patches on upstream 3895c3a).

## Lockfile reproducibility

`requirements.lock` is the `uv pip compile` output of 2026-11-02, with
fiber-kit and SLAy pinned to commits since.  Install the pinned environment
byte-for-byte:

```sh
pip install -r requirements.lock
pip install -e .                        # install ns3-sort-bench itself
pytest                                   # the whole suite should pass
```

Regenerate after a deliberate dep bump, keeping every other pin (the lock is
read as the preference) and moving only the package named:

```sh
uv pip compile --python-version 3.13 --extra all pyproject.toml \
    --output-file requirements.lock --no-sources --upgrade-package <name>
```

`--no-sources` is needed since SLAy is in `[all]`: its `pyproject.toml` sends uv
to the pytorch.org CUDA index for torch, which is not where this lock takes torch
from (PyPI).  A git dependency is not held by the existing lock: uv moves
`fiber-kit` and `slay` to their main unless they are pinned, e.g. with
`--override <file>` holding `fiber-kit @ git+https://github.com/Gravios/fiber-kit.git@<sha>`.

CI should fail any `pyproject.toml` change whose `requirements.lock`
wasn't regenerated in the same commit — otherwise the lockfile silently
drifts from what the ranges resolve to.

## Using the GPU and the cores

Check the GPU is seen by the torch the sorters will use:

```sh
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

With it, Kilosort 4 and SLAy run on CUDA, DARTsort gets 4 GPU workers, and
SpyKING CIRCUS 2's template matching runs on CUDA (torch engine).  Without
it they fall back to the CPU.  `ns3sb run --n-jobs` (default one per
physical core) sets the CPU worker processes; `ns3sb` caps
`OPENBLAS_NUM_THREADS` at 32 unless you export it.  See
[docs/cli.md → Cores and GPU](docs/cli.md#cores-and-gpu---n-jobs).

## Per-sorter install notes

### DARTsort (GPU, torch >= 2.2, Python >= 3.12)

Upstream metadata takes care of both constraints.  CUDA 12+ wheels are
pulled automatically.  Resource contract: `gpu_vram_mb >= 12288` —
check with `ns3sb contract dartsort`.

### Kilosort 4

GPU-strongly-recommended (CPU fallback exists but is impractical).
Pulls faiss-cpu 1.15.1 for the splits step.  Default `do_CAR=True`
subtracts spikes on sub-10-channel geometries (see
`claude/spikeinterface-g6-comparison.md`) — the Kilosort 4 adapter here
disables CAR by default; pass `run_kwargs={"do_CAR": True}` to override.

### SpikeForest corpus

The kachery storage layer needs network access to `kachery-cloud`
servers.  First corpus pull may take several minutes while the client
authenticates.

### IBL ONE API

Needs an IBL account.  Initialise ONE before the first corpus pull:

```sh
pip install -e '.[ibl]'
python -c "from one.api import ONE; ONE.setup()"
```

### fiber-kit (not on PyPI)

Installed from git via the `[fiber-kit]` extra.  The lockfile pins the
exact SHA validated by the smoke tests; override with:

```sh
pip install 'fiber-kit @ git+https://github.com/Gravios/fiber-kit.git@<branch-or-sha>'
```

## Minimum install for the Pröpper simulator + BOTM reference

```sh
pip install -e .
```

Core deps only; no torch, no GPU, no git clones.  Enough to run every
test in `tests/`, the Pröpper simulator corpus, the BOTM reference
adapter, and all three scorers.  First install footprint under 400 MB.
