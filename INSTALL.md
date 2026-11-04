# Installation

## TL;DR

```sh
# Core only (BOTM reference + simulator + harness; ~10 packages, no GPU)
pip install ns3-sort-bench

# CPU-only sorter bundle (no torch, no CUDA wheels; ~400 MB)
pip install ns3-sort-bench[all-cpu]

# Everything except SLAy (~148 packages, ~3 GB with CUDA 13 wheels)
pip install ns3-sort-bench[all]

# Reproducible — install the exact resolved environment the smoke tests pass against
pip install -r requirements.lock
pip install -e .
```

## What `[all]` includes

Resolved against **Python 3.13.18, linux-x86_64, 2026-11-02** (see
`requirements.lock` for the full 148-package manifest).  Core
`pyproject.toml` dependencies are strict `==` pins because benchmark
reproducibility is the whole point — a point release of SpikeInterface
changing spike-matching semantics mid-benchmark would silently
invalidate stored baselines.

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

For a core-only install (`pip install ns3-sort-bench`), add the test
extra explicitly:

```sh
pip install 'ns3-sort-bench[test]'
```

Add `[dev]` instead if you want ruff for linting / formatting alongside
pytest.

## What `[all]` does NOT include — SLAy

SLAy's upstream `pyproject.toml` pins `requires-python = "~=3.11.0"`
(Python 3.11.x only), which conflicts with the rest of the stack
running on Python 3.13.  Install SLAy standalone in its own 3.11 env:

```sh
pyenv shell 3.11.x                    # or `conda create -n slay python=3.11`
python -m venv .venv-slay
source .venv-slay/bin/activate
pip install ns3-sort-bench[slay]
```

The `slay_vs_fiberlink` runner handles the two-env split by writing an
intermediate `SortedResult` cache that each env reads from its own
Python interpreter.

## Lockfile reproducibility

`requirements.lock` is the exact `uv pip compile` output from
2026-11-02.  Install the pinned environment byte-for-byte:

```sh
pip install -r requirements.lock
pip install -e .                        # install ns3-sort-bench itself
pytest                                   # 11/11 should pass
```

Regenerate after a deliberate dep bump:

```sh
uv pip compile --python-version 3.13 --extra all pyproject.toml \
    --output-file requirements.lock
```

CI should fail any `pyproject.toml` bump PR whose `requirements.lock`
wasn't regenerated in the same commit — otherwise the lockfile silently
drifts from the pins.

## Per-sorter install notes

### DARTsort (GPU, torch >= 2.2, Python >= 3.12)

Upstream metadata takes care of both constraints.  CUDA 12+ wheels are
pulled automatically.  Resource contract: `gpu_vram_mb >= 12288` —
check with `ns3-sort-bench list-contracts | grep -A3 dartsort`.

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
pip install ns3-sort-bench[ibl]
python -c "from one.api import ONE; ONE.setup()"
```

### fiber-kit (not on PyPI)

Installed from git via the `[fiber-kit]` extra.  The lockfile pins the
exact SHA validated by the smoke tests; override with:

```sh
pip install 'ns3-sort-bench[fiber-kit] @ \
    git+https://github.com/Gravios/fiber-kit.git@<branch-or-sha>'
```

## Minimum install for the Pröpper simulator + BOTM reference

```sh
pip install ns3-sort-bench
```

Core deps only; no torch, no GPU, no git clones.  Enough to run every
test in `tests/`, the Pröpper simulator corpus, the BOTM reference
adapter, and all three scorers.  First install footprint under 400 MB.
