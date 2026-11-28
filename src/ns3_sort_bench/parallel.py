"""CPU parallelism shared by the SpikeInterface-based adapters.

SpikeInterface runs every chunked job (filtering, whitening, caching a
recording, peak detection, template matching, waveform extraction) with
its global job kwargs, and those default to ``n_jobs=1``.  Nothing in the
bench set them, so MountainSort 5's preprocessing and cache, all of
SpyKING CIRCUS 2, Kilosort 4's binary copy and SLAy's analyzer ran on
one process however many cores the machine had.

``ns3sb run --n-jobs N`` (default: every CPU this process may use) puts
``n_jobs`` into the adapter kwargs, and the SI adapters run their sort
inside :func:`si_job_kwargs`.

GPU: Kilosort 4 and SLAy pick CUDA on their own.  SpyKING CIRCUS 2's
template matching defaults to numpy on the CPU, and DARTsort runs one
GPU worker; their adapters switch to CUDA / more GPU workers when
:func:`cuda_available` and the adapter's yaml doesn't say otherwise.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional


def available_cpus() -> int:
    """CPUs this process may run on (taskset / cgroup aware where possible)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def physical_cores() -> int:
    """Physical cores among the available CPUs (SMT siblings not counted)."""
    cpus = available_cpus()
    try:
        import psutil
        phys = psutil.cpu_count(logical=False) or cpus
        logical = psutil.cpu_count(logical=True) or cpus
        return max(1, min(cpus, cpus * phys // logical))
    except Exception:
        return cpus


def resolve_n_jobs(n_jobs: Optional[int]) -> int:
    """None or 0 → one job per physical core (on a 64-core / 128-thread
    machine, 64: the SMT sibling adds little to numeric work and doubles
    the processes and their memory); negative → that many fewer + 1 of
    the available CPUs (-1 = all, -2 = all but one), as joblib and
    SpikeInterface count."""
    cpus = available_cpus()
    if n_jobs is None or int(n_jobs) == 0:
        return physical_cores()
    n = int(n_jobs)
    if n < 0:
        return max(1, cpus + 1 + n)
    return n


def cuda_available() -> bool:
    """True when torch is installed and sees a CUDA device."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@contextmanager
def si_job_kwargs(n_jobs: Optional[int]):
    """Set SpikeInterface's global ``n_jobs`` for the duration of a sort.

    The earlier global job kwargs are restored afterwards, so one
    adapter's setting never leaks into the next.  Without SpikeInterface
    this does nothing.
    """
    try:
        import spikeinterface as si
    except ImportError:
        yield None
        return
    before = dict(si.get_global_job_kwargs())
    n = resolve_n_jobs(n_jobs)
    si.set_global_job_kwargs(n_jobs=n)
    try:
        yield n
    finally:
        si.set_global_job_kwargs(**before)
