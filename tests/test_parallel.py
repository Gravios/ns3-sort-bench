"""SpikeInterface job kwargs are set for a sort and restored after it."""
import pytest

from ns3_sort_bench.parallel import available_cpus, resolve_n_jobs, si_job_kwargs


def test_resolve_n_jobs():
    c = available_cpus()
    assert resolve_n_jobs(None) == c
    assert resolve_n_jobs(0) == c
    assert resolve_n_jobs(-1) == c
    assert resolve_n_jobs(-2) == max(1, c - 1)
    assert resolve_n_jobs(-10_000) == 1
    assert resolve_n_jobs(5) == 5


def test_si_job_kwargs_sets_and_restores():
    si = pytest.importorskip("spikeinterface")
    before = dict(si.get_global_job_kwargs())
    with si_job_kwargs(3) as n:
        assert n == 3
        assert si.get_global_job_kwargs()["n_jobs"] == 3
    assert si.get_global_job_kwargs() == before


def test_si_job_kwargs_restores_after_error():
    si = pytest.importorskip("spikeinterface")
    before = dict(si.get_global_job_kwargs())
    with pytest.raises(RuntimeError):
        with si_job_kwargs(2):
            raise RuntimeError("sort failed")
    assert si.get_global_job_kwargs() == before


def test_si_sorter_runs_inside_job_kwargs(monkeypatch, tmp_path):
    """The SI adapter's run_sorter call sees the requested n_jobs."""
    si = pytest.importorskip("spikeinterface")
    import numpy as np
    import spikeinterface.core as sc
    import spikeinterface.sorters as ss
    from ns3_sort_bench.adapters.si_sorters import MountainSort5Adapter

    seen = {}

    def fake_run_sorter(**kw):
        seen["n_jobs"] = si.get_global_job_kwargs()["n_jobs"]
        return sc.NumpySorting.from_unit_dict(
            [{0: np.array([10, 20])}], sampling_frequency=1000.0)

    monkeypatch.setattr(ss, "run_sorter", fake_run_sorter)
    rec = sc.NumpyRecording([np.zeros((1000, 2), np.float32)], 1000.0)
    a = MountainSort5Adapter()
    a._available = True
    r = a.sort(rec, n_jobs=4)
    assert seen["n_jobs"] == 4
    assert r.extras["n_jobs"] == 4


def test_sc2_matching_on_cuda_only_with_a_gpu(monkeypatch):
    import ns3_sort_bench.adapters.si_sorters as ss
    a = ss.SpykingCircus2Adapter.__new__(ss.SpykingCircus2Adapter)
    monkeypatch.setattr(ss, "cuda_available", lambda: False)
    assert a._device_kwargs({"x": 1}) == {"x": 1}
    monkeypatch.setattr(ss, "cuda_available", lambda: True)
    kw = a._device_kwargs({"x": 1, "job_kwargs": {"chunk_duration": "2s"}})
    assert kw["matching"]["matching_kwargs"] == {"engine": "torch",
                                                 "torch_device": "cuda"}
    assert kw["job_kwargs"] == {"chunk_duration": "2s", "mp_context": "spawn"}
    # the yaml's own matching wins
    mine = {"method": "circus-omp", "matching_kwargs": {"engine": "numpy"}}
    assert a._device_kwargs({"matching": mine})["matching"] is mine


def test_dartsort_jobs(monkeypatch):
    import ns3_sort_bench.adapters.dartsort as ds
    monkeypatch.setattr(ds, "cuda_available", lambda: True)
    kw = ds.dartsort_jobs({}, 64)
    assert kw == {"n_jobs_cpu": 64, "n_jobs_gpu": ds._DEFAULT_GPU_JOBS}
    assert ds.dartsort_jobs({"n_jobs_gpu": 1, "n_jobs_cpu": 2}, 64) == \
        {"n_jobs_gpu": 1, "n_jobs_cpu": 2}
    monkeypatch.setattr(ds, "cuda_available", lambda: False)
    assert ds.dartsort_jobs({}, 8) == {"n_jobs_cpu": 8}
