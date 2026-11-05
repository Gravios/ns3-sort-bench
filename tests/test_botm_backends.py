"""Backend-correctness tests for the BOTM reference adapter's SIC-B hot path.

The adapter can compute its discriminant matrix two ways:
  - numpy   (per-channel np.correlate, outer loop over templates)
  - torch   (one batched conv1d — CUDA when available, else CPU-BLAS)

These tests pin:
  (a) the numpy backend is deterministic and shape-correct;
  (b) the torch backend (CPU) matches numpy within float32 accumulation
      tolerance — so we can swap in GPU without silently changing spike
      counts;
  (c) backend selection honours the explicit kwarg / env override.

The tests do NOT require CUDA; the torch backend here is pinned to CPU
so they run in any environment with torch installed.  If torch isn't
installed, the torch tests are skipped.
"""
from __future__ import annotations

import numpy as np
import pytest

from ns3_sort_bench.adapters.botm_ref import (
    _discriminant_all_numpy,
    _pick_discriminant_backend,
)

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ─── numpy discriminant: shape + ground-truth values ─────────────────────────
def test_numpy_discriminant_shape_and_correctness():
    rng = np.random.default_rng(0)
    n_samp, n_ch, T, k = 300, 4, 3, 32
    residual = rng.standard_normal((n_samp, n_ch))
    f = rng.standard_normal((T, k, n_ch))
    c = rng.standard_normal(T)

    D = _discriminant_all_numpy(residual, f, c, k)
    assert D.shape == (T, n_samp - k + 1)

    # One element against an explicit correlate
    u, t = 1, 50
    ref = sum(np.correlate(residual[:, ch], f[u, :, ch], "valid")[t]
              for ch in range(n_ch)) + c[u]
    assert D[u, t] == pytest.approx(ref)


def test_numpy_discriminant_returns_empty_when_residual_shorter_than_template():
    """A pathological short residual — SIC-B's loop must not crash on it."""
    residual = np.zeros((10, 2))
    f = np.zeros((2, 32, 2))
    c = np.zeros(2)
    D = _discriminant_all_numpy(residual, f, c, 32)
    assert D.shape == (2, 0)


# ─── backend resolver ────────────────────────────────────────────────────────
def test_backend_resolver_numpy_force():
    fn, name = _pick_discriminant_backend("numpy")
    assert name == "numpy"
    assert fn is _discriminant_all_numpy


def test_backend_resolver_auto_falls_to_numpy_without_torch(monkeypatch):
    """If torch isn't importable, auto must land on numpy without raising."""
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *a, **kw):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("blocked for test")
        return real_import(name, *a, **kw)

    # Also clear cached torch module so _pick_discriminant_backend re-imports
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    monkeypatch.setattr(builtins, "__import__", _blocked)
    fn, name = _pick_discriminant_backend("auto")
    assert name == "numpy"
    assert fn is _discriminant_all_numpy


def test_backend_resolver_hard_cuda_request_raises_without_cuda(monkeypatch):
    """When the user forces cuda but it isn't available, fail loudly —
    don't silently run on CPU and pretend the GPU was used."""
    if not _HAS_TORCH:
        import builtins
        real_import = builtins.__import__

        def _blocked(name, *a, **kw):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("blocked for test")
            return real_import(name, *a, **kw)

        monkeypatch.setitem(__import__("sys").modules, "torch", None)
        monkeypatch.setattr(builtins, "__import__", _blocked)
        with pytest.raises(RuntimeError, match="torch is not installed"):
            _pick_discriminant_backend("cuda")
    else:
        import torch
        if torch.cuda.is_available():
            pytest.skip("CUDA actually available — cannot test the no-CUDA path")
        with pytest.raises(RuntimeError, match="cuda.*is_available.*False"):
            _pick_discriminant_backend("cuda")


# ─── torch backend: matches numpy within float32 tol ─────────────────────────
@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_cpu_matches_numpy_discriminant_within_f32_tol():
    """One batched conv1d on CPU must produce the same discriminant matrix
    as the per-channel numpy loop, within float32 accumulation tolerance.
    """
    rng = np.random.default_rng(1)
    n_samp, n_ch, T, k = 2000, 4, 3, 32
    residual = rng.standard_normal((n_samp, n_ch))
    f = rng.standard_normal((T, k, n_ch))
    c = rng.standard_normal(T)

    fn, name = _pick_discriminant_backend("torch-cpu")
    assert name == "torch:cpu"

    D_np = _discriminant_all_numpy(residual, f, c, k)
    D_t  = fn(residual, f, c, k)
    assert D_t.shape == D_np.shape
    # float32 accumulation → rtol ~1e-4, atol scaled by data magnitude
    assert np.allclose(D_t, D_np, rtol=1e-3,
                       atol=1e-3 * float(np.max(np.abs(D_np))))


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_cpu_argmax_matches_numpy_on_a_structured_signal():
    """On a signal where one (template, time) is the clear peak, both
    backends must pick the same (u, t) argmax — that's what SIC-B acts on.
    """
    rng = np.random.default_rng(2)
    n_samp, n_ch, T, k = 2000, 4, 3, 32
    residual = rng.standard_normal((n_samp, n_ch)) * 0.1
    # Inject a clear template-0 match at time 500
    tpl0 = rng.standard_normal((k, n_ch))
    residual[500:500 + k] += tpl0 * 5.0
    f = np.stack([tpl0] + [rng.standard_normal((k, n_ch)) for _ in range(T - 1)])
    c = np.zeros(T)

    D_np = _discriminant_all_numpy(residual, f, c, k)
    fn, _ = _pick_discriminant_backend("torch-cpu")
    D_t = fn(residual, f, c, k)

    assert np.unravel_index(np.argmax(D_np), D_np.shape) \
        == np.unravel_index(np.argmax(D_t),  D_t.shape)
