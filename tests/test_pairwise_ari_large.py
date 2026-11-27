"""The ARI is exact on lab-sized groups: the pair-count products pass 2**63
long before the spike counts do."""
import numpy as np
import pytest

from ns3_sort_bench.scorers.pairwise import _ari, _contingency


@pytest.mark.parametrize("n, k_ref, k_cand", [(2_000, 5, 7), (3_000_000, 30, 150)])
def test_ari_matches_sklearn(n, k_ref, k_cand):
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    ref = rng.integers(0, k_ref, n)
    cand = np.where(rng.random(n) < 0.8, ref * (k_cand // k_ref), rng.integers(0, k_cand, n))
    tab, _, _ = _contingency(ref, cand)
    assert _ari(tab) == pytest.approx(sk.adjusted_rand_score(ref, cand), abs=1e-9)
