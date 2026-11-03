"""Pursue variants for the fiber_deconv lattice ablation.

Reimplements `fiber_deconv.pursue()` with a configurable overlap-resolution
rule.  Reuses every other primitive (score cache, atom bank, cross-corr
cache, amplitude estimation) from `fiber_deconv` directly, so this file
only has to carry the branching logic for the four rules.

The rules follow Pröpper 2015 §3.4.4 exactly.  Let `c0[j, t]` be the
whitened inner product between the signal and atom j at offset t, `n_k =
‖A_k‖²` the atom's self-energy in whitened space, and `λ_k = 1/σ_a(k)²`
the amplitude-prior inverse variance.  The per-sample gain is

    G[j, t] = 2 a[j, t] c[j, t] − a[j, t]² n_j − λ_j (a[j, t] − 1)²
    a*[j, t] = clip((c[j, t] + λ_j) / (n_j + λ_j), amin_j, amax_j)

Choose the highest-gain cell (i, s); let g_pick = G[i, s] and
a_pick = a*[i, s].  After the pick, the four rules update differently:

    OMP   : subtract atom i at s from c, then JOINTLY refit all placed
            amplitudes against c0 and recompute c from scratch
    SIC-B : subtract atom i at s with amp a_pick from c; keep all other
            placements' amplitudes fixed
    SIC-A : SIC-B + add an overlap-prior penalty to the chosen cell's
            future discriminant at a nearby window (approximation of
            Pröpper's eq 3.21 ln(p(i,t_0|s^i(t_0))))
    SIC-C : SIC-B + add g_pick back into the subtracted window — makes
            subsequent discriminants read higher at the collision site,
            biasing toward overlap detection

Shape conventions and primitive imports mirror fiber_deconv.py.
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

from fiber_kit.fiber_deconv import (
    _gains, _refit, _scores, gcol,
)


PURSUE_RULES = ("omp", "sic-a", "sic-b", "sic-c")


def pursue_variant(y: np.ndarray, bank, thr: float, refrac: int, *,
                   max_iter: int = 64,
                   refit_rule: str = "omp",
                   sic_a_penalty_factor: float = 1.0) -> Tuple[List[Tuple[int, int, float, float]], np.ndarray]:
    """Lattice-variant pursuit on a whitened segment `y` with configurable
    overlap-resolution rule.

    Parameters
    ----------
    y : (L, C) whitened signal (identical convention to fiber_deconv.pursue)
    bank : Bank namedtuple as returned by fiber_deconv.build_bank
    thr : noise-null threshold (fiber_deconv.null_threshold result)
    refrac : refractory block, samples
    max_iter : hard cap on placements per segment
    refit_rule : one of PURSUE_RULES.  "omp" matches fiber_deconv's
        production path; the three SIC variants mirror Pröpper's 3.4.4.
    sic_a_penalty_factor : for refit_rule="sic-a", multiplier on the
        overlap-prior penalty applied to the chosen cell's future
        discriminant.  Pröpper's eq 3.21 uses the model's full ln(p(...))
        which depends on firing-rate estimation we don't carry in this
        reimplementation; the factor * thr proxy keeps the effect in the
        same order of magnitude.  Default 1.0 reproduces Pröpper's
        qualitative finding (SIC-A misses low-SNR overlaps); set to 0.0
        to make SIC-A behave identically to SIC-B.

    Returns
    -------
    placements : list of (atom_index, offset_sample, amplitude, gain) tuples
                 for every placed atom, in placement order
    residual : (L, C) y minus the placed-templates reconstruction

    Shape notes: `at`/`st`/`amps` arrays are length N_placements; `gains`
    carries the gain AT THE MOMENT of selection (not the final one, which
    for OMP can differ after joint refit).
    """
    if refit_rule not in PURSUE_RULES:
        raise ValueError(f"refit_rule must be one of {PURSUE_RULES}, got {refit_rule!r}")

    La = bank.atoms.shape[1]
    c0 = _scores(y, bank)
    if c0.shape[1] == 0:
        return [], y.copy()
    KR, S = c0.shape
    blocked = np.zeros((KR, S), bool)
    c = c0.copy()
    at: List[int] = []
    st: List[int] = []
    gains: List[float] = []
    amps = np.zeros(0)

    for _ in range(max_iter):
        g, a_all = _gains(c, bank)
        g[blocked] = -np.inf
        flat = int(np.argmax(g))
        i, s = divmod(flat, S)
        g_pick = float(g[i, s])
        if not np.isfinite(g_pick) or g_pick < thr:
            break

        at.append(i); st.append(s); gains.append(g_pick)

        if refit_rule == "omp":
            # fiber_deconv's production path: joint refit of all amps + full
            # score recompute from c0.
            amps = _refit(c0, bank, at, st)
            c = c0.copy()
            for ik, sk, ak in zip(at, st, amps):
                lo, hi = max(0, sk - La + 1), min(S, sk + La)
                c[:, lo:hi] -= ak * gcol(bank, ik)[:, (lo - sk + La - 1):(hi - sk + La - 1)]
        else:
            # SIC family: subtract the just-placed atom at its own best
            # amplitude; keep earlier amplitudes fixed.  No joint refit.
            a_pick = float(a_all[i, s])
            amps = np.concatenate([amps, [a_pick]])
            lo, hi = max(0, s - La + 1), min(S, s + La)
            delta = a_pick * gcol(bank, i)[:, (lo - s + La - 1):(hi - s + La - 1)]
            c[:, lo:hi] -= delta

            if refit_rule == "sic-a":
                # Overlap-prior penalty.  Pröpper's eq 3.21 adds
                # ln(p(i,t_0|s^i(t_0))) to the chosen cell's d — in the
                # gain parametrisation that's a subtract from G, which we
                # implement by nudging c[i, nearby] downward.  Window is
                # the atom's own support, factor controls strength.
                penalty = sic_a_penalty_factor * thr
                i_unit_rows = (bank.unit == bank.unit[i])
                c[i_unit_rows, lo:hi] -= penalty
            elif refit_rule == "sic-c":
                # Add g_pick back into the subtracted window at the
                # chosen cell only — raises its own future discriminant at
                # the collision site.  Pröpper reports SIC-C is more
                # sensitive but introduces false positives.
                c[i, lo:hi] += g_pick

        blocked[bank.unit == bank.unit[i],
                max(0, s - refrac):min(S, s + refrac + 1)] = True

    # Residual: subtract all placed templates (match fiber_deconv.pursue)
    r = y.copy()
    for ik, sk, ak in zip(at, st, amps):
        r[sk:sk + La] -= ak * bank.atoms[ik]

    return [(ik, sk, float(ak), gk) for ik, sk, ak, gk in zip(at, st, amps, gains)], r
