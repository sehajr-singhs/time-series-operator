"""Tipping-point detection: watching the attractor's geometry warp.

A system doesn't usually announce a bifurcation in its raw time series — the
local readings look normal right up to the moment the shape of the attractor
changes. This module implements a data-driven largest-Lyapunov-exponent
estimator (Wolf's method, from scratch) that reads the *geometry* of the
reconstructed phase space:

    lambda1 > 0  -> nearby trajectories diverge exponentially
                   (chaotic / approaching a bifurcation / tipping)
    lambda1 <= 0 -> the attractor is a stable point or limit cycle

We sweep the Lorenz parameter rho from 1 (fixed point) through the homoclinic
explosion (rho ~ 24.7, onset of chaos) up to 40 (fully chaotic butterfly) and
show the detector flags the tipping point from a single scalar channel.
"""

from __future__ import annotations

import hashlib

import numpy as np

from . import attractors
from .embedding import embed_signal
from .pipeline import normalize


def wolf_lambda1(states, dt=1.0, theiler=50, track_steps=40,
                 n_fiducial=24, rng=0):
    """Largest Lyapunov exponent from a reconstructed trajectory (Wolf 1985).

    For each fiducial point we find its nearest neighbour *away in time*
    (Theiler window, so we don't pair a point with itself), then watch how
    that tiny separation grows over ``track_steps``. Exponential growth means
    chaos:

        lambda1 = <ln(d_k / d_0)> / (track_steps * dt)

    Averaged over several fiducial points for stability.

    Parameters
    ----------
    states  : (T, d) embedded trajectory
    dt      : time between successive rows
    """
    rng = np.random.default_rng(rng)
    T = states.shape[0]
    if T < 400:
        return float("nan"), 0
    # subsample hard enough that the pairwise work stays bounded (Wolf
    # estimates saturate well before ~2k reconstructed points)
    subsample = max(1, T // 2000)
    S = states[::subsample]
    n = len(S)
    track = min(track_steps, n // 4)

    # pairwise squared distances in row-blocks (memory-bounded vectorized:
    # the full n x n x d broadcast thrashes memory for long trajectories)
    D = np.full((n, n), np.inf, dtype=np.float64)
    mask = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < theiler
    block = 256
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        d = ((S[i0:i1, None, :] - S[None, :, :]) ** 2).sum(-1)
        D[i0:i1] = d
    D[mask] = np.inf
    np.fill_diagonal(D, np.inf)

    picks = rng.choice(n, size=min(n_fiducial, n), replace=False)
    lam = []
    for i in picks:
        j = int(np.nanargmin(D[i]))
        d0 = max(np.sqrt(D[i, j]), 1e-9)
        # track the two points forward, rescaling if they separate too far
        d_prev = d0
        log_sum = 0.0
        k = 0
        i_k, j_k = i, j
        for step in range(track):
            i_k += 1
            j_k += 1
            if i_k >= n or j_k >= n:
                break
            d_new = max(np.linalg.norm(S[i_k] - S[j_k]), 1e-12)
            if d_new > 2.0 * d_prev or d_new < 0.5 * d_prev:
                # rescale: find a new neighbour in the same direction
                cand = np.linalg.norm(S[i_k] - S, axis=1)
                cand[mask[i_k]] = np.inf
                cand[i_k] = np.inf
                j_new = int(np.nanargmin(cand))
                d_new = max(cand[j_new], 1e-12)
                j_k = j_new
            if d_prev > 0:
                log_sum += np.log(d_new / max(d_prev, 1e-12))
                k += 1
            d_prev = d_new
        if k > 0:
            lam.append(log_sum / (k * dt * subsample))
    if not lam:
        return float("nan"), 0
    return float(np.mean(lam)), int(len(lam))


_TIPPING_CACHE = {}


def tipping_metrics(series, dt=1.0, max_len=4000):
    """One-shot tipping report for any scalar series.

    Returns dict with the largest Lyapunov exponent, the dominant DMD
    eigenvalue magnitude (|mu| -> 1 means persistent oscillation) and a
    composite tipping score in [0, 1] (0 = laminar, 1 = strongly chaotic).
    The Wolf estimator dominates probe cost, so results are cached by the
    normalized series values (they do not depend on any model).
    """
    x = normalize(np.asarray(series, dtype=float))
    if len(x) > max_len:
        x = x[:: max(1, len(x) // max_len)]
    key = (len(x), float(x[0]) if len(x) else 0.0,
           hashlib.md5(x.tobytes()).hexdigest() if len(x) else "")
    if key in _TIPPING_CACHE:
        return dict(_TIPPING_CACHE[key])
    if len(x) > max_len:
        x = x[:: max(1, len(x) // max_len)]
    # degenerate (constant / collapsed) series: no attractor to read
    if float(np.std(x)) < 1e-6:
        return {"lambda1": float("nan"), "dmd_mu_top": float("nan"),
                "tipping_score": float("nan"), "tau": 0, "dim": 0,
                "n_fiducial": 0}
    S, (tau, m) = embed_signal(x, max_lag=min(200, len(x) // 3))
    lam, nfid = wolf_lambda1(S, dt=dt)

    # dominant Koopman/DMD eigenvalue magnitude: persistence of the oscillation
    from .koopman import exact_dmd
    X, Y = S[:-1], S[1:]
    if len(X) > 20000:
        X, Y = X[::2], Y[::2]
    U, s, _ = np.linalg.svd(X.T, full_matrices=False)
    if s.size and s[0] > 1e-8:
        model = exact_dmd(X, Y, rank=min(8, X.shape[1]))
        mu_top = float(np.max(np.abs(model["eigenvalues"])))
    else:
        mu_top = float("nan")

    # composite: positive lambda1 is the tipping tell; |mu| near 1 is a
    # "sustained structure" tell. Map to [0,1] with a soft logistic.
    if np.isfinite(lam):
        score = 1.0 / (1.0 + np.exp(-4.0 * lam))
    else:
        score = float("nan")
    out = {"lambda1": lam, "dmd_mu_top": mu_top, "tipping_score": score,
           "tau": int(tau), "dim": int(m), "n_fiducial": nfid}
    _TIPPING_CACHE[key] = dict(out)
    return out


def bifurcation_sweep(rho_values, dt=0.01, n=16000, discard=4000, seed=0):
    """Tipping score of the Lorenz x-channel as rho sweeps through chaos onset.

    Long integration (160k steps) so stable regimes show cleanly negative
    Lyapunov exponents and transient chaos doesn't dominate the reading.
    Returns dict with per-rho lambda1, DMD radius and tipping score.
    """
    out = {"rho": [], "lambda1": [], "dmd_mu_top": [], "tipping_score": []}
    for rho in rho_values:
        y = attractors.lorenz_trajectory(dt=dt, n=n, discard=discard, rho=rho)
        m = tipping_metrics(y[:, 0], dt=dt)
        out["rho"].append(float(rho))
        out["lambda1"].append(m["lambda1"])
        out["dmd_mu_top"].append(m["dmd_mu_top"])
        out["tipping_score"].append(m["tipping_score"])
    return out


def detect_tipping(sweep, window=3, lam_threshold=0.05):
    """First rho where the Lyapunov exponent durably crosses into positive
    territory (sustained exponential divergence) — the detected tipping point.

    Uses lambda1 directly (not the soft score) and requires `window`
    consecutive rho values above threshold so transient spikes don't fire it.
    """
    lam = np.asarray(sweep["lambda1"], dtype=float)
    rho = np.asarray(sweep["rho"], dtype=float)
    for i in range(len(lam) - window):
        if np.all(np.isfinite(lam[i:i + window])) and \
           np.all(lam[i:i + window] > lam_threshold):
            return float(rho[i]), float(lam_threshold)
    return float("nan"), float(lam_threshold)
