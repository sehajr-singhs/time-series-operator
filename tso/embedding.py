"""Takens' embedding theorem, implemented from scratch.

Given a single scalar observation x(t), we reconstruct a phase space that is
topologically equivalent to the hidden multi-dimensional dynamics by forming
delay vectors

    s(t) = [x(t), x(t - tau), x(t - 2*tau), ..., x(t - (m-1)*tau)]

This is the step that turns "a row of numbers" into a manifold the model can
reason about geometrically.
"""

from __future__ import annotations

import numpy as np


def autocorrelation(x, max_lag=300):
    """Autocorrelation r(lag) for lag = 1..max_lag."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = float(np.dot(x, x) / max(len(x), 1))
    if var <= 0.0:
        return np.zeros(max_lag)
    max_lag = min(max_lag, len(x) - 2)
    ac = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        ac[lag - 1] = float(np.dot(x[:-lag], x[lag:])) / var / (len(x) - lag) * len(x)
    return ac


def suggest_delay(x, max_lag=300, depth_threshold=0.1):
    """Pick the Takens delay tau from the autocorrelation.

    Rule (in order of preference):
      1. a *deep* zero crossing (r < -0.1) — clean decorrelation time;
      2. the first local minimum — the least-correlated lag before the
         autocorrelation bounces back (proxy for the first minimum of
         mutual information, the Fraser-Swinney choice);
      3. the 1/e decay time;
      4. max_lag.
    Chaotic signals like the Lorenz butterfly never cross zero (their
    correlation decays but oscillates), so rule 2 is what rescues them.
    """
    x = np.asarray(x, dtype=float)
    max_lag = min(max_lag, max(len(x) - 2, 1))
    if max_lag < 1:
        return 1
    ac = autocorrelation(x, max_lag)
    if not np.any(np.isfinite(ac)):
        return 1
    signs = np.sign(ac)
    crosses = np.where(np.diff(signs) != 0)[0] + 1
    if len(crosses):
        c = int(crosses[0])
        if ac[c - 1] < -depth_threshold:
            return c
    mins = np.where((ac[1:-1] <= ac[:-2]) & (ac[1:-1] <= ac[2:]))[0] + 1
    if len(mins):
        return int(mins[0])
    decay = np.where(ac < 1.0 / np.e)[0]
    if len(decay):
        return int(decay[0]) + 1
    return max_lag


# backwards-compatible alias
first_zero_autocorr = suggest_delay


def takens_embed(x, delay, dim):
    """Delay-embed a scalar series into (rows, dim) phase-space coordinates.

    Row i of the output corresponds to time index i of the input signal:
    S[i] = [x[i], x[i+delay], ..., x[i+(dim-1)*delay]].
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = n - (dim - 1) * delay
    if m <= 1:
        raise ValueError("series too short for this (delay, dim)")
    idx = np.arange(dim)[:, None] * delay + np.arange(m)[None, :]
    return x[idx].T


def false_nearest_fraction(states, delay, max_n=4000, rng=None):
    """Fraction of false nearest neighbours for the current embedding dim.

    A point's neighbour in m dimensions is "false" if adding one more delay
    coordinate yanks them apart. When the fraction collapses, the attractor is
    fully unfolded.
    """
    rng = np.random.default_rng(rng)
    d = states.shape[1]
    m = states.shape[0]
    n = min(m, max_n)
    pick = rng.choice(m, size=n, replace=False)
    A = states[pick]
    # distances in current d dimensions (squared)
    D = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    j = D.argmin(1)
    dist_d = np.sqrt(D[np.arange(n), j])
    # next coordinate available only when BOTH the point and its neighbour
    # have room for one more delay coordinate
    next_ok = (pick + d * delay) < m
    nbr_ok = (pick[j] + d * delay) < m
    both = next_ok & nbr_ok
    if not both.any():
        return 0.0
    nxt = states[pick[both] + d * delay, 0]
    nxt_j = states[pick[j[both]] + d * delay, 0]
    num = np.abs(nxt - nxt_j)
    den = dist_d[both]
    ratio = num / np.maximum(den, 1e-12)
    thr = np.maximum(10.0, 2.0 / np.maximum(den, 1e-12))
    return float((ratio > thr).mean())


def suggest_dim(x, delay, max_dim=6, fnn_threshold=0.02, verbose=False):
    """Pick the embedding dimension where false neighbours collapse."""
    n = len(x)
    prev_frac = 1.0
    for d in range(1, max_dim + 1):
        if (d - 1) * delay >= n - 2:
            break
        states = takens_embed(x, delay, d)
        frac = false_nearest_fraction(states, delay)
        if verbose:
            print(f"  dim={d}: FNN fraction = {frac:.3f}")
        if frac < fnn_threshold and d >= 2:
            return d
        prev_frac = frac
    return max_dim


def embed_signal(x, delay=None, dim=None, max_lag=300, verbose=False):
    """One-call helper: choose (delay, dim) from data, then embed."""
    x = np.asarray(x, dtype=float)
    if delay is None:
        delay = first_zero_autocorr(x, max_lag)
    if dim is None:
        dim = suggest_dim(x, delay, verbose=verbose)
    if verbose:
        print(f"Takens: tau={delay}, m={dim} (from {len(x)} samples)")
    return takens_embed(x, delay, dim), (delay, dim)
