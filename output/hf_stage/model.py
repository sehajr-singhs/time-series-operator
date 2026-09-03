"""Time-Series Operator (TSO) — frozen-lift zero-shot forecasting.

Self-contained inference module for the hub checkpoint. The operator was
pretrained with four pretexts (reconstruction, Koopman-linear dynamics,
scale covariance, arrow of time) on a 40-series corpus across 8 domains,
then transfer is a *closed-form* Koopman fit on the frozen latent — zero
gradient steps on the target series.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

EMBED_DIM = 5


class FoundationOperator(nn.Module):
    def __init__(self, state_dim=EMBED_DIM, latent_dim=256, hidden=768):
        super().__init__()
        self.state_dim, self.latent_dim = state_dim, latent_dim
        self.enc = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, state_dim),
        )
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)
        self.scale_map = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )
        self.arrow_head = nn.Sequential(
            nn.Conv1d(latent_dim, 16, 9, padding=4), nn.Tanh(),
            nn.Conv1d(16, 1, 9, padding=4),
        )

    def phi(self, s): return self.enc(s)
    def psi(self, z): return self.dec(z)
    def step(self, z): return self.K(z)


def normalize(x):
    x = np.asarray(x, dtype=float)
    m, s = float(np.nanmean(x)), float(np.nanstd(x))
    return (x - m) / (s + 1e-8)


def autocorrelation(x, max_lag=300):
    """Autocorrelation r(lag) for lag = 1..max_lag (kernel-exact)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = float(np.dot(x, x) / max(len(x), 1))
    if var <= 0.0:
        return np.zeros(max_lag)
    max_lag = min(max_lag, len(x) - 2)
    ac = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        ac[lag - 1] = float(np.dot(x[:-lag], x[lag:])) / var             / (len(x) - lag) * len(x)
    return ac


def suggest_delay(x, max_lag=300, depth_threshold=0.1):
    """Takens delay: deep zero crossing, else first local minimum
    (Fraser-Swinney proxy), else 1/e decay (kernel-exact)."""
    x = np.asarray(x, dtype=float)
    max_lag = min(max_lag, max(len(x) - 2, 1))
    if max_lag < 1:
        return 1
    ac = autocorrelation(x, max_lag)
    if not np.any(np.isfinite(ac)):
        return 1
    crosses = np.where(np.diff(np.sign(ac)) != 0)[0] + 1
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


def takens_embed(x, delay, dim):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = n - (dim - 1) * delay
    if m <= 1:
        raise ValueError("series too short for this (delay, dim)")
    idx = np.arange(dim)[:, None] * delay + np.arange(m)[None, :]
    return x[idx].T


def false_nearest_fraction(states, delay, max_n=4000, rng=None):
    """Fraction of false nearest neighbours (kernel-exact)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(states)
    if n < 2:
        return 1.0
    if n > max_n:
        idx = np.sort(rng.choice(n, max_n, replace=False))
        states = states[idx]
        n = max_n
    m = states.shape[1]
    frac = 0.0
    for i in range(n):
        d = np.sum((states - states[i]) ** 2, axis=1)
        d[i] = np.inf
        j = int(np.argmin(d))
        dx = abs(states[i, 0] - states[j, 0])
        dy = abs(states[i, m - 1] - states[j, m - 1]) if m > 1 else 0.0
        frac += 1.0 if (dy > 2.0 * dx and dx > 1e-12) else 0.0
    return float(frac / n)


def suggest_dim(x, delay, max_dim=6, fnn_threshold=0.02):
    """Embedding dimension where false neighbours collapse (kernel-exact)."""
    n = len(x)
    for d in range(1, max_dim + 1):
        if (d - 1) * delay >= n - 2:
            break
        states = takens_embed(x, delay, d)
        frac = false_nearest_fraction(states, delay)
        if frac < fnn_threshold and d >= 2:
            return d
    return max_dim


def embed_signal(x, delay=None, dim=None, max_lag=300):
    """Choose (delay, dim) from data, then embed (kernel-exact)."""
    x = np.asarray(x, dtype=float)
    if delay is None:
        delay = suggest_delay(x, max_lag)
    if dim is None:
        dim = suggest_dim(x, delay)
    return takens_embed(x, delay, dim), (delay, dim)


def _rmse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2) ** 0.5)


def zero_shot_forecast(model, series, train_frac=0.7, horizon_frac=0.2,
                       device="cpu"):
    """Fit a linear Koopman operator on the frozen latent, then roll it out
    (kernel-exact: rank-reduced SVD fit, adaptive ridge, spectral clipping,
    envelope projection). Returns skill vs persistence + corr."""
    x = normalize(np.asarray(series, dtype=float))
    max_lag = min(150, max(len(x) // 6, 1))
    S, (tau, _) = embed_signal(x, dim=EMBED_DIM, max_lag=max_lag)
    n = len(S)
    split = int(n * train_frac)
    horizon = min(int(n * horizon_frac), n - split - 1, 100)
    with torch.no_grad():
        Z = model.phi(torch.tensor(S, dtype=torch.float32,
                                   device=device)).cpu().numpy()
    Ztr = Z[: split - 1]
    mu = Ztr.mean(axis=0)
    Zc = Ztr - mu
    U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
    R = max(2, min(8, len(Ztr) // 16, len(s)))
    Vr = Vt[:R]
    zr = Zc @ Vr.T
    ridge = 1e-1 * float(np.mean(zr ** 2))
    A, B = zr[:-1], zr[1:]
    AtA = A.T @ A + ridge * np.eye(R)
    Kfit_r = np.linalg.solve(AtA, A.T @ B)
    w, V = np.linalg.eig(Kfit_r)
    w = np.where(np.abs(w) > 1.02, w / np.abs(w) * 1.02, w)
    Kfit_r = (V * w[None, :]) @ np.linalg.inv(V)
    Kfit = Vr.T @ Kfit_r @ Vr
    cap = 1.2 * float(np.percentile(np.linalg.norm(Ztr, axis=1), 99)) + 1e-9
    zk = Z[split - 1]
    z_path = [zk]
    for _ in range(horizon):
        zk = Kfit @ zk
        nrm = float(np.linalg.norm(zk))
        if nrm > cap:
            zk = zk * (cap / nrm)
        z_path.append(zk)
    z_path = np.array(z_path).real
    with torch.no_grad():
        pred_emb = model.psi(torch.tensor(z_path, dtype=torch.float32,
                                          device=device)).cpu().numpy()
    true_vals = S[split - 1: split + horizon, 0]
    pred_vals = pred_emb[:, 0]
    e_koop = _rmse(pred_vals, true_vals)
    pers = np.full(len(true_vals), true_vals[0])
    e_pers = _rmse(pers, true_vals)
    skill = 100.0 * (e_pers - e_koop) / max(e_pers, 1e-12)
    corr = float(np.corrcoef(pred_vals, true_vals)[0, 1])         if len(true_vals) > 2 else float("nan")
    return {"true": true_vals, "pred": pred_vals,
            "skill_pct": skill, "corr": corr, "horizon": horizon,
            "tau": int(tau)}
