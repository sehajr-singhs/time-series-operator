"""The TSO end-to-end pipeline.

    signal -> normalize -> Takens embedding (phase space)
           -> Koopman lift (linearize chaos)      -> forecast + metrics
           -> neural vector field (learn the flow) -> attractor reproduction

Also exposes a light "scale space" analysis: the same Koopman fit run at
multiple decimations, to show how the model sees the same system through
different temporal lenses (the multi-scale leg of the TSO vision).
"""

from __future__ import annotations

import numpy as np

from .embedding import embed_signal
from .koopman import edmd_rff, exact_dmd, forecast


def normalize(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-12)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def persistence_baseline(train_tail, horizon):
    """Naive forecast: repeat the last observed value."""
    return np.full(horizon, train_tail)


def tso_forecast(signal, horizon=None, embed_dim=None, delay=None, rank=None,
                 lift_dim=128, method="edmd", train_frac=0.7, seed=0,
                 verbose=True):
    """Fit Koopman on the first train_frac of the embedded series, then
    forecast the rest in closed form. Returns results dict with:
      - model, metrics (rmse_koopman, rmse_persist, skill)
      - embedded series + alignment so callers can plot.
    """
    x = normalize(np.asarray(signal, dtype=float))
    S, (tau, m) = embed_signal(x, delay=delay, dim=embed_dim, verbose=verbose)

    X, Y = S[:-1], S[1:]              # snapshot pairs of the flow
    split = int(len(S) * train_frac)
    Xtr, Ytr = X[:split], Y[:split]

    if method == "edmd":
        model = edmd_rff(Xtr, Ytr, lift_dim=lift_dim, rank=rank, seed=seed)
    else:
        model = exact_dmd(Xtr, Ytr, rank=rank)

    horizon = horizon if horizon is not None else len(S) - split
    horizon = min(horizon, len(S) - split)
    steps = max(1, horizon)

    pred = forecast(model, Xtr[-1], steps)          # (steps+1, d) embedded

    # column 0 of the embedding is the signal itself. pred[k] corresponds to
    # embedded row (split-1+k): row split-1 is the last training state.
    true_vals = S[split - 1 : split - 1 + steps + 1, 0]
    pred_vals = pred[:, 0]

    e_koop = rmse(pred_vals, true_vals)
    e_pers = rmse(persistence_baseline(Xtr[-1, 0], steps + 1), true_vals)
    skill = 100.0 * (e_pers - e_koop) / max(e_pers, 1e-12)

    if verbose:
        print(f"  Koopman RMSE      = {e_koop:.4f}")
        print(f"  Persistence RMSE  = {e_pers:.4f}")
        print(f"  Skill over persistence = {skill:+.1f}%")

    return {
        "model": model,
        "signal": x,
        "embedded": S,
        "split": split,
        "pred_embedded": pred,
        "pred": pred_vals,
        "true": true_vals,
        "tau": tau,
        "dim": m,
        "rmse_koopman": e_koop,
        "rmse_persistence": e_pers,
        "skill": skill,
        "method": method,
    }


def scale_space(signal, scales=(1, 2, 4, 8), rank=8, seed=0):
    """Fit plain DMD on the series decimated by 1,2,4,8 and return the top
    Koopman eigenvalue magnitudes per scale — the "same physics, different
    temporal lens" view. Eigenvalues that stay on the unit circle at every
    scale are the scale-covariant skeleton of the system.
    """
    x = normalize(np.asarray(signal, dtype=float))
    out = {}
    for s in scales:
        if s > 1:
            # block-average decimation: a crude low-pass "renormalization" step
            k = len(x) - len(x) % s
            xs = x[:k].reshape(-1, s).mean(axis=1)
        else:
            xs = x
        S, (tau, _) = embed_signal(xs, max_lag=min(150, len(xs) // 3))
        X, Y = S[:-1], S[1:]
        model = exact_dmd(X, Y, rank=rank)
        w = np.sort(np.abs(model["eigenvalues"]))[::-1][:6]
        out[s] = {"eigenvalues": model["eigenvalues"], "top_magnitudes": w,
                  "n": len(xs), "tau": tau}
    return out
