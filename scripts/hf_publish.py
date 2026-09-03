#!/usr/bin/env python3
"""Publish the best v11 TSO checkpoint to the Hugging Face Hub.

Builds a self-contained model repo:
  - model.py        (FoundationOperator + zero-shot forecast, no tso import)
  - config.json     (architecture + training config)
  - foundation_model.pt (state dict)
  - metrics.json    (full in-kernel evaluation)
  - README.md       (model card with results, usage, solar-cycle discovery)
  - figures         (pretrain curves, solar cycle, latent geometry, zero-shot)

Usage: python scripts/hf_publish.py --seed 0 --repo sehajrsingh/tso-foundation-v11
"""
import argparse
import json
import os
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PY = '''\
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
        ac[lag - 1] = float(np.dot(x[:-lag], x[lag:])) / var \
            / (len(x) - lag) * len(x)
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
    corr = float(np.corrcoef(pred_vals, true_vals)[0, 1]) \
        if len(true_vals) > 2 else float("nan")
    return {"true": true_vals, "pred": pred_vals,
            "skill_pct": skill, "corr": corr, "horizon": horizon,
            "tau": int(tau)}
'''

README_TMPL = """---
language:
- en
license: mit
tags:
- time-series
- foundation-model
- koopman-operator
- dynamical-systems
- zero-shot
datasets:
- sehajrsingh/tso-foundation-corpus-v11
pipeline_tag: time-series-forecasting
---

# TSO Foundation Model — {ver}

**Time is geometry.** The Time-Series Operator (TSO) is a foundation model
that learns the *shape* of dynamical systems instead of token-chunking
numbers. Pretraining uses four self-supervised pretexts — reconstruction,
Koopman-linear dynamics, scale covariance (a renormalization leg), and the
arrow of time — on {n_series} series across {n_domains} domains
(electricity grids, meteorology, ECG, finance, epidemiology, economics,
solar physics, chaotic systems). Transfer to a never-seen series is a
**closed-form Koopman fit on the frozen latent**: zero gradient steps.

## Results ({ver}, latent {latent} / hidden {hidden},
{iters:,} iters, {hw})

| Metric | Value |
|---|---|
| Zero-shot wins vs per-series GRU (40 series, in-kernel) | {gru_wins}/40 ({gru_pct}%) |
| Median frozen skill vs persistence | {median_skill:+.1f}% (GRU: {median_gru:+.1f}%) |
| Solar-cycle rediscovery (held-out sunspots) | {solar_months:.0f} mo vs known 132 ({solar_years:.1f} yr) |
| Arrow-of-time pretext accuracy | {arrow_acc:.1f}% |

The frozen operator **rediscovers the ~11-year Schwabe solar cycle** from a
scale-space scan of its fitted Koopman eigenvalues on the held-out sunspot
series — a purely structural, unsupervised discovery.

## Zero-shot usage

```python
import torch
from model import FoundationOperator, zero_shot_forecast

model = FoundationOperator(latent_dim={latent}, hidden={hidden})
model.load_state_dict(torch.load("foundation_model.pt", map_location="cpu"))

series = [...]  # your 1-D series (any domain, any sampling rate)
res = zero_shot_forecast(model, series)
print(res["skill_pct"], res["corr"])   # skill vs persistence on the test split
```

No training on your data is needed: the Koopman operator is fitted in closed
form on the frozen latent (ridge-regularized), then rolled out with an
envelope projection that keeps the trajectory on the observed attractor.

## Architecture

1. **Takens embeddings** — raw series → delay-embedding (per-series tau).
2. **Scale space** — fine + coarsened (renormalized) embeddings; the
   `scale_map` forces scale covariance, so periods like the solar cycle
   reappear as clean eigenmodes at coarse scales.
3. **Koopman lift** — a deep encoder flattens the nonlinear attractor into a
   latent where dynamics are approximately linear (`K`).
4. **Arrow of time** — a conv head classifies forward vs reversed windows.

## Honest limitations

- Spike / near-unit-root series (Dogecoin, covid-india) still defeat any
  closed-form probe; persistence is unbeatable there.
- The frozen *linear* probe plateaus: pretraining past pretext saturation
  (≈25k iters at this width) does not buy transfer — capacity and corpus
  breadth are the levers.
- Pretext losses pay off only above ~15k iterations.

## Reproduce

Training protocol: {train_cmd}. Full study + manuscript:
`output/study/paper/main.pdf` in the source repo.

## Citation

```bibtex
@misc{{tso-{ver},
  title={{Time is geometry: an operator foundation model that learns the shape of dynamical systems}},
  author={{{{TSO}} Project}},
  year={{2026}}
}}
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="dir containing metrics.json + foundation_model.pt")
    ap.add_argument("--ver", default="v11")
    ap.add_argument("--hw", default="T4 + AMP")
    ap.add_argument("--train-cmd", default="`scripts/modal_v11.py` (Modal, T4); corpus `scripts/build_corpus_modal.py`")
    ap.add_argument("--latent", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--iters", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(ROOT, args.src)
    assert os.path.exists(os.path.join(src, "metrics.json")), f"no metrics in {src}"

    metrics = json.load(open(os.path.join(src, "metrics.json")))
    cfg = metrics.get("config") or {}
    zs, gru = metrics["zero_shot"], metrics["gru_baseline"]
    if cfg:
        args.latent = cfg["latent_dim"]
        args.hidden = cfg["hidden"]
        args.iters = cfg["iters"]
        args.seed = metrics["seed"]
    else:
        pre = metrics["pretrain"]
        args.latent = pre["latent_dim"]
        args.hidden = pre["hidden"]
        args.iters = pre["iters"]
    wins = sum(1 for n in zs
               if (isinstance(zs[n], dict) and zs[n].get("skill_pct", -1e9)
                   > gru.get(n, -1e9)))
    solar = metrics.get("solar_cycle") or {}
    arrow = (metrics["pretrain"].get("pretext_losses") or {}).get("arrow_acc")

    # stage in a clean build dir
    build = os.path.join(ROOT, "output", "hf_stage")
    shutil.rmtree(build, ignore_errors=True)
    os.makedirs(build)
    ck = os.path.join(src, "foundation_model.pt")
    assert os.path.exists(ck), f"no checkpoint in {src}"
    shutil.copy(ck, os.path.join(build, "foundation_model.pt"))
    shutil.copy(os.path.join(src, "metrics.json"), os.path.join(build, "metrics.json"))
    for f in ("pretrain_curves.png", "solar_cycle.png", "latent_geometry.png"):
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(build, f))
    for f in os.listdir(src):
        if f.startswith("zero_shot_"):
            shutil.copy(os.path.join(src, f), os.path.join(build, f))
    with open(os.path.join(build, "model.py"), "w") as fh:
        fh.write(MODEL_PY)
    with open(os.path.join(build, "config.json"), "w") as fh:
        json.dump({"latent_dim": args.latent, "hidden": args.hidden,
                   "iters": args.iters, "seed": args.seed,
                   "embed_dim": 5, "corpus": metrics.get("corpus_note",
                                                          "40 series / 8 domains"),
                   "version": args.ver, "device": metrics["device"],
                   "gpu": metrics.get("gpu")}, fh, indent=2)
    domains = sorted({m["domain"] for m in metrics["corpus"]})
    median_skill = float(np.median([zs[n]["skill_pct"] for n in zs]))
    gru_vals = [g for g in gru.values() if isinstance(g, (int, float))]
    median_gru = float(np.median(gru_vals))
    solar_months = solar.get("period_months")
    if not solar_months:
        solar_months = float("nan")
    readme = README_TMPL.format(
        ver=args.ver, hw=args.hw, train_cmd=args.train_cmd,
        n_series=len(zs), n_domains=len(domains),
        latent=args.latent, hidden=args.hidden, iters=args.iters,
        gru_wins=wins, gru_pct=100.0 * wins / len(zs),
        median_skill=median_skill, median_gru=median_gru,
        solar_months=solar_months, solar_years=solar_months / 12.0,
        arrow_acc=100.0 * arrow if arrow else float("nan"))
    with open(os.path.join(build, "README.md"), "w") as fh:
        fh.write(readme)

    print(f"staged {len(os.listdir(build))} files in {build}")
    print(f"  zero-shot wins vs GRU: {wins}/{len(zs)} "
          f"({100.0 * wins / len(zs):.0f}%)")
    print(f"  solar: {solar_months:.1f} months" if solar_months == solar_months
          else "  solar: n/a")
    print("To publish: huggingface-cli upload <repo> <build>")


if __name__ == "__main__":
    main()