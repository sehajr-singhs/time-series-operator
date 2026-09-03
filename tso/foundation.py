"""The scaled leg: a small *foundation operator* pretrained across domains.

The premise of the TSO vision is that a model should not learn "retail sales
physics" or "heart physics" — it should learn the *universal geometry of
fluctuations*: how a system's phase space is structured, how fast noise rides
on slow trends, how the arrow of time shows up in the shape of the attractor.

This module implements that as a real, trainable object:

    FoundationOperator
        enc : state -> Koopman latent     (learned eigencoordinates, shared)
        K   : latent -> latent            (one linear operator for ALL systems)
        dec : latent -> state             (back to physical coordinates)
        scale_map : fine-latent -> coarse-latent (renormalization-group leg)
        arrow_head : latent -> logit      (the arrow-of-time pretext)

Pretraining objectives (each a self-supervised "temporal pretext" task):

    L_recon   reconstruct the phase space
    L_dyn     one-step linear dynamics in the latent (deep Koopman)
    L_scale   *scale covariance*: the same system decimated 2x must land in
              the same latent geometry — the RG-style "how micro becomes macro"
    L_arrow   *time reversibility*: can the model tell forward from backward?

Zero-shot transfer: after pretraining, the nonlinear lift enc/dec is frozen.
For a NEVER-SEEN system we only fit the linear Koopman matrix in the latent
(closed-form least squares) and forecast in closed form. If the lift learned
universal operator structure, this beats training the same model from scratch
on the target alone.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .embedding import embed_signal, takens_embed
from .pipeline import normalize, rmse, persistence_baseline
from .bifurcation import tipping_metrics

EMBED_DIM = 5          # fixed phase-space dimension for the shared encoder
WINDOW = 128           # consecutive states per training window (long enough
                       # for the arrow-of-time signal to appear in the data)


# ---------------------------------------------------------------------------
# Corpus loading (robust CSV -> 1-D series)
# ---------------------------------------------------------------------------

def _find_column(df, hints):
    """Pick the best numeric column: exact hint match, then substring, then
    the most variable numeric column."""
    cols = list(df.columns)
    for h in hints:
        for c in cols:
            if str(c).strip().lower() == h.strip().lower():
                return c
    for h in hints:
        for c in cols:
            if h.strip().lower() in str(c).strip().lower():
                return c
    best, best_var = None, -1.0
    for c in cols:
        low = str(c).strip().lower()
        if low in ("unnamed: 0", "unnamed", "index", "id", "row", "date",
                   "time", "datetime", "timestamp"):
            continue
        try:
            v = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            continue
        if v.notna().sum() < 64:
            continue
        # skip index-like columns (0..n-1 ramps) that masquerade as signals
        vv = v.to_numpy(dtype=float)
        if np.allclose(vv[: min(len(vv), 1000)],
                       np.arange(min(len(vv), 1000))):
            continue
        var = float(v.var())
        if np.isfinite(var) and var > best_var:
            best, best_var = c, var
    return best


def load_series_from_csv(path, hints=None, max_len=4096, min_len=128,
                         name=None, detrend=True):
    """Extract one clean 1-D series from an arbitrary CSV.

    ``detrend``: a series whose lag-1 autocorrelation is ~1 (unit root — a
    random walk, an asset price) is not a well-posed Koopman target; we take
    first differences (returns), which is the stationary, linearizable signal.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"cannot read {path}: {e}")
    col = _find_column(df, hints or [])
    if col is None:
        raise ValueError(f"no usable numeric column in {path}")
    s = pd.to_numeric(df[col], errors="coerce").to_numpy()
    s = s[np.isfinite(s)]
    if len(s) < min_len:
        raise ValueError(f"{path}: only {len(s)} valid samples (< {min_len})")
    if detrend and len(s) > 64:
        ac1 = float(np.corrcoef(s[:-1], s[1:])[0, 1])
        if np.isfinite(ac1) and ac1 > 0.999:  # near unit root -> difference
            s = np.diff(s)
    if len(s) < 2 or float(np.std(s)) < 1e-6:
        raise ValueError(f"{path}: degenerate series after extraction")
    if len(s) > max_len:
        step = max(1, len(s) // max_len)
        s = s[::step][:max_len]
    s = normalize(s)
    return (name or os.path.basename(path)), s


# ---------------------------------------------------------------------------
# Multi-scale preparation (the two "temporal lenses")
# ---------------------------------------------------------------------------

def prepare_pair(series, delay_f=None, delay_c=None):
    """Fine scale: embed at native rate. Coarse scale: block-average decimate
    by 2 (a crude low-pass / renormalization step) and re-embed.

    Robust to short series: the delay is capped so the embedding always fits.
    """
    x = np.asarray(series, dtype=float)
    max_lag = min(150, max(len(x) // 6, 1))

    def safe_embed(xs, delay):
        try:
            return embed_signal(xs, delay=delay, dim=EMBED_DIM,
                                max_lag=max_lag)
        except ValueError:
            return takens_embed(xs, 1, EMBED_DIM), (1, EMBED_DIM)

    Sf, (tau_f, _) = safe_embed(x, delay_f)
    xc = x[: len(x) - len(x) % 2].reshape(-1, 2).mean(axis=1)
    Sc, (tau_c, _) = safe_embed(xc, delay_c)
    return {"fine": Sf, "fine_next": Sf[1:], "tau_f": int(tau_f),
            "coarse": Sc, "tau_c": int(tau_c), "series": x}


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class FoundationOperator(nn.Module):
    def __init__(self, state_dim=EMBED_DIM, latent_dim=24, hidden=96):
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
        # arrow-of-time head: a small temporal convolution over the latent
        # window. Pooled statistics are order-invariant (a stationary process
        # has symmetric lag-1 autocovariance), so the head must read the
        # trajectory's *direction* — exactly what a conv filter does.
        self.arrow_head = nn.Sequential(
            nn.Conv1d(latent_dim, 16, 9, padding=4), nn.Tanh(),
            nn.Conv1d(16, 1, 9, padding=4),
        )

    def phi(self, s): return self.enc(s)
    def psi(self, z): return self.dec(z)
    def step(self, z): return self.K(z)


def foundation_loss(model, pair, device="cpu", w=(1.0, 0.5, 0.5, 1.0),
                    windows_per_iter=8, joint_probe=False,
                    w_probe=0.3, w_spec=0.05, probe_steps=(2, 4, 8)):
    """Combined pretext loss on one sample: recon + linear dynamics +
    scale covariance + arrow of time, plus (optionally) the v12 joint-probe
    terms: multi-step Koopman roll-out consistency (the latent must stay
    linearly predictable h steps ahead, exactly what the zero-shot probe
    exploits) and a unit-circle penalty on unstable modes (|lambda| > 1),
    which keeps closed-form roll-outs from diverging without damping the
    physical oscillatory modes the solar discovery needs."""
    w_recon, w_dyn, w_scale, w_arrow = w
    Sf = torch.tensor(pair["fine"], dtype=torch.float32, device=device)
    Sfn = torch.tensor(pair["fine_next"], dtype=torch.float32, device=device)
    Sc = torch.tensor(pair["coarse"], dtype=torch.float32, device=device)

    # --- random windows of consecutive states (window shrinks on short
    # series so every sample in the corpus can be trained on). With the
    # joint probe the windows are pulled back by max_h so the h-step
    # targets exist inside the series. ---
    n = Sf.shape[0] - 1
    max_h = max(probe_steps) if joint_probe else 0
    w_eff = min(WINDOW, max(n - max_h - 1, 2))
    hi = max(n - w_eff - max_h, 0)
    if n - w_eff >= 1 and hi >= 1:
        starts = torch.randint(0, hi, (windows_per_iter,))
    else:
        starts = torch.zeros(windows_per_iter, dtype=torch.long)
    idx = starts[:, None] + torch.arange(w_eff)
    S = Sf[idx]                      # (Wb, W, d)
    Sp = Sfn[idx]                    # (Wb, W, d)
    z = model.phi(S)
    zp = model.phi(Sp)
    z_next = model.step(z)

    l_recon = ((model.psi(z) - S) ** 2).mean()
    l_dyn = ((model.psi(z_next) - Sp) ** 2).mean() + \
            ((z_next - zp) ** 2).mean()

    # --- v12 joint-probe terms: multi-step linear roll-out consistency ---
    l_probe = torch.zeros((), device=device)
    l_spec = torch.zeros((), device=device)
    if joint_probe:
        for h in probe_steps:
            z_cur = z
            for _ in range(h):
                z_cur = model.step(z_cur)
            Sh = Sfn[idx + h - 1]           # states at t+h
            zh = model.phi(Sh)
            l_probe = l_probe + \
                ((model.psi(z_cur) - Sh) ** 2).mean() + \
                ((z_cur - zh) ** 2).mean()
        wv, _ = torch.linalg.eig(model.K.weight)  # (K,) complex
        lam = torch.abs(wv)
        l_spec = torch.relu(lam - 1.0).square().mean()

    # --- scale covariance: pooled fine latent vs coarse latent ---
    zf = z.reshape(-1, model.latent_dim)          # flatten windows
    n2 = min(len(zf) // 2, Sc.shape[0])
    zc = model.phi(Sc)                            # coarse states lifted to latent
    pooled = (zf[: 2 * n2: 2] + zf[1: 2 * n2: 2]) / 2.0
    l_scale = ((model.scale_map(pooled) - zc[:n2]) ** 2).mean()

    # --- arrow of time: forward vs reversed windows, classified by a conv
    # head that reads the direction of the latent trajectory.
    lab_f = torch.ones(windows_per_iter, 1, device=device)
    lab_b = torch.zeros(windows_per_iter, 1, device=device)
    zt = z.transpose(1, 2)                    # (Wb, K, W) for conv
    logit_f = model.arrow_head(zt).mean(dim=2)
    logit_b = model.arrow_head(zt.flip(2)).mean(dim=2)
    l_arrow = nn.functional.binary_cross_entropy_with_logits(
        torch.cat([logit_f, logit_b]), torch.cat([lab_f, lab_b]))

    loss = w_recon * l_recon + w_dyn * l_dyn + w_scale * l_scale + \
           w_arrow * l_arrow + w_probe * l_probe + w_spec * l_spec
    with torch.no_grad():
        acc = 0.5 * ((logit_f > 0).float().mean() + (logit_b < 0).float().mean())
    return loss, {"recon": float(l_recon.detach()), "dyn": float(l_dyn.detach()),
                  "scale": float(l_scale.detach()), "arrow": float(l_arrow.detach()),
                  "arrow_acc": float(acc),
                  "probe": float(l_probe.detach()),
                  "spec": float(l_spec.detach())}


def pretrain_foundation(samples, iters=2400, lr=1e-3, latent_dim=24,
                        hidden=96, seed=0, device="cpu", print_every=300,
                        ckpt_path=None, resume=False, amp=False,
                        joint_probe=False, w_probe=0.3, w_spec=0.05,
                        dyn_w=0.5):
    """Pretrain the shared operator across all training samples.

    samples : list of prepared pairs (dicts from prepare_pair).
    amp     : mixed precision on CUDA (autocast + GradScaler).
    joint_probe : v12 mode — add multi-step Koopman roll-out consistency
        and unit-circle spectrum regularization (see foundation_loss).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FoundationOperator(latent_dim=latent_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_iter = 0
    if resume and ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_iter = int(ck["iter"]) + 1
        print(f"  resumed pretraining at iter {start_iter}")
    if not samples:
        raise ValueError("pretrain_foundation needs at least one sample")

    hist, losses = [], {k: [] for k in ("recon", "dyn", "scale", "arrow",
                                         "arrow_acc", "probe", "spec")}
    last = {k: 0.0 for k in losses}
    for it in range(start_iter, iters):
        pair = samples[int(rng.integers(0, len(samples)))]
        with torch.autocast(device_type=device, dtype=torch.float16,
                            enabled=use_amp):
            loss, parts = foundation_loss(model, pair, device=device,
                                          w=(1.0, dyn_w, 0.5, 1.0),
                                          joint_probe=joint_probe,
                                          w_probe=w_probe, w_spec=w_spec)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        hist.append(float(loss.detach()))
        for k, v in parts.items():
            losses[k].append(v); last[k] = v
        if print_every and (it % print_every == 0 or it == iters - 1):
            print(f"  pretrain iter {it:5d}: loss={float(loss.detach()):.5f} "
                  f"(recon {last['recon']:.4f} dyn {last['dyn']:.4f} "
                  f"scale {last['scale']:.4f} arrow {last['arrow']:.4f} "
                  f"arrow-acc {last['arrow_acc']:.2f})")
        if ckpt_path and (it % 600 == 0 or it == iters - 1):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "iter": it}, ckpt_path)
    model.eval()
    agg = {k: (float(np.mean(v[-200:])) if v else float("nan"))
           for k, v in losses.items()}
    return model, hist, agg, losses


# ---------------------------------------------------------------------------
# Zero-shot transfer (fit ONLY the linear Koopman matrix on the new system)
# ---------------------------------------------------------------------------

def zero_shot_forecast(model, series, train_frac=0.7, horizon_frac=0.2,
                       device="cpu", delay=None, ridge=None,
                       cap_scale=1.2, metric="persistence"):
    """Forecast a never-seen series with the frozen lift.

    1. embed the series (per-system tau, fixed dim),
    2. lift with the PRETRAINED encoder (frozen),
    3. fit the linear Koopman matrix by closed-form least squares on the
       training portion of the latent trajectory,
    4. iterate the matrix and decode — a forecast with zero gradient steps.

    Roll-out safety: each latent state is softly projected back onto the
    empirical envelope of the training latent (1.2 x 99th-percentile norm),
    so spuriously unstable fitted modes cannot drive the forecast off the
    attractor region the encoder actually saw. `window_flat` flags series
    whose test window is quasi-constant, where any skill-vs-persistence
    number degenerates into a noise race.
    """
    x = normalize(np.asarray(series, dtype=float))
    max_lag = min(150, max(len(x) // 6, 1))
    S, (tau, _) = embed_signal(x, delay=delay, dim=EMBED_DIM, max_lag=max_lag)
    n = len(S)
    split = int(n * train_frac)
    # horizon capped at 100 steps: iterating a fitted linear map hundreds of
    # steps through noise-dominated dynamics is not a meaningful test
    horizon = min(int(n * horizon_frac), n - split - 1, 100)
    with torch.no_grad():
        Z = model.phi(torch.tensor(S, dtype=torch.float32, device=device)).cpu().numpy()
    Ztr, Zte = Z[: split - 1], Z[split:]
    # rank-reduced linear probe: fit K on the top-R PCA subspace of the
    # latent (DMD-style truncation). Keeps the probe stable when the series
    # is short (covid: ~40 training rows) and filters the noise directions.
    mu = Ztr.mean(axis=0)
    Zc = Ztr - mu
    U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
    R = max(2, min(8, len(Ztr) // 16, len(s)))
    Vr = Vt[:R]                              # (R, K) PCA directions
    zr = Zc @ Vr.T                           # (n, R) reduced coordinates
    # ridge-regularized fit when requested (default: adaptive). The plain
    # least-squares Koopman fit is unstable on short/noisy latents; a small
    # ridge keeps spurious modes from exploding while physical modes
    # (|lambda| ~ 1) are unaffected.
    if ridge is None:
        ridge = 1e-1 * float(np.mean(zr ** 2))
    A = zr[:-1]
    B = zr[1:]
    AtA = A.T @ A + ridge * np.eye(R)
    Kfit_r = np.linalg.solve(AtA, A.T @ B)
    # spectral clipping: pull eigenvalues above the unit disk back so the
    # closed-form iteration cannot diverge (a fitted mode on noise can be
    # spuriously unstable; physical modes sit at |lambda| ~ 1)
    w, V = np.linalg.eig(Kfit_r)
    w = np.where(np.abs(w) > 1.02, w / np.abs(w) * 1.02, w)
    Kfit_r = (V * w[None, :]) @ np.linalg.inv(V)
    Kfit = Vr.T @ Kfit_r @ Vr                # back in full latent space

    # empirical envelope of the training latent: soft norm cap so the
    # closed-form iteration stays on the region the encoder can decode
    cap = cap_scale * float(np.percentile(
        np.linalg.norm(Ztr, axis=1), 99)) + 1e-9
    zk = Z[split - 1]
    cap_hits = 0
    z_path = [zk]
    for _ in range(horizon):
        zk = Kfit @ zk
        nrm = float(np.linalg.norm(zk))
        if nrm > cap:
            zk = zk * (cap / nrm)
            cap_hits += 1
        z_path.append(zk)
    z_path = np.array(z_path).real
    with torch.no_grad():
        pred_emb = model.psi(torch.tensor(z_path, dtype=torch.float32,
                                          device=device)).cpu().numpy()

    true_vals = S[split - 1: split + horizon, 0]
    pred_vals = pred_emb[:, 0]
    e_koop = rmse(pred_vals, true_vals)
    e_pers = rmse(persistence_baseline(true_vals[0], len(true_vals)),
                  true_vals)
    skill = 100.0 * (e_pers - e_koop) / max(e_pers, 1e-12)
    flat = bool(e_pers < 0.02 * max(float(np.std(S[:, 0])), 1e-12))

    # latent linearity on the test portion (rows = snapshots)
    zerr = np.mean(np.linalg.norm(Zte[1:] - Zte[:-1] @ Kfit.T, axis=1)) / \
        max(np.mean(np.linalg.norm(Zte[1:], axis=1)), 1e-9)

    # shape correlation: does the forecast track the curve even if the level
    # drifts? (RMSE skill vs persistence punishes smooth series brutally)
    corr = float(np.corrcoef(pred_vals, true_vals)[0, 1]) if len(true_vals) > 2 \
        else float("nan")

    tip = tipping_metrics(x)
    return {"skill_pct": skill, "skill_metric": metric, "rmse": e_koop,
            "persistence_rmse": e_pers, "window_flat": flat,
            "latent_linearity": float(zerr), "corr": corr, "tau": int(tau),
            "horizon": horizon, "cap_hits": cap_hits,
            "lambda1": tip["lambda1"], "tipping_score": tip["tipping_score"],
            "pred": pred_vals, "true": true_vals, "Kfit": Kfit}


def scratch_baseline(series, iters=700, latent_dim=24, hidden=96, seed=0,
                     device="cpu", train_frac=0.7, horizon_frac=0.2):
    """The control: the SAME architecture, but trained from scratch on the
    target series' training split only (no cross-domain pretraining)."""
    x = normalize(np.asarray(series, dtype=float))
    pair = prepare_pair(x)
    Sf = torch.tensor(pair["fine"], dtype=torch.float32, device=device)
    Sfn = torch.tensor(pair["fine_next"], dtype=torch.float32, device=device)
    n = Sf.shape[0]
    split = int(n * train_frac)
    Xtr, Ytr = Sf[: split - 1], Sfn[: split - 1]

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FoundationOperator(latent_dim=latent_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for it in range(iters):
        idx = rng.integers(0, Xtr.shape[0], size=WINDOW)
        S, Sp = Xtr[idx], Ytr[idx]
        z = model.phi(S); zp = model.phi(Sp); zn = model.step(z)
        loss = ((model.psi(z) - S) ** 2).mean() + \
               ((model.psi(zn) - Sp) ** 2).mean() + ((zn - zp) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    res = zero_shot_forecast(model, x, train_frac=train_frac,
                             horizon_frac=horizon_frac, device=device)
    res["scratch_iters"] = iters
    return res


# ---------------------------------------------------------------------------
# Shared-latent geometry (the "all attractors in one space" artwork)
# ---------------------------------------------------------------------------

def dominant_period(Kfit, dt_samples=1.0):
    """Period of the dominant oscillatory Koopman mode of a fitted latent
    operator. Used to show the operator *discovers* physical periods
    (e.g. the ~11-year solar cycle) from a zero-shot fit."""
    w = np.linalg.eigvals(Kfit)
    osc = w[np.abs(np.imag(w)) > 1e-6]
    if len(osc) == 0:
        return float("nan")
    periods = 2.0 * np.pi / np.abs(np.angle(osc))
    return float(periods[np.argmax(np.abs(osc))] * dt_samples)


def solar_cycle_discovery(model, series, coarsenings=(1, 2, 4, 8, 16, 32),
                          device="cpu"):
    """Recover the dominant physical period of a held-out series from the
    FROZEN pretrained operator, by coarsening the series (RG-style).

    The solar cycle is real but buried: at full rate, AM-modulated noise
    dominates the one-step linear fit, whose top modes are the trend and
    short noisy pairs. Coarsening the series renormalizes the fluctuation
    spectrum, turning the cycle into a nearly-sinusoidal mode that a linear
    Koopman fit locks onto. Period x coarsening is then scale-covariant,
    converging to the true physical period at the coarsest resolvable scale.

    Returns a dict with per-scale detections and the converged estimate.
    """
    x = normalize(np.asarray(series, dtype=float))
    rows = []
    for c in coarsenings:
        n = len(x) // c
        if n < 96:
            continue
        xs = x[:n * c].reshape(n, c).mean(axis=1)
        S, (_, _) = embed_signal(xs, dim=EMBED_DIM,
                                 max_lag=min(150, max(len(xs) // 6, 1)))
        n2 = len(S)
        split = int(n2 * 0.7)
        if split < 32:
            continue
        with torch.no_grad():
            Z = model.phi(torch.tensor(S, dtype=torch.float32,
                                       device=device)).cpu().numpy()
        Ztr = Z[:split - 1]
        mu = Ztr.mean(axis=0)
        Zc = Ztr - mu
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        R = max(2, min(8, len(Ztr) // 16, len(Vt)))
        Vr = Vt[:R]
        zr = Zc @ Vr.T
        ridge = 1e-1 * float(np.mean(zr ** 2))
        A, B = zr[:-1], zr[1:]
        Kfit_r = np.linalg.solve(A.T @ A + ridge * np.eye(R), A.T @ B)
        w, V = np.linalg.eig(Kfit_r)
        w = np.where(np.abs(w) > 1.02, w / np.abs(w) * 1.02, w)
        Kfit = Vr.T @ ((V * w[None, :]) @ np.linalg.inv(V)) @ Vr
        wf = np.linalg.eigvals(Kfit)
        osc = wf[np.abs(np.imag(wf)) > 1e-6]
        if len(osc) == 0:
            rows.append({"coarsening": int(c), "period_at_scale": None,
                         "period_months": None, "amp": None})
            continue
        periods = 2.0 * np.pi / np.abs(np.angle(osc))
        i = int(np.argmax(np.abs(osc)))
        rows.append({"coarsening": int(c),
                     "period_at_scale": float(periods[i]),
                     "period_months": float(periods[i] * c),
                     "amp": float(np.abs(osc[i]))})
    # converged estimate: coarsest scale with a detection (most renormalized)
    est = None
    for r in reversed(rows):
        if r["period_months"] is not None:
            est = r["period_months"]
            break
    return {"rows": rows, "period_months": est,
            "known_cycle_months": 132.0}


def gru_baseline(states, iters=500, hidden=24, lr=1e-3, train_frac=0.7,
                 horizon_frac=0.2, seed=0, device="cpu"):
    """A from-scratch GRU autoregressive token forecaster on the embedded
    states: one-step training, closed-loop roll-out. The 'sequence model'
    comparison point for the operator approach."""
    import torch.nn as nn

    class _GRU(nn.Module):
        def __init__(self, dim, h):
            super().__init__()
            self.gru = nn.GRUCell(dim, h)
            self.out = nn.Linear(h, dim)

        def forward(self, s, h=None):
            if h is None:
                h = torch.zeros(s.shape[0], self.gru.hidden_size,
                                dtype=s.dtype, device=s.device)
            h = self.gru(s, h)
            return self.out(h), h

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(states)
    split = int(n * train_frac)
    horizon = min(int(n * horizon_frac), n - split - 1, 100)
    X, Y = states[: split - 1], states[1:split]
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(Y, dtype=torch.float32, device=device)
    model = _GRU(X.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        b = 64
        idx = rng.integers(0, len(xt), size=b)
        s, t = xt[idx], yt[idx]
        opt.zero_grad()
        pred, _ = model(s)
        ((pred - t) ** 2).mean().backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        h = None
        preds = [states[split - 1]]
        s = torch.tensor(states[split - 1], dtype=torch.float32,
                         device=device)[None]
        for _ in range(horizon):
            p, h = model(s, h)
            preds.append(p.cpu().numpy()[0])
            s = p
    pred_vals = np.array(preds)[:, 0]
    true_vals = states[split - 1: split + horizon, 0]
    from .pipeline import rmse, persistence_baseline
    e = rmse(pred_vals, true_vals)
    ep = rmse(persistence_baseline(true_vals[0], len(true_vals)), true_vals)
    corr = float(np.corrcoef(pred_vals, true_vals)[0, 1]) \
        if len(true_vals) > 2 else float("nan")
    return {"skill_pct": 100.0 * (ep - e) / max(ep, 1e-12), "corr": corr,
            "horizon": horizon}


def few_shot_baseline(pretrained, series, iters=200, latent_dim=24, hidden=96,
                      seed=0, device="cpu", train_frac=0.7, horizon_frac=0.2):
    """The foundation-model move: warm-start from the pretrained operator and
    fine-tune briefly on the target's training split, then probe. Compared
    against ``scratch_baseline`` (same architecture, random init, more iters)
    this shows whether cross-domain pretraining actually transfers.
    """
    x = normalize(np.asarray(series, dtype=float))
    pair = prepare_pair(x)
    Sf = torch.tensor(pair["fine"], dtype=torch.float32, device=device)
    Sfn = torch.tensor(pair["fine_next"], dtype=torch.float32, device=device)
    n = Sf.shape[0]
    split = int(n * train_frac)
    Xtr, Ytr = Sf[: split - 1], Sfn[: split - 1]

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FoundationOperator(latent_dim=latent_dim, hidden=hidden).to(device)
    model.load_state_dict(pretrained.state_dict())   # the transfer
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for it in range(iters):
        idx = rng.integers(0, Xtr.shape[0], size=min(WINDOW, Xtr.shape[0]))
        S, Sp = Xtr[idx], Ytr[idx]
        z = model.phi(S); zp = model.phi(Sp); zn = model.step(z)
        loss = ((model.psi(z) - S) ** 2).mean() + \
               ((model.psi(zn) - Sp) ** 2).mean() + ((zn - zp) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    res = zero_shot_forecast(model, x, train_frac=train_frac,
                             horizon_frac=horizon_frac, device=device)
    res["ft_iters"] = iters
    return res


def latent_geometry(model, samples, max_pts=1200, seed=0, device="cpu"):
    """Encode every sample's phase space into the shared latent, PCA to 2D."""
    from sklearn.decomposition import PCA  # local import, corpus util only
    rng = np.random.default_rng(seed)
    all_z, labels = [], []
    for name, pair in samples:
        S = pair["fine"]
        pick = rng.choice(len(S), size=min(max_pts, len(S)), replace=False)
        with torch.no_grad():
            z = model.phi(torch.tensor(S[pick], dtype=torch.float32,
                                       device=device)).cpu().numpy()
        all_z.append(z); labels += [name] * len(z)
    Z = np.vstack(all_z)
    p = PCA(n_components=2).fit_transform(Z)
    return p, np.array(labels)
