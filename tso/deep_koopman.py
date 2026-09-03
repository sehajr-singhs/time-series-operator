"""Deep Koopman eigenfunction learning (from scratch).

The RFF lift in ``koopman.py`` is a *fixed* nonlinearity: it cannot adapt to
the system it observes. The deep-Koopman leg learns the lift itself. An
autoencoder

    encoder  phi : state  -> latent      (the learned Koopman eigencoordinates)
    linear   K   : latent -> latent      (the linearized flow, shared operator)
    decoder  psi : latent -> state       (back to physical coordinates)

is trained so that in the latent space the dynamics are *exactly* linear:

    psi(K phi(x_k)) ~= x_{k+1}           (one-step consistency)
    K phi(x_k)     ~= phi(x_{k+1})       (linearity of the lift)

Once trained, forecasting is closed form again — iterate the learned matrix —
but now the coordinates were *discovered from the data's own geometry* instead
of being a hand-picked Fourier lift. We benchmark both lifts on the Lorenz
butterfly: linearity error (how linear the flow is in the latent space) and
forecast skill over persistence.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from .pipeline import rmse, persistence_baseline, normalize
from .embedding import embed_signal
from .koopman import edmd_rff, forecast as koopman_forecast


class KoopmanAE(nn.Module):
    """phi / K / psi — the learned Koopman eigenfunction decomposition.

    Parameters
    ----------
    state_dim  : physical (embedded) state dimension
    latent_dim : dimension of the learned Koopman coordinates
    hidden     : width of the MLP layers
    """

    def __init__(self, state_dim, latent_dim=24, hidden=96):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
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
        # the shared linear Koopman operator (bias-free by construction)
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

    # -- forward pieces ----------------------------------------------------
    def phi(self, s):
        return self.enc(s)

    def psi(self, z):
        return self.dec(z)

    def step(self, z):
        """Advance one step in the latent Koopman coordinates."""
        return self.K(z)

    def forward(self, s):
        """(recon, next_recon, z_next): the three quantities used by losses."""
        z = self.phi(s)
        z_next = self.step(z)
        return self.psi(z), self.psi(z_next), z_next


def train_deep_koopman(states, iters=2500, batch=256, lr=1e-3,
                       latent_dim=24, hidden=96, seed=0, device="cpu",
                       print_every=500, ckpt_path=None, resume=False,
                       track_history=True, consistency_steps=3):
    """Fit a KoopmanAE on an embedded trajectory.

    Loss = reconstruction + one-step dynamics + latent linearity + multi-step
    consistency (the operator must stay accurate when iterated, which is what
    forecasting actually does):

        L = ||psi(phi(s)) - s||^2
          + ||psi(K phi(s)) - s'||^2          (predict the next physical state)
          + ||K phi(s) - phi(s')||^2          (the lift must make flow linear)
          + ||psi(K^j phi(s)) - s_{t+j}||^2   (unrolled consistency, j=1..C)

    Returns (model, loss_history) where loss_history is a list of the total
    loss at each iteration (empty if track_history is False).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X = torch.tensor(states[:-1], dtype=torch.float32, device=device)
    Y = torch.tensor(states[1:], dtype=torch.float32, device=device)
    n = X.shape[0]
    dim = X.shape[1]

    model = KoopmanAE(dim, latent_dim=latent_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start_iter, history = 0, []

    if resume and ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_iter = int(ck["iter"]) + 1
        history = list(ck.get("history", []))
        print(f"  resumed from {ckpt_path} at iter {start_iter}")

    def step_batch(idx):
        s, s_next = X[idx], Y[idx]
        recon, next_recon, z_next = model(s)
        loss = ((recon - s) ** 2).mean()
        loss = loss + ((next_recon - s_next) ** 2).mean()
        loss = loss + ((z_next - model.phi(s_next)) ** 2).mean()
        # unroll the linear operator: it must predict s at +1..+C steps
        if consistency_steps > 0:
            zk = z_next
            unrolled = 0.0
            for j in range(1, consistency_steps + 1):
                s_target = X[(idx + j) % n]
                unrolled = unrolled + ((model.psi(zk) - s_target) ** 2).mean()
                zk = model.step(zk)
            loss = loss + 0.5 * unrolled / consistency_steps
        return loss

    hi = n - consistency_steps - 1
    last = float("nan")
    for it in range(start_iter, iters):
        idx = rng.integers(0, max(hi, 1), size=batch)
        opt.zero_grad()
        loss = step_batch(idx)
        loss.backward()
        opt.step()
        last = float(loss.item())
        if track_history:
            history.append(last)
        if print_every and (it % print_every == 0 or it == iters - 1):
            print(f"  deep-Koopman iter {it:5d}: loss={last:.6f}")
        if ckpt_path and (it % 500 == 0 or it == iters - 1):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "iter": it, "history": history}, ckpt_path)

    model.eval()
    return model, history


def linearity_error(model, X, Y, device="cpu"):
    """Mean relative one-step error of the linear law in the latent space:

        e = mean(||K phi(x) - phi(y)||) / mean(||phi(y)||)
    """
    with torch.no_grad():
        z = model.phi(torch.tensor(X, dtype=torch.float32, device=device))
        zp = model.phi(torch.tensor(Y, dtype=torch.float32, device=device))
        err = (model.step(z) - zp).norm(dim=1).mean()
        scale = zp.norm(dim=1).mean()
    return float(err / max(float(scale), 1e-9))


def deep_koopman_forecast(model, s0, steps, device="cpu"):
    """Closed-form forecast by iterating the *learned* linear operator."""
    z = model.phi(torch.tensor(np.atleast_2d(s0), dtype=torch.float32,
                               device=device))
    out = []
    with torch.no_grad():
        zk = z
        for _ in range(steps + 1):
            out.append(model.psi(zk).cpu().numpy()[0])
            zk = model.step(zk)
    return np.array(out)


def benchmark_vs_rff(states, delay, train_frac=0.7, iters=3000, seed=0,
                     device="cpu", latent_dim=48, horizon=None, out_dir=None,
                     label="benchmark", verbose=True):
    """Head-to-head: learned Koopman coordinates vs the fixed RFF lift.

    Both are fit on the first ``train_frac`` of the embedded trajectory and
    forecast the rest. Metrics:

      linearity_err  : one-step linearization error in the respective latent
      skill_pct      : % RMSE improvement over persistence on the test horizon
    """
    n = len(states)
    split = int(n * train_frac)
    Xtr, Ytr = states[: split - 1], states[1:split]
    horizon = horizon if horizon is not None else n - split
    x0 = states[split - 1]
    true_vals = states[split - 1: split + horizon, 0]

    # --- fixed RFF lift (current TSO) ---
    rff = edmd_rff(Xtr, Ytr, lift_dim=128, seed=seed)
    pred_rff = koopman_forecast(rff, x0, horizon)[:, 0]
    skill_rff = 100.0 * (rmse(persistence_baseline(x0[0], horizon + 1),
                              true_vals) - rmse(pred_rff, true_vals)) \
        / max(rmse(persistence_baseline(x0[0], horizon + 1), true_vals), 1e-12)
    # linearity of the fixed lift (identity + cos/sin features), measured on
    # HELD-OUT snapshot pairs (the fit pairs would be ~0 by construction):
    # both lifts are judged on dynamics they never saw.
    Xte, Yte = states[split - 1: n - 1], states[split: n]
    z_rff = rff["lift"](Xte)
    zp_rff = rff["lift"](Yte)
    err_rff = float(np.mean(np.linalg.norm(zp_rff - z_rff @ rff["operator"].T,
                                           axis=1)) /
                    max(np.mean(np.linalg.norm(zp_rff, axis=1)), 1e-9))

    # --- learned lift (deep Koopman): trained on the SAME split as the RFF
    # model so the comparison is fair (neither sees the test horizon) ---
    model, hist = train_deep_koopman(states[:split], iters=iters, seed=seed,
                                     latent_dim=latent_dim, device=device,
                                     print_every=0, consistency_steps=4)
    pred_dk = deep_koopman_forecast(model, x0, horizon, device=device)[:, 0]
    skill_dk = 100.0 * (rmse(persistence_baseline(x0[0], horizon + 1),
                             true_vals) - rmse(pred_dk, true_vals)) \
        / max(rmse(persistence_baseline(x0[0], horizon + 1), true_vals), 1e-12)
    err_dk = linearity_error(model, Xte, Yte, device=device)

    better_lin = "learned" if err_dk < err_rff else "RFF"
    out = {
        "label": label,
        "horizon": int(horizon),
        "rff": {"linearity_err": round(err_rff, 5), "skill_pct": round(skill_rff, 2),
                "lift_dim": 2 * 128 + 5},
        "deep": {"linearity_err": round(err_dk, 5), "skill_pct": round(skill_dk, 2),
                 "final_loss": float(hist[-1]) if hist else None,
                 "lift_dim": latent_dim},
        "comparison": {
            "more_linear": better_lin,
            "linearity_ratio": round(max(err_rff, err_dk) / max(min(err_rff, err_dk), 1e-12), 2),
            "skill_pts": round(skill_dk - skill_rff, 2),
        },
    }
    if verbose:
        print(f"  [{label}] fixed RFF lift : linearity err={err_rff:.4f} (held-out), "
              f"skill={skill_rff:+.1f}%")
        print(f"  [{label}] learned lift   : linearity err={err_dk:.4f} (held-out), "
              f"skill={skill_dk:+.1f}%")
        print(f"  -> {better_lin} lift is {out['comparison']['linearity_ratio']:.2f}x more "
              f"linear; skill delta {out['comparison']['skill_pts']:+.1f} pts")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "deep_koopman_benchmark.json"), "w") as fh:
            json.dump(out, fh, indent=2)
    return out
