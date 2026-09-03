"""Production-style training loop for the TSO.

The research scripts above train in one flat loop; this module is the leg that
scales: batched sampling, automatic mixed precision (autocast + GradScaler on
CUDA, transparent fp32 fallback on CPU), checkpointing with resume, and a
*continuous* Neural-ODE-style query API — ask for the model's state at ANY
time t, not just at the grid of observed ticks (``continuous_flow``).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch


def pick_device(prefer="cuda"):
    """cuda if available, else mps (Apple), else cpu."""
    if prefer == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AmpContext:
    """Mixed-precision wrapper: real AMP on CUDA, no-op elsewhere.

    Usage::

        ctx = AmpContext(device)
        for it in ...:
            with ctx.autocast():
                loss = compute_loss(model, batch)
            ctx.step(opt, loss)
    """

    def __init__(self, device, enabled=True):
        self.device = device
        self.enabled = enabled and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled)

    def autocast(self):
        return torch.autocast(device_type=self.device, dtype=torch.float16,
                              enabled=self.enabled)

    def step(self, opt, loss):
        self.scaler.scale(loss).backward()
        self.scaler.step(opt)
        self.scaler.update()


def train_batched(model, X, Y, iters=2000, batch=256, lr=1e-3, seed=0,
                  device="cpu", amp=True, ckpt_path=None, resume=False,
                  print_every=500, track_history=True, extra_loss=None):
    """Generic batched training with AMP + checkpointing + resume.

    Parameters
    ----------
    X, Y       : (n, d) torch tensors — snapshot pairs (state, next state)
    extra_loss : optional callable(model, s, s_next, loss) -> loss to add
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    model = model.to(device)
    X, Y = X.to(device), Y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    amp_ctx = AmpContext(device, enabled=amp)

    start_iter, history = 0, []
    if resume and ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_iter = int(ck["iter"]) + 1
        history = list(ck.get("history", []))
        print(f"  [train_loop] resumed at iter {start_iter} from {ckpt_path}")

    t0 = time.perf_counter()
    last = float("nan")
    for it in range(start_iter, iters):
        idx = rng.integers(0, n, size=batch)
        s, s_next = X[idx], Y[idx]
        opt.zero_grad()
        with amp_ctx.autocast():
            loss = ((model(s) - s_next) ** 2).mean()
            if extra_loss is not None:
                loss = loss + extra_loss(model, s, s_next)
        amp_ctx.step(opt, loss)
        last = float(loss.item())
        if track_history:
            history.append(last)
        if print_every and (it % print_every == 0 or it == iters - 1):
            print(f"  [train_loop] iter {it:5d}: loss={last:.6f}")
        if ckpt_path and (it % 500 == 0 or it == iters - 1):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "iter": it, "history": history}, ckpt_path)

    wall = time.perf_counter() - t0
    model.eval()
    return model, history, {"device": device, "amp": amp_ctx.enabled,
                            "iters": iters, "wall_s": round(wall, 2),
                            "ms_per_iter": round(1000.0 * wall / max(iters - start_iter, 1), 3)}


def continuous_flow(model, x0, t_eval, rk4_dt=None, device="cpu"):
    """Neural-ODE-style *continuous querying* along the learned flow.

    Instead of only producing states at integer ticks, this integrates the
    learned vector field with fine RK4 substeps and then evaluates the
    trajectory at the *arbitrary* times requested in ``t_eval`` — the model
    has an answer for every instant, not just the observed sampling grid.

    ``model`` here is a callable vector field (like the neural_field module's
    VectorField, or any f_theta(s) -> ds/dt).
    """
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=float))
    t_max = float(np.max(t_eval))
    if rk4_dt is None:
        rk4_dt = max(t_max / 200.0, 1e-3)
    n_sub = max(2, int(np.ceil(t_max / rk4_dt)))
    steps = np.linspace(0.0, t_max, n_sub + 1)

    y = torch.tensor(np.asarray(x0, dtype=np.float32), device=device)
    traj = [y.cpu().numpy().copy()]
    with torch.no_grad():
        for i in range(n_sub):
            dt = steps[i + 1] - steps[i]
            def field(v):
                return model(v)
            k1 = field(y)
            k2 = field(y + 0.5 * dt * k1)
            k3 = field(y + 0.5 * dt * k2)
            k4 = field(y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            traj.append(y.cpu().numpy().copy())
    traj = np.array(traj)
    # dense output: linear interpolation onto the requested times
    out = np.empty((len(t_eval), traj.shape[1]))
    for i, t in enumerate(t_eval):
        frac = t / max(t_max, 1e-12) * n_sub
        i0 = min(int(np.floor(frac)), n_sub)
        i1 = min(i0 + 1, n_sub)
        a = frac - i0
        out[i] = (1.0 - a) * traj[i0] + a * traj[i1]
    return out


def benchmark_loop(states, iters=300, batch=128, device="auto", out_dir=None,
                   label="bench"):
    """Wall-clock benchmark of the batched AMP loop (CPU here, GPU in kernel).

    Fits a tiny one-layer net to predict s_{k+1} from s_k — purely a
    throughput probe of the training machinery.
    """
    device = pick_device(device)
    X = torch.tensor(states[:-1], dtype=torch.float32)
    Y = torch.tensor(states[1:], dtype=torch.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 64), torch.nn.Tanh(),
        torch.nn.Linear(64, X.shape[1]))
    model, hist, info = train_batched(model, X, Y, iters=iters, batch=batch,
                                      device=device, amp=True, print_every=0)
    out = {"label": label, **info,
           "final_loss": float(hist[-1]) if hist else None}
    print(f"  [bench] device={info['device']} amp={info['amp']} "
          f"{info['ms_per_iter']:.2f} ms/iter over {iters} iters")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "train_loop_bench.json"), "w") as fh:
            json.dump(out, fh, indent=2)
    return out
