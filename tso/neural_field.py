"""Learning the attractor's vector field with a small neural ODE-style net.

The TSO's second leg: a neural field f_theta(s) that approximates the
differential operator ds/dt of the reconstructed phase space. Once learned,
the model is no longer a "next-token predictor" — it is a vector field, and
any trajectory can be produced by integrating it (RK4, from scratch).

Training signal: finite-difference derivatives of the observed (embedded)
trajectory, plus a multi-step consistency term so the field agrees with the
actual flow over a short horizon.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class VectorField(nn.Module):
    """MLP mapping phase-space state -> instantaneous velocity."""

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, s):
        return self.net(s)


def train_vector_field(states, dt, iters=2500, lr=1e-3, batch=256,
                       consistency_steps=4, seed=0, device="cpu",
                       print_every=500):
    """Fit f_theta on an embedded trajectory.

    states : (T, d) Takens-embedded trajectory (already normalized).

    Returns (model, final_loss).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X = torch.tensor(states[:-1], dtype=torch.float32, device=device)
    dX = torch.tensor(np.diff(states, axis=0) / dt,
                      dtype=torch.float32, device=device)
    n = X.shape[0]
    dim = X.shape[1]

    model = VectorField(dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    last_loss = float("nan")
    for it in range(iters):
        idx = rng.integers(0, n, size=batch)
        s = X[idx]
        target = dX[idx]
        # derivative-matching loss
        loss = ((model(s) - target) ** 2).mean()
        # multi-step Euler consistency: f must integrate correctly over a few ticks
        if consistency_steps > 0:
            s_k = s.clone()
            step_loss = 0.0
            for _ in range(consistency_steps):
                s_k = s_k + dt * model(s_k)
                step_loss = step_loss + ((s_k - X[(idx + 1 + _) % n]) ** 2).mean()
            loss = 0.5 * loss + 0.5 * step_loss / consistency_steps
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        if print_every and (it % print_every == 0 or it == iters - 1):
            print(f"  vector-field iter {it:5d}: loss={last_loss:.6f}")

    model.eval()
    return model, last_loss


def integrate_model(model, x0, dt, steps):
    """RK4 integration along the learned field — a forecast as pure geometry.

    This is where the model "runs the physics": it never looks up a pattern,
    it rides the vector field.
    """
    x0 = np.asarray(x0, dtype=np.float32)
    with torch.no_grad():
        traj = np.empty((steps + 1, x0.size), dtype=np.float32)
        traj[0] = x0
        y = torch.tensor(x0)
        for i in range(steps):
            def field(v):
                return model(v[None]).squeeze(0)
            k1 = field(y)
            k2 = field(y + 0.5 * dt * k1)
            k3 = field(y + 0.5 * dt * k2)
            k4 = field(y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            traj[i + 1] = y.numpy()
    return traj


def integrate_vectorized(model, x0, dt, steps):
    """Same as integrate_model but operates on full trajectories (RK4, batched)."""
    x0 = np.asarray(x0, dtype=np.float32)
    with torch.no_grad():
        y = torch.tensor(x0)
        traj = [y.numpy().copy()]
        for i in range(steps):
            def field(v):
                return model(v)
            k1 = field(y)
            k2 = field(y + 0.5 * dt * k1)
            k3 = field(y + 0.5 * dt * k2)
            k4 = field(y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            traj.append(y.numpy().copy())
    return np.array(traj)
