"""Koopman operator theory, implemented from scratch in NumPy.

The core idea of the TSO: instead of predicting the next value of a chaotic
signal (hopeless — the butterfly effect guarantees divergence), we *lift* the
state into a higher-dimensional feature space where the dynamics become (nearly)
linear, i.e. where the flow is governed by a matrix that we can diagonalize.

    x_{k+1} ~= K phi(x_k)          (linear, in lifted space)

Two constructions:

* ``exact_dmd``   — the classic Dynamic Mode Decomposition: find the best
                    linear map X -> Y via SVD, then eigen-decompose it.
* ``edmd_rff``    — Extended DMD with random Fourier features. This is the
                    "Koopman lift": phi(x) = [x, cos(Wx+c), sin(Wx+c)], so a
                    chaotic twist is unfolded into a linear flow whose
                    eigenvalues are directly the frequencies of the motion.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Linear Koopman operator (exact DMD)
# --------------------------------------------------------------------------

def exact_dmd(X, Y, rank=None):
    """Best linear operator K = argmin ||Y - K X||_F, diagonalized.

    Parameters
    ----------
    X : (m, d)  snapshots at time k (rows = snapshots, cols = state dims)
    Y : (m, d)  snapshots at time k+1
    rank : int or None — truncation of the SVD (noise filter)

    Returns a dict with the Koopman eigenvalues, modes, amplitudes and the
    linear operator itself.
    """
    # classic DMD convention: columns are snapshots -> transpose
    Xt, Yt = X.T, Y.T
    U, s, Vt = np.linalg.svd(Xt, full_matrices=False)
    s0 = s[0] if s.size else 1.0
    r = rank if rank is not None else int(np.count_nonzero(s > 1e-10 * s0))
    r = max(1, min(r, s.size))
    U, s, Vt = U[:, :r], s[:r], Vt[:r]

    # reduced operator acting on the SVD coordinates
    A_tilde = U.T @ Yt @ Vt.T @ np.diag(1.0 / s)

    # eigendecomposition: eigenvalues = growth/decay + rotation rates
    w, W = np.linalg.eig(A_tilde)

    # exact DMD modes in the original state space, and their amplitudes b
    Phi = Yt @ Vt.T @ np.diag(1.0 / s) @ W
    b = np.linalg.lstsq(Phi, Xt[:, 0], rcond=None)[0]

    return {
        "kind": "dmd",
        "eigenvalues": w,
        "modes": Phi,
        "amplitudes": b,
        "operator": U @ A_tilde @ U.T,
        "rank": r,
        "singular_values": s,
    }


# --------------------------------------------------------------------------
# Nonlinear lift (extended DMD with random Fourier features)
# --------------------------------------------------------------------------

def _rff_lift_factory(W, c, scale):
    """phi(x) = [x, cos(Wx+c)/s, sin(Wx+c)/s] — identity is coordinate 0..d-1."""
    def lift(x):
        x = np.atleast_2d(x)
        feats = np.cos(x @ W + c) / scale
        feats = np.concatenate([feats, np.sin(x @ W + c) / scale], axis=1)
        return np.concatenate([x, feats], axis=1)
    return lift


def edmd_rff(X, Y, lift_dim=128, sigma=1.0, rank=None, seed=0):
    """Extended DMD with a random-Fourier Koopman lift.

    The chaotic state is embedded in a higher-dimensional feature space where
    the evolution operator is fit linearly (exact DMD on the lifted snapshots).
    Because the lift contains the identity, forecasting the first d coordinates
    of the lifted state IS the forecast of the physical state.
    """
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, sigma, size=(d, lift_dim))
    c = rng.uniform(0.0, 2.0 * np.pi, size=lift_dim)
    scale = np.sqrt(lift_dim)

    lift = _rff_lift_factory(W, c, scale)
    Z, Zp = lift(X), lift(Y)

    model = exact_dmd(Z, Zp, rank=rank)
    model["kind"] = "edmd-rff"
    model["lift"] = lift
    model["state_dim"] = d
    return model


# --------------------------------------------------------------------------
# Forecasting with a Koopman model
# --------------------------------------------------------------------------

def forecast(model, x0, steps):
    """Propagate x0 forward `steps` steps along the (linearized) Koopman flow.

    x_k = Phi * diag(mu^k) * b   =>   closed-form, no iterative integration.
    """
    w = model["eigenvalues"]
    Phi = model["modes"]
    b = model["amplitudes"]

    if "lift" in model:
        x0 = model["lift"](np.atleast_2d(x0))[0]

    # powers mu^k for k = 0..steps, shape (rank, steps+1)
    k = np.arange(steps + 1)
    mu_pow = w[:, None] ** k[None, :]
    z = Phi @ (b[:, None] * mu_pow)

    if "lift" in model:
        return z[: model["state_dim"], :].real.T
    return z.real.T


def spectrum(model):
    """Return (eigenvalues, magnitudes) for plotting the Koopman spectrum."""
    w = model["eigenvalues"]
    return w, np.abs(w)
