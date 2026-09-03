"""Ground-truth dynamical systems, integrated from scratch with RK4.

These are the "physical reality" the TSO observes. The model only ever sees
one scalar channel (e.g. x), and has to reconstruct the rest of the phase
space (Takens) and the hidden dynamics (Koopman / neural field).
"""

from __future__ import annotations

import numpy as np


def rk4(f, y0, dt, n_steps):
    """Integrate dy/dt = f(y) with a 4th-order Runge-Kutta stepper.

    Parameters
    ----------
    f : callable(y) -> array
        Vector field (the differential operator of the system).
    y0 : array_like
        Initial state.
    dt : float
        Time step.
    n_steps : int
        Number of steps to take (returns n_steps + 1 states).

    Returns
    -------
    ndarray, shape (n_steps + 1, len(y0))
    """
    y = np.asarray(y0, dtype=float)
    ys = np.empty((n_steps + 1, y.size))
    ys[0] = y
    for i in range(n_steps):
        k1 = np.asarray(f(y), dtype=float)
        k2 = np.asarray(f(y + 0.5 * dt * k1), dtype=float)
        k3 = np.asarray(f(y + 0.5 * dt * k2), dtype=float)
        k4 = np.asarray(f(y + dt * k3), dtype=float)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        ys[i + 1] = y
    return ys


def lorenz_field(y, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """The Lorenz 1963 atmospheric-convection system (the butterfly)."""
    x, yv, z = y
    return np.array([
        sigma * (yv - x),
        x * (rho - z) - yv,
        x * yv - beta * z,
    ])


def rossler_field(y, a=0.2, b=0.2, c=5.7):
    """The Rossler system: an even simpler chaotic attractor (one nonlinearity)."""
    x, yv, z = y
    return np.array([-yv - z, x + a * yv, b + z * (x - c)])


def lorenz_trajectory(y0=(1.0, 1.0, 1.0), dt=0.01, n=20000, discard=1000,
                      sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """A Lorenz trajectory with the transient (approach to the attractor) removed."""
    total = n + discard
    ys = rk4(lambda y: lorenz_field(y, sigma, rho, beta), y0, dt, total)
    return ys[discard:]


def rossler_trajectory(y0=(0.0, -6.0, 0.0), dt=0.02, n=20000, discard=2000,
                       a=0.2, b=0.2, c=5.7):
    total = n + discard
    ys = rk4(lambda y: rossler_field(y, a, b, c), y0, dt, total)
    return ys[discard:]
