#!/usr/bin/env python3
"""Synthetic end-to-end TSO demo on the Lorenz butterfly.

The model sees ONE scalar channel (x, with 1% sensor noise) and has to
reconstruct the phase space, linearize the chaos (Koopman) and learn the flow
(neural field) — then we draw the artwork.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tso import attractors
from tso.demo import run_full_analysis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "lorenz")


def main():
    print("=" * 70)
    print("TSO synthetic demo: the Lorenz attractor (chaotic convection)")
    print("=" * 70)

    rng = np.random.default_rng(7)
    true = attractors.lorenz_trajectory(n=20000, dt=0.01)          # ground truth
    x_obs = true[:, 0] + 0.01 * rng.normal(size=true.shape[0])     # one noisy channel

    results = run_full_analysis(x_obs, "lorenz-x", OUT, true3d=true,
                                train_neural=True, nf_iters=2000)

    print("\n" + "=" * 70)
    print("RESULTS — Lorenz demo")
    print("=" * 70)
    print(f"  observed: 1 channel of a 3-D chaotic system ({len(x_obs)} samples)")
    print(f"  Takens:   tau={results['tau']}, dim={results['dim']}")
    print(f"  eDMD-RFF: RMSE={results['edmd']['rmse']:.4f}  "
          f"skill over persistence={results['edmd']['skill_pct']:+.1f}%")
    print(f"  plain DMD: RMSE={results['dmd']['rmse']:.4f}  "
          f"skill={results['dmd']['skill_pct']:+.1f}%")
    nf = results.get("neural_field", {})
    if nf:
        print(f"  neural field: loss={nf['final_loss']:.5f}, "
              f"short-horizon corr={nf['short_horizon_corr']:.3f}")
    print(f"  scale space top |lambda|: {results['scale_space']}")
    print(f"\n  artwork written to: {OUT}/")
    for f in results["files"]:
        print(f"    - {f}")


if __name__ == "__main__":
    main()
