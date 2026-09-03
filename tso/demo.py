"""Full TSO analysis + artwork orchestration, shared by both demo scripts."""

from __future__ import annotations

import json
import os

import numpy as np

from . import neural_field, viz
from .pipeline import scale_space, tso_forecast
from .koopman import spectrum


def _corr(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_full_analysis(signal, label, out_dir, true3d=None, seed=0,
                      train_neural=True, nf_iters=1800, verbose=True):
    """End-to-end: embed -> Koopman -> neural field -> artwork -> metrics.

    Parameters
    ----------
    signal : 1-D real time series (one observed channel).
    true3d : optional (n,3) ground-truth phase space (Lorenz demo only).
    """
    os.makedirs(out_dir, exist_ok=True)
    results = {"label": label, "n": int(len(signal))}

    # --- 1. Koopman forecast (eDMD-RFF and plain DMD) ---
    if verbose:
        print(f"\n[{label}] Koopman forecast (eDMD with random-Fourier lift)")
    res = tso_forecast(signal, method="edmd", seed=seed, verbose=verbose)
    res_dmd = tso_forecast(signal, method="dmd", seed=seed, verbose=False)
    results.update({
        "tau": int(res["tau"]), "dim": int(res["dim"]),
        "edmd": {"rmse": res["rmse_koopman"], "persistence_rmse": res["rmse_persistence"],
                 "skill_pct": res["skill"]},
        "dmd": {"rmse": res_dmd["rmse_koopman"], "skill_pct": res_dmd["skill"]},
    })

    # --- 2. scale space ---
    ss = scale_space(signal, scales=(1, 2, 4, 8))
    results["scale_space"] = {f"x{s}": [round(float(m), 4) for m in v["top_magnitudes"]]
                              for s, v in ss.items()}

    # --- 3. neural vector field (learns the flow, not the next token) ---
    nf = {}
    if train_neural:
        if verbose:
            print(f"[{label}] Neural vector field (learns ds/dt on the attractor)")
        states = res["embedded"]
        subsample = max(1, len(states) // 5000)
        Ss = states[::subsample]
        dt = 1.0  # sample spacing in normalized time
        model, final_loss = neural_field.train_vector_field(
            Ss, dt, iters=nf_iters, seed=seed,
            print_every=400 if verbose else 0)
        # start integration at the last training state of the pipeline split;
        # the true continuation of the embedded series from that same state
        start_idx = int(res["split"]) - 1
        steps = min(2000, len(states) - start_idx - 1)
        x0 = states[start_idx]
        traj = neural_field.integrate_model(model, x0, dt, steps)
        true_cont = states[start_idx: start_idx + steps + 1]
        # chaos: exact orbits diverge after ~1/Lyapunov time, so measure the
        # correlation *before* divergence (h=8 steps) and the long-horizon
        # shape agreement separately
        h = min(8, steps)
        corr_short = _corr(traj[:h, 0], true_cont[:h, 0])
        corr_shape = _corr(traj[::10, 0], true_cont[::10, 0])
        nf = {"final_loss": final_loss, "short_horizon_corr": corr_short,
              "shape_corr": corr_shape, "steps": steps}
        results["neural_field"] = nf
        if verbose:
            print(f"  neural field: loss={final_loss:.5f}, "
                  f"short-horizon corr={corr_short:.3f}")

    # --- 4. artwork ---
    files = []
    def save(plot_fn, *a, **kw):
        p = os.path.join(out_dir, kw.pop("fname"))
        plot_fn(*a, p, **kw)   # path is the plot function's positional arg
        files.append(p)

    if true3d is not None:
        save(viz.plot_butterfly, true3d, fname="attractor_butterfly.png")
        n = min(len(true3d), len(res["embedded"]))
        save(viz.plot_reconstruction, true3d[:n], res["embedded"][:n], res["tau"],
             fname="takens_reconstruction.png")
    save(viz.plot_koopman_spectrum, res["model"], fname="koopman_spectrum.png")
    save(viz.plot_forecast, res, fname="forecast.png")
    save(viz.plot_scale_space, ss, fname="scale_space.png")

    if train_neural:
        save(viz.plot_learned_field,
             true3d if true3d is not None else res["embedded"],
             traj, res["tau"], fname="learned_vector_field.png")

    mp = os.path.join(out_dir, "masterpiece.png")
    viz.make_masterpiece(butterfly=true3d, embedded=res["embedded"],
                         delay=res["tau"], model=res["model"], result=res,
                         out_path=mp)
    files.append(mp)

    results["files"] = sorted(os.path.basename(f) for f in files)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return results
