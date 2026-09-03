#!/usr/bin/env python3
"""TSO scaled leg: multi-domain corpus (via the legacy Kaggle API) ->
deep-Koopman eigenfunction learning -> bifurcation detector -> GPU training
loop benchmark -> foundation pretraining -> zero-shot transfer -> artwork.

Run (local, CPU):

    python scripts/pretrain_foundation.py --iters 2400
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from tso import attractors, viz  # noqa: E402
from tso.foundation import (  # noqa: E402
    prepare_pair, load_series_from_csv, pretrain_foundation,
    zero_shot_forecast, scratch_baseline, few_shot_baseline, latent_geometry,
)
from tso.deep_koopman import benchmark_vs_rff  # noqa: E402
from tso.bifurcation import (  # noqa: E402
    bifurcation_sweep, detect_tipping, tipping_metrics,
)
from tso.train_loop import benchmark_loop, pick_device  # noqa: E402
from tso.embedding import embed_signal  # noqa: E402
from tso.pipeline import normalize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "corpus")
OUT = os.path.join(ROOT, "output", "foundation")

# ---------------------------------------------------------------------------
# Corpus specification (each a distinct physical/economic/physiological domain)
# ---------------------------------------------------------------------------

CORPUS = [
    # name, kaggle ref (None = local/synthetic), filename glob, column hints,
    # domain, held_out
    ("ecg-hrv", "sumit042004/cardiac-arrhythmia-ecg-dataset-mit-bih",
     "Cardiac_arrhythmia_dataset.csv", ["rr_prev"], "physiology", False),
    ("grid-energy", "robikscube/hourly-energy-consumption",
     "AEP_hourly.csv", ["AEP_MW"], "energy-grid", False),
    ("weather-aus", "jsphyg/weather-dataset-rattle-package",
     "weatherAUS.csv", ["Temp3pm"], "meteorology", False),
    ("bitcoin", "sudalairajkumar/cryptocurrencypricehistory",
     "coin_Bitcoin.csv", ["Close"], "finance", False),
    ("sunspots", "robervalt/sunspots",
     "Sunspots.csv", ["Sunspot Number", "Sunspot"], "solar-physics", True),
    ("airline", "rakannimer/air-passengers",
     "AirPassengers.csv", ["#Passengers"], "economics", False),
    ("covid-us", "imdevskp/corona-virus-report",
     "covid_19_clean_complete.csv", ["Confirmed"], "epidemiology", True),
]


def kaggle_download(ref, dest):
    """Legacy Kaggle API download (authenticated CLI)."""
    os.makedirs(dest, exist_ok=True)
    print(f"  downloading {ref} (legacy API) ...")
    proc = subprocess.run(["kaggle", "datasets", "download", "-d", ref,
                           "-p", dest, "--unzip"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"kaggle download failed for {ref}: "
                           f"{proc.stderr[-600:]}")
    print(f"  -> {ref} downloaded")


def locate_file(dest, glob_name):
    if os.path.exists(os.path.join(dest, glob_name)):
        return os.path.join(dest, glob_name)
    hits = [os.path.join(dest, f) for f in os.listdir(dest)
            if f.lower().endswith(".csv")]
    return hits[0] if hits else None


def load_covid(path, max_len=4096, min_len=128):
    """Daily new confirmed cases for the US — a real epidemic curve."""
    import pandas as pd
    df = pd.read_csv(path)
    if "Country/Region" in df.columns:
        df = df[df["Country/Region"] == "US"]
    if df.empty:
        df = pd.read_csv(path)
    s = pd.to_numeric(df["Confirmed"], errors="coerce").dropna().to_numpy()
    s = np.diff(s)                      # new cases per day
    s = s[np.isfinite(s)]
    if len(s) < min_len:
        raise ValueError("covid series too short")
    if len(s) > max_len:
        s = s[:: len(s) // max_len][:max_len]
    return "covid-us", normalize(s)


def build_corpus(skip_download=False, verbose=True):
    """Download (if needed) and prepare every sample -> list of pairs."""
    samples, meta = [], []
    for name, ref, fglob, hints, domain, held in CORPUS:
        try:
            if name == "ecg-hrv":
                path = os.path.join(ROOT, "data",
                                    "Cardiac_arrhythmia_dataset.csv")
                if not os.path.exists(path):
                    kaggle_download(ref, os.path.join(ROOT, "data"))
            elif name == "lorenz":
                continue  # added below, synthetic
            else:
                dest = os.path.join(DATA, name)
                has_csv = (os.path.isdir(dest) and any(
                    f.lower().endswith(".csv") for f in os.listdir(dest)))
                if not has_csv and not skip_download:
                    kaggle_download(ref, dest)
                path = locate_file(dest, fglob)
            if name == "covid-us":
                lbl, s = load_covid(path)
            else:
                lbl, s = load_series_from_csv(path, hints=hints, name=name)
            if verbose:
                print(f"  corpus + {lbl:14s} [{domain:12s}] "
                      f"len={len(s):5d}  held_out={held}")
            pair = prepare_pair(s)
            samples.append((lbl, pair))
            meta.append({"name": lbl, "domain": domain, "held_out": held,
                         "n": int(len(s)), "tau_f": pair["tau_f"],
                         "tau_c": pair["tau_c"]})
        except Exception as e:
            print(f"  !! skipped {name}: {e}")
    # synthetic physics domain: the Lorenz butterfly's x channel
    rng = np.random.default_rng(7)
    true = attractors.lorenz_trajectory(n=20000, dt=0.01)
    s = true[:, 0] + 0.01 * rng.normal(size=true.shape[0])
    s = normalize(s)
    pair = prepare_pair(s)
    samples.append(("lorenz-x", pair))
    meta.append({"name": "lorenz-x", "domain": "physics", "held_out": False,
                 "n": len(s), "tau_f": pair["tau_f"], "tau_c": pair["tau_c"]})
    return samples, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2400)
    ap.add_argument("--dk-iters", type=int, default=2200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-dk", action="store_true",
                    help="skip the deep-Koopman benchmark phase")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="skip the bifurcation sweep phase")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    print(f"device = {device}")

    print("=" * 70)
    print("TSO scaled leg — multi-domain corpus -> foundation operator")
    print("=" * 70)

    # --- corpus -----------------------------------------------------------
    print("\n[1/6] corpus (legacy Kaggle API)")
    samples, meta = build_corpus(skip_download=args.skip_download)
    held = [m for m in meta if m["held_out"]]
    print(f"  {len(samples)} samples, {len(held)} held out for zero-shot: "
          f"{[m['name'] for m in held]}")

    # --- deliverable 1: deep Koopman vs fixed RFF lift ---------------------
    print("\n[2/6] deep-Koopman eigenfunction learning vs the RFF lift")
    true = attractors.lorenz_trajectory(n=20000, dt=0.01)
    x_obs = true[:, 0]
    S, (tau, _) = embed_signal(normalize(x_obs), dim=5)
    dk_bench = {}
    if not args.skip_dk:
        dk_bench = benchmark_vs_rff(S, tau, iters=args.dk_iters, seed=0,
                                    device=device, out_dir=args.out,
                                    label="lorenz-deep-koopman", verbose=True)
    else:
        dk_bench = json.load(open(os.path.join(args.out,
                                               "deep_koopman_benchmark.json")))

    # --- deliverable 2: bifurcation / tipping detector ---------------------
    print("\n[3/6] bifurcation detector (Lorenz rho sweep, one channel)")
    rho_vals = np.concatenate([
        np.arange(1.0, 24.0, 2.0),
        np.arange(24.0, 30.0, 0.5),
        np.arange(30.0, 41.0, 2.0),
    ])
    if not args.skip_sweep:
        sweep = bifurcation_sweep(rho_vals)
        detected_rho, thr = detect_tipping(sweep)
        with open(os.path.join(args.out, "bifurcation.json"), "w") as fh:
            json.dump({"detected_rho": detected_rho, "threshold": thr,
                       "sweep": sweep}, fh, indent=2)
    else:
        bj = json.load(open(os.path.join(args.out, "bifurcation.json")))
        sweep, detected_rho, thr = bj["sweep"], bj["detected_rho"], bj["threshold"]
    print(f"  detected tipping at rho ~ {detected_rho:.1f} "
          f"(theoretical chaos onset ~ 24.7)")

    # --- deliverable 3: GPU training loop benchmark ------------------------
    print("\n[4/6] production training loop (batched + AMP + checkpointing)")
    bench = None
    if not args.skip_bench:
        bench = benchmark_loop(S, iters=250, device=device, out_dir=args.out,
                               label=f"lorenz-{device}")

    # --- deliverable 4: foundation pretraining + zero-shot transfer --------
    print("\n[5/6] foundation pretraining across domains")
    train_pairs = [(n, p) for (n, p), m in zip(samples, meta)
                   if not m["held_out"]]
    model, hist, parts_agg, parts = pretrain_foundation(
        [p for _, p in train_pairs], iters=args.iters, seed=0, device=device,
        ckpt_path=os.path.join(args.out, "foundation_model.pt"))
    torch.save(model.state_dict(), os.path.join(args.out, "foundation_model.pt"))
    viz.plot_pretrain_curves(hist, parts,
                             os.path.join(args.out, "pretrain_curves.png"))

    print("\n[6/6] zero-shot transfer on never-seen systems")
    zs_results, scratch_results, fewshot_results = {}, {}, {}
    for n, p in samples:
        name = n
        res = zero_shot_forecast(model, p["series"], device=device)
        zs_results[name] = {k: v for k, v in res.items()
                            if not isinstance(v, np.ndarray)}
        print(f"  frozen   {name:14s}: skill={res['skill_pct']:+6.1f}%  "
              f"corr={res['corr']:+.2f}  tau={res['tau']}  "
              f"lambda1={res['lambda1']:.3f}")
    for m in held:
        name = m["name"]
        pair = dict(samples)[name]
        scratch = scratch_baseline(pair["series"], iters=700, seed=0,
                                   device=device)
        scratch_results[name] = {k: v for k, v in scratch.items()
                                 if not isinstance(v, np.ndarray)}
        print(f"  scratch  {name:14s}: skill={scratch['skill_pct']:+6.1f}%  "
              f"corr={scratch['corr']:+.2f}")
        few = few_shot_baseline(model, pair["series"], iters=200, seed=0,
                                device=device)
        fewshot_results[name] = {k: v for k, v in few.items()
                                 if not isinstance(v, np.ndarray)}
        print(f"  few-shot {name:14s}: skill={few['skill_pct']:+6.1f}%  "
              f"corr={few['corr']:+.2f}  (200 iters from pretrained)")

    # artwork: latent geometry + zero-shot forecasts
    viz.plot_latent_geometry(*latent_geometry(model, samples, seed=0,
                                              device=device),
                             os.path.join(args.out, "latent_geometry.png"))
    for name, m in zip([x[0] for x in samples], meta):
        if not m["held_out"]:
            continue
        pair = dict(samples)[name]
        res = zero_shot_forecast(model, pair["series"], device=device)
        pers = np.full(len(res["true"]), res["true"][0])
        viz.plot_zero_shot(res["true"], res["pred"],
                           os.path.join(args.out, f"zero_shot_{name}.png"),
                           f"Zero-shot TSO forecast — {name} (never seen during pretraining)",
                           res["skill_pct"], baseline=pers)

    # deep-Koopman forecast overlay
    split = int(len(S) * 0.7)
    x0 = S[split - 1]
    horizon = len(S) - split
    from tso.koopman import edmd_rff, forecast as koop_forecast
    from tso.deep_koopman import deep_koopman_forecast
    rff_m = edmd_rff(S[: split - 1], S[1:split], lift_dim=128, seed=0)
    pred_rff = koop_forecast(rff_m, x0, horizon)[:, 0]
    pred_dk = deep_koopman_forecast(model, x0, horizon, device=device)[:, 0]
    true_vals = S[split - 1: split + horizon, 0]
    viz.plot_deep_koopman_bench(true_vals, pred_rff, pred_dk,
                                os.path.join(args.out, "deep_koopman_bench.png"),
                                dk_bench["rff"]["skill_pct"],
                                dk_bench["deep"]["skill_pct"])

    viz.plot_bifurcation(sweep, os.path.join(args.out, "bifurcation.png"),
                         detected_rho)

    # --- summary metrics ---------------------------------------------------
    results = {
        "device": device,
        "deep_koopman": dk_bench,
        "bifurcation": {"detected_rho": detected_rho,
                        "theoretical_onset": 24.74, "threshold": thr},
        "train_loop": bench,
        "pretrain": {"iters": args.iters,
                     "final_loss": float(hist[-1]) if hist else None,
                     "pretext_losses": parts_agg},
        "corpus": meta,
        "zero_shot": zs_results,
        "scratch_baseline": scratch_results,
        "few_shot_baseline": fewshot_results,
        "transfer_summary": {
            k: {"frozen_skill": round(zs_results[k]["skill_pct"], 2),
                "fewshot_skill": round(fewshot_results[k]["skill_pct"], 2),
                "scratch_skill": round(scratch_results[k]["skill_pct"], 2),
                "fewshot_vs_scratch_pts": round(fewshot_results[k]["skill_pct"] -
                                                 scratch_results[k]["skill_pct"], 2)}
            for k in scratch_results
        },
        "files": sorted(os.listdir(args.out)),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  -> all outputs in {args.out}/")
    for f in sorted(os.listdir(args.out)):
        print(f"     {f}")


if __name__ == "__main__":
    main()
