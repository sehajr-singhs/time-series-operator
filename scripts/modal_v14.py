#!/usr/bin/env python3
"""v14 multi-seed: the plateau-break recipe on GPU (Modal, T4, AMP).

Reproduces the Kaggle v14 kernel protocol exactly, with the torch seed as
the only variable: real 40-series corpus x8 + the deterministic 240-series
balanced dynamics battery (seed 11), latent 256 / hidden 768, 25k iters,
dyn_w=2.5. Purpose: does the 14/23 shared-corpus break survive seeds 1-3
with a pooled significant p-value vs v9?

Outputs per seed (volume /data/out-v14sN/): foundation_model.pt,
metrics.json (zero-shot probe over the 40 real series + solar discovery).

Usage:
    modal run scripts/modal_v14.py --seeds 0,1,2
    modal run scripts/modal_v14.py --seeds 99 --iters 30 --latent 32 \
        --hidden 64      # smoke test
"""
import os

import modal

APP = modal.App("tso-v14")

VOL = modal.Volume.from_name("tso-v14-outputs", create_if_missing=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE = (
    modal.Image.debian_slim()
    .env({"MPLBACKEND": "Agg"})
    .pip_install("torch", "numpy", "matplotlib", "pandas", "scipy",
                 "scikit-learn")
    .add_local_dir(os.path.join(ROOT, "kaggle_kernel_tso"), "/repo")
    .add_local_dir(os.path.join(ROOT, "output", "modal_corpus"), "/corpus")
)


@APP.function(image=IMAGE, gpu="T4", volumes={"/data": VOL}, timeout=5400,
              cpu=4.0, memory=16384)
def train_seed(seed: int, iters: int = 25000, latent: int = 256,
               hidden: int = 768, dyn_w: float = 2.5) -> str:
    import json
    import os
    import pickle
    import sys

    import numpy as np

    sys.path.insert(0, "/repo")
    import main as K  # merged single-file kernel (v14 code)

    OUT = f"/data/out-v14s{seed}"
    os.makedirs(OUT, exist_ok=True)

    device = K.pick_device("cuda")
    print(f"[v14 seed {seed}] device={device} "
          f"({K.torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")

    with open("/corpus/corpus40.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    samples, meta = bundle["samples"], bundle["meta"]
    real_train = [p for (n, p), m in zip(samples, meta) if not m["held_out"]]

    # deterministic v14 battery (seed 11, 240 series, n=2600) — same as the
    # Kaggle kernel; only the torch seed varies across runs.
    battery = K.load_dynamics_battery(n_series=240)
    battery_pairs = []
    for name, s in battery:
        try:
            battery_pairs.append(K.prepare_pair(s))
        except Exception as e:
            print(f"  !! {name} failed to embed: {e}")
    train_pairs = real_train * 8 + battery_pairs
    print(f"[v14 seed {seed}] corpus: {len(real_train)} real x8 + "
          f"{len(battery_pairs)} synthetic = {len(train_pairs)} entries")

    # ---- stage 1: pretraining (AMP on GPU, checkpointed) -------------------
    print(f"[v14 seed {seed}] pretraining latent={latent} hidden={hidden} "
          f"iters={iters} dyn_w={dyn_w}")
    model, hist, parts_agg, parts = K.pretrain_foundation(
        train_pairs, iters=iters, latent_dim=latent, hidden=hidden, seed=seed,
        device=device, amp=(device == "cuda"), print_every=5000,
        ckpt_path=os.path.join(OUT, "foundation_model.pt"), dyn_w=dyn_w)
    K.torch.save(model.state_dict(), os.path.join(OUT, "foundation_model.pt"))
    K.plot_pretrain_curves(hist, parts, os.path.join(OUT,
                                                     "pretrain_curves.png"))
    VOL.commit()
    print(f"[v14 seed {seed}] pretraining done: final loss {hist[-1]:.5f}")

    # ---- stage 2: zero-shot probe over the 40 real series + solar ----------
    zs, scratch = {}, {}
    sunspot_series = None
    for n, p in samples:
        res = K.zero_shot_forecast(model, p["series"], device=device)
        zs[n] = {k: v for k, v in res.items()
                 if not isinstance(v, np.ndarray)}
        if n == "sunspots":
            sunspot_series = p["series"]
        print(f"[v14 seed {seed}] frozen {n:22s}: skill={res['skill_pct']:+7.1f}%")
    for m in meta:
        if not m["held_out"]:
            continue
        pair = dict(samples)[m["name"]]
        sc = K.scratch_baseline(pair["series"], iters=400, seed=0,
                                device=device)
        scratch[m["name"]] = {k: v for k, v in sc.items()
                              if not isinstance(v, np.ndarray)}
    K.plot_latent_geometry(*K.latent_geometry(model, samples, seed=0,
                                              device=device),
                           os.path.join(OUT, "latent_geometry.png"))

    solar = None
    if sunspot_series is not None:
        disc = K.solar_cycle_discovery(model, sunspot_series, device=device)
        months = disc["period_months"]
        K.plot_solar_discovery(disc["rows"],
                               os.path.join(OUT, "solar_cycle.png"),
                               known_months=disc["known_cycle_months"])
        solar = {"period_months": float(months) if months else None,
                 "period_years": (float(months) / 12.0) if months else None,
                 "known_cycle_months": disc["known_cycle_months"],
                 "rows": disc["rows"]}
        print(f"[v14 seed {seed}] SOLAR: {months:.0f} mo vs known "
              f"{disc['known_cycle_months']:.0f}")

    results = {
        "version": "v14", "device": device, "seed": seed,
        "cuda": K.torch.cuda.is_available(),
        "gpu": (K.torch.cuda.get_device_name(0)
                if K.torch.cuda.is_available() else None),
        "config": {"iters": iters, "latent_dim": latent, "hidden": hidden,
                   "amp": device == "cuda", "dyn_w": dyn_w,
                   "mode": "v14-balanced-koopman"},
        "pretrain": {"iters": len(hist), "latent_dim": latent,
                     "hidden": hidden, "dyn_w": dyn_w, "seed": seed,
                     "final_loss": float(hist[-1]) if hist else None,
                     "pretext_losses": parts_agg, "mode":
                     "v14-balanced-koopman"},
        "corpus": meta,
        "corpus_note": f"{len(real_train)} real series x8 + "
                       f"{len(battery_pairs)} synthetic dynamics "
                       f"(universal battery, seed 11)",
        "zero_shot": zs,
        "gru_baseline": {},
        "scratch_baseline": scratch,
        "solar_cycle": solar,
        "files": sorted(os.listdir(OUT)),
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    VOL.commit()
    print(f"[v14 seed {seed}] DONE")
    return OUT


@APP.local_entrypoint()
def main(seeds: str = "0,1,2", iters: int = 25000, latent: int = 256,
         hidden: int = 768, dyn_w: float = 2.5):
    calls = [train_seed.spawn(s, iters, latent, hidden, dyn_w)
             for s in map(int, seeds.split(","))]
    print(f"spawned v14 seeds {seeds} on Modal (T4, latent {latent}, "
          f"hidden {hidden}, {iters} iters, dyn_w={dyn_w}) — waiting...")
    for c in calls:
        out = c.get()
        print("finished:", out)


if __name__ == "__main__":
    main()
