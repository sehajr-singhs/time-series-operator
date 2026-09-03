#!/usr/bin/env python3
"""v11: GPU-scaled TSO pretraining on Modal (T4, AMP).

Controlled capacity-scaling experiment vs v9/v10 (Kaggle CPU, latent 128 /
hidden 384, 25k iters): same 40-series corpus, same protocol, but latent 256 /
hidden 768, 25k iters, 3 seeds, mixed precision on a T4. This tests the
paper's prescription — width is the lever, not raw iteration count.

Outputs per seed (volume /data/out-seedN/): foundation_model.pt,
metrics.json, pretrain_curves.png, latent_geometry.png, solar_cycle.png,
zero-shot plots for the held-out series.

Usage:
    modal run scripts/modal_v11.py            # seeds 0,1,2
    modal run scripts/modal_v11.py --seeds 0 # single seed
"""
import os

import modal

APP = modal.App("tso-v11")

VOL = modal.Volume.from_name("tso-v11-outputs", create_if_missing=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE = (
    modal.Image.debian_slim()
    .env({"MPLBACKEND": "Agg"})
    .pip_install("torch", "numpy", "matplotlib", "pandas", "scipy",
                 "scikit-learn")
    .add_local_dir(os.path.join(ROOT, "kaggle_kernel_tso"), "/repo")
    .add_local_dir(os.path.join(ROOT, "output", "modal_corpus"), "/corpus")
)

ITERS = 25000
LATENT = 256
HIDDEN = 768


@APP.function(image=IMAGE, gpu="T4", volumes={"/data": VOL}, timeout=3600,
              cpu=4.0, memory=16384)
def train_seed(seed: int, iters: int = ITERS, latent: int = LATENT,
               hidden: int = HIDDEN, joint_probe: bool = False,
               out_tag: str = "out-seed") -> str:
    import json
    import os
    import pickle
    import sys

    import numpy as np

    sys.path.insert(0, "/repo")
    import main as K  # the merged single-file kernel module

    OUT = f"/data/{out_tag}{seed}"
    os.makedirs(OUT, exist_ok=True)

    device = K.pick_device("cuda")
    print(f"[seed {seed}] device={device} "
          f"({K.torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")

    with open("/corpus/corpus40.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    samples, meta = bundle["samples"], bundle["meta"]
    held = [m["name"] for m in meta if m["held_out"]]
    train_pairs = [p for (n, p), m in zip(samples, meta) if not m["held_out"]]
    print(f"[seed {seed}] {len(samples)} series, held out: {held}")

    # ---- stage 1: pretraining (AMP on GPU, checkpointed) -------------------
    print(f"[seed {seed}] pretraining latent={latent} hidden={hidden} "
          f"iters={iters}")
    model, hist, parts_agg, parts = K.pretrain_foundation(
        train_pairs, iters=iters, latent_dim=latent, hidden=hidden, seed=seed,
        device=device, amp=(device == "cuda"), print_every=5000,
        ckpt_path=os.path.join(OUT, "foundation_model.pt"),
        joint_probe=joint_probe)
    if joint_probe:
        print(f"[seed {seed}] v12 joint-probe mode: multi-step Koopman "
              f"roll-out + unit-circle spectrum regularization")
    K.torch.save(model.state_dict(), os.path.join(OUT, "foundation_model.pt"))
    K.plot_pretrain_curves(hist, parts, os.path.join(OUT,
                                                     "pretrain_curves.png"))
    VOL.commit()
    print(f"[seed {seed}] pretraining done: final loss {hist[-1]:.5f}")

    # ---- stage 2: zero-shot probe + in-kernel baselines --------------------
    zs, gru, scratch = {}, {}, {}
    sunspot_series = None
    for n, p in samples:
        res = K.zero_shot_forecast(model, p["series"], device=device)
        zs[n] = {k: v for k, v in res.items()
                 if not isinstance(v, np.ndarray)}
        if n == "sunspots":
            sunspot_series = p["series"]
        try:
            g = K.gru_baseline(p["fine"], iters=300, seed=0, device=device)
            gru[n] = g["skill_pct"]
        except Exception as e:
            print(f"  !! gru {n}: {e}")
        print(f"[seed {seed}] frozen {n:22s}: skill={res['skill_pct']:+7.1f}% "
              f"corr={res['corr']:+.2f} gru={gru.get(n, float('nan')):+7.1f}%")
    for m in meta:
        if not m["held_out"]:
            continue
        pair = dict(samples)[m["name"]]
        sc = K.scratch_baseline(pair["series"], iters=400, seed=0,
                                device=device)
        scratch[m["name"]] = {k: v for k, v in sc.items()
                              if not isinstance(v, np.ndarray)}
        print(f"[seed {seed}] scratch {m['name']:22s}: "
              f"skill={sc['skill_pct']:+7.1f}%  corr={sc['corr']:+.2f}")
        res = K.zero_shot_forecast(model, pair["series"], device=device)
        pers = np.full(len(res["true"]), res["true"][0])
        K.plot_zero_shot(res["true"], res["pred"],
                         os.path.join(OUT, f"zero_shot_{m['name']}.png"),
                         f"Zero-shot TSO forecast - {m['name']} (V11 seed "
                         f"{seed})", res["skill_pct"], baseline=pers)

    K.plot_latent_geometry(*K.latent_geometry(model, samples, seed=0,
                                              device=device),
                           os.path.join(OUT, "latent_geometry.png"))

    # ---- stage 3: the solar-cycle discovery --------------------------------
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
        if months is None:
            print(f"[seed {seed}] SOLAR: no oscillatory mode at any scale")
        else:
            print(f"[seed {seed}] SOLAR-CYCLE DISCOVERY: "
                  f"{months:.0f} months ({months / 12:.1f} yr) vs known "
                  f"{disc['known_cycle_months']:.0f}")

    results = {
        "version": "v11", "device": device, "seed": seed,
        "cuda": K.torch.cuda.is_available(),
        "gpu": (K.torch.cuda.get_device_name(0)
                if K.torch.cuda.is_available() else None),
        "config": {"iters": iters, "latent_dim": latent, "hidden": hidden,
                   "amp": device == "cuda", "joint_probe": joint_probe,
                   "mode": "v12-joint-probe" if joint_probe else "v11"},
        "pretrain": {"iters": len(hist), "final_loss": float(hist[-1])
                     if hist else None, "pretext_losses": parts_agg},
        "corpus": meta,
        "zero_shot": zs,
        "gru_baseline": gru,
        "scratch_baseline": scratch,
        "solar_cycle": solar,
        "transfer_summary": {
            k: {"frozen_skill": round(zs[k]["skill_pct"], 2),
                "scratch_skill": round(scratch[k]["skill_pct"], 2)}
            for k in scratch},
        "files": sorted(os.listdir(OUT)),
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    VOL.commit()
    print(f"[seed {seed}] DONE")
    return OUT


@APP.local_entrypoint()
def main(seeds: str = "0,1,2", iters: int = ITERS, latent: int = LATENT,
         hidden: int = HIDDEN, joint_probe: bool = False,
         out_tag: str = "out-seed"):
    """Config passes as CLI args (env vars do NOT propagate through
    `modal run` on this platform — verified the hard way):
    modal run scripts/modal_v11.py --seeds 0 --iters 100 --latent 32 --hidden 64
    modal run scripts/modal_v11.py --joint-probe True --out-tag out-v12
    """
    calls = [train_seed.spawn(s, iters, latent, hidden, joint_probe, out_tag)
             for s in map(int, seeds.split(","))]
    print(f"spawned seeds {seeds} on Modal (T4, latent {latent}, "
          f"hidden {hidden}, {iters} iters, joint_probe={joint_probe}, "
          f"tag={out_tag}) — waiting...")
    for c in calls:
        out = c.get()
        print("finished:", out)


if __name__ == "__main__":
    main()