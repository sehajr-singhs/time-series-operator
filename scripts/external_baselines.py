#!/usr/bin/env python3
"""External foundation-model baselines on the exact TSO protocol.

Runs Chronos (Amazon, chronos-t5-small), and optionally Moirai
(Salesforce, uni2ts), on the same 40-series corpus with the same
train/test split and the same skill-vs-persistence metric that the TSO
zero-shot probe uses. No in-domain fine-tuning for any model: Chronos and
Moirai are used frozen, TSO's probe is the closed-form Koopman fit.

Output: output/external_baselines.json (per-series skill for each model)
and a printed head-to-head table.

Usage:
    python scripts/external_baselines.py            # Chronos only
    python scripts/external_baselines.py --moirai  # + Moirai-small (heavy)
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_corpus():
    bundle = pickle.load(open(os.path.join(
        ROOT, "output", "modal_corpus", "corpus40.pkl"), "rb"))
    return bundle["samples"], bundle["meta"]


def split_horizon(x_norm):
    """Mirror the TSO probe's split/horizon on the normalized series."""
    max_lag = min(150, max(len(x_norm) // 6, 1))
    # the probe embeds with (delay, dim=5); first column is x itself, so the
    # probe's train/test boundary is split on the embedded row count
    n = len(x_norm)
    split = int(n * 0.7)
    horizon = min(int(n * 0.2), n - split - 1, 100)
    return split, horizon, max_lag


def skill_vs_persistence(pred, true):
    e = float(np.mean((pred - true) ** 2) ** 0.5)
    ep = float(np.mean((np.full(len(true), true[0]) - true) ** 2) ** 0.5)
    return 100.0 * (ep - e) / max(ep, 1e-12)


def run_chronos(samples, device="cpu"):
    """Chronos-t5-small, frozen, zero-shot (no training)."""
    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError as e:
        print("chronos not installed:", e)
        return {}
    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small")
    out = {}
    t0 = time.time()
    for i, (name, pair) in enumerate(samples):
        x = pair["series"].astype(float)
        x = (x - float(np.nanmean(x))) / (float(np.nanstd(x)) + 1e-8)
        split, horizon, _ = split_horizon(x)
        ctx = torch.tensor(x[:split], dtype=torch.float32)
        with torch.no_grad():
            fc = pipe.predict(ctx, prediction_length=horizon,
                              num_samples=64)
        pred = fc[0].median(dim=0).values.numpy()
        true = x[split: split + horizon]
        if len(pred) < len(true):
            true = true[: len(pred)]
        out[name] = {
            "skill_pct": skill_vs_persistence(pred, true),
            "corr": float(np.corrcoef(pred, true)[0, 1])
            if len(true) > 2 else float("nan"),
            "horizon": int(len(pred)),
        }
        print(f"  chronos {name:22s}: skill={out[name]['skill_pct']:+8.1f}%"
              f"  corr={out[name]['corr']:+.2f}  ({i + 1}/{len(samples)})",
              flush=True)
        # incremental save: a timeout must not lose finished series
        json.dump({"chronos": out},
                  open(os.path.join(ROOT, "output",
                                    "external_baselines.json"), "w"),
                  indent=1)
    print(f"  chronos total {time.time() - t0:.0f}s")
    return out


def run_moirai(samples):
    """Moirai-small-1.1, frozen, zero-shot."""
    try:
        import torch
        from uni2ts.eval_util.plot import load_model
        from einops import rearrange
    except ImportError as e:
        print("moirai not available:", e)
        return {}
    model = load_model("hf", "Salesforce/moirai-small-1.1")
    out = {}
    for name, pair in samples:
        x = pair["series"].astype(float)
        x = (x - float(np.nanmean(x))) / (float(np.nanstd(x)) + 1e-8)
        split, horizon, _ = split_horizon(x)
        ctx = torch.tensor(x[:split], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model.predict(ctx, prediction_length=horizon)
        pred = pred.median(dim=0).values.numpy()
        true = x[split: split + len(pred)]
        out[name] = {
            "skill_pct": skill_vs_persistence(pred, true),
            "corr": float(np.corrcoef(pred, true)[0, 1])
            if len(true) > 2 else float("nan"),
            "horizon": int(len(pred)),
        }
        print(f"  moirai {name:22s}: skill={out[name]['skill_pct']:+8.1f}%",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moirai", action="store_true")
    args = ap.parse_args()

    samples, meta = load_corpus()
    print(f"corpus: {len(samples)} series")

    dest = os.path.join(ROOT, "output", "external_baselines.json")
    res = {}
    if os.path.exists(dest):
        res = json.load(open(dest))
    res["chronos"] = run_chronos(samples)
    if args.moirai:
        res["moirai"] = run_moirai(samples)

    json.dump(res, open(dest, "w"), indent=1)
    print("saved", dest)


if __name__ == "__main__":
    main()