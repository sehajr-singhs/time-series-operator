#!/usr/bin/env python3
"""Fetch and analyze the v14 (balanced corpus + forced Koopman) Kaggle run.

v14 protocol: v11 capacity (latent 256/hidden 768, 25k iters) on CPU,
with a regime-balanced synthetic battery (smooth/seasonal/trend/spiky
families added to the v13 dynamics battery) and the Koopman dynamics
pretext re-weighted to dyn_w=2.5.

Usage: python scripts/v14_compare.py [--fetch]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "kaggle_kernel_v14")


def fetch():
    os.makedirs(OUT, exist_ok=True)
    r = subprocess.run(["kaggle", "kernels", "output",
                        "sehajrsingh/tso-foundation-model-v14", "-p", OUT],
                       capture_output=True, text=True)
    print((r.stdout or r.stderr)[-300:])


def summarize(m, name="v14"):
    zs, gru = m["zero_shot"], m["gru_baseline"]
    wins = sum(1 for n in zs if zs[n]["skill_pct"] > gru.get(n, -1e9))
    pos = sum(1 for n in zs if zs[n]["skill_pct"] > 0)
    sk = float(np.median([zs[n]["skill_pct"] for n in zs]))
    diffs = [zs[n]["skill_pct"] - gru[n] for n in zs
             if isinstance(gru.get(n), (int, float))]
    md = float(np.median(diffs)) if diffs else float("nan")
    solar = (m.get("solar_cycle") or {}).get("period_months")
    pre = m.get("pretrain") or {}
    out = dict(n_series=len(zs), wins=wins, positive=pos,
               median_skill=sk, median_vs_gru=md,
               solar_months=solar, final_loss=pre.get("final_loss"),
               mode=pre.get("mode"))
    print(f"[{name}] {json.dumps(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    p = os.path.join(OUT, "metrics.json")
    if not os.path.exists(p):
        print(f"no metrics yet at {p}; run --fetch once the kernel completes")
        return
    m = json.load(open(p))
    summarize(m)
    # quick per-series deltas vs the v9 (baseline plateau) run for the paper
    p9 = os.path.join(ROOT, "output", "kaggle_kernel_v9", "metrics.json")
    if os.path.exists(p9):
        m9 = json.load(open(p9))
        rows = []
        for n in m["zero_shot"]:
            if n in m9["zero_shot"] and n in m["gru_baseline"]:
                rows.append((n, m9["zero_shot"][n]["skill_pct"],
                             m["zero_shot"][n]["skill_pct"]))
        big = sorted(rows, key=lambda r: abs(r[1] - r[2]), reverse=True)[:6]
        print("largest v9-vs-v14 deltas (series, v9, v14):")
        for n, a, b in big:
            print(f"  {n:20s} {a:+9.1f} {b:+9.1f}  (d={b - a:+9.1f})")


if __name__ == "__main__":
    main()
