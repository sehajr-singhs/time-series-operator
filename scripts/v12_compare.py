#!/usr/bin/env python3
"""Fetch and analyze the v12 (joint-probe) Kaggle run.

Head-to-head vs v11 (GPU) and v9/v10 (CPU): in-kernel wins vs GRU,
positive counts, median skill, pretext convergence, solar discovery.
Also re-probes the v12 checkpoint on the shared 23-series corpus.

Usage: python scripts/v12_compare.py
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "kaggle_kernel_v12")


def fetch():
    os.makedirs(OUT, exist_ok=True)
    r = subprocess.run(["kaggle", "kernels", "output",
                        "sehajrsingh/tso-foundation-model-v12", "-p", OUT],
                       capture_output=True, text=True)
    print(r.stdout[-300:] if r.stdout else r.stderr[-300:])


def load(name):
    p = os.path.join(OUT, "metrics.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def summarize(m, name):
    zs, gru = m["zero_shot"], m["gru_baseline"]
    wins = sum(1 for n in zs if zs[n]["skill_pct"] > gru.get(n, -1e9))
    pos = sum(1 for n in zs if zs[n]["skill_pct"] > 0)
    sk = np.median([zs[n]["skill_pct"] for n in zs])
    gk = [gru[n] for n in zs if isinstance(gru.get(n), (int, float))]
    from math import comb
    d = np.array([zs[n]["skill_pct"] - gru.get(n, -1e9) for n in zs])
    d = d[np.isfinite(d)]
    p = min(1.0, sum(comb(len(d), k)
                     for k in range(int((d > 0).sum()) + 1, len(d) + 1))
            * 0.5 ** len(d) * 2)
    print(f"{name:28s} wins {wins:2d}/40  pos {pos:2d}/40  "
          f"med {sk:+7.1f}%  gru-med {np.median(gk):+7.1f}%  p={p:.3f}")
    return dict(wins=wins, pos=pos, med=float(sk))


def main():
    fetch()
    m = load(OUT)
    if m is None:
        print("v12 metrics not fetched yet")
        return
    print("device:", m.get("device"), "gpu:", m.get("gpu"))
    print("mode:", m["config"].get("mode"),
          "iters:", m["pretrain"]["iters"])
    pa = m["pretrain"].get("pretext_losses", {})
    print("arrow acc:", round(pa.get("arrow_acc", float("nan")), 3),
          "probe:", round(pa.get("probe", float("nan")), 4),
          "spec:", round(pa.get("spec", float("nan")), 5))
    solar = (m.get("solar_cycle") or {}).get("period_months")
    print("solar:", solar)
    summarize(m, "v12 (joint probe, CPU)")
    v9 = json.load(open(os.path.join(ROOT, "output", "kaggle_kernel_v9",
                                     "metrics.json")))
    v11 = json.load(open(os.path.join(ROOT, "output", "kaggle_kernel_v11",
                                      "seed0", "metrics.json")))
    summarize(v9, "v9  (25k, 128/384, CPU)")
    summarize(v11, "v11 (25k, 256/768, GPU)")


if __name__ == "__main__":
    main()