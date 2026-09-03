#!/usr/bin/env python3
"""Fetch and summarize the v16/v17 corpus-scale Kaggle runs.

v16 was pushed with the v14-size default (protocol error, superseded by
v17); v17 is the real corpus-scale test: the 5988-series balanced battery
(randomized parameter sweeps, v14 family balance) with z-scored emission
(the v15 divergence fix), v14's exact recipe otherwise (lat 256 · hid 768,
25k iters, dyn_w=2.5).

Usage: python scripts/v17_compare.py [--fetch]
"""
import argparse
import json
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fetch(name, slug):
    out = os.path.join(ROOT, "output", name)
    os.makedirs(out, exist_ok=True)
    r = subprocess.run(["kaggle", "kernels", "output", slug, "-p", out],
                       capture_output=True, text=True)
    print((r.stdout or r.stderr)[-200:])


def summarize(m, name):
    zs, gru = m["zero_shot"], m["gru_baseline"]
    wins = sum(1 for n in zs if zs[n]["skill_pct"] > gru.get(n, -1e9))
    pos = sum(1 for n in zs if zs[n]["skill_pct"] > 0)
    sk = float(np.median([zs[n]["skill_pct"] for n in zs]))
    solar = (m.get("solar_cycle") or {}).get("period_months")
    pre = m.get("pretrain") or {}
    note = m.get("corpus_note", "")
    print(f"[{name}] n={len(zs)} wins={wins} positive={pos} "
          f"med_skill={sk:+.1f} solar={solar} loss={pre.get('final_loss')} "
          f"iters={pre.get('iters')}")
    print(f"  corpus: {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    for name, slug in [("kaggle_kernel_v16", "sehajrsingh/tso-foundation-model-v16"),
                       ("kaggle_kernel_v17", "sehajrsingh/tso-foundation-model-v17")]:
        p = os.path.join(ROOT, "output", name, "metrics.json")
        if args.fetch or not os.path.exists(p):
            fetch(name, slug)
        if os.path.exists(p):
            summarize(json.load(open(p)), name)


if __name__ == "__main__":
    main()