#!/usr/bin/env python3
"""v18 analysis: fetch + summarize the matched-compute experiment.

TSO v14 (published checkpoint) vs Chronos-t5-small (frozen, fine-tuned at
matched parameter-steps, fine-tuned at generous equal steps), all on the
identical 40-series protocol.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "kaggle_kernel_v18")


def load():
    p = os.path.join(OUT, "metrics.json")
    if not os.path.exists(p):
        sys.exit(f"no {p} — run `kaggle kernels output` first")
    return json.load(open(p))


def main():
    d = load()
    cfg, per, summ, h2h = d["config"], d["per_series"], d["summary"], d["h2h"]
    print("=" * 78)
    print("v18 matched-compute experiment")
    print("=" * 78)
    print(f"  TSO v14:            {cfg['p_tso']:,} params, {cfg['n_tso_iters']:,} iters")
    print(f"  chronos-t5-small:   {cfg['p_chronos']:,} params")
    print(f"  matched budget:     {cfg['n_matched']:,} steps "
          f"(= {cfg['n_tso_iters']:,} x {cfg['p_tso'] / cfg['p_chronos']:.3f})")
    print(f"  generous budget:    {cfg['n_total']:,} steps "
          f"({cfg['n_total'] / cfg['n_matched']:.1f}x matched)")
    print(f"  pool:               {cfg['pool_entries']} entries (v14 corpus "
          f"distribution)")
    print()
    print(f"{'model':24s} {'median':>8s} {'positive':>8s} {'n':>4s}")
    for k, v in summ.items():
        print(f"{k:24s} {v['median']:+8.1f} {v['positive']:>8d} {v['n']:>4d}")
    print()
    print("head-to-head (per-series skill wins):")
    for k, v in h2h.items():
        a, b = k.split("_vs_")
        print(f"  {a:18s} {v['a']:>3d}  vs  {b:18s} {v['b']:>3d}  "
              f"(n={v['n']})")
    # per-series table for the paper
    print()
    print("per-series skills:")
    names = list(per.keys())
    for n in names:
        r = per[n]
        tso = r.get("tso_v14", {}).get("skill_pct")
        frz = r.get("chronos_frozen", {}).get("skill_pct")
        mat = r.get("chronos_matched", {}).get("skill_pct")
        print(f"  {n:22s} tso={tso:+8.1f}  frz={frz:+8.1f}  mat={mat:+8.1f}")

    # signed-rank test: TSO vs fine-tuned Chronos on shared series
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    if wilcoxon:
        for a, b in (("tso_v14", "chronos_matched"),
                     ("tso_v14", "chronos_frozen")):
            va = [per[n][a]["skill_pct"] for n in names
                  if per[n].get(a) and per[n].get(b)]
            vb = [per[n][b]["skill_pct"] for n in names
                  if per[n].get(a) and per[n].get(b)]
            if len(va) > 5:
                w, p = wilcoxon(va, vb)
                print(f"\n  wilcoxon {a} vs {b}: p={p:.4f} "
                      f"(n={len(va)}, W={w:.0f})")
    return d


if __name__ == "__main__":
    main()