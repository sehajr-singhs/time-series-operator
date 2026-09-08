#!/usr/bin/env python3
"""v19 final analysis: TSO (real-corpus pretrain) vs Chronos on the held-out
real probe.

Reads output/kaggle_kernel_v19/final_summary.json written by the kernel and
renders the decisive numbers: median skill, wins vs persistence, and
one-sided Wilcoxon p-values for every head-to-head on the held-out probe
(series never seen in either training corpus).

The v19 protocol fixes both v18 lessons:
  - true frozen = snapshot BEFORE fine-tuning (v18's "frozen" was
    contaminated by in-place training);
  - matched/generous = loaded from saved checkpoints, never the live model.
"""

import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = os.path.join(ROOT, "output", "kaggle_kernel_v19")


def agg(ps):
    sk = [v["skill_pct"] for v in ps]
    return {
        "n": len(sk),
        "wins_vs_persistence": int(sum(s > 0 for s in sk)),
        "median": float(np.median(sk)),
        "mean": float(np.mean(sk)),
        "sd": float(np.std(sk)),
        "min": float(np.min(sk)),
        "max": float(np.max(sk)),
    }


def wilcoxon(a, b):
    """One-sided Wilcoxon signed-rank: is a > b?"""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 6 or np.all(d == 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(d, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


def main():
    m = json.load(open(os.path.join(K, "final_summary.json")))
    cfg = m["config"]
    per = m["per_series"]
    held_names = [c["name"] for c in m["corpus"] if c.get("held_out")]
    print("=" * 72)
    print("v19 real-corpus rematch — final analysis")
    print("=" * 72)
    print(f"config: {cfg['experiment']}")
    print(f"  TSO      {cfg['p_tso']:,} params x {cfg['n_tso_iters']:,} iters "
          f"(lat {cfg.get('latent_dim')}, hid {cfg.get('hidden')})")
    print(f"  Chronos  {cfg['p_chronos']:,} params; matched = "
          f"{cfg['n_matched']:,} steps, generous = {cfg['n_total']:,} steps")
    print(f"  corpus:  {cfg['pool_entries']} pool entries; device {cfg['device']}")
    print(f"  held-out probe: {len(held_names)} series\n")

    models = ("tso", "chronos_frozen", "chronos_matched", "chronos_generous")
    labs = {
        "tso": "TSO (operator)",
        "chronos_frozen": "Chronos frozen",
        "chronos_matched": "Chronos fine-tuned (matched)",
        "chronos_generous": "Chronos fine-tuned (equal steps)",
    }
    print("held-out probe aggregates:")
    for k in models:
        ps = [per[n][k] for n in held_names if per[n].get(k)]
        a = agg(ps)
        print(f"  {labs[k]:32s} med {a['median']:+7.2f}  "
              f"mean {a['mean']:+7.2f}  wins {a['wins_vs_persistence']:2d}/"
              f"{a['n']}")

    print("\nhead-to-head (one-sided Wilcoxon, a > b):")
    for ab in ("tso_vs_frozen", "tso_vs_matched", "tso_vs_generous",
               "matched_vs_frozen", "generous_vs_frozen"):
        h = m["h2h"]["held"].get(ab)
        if not h:
            continue
        a, b = ab.split("_vs_")
        d = [per[n][a]["skill_pct"] - per[n][b]["skill_pct"]
             for n in held_names if per[n].get(a) and per[n].get(b)]
        p = wilcoxon([x for x in d], [0.0] * len(d)) if False else None
        va = [per[n][a]["skill_pct"] for n in held_names
              if per[n].get(a) and per[n].get(b)]
        vb = [per[n][b]["skill_pct"] for n in held_names
              if per[n].get(a) and per[n].get(b)]
        p = wilcoxon(va, vb)
        print(f"  {labs[a]:32s} vs {labs[b]:30s} "
              f"{h['a_wins']}W/{h['b_wins']}L  p={p:.3f}")

    # fine-tune effect: does fine-tuning at ANY budget help Chronos on this
    # corpus? (the v18 question, now on real data)
    d = [per[n]["chronos_matched"]["skill_pct"]
         - per[n]["chronos_frozen"]["skill_pct"]
         for n in held_names
         if per[n].get("chronos_matched") and per[n].get("chronos_frozen")]
    p = wilcoxon(
        [per[n]["chronos_matched"]["skill_pct"] for n in held_names
         if per[n].get("chronos_matched") and per[n].get("chronos_frozen")],
        [per[n]["chronos_frozen"]["skill_pct"] for n in held_names
         if per[n].get("chronos_matched") and per[n].get("chronos_frozen")])
    print(f"\nmatched fine-tune effect: {sum(1 for x in d if x > 0)}/"
          f"{len(d)} series improve, p={p:.3f} (one-sided)")
    dg = [per[n]["chronos_generous"]["skill_pct"]
          - per[n]["chronos_frozen"]["skill_pct"]
          for n in held_names
          if per[n].get("chronos_generous") and per[n].get("chronos_frozen")]
    pg = wilcoxon(
        [per[n]["chronos_generous"]["skill_pct"] for n in held_names
         if per[n].get("chronos_generous") and per[n].get("chronos_frozen")],
        [per[n]["chronos_frozen"]["skill_pct"] for n in held_names
         if per[n].get("chronos_generous") and per[n].get("chronos_frozen")])
    print(f"generous fine-tune effect: {sum(1 for x in dg if x > 0)}/"
          f"{len(dg)} series improve, p={pg:.3f} (one-sided)")

    if m.get("pretrain"):
        pt = m["pretrain"]
        print(f"\nTSO pretrain: {pt['iters']} iters, final loss "
              f"{pt['final_loss']:.4f} (lat {pt['latent_dim']}, "
              f"hid {pt['hidden']}, dyn_w {pt['dyn_w']})")

    out = os.path.join(K, "analysis_summary.json")
    json.dump({"held_names": held_names,
               "aggregates_held": m["aggregates"]["held"],
               "h2h_held": m["h2h"]["held"],
               "config": cfg,
               "pretrain": m.get("pretrain")}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()