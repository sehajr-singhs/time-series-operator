#!/usr/bin/env python3
"""Pooled multi-seed significance: v14 (4 seeds) vs v9 on the shared 23-series
corpus, per-series frozen zero-shot skill with the current probe.

Reprobes every checkpoint on the identical local corpus (the apples-to-apples
protocol), then tests:
  1. pooled positive-fraction: all v14 seeds vs v9's rate (binomial),
  2. per-series sign test on median-of-seeds v14 skill vs v9 skill,
  3. each seed's wins/losses vs v9 (binomial), for the record.

Writes output/study/reprobe_skills.json {tag: {series: skill}} for the figures.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "output", "study")
sys.path.insert(0, ROOT)

from tso.foundation import FoundationOperator, zero_shot_forecast  # noqa: E402


def reprobe(tag, ckpath, lat, hid, meta, z, cache):
    """Per-series skills for one checkpoint; cached by tag."""
    if tag in cache:
        return cache[tag]
    import torch
    m = FoundationOperator(latent_dim=lat, hidden=hid)
    m.load_state_dict(torch.load(ckpath, map_location="cpu",
                                 weights_only=True))
    m.eval()
    out = {}
    for mm in meta:
        out[mm["name"]] = zero_shot_forecast(m, z[mm["name"]])["skill_pct"]
    cache[tag] = out
    json.dump(cache, open(os.path.join(STUDY, "reprobe_skills.json"), "w"),
              indent=1)
    return out


def main():
    meta = json.load(open(os.path.join(STUDY, "corpus_meta.json")))
    z = np.load(os.path.join(STUDY, "corpus.npz"), allow_pickle=True)
    p = os.path.join(STUDY, "reprobe_skills.json")
    cache = json.load(open(p)) if os.path.exists(p) else {}

    models = [
        ("v9", os.path.join(ROOT, "output", "kaggle_kernel_v9",
                            "foundation_model.pt"), 128, 384),
        ("v14s0", os.path.join(ROOT, "output", "kaggle_kernel_v14",
                               "foundation_model.pt"), 256, 768),
    ]
    for s in (1, 2, 3):
        models.append((f"v14s{s}",
                       os.path.join(ROOT, "output",
                                    f"kaggle_kernel_v14_seed{s}",
                                    "foundation_model.pt"), 256, 768))
    # v16: protocol error run that duplicated the v14 recipe end-to-end
    # (240-series battery instead of 5988) — a free fifth seed.
    models.append(("v16", os.path.join(ROOT, "output", "kaggle_kernel_v16",
                                        "foundation_model.pt"), 256, 768))
    skills = {t: reprobe(t, c, l, h, meta, z, cache) for t, c, l, h in models}

    names = [mm["name"] for mm in meta]
    v9 = np.array([skills["v9"][n] for n in names])
    seeds = ["v14s0", "v14s1", "v14s2", "v14s3", "v16"]
    S = np.array([[skills[t][n] for n in names] for t in seeds])  # (4, 23)

    print(f"shared corpus: {len(names)} series / "
          f"{len({mm['domain'] for mm in meta})} domains")
    for i, t in enumerate(seeds):
        pos = int((S[i] > 0).sum())
        d = S[i] - v9
        w = int((d > 0).sum()); l = int((d < 0).sum())
        pv = stats.binomtest(w, w + l, 0.5, alternative="greater").pvalue
        print(f"{t}: positive {pos}/{len(names)} | vs v9 {w}W/{l}L "
              f"p={pv:.4f} | median skill {np.median(S[i]):+.2f}")

    # pooled 1: all 92 seed-series draws vs v9's rate
    pooled = S.flatten()
    n = len(pooled); k = int((pooled > 0).sum())
    p_v9 = float((v9 > 0).mean())
    p1 = stats.binomtest(k, n, p_v9, alternative="greater").pvalue
    p_chance = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
    print(f"\npooled {n} seed-series: positive {k}/{n} "
          f"(v9 rate {p_v9:.3f})")
    print(f"  vs v9 rate: p={p1:.4f} | vs chance: p={p_chance:.4f}")

    # pooled 2: median-of-seeds per series vs v9, sign test
    med = np.median(S, axis=0)
    d = med - v9
    w = int((d > 0).sum()); l = int((d < 0).sum()); t_ = int((d == 0).sum())
    p2 = stats.binomtest(w, w + l, 0.5, alternative="greater").pvalue
    print(f"median-of-seeds vs v9: {w}W/{l}L/{t_}T, p={p2:.4f}, "
          f"median delta {np.median(d):+.2f}")

    # paired t-test on median-of-seeds deltas (normal approx, dependent)
    tt = stats.ttest_rel(med, v9)
    print(f"paired t (median-of-seeds vs v9): t={tt.statistic:.3f}, "
          f"p={tt.pvalue:.4f}")

    # Wilcoxon signed-rank on the medians
    wr = stats.wilcoxon(med, v9, alternative="greater")
    print(f"wilcoxon (median-of-seeds > v9): p={wr.pvalue:.4f}")

    json.dump(skills, open(p, "w"), indent=1)
    print("per-series skills cached ->", p)


if __name__ == "__main__":
    main()