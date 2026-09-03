#!/usr/bin/env python3
"""Paired per-series zero-shot skills for v9 vs v12 (same probe, same 23-series
corpus) so the plateau claim in the paper carries a real sign-test p-value.
Usage: python scripts/pair_v9_v12.py  (writes output/study/pair_v9_v12.json)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "output", "study")
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from tso.foundation import FoundationOperator, zero_shot_forecast  # noqa: E402

MODELS = [
    ("v9", os.path.join(ROOT, "output", "kaggle_kernel_v9", "foundation_model.pt"), 128, 384),
    ("v12", os.path.join(ROOT, "output", "kaggle_kernel_v12", "foundation_model.pt"), 256, 768),
]
if len(sys.argv) > 1:
    MODELS = [m for m in MODELS if m[0] == sys.argv[1]]

meta = json.load(open(os.path.join(STUDY, "corpus_meta.json")))
z = np.load(os.path.join(STUDY, "corpus.npz"), allow_pickle=True)
names = [m["name"] for m in meta]

skills = {}
cache_path = os.path.join(STUDY, "pair_v9_v12.json")
if os.path.exists(cache_path):
    skills = json.load(open(cache_path))["skills"]
for tag, ckpath, lat, hid in MODELS:
    m = FoundationOperator(latent_dim=lat, hidden=hid)
    m.load_state_dict(torch.load(ckpath, map_location="cpu", weights_only=True))
    m.eval()
    s = {}
    for i, n in enumerate(names):
        r = zero_shot_forecast(m, z[n])
        s[n] = float(r["skill_pct"])
        print(f"  {tag} {n}: {s[n]:+.1f}", flush=True)
    skills[tag] = s

if "v9" not in skills or "v12" not in skills:
    json.dump({"skills": skills}, open(cache_path, "w"))
    print("partial write for " + str(list(skills.keys())) + "; rerun for the other model")
    sys.exit(0)

a = np.array([skills["v9"][n] for n in names])
b = np.array([skills["v12"][n] for n in names])
wins = int(np.sum(a > b))
losses = int(np.sum(b > a))
ties = int(np.sum(a == b))
n_eff = wins + losses
from math import comb
p = sum(comb(n_eff, k) for k in range(0, min(wins, losses) + 1)) / (2 ** n_eff)
out = {
    "series": names,
    "v9": skills["v9"],
    "v12": skills["v12"],
    "v9_wins": wins,
    "v12_wins": losses,
    "ties": ties,
    "n_effective": n_eff,
    "two_sided_binomial_p": p,
    "v9_median": float(np.median(a)),
    "v12_median": float(np.median(b)),
}
json.dump(out, open(os.path.join(STUDY, "pair_v9_v12.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (dict, list))}, indent=1))
