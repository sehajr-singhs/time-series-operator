#!/usr/bin/env python3
"""Paired per-series zero-shot skills for two checkpoints on the shared
23-series corpus (same probe) with a two-sided binomial sign test.

Usage:
  python scripts/pair_models.py A:path/to/modelA.pt:lat:hid B:path/to/modelB.pt:lat:hid
  e.g. python scripts/pair_models.py \
         v9:output/kaggle_kernel_v9/foundation_model.pt:128:384 \
         v13:output/kaggle_kernel_v13/foundation_model.pt:128:384

Writes output/study/pair_{A}_{B}.json and prints the test.
"""
import json
import os
import sys
from math import comb

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "output", "study")
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from tso.foundation import FoundationOperator, zero_shot_forecast  # noqa: E402


def parse(spec):
    tag, ck, lat, hid = spec.split(":")
    if not os.path.isabs(ck):
        ck = os.path.join(ROOT, ck)
    return tag, ck, int(lat), int(hid)


A, B = parse(sys.argv[1]), parse(sys.argv[2])

meta = json.load(open(os.path.join(STUDY, "corpus_meta.json")))
z = np.load(os.path.join(STUDY, "corpus.npz"), allow_pickle=True)
names = [m["name"] for m in meta]

cache = os.path.join(STUDY, "pair_skills.json")
skills = json.load(open(cache)) if os.path.exists(cache) else {}

for tag, ck, lat, hid in (A, B):
    if tag in skills:
        print(f"  {tag}: cached", flush=True)
        continue
    m = FoundationOperator(latent_dim=lat, hidden=hid)
    m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    m.eval()
    s = {}
    for i, n in enumerate(names):
        r = zero_shot_forecast(m, z[n])
        s[n] = float(r["skill_pct"])
        print(f"  {tag} {n}: {s[n]:+.1f}", flush=True)
    skills[tag] = s
    json.dump(skills, open(cache, "w"))

a = np.array([skills[A[0]][n] for n in names])
b = np.array([skills[B[0]][n] for n in names])
wins = int(np.sum(a > b))
losses = int(np.sum(b > a))
ties = int(np.sum(a == b))
n_eff = wins + losses
p = sum(comb(n_eff, k) for k in range(0, min(wins, losses) + 1)) / (2 ** n_eff)
out = {
    "series": names, A[0]: skills[A[0]], B[0]: skills[B[0]],
    f"{A[0]}_wins": wins, f"{B[0]}_wins": losses, "ties": ties,
    "n_effective": n_eff, "two_sided_binomial_p": p,
    f"{A[0]}_median": float(np.median(a)), f"{B[0]}_median": float(np.median(b)),
}
json.dump(out, open(os.path.join(STUDY, f"pair_{A[0]}_{B[0]}.json"), "w"),
          indent=1)
print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (dict, list))},
                 indent=1))
