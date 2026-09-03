#!/usr/bin/env python3
"""Rebuild the exact 40-series Kaggle kernel corpus from the local
data/corpus directory and save it for the Modal GPU run.

Uses the merged single-file kernel module (kaggle_kernel_tso/main.py) so the
prepared pairs are byte-identical to what v9/v10 trained on.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle_kernel_tso"))

import main as K  # noqa: E402  (merged single-file kernel)

K.INPUT = os.path.join(ROOT, "data", "corpus")
K.OUT = os.path.join(ROOT, "output", "modal_corpus")
os.makedirs(K.OUT, exist_ok=True)

out = K.build_corpus()
samples, meta = out[0], out[1]

import pickle

pairs = {}
for item in samples:
    name, pair = item[0], item[1]
    pairs[name] = pair

with open(os.path.join(K.OUT, "corpus40.pkl"), "wb") as fh:
    pickle.dump({"samples": [(n, p) for n, p in pairs.items()],
                 "meta": meta}, fh)
with open(os.path.join(K.OUT, "corpus40_meta.json"), "w") as fh:
    json.dump([{k: m[k] for k in ("name", "domain", "held_out", "n", "tau_f")}
               for m in meta], fh, indent=1)

print(f"TOTAL: {len(samples)} series, held out: "
      f"{[m['name'] for m in meta if m['held_out']]}")
print("saved", os.path.join(K.OUT, "corpus40.pkl"),
      os.path.getsize(os.path.join(K.OUT, "corpus40.pkl")), "bytes")