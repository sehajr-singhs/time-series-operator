#!/usr/bin/env python3
"""Evaluate the TRUE frozen chronos-t5-small on the exact v18 kernel
protocol (z-scored, 0.7/0.2 split, 64-sample median, horizon<=100).

The v18 kernel's "chronos_frozen" column is contaminated: fine-tuning
mutated pipe.model.model in place, so the kernel's frozen pipe actually
served the 25k-step model (corr 0.986 with its generous column). This
script measures the real pretrained baseline under the byte-identical
protocol so the matched-compute comparison is honest.

Usage: python scripts/eval_v18_frozen.py [start] [end]
Writes output/kaggle_kernel_v18/chronos_frozen_true_per_series.json
"""

import json
import os
import sys
import time

import numpy as np
import torch

CORPUS = "output/modal_corpus/corpus40.npz"
META = "output/modal_corpus/corpus40_meta.json"
OUT = "output/kaggle_kernel_v18/chronos_frozen_true_per_series.json"


def main():
    sl = slice(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 \
        else slice(None)

    from chronos import ChronosPipeline

    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small")
    cfg = pipe.model.config

    arr = np.load(CORPUS, allow_pickle=True)
    meta = json.load(open(META))
    names = [m["name"] for m in meta]
    series_list = [np.asarray(arr[nm], dtype=float) for nm in names]

    def eval_chronos(pp, series):
        x = np.asarray(series, dtype=float)
        x = (x - float(np.nanmean(x))) / (float(np.nanstd(x)) + 1e-8)
        split = int(len(x) * 0.7)
        horizon = min(int(len(x) * 0.2), len(x) - split - 1, 100)
        if horizon < 1:
            return None
        ctx = torch.tensor(x[:split], dtype=torch.float32)
        with torch.no_grad():
            fc = pp.predict(ctx, prediction_length=horizon, num_samples=64)
        pred = fc[0].median(dim=0).values.numpy()
        true = x[split: split + horizon][: len(pred)]
        e = float(np.mean((pred - true) ** 2) ** 0.5)
        ep = float(np.mean((np.full(len(true), true[0]) - true) ** 2) ** 0.5)
        return {"skill_pct": 100.0 * (ep - e) / max(ep, 1e-12),
                "corr": float(np.corrcoef(pred, true)[0, 1])
                if len(true) > 2 else float("nan"),
                "horizon": int(len(pred))}

    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT))
    t0 = time.time()
    for i in range(*sl.indices(len(series_list))):
        out[names[i]] = eval_chronos(pipe, series_list[i])
        print(f"  {i:2d} {names[i]:22s} skill={out[names[i]]['skill_pct']:+7.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)  # save after every series
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"done: {len(out)} series in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()