#!/usr/bin/env python3
"""Compare kernel runs v9 and v10 on an identical protocol and update the
study paper_data / scaling figure once v10 lands.

    python scripts/v10_compare.py fetch   # kaggle kernels output -> output/kaggle_kernel_v10
    python scripts/v10_compare.py compare # print the v9-vs-v10 delta table
    python scripts/v10_compare.py figures # reprobe v10 + redraw fig_scaling + paper_data
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "output", "study")
V9 = os.path.join(ROOT, "output", "kaggle_kernel_v9")
V10 = os.path.join(ROOT, "output", "kaggle_kernel_v10")

import numpy as np


def load_metrics(tag):
    d = {"v9": V9, "v10": V10}[tag]
    return json.load(open(os.path.join(d, "metrics.json")))


def fetch():
    os.system("cd %s && rm -rf output/kaggle_kernel_v10 && "
              "kaggle kernels output sehajrsingh/tso-foundation-model-v10 "
              "-p output/kaggle_kernel_v10" % ROOT)


def compare():
    a, b = load_metrics("v9"), load_metrics("v10")
    za, zb = a["zero_shot"], b["zero_shot"]
    print("pretrain:", a["pretrain"].get("iters"), "->", b["pretrain"].get("iters"),
          "iters | arrow acc:",
          round(a["pretrain"]["pretext_losses"].get("arrow_acc", 0) * 100, 1),
          "% ->", round(b["pretrain"]["pretext_losses"].get("arrow_acc", 0) * 100, 1), "%")
    print("solar: v9 %.1f mo | v10 %.1f mo (known %.1f)"
          % (a["solar_cycle"]["period_months"],
             b["solar_cycle"]["period_months"],
             a["solar_cycle"]["known_cycle_months"]))
    series = sorted(set(za) & set(zb))
    rows = []
    for s in series:
        sa, sb = za[s]["skill_pct"], zb[s]["skill_pct"]
        rows.append((s, sa, sb, sb - sa))
    rows.sort(key=lambda r: -abs(r[3]))
    print("\n%-20s %10s %10s %10s" % ("series", "v9 skill", "v10 skill", "delta"))
    for s, sa, sb, d in rows:
        print("%-20s %+9.1f%% %+9.1f%% %+9.1f" % (s, sa, sb, d))
    va = np.array([r[1] for r in rows]); vb = np.array([r[2] for r in rows])
    wins = int((vb > va).sum()); losses = int((vb < va).sum()); ties = len(rows) - wins - losses
    print("\nn=%d: v10 wins %d, v9 wins %d, ties %d | median delta %+.1f pts"
          % (len(rows), wins, losses, ties, np.median(vb - va)))
    pos_a = sum(1 for s in series if za[s]["skill_pct"] > 0)
    pos_b = sum(1 for s in series if zb[s]["skill_pct"] > 0)
    print("positive: v9 %d/%d -> v10 %d/%d" % (pos_a, len(series), pos_b, len(series)))
    if wins + losses >= 4:
        from scipy.stats import binomtest
        p = binomtest(wins, wins + losses, 0.5, alternative="two-sided").pvalue
        print("paired sign test p = %.4f" % p)
    print("\ntransfer_summary:")
    for k, v in b.get("transfer_summary", {}).items():
        print("  %s: frozen %+.1f%% vs scratch %+.1f%%" % (k, v["frozen_skill"], v["scratch_skill"]))


def figures():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import study_figures as sf
    # add v10 to the kernel reprobe list by appending its entry to the cache
    cache = os.path.join(STUDY, "reprobe.json")
    out = json.load(open(cache)) if os.path.exists(cache) else {}
    if "v10" not in out:
        import torch
        from tso.foundation import FoundationOperator, zero_shot_forecast
        meta = json.load(open(os.path.join(STUDY, "corpus_meta.json")))
        z = np.load(os.path.join(STUDY, "corpus.npz"), allow_pickle=True)
        ck = os.path.join(V10, "foundation_model.pt")
        if not os.path.exists(ck):
            print("no v10 checkpoint at", ck); return
        m = FoundationOperator(latent_dim=128, hidden=384)
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
        m.eval()
        pos = sum(1 for mm in meta
                  if zero_shot_forecast(m, z[mm["name"]])["skill_pct"] > 0)
        out["v10"] = [pos, len(meta)]
        json.dump(out, open(cache, "w"))
        print("reprobed v10: %d/%d" % (pos, len(meta)))
    os.system("cd %s && python scripts/study_figures.py scaling" % ROOT)


if __name__ == "__main__":
    {"fetch": fetch, "compare": compare, "figures": figures}[sys.argv[1]]()
