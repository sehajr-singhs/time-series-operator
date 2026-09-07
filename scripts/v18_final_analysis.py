#!/usr/bin/env python3
"""v18 final analysis: TSO vs frozen Chronos vs fine-tuned Chronos.

Merges the kernel's eval (tso_v14, chronos_frozen, chronos_generous = the
25k-step fine-tune, evaluated in-kernel) with the local re-eval of the TRUE
compute-matched checkpoint (chronos_matched = 1127 steps, equal parameter-steps
to TSO's 25k x 2.08M params).

Compute matching (exact):
    TSO v14:    2,080,678 params x 25,000 steps = 5.20e10 parameter-steps
    Chronos:   46,154,240 params x 1,127 steps  = 5.20e10 parameter-steps
    Generous:  46,154,240 params x 25,000 steps (22x the matched budget)

Writes output/kaggle_kernel_v18/final_summary.json
"""

import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = os.path.join(ROOT, "output", "kaggle_kernel_v18")


def agg(ps):
    sk = [v["skill_pct"] for v in ps]
    return {
        "n": len(sk),
        "wins_vs_persistence": int(sum(s > 0 for s in sk)),
        "pos_median": float(np.median(sk)),
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
    m = json.load(open(os.path.join(K, "metrics.json")))
    per = m["per_series"]
    matched = json.load(open(os.path.join(K, "chronos_matched_per_series.json")))
    names = list(per.keys())
    assert len(names) == 40, f"expected 40 series, got {len(names)}"
    missing = [n for n in names if n not in matched]
    assert not missing, f"matched eval incomplete: {missing}"

    # NOTE: the kernel's "chronos_frozen" column is contaminated — the
    # frozen pipe shared the model object that fine-tuning mutated in
    # place (corr 0.986 with the generous column). The TRUE frozen
    # baseline was re-measured post-hoc on the identical protocol by
    # scripts/eval_v18_frozen.py. The kernel's "chronos_matched" column
    # was actually the final 25k model (= the true generous checkpoint).
    frozen_true = json.load(open(os.path.join(
        K, "chronos_frozen_true_per_series.json")))
    tso = {n: per[n]["tso_v14"]["skill_pct"] for n in names}
    frozen = {n: frozen_true[n]["skill_pct"] for n in names}
    generous = {n: per[n]["chronos_matched"]["skill_pct"] for n in names}
    matched_m = {n: matched[n]["skill_pct"] for n in names}

    summary = {
        "config": {
            "p_tso": m["config"]["p_tso"],
            "p_chronos": m["config"]["p_chronos"],
            "n_tso_iters": m["config"]["n_tso_iters"],
            "n_matched": m["config"]["n_matched"],
            "n_total": m["config"]["n_total"],
            "tso_param_steps": m["config"]["p_tso"] * m["config"]["n_tso_iters"],
            "matched_param_steps": m["config"]["p_chronos"] * m["config"]["n_matched"],
            "generous_param_steps": m["config"]["p_chronos"] * m["config"]["n_total"],
            "protocol": "identical z-scored windows, 0.7/0.2 split, 64-sample median",
        },
        "aggregates": {
            "tso_v14": agg([{"skill_pct": v} for v in tso.values()]),
            "chronos_frozen": agg([{"skill_pct": v} for v in frozen.values()]),
            "chronos_matched_1127": agg([{"skill_pct": v} for v in matched_m.values()]),
            "chronos_generous_25k": agg([{"skill_pct": v} for v in generous.values()]),
        },
        "head_to_head": {
            "tso_vs_frozen": {
                "tso_wins": int(sum(tso[n] > frozen[n] for n in names)),
                "chronos_wins": int(sum(frozen[n] > tso[n] for n in names)),
                "wilcoxon_chronos_gt_tso": wilcoxon(
                    [frozen[n] for n in names], [tso[n] for n in names]
                ),
            },
            "tso_vs_matched": {
                "tso_wins": int(sum(tso[n] > matched_m[n] for n in names)),
                "chronos_wins": int(sum(matched_m[n] > tso[n] for n in names)),
                "wilcoxon_chronos_gt_tso": wilcoxon(
                    [matched_m[n] for n in names], [tso[n] for n in names]
                ),
            },
            "matched_vs_frozen": {
                "matched_wins": int(sum(matched_m[n] > frozen[n] for n in names)),
                "frozen_wins": int(sum(frozen[n] > matched_m[n] for n in names)),
                "wilcoxon_matched_gt_frozen": wilcoxon(
                    [matched_m[n] for n in names], [frozen[n] for n in names]
                ),
            },
            "generous_vs_frozen": {
                "generous_wins": int(sum(generous[n] > frozen[n] for n in names)),
                "frozen_wins": int(sum(frozen[n] > generous[n] for n in names)),
                "wilcoxon_generous_gt_frozen": wilcoxon(
                    [generous[n] for n in names], [frozen[n] for n in names]
                ),
            },
        },
        "per_series": {
            n: {
                "tso_v14": tso[n],
                "chronos_frozen": frozen[n],
                "chronos_matched": matched_m[n],
                "chronos_generous": generous[n],
                "matched_corr": matched[n]["corr"],
                "horizon": matched[n]["horizon"],
            }
            for n in names
        },
    }

    out = os.path.join(K, "final_summary.json")
    json.dump(summary, open(out, "w"), indent=1)
    print("wrote", out)

    # console digest
    a = summary["aggregates"]
    print("\n== aggregates (skill% vs persistence, median) ==")
    for k, v in a.items():
        print(f"  {k:26s} wins {v['wins_vs_persistence']:2d}/{v['n']}  "
              f"med {v['pos_median']:+6.2f}  mean {v['mean']:+6.2f}")
    print("\n== head-to-head ==")
    for k, v in summary["head_to_head"].items():
        print(f"  {k:22s} {v}")


if __name__ == "__main__":
    main()