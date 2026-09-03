#!/usr/bin/env python3
"""Study figures + paper_data.json for the NMI-style manuscript.

Aggregates: local multi-seed study (output/study/*.json), the Kaggle kernel
runs (output/kaggle_kernel_run/metrics.json = v5, output/kaggle_kernel_v7/
metrics.json = v7) into one machine-readable summary and renders the figures
used by the paper.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tso import viz  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "output", "study")
FIGS = os.path.join(STUDY, "figs")
os.makedirs(FIGS, exist_ok=True)
BG, FG, ACCENT, ORANGE, MAGENTA, BLUE = (viz.BG, viz.FG, viz.ACCENT,
                                         viz.ORANGE, viz.MAGENTA, viz.BLUE)
GOLD = "#d4a017"
plt.rcParams.update(viz.plt.rcParams)


def load_study():
    tr = json.load(open(os.path.join(STUDY, "transfer.json")))
    bl = json.load(open(os.path.join(STUDY, "baselines.json")))
    meta = json.load(open(os.path.join(STUDY, "corpus_meta.json")))
    full = [tr[k] for k in sorted(tr) if k.startswith("full_s")]
    return tr, bl, meta, full


def agg():
    tr, bl, meta, full = load_study()
    v5 = json.load(open(os.path.join(ROOT, "output", "kaggle_kernel_run",
                                     "metrics.json")))
    v7 = json.load(open(os.path.join(ROOT, "output", "kaggle_kernel_v7",
                                     "metrics.json")))
    names = sorted(tr["full_s0"])
    domain = {m["name"]: m["domain"] for m in meta}
    per = {}
    for n in names:
        fs = [r[n]["skill_pct"] for r in full if n in r
              and np.isfinite(r[n]["skill_pct"])]
        per[n] = {
            "domain": domain[n],
            "tso_mean": float(np.mean(fs)) if fs else None,
            "tso_std": float(np.std(fs)) if len(fs) > 1 else None,
            "tso_median": float(np.median(fs)) if fs else None,
            "no_scale": tr["no_scale_s0"][n]["skill_pct"],
            "no_arrow": tr["no_arrow_s0"][n]["skill_pct"],
            "gru": bl["gru"].get(n, {}).get("skill_pct"),
            "scratch": bl["scratch"].get(n, {}).get("skill_pct"),
        }
    v9 = json.load(open(os.path.join(ROOT, "output", "kaggle_kernel_v9",
                                     "metrics.json")))
    return {"per_series": per, "kernel_v5": v5, "kernel_v7": v7,
            "kernel_v9": v9}


def fig_comparison(data):
    """Per-series frozen TSO vs GRU vs scratch, domain-grouped, sorted."""
    per = data["per_series"]
    order = sorted(per, key=lambda n: (per[n]["domain"], -(
        per[n]["tso_median"] if per[n]["tso_median"] is not None else -1e9)))
    x = np.arange(len(order))
    y_tso = np.array([per[n]["tso_median"] for n in order], dtype=float)
    y_lo = np.array([per[n]["tso_median"] - per[n]["tso_std"]
                     if per[n]["tso_std"] else per[n]["tso_median"]
                     for n in order], dtype=float)
    y_hi = np.array([per[n]["tso_median"] + per[n]["tso_std"]
                     if per[n]["tso_std"] else per[n]["tso_median"]
                     for n in order], dtype=float)
    y_gru = np.array([per[n]["gru"] if per[n]["gru"] is not None else np.nan
                      for n in order], dtype=float)
    y_scr = np.array([per[n]["scratch"] if per[n]["scratch"] is not None
                      else np.nan for n in order], dtype=float)
    fig, ax = plt.subplots(figsize=(15, 6), dpi=160)
    ax.errorbar(x, y_tso, yerr=[y_tso - y_lo, y_hi - y_tso],
                fmt="o", color=ACCENT, markersize=5, capsize=3, ecolor=ACCENT,
                alpha=0.9, label="TSO frozen (median over 3 seeds)")
    ax.plot(x, y_gru, "s", color=MAGENTA, markersize=4, alpha=0.85,
            label="GRU autoregressive (per-series)")
    ax.plot(x, y_scr, "^", color=ORANGE, markersize=5, alpha=0.9,
            label="TSO scratch (per-series, no pretraining)")
    ax.axhline(0, color="#3a4154", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("forecast skill over persistence (%)")
    ax.set_title("Frozen zero-shot TSO vs GRU tokens vs per-series scratch "
                 "on 23 series / 8 domains", fontsize=12)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def fig_ablations(data):
    per = data["per_series"]
    names = sorted(per)
    x = np.arange(len(names))
    y_full = [per[n]["tso_median"] if per[n]["tso_median"] is not None
              else np.nan for n in names]
    y_ns = [per[n]["no_scale"] for n in names]
    y_na = [per[n]["no_arrow"] for n in names]
    fig, ax = plt.subplots(figsize=(15, 5.5), dpi=160)
    ax.plot(x, y_full, "o-", color=ACCENT, markersize=4, alpha=0.9,
            label="full (all four pretexts)")
    ax.plot(x, y_ns, "s--", color=BLUE, markersize=4, alpha=0.85,
            label="ablation: no scale covariance")
    ax.plot(x, y_na, "d-.", color=MAGENTA, markersize=4, alpha=0.85,
            label="ablation: no arrow of time")
    ax.axhline(0, color="#3a4154", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("frozen zero-shot skill over persistence (%)")
    ax.set_title("Pretext ablations: each pretext converges, but at 1.4k "
                 "iterations the pretext-free latent probes better — the "
                 "pretexts only pay off with scale (see scaling figure)",
                 fontsize=11)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_ablations.png"), bbox_inches="tight")
    plt.close(fig)


def reprobe_kernel_models():
    """Evaluate the kernel checkpoints with the CURRENT probe on the SAME
    local corpus, so the scaling comparison is apples-to-apples (probe
    versions changed between runs). Results are cached on disk keyed by tag:
    run `python scripts/study_figures.py reprobe` to (re)compute them."""
    cache = os.path.join(STUDY, "reprobe.json")
    out = {}
    if os.path.exists(cache):
        out = json.load(open(cache))
    kernels = [
        ("v5", os.path.join(ROOT, "output", "kaggle_kernel_run",
                             "foundation_model.pt"), 48, 128),
        ("v7", os.path.join(ROOT, "output", "kaggle_kernel_v7",
                             "foundation_model.pt"), 96, 256),
        ("v9", os.path.join(ROOT, "output", "kaggle_kernel_v9",
                             "foundation_model.pt"), 128, 384),
        ("v10", os.path.join(ROOT, "output", "kaggle_kernel_v10",
                              "foundation_model.pt"), 128, 384),
    ]
    for s in (0, 1, 2):
        kernels.append(
            (f"v11s{s}", os.path.join(ROOT, "output", "kaggle_kernel_v11",
                                      f"seed{s}", "foundation_model.pt"),
             256, 768))
    kernels.append(
        ("v12", os.path.join(ROOT, "output", "kaggle_kernel_v12",
                             "foundation_model.pt"), 256, 768))
    kernels.append(
        ("v13", os.path.join(ROOT, "output", "kaggle_kernel_v13",
                             "foundation_model.pt"), 128, 384))
    kernels.append(
        ("v14", os.path.join(ROOT, "output", "kaggle_kernel_v14",
                             "foundation_model.pt"), 256, 768))
    for s in (1, 2, 3):
        kernels.append(
            (f"v14s{s}", os.path.join(ROOT, "output",
                                      f"kaggle_kernel_v14_seed{s}",
                                      "foundation_model.pt"), 256, 768))
    todo = [(t, c, l, h) for t, c, l, h in kernels
            if t not in out and os.path.exists(c)]
    if not todo:
        return out
    import torch
    from tso.foundation import FoundationOperator, zero_shot_forecast
    meta = json.load(open(os.path.join(STUDY, "corpus_meta.json")))
    z = np.load(os.path.join(STUDY, "corpus.npz"), allow_pickle=True)
    for tag, ckpath, lat, hid in todo:
        m = FoundationOperator(latent_dim=lat, hidden=hid)
        m.load_state_dict(torch.load(ckpath, map_location="cpu",
                                     weights_only=True))
        m.eval()
        pos = 0
        for mm in meta:
            r = zero_shot_forecast(m, z[mm["name"]])
            if r["skill_pct"] > 0:
                pos += 1
        out[tag] = [pos, len(meta)]
        json.dump(out, open(cache, "w"))
        print(f"  reprobed {tag}: {pos}/{len(meta)}")
    return out


def fig_scaling(data):
    """Positive-fraction of frozen zero-shot vs training scale. Local study
    (seeds 0-2) and the two Kaggle kernel runs, all evaluated with the
    current probe on the same corpus."""
    per = data["per_series"]
    local_pos = sum(1 for n in per
                    if per[n]["tso_median"] is not None and
                    per[n]["tso_median"] > 0)
    local_n = len(per)
    k = reprobe_kernel_models()
    v5_pos, v5_n = k.get("v5", (2, 23))
    v7_pos, v7_n = k.get("v7", (13, 23))
    v9_pos, v9_n = k.get("v9", (0, 23))
    v10_pos, v10_n = k.get("v10", (0, 23))
    v11s = [(k.get(f"v11s{s}", (0, 23))) for s in (0, 1, 2)]
    labels = ["kernel v5\n(7-series, 4k iters,\nlat 48 · hid 128) —\nreprobed",
              "local ×3 seeds\n(23-series, 1.4k iters,\nlat 48 · hid 128)",
              "kernel v7\n(23-series, 15k iters,\nlat 96 · hid 256) —\nreprobed",
              "kernel v9\n(40-series, 25k iters,\nlat 128 · hid 384) —\nreprobed",
              "kernel v10\n(40-series, 60k iters,\nlat 128 · hid 384) —\nreprobed",
              "kernel v11 seed 0\n(40-series, 25k iters,\nlat 256 · hid 768, T4)",
              "kernel v11 seed 1\n(40-series, 25k iters,\nlat 256 · hid 768, T4)",
              "kernel v11 seed 2\n(40-series, 25k iters,\nlat 256 · hid 768, T4)",
              "kernel v12\n(40-series, 25k iters,\nlat 256 · hid 768,\njoint probe) — reprobed",
              "kernel v13\n(40 real + 176 dynamics,\n25k iters, lat 128 · hid 384,\nuniversal corpus) — reprobed"]
    v12_pos, v12_n = k.get("v12", (0, 23))
    v13_pos, v13_n = k.get("v13", (0, 23))
    v14_pos, v14_n = k.get("v14", (0, 23))
    v14s = [k.get(f"v14s{s}", (0, 23)) for s in (1, 2, 3)]
    vals = [v5_pos / v5_n, local_pos / local_n, v7_pos / v7_n,
            v9_pos / v9_n, v10_pos / v10_n]
    vals += [p / n for p, n in v11s]
    vals += [v12_pos / v12_n, v13_pos / v13_n, v14_pos / v14_n]
    vals += [p / n for p, n in v14s]
    labels += ["kernel v14\n(40 real + 192 balanced\ndynamics, 25k iters,\nlat 256 · hid 768,\ndyn-w 2.5) — reprobed"]
    labels += [f"v14 seed {s}\n(same recipe, seed {s},\nreprobed)" for s in (1, 2, 3)]
    nbars = len(vals)
    fig, ax = plt.subplots(figsize=(18.5, 5.6), dpi=160)
    x = np.arange(nbars)
    colors = ([BLUE, BLUE, ACCENT, MAGENTA, ORANGE] + [GOLD] * 3
              + ["#7ee787", "#58c4f0", "#ffb86c", "#ffb86c", "#ffb86c",
                 "#ffb86c"])
    bars = ax.bar(x, vals, color=colors, alpha=0.9, width=0.62)
    for b, (lab, v) in enumerate(zip(labels, vals)):
        ax.text(b, v + 0.01, f"{v:.0%}", ha="center", color=FG, fontsize=9)
    ax.text(3, vals[3] + 0.10, "20/40 in-kernel\n(40-series corpus)",
            ha="center", color=MAGENTA, fontsize=8)
    ax.text(4, vals[4] + 0.10, "16/40 in-kernel\n(saturated pretexts)",
            ha="center", color=ORANGE, fontsize=8)
    ax.text(5.5, max(vals[5:8]) + 0.11,
            "in-kernel 29/40 · 21/40 · 28/40\n(seed variance > width gain)",
            ha="center", color=GOLD, fontsize=8)
    ax.text(8, v12_pos / v12_n + 0.05, "joint probe", ha="center",
            color="#7ee787", fontsize=8)
    ax.text(9, v13_pos / v13_n + 0.05, "corpus breadth", ha="center",
            color="#58c4f0", fontsize=8)
    ax.text(10.5, max(vals[10:]) + 0.11,
            "v14 × 4 seeds: 14/23 · 13/23 · 13/23 · 13/23\n(all seeds above the v9 plateau;\npositive median in every seed)",
            ha="center", color="#ffb86c", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.0)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fraction of series with frozen zero-shot skill > 0")
    ax.set_title("Scaling the operator: corpus and width raise zero-shot "
                 "transfer; seed variance and the frozen probe dominate at "
                 "fixed scale (joint-probe v12 and dynamics-corpus v13: "
                 "see text; v14 = balanced corpus + forced Koopman \n"
                 "linearity at v11 width breaks the plateau: 14/23 reprobed, "
                 "31/40 in-kernel, first positive median skill)",
                 fontsize=11.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_scaling.png"), bbox_inches="tight")
    plt.close(fig)
    return {"local": {"positive": local_pos, "n": local_n},
            "v5": {"positive": v5_pos, "n": v5_n},
            "v7": {"positive": v7_pos, "n": v7_n},
            "v9": {"positive": v9_pos, "n": v9_n},
            "v10": {"positive": v10_pos, "n": v10_n},
            "v11": {"seeds": [{"positive": p, "n": n}
                                for p, n in v11s]},
            "v12": {"positive": v12_pos, "n": v12_n},
            "v13": {"positive": v13_pos, "n": v13_n},
            "v14": {"positive": v14_pos, "n": v14_n}}


def fig_reversibility():
    """Weiss T3 on held-out data vs in the frozen latent (full_s0)."""
    from tso.foundation import FoundationOperator, prepare_pair
    import torch
    from tso.pipeline import normalize
    meta = json.load(open(os.path.join(STUDY, "corpus_meta.json")))
    z = np.load(os.path.join(STUDY, "corpus.npz"), allow_pickle=True)
    ck = torch.load(os.path.join(STUDY, "foundation_full_s0.pt"),
                    map_location="cpu", weights_only=True)
    model = FoundationOperator(latent_dim=48, hidden=128)
    model.load_state_dict(ck["model"])
    model.eval()
    def t3(s):
        dx = np.diff(np.asarray(s, dtype=float), axis=0)
        m2 = np.mean(dx ** 2)
        return float(np.mean(dx ** 3) / m2 ** 1.5) if m2 > 1e-12 else np.nan
    rows = []
    for m in meta:
        name = m["name"]
        pair = prepare_pair(z[name])
        with torch.no_grad():
            zl = model.phi(torch.tensor(pair["fine"],
                                        dtype=torch.float32)).numpy()
        rows.append((name, m["domain"], t3(z[name]), t3(zl)))
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    dom_col = {}
    uniq = sorted({r[1] for r in rows})
    cm = plt.get_cmap("turbo")
    for i, d in enumerate(uniq):
        dom_col[d] = cm(i / max(len(uniq) - 1, 1))
    for name, d, td, tl in rows:
        ax.scatter(td, tl, s=34, color=dom_col.get(d, "white"),
                   edgecolors="white", linewidths=0.3, alpha=0.9)
        ax.annotate(name.split("-")[-1], (td, tl), fontsize=6, color="#8b93a7",
                    xytext=(3, 3), textcoords="offset points")
    vals = [abs(v) for r in rows for v in (r[2], r[3]) if np.isfinite(v)]
    lim = max(vals) * 1.2 if vals else 1.0
    ax.plot([-lim, lim], [-lim, lim], color="#2c3450", linewidth=1,
            linestyle="--", label="latent == data (perfect arrow preservation)")
    ax.axhline(0, color="#3a4154", linewidth=0.6)
    ax.axvline(0, color="#3a4154", linewidth=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Weiss T3 of the raw series (irreversibility of the data)")
    ax.set_ylabel("Weiss T3 of the frozen latent trajectory")
    ax.set_title("The arrow of time in the latent: the operator learns to "
                 "classify direction (94%) yet preserves only part of the "
                 "data's third-order asymmetry", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_reversibility.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig_efficiency():
    """Data-efficiency crossover on the held-out sunspots series: the frozen
    operator beats per-target scratch when training data is scarce."""
    eff = json.load(open(os.path.join(STUDY, "efficiency.json")))
    pts = sorted((k, v) for k, v in eff.items() if v["name"] == "sunspots")
    tf = [v["train_frac"] for _, v in pts]
    fz = [v["frozen_skill"] for _, v in pts]
    sc = [v["scratch_skill"] for _, v in pts]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(tf, fz, "o-", color=ACCENT, linewidth=1.8, markersize=6,
            label="frozen pretrained operator")
    ax.plot(tf, sc, "s--", color=ORANGE, linewidth=1.8, markersize=6,
            label="per-target scratch (same architecture)")
    ax.axhline(0, color="#3a4154", linewidth=0.8)
    ax.set_xlabel("training fraction of the held-out sunspots series")
    ax.set_ylabel("zero-shot skill over persistence (%)")
    ax.set_title("The data-efficiency crossover: transfer wins when data is "
                 "scarce, per-target training wins when it is abundant",
                 fontsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_efficiency.png"), bbox_inches="tight")
    plt.close(fig)


def fig_external():
    """TSO (v14, best run: balanced corpus + forced Koopman linearity) vs
    external foundation models (Chronos, Moirai) vs the GRU token baseline,
    all on the same 40-series protocol with the same skill-vs-persistence
    metric."""
    src = os.path.join(ROOT, "output", "external_baselines.json")
    if not os.path.exists(src):
        print("  skip fig_external: no external_baselines.json")
        return None
    ext = json.load(open(src))
    if "chronos" not in ext or not ext["chronos"]:
        print("  skip fig_external: chronos empty")
        return None
    v14 = json.load(open(os.path.join(ROOT, "output", "kaggle_kernel_v14",
                                      "metrics.json")))
    zs, gru = v14["zero_shot"], v14["gru_baseline"]
    names = [m["name"] for m in v14["corpus"]]
    series = [n for n in names if n in zs and n in ext["chronos"]]

    def skill(m, n):
        return m[n]["skill_pct"] if n in m else float("nan")

    models = {"TSO v14 (frozen)": zs, "Chronos-t5-small": ext["chronos"]}
    if "moirai" in ext and ext["moirai"]:
        models["Moirai-small"] = ext["moirai"]
    models["GRU (per-series)"] = {n: {"skill_pct": gru[n]}
                                  for n in series if n in gru}

    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=160)
    x = np.arange(len(series))
    w = 0.72 / len(models)
    colors = [GOLD, ACCENT, ORANGE, MAGENTA]
    for k, (lab, m) in enumerate(models.items()):
        vals = [skill(m, n) for n in series]
        med = float(np.nanmedian(vals))
        ax.bar(x + (k - len(models) / 2 + 0.5) * w, vals, width=w * 0.9,
               color=colors[k], alpha=0.85, label=f"{lab} (median {med:+.0f}%)")
    ax.axhline(0, color=FG, lw=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("weather-", "").replace("grid-", "").
                        replace("coin-", "").replace("covid-", "")[:12]
                        for n in series], rotation=70, fontsize=6.5)
    ax.set_ylabel("skill vs persistence (%)")
    ax.set_title("Zero-shot transfer on 40 held-out series: TSO vs external "
                 "foundation models and the GRU token baseline", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_external.png"), bbox_inches="tight")
    plt.close(fig)
    # summary numbers for the paper
    out = {}
    for lab, m in models.items():
        vals = [skill(m, n) for n in series]
        wins = sum(1 for n in series if skill(m, n) > 0)
        out[lab] = {"median": round(float(np.nanmedian(vals)), 1),
                    "positive": wins, "n": len(series)}
    # head-to-head: does the operator beat Chronos on each shared series?
    h2h = sum(1 for n in series
              if skill(zs, n) > skill(ext["chronos"], n))
    out["chronos_h2h_wins"] = {"tso": h2h, "chronos": len(series) - h2h,
                                "n": len(series)}
    print("  external summary:", out)
    return out


def fig_solar(kernel_dir="kaggle_kernel_v9"):
    """Scale-covariant solar-cycle discovery with the frozen v8 kernel latent."""
    import sys as _sys
    _sys.path.insert(0, ROOT)
    import torch
    from tso.foundation import solar_cycle_discovery, FoundationOperator
    from tso.viz import plot_solar_discovery
    ck_path = os.path.join(ROOT, "output", kernel_dir, "foundation_model.pt")
    if not os.path.exists(ck_path):
        print("  skip fig_solar: no", ck_path)
        return None
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = FoundationOperator(latent_dim=128, hidden=384)
    model.load_state_dict(ck)
    model.eval()
    corpus = np.load(os.path.join(STUDY, "corpus.npz"), allow_pickle=True)
    d = solar_cycle_discovery(model, corpus["sunspots"])
    plot_solar_discovery(d["rows"],
                         os.path.join(FIGS, "fig_solar_discovery.png"),
                         known_months=d["known_cycle_months"])
    print("  solar discovery:", d["period_months"], "months vs",
          d["known_cycle_months"])
    return d


def main():
    import sys as _sys
    want = set(_sys.argv[1:]) if len(_sys.argv) > 1 else None
    data = agg()
    if not want or "comparison" in want:
        fig_comparison(data)
    if not want or "ablations" in want:
        fig_ablations(data)
    scaling = None
    if not want or "scaling" in want:
        scaling = fig_scaling(data)
    if want and "reprobe" in want:
        reprobe_kernel_models()
        return
    if not want or "reversibility" in want:
        fig_reversibility()
    if not want or "efficiency" in want:
        if os.path.exists(os.path.join(STUDY, "efficiency.json")):
            fig_efficiency()
    if not want or "solar" in want:
        fig_solar()
    if not want or "external" in want:
        fig_external()
    with open(os.path.join(STUDY, "paper_data.json"), "w") as fh:
        json.dump(data, fh, indent=1)
    print("figures + paper_data.json written to", STUDY)
    print("scaling:", scaling)


if __name__ == "__main__":
    main()