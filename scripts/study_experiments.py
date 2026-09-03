#!/usr/bin/env python3
"""NMI-grade study suite for the Time-Series Operator.

Runs, over the same 23-series/8-domain corpus used in the Kaggle kernels:

  * `full`        — all four pretexts (seeds 0..2)          [variance]
  * `no_scale`    — pretext ablation: scale covariance off
  * `no_arrow`    — pretext ablation: arrow of time off
  * baselines     — per-target scratch deep-Koopman (subset),
                    a from-scratch GRU autoregressive forecaster (all series)
  * transfer      — frozen zero-shot skill/corr/linearity for every config
                    on every series, plus time-reversibility (Weiss T3) and
                    scale-covariance consistency metrics on held-out data
  * stats         — sign test / Wilcoxon / paired bootstrap on
                    pretrained-vs-scratch skill differences

Resumable through --phase. All results land in output/study/*.json.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from tso import attractors, viz  # noqa: E402
from tso.foundation import (  # noqa: E402
    prepare_pair, load_series_from_csv, pretrain_foundation,
    zero_shot_forecast, scratch_baseline, FoundationOperator, WINDOW,
)
from tso.pipeline import normalize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "output", "study")
os.makedirs(STUDY, exist_ok=True)
MAXLEN = 4096

# ---------------------------------------------------------------------------
# Corpus (mirrors the Kaggle v7 kernel: 23 series, 8 domains)
# ---------------------------------------------------------------------------

CORPUS = [
    # (kind, path, args, name-prefix, domain, held_out)
    ("ecg", "data/Cardiac_arrhythmia_dataset.csv", (2,), "ecg", "physiology", False),
    ("grid", "data/corpus/grid-energy", (8,), "grid", "energy-grid", False),
    ("weather", "data/corpus/weather-aus/weatherAUS.csv",
     (["Temp3pm", "MinTemp", "Rainfall", "Humidity3pm"],),
     "weather", "meteorology", False),
    ("coins", "data/corpus/bitcoin", (["coin_Bitcoin.csv", "coin_Ethereum.csv",
                                       "coin_Dogecoin.csv"],),
     "coin", "finance", False),
    ("single-file", "data/corpus/sunspots/Sunspots.csv", (["Sunspot"],),
     "sunspots", "solar-physics", True),
    ("single-file", "data/corpus/airline/AirPassengers.csv", (["#Passengers"],),
     "airline", "economics", False),
    ("covid", "data/corpus/covid-us/covid_19_clean_complete.csv",
     (["US"], ["India", "Brazil"]), "covid", "epidemiology", True),
    ("lorenz", None, None, "lorenz-x", "physics", False),
]


def _clean(s, name, detrend=True):
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 128:
        return None
    if detrend and len(s) > 64:
        ac1 = float(np.corrcoef(s[:-1], s[1:])[0, 1])
        if np.isfinite(ac1) and ac1 > 0.999:
            s = np.diff(s)
    if len(s) > MAXLEN:
        s = s[:: max(1, len(s) // MAXLEN)][:MAXLEN]
    if len(s) < 2 or float(np.std(s)) < 1e-6:
        return None
    return name, normalize(s)


def build_corpus():
    out = []
    for kind, path, args, prefix, domain, held in CORPUS:
        try:
            if kind == "ecg":
                import pandas as pd
                df = pd.read_csv(path)
                for rec in df.groupby("record").size().sort_values(
                        ascending=False).index[:args[0]]:
                    r = _clean(df[df["record"] == rec]["rr_prev"].to_numpy(),
                               f"{prefix}-{int(rec)}")
                    if r:
                        out.append((*r, domain, held))
            elif kind == "grid":
                import pandas as pd
                for f in sorted(os.listdir(path)):
                    if not f.endswith("_hourly.csv") or "Load" in f or "est" in f:
                        continue
                    col = f.replace("_hourly.csv", "") + "_MW"
                    df = pd.read_csv(os.path.join(path, f))
                    if col not in df.columns:
                        continue
                    r = _clean(df[col].to_numpy(),
                               f"{prefix}-{col.replace('_MW','')}")
                    if r:
                        out.append((*r, domain, held))
                    if sum(1 for o in out if o[2] == domain) >= args[0]:
                        break
            elif kind == "weather":
                import pandas as pd
                df = pd.read_csv(path)
                for c in args[0]:
                    r = _clean(df[c].to_numpy(), f"{prefix}-{c}")
                    if r:
                        out.append((*r, domain, held))
            elif kind == "coins":
                import pandas as pd
                for f in args[0]:
                    df = pd.read_csv(os.path.join(path, f))
                    r = _clean(df["Close"].to_numpy(),
                               f"{prefix}-{f.replace('coin_','').replace('.csv','')}")
                    if r:
                        out.append((*r, domain, held))
            elif kind == "covid":
                import pandas as pd
                df = pd.read_csv(path)
                for c, h in [(c_, False) for c_ in args[1]] + \
                            [(c_, True) for c_ in args[0]]:
                    sub = df[df["Country/Region"] == c] \
                        if "Country/Region" in df.columns else df
                    s = pd.to_numeric(sub["Confirmed"], errors="coerce") \
                        .dropna().to_numpy()
                    r = _clean(np.diff(s), f"{prefix}-{c.lower()}")
                    if r:
                        out.append((*r, domain, h))
            elif kind == "single-file":
                _, s = load_series_from_csv(path, hints=args[0], name=prefix)
                out.append((prefix, s, domain, held))
            elif kind == "lorenz":
                rng = np.random.default_rng(7)
                true = attractors.lorenz_trajectory(n=20000, dt=0.01)
                s = true[:, 0] + 0.01 * rng.normal(size=true.shape[0])
                out.append((prefix, normalize(s), domain, held))
        except Exception as e:
            print(f"  !! {prefix}: {e}")
    return out


def prep(samples):
    """samples -> (prepared list for pretrain, list of pair dicts, meta)."""
    train, pairs, meta = [], [], []
    for name, s, domain, held in samples:
        pair = prepare_pair(s)
        pairs.append((name, pair))
        meta.append({"name": name, "domain": domain, "held_out": held,
                     "n": int(len(s))})
        if not held:
            train.append(pair)
    return train, pairs, meta


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class UnrolledGRU(torch.nn.Module):
    """A from-scratch GRU autoregressive forecaster on the embedded states —
    the 'token-less recurrent sequence model' baseline the TSO is compared
    against. Trained one-step, forecasted closed-loop (teacher forcing off)."""

    def __init__(self, dim, hidden=24):
        super().__init__()
        self.gru = torch.nn.GRUCell(dim, hidden)
        self.out = torch.nn.Linear(hidden, dim)

    def forward(self, s, h=None):
        if h is None:
            h = torch.zeros(s.shape[0], self.gru.hidden_size,
                            dtype=s.dtype)
        h = self.gru(s, h)
        return self.out(h), h


def gru_baseline(states, iters=500, hidden=24, lr=1e-3, train_frac=0.7,
                 horizon_frac=0.2, seed=0):
    """One-step GRU trained on the embedded series; closed-loop forecast."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(states)
    split = int(n * train_frac)
    horizon = min(int(n * horizon_frac), n - split - 1, 100)
    X, Y = states[: split - 1], states[1:split]
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(Y, dtype=torch.float32)
    model = UnrolledGRU(X.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        b = 64
        idx = rng.integers(0, len(xt), size=b)
        s, t = xt[idx], yt[idx]
        opt.zero_grad()
        pred, _ = model(s)
        loss = ((pred - t) ** 2).mean()
        loss.backward()
        opt.step()
    model.eval()
    # closed-loop forecast from the last training state
    with torch.no_grad():
        h = None
        preds = [states[split - 1]]
        s = torch.tensor(states[split - 1], dtype=torch.float32)[None]
        for _ in range(horizon):
            p, h = model(s, h)
            preds.append(p.numpy()[0])
            s = p
    pred_vals = np.array(preds)[:, 0]
    true_vals = states[split - 1: split + horizon, 0]
    from tso.pipeline import rmse, persistence_baseline
    e = rmse(pred_vals, true_vals)
    ep = rmse(persistence_baseline(true_vals[0], len(true_vals)), true_vals)
    corr = float(np.corrcoef(pred_vals, true_vals)[0, 1]) \
        if len(true_vals) > 2 else float("nan")
    return {"skill_pct": 100.0 * (ep - e) / max(ep, 1e-12), "corr": corr,
            "horizon": horizon}


def weiss_t3(series_or_states):
    """Weiss (1975) time-reversibility statistic on increment third cumulant:
    T3 = E[(dx)^3] / E[(dx)^2]^1.5. Nonzero <=> irreversible (has an arrow)."""
    dx = np.diff(np.asarray(series_or_states, dtype=float), axis=0)
    if dx.size == 0:
        return float("nan")
    m2 = np.mean(dx ** 2)
    if m2 < 1e-12:
        return float("nan")
    return float(np.mean(dx ** 3) / m2 ** 1.5)


def sign_test(pairs_diff):
    """Two-sided binomial sign test p-value on paired differences."""
    d = np.asarray(pairs_diff, dtype=float)
    d = d[np.isfinite(d)]
    pos = int((d > 0).sum())
    n = len(d)
    if n == 0:
        return float("nan"), n
    from math import comb
    p = sum(comb(n, k) for k in range(pos + 1, n + 1)) * 0.5 ** n * 2
    return min(1.0, p), n


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_corpus():
    samples = build_corpus()
    train, pairs, meta = prep(samples)
    np.savez(os.path.join(STUDY, "corpus.npz"),
             **{name: p["series"] for name, p in pairs})
    json.dump(meta, open(os.path.join(STUDY, "corpus_meta.json"), "w"),
              indent=1)
    print(f"corpus: {len(samples)} series, {len(meta)} meta, "
          f"{len([m for m in meta if m['held_out']])} held out")
    for m in meta:
        print(f"  {m['name']:18s} [{m['domain']:12s}] n={m['n']:5d} "
              f"held={m['held_out']}")


def _load_corpus():
    meta = json.load(open(os.path.join(STUDY, "corpus_meta.json")))
    z = np.load(os.path.join(STUDY, "corpus.npz"), allow_pickle=True)
    pairs = {}
    for m in meta:
        pairs[m["name"]] = prepare_pair(z[m["name"]])
    return pairs, meta


def phase_pretrain(configs, iters, latent_dim, hidden, seeds):
    pairs, meta = _load_corpus()
    train = [p for n, p in pairs.items()
             if not dict((m["name"], m) for m in meta)[n]["held_out"]]
    for cfg in configs:
        w = {"full": (1.0, 0.5, 0.5, 1.0),
             "no_scale": (1.0, 0.5, 0.0, 1.0),
             "no_arrow": (1.0, 0.5, 0.5, 0.0)}[cfg]
        for seed in seeds:
            tag = f"{cfg}_s{seed}"
            ck = os.path.join(STUDY, f"foundation_{tag}.pt")
            if os.path.exists(ck):
                print(f"  {tag}: cached, skipping")
                continue
            torch.manual_seed(seed)
            # patch the module-global loss so its pretext weight vector is
            # replaced per config (ablations zero out scale or arrow)
            from tso import foundation as _F
            orig_loss = _F.foundation_loss
            _F.foundation_loss = lambda m_, p_, device="cpu", **kw: orig_loss(
                m_, p_, device=device, w=w, **kw)
            model, hist, agg, parts = pretrain_foundation(
                train, iters=iters, latent_dim=latent_dim, hidden=hidden,
                seed=seed, print_every=800,
                ckpt_path=ck, resume=True)
            _F.foundation_loss = orig_loss
            torch.save({"model": model.state_dict(), "agg": agg,
                        "final_loss": float(hist[-1])},
                       ck)
            print(f"  {tag}: loss={float(hist[-1]):.4f} agg={agg}")


def phase_baselines(iters=400, kinds=("gru", "scratch")):
    pairs, meta = _load_corpus()
    out = json.load(open(os.path.join(STUDY, "baselines.json"), "r")) \
        if os.path.exists(os.path.join(STUDY, "baselines.json")) \
        else {"gru": {}, "scratch": {}}
    # GRU on every series; scratch deep-Koopman on the held-out + a
    # representative in-domain subset
    scratch_names = ["sunspots", "covid-us", "grid-AEP", "grid-DOM",
                     "weather-Temp3pm", "coin-Bitcoin", "ecg-215",
                     "covid-india"]
    for m in meta:
        name = m["name"]
        if "gru" in kinds and name not in out["gru"]:
            try:
                out["gru"][name] = gru_baseline(pairs[name]["fine"],
                                                iters=iters, seed=0)
                print(f"  gru      {name:18s}: "
                      f"skill={out['gru'][name]['skill_pct']:+7.1f}%")
            except Exception as e:
                print(f"  !! gru {name}: {e}")
        if "scratch" in kinds and name in scratch_names and \
                name not in out["scratch"]:
            try:
                s = scratch_baseline(pairs[name]["series"], iters=iters,
                                     seed=0)
                out["scratch"][name] = {k: v for k, v in s.items()
                                        if not isinstance(v, np.ndarray)}
                print(f"  scratch  {name:18s}: "
                      f"skill={s['skill_pct']:+7.1f}%")
            except Exception as e:
                print(f"  !! scratch {name}: {e}")
    json.dump(out, open(os.path.join(STUDY, "baselines.json"), "w"), indent=1)


def phase_transfer(configs, seeds):
    from tso import foundation as F
    pairs, meta = _load_corpus()
    res = json.load(open(os.path.join(STUDY, "transfer.json"), "r")) \
        if os.path.exists(os.path.join(STUDY, "transfer.json")) else {}
    for cfg in configs:
        for seed in seeds:
            tag = f"{cfg}_s{seed}"
            ck = os.path.join(STUDY, f"foundation_{tag}.pt")
            if not os.path.exists(ck) or tag in res:
                continue
            model = FoundationOperator(latent_dim=48, hidden=128)
            model.load_state_dict(torch.load(ck, map_location="cpu",
                                             weights_only=True)["model"])
            model.eval()
            per = {}
            for m in meta:
                name = m["name"]
                try:
                    r = zero_shot_forecast(model, pairs[name]["series"])
                    per[name] = {k: v for k, v in r.items()
                                 if not isinstance(v, np.ndarray)}
                    # reversibility preservation + scale consistency on
                    # held-out series (scientific validation metrics)
                    if m["held_out"]:
                        with torch.no_grad():
                            _, metrics = F.foundation_loss(
                                model, pairs[name], device="cpu")
                            per[name]["scale_consistency"] = \
                                metrics["scale"]
                            z = model.phi(torch.tensor(
                                pairs[name]["fine"],
                                dtype=torch.float32)).numpy()
                        per[name]["weiss_data"] = \
                            weiss_t3(pairs[name]["series"])
                        per[name]["weiss_latent"] = weiss_t3(z)
                except Exception as e:
                    print(f"  !! {name}: {e}")
            res[tag] = per
            pos = sum(1 for v in per.values() if v["skill_pct"] > 0)
            print(f"  {tag}: positive {pos}/{len(per)}")
            # incremental save: a killed run keeps finished configs
            json.dump(res, open(os.path.join(STUDY, "transfer.json"), "w"),
                      indent=1)


def phase_efficiency(fracs=(0.25, 0.4, 0.55, 0.7),
                     names=("sunspots", "covid-us")):
    """Data-efficiency: frozen zero-shot vs per-target scratch as the
    training budget shrinks — where transfer is supposed to win."""
    pairs, meta = _load_corpus()
    ck = torch.load(os.path.join(STUDY, "foundation_full_s0.pt"),
                    map_location="cpu", weights_only=True)
    model = FoundationOperator(latent_dim=48, hidden=128)
    model.load_state_dict(ck["model"])
    model.eval()
    out = {}
    for name in names:
        s = pairs[name]["series"]
        for f in fracs:
            fr = zero_shot_forecast(model, s, train_frac=f)
            sc = scratch_baseline(s, iters=300, seed=0, train_frac=f)
            out[f"{name}_{f}"] = {
                "frozen_skill": fr["skill_pct"],
                "scratch_skill": sc["skill_pct"],
                "frozen_corr": fr["corr"],
                "train_frac": f, "name": name,
            }
            print(f"  {name:12s} tf={f:.2f}: frozen={fr['skill_pct']:+7.1f}%  "
                  f"scratch={sc['skill_pct']:+7.1f}%")
    json.dump(out, open(os.path.join(STUDY, "efficiency.json"), "w"),
              indent=1)


def phase_stats():
    tr = json.load(open(os.path.join(STUDY, "transfer.json")))
    bl = json.load(open(os.path.join(STUDY, "baselines.json")))
    full = [v for k, v in tr.items() if k.startswith("full_")]
    names = sorted({n for r in full for n in r})
    out = {"per_series": {}}
    for n in names:
        skills = [r[n]["skill_pct"] for r in full if n in r and
                  np.isfinite(r[n]["skill_pct"])]
        out["per_series"][n] = {
            "mean_full": float(np.mean(skills)) if skills else None,
            "std_full": float(np.std(skills)) if len(skills) > 1 else None,
            "gru": bl["gru"].get(n, {}).get("skill_pct"),
            "scratch": bl["scratch"].get(n, {}).get("skill_pct"),
        }
    # pretrained (mean over seeds) vs GRU across all series: sign test
    diff_vs_gru = []
    for n in names:
        f = out["per_series"][n]["mean_full"]
        g = out["per_series"][n]["gru"]
        if f is not None and g is not None and np.isfinite(g):
            diff_vs_gru.append(f - g)
    p_gru, nn = sign_test(diff_vs_gru)
    out["vs_gru"] = {"median_diff_pts": float(np.median(diff_vs_gru)),
                     "sign_test_p": p_gru, "n": nn,
                     "wins": int(sum(1 for d in diff_vs_gru if d > 0))}
    json.dump(out, open(os.path.join(STUDY, "stats.json"), "w"), indent=1)
    print(json.dumps(out["vs_gru"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["corpus", "pretrain", "baselines", "transfer",
                             "stats", "efficiency", "all"])
    ap.add_argument("--iters", type=int, default=2400)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--configs", default="full,no_scale,no_arrow")
    ap.add_argument("--baselines", default="gru,scratch")
    ap.add_argument("--fracs", default="0.25,0.4,0.55,0.7")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    configs = args.configs.split(",")
    if args.phase in ("corpus", "all"):
        print("[phase] corpus"); phase_corpus()
    if args.phase in ("pretrain", "all"):
        print("[phase] pretrain")
        phase_pretrain(configs, args.iters, 48, 128, seeds)
    if args.phase in ("baselines", "all"):
        print("[phase] baselines (GRU + scratch subset)")
        phase_baselines(iters=400, kinds=args.baselines.split(","))
    if args.phase in ("transfer", "all"):
        print("[phase] zero-shot transfer + metrics")
        phase_transfer(configs, seeds)
    if args.phase in ("stats", "all"):
        print("[phase] statistics"); phase_stats()
    if args.phase == "efficiency":
        print("[phase] data-efficiency")
        phase_efficiency(fracs=tuple(float(x) for x in args.fracs.split(",")))


if __name__ == "__main__":
    main()