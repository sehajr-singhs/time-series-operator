"""Kernel driver: uses the inlined tso namespace (merged by
scripts/build_kernel_single.py), so no `from tso import ...` is allowed here.

v13 (this driver): corpus-breadth experiment at the v9/v10 width (latent
128 / hidden 384, 25k iters, identical 40-series in-kernel probe + GRU +
scratch + solar-cycle protocol). The only change vs v9 is the pretraining
corpus: 34 real series tiled x8 plus a ~176-series universal dynamics
battery (attractor parameter sweeps, maps, stochastic regimes).
Joint-probe loss is off (v12 negative result).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

INPUT = "/kaggle/input"
OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)

MAXLEN = 4096


def ensure_gpu_compatible():
    """If the assigned GPU can't run the image's torch (Kaggle sometimes
    gives a P100 sm_60 that modern torch no longer ships kernels for),
    install a compatible torch and restart once. A marker file prevents any
    install+restart loop; on second failure we just fall back to CPU."""
    def gpu_arch_supported():
        if not torch.cuda.is_available():
            return False
        archs = torch.cuda.get_arch_list()
        cap = torch.cuda.get_device_capability(0)
        tag = f"sm_{cap[0]}{cap[1]}"
        if archs and not any(tag in a or f"compute_{cap[0]}{cap[1]}" in a
                             for a in archs):
            print(f"  torch arch list {archs} does not cover {tag}")
            return False
        try:
            x = torch.ones(8, device="cuda")
            _ = (x + x).sum().item()
            return True
        except Exception as e:
            print(f"  CUDA test op failed: {e}")
            return False

    if gpu_arch_supported():
        cap = torch.cuda.get_device_capability(0)
        print(f"  GPU OK: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
        return
    if not torch.cuda.is_available():
        print("  no CUDA available; running on CPU")
        return
    cap = torch.cuda.get_device_capability(0)
    marker = os.path.join(OUT, "torch_reinstall_attempted")
    if os.path.exists(marker):
        print(f"  GPU sm_{cap[0]}{cap[1]} still unusable after reinstall "
              f"attempt — running on CPU")
        return
    open(marker, "w").close()
    print(f"  GPU {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} not "
          f"supported by image torch; installing torch 2.4.1 (cu121)")
    rc = os.system("pip install -q torch==2.4.1 "
                   "--index-url https://download.pytorch.org/whl/cu121")
    print(f"  pip install rc={rc}; restarting with the new torch")
    os.execv(sys.executable, [sys.executable] + sys.argv)


def find_csv(fname):
    """Locate a dataset file anywhere under /kaggle/input — mount dir names
    vary (slug, owner-slug), so a recursive filename scan is robust."""
    if not os.path.isdir(INPUT):
        return None
    for root, _dirs, files in os.walk(INPUT):
        if fname in files:
            return os.path.join(root, fname)
    for root, _dirs, files in os.walk(INPUT):
        for f in files:
            if f.lower().endswith(".csv"):
                return os.path.join(root, f)
    return None


def prepare_series(s, name, detrend=True):
    """Clean one 1-D series: dropna, unit-root diff, subsample, normalize."""
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


def load_ecg_records(path, n=3):
    df = pd.read_csv(path)
    out = []
    for rec in df.groupby("record").size().sort_values(
            ascending=False).index[:n]:
        r = prepare_series(df[df["record"] == rec]["rr_prev"].to_numpy(),
                           f"ecg-{int(rec)}")
        if r:
            out.append(r)
    return out


def load_grid_regions(path):
    d = os.path.dirname(path)
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith("_hourly.csv") or "Load" in f or "est" in f:
            continue
        col = f.replace("_hourly.csv", "") + "_MW"
        try:
            df = pd.read_csv(os.path.join(d, f))
        except Exception:
            continue
        if col not in df.columns:
            continue
        r = prepare_series(df[col].to_numpy(), col.replace("_MW", ""))
        if r:
            out.append(r)
    return out


def load_weather_cols(path, cols):
    df = pd.read_csv(path)
    out = []
    for c in cols:
        r = prepare_series(pd.to_numeric(df[c], errors="coerce").to_numpy(),
                           "weather-" + c)
        if r:
            out.append(r)
    return out


def load_crypto_coins(path, n=8):
    """The 8 coins with the longest observed histories."""
    d = os.path.dirname(path)
    cands = []
    for f in sorted(os.listdir(d)):
        if not f.startswith("coin_") or not f.endswith(".csv"):
            continue
        try:
            df = pd.read_csv(os.path.join(d, f), usecols=["Close"])
        except Exception:
            continue
        s = df["Close"].dropna().to_numpy()
        if len(s) >= 256:
            cands.append((len(s), f))
    cands.sort(reverse=True)
    out = []
    for _, f in cands[:n]:
        df = pd.read_csv(os.path.join(d, f))
        r = prepare_series(df["Close"].to_numpy(),
                           f.replace("coin_", "").replace(".csv", ""))
        if r:
            out.append(r)
    return out


def load_covid_countries(path, held, train):
    df = pd.read_csv(path)
    out = []
    for c, h in [(c_, False) for c_ in train] + [(c_, True) for c_ in held]:
        sub = df[df["Country/Region"] == c] if "Country/Region" in df.columns \
            else df
        s = pd.to_numeric(sub["Confirmed"], errors="coerce").dropna().to_numpy()
        r = prepare_series(np.diff(s), f"covid-{c.lower()}")
        if r:
            out.append((r[0], r[1], h))
    return out


def load_synthetic():
    rng = np.random.default_rng(7)
    out = []
    true = lorenz_trajectory(n=20000, dt=0.01)
    r = prepare_series(true[:, 0] + 0.01 * rng.normal(size=true.shape[0]),
                       "lorenz-x")
    if r:
        out.append(r)
    ross = rossler_trajectory(n=20000, dt=0.02)
    r = prepare_series(ross[:, 0], "rossler-x")
    if r:
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# V13: universal dynamics battery. Tests the corpus-breadth hypothesis the
# Chronos comparison pinned down (Sec. external): a pretraining corpus of
# *diverse dynamics* — attractors swept across their bifurcation diagrams,
# dissipative maps, and stochastic regimes (long memory, volatility
# clustering, regime switching, bursting, synchronization) — should move the
# frozen-probe plateau more than extra iterations, width, or probe
# regularization did in v10/v11/v12. All entries are synthetic, deterministic
# (fixed seed/configs) and normalized by prepare_series.
# ---------------------------------------------------------------------------

def _rk4_obs(f, y0, n, dt, obs, burn=300):
    y = np.asarray(y0, dtype=float).copy()
    for _ in range(burn):
        k1 = f(y); k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    out = np.empty(n)
    for i in range(n):
        k1 = f(y); k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[i] = obs(y)
    return out


def load_dynamics_battery(n_series=240, n=2600, seed=11):
    """Deterministic battery of synthetic dynamical systems. Returns a list
    of (name, series) with name = f"syn-{family}-{cfg}". Continuous flows are
    observed through one scalar channel; every flow also gets a 5%-noise
    twin so the operator sees observation noise, not just clean attractors.
    """
    out = []

    def emit(family, cfg, x, noisy=False):
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):      # divergent config: drop, no leak
            return
        if noisy:
            r = np.random.default_rng(hash((family, cfg)) & 0xffffffff)
            x = x + 0.05 * float(np.std(x)) * r.normal(size=len(x))
        tag = f"syn-{family}-{cfg}"
        r = prepare_series(x, tag)
        if r:
            out.append(r)

    # --- continuous flows (RK4, one scalar channel) ---
    lz = lambda rho: lambda y: np.array(  # noqa: E731
        [10.0 * (y[1] - y[0]),
         y[0] * (rho - y[2]) - y[1],
         y[0] * y[1] - (8.0 / 3.0) * y[2]])
    for rho in (8.0, 14.0, 24.0, 28.0, 60.0, 99.0):
        x = _rk4_obs(lz(rho), (1.0, 1.0, 1.0), n, 0.008,
                     lambda y: float(y[0]), burn=500)
        emit("lorenz", f"rho{rho:g}", x)
        emit("lorenz", f"rho{rho:g}", x, noisy=True)
    rs = lambda a: lambda y: np.array(  # noqa: E731
        [-y[1] - y[2], y[0] + a * y[1], 2.0 + y[2] * (y[0] - 4.0)])
    for a in (0.05, 0.15, 0.25, 0.35, 0.386, 0.5):
        x = _rk4_obs(rs(a), (1.0, 1.0, 1.0), n, 0.02,
                     lambda y: float(y[0]), burn=800)
        emit("rossler", f"a{a:g}", x)
        emit("rossler", f"a{a:g}", x, noisy=True)
    # Duffing has an explicit drive term; integrate with a stored phase
    for g in (0.1, 0.25, 0.32, 0.37, 0.4, 0.6):
        def f(y, g=g, w=1.2, phase=[0.0]):
            phase[0] += 0.04
            return np.array([y[1], -0.3 * y[1] + y[0] - y[0] ** 3
                             + g * np.cos(w * phase[0])])
        x = _rk4_obs(f, (0.0, 0.1), n, 0.04, lambda y: float(y[0]), burn=1000)
        emit("duffing", f"g{g:g}", x)
        emit("duffing", f"g{g:g}", x, noisy=True)
    for mu in (0.1, 0.5, 1.0, 5.0, 10.0, 20.0):
        def f(y, mu=mu, w=1.0, phase=[0.0]):
            phase[0] += 0.04
            return np.array([y[1],
                             mu * (1.0 - y[0] ** 2) * y[1] - w * w * y[0]])
        x = _rk4_obs(f, (0.1, 0.0), n, 0.04, lambda y: float(y[0]), burn=1000)
        emit("vanderpol", f"mu{mu:g}", x)
        emit("vanderpol", f"mu{mu:g}", x, noisy=True)

    # --- discrete maps ---
    for r in (3.2, 3.5, 3.56, 3.7, 3.8, 3.9, 3.95, 3.99):
        x = np.empty(n); x[0] = 0.3
        for i in range(1, n):
            x[i] = r * x[i - 1] * (1.0 - x[i - 1])
        emit("logistic", f"r{r:g}", x)
        emit("logistic", f"r{r:g}", x, noisy=True)
    for a, b in ((1.2, 0.3), (1.3, 0.3), (1.4, 0.3), (1.4, 0.25),
                 (1.05, 0.3), (1.16, 0.2)):
        x = np.empty(n); x[0] = 0.1; y = 0.1
        for i in range(1, n):
            xn = 1.0 - a * x[i - 1] ** 2 + y
            y = b * x[i - 1]
            x[i] = xn
        emit("henon", f"a{a:g}-b{b:g}", x)

    # --- stochastic / structural regimes ---
    rng = np.random.default_rng(seed)
    for i, H in enumerate(np.linspace(0.08, 0.92, 10)):   # fractional noise
        freqs = np.fft.rfftfreq(n)
        s = np.zeros(len(freqs))
        s[1:] = freqs[1:] ** (-H - 0.5)
        ph = rng.uniform(0, 2 * np.pi, len(freqs))
        c = s * np.exp(1j * ph)
        x = np.fft.irfft(c, n)
        emit("arfima", f"H{H:.2f}", x)
    for i in range(10):                                    # GARCH vol clustering
        w_, a_, b_ = 0.02 + 0.02 * i / 9, 0.05 + 0.05 * i / 9, 0.85
        sig2 = 1.0; x = np.empty(n)
        for t in range(n):
            e = rng.normal(0, np.sqrt(sig2))
            x[t] = e
            sig2 = w_ + a_ * e * e + b_ * sig2
        emit("garch", f"a{a_:.3f}-b{b_:.2f}", x)
    for i in range(10):                                    # regime-switching AR
        ps = 0.002 + 0.004 * i
        x = np.empty(n); x[0] = 0.0; reg = 0
        for t in range(1, n):
            if rng.uniform() < ps:
                reg = 1 - reg
            mu = 0.5 if reg else -0.5
            x[t] = mu + 0.6 * (x[t - 1] - mu) + 0.6 * rng.normal()
        emit("regime", f"ps{ps:.4f}", x)
    for i, K in enumerate((0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)):  # sync
        m = 16
        th = rng.uniform(0, 2 * np.pi, m)
        w = rng.uniform(0.7, 1.3, m)
        x = np.empty(n)
        for t in range(n):
            x[t] = np.mean(np.sin(th))
            rho_ = np.abs(np.mean(np.exp(1j * th)))
            for j in range(m):
                th[j] += w[j] * 0.1 + K * rho_ * np.sin(
                    np.angle(np.mean(np.exp(1j * th))) - th[j]) * 0.1 \
                    + 0.15 * rng.normal()
        emit("kuramoto", f"K{K:g}", x)
    for i in range(10):                                    # quasiperiodic + 1/f
        r1, r2 = 0.009, 0.009 * (1.0 + np.sqrt(5.0)) / 2.0 * 0.999
        t = np.arange(n)
        x = np.sin(2 * np.pi * r1 * t) + np.sin(2 * np.pi * r2 * t)
        slow = np.cumsum(rng.normal(0, 0.004, n))
        x = x * (1.0 + 0.3 * np.sin(slow)) \
            + np.convolve(rng.normal(0, 1, n),
                          np.exp(-np.arange(40) / 12.0), "same") * 0.2
        emit("quasiper", f"tone{i}", x)
    for i in range(8):                                     # bursty spiking
        base = 0.05 + 0.03 * i
        v = np.empty(n); v[0] = 0.0; burst = 0.0
        for t in range(1, n):
            burst = max(0.0, burst - 0.01)
            lam = base + 3.0 * (1.0 if rng.uniform() < burst else 0.0)
            v[t] = v[t - 1] * 0.9 + (1.0 if rng.poisson(lam) > 0 else 0.0)
            if rng.uniform() < 0.004:
                burst = 1.0
        emit("burst", f"b{i}", v)
    # --- v14: smooth / seasonal / trend / spiky families. The v13 result
    # (better on explosive series, worse on smooth periodic ones) showed
    # the battery was chaos-heavy; these families rebalance it toward the
    # smooth regime the zero-shot probe must handle (sunspots, airline,
    # weather). ---
    for i, seas in enumerate((6, 12, 24, 52, 104, 365)):  # seasonal + trend
        t = np.arange(n)
        trend = 0.02 * t / n + 0.3 * np.sin(t / 700.0)
        x = np.sin(2 * np.pi * t / seas) + 0.35 * np.sin(4 * np.pi * t / seas)
        x = x + trend + 0.15 * rng.normal(size=n)
        emit("seasonal", f"s{seas}", x)
        emit("seasonal", f"s{seas}", x, noisy=True)
    for i, tau in enumerate((30, 80, 150, 300, 600)):     # damped oscillations
        t = np.arange(n)
        w = 2 * np.pi / tau
        x = np.exp(-t / (4.0 * tau)) * np.sin(w * t + 0.7)
        x = x + 0.05 * rng.normal(size=n)
        emit("damped", f"tau{tau}", x)
        emit("damped", f"tau{tau}", x, noisy=True)
    for i, prd in enumerate((20, 40, 80, 160)):            # square / sawtooth
        t = np.arange(n)
        sq = np.sign(np.sin(2 * np.pi * t / prd))
        sw = 2 * ((t / prd) % 1.0) - 1.0
        for nm, wv in (("square", sq), ("saw", sw)):
            x = wv + 0.08 * rng.normal(size=n)
            emit(nm, f"p{prd}", x)
    for i, phi in enumerate((0.4, 0.6, 0.8, 0.92)):        # AR(2) with osc root
        x = np.empty(n); x[:2] = 0.0
        for t in range(2, n):
            x[t] = 2.0 * phi * np.cos(0.35) * x[t - 1] \
                - phi * phi * x[t - 2] + 0.3 * rng.normal()
        emit("ar2osc", f"phi{phi:g}", x)
    for i, drift in enumerate((0.0, 0.001, 0.005, 0.02)):  # smooth random walk
        x = np.cumsum(rng.normal(drift, 1.0, n))
        x = x + 0.5 * np.sin(np.arange(n) / 300.0)
        emit("rw", f"d{drift:g}", x)
    for i in range(6):                                     # Poisson spike train
        rate = 0.01 + 0.02 * i
        x = (rng.poisson(rate, n) > 0).astype(float)
        x = x + 0.02 * rng.normal(size=n)
        emit("poisson", f"r{rate:.3f}", x)
    for i in range(8):                                     # trend + steps
        x = np.cumsum(rng.normal(0, 0.02, n))
        for _ in range(3 + i % 4):
            t0 = rng.integers(200, n - 200)
            x[t0:] += rng.choice([-1.5, 1.5])
        x = x + 0.3 * np.sin(np.arange(n) / 120.0)
        emit("trendstep", f"k{i}", x)
    for i, r in enumerate((0.3, 0.5, 0.7, 0.9)):           # autoregressive AR(1)
        x = np.empty(n); x[0] = 0.0
        for t in range(1, n):
            x[t] = r * x[t - 1] + rng.normal(0, 0.5)
        emit("ar1", f"r{r:g}", x)
    for i in range(10):                                    # heartbeat template jitter
        t = np.arange(n)
        x = np.zeros(n)
        beat = 0
        while beat < n:
            span = np.arange(beat, min(beat + 30, n))
            x[span] += np.exp(-0.5 * ((span - beat - 2) / 1.5) ** 2) * 1.0 \
                - 0.35 * np.exp(-0.5 * ((span - beat - 10) / 3.0) ** 2)
            beat += int(round(58 + 9 * (i / 9) + 4 * rng.uniform()))
        x = x + 0.08 * np.convolve(
            rng.normal(0, 1, n), np.exp(-np.arange(10) / 4.0), "same")
        emit("heartbeat", f"rate{i}", x)
    if len(out) > n_series:
        out = out[:n_series]
    return out


def build_corpus():
    samples, meta = [], []
    held_flags = {}

    def add(name, series, domain, held_out):
        try:
            pair = prepare_pair(series)
        except Exception as e:
            print(f"  !! {name} failed to embed: {e}")
            return
        samples.append((name, pair))
        held_flags[name] = held_out
        meta.append({"name": name, "domain": domain, "held_out": held_out,
                     "n": int(len(series)), "tau_f": pair["tau_f"]})
        print(f"  corpus + {name:18s} [{domain:12s}] len={len(series):5d} "
              f"held_out={held_out}")

    p = find_csv("Cardiac_arrhythmia_dataset.csv")     # physiology
    if p:
        for name, s in load_ecg_records(p, 3):
            add(name, s, "physiology", False)
    p = find_csv("AEP_hourly.csv")                       # energy grid (11)
    if p:
        for name, s in load_grid_regions(p):
            add(name, s, "energy-grid", False)
    p = find_csv("weatherAUS.csv")                       # meteorology (9)
    if p:
        for name, s in load_weather_cols(
                p, ["Temp3pm", "MinTemp", "MaxTemp", "Temp9am",
                    "Humidity3pm", "Humidity9am", "Pressure3pm",
                    "Pressure9am", "Rainfall"]):
            add(name, s, "meteorology", False)
    p = find_csv("coin_Bitcoin.csv")                     # finance (8 coins)
    if p:
        for name, s in load_crypto_coins(p, 8):
            add(name, s, "finance", False)
    p = find_csv("Sunspots.csv")                         # solar (held out)
    if p:
        _, s = load_series_from_csv(p, hints=["Sunspot"], name="sunspots")
        add("sunspots", s, "solar-physics", True)
    p = find_csv("AirPassengers.csv")                    # economics
    if p:
        _, s = load_series_from_csv(p, hints=["#Passengers"], name="airline")
        add("airline", s, "economics", False)
    p = find_csv("covid_19_clean_complete.csv")          # epidemiology
    if p:
        for name, s, h in load_covid_countries(
                p, ["US"], ["India", "Brazil", "Germany", "UK", "France"]):
            add(name, s, "epidemiology", h)
    for name, s in load_synthetic():                     # physics
        add(name, s, "physics", False)
    return samples, meta, held_flags


def main():
    print("=" * 70)
    print("TSO foundation pretraining V14 — balanced corpus, forced "
          "Koopman linearity")
    print("=" * 70)
    ensure_gpu_compatible()
    print(torch.__version__, "cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    device = pick_device("cuda")

    print("\n[1/5] corpus from attached datasets")
    try:
        print("  /kaggle/input:", os.listdir(INPUT))
    except OSError:
        print("  /kaggle/input: not mounted!")
    samples, meta, held_flags = build_corpus()
    held = [m["name"] for m in meta if m["held_out"]]
    print(f"  {len(samples)} real series, held out: {held}")
    _big = int(os.environ.get("V15_TARGET", 0))
    if _big > 0:
        battery = gen_v15_battery(target=_big)
        print(f"  + {len(battery)} synthetic dynamics series "
              f"(V15 corpus-scale battery, target {_big}, seed 11)")
    else:
        battery = load_dynamics_battery(n_series=240)
        print(f"  + {len(battery)} synthetic dynamics series "
              f"(attractor sweeps, maps, stochastic regimes)")

    print("\n[2/5] AMP training-loop benchmark")
    try:
        bench = benchmark_loop(np.random.default_rng(0).normal(size=(4000, 5)),
                               iters=300, device=device, out_dir=OUT,
                               label=f"gpu-{device}")
    except Exception as e:
        print(f"  GPU bench failed ({e}); falling back to CPU")
        bench = benchmark_loop(np.random.default_rng(0).normal(size=(4000, 5)),
                               iters=200, device="cpu", out_dir=OUT,
                               label="cpu-fallback")

    # V14 protocol (env-tunable; kernel default = the GPU run):
    #   - balanced corpus: real x8 + rebalanced battery (~45% chaotic
    #     flows/maps, ~55% smooth/seasonal/stochastic families) — the v13
    #     result said chaos-heavy pretraining hurts smooth periodic series.
    #   - dyn_w=2.5: force the latent to be Koopman-linear during training
    #     (the frozen probe's core assumption; v12 tested an add-on probe
    #     loss, this strengthens the built-in linear-step pretext instead).
    #   - latent 256 / hidden 768 (v11's width) on GPU, 25k iters.
    N_ITERS = int(os.environ.get("V14_ITERS", 25_000))
    LATENT = int(os.environ.get("V14_LATENT", 256))
    HIDDEN = int(os.environ.get("V14_HIDDEN", 768))
    DYN_W = float(os.environ.get("V14_DYN_W", 2.5))
    SEED = int(os.environ.get("V14_SEED", 0))
    real_train = [(p, m) for (n, p), m in zip(samples, meta)
                  if not m["held_out"]]
    battery_pairs = []
    for name, s in battery:
        try:
            battery_pairs.append(prepare_pair(s))
        except Exception as e:
            print(f"  !! {name} failed to embed: {e}")
    train_pairs = [p for p, _ in real_train] * 8 + battery_pairs
    print(f"\n[3/5] pretraining (latent {LATENT} / hidden {HIDDEN}, "
          f"{N_ITERS} iters, dyn_w={DYN_W}, seed={SEED}, "
          f"device={device}, corpus={len(train_pairs)} entries "
          f"= {len(real_train)} real x8 + {len(battery_pairs)} synthetic)")
    try:
        model, hist, parts_agg, parts = pretrain_foundation(
            train_pairs, iters=N_ITERS, latent_dim=LATENT, hidden=HIDDEN,
            seed=SEED, device=device, amp=(device == "cuda"),
            print_every=5_000, dyn_w=DYN_W,
            ckpt_path=os.path.join(OUT, "foundation_model.pt"),
            joint_probe=False)
    except Exception as e:
        print(f"  pretrain failed ({e}); falling back to CPU")
        device = "cpu"
        model, hist, parts_agg, parts = pretrain_foundation(
            train_pairs, iters=N_ITERS, latent_dim=LATENT, hidden=HIDDEN,
            seed=SEED, device="cpu", amp=False, print_every=5_000,
            dyn_w=DYN_W, ckpt_path=os.path.join(OUT, "foundation_model.pt"),
            joint_probe=False)
    torch.save(model.state_dict(), os.path.join(OUT, "foundation_model.pt"))
    plot_pretrain_curves(hist, parts, os.path.join(OUT, "pretrain_curves.png"))

    print("\n[4/5] zero-shot transfer + in-kernel baselines")
    zs, scratch, gru = {}, {}, {}
    sunspot_series = None
    for n, p in samples:
        res = zero_shot_forecast(model, p["series"], device=device)
        zs[n] = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
        if n == "sunspots":
            sunspot_series = p["series"]
        try:
            g = gru_baseline(p["fine"], iters=300, seed=0, device=device)
            gru[n] = g["skill_pct"]
        except Exception as e:
            print(f"  !! gru {n}: {e}")
        print(f"  frozen {n:22s}: skill={res['skill_pct']:+7.1f}%  "
              f"corr={res['corr']:+.2f}  gru={gru.get(n, float('nan')):+7.1f}%")
    for m in meta:
        if not m["held_out"]:
            continue
        pair = dict(samples)[m["name"]]
        sc = scratch_baseline(pair["series"], iters=400, seed=0, device=device)
        scratch[m["name"]] = {k: v for k, v in sc.items()
                              if not isinstance(v, np.ndarray)}
        print(f"  scratch {m['name']:22s}: skill={sc['skill_pct']:+7.1f}%  "
              f"corr={sc['corr']:+.2f}")
        res = zero_shot_forecast(model, pair["series"], device=device)
        pers = np.full(len(res["true"]), res["true"][0])
        plot_zero_shot(res["true"], res["pred"],
                       os.path.join(OUT, f"zero_shot_{m['name']}.png"),
                       f"Zero-shot TSO forecast — {m['name']} (V14 pretrained)",
                       res["skill_pct"], baseline=pers)

    plot_latent_geometry(*latent_geometry(model, samples, seed=0,
                                          device=device),
                         os.path.join(OUT, "latent_geometry.png"))

    # ---------------- the solar-cycle discovery ---------------------------
    # The cycle is AM-modulated noise at full rate; the single-scale fit
    # cannot see it. Coarsening renormalizes the spectrum: the operator's
    # frozen latent + linear Koopman fit converge to the true ~132-month
    # Schwabe period at coarse scales (scale covariance).
    solar = None
    if sunspot_series is not None:
        disc = solar_cycle_discovery(model, sunspot_series, device=device)
        months = disc["period_months"]
        plot_solar_discovery(disc["rows"], os.path.join(OUT, "solar_cycle.png"),
                             known_months=disc["known_cycle_months"])
        solar = {"period_months": float(months) if months else None,
                 "period_years": (float(months) / 12.0) if months else None,
                 "known_cycle_months": disc["known_cycle_months"],
                 "rows": disc["rows"]}
        if months is None:
            print("  SOLAR-CYCLE DISCOVERY: no oscillatory mode found at any "
                  "scale")
        else:
            print(f"  SOLAR-CYCLE DISCOVERY: the frozen operator on the held-out "
                  f"sunspot series reads a dominant period of {months:.0f} "
                  f"months ({months / 12:.1f} years) at the coarsest scale — "
                  f"against the known Schwabe cycle of "
                  f"{disc['known_cycle_months']:.0f} months (~11 years).")

    results = {
        "version": ("v15" if _big > 0 else "v14"), "device": device,
        "cuda": torch.cuda.is_available(),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None),
        "train_loop": bench,
        "pretrain": {"iters": len(hist) or N_ITERS, "latent_dim": LATENT,
                     "hidden": HIDDEN, "dyn_w": DYN_W, "seed": SEED,
                     "requested_iters": N_ITERS,
                     "joint_probe": False,
                     "mode": ("v15-corpus-scale" if _big > 0
                              else "v14-balanced-koopman"),
                     "amp": device == "cuda" and torch.cuda.is_available(),
                     "final_loss": float(hist[-1]) if hist else None,
                     "pretext_losses": parts_agg},
        "corpus": meta,
        "corpus_note": f"{len(real_train)} real series x8 + "
                       f"{len(battery_pairs)} synthetic dynamics "
                       f"(universal battery, seed 11)",
        "zero_shot": zs,
        "gru_baseline": gru,
        "scratch_baseline": scratch,
        "solar_cycle": solar,
        "transfer_summary": {
            k: {"frozen_skill": round(zs[k]["skill_pct"], 2),
                "scratch_skill": round(scratch[k]["skill_pct"], 2)}
            for k in scratch
        },
        "files": sorted(os.listdir(OUT)),
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\n  -> outputs in /kaggle/working:")
    for f in sorted(os.listdir(OUT)):
        print(f"     {f}")
    print("DONE")


# ===================== v15: corpus-scale dynamics battery =====================
# Ported verbatim from scripts/build_v15_battery.py (locally tested; deterministic
# seed-11 generation so every run shares the identical corpus). FRAC mirrors the
# v14 kernel battery family balance.

FRAC = {
    "lorenz": 0.062, "rossler": 0.062, "duffing": 0.062, "vanderpol": 0.062,
    "logistic": 0.083, "henon": 0.031,
    "seasonal": 0.062, "quasiper": 0.052, "damped": 0.052, "heartbeat": 0.052,
    "saw": 0.021, "square": 0.021, "ar2osc": 0.021,
    "arfima": 0.052, "garch": 0.052, "regime": 0.052, "burst": 0.042,
    "trendstep": 0.042, "poisson": 0.031, "ar1": 0.021, "rw": 0.021,
    "kuramoto": 0.042,
}
TWIN = {"lorenz", "rossler", "duffing", "vanderpol", "logistic", "kuramoto"}


def rk4_obs(f, y0, n, dt, obs, burn=400):
    y = np.asarray(y0, dtype=float)
    ys = np.empty(n)
    for i in range(-burn, n):
        def deriv(v):
            return np.asarray(f(v), dtype=float)
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if i >= 0:
            ys[i] = obs(y)
        if not np.all(np.isfinite(y)):
            return None
    return ys


def gen(seed, target, n):
    rng = np.random.default_rng(seed)
    out = {}
    _idx = [0]

    def unique(family, cfg):
        _idx[0] += 1
        return f"{family}-{cfg}-{_idx[0]}"

    def emit(family, cfg, x, noisy=False):
        x = np.asarray(x, dtype=float)
        if x is None or not np.all(np.isfinite(x)):
            return False
        if noisy:
            r = np.random.default_rng(hash((family, cfg)) & 0xffffffff)
            x = x + 0.05 * float(np.std(x)) * r.normal(size=len(x))
        out[f"syn-{unique(family, cfg)}"] = x
        return True

    def draw_until(family, count):
        """Draw random configs until `count` FINAL series (incl. twins)."""
        made = 0
        tries = 0
        while made < count and tries < 50 * count + 200:
            tries += 1
            cfg, x = make_family(family, rng, n)
            if x is None:
                continue
            twin = family in TWIN
            if emit(family, cfg, x, noisy=False):
                made += 1
            if twin and made < count and emit(family, cfg + "n", x, noisy=True):
                made += 1

    def make_family(family, rng, n):
        if family == "lorenz":
            rho = float(rng.uniform(10, 99))
            return f"r{rho:.2f}", rk4_obs(
                lambda y, rho=rho: np.array(
                    [10.0 * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1],
                     y[0] * y[1] - (8.0 / 3.0) * y[2]]),
                (1.0, 1.0, 1.0), n, 0.008, lambda y: float(y[0]))
        if family == "rossler":
            a = float(rng.uniform(0.06, 0.5))
            return f"a{a:.3f}", rk4_obs(
                lambda y, a=a: np.array(
                    [-y[1] - y[2], y[0] + a * y[1], 2.0 + y[2] * (y[0] - 4.0)]),
                (1.0, 1.0, 1.0), n, 0.02, lambda y: float(y[0]))
        if family == "duffing":
            g = float(rng.uniform(0.1, 0.6))
            def duff(y, g=g, phase=[0.0]):
                phase[0] += 0.04
                return np.array([y[1], -0.3 * y[1] + y[0] - y[0] ** 3
                                 + g * np.cos(1.2 * phase[0])])
            return f"g{g:.3f}", rk4_obs(duff, (0.0, 0.1), n, 0.04,
                                        lambda y: float(y[0]))
        if family == "vanderpol":
            mu = float(rng.uniform(0.1, 20.0))
            return f"mu{mu:.2f}", rk4_obs(
                lambda y, mu=mu: np.array(
                    [y[1], mu * (1.0 - y[0] ** 2) * y[1] - 1.0 * y[0]]),
                (0.1, 0.0), n, 0.04, lambda y: float(y[0]))
        if family == "logistic":
            r = float(rng.uniform(3.2, 4.0))
            x = np.empty(n); x[0] = 0.3
            for j in range(1, n):
                x[j] = r * x[j - 1] * (1.0 - x[j - 1])
            return f"r{r:.3f}", x
        if family == "henon":
            a = float(rng.uniform(1.0, 1.4)); b = float(rng.uniform(0.2, 0.32))
            x = np.empty(n); x[0] = 0.1; yy = 0.1
            for j in range(1, n):
                xn = 1.0 - a * x[j - 1] ** 2 + yy
                yy = b * x[j - 1]
                x[j] = xn
            return f"a{a:.3f}-b{b:.3f}", x
        if family == "seasonal":
            t = np.arange(n)
            seas = float(rng.choice([12, 24, 28, 52, 60, 120]))
            trend = float(rng.uniform(-0.0006, 0.0006))
            x = np.sin(2 * np.pi * t / seas) + 0.4 * np.sin(
                2 * np.pi * 2 * t / seas + rng.uniform(0, 6))
            x += trend * t + rng.normal(0, 0.08, n)
            return f"s{seas:g}", x
        if family == "quasiper":
            t = np.arange(n)
            r1 = float(rng.uniform(0.005, 0.03))
            r2 = r1 * (1.0 + np.sqrt(5.0)) / 2.0 * float(rng.uniform(0.98, 1.0))
            x = np.sin(2 * np.pi * r1 * t) + np.sin(2 * np.pi * r2 * t)
            x *= 1.0 + 0.3 * np.sin(np.cumsum(rng.normal(0, 0.003, n)))
            return f"r{r1:.4f}", x
        if family == "damped":
            t = np.arange(n)
            f0 = float(rng.uniform(0.005, 0.04))
            zeta = float(rng.uniform(0.0005, 0.004))
            x = np.exp(-zeta * t) * np.sin(2 * np.pi * f0 * t)
            return f"f{f0:.4f}-z{zeta:.4f}", x
        if family == "heartbeat":                       # ECG-like pulse train
            t = np.arange(n)
            per = float(rng.uniform(40, 400))
            ph = (t % per) / per
            x = np.exp(-((ph - 0.1) ** 2) / 0.002) - 0.15 * np.exp(
                -((ph - 0.5) ** 2) / 0.01) + 0.06 * rng.normal(size=n)
            return f"p{per:.0f}", x
        if family == "saw":
            t = np.arange(n)
            per = float(rng.uniform(30, 300))
            x = 2.0 * ((t % per) / per) - 1.0 + 0.1 * rng.normal(size=n)
            return f"p{per:.0f}", x
        if family == "square":
            t = np.arange(n)
            per = float(rng.uniform(30, 300))
            x = np.sign(np.sin(2 * np.pi * t / per)) + 0.1 * rng.normal(size=n)
            return f"p{per:.0f}", x
        if family == "ar2osc":
            phi = float(rng.uniform(0.3, 0.98))
            x = np.empty(n); x[0] = 0.0; x[1] = 0.1
            for j in range(2, n):
                x[j] = 1.9 * phi * x[j - 1] - phi * phi * x[j - 2] \
                    + 0.05 * rng.normal()
            return f"phi{phi:.3f}", x
        if family == "arfima":
            H = float(rng.uniform(0.1, 0.9))
            freqs = np.fft.rfftfreq(n)
            s = np.zeros(len(freqs)); s[1:] = freqs[1:] ** (-H - 0.5)
            c = s * np.exp(1j * rng.uniform(0, 2 * np.pi, len(freqs)))
            return f"H{H:.2f}", np.fft.irfft(c, n)
        if family == "garch":
            w = float(rng.uniform(0.01, 0.05))
            a = float(rng.uniform(0.05, 0.15)); b = float(rng.uniform(0.8, 0.92))
            sig2 = 1.0; x = np.empty(n)
            for t in range(n):
                e = rng.normal(0, np.sqrt(sig2)); x[t] = e
                sig2 = w + a * e * e + b * sig2
            return f"a{a:.3f}-b{b:.3f}", x
        if family == "regime":
            ps = float(rng.uniform(0.001, 0.008))
            x = np.empty(n); x[0] = 0.0; reg = 0
            for t in range(1, n):
                if rng.uniform() < ps:
                    reg = 1 - reg
                mu = 0.5 if reg else -0.5
                x[t] = mu + 0.6 * (x[t - 1] - mu) + 0.6 * rng.normal()
            return f"ps{ps:.4f}", x
        if family == "burst":
            x = np.empty(n); x[0] = 0.0; hot = False
            for t in range(1, n):
                if rng.uniform() < 0.004:
                    hot = not hot
                a = 0.95 if not hot else 0.55
                s = 0.05 if not hot else 0.8
                x[t] = a * x[t - 1] + s * rng.normal()
            return f"h{int(hot)}", x
        if family == "trendstep":                       # trend + random steps
            t = np.arange(n)
            x = float(rng.uniform(0.0003, 0.002)) * t
            pos = 0
            while pos < n:
                w = int(rng.integers(30, 400))
                x[pos:pos + w] += float(rng.choice([-1, 1])) \
                    * float(rng.uniform(0.5, 4.0))
                pos += w
            x += rng.normal(0, 0.05, n)
            return "s", x
        if family == "poisson":
            lam = float(rng.uniform(0.5, 6.0))
            x = np.empty(n); x[0] = lam
            for t in range(1, n):
                lam = max(0.2, lam + 0.02 * rng.normal())
                x[t] = rng.poisson(lam)
            return f"l{lam:.2f}", x
        if family == "ar1":
            r = float(rng.uniform(-0.95, 0.95))
            x = np.empty(n); x[0] = 0.0
            for t in range(1, n):
                x[t] = r * x[t - 1] + 0.3 * rng.normal()
            return f"r{r:.3f}", x
        if family == "rw":
            x = np.cumsum(rng.normal(float(rng.uniform(-0.02, 0.02)), 0.1, n))
            return "w", x
        if family == "kuramoto":
            Kc = float(rng.uniform(0.0, 4.0))
            m = int(rng.integers(8, 32))
            th = rng.uniform(0, 2 * np.pi, m)
            w = rng.uniform(0.7, 1.3, m)
            x = np.empty(n)
            for t in range(n):
                x[t] = np.mean(np.sin(th))
                rho_ = np.abs(np.mean(np.exp(1j * th)))
                ang = np.angle(np.mean(np.exp(1j * th)))
                for j in range(m):
                    th[j] += w[j] * 0.1 + Kc * rho_ * np.sin(ang - th[j]) \
                        * 0.1 + 0.15 * rng.normal()
            return f"K{Kc:.2f}", x
        return None, None

    for fam, frac in FRAC.items():
        draw_until(fam, max(1, round(target * frac)))
    return out



def gen_v15_battery(target=5988, n=2600, seed=11):
    """Deterministic v15 battery: list of (name, series) like load_dynamics_battery."""
    d = gen(seed, target, n)
    return [(k, v) for k, v in sorted(d.items())]




if __name__ == "__main__":
    main()