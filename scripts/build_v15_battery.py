#!/usr/bin/env python3
"""v15: generate the large balanced dynamics battery once, deterministically.

The Chronos lesson (paper Sec.~external): corpus breadth is the gap. v15
scales v14's balanced-battery recipe from 192 to thousands of series by
randomizing the dynamical-system parameters (fixed RNG seed -> reproducible
corpus shared by every run/platform via a Kaggle dataset mount).

Family mix mirrors v14 exactly (the plateau-breaking balance), measured from
the v14 kernel battery: chaotic flows lorenz/rossler/duffing/vanderpol
~25%, maps logistic/henon ~11%, coupled kuramoto ~4%, smooth/seasonal
~28%, stochastic/structural ~31%. Noise twins follow the v14 policy
(flows/maps/kuramoto get a 5%-noise twin).

Usage:
    python scripts/build_v15_battery.py            # ~6000 series, n=2600
    python scripts/build_v15_battery.py --target 300 --n 800 --seed 11

Output:
    output/v15_corpus/battery.npz   (name -> float32 series)
    output/v15_corpus/meta.json
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# v14 family fractions of the final series count (from the kernel battery)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=6000,
                    help="final series count (mix mirrors v14)")
    ap.add_argument("--n", type=int, default=2600, help="series length")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "v15_corpus"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out = gen(args.seed, args.target, args.n)
    from collections import Counter
    fam = Counter(k.split("-")[1] for k in out)
    np.savez_compressed(os.path.join(args.out, "battery.npz"),
                        **{k: v.astype(np.float32) for k, v in out.items()})
    meta = {"n_series": len(out), "n": args.n, "seed": args.seed,
            "families": dict(fam)}
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    tot = len(out)
    print(f"wrote {tot} series (n={args.n}) to {args.out}")
    for k, v in sorted(fam.items()):
        print(f"  {k:11s} {v:5d}  {100 * v / tot:5.1f}%")


if __name__ == "__main__":
    main()
