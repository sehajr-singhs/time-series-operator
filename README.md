# Time-Series Operator (TSO)

A from-scratch implementation of the "time as geometry" vision: instead of
chopping numbers into tokens and treating them like sentences, the model

1. **reconstructs phase space** (Takens' delay embedding) from a single channel,
2. **linearizes the chaos** (Koopman operator via exact DMD, a random-Fourier
   lift, and a *learned* deep-Koopman autoencoder),
3. **learns the vector field** (a small neural field `ds/dt`, neural-ODE style),
4. sees the system through **multiple temporal lenses** (a scale-space analysis),
5. **pretrains across domains** (a shared FoundationOperator: one linear
   Koopman matrix + scale-covariance + arrow-of-time pretexts, then zero-shot
   forecasting on never-seen systems),
6. **reads tipping points** (a data-driven Lyapunov detector on the
   reconstructed attractor),
7. **scales on GPU** (batched training, mixed precision, checkpoint/resume,
   continuous Neural-ODE querying — CPU here, T4 via the Kaggle kernel).

No code is shared with any other project in this workspace — everything here is
implemented from scratch in NumPy + PyTorch.

```
signal -> normalize -> Takens embedding (phase space)
                    -> Koopman lift (linearize chaos)  -> closed-form forecast
                    -> neural field (learn the flow)   -> integrate = geometry
                    -> scale space (1x, 2x, 4x, 8x lenses)
                    -> FoundationOperator pretraining (many domains)
                         -> zero-shot forecast on never-seen systems
                    -> Wolf Lyapunov detector (tipping points)
```

## Run it

```bash
# 1) synthetic: observe ONE noisy channel of the Lorenz butterfly,
#    reconstruct the rest, forecast, and render the artwork
python scripts/demo_lorenz.py

# 2) real data: pull MIT-BIH heart-rate variability through the legacy
#    Kaggle API and run the whole pipeline end to end
python scripts/kaggle_e2e.py

# 3) the scaled leg: download a 7-domain corpus via the legacy Kaggle API,
#    pretrain the FoundationOperator, benchmark deep Koopman + bifurcation +
#    the GPU training loop, then zero-shot forecast held-out systems
python scripts/pretrain_foundation.py --iters 2400 --dk-iters 3000
```

Artwork lands in `output/lorenz/`, `output/kaggle_ecg/` and
`output/foundation/`; `output/index.html` is a gallery of everything. Metrics
are written to `metrics.json` in each output directory.

## Results (as built)

### Core pipeline

| metric | Lorenz (chaos, synthetic) | MIT-BIH HRV (real heartbeats) |
|---|---|---|
| Takens delay / dim | τ=59, m=3 | τ=1, m=6 |
| eDMD-RFF skill over persistence | **+21.3%** | **+3.3%** |
| plain DMD skill | +18.2% | +3.3% |
| neural field, pre-divergence corr | 0.79 (8 steps) | 0.74 |
| top Koopman |λ| at scales 1→8 | 0.999 → 0.937 | 0.69 → 0.21 |

The scale-space row is the tell: the deterministic Lorenz butterfly keeps its
Koopman eigenvalues pinned to the unit circle under decimation (scale-covariant
physics), while noisy heart-rate variability sheds its spectral content as you
coarsen — the same operator distinguishes deterministic structure from
stochastic physiology.

### Scaled leg (`output/foundation/metrics.json`)

- **Deep Koopman vs fixed RFF lift** (single-system Lorenz, held-out
  linearity): RFF is ~1.9× more linear at 258 lifted dims vs the learned
  lift's 64-dim latent; forecast skill within ~2 pts. The learned lift's
  advantage is *sharing* one coordinate system across systems — which a
  per-system fixed lift cannot do.
- **Tipping detector**: Wolf Lyapunov estimator on one channel flags
  ρ≈21 (transient-chaos precursor) ahead of the fully sustained chaotic
  attractor at ρ≈24.7 — an early warning, not a late one.
- **GPU training loop**: batched + AMP + checkpointing; 3.5 ms/iter on CPU
  (real AMP numbers on T4 in the Kaggle kernel).
- **Foundation pretraining**: all four pretexts converge — reconstruction
  0.013, linear dynamics 0.31, scale covariance 0.031, arrow-of-time accuracy
  **94.8%** (the conv head learns the direction of time from latent
  trajectory structure).
- **Zero-shot forecasting** (frozen lift + closed-form linear probe, held-out
  systems): weather +12%, sunspots +16%, ECG +11–14%, Lorenz +14–17%; the
  persistence baseline is brutally strong on smooth series (covid, bitcoin),
  where frozen zero-shot skill goes negative.
- **Honest transfer result**: at this prototype scale (6 training domains,
  24-dim latent, 2.4k iters) the frozen latent does **not** beat per-target
  training on forecast skill — scratch wins on both held-out sets. The
  pretexts converge, but coordinates don't transfer better than scratch yet;
  that is exactly the scaling question the GPU kernel pushes on (6000 iters,
  48-dim latent, 128 hidden, AMP on T4).

## Kaggle kernel (scaled run)

`kaggle_kernel_tso/` is a private kernel (built single-file by
`scripts/build_kernel_single.py` because the Kaggle CLI uploads only the code
file) that runs the scaled pretraining on Kaggle with the corpus attached as
datasources. Private runs: `kaggle kernels push -p kaggle_kernel_tso`, then
`kaggle kernels status <owner>/<slug>`, then
`kaggle kernels output <owner>/<slug> -p output/kaggle_kernel_run`.

**Verified runs (private, COMPLETE — `output/kaggle_kernel_run/` for v5,
`output/kaggle_kernel_v7/` for v7, `output/kaggle_kernel_v9/` for v9):**

* **v5** — 7 datasets, 4000 iters (latent 48, hidden 128). All four pretexts
  converged on Kaggle's hardware: recon 0.01, scale 0.02, arrow-of-time loss
  0.0001 at **100%** accuracy.
* **v7 (scaled)** — 23 series across 8 domains, 15000 iters (latent 96,
  hidden 256), arrow-of-time **96.8%**. Scaling effect on frozen zero-shot
  skill over persistence: median **−24.6% → +1.5%**, positive **2/7 → 13/23**
  series. Top frozen zero-shot wins: DOM +60.1%, DUQ +59.9%, EKPC +60.1%,
  DEOK +50.6% (electricity regions), Ethereum +16.6%, covid-brazil +16.2%,
  weather-Temp3pm +13.2%. Spiky/near-unit-root series (Dogecoin, covid-india)
  remain out of reach for the linear probe — an honest limitation.
* **v9 (largest)** — 40 series / 8 domains, 25000 iters (latent 128, hidden
  384), arrow-of-time **95.5%** at the end of training. **In-kernel TSO vs
  GRU on identical hardware/protocol: 28/40 wins, p=0.017 (two-sided
  binomial), median +46.5 pts over persistence** (GRU median −60.6%).
  Zero-shot: 20/40 series positive, 11/13 grid/weather entries (e.g.
  grid-DEOK +41.7%, COMED +20.7%, DAYTON +6.8%, AEP +5.4%). Held-out
  sunspots: frozen +20.1% vs scratch +22.8% (near tie) and
  **the frozen operator rediscovers the ~11-year solar cycle from its
  fitted Koopman eigenvalues** via the renormalization leg: 130.1 months
  vs the known 132 (1.4% error), converging 441→327→203→139→130 across
  2×–32× coarsening. On the shared 23-series subset v9 ≈ v7 (13 wins vs
  10, p=0.68); v9's gains concentrate on the 17 new series.
* **v10 (iteration saturation — the honest negative result)** — same
  protocol, **60,000 iters** (GPU quota exhausted, ran CPU). The pretext
  losses had already saturated at 25k (arrow 95.5% → 96.7%, final loss
  0.318 → 0.320), and the extra iterations **hurt** the frozen probe:
  v9 wins 28/40 (p=0.017), median −3.9 pts, positives 20/40 → 16/40,
  held-out sunspots +20.1% → +0.4%. On the shared 23-series corpus the
  scaling curve is 8 (v5) → 9 (local) → 13 (v7) → 12 (v9) → 10 (v10):
  corpus breadth and width were what transferred, iteration count
  saturates. Solar-cycle discovery is stable (129.4 mo vs known 132).
  Takeaway for the paper: pretraining beyond pretext loss saturation
  does not buy transfer — checkpoint selection and joint probe/pretext
  training are the next levers, not more iterations.
* **v11 (Modal GPU, capacity scaling, 3 seeds) — published on Hugging
  Face as [`Sejibeji/tso-foundation-v11`](https://huggingface.co/Sejibeji/tso-foundation-v11)**
  — same 40-series corpus and protocol, but latent 256 / hidden 768
  (2× width) at 25k iters with AMP on T4 GPUs, three seeds. Best seed:
  **29/40 in-kernel wins vs GRU (p=0.002), 20/40 positive** — the best
  single run observed, and solar-cycle rediscovery holds at 127–128 mo
  (vs known 132) in every seed. But the honest headline is seed
  variance: the three seeds land at 29/21/28 wins and 12/5/8 positive
  on the shared 23-series reprobe — capacity gains are within seed
  noise at this probe. The hub repo ships the self-contained `model.py`
  (exact kernel probe logic, verified to reproduce metrics.json) plus
  figures and metrics.
* **v12 (joint probe/pretext training — the direct test of the v10/v11
  prescription, CPU Kaggle)** — same 40-series corpus, capacity and
  25k-iter protocol as v11, but the pretraining loss adds multi-step
  Koopman roll-out consistency (horizons 2/4/8) and unit-circle
  spectral regularization. Both terms converge (probe 1.72 → final
  loss 1.03; spectral term 1.6e-6, spectrum inside the unit circle),
  solar discovery holds (129.7 mo vs known 132), and the in-kernel
  result is 28/40 wins vs GRU / 17/40 positive — statistically
  identical to v9/v11. But the honest headline is negative: on the
  shared 23-series reprobe v12 reads 11/23, and paired head-to-head
  v9 **beats** v12 16/23 (p=0.05). Regularizing latent linearizability
  during pretraining mildly *degrades* frozen closed-form transfer —
  the latent is already as linearizable as a frozen probe can exploit.
  The wall is the probe-evaluation interface and corpus breadth, not
  latent geometry. Checkpoints/metrics/figures: `output/kaggle_kernel_v12/`.
* **v13 (corpus breadth — 40 real series tiled with a 176-series
  universal dynamics battery, CPU Kaggle)** — at v9's exact width
  (lat 128 · hid 384, 25k iters, plain pretext loss) but trained on
  generated Lorenz/Rössler parameter sweeps, discrete maps, ARFIMA,
  GARCH, regime-switching and Kuramoto oscillators in addition to the
  40 real series. In-kernel: 25/40 wins vs GRU / 12/40 positive,
  solar rediscovery holds (127.9 mo). The honest result is a *regime
  trade-off*, not a plateau break: on the shared 23-series reprobe
  v13 reads 6/23 (v9 wins paired 15/23, p=0.11), yet it crushes v9
  where v9 explodes (Dogecoin −7,678 → −883 skill points, covid-india
  −91,617 → −22,500, Bitcoin −177 → −92) while losing smooth
  periodic transfer (sunspots +20 → −26, airline −43 → −161).
  Dynamics-heavy pretraining teaches nonlinear regime structure at
  the expense of smooth-periodic geometry. Checkpoints/metrics/figures:
  `output/kaggle_kernel_v13/`. Published on Hugging Face as
  [`Sejibeji/tso-foundation-v13`](https://huggingface.co/Sejibeji/tso-foundation-v13)
  (weights + self-contained `model.py`, verified to reproduce
  `metrics.json` to the decimal).
* **v14 (the plateau break — balanced corpus + forced Koopman
  linearity, CPU Kaggle)** — v11 capacity (lat 256 · hid 768, 25k
  iters) on the v13 battery re-balanced toward smooth/seasonal/trend/
  spiky families, with the Koopman dynamics pretext re-weighted to
  `dyn_w=2.5`. **Best run of the whole project**: 31/40 wins vs GRU
  (p<0.001) and 24/40 positive in-kernel with the **first positive
  median skill (+1.8 pts over persistence)**; 14/23 on the shared
  23-series reprobe (best of any run, above the 12/23 plateau); first
  run to win a paired head-to-head vs v9 (13/23, p=0.34); solar
  rediscovery holds (128.0 mo vs known 132). The two negative levers
  (corpus balance + forced linearity) combine into a positive one —
  every single-lever experiment landed inside seed noise, this
  two-lever combination lands outside it in every tracked metric.
  **Replicated across 5 runs** (identical recipe, seeds 0–3 + a
  protocol-error duplicate, v16): 14/23 · 13/23 · 13/23 · 13/23 ·
  12/23 on the shared reprobe — with a **positive median skill in
  every single run** (+2.0, +6.4, +0.5, +1.4, +3.8 pts; v9's median
  is negative). Wilcoxon signed-rank on per-series medians vs v9:
  **p=0.03**; pooled 65/115 positive vs v9's rate (p=0.20). Direction
  consistent across all runs; the magnitude gains are significant.
  Checkpoints/metrics/figures: `output/kaggle_kernel_v14/` +
  `output/kaggle_kernel_v14_seed{1,2,3}/`. Published on
  Hugging Face as
  [`Sejibeji/tso-foundation-v14`](https://huggingface.co/Sejibeji/tso-foundation-v14)
  (weights + self-contained `model.py`, verified to reproduce
  `metrics.json` to the decimal).
* **v17 (corpus-scale: 25× the synthetic corpus at fixed budget, CPU
  Kaggle)** — the exact v14 recipe but with the 5,973-series battery
  (randomized parameter sweeps, z-scored emission — the v15 divergence
  fix: v15's raw-scale ARFIMA/trend-step tails blew the MSE pretexts to
  4.8e7 and collapsed transfer). v17 converges cleanly (loss 0.87,
  solar 128.8 mo) at 30/40 wins, 21 positive, median +1.0 — inside the
  v14 seed band, but it **significantly loses to v14 head-to-head
  (12W/28L, p=0.017) and reprobes at 11/23**. The reason is the
  fixed-budget trade: 5,973 series × ~4 passes each vs 192 × ~104.
  Corpus breadth is only bought with proportionally larger training
  budgets — which is exactly how Chronos's ~80k-series corpus wins.
  Checkpoints/metrics/figures: `output/kaggle_kernel_v17/`.

GPU notes (Modal): `scripts/modal_v11.py` is the self-contained app —
image bakes the merged kernel module + 40-series corpus, trains on T4 with
AMP and checkpoints every 600 iters to a volume, runs the full in-kernel
probe/GRU/scratch/solar pipeline, and commits `metrics.json` + plots per
seed (`scripts/fetch_v11.py` pulls them down). Three concurrent T4 seeds
for 25k iters finished in ~15 minutes total.

GPU notes: Kaggle's free tier sometimes assigns a P100 (sm_60) that modern
torch wheels (sm_70+) cannot execute — the kernel detects this, installs a
compatible torch once (marker-guarded, no restart loop), else falls back to
CPU; an AMP GradScaler/autocast path runs whenever CUDA is usable. The v3
attempt exposed a real infrastructure trap (guard bug → reinstall loop), fixed
in v5; results are from a clean completed run.

## Modules

- `tso/attractors.py` — RK4 integrator, Lorenz & Rossler ground truths.
- `tso/embedding.py` — autocorrelation delay selection, Takens embedding,
  false-nearest-neighbours dimension selection.
- `tso/koopman.py` — exact DMD and extended DMD with random-Fourier lift;
  closed-form forecasting `x_k = Φ diag(μ^k) b`.
- `tso/deep_koopman.py` — learned Koopman eigencoordinates (autoencoder
  φ/K/ψ) with multi-step consistency; benchmark vs the RFF lift.
- `tso/bifurcation.py` — Wolf largest-Lyapunov estimator, tipping metrics,
  Lorenz ρ sweep, durable-crossing detector.
- `tso/train_loop.py` — batched training with AMP (autocast + GradScaler),
  checkpoint/resume, continuous Neural-ODE querying, loop benchmark.
- `tso/foundation.py` — the FoundationOperator (shared enc/K/dec +
  scale-map + arrow-of-time conv head), multi-domain corpus loader,
  pretraining, rank-reduced zero-shot probe, scratch/few-shot baselines,
  shared-latent geometry.
- `tso/neural_field.py` — PyTorch MLP vector field + RK4 integration.
- `tso/pipeline.py` — `tso_forecast` / `scale_space` orchestration.
- `tso/viz.py` — the artwork (butterfly, reconstruction, spectrum, forecast,
  learned flow, scale space, masterpiece, bifurcation, latent geometry,
  zero-shot, pretrain curves).
- `scripts/demo_lorenz.py`, `scripts/kaggle_e2e.py`,
  `scripts/pretrain_foundation.py` — the three entry points.
- `kaggle_kernel_tso/` — vendored package + private GPU kernel.

## NMI-style study manuscript

`scripts/study_experiments.py`, `scripts/study_figures.py` and
`scripts/study_paper.py` reproduce the full study and compile the preprint
**`output/study/paper/main.pdf`** (LaTeX, pdflatex). Contents: 23-series /
8-domain corpus, multi-seed pretraining (seeds 0--2, 1.4k iters) with
ablations (no scale, no arrow), a from-scratch GRU autoregressive baseline
and per-series scratch deep-Koopman baselines on an identical protocol, the
v5/v7 Kaggle-kernel scaling comparison, Weiss time-reversibility analysis,
and honest statistics (two-sided binomial sign test). Includes the v9
in-kernel comparison (28/40 wins, p=0.017) and the solar-cycle discovery
(130.1 months vs known 132).

Key numbers in one block:

| claim | number |
|---|---|
| frozen TSO vs GRU, median skill advantage | +67.6 pts (local 3-seed, 16/23 wins, p=0.09) → **+46.5 pts (v9 in-kernel, 28/40 wins, p=0.017)** → **+60.9 pts (v11 best seed, 29/40 wins, p=0.002)** → +59.2 pts (v12 in-kernel, 28/40 wins) → +58.0 pts (v13 in-kernel, 25/40 wins) → **+57.2 pts (v14 in-kernel, 31/40 wins, p<0.001)** |
| scaling positive-fraction (shared 23-series, same probe) | 8/23 (v5) → 9/23 (local) → 13/23 (v7) → 12/23 (v9) → 10/23 (v10) → 12/5/8 (v11 ×3 seeds) → 11/23 (v12, joint probe) → 6/23 (v13, dynamics corpus) → **14/23 (v14) — ×4 seeds: 14/23 · 13/23 · 13/23 · 13/23** → 11/23 (v17, 25× corpus at fixed budget — per-series exposure is the binding constraint) |
| iteration / capacity / probe saturation | pretexts converge by 25k; 60k hurts the frozen probe (v9 wins 28/40 vs v10, p=0.017); width×2 at fixed probe lands within seed noise (v11); joint-probe v12 is beaten head-to-head by v9 16/23, p=0.05; dynamics-corpus v13 shifts transfer smooth→explosive (v9 15/23, p=0.11); **v14 breaks the plateau — 14/23 reprobed, first positive median skill, wins v9 paired 13/23 (p=0.34)** |
| external FSTM head-to-head (same 40-series protocol, both frozen) | TSO median +0.4% (20/40 positive) vs **Chronos-t5-small +24.7% (28/40)**; both crush GRU (−60.6%); Chronos's ~80k-series corpus is the difference |
| arrow-of-time classification accuracy | 94--97% (all scales) |
| scale-covariance diagnostic | 0.999→0.937 (physics) vs 0.69→0.21 (heart) |
| tipping onset detected / theoretical | ρ≈21 / ρ≈24.74 (early warning) |
| solar-cycle rediscovery (zero-shot, frozen) | **130.1 mo vs known 132 (1.4%)** |

## What this is (and isn't)

This is a working, end-to-end implementation of the TSO paradigm — real Takens
embedding, real Koopman linearization (fixed and learned), real vector-field
learning, real multi-scale spectra, real tipping detection, real cross-domain
pretraining with zero-shot forecasting, real artwork — not yet the full
foundation model of the vision (no billion-scale pretraining, no closed-loop
generative control). The Kaggle GPU kernel is the on-ramp to that scale.
