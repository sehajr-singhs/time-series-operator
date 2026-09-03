---
language:
- en
license: mit
tags:
- time-series
- foundation-model
- koopman-operator
- dynamical-systems
- zero-shot
datasets:
- sehajrsingh/tso-foundation-corpus-v11
pipeline_tag: time-series-forecasting
---

# TSO Foundation Model — v17

**Time is geometry.** The Time-Series Operator (TSO) is a foundation model
that learns the *shape* of dynamical systems instead of token-chunking
numbers. Pretraining uses four self-supervised pretexts — reconstruction,
Koopman-linear dynamics, scale covariance (a renormalization leg), and the
arrow of time — on 40 series across 8 domains
(electricity grids, meteorology, ECG, finance, epidemiology, economics,
solar physics, chaotic systems). Transfer to a never-seen series is a
**closed-form Koopman fit on the frozen latent**: zero gradient steps.

## Results (v17, latent 256 / hidden 768,
25,000 iters, CPU Kaggle (25k iters))

| Metric | Value |
|---|---|
| Zero-shot wins vs per-series GRU (40 series, in-kernel) | 30/40 (75.0%) |
| Median frozen skill vs persistence | +1.0% (GRU: -60.6%) |
| Solar-cycle rediscovery (held-out sunspots) | 129 mo vs known 132 (10.7 yr) |
| Arrow-of-time pretext accuracy | 82.8% |

The frozen operator **rediscovers the ~11-year Schwabe solar cycle** from a
scale-space scan of its fitted Koopman eigenvalues on the held-out sunspot
series — a purely structural, unsupervised discovery.

## Zero-shot usage

```python
import torch
from model import FoundationOperator, zero_shot_forecast

model = FoundationOperator(latent_dim=256, hidden=768)
model.load_state_dict(torch.load("foundation_model.pt", map_location="cpu"))

series = [...]  # your 1-D series (any domain, any sampling rate)
res = zero_shot_forecast(model, series)
print(res["skill_pct"], res["corr"])   # skill vs persistence on the test split
```

No training on your data is needed: the Koopman operator is fitted in closed
form on the frozen latent (ridge-regularized), then rolled out with an
envelope projection that keeps the trajectory on the observed attractor.

## Architecture

1. **Takens embeddings** — raw series → delay-embedding (per-series tau).
2. **Scale space** — fine + coarsened (renormalized) embeddings; the
   `scale_map` forces scale covariance, so periods like the solar cycle
   reappear as clean eigenmodes at coarse scales.
3. **Koopman lift** — a deep encoder flattens the nonlinear attractor into a
   latent where dynamics are approximately linear (`K`).
4. **Arrow of time** — a conv head classifies forward vs reversed windows.

## Honest limitations

- Spike / near-unit-root series (Dogecoin, covid-india) still defeat any
  closed-form probe; persistence is unbeatable there.
- The frozen *linear* probe plateaus: pretraining past pretext saturation
  (≈25k iters at this width) does not buy transfer — capacity and corpus
  breadth are the levers.
- Pretext losses pay off only above ~15k iterations.

## Reproduce

Training protocol: `scripts/modal_v11.py` (Modal, T4); corpus `scripts/build_corpus_modal.py`. Full study + manuscript:
`output/study/paper/main.pdf` in the source repo.

## Citation

```bibtex
@misc{tso-v17,
  title={Time is geometry: an operator foundation model that learns the shape of dynamical systems},
  author={{TSO} Project},
  year={2026}
}
```
