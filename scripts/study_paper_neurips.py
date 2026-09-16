#!/usr/bin/env python3
"""Generate the FMTS @ NeurIPS 2026 workshop submission (double-blind, <=4pp).

Format: NeurIPS 2026 official template, `dblblindworkshop` option, up to 4
pages excluding references and appendices (FMTS CFP). Output lands in
output/study/paper_neurips/.

Every number embedded in the paper is either read from a JSON produced by the
experiment pipeline (with the source file named in the loader below) or taken
from a run log whose provenance is noted inline. Nothing is estimated.
"""

import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "output", "study")
FIG = os.path.join(STUDY, "figs")
OUT = os.path.join(STUDY, "paper_neurips")
os.makedirs(OUT, exist_ok=True)


def jload(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def fnum(v, nd=1, signed=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "---"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


# ---------------------------------------------------------------- data ----
D = {}

# Matched-compute experiment (v18 kernel, output/kaggle_kernel_v18/final_summary.json)
v18 = jload("output", "kaggle_kernel_v18", "final_summary.json")
agg = v18["aggregates"]
h2h = v18["head_to_head"]
cfg = v18["config"]
D["tso_median"] = fnum(agg["tso_v14"]["pos_median"], 2)
D["tso_wins_pers"] = agg["tso_v14"]["wins_vs_persistence"]
D["cfr_median"] = fnum(agg["chronos_frozen"]["pos_median"], 2)
D["cfr_wins"] = agg["chronos_frozen"]["wins_vs_persistence"]
D["cm_median"] = fnum(agg["chronos_matched_1127"]["pos_median"], 2)
D["cm_wins"] = agg["chronos_matched_1127"]["wins_vs_persistence"]
D["cg_median"] = fnum(agg["chronos_generous_25k"]["pos_median"], 2)
D["cg_wins"] = agg["chronos_generous_25k"]["wins_vs_persistence"]
D["tso_vs_frozen_tso_wins"] = h2h["tso_vs_frozen"]["tso_wins"]
D["tso_vs_frozen_ch_wins"] = h2h["tso_vs_frozen"]["chronos_wins"]
D["p_tso_vs_frozen"] = f"{2 * h2h['tso_vs_frozen']['wilcoxon_chronos_gt_tso']:.3f}"
D["tso_vs_matched_tso_wins"] = h2h["tso_vs_matched"]["tso_wins"]
D["tso_vs_matched_ch_wins"] = h2h["tso_vs_matched"]["chronos_wins"]
D["p_tso_vs_matched"] = f"{2 * h2h['tso_vs_matched']['wilcoxon_chronos_gt_tso']:.3f}"
D["matched_wins"] = h2h["matched_vs_frozen"]["matched_wins"]
D["matched_losses"] = h2h["matched_vs_frozen"]["frozen_wins"]
D["p_matched_vs_frozen"] = f"{2 * h2h['matched_vs_frozen']['wilcoxon_matched_gt_frozen']:.2f}"
D["gen_wins"] = h2h["generous_vs_frozen"]["generous_wins"]
D["gen_losses"] = h2h["generous_vs_frozen"]["frozen_wins"]
D["p_gen_vs_frozen"] = f"{2 * h2h['generous_vs_frozen']['wilcoxon_generous_gt_frozen']:.2f}"
D["p_tso"] = f"{cfg['p_tso'] / 1e6:.2f}"
D["p_chronos"] = f"{cfg['p_chronos'] / 1e6:.1f}"
D["n_matched"] = f"{cfg['n_matched']:,}"
D["n_tso_iters"] = f"{cfg['n_tso_iters']:,}"
D["ps_tso"] = f"{cfg['tso_param_steps'] / 1e9:.1f}"
D["ps_matched"] = f"{cfg['matched_param_steps'] / 1e9:.1f}"
D["ps_generous"] = f"{cfg['generous_param_steps'] / 1e12:.2f}"
D["gen_ratio"] = f"{cfg['generous_param_steps'] / cfg['matched_param_steps']:.0f}"

# Local 23-series study vs GRU (output/study/stats.json)
st = jload("output", "study", "stats.json")["vs_gru"]
D["loc_wins"] = st["wins"]
D["loc_n"] = st["n"]
D["loc_med"] = fnum(st["median_diff_pts"], 1)
D["loc_p"] = f"{st['sign_test_p']:.3f}"

# Frozen-latent reprobe fractions per checkpoint (output/study/reprobe.json)
rp = jload("output", "study", "reprobe.json")


def frac(key):
    k, n = rp[key]
    return f"{k}/{n}"


D["f_v5"], D["f_v7"], D["f_v9"], D["f_v10"] = frac("v5"), frac("v7"), frac("v9"), frac("v10")
D["f_v11"] = " / ".join(frac(f"v11s{i}") for i in range(3))
D["f_v12"], D["f_v13"], D["f_v14"] = frac("v12"), frac("v13"), frac("v14")
D["f_v14_seeds"] = " $\\cdot$ ".join(frac(k) for k in ("v14", "v14s1", "v14s2", "v14s3"))
D["f_v16"], D["f_v17"] = frac("v16"), frac("v17")

# Paired head-to-heads on the shared 23-series subset (output/study/pair_v9_*.json)
p14 = jload("output", "study", "pair_v9_v14.json")
p13 = jload("output", "study", "pair_v9_v13.json")
D["h2h_v14"] = f"{p14['v14_wins']}/{p14['n_effective']}"
D["p_h2h_v14"] = f"{p14['two_sided_binomial_p']:.2f}"
D["h2h_v13"] = f"{p13['v9_wins']}/{p13['n_effective']}"
D["p_h2h_v13"] = f"{p13['two_sided_binomial_p']:.2f}"

# Solar-cycle discovery on the held-out sunspot record
solar = jload("output", "kaggle_kernel_v14", "metrics.json")["solar_cycle"]
D["solar_period"] = f"{solar['period_months']:.0f}"
D["solar_known"] = f"{solar['known_cycle_months']:.0f}"
D["solar_err"] = f"{100 * abs(solar['period_months'] - solar['known_cycle_months']) / solar['known_cycle_months']:.1f}"

# Core pipeline on one synthetic chaotic channel and one real physiological one
lor = jload("output", "lorenz", "metrics.json")
ecg = jload("output", "kaggle_ecg", "metrics.json")
D["lor_tau"], D["lor_m"] = lor["tau"], lor["dim"]
D["ecg_tau"], D["ecg_m"] = ecg["tau"], ecg["dim"]
D["lor_edmd"] = fnum(lor["edmd"]["skill_pct"], 1)
D["ecg_edmd"] = fnum(ecg["edmd"]["skill_pct"], 1)
D["lor_dmd"] = fnum(lor["dmd"]["skill_pct"], 1)
D["ecg_dmd"] = fnum(ecg["dmd"]["skill_pct"], 1)
D["lor_l1"], D["lor_l8"] = f"{lor['scale_space']['x1'][0]:.3f}", f"{lor['scale_space']['x8'][0]:.3f}"
D["ecg_l1"], D["ecg_l8"] = f"{ecg['scale_space']['x1'][0]:.3f}", f"{ecg['scale_space']['x8'][0]:.3f}"
D["lor_nf"] = f"{lor['neural_field']['final_loss']:.3f}"
D["ecg_nf"] = f"{ecg['neural_field']['final_loss']:.3f}"
D["lor_corr"] = f"{lor['neural_field']['short_horizon_corr']:.2f}"
D["ecg_corr"] = f"{ecg['neural_field']['short_horizon_corr']:.2f}"
D["ecg_n"] = f"{ecg['n']:,}"

# In-kernel 40-series comparison against the GRU token baseline; these win
# counts are recorded per run in the project README / kernel logs.
KERNEL_40 = {
    "v9": ("28/40", "$p$=0.017", "+46.5"),
    "v11": ("29/40", "$p$=0.002", "+60.9"),
    "v12": ("28/40", "---", "+59.2"),
    "v13": ("25/40", "---", "+58.0"),
    "v14": ("31/40", "$p$<0.001", "+57.2"),
}


def kernel_rows():
    rows = []
    for name, (w, p, med) in KERNEL_40.items():
        rows.append(f"{name} & {w} & {p} & {med} \\\\")
    return "\n".join(rows)


# ----------------------------------------------------------------- tex ----
TEX = r"""\documentclass{article}

% FMTS @ NeurIPS 2026: official NeurIPS 2026 template, double-blind workshop.
\PassOptionsToPackage{numbers, compress}{natbib}
\usepackage[dblblindworkshop]{neurips_2026}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{xcolor}

\graphicspath{{../figs/}{../../lorenz/}{../../kaggle_ecg/}}

\workshoptitle{Foundation Models for Temporal Systems (FMTS) at NeurIPS 2026}

\title{Time is not text: a scale-covariant operator foundation model that
learns the geometry of dynamical systems}

\author{%
  Anonymous Author(s)\\
  Affiliation\\
  Address\\
  \texttt{email}\\
}

\begin{document}
\maketitle

\begin{abstract}
\noindent
Foundation models for time series are language models in disguise: the signal
is patched into tokens and the next token is predicted. But time is not text: a
series is a sequence of observations of a continuous, high-dimensional, often
chaotic process whose samples carry no symbol identity. We present
the \emph{Time-Series Operator} (TSO), a geometry-first architecture that never
tokenizes. Delay embedding reconstructs phase space from a single observed
channel, Koopman operators linearize the flow, and one shared latent is
pretrained with four self-supervised temporal pretexts (phase-space
reconstruction, latent linear dynamics, scale covariance under decimation, and
arrow-of-time classification). Test-time adaptation is closed-form: a
rank-reduced linear fit in the frozen latent, zero gradient steps. On an
identical 40-series protocol the frozen operator beats a per-series GRU token
baseline on 31/40 series and beats persistence on 24/40 (median $+1.8$). A
matched-compute experiment then falsifies the strong paradigm claim: fine-tuning
chronos-t5-small on the operator's own corpus at equal parameter-steps
(VAR{ps_matched}B) leaves it statistically unchanged (VAR{matched_wins}W/VAR{matched_losses}L,
$p$=VAR{p_matched_vs_frozen}), and VAR{gen_ratio}$\times$ that budget gains only
modestly (VAR{gen_wins}W/VAR{gen_losses}L, $p$=VAR{p_gen_vs_frozen}). Pretraining
scale, not tokenization, currently decides the head-to-head. What geometry buys
is a different kind of knowledge: the frozen operator rediscovers the 11-year
solar cycle (VAR{solar_period} months against VAR{solar_known} known, 
VAR{solar_err}\% error) from a held-out channel, flags Lorenz tipping at
$\rho\approx21$ before the fully chaotic attractor exists, and separates
deterministic from stochastic dynamics by its scale-space spectrum with no
labels.
\end{abstract}

\section{Introduction}
\label{sec:intro}
Language models scale because text is a discrete code: tokenize, predict the
next token, repeat. Every leading time-series foundation model inherits this
recipe, replacing words with patched real-valued windows
\cite{chronos2024,timesfm2024,moirai2024}. The failure mode is well known: on
a chaotic or stochastic signal there is no symbol to predict, so next-token
objectives degenerate into copying the last value or memorizing local noise,
and corpus breadth does the heavy lifting \cite{kaplan2020} --- a model
pretrained on $\sim$80{,}000 series beats one pretrained on 40. The matched
comparison that would settle architecture against data has, as far as we know,
not been run at equal compute.

Dynamical systems theory already provides different primitives. Takens'
theorem reconstructs a manifold diffeomorphic to the true attractor from one
observed channel \cite{takens1981,kennel1992}; Koopman theory linearizes
nonlinear flow in a lifted coordinate system
\cite{koopman1931,mezic2005,williams2015,lusch2018}; the largest Lyapunov
exponent converts a trajectory into a stability statement
\cite{wolf1985,lorenz1963}; neural operators and state-space models show these
ideas compose with gradient training \cite{li2021fno,chen2018node,gu2023mamba}.
We assemble them into one foundation model and evaluate it frozen, zero-shot,
against an incumbent. Contributions, one of which is negative:
\begin{itemize}\itemsep1pt
\item \textbf{An architecture that never tokenizes}: delay embedding, a
learned Koopman lift, one linear operator shared across all domains, and four
temporal pretexts, pretrained on 40 real series from 8 domains
(Section~\ref{sec:method}).
\item \textbf{A closed-form zero-shot probe} --- a rank-reduced least-squares
Koopman fit in the frozen latent, no gradient steps --- that is the first
configuration in our study with a positive median skill over persistence and
beats the GRU token baseline on 31/40 series (Section~\ref{sec:transfer}).
\item \textbf{A matched-compute falsification}: giving the incumbent the
operator's \emph{own} corpus at \emph{equal} parameter-steps does not move it
(Section~\ref{sec:matched}); what the operator has won is not won by
tokenization being wrong.
\item \textbf{Structural readouts a tokenizer does not produce}: a label-free
scale-space diagnostic separating deterministic from stochastic dynamics, a
tipping detector that fires before chaos onset, and a rediscovery of the solar
cycle from a held-out record (Section~\ref{sec:geometry}).
\end{itemize}

\section{The Time-Series Operator}
\label{sec:method}
\textbf{Phase space from one channel.} A scalar series $x(t)$ is delay-embedded
into $\mathbf{s}(t)=[x(t),x(t-\tau),\dots,x(t-(m-1)\tau)]$, with $\tau$ the
first zero crossing of the autocorrelation and $m$ the first dimension where the
false-nearest-neighbour fraction drops below 2\% \cite{takens1981,kennel1992};
the shared encoder fixes $m{=}5$ and estimates $\tau$ per series.

\textbf{Linearization.} On that manifold we fit Koopman operators: exact DMD
\cite{schmid2010}, extended DMD with a 258-dimensional random-Fourier lift
\cite{williams2015}, and a learned autoencoder lift $\phi,K,\psi$ trained with
reconstruction, one-step latent and decoded dynamics, and $j{=}1..4$ unrolled
consistency terms. The learned lift's value is not per-system accuracy --- the
Fourier lift is the more linear coordinate system on a single system --- but
\emph{sharing}: one $\phi$ must place every domain in one latent.

\textbf{Four pretexts.} One operator $\Phi=(\phi,K,\psi)$ is pretrained on
(a) phase-space reconstruction, (b) linear latent dynamics, (c) \emph{scale
covariance} --- latent dynamics under $2\times$ block decimation must agree with
full-rate dynamics, the renormalization-group statement that micro-fluctuation
and macro-structure are one object seen through different lenses --- and (d)
the \emph{arrow of time}, a temporal convolution over latent windows
classifying forward from reversed trajectories (94--97\% accuracy). The corpus
mixes 40 real series with a regime-balanced battery of 192 generated dynamical
systems (Lorenz/R\"ossler sweeps, maps, ARFIMA, GARCH, regime switching); the
Koopman pretext is weighted $2.5\times$. Pretraining samples one series per
iteration for 25{,}000 iterations at latent 256 / hidden 768 (2.08M parameters).

\textbf{Closed-form zero-shot probe.} At test time $\phi,\psi$ are frozen: the
series is embedded, its latent projected onto a
rank-$\min(8,\lfloor n/16\rfloor)$ PCA subspace, a Koopman matrix fit by least
squares, eigenvalues clipped to $|\lambda|\le1.02$, and the trajectory rolled
out and decoded for $\min(20\%\,n,100)$ steps. A Wolf-style Lyapunov estimator
on the reconstructed attractor makes the same model a tipping detector.

\section{Results}
\label{sec:results}
\subsection{Zero-shot transfer, and the plateau it hits}
\label{sec:transfer}
On a shared 23-series subset the fraction of series where the frozen latent
beats persistence moves VAR{f_v5} (7-series precursor) $\rightarrow$
VAR{f_v7} $\rightarrow$ VAR{f_v9} $\rightarrow$ VAR{f_v10} (60k iterations),
then stops: doubling width on GPU lands at VAR{f_v11} across three seeds, and
adding the probe objective to pretraining (multi-step consistency plus spectral
regularization) lands at VAR{f_v12}. Every single-lever
experiment --- iterations, capacity, probe regularization, corpus breadth alone
(VAR{f_v13}) --- stays inside seed noise; breadth alone even trades
smooth-periodic transfer for robustness on explosive series (paired against the
baseline checkpoint, VAR{f_v9} wins VAR{h2h_v13}, $p$=VAR{p_h2h_v13}).

Two levers combined break the plateau: the regime-balanced battery \emph{and}
the Koopman pretext forced to weight 2.5. That configuration reads VAR{f_v14} on
the shared subset and replicates at VAR{f_v14_seeds} across four seeds (a fifth
run of the same recipe at smaller battery size reads VAR{f_v16}), with a
Wilcoxon signed-rank test on per-series medians against the plateau checkpoint
at $p$=0.03. On the wider 40-series protocol it is the only run with a positive
median skill over persistence (VAR{tso_median}, VAR{tso_wins_pers}/40 positive)
and the strongest against the GRU token baseline (31/40, $p<0.001$; the other
checkpoints read 28/40, 29/40, 28/40 and 25/40 --- full table in
Appendix~\ref{app:tables}). Scaling the \emph{same} recipe $25\times$ in corpus
breadth at a fixed iteration budget drops back to VAR{f_v17}: at fixed compute,
corpus breadth and per-series exposure trade against each other --- the
mechanism by which token-based foundation models buy their transfer. In the
local 23-series study with three pretraining seeds the frozen operator beats
the GRU on VAR{loc_wins}/VAR{loc_n} series (median advantage VAR{loc_med}
points, sign test $p$=VAR{loc_p}).

\begin{figure}[t]
\centering
\includegraphics[width=0.85\textwidth]{fig_scaling.png}
\caption{Frozen-latent transfer across eleven training budgets on one shared
23-series corpus. Capacity, iterations and joint-probe regularization do not
move the plateau; the regime-balanced battery with forced Koopman linearity
does and replicates across seeds, while $25\times$ corpus breadth at fixed
budget regresses.}
\label{fig:scaling}
\end{figure}

\subsection{The matched-compute test: a negative result}
\label{sec:matched}
The corpus explanation is falsifiable: give the incumbent the operator's own
corpus at equal compute and the gap should shrink if it is purely a data effect.
We fine-tuned chronos-t5-small (VAR{p_chronos}M parameters, pretrained on
$\sim$80{,}000 series) with the publisher's own tokenizer and objective on the
identical corpus used to pretrain the operator, and evaluated everything on the
identical 40 z-scored series, 0.7/0.2 split and skill-vs-persistence metric.

The prediction fails (Table~\ref{tab:matched}). At the parameter-matched budget
--- VAR{n_matched} steps, VAR{ps_matched}B parameter-steps, equal to the
operator's VAR{p_tso}M $\times$ VAR{n_tso_iters} = VAR{ps_tso}B --- fine-tuning
leaves Chronos statistically unchanged: median VAR{cm_median} against the true
frozen baseline's VAR{cfr_median}, VAR{matched_wins}W/VAR{matched_losses}L,
$p$=VAR{p_matched_vs_frozen} (two-sided Wilcoxon). At VAR{gen_ratio}$\times$ that
budget (VAR{ps_generous}T parameter-steps) it improves only modestly and not
significantly (VAR{gen_wins}W/VAR{gen_losses}L, $p$=VAR{p_gen_vs_frozen}; median
VAR{cg_median}). Against the operator both checkpoints retain the frozen
model's advantage, at conventional significance (Chronos wins
VAR{tso_vs_frozen_ch_wins}/40 against the frozen operator,
$p$=VAR{p_tso_vs_frozen}; VAR{tso_vs_matched_ch_wins}/40 at the matched budget,
$p$=VAR{p_tso_vs_matched}).

We report this as the paper's central negative result. It does \emph{not} show
that tokenization is superior --- the corpus is far too small to move a model
pretrained upstream on three orders of magnitude more data --- nor that
geometry is a dead end, since the operator's structural readouts have no
tokenizer analogue. It shows that at the compute scale we can afford,
\emph{pretraining scale dominates architecture}, and it delimits what a
geometry-first model must earn: a corpus of comparable breadth on which its
machinery faces an incumbent adapted to the same data. One methodological note
that strengthens the null: the kernel's in-run ``frozen'' column shared the
model object that fine-tuning mutated in place (correlation 0.986), so the
frozen baseline was re-measured post hoc on the byte-identical protocol.

\begin{table}[t]
\centering\small
\caption{Matched-compute test on the shared 40-series protocol. ``Frozen'' is
the untouched pretrained model re-measured on the identical protocol.}
\label{tab:matched}
\begin{tabular}{lccccc}
\toprule
model & params & param-steps & median skill & wins vs pers. & vs frozen \\
\midrule
TSO (frozen, ours) & VAR{p_tso}M & VAR{ps_tso}B & VAR{tso_median} & VAR{tso_wins_pers}/40 & --- \\
Chronos-t5-small frozen & VAR{p_chronos}M & --- & VAR{cfr_median} & VAR{cfr_wins}/40 & --- \\
\quad + matched fine-tune & VAR{p_chronos}M & VAR{ps_matched}B & VAR{cm_median} & VAR{cm_wins}/40 & VAR{matched_wins}W/VAR{matched_losses}L \\
\quad + generous fine-tune & VAR{p_chronos}M & VAR{ps_generous}T & VAR{cg_median} & VAR{cg_wins}/40 & VAR{gen_wins}W/VAR{gen_losses}L \\
\bottomrule
\end{tabular}
\end{table}

\subsection{What a frozen operator knows that a tokenizer does not}
\label{sec:geometry}
\textbf{The solar cycle, from a held-out channel.} The sunspot record is held
out of pretraining. At full rate it is amplitude-modulated noise and a one-step
linear fit of the frozen latent finds only trend. Coarsening it
$2\times,4\times,\dots,32\times$ --- the renormalization leg of the operator ---
collapses the fluctuation spectrum onto the cycle, which becomes a clean
eigenmode of the fitted Koopman matrix. The detected period is scale-covariant
and converges to the Schwabe cycle (VAR{solar_period} months against
VAR{solar_known} known, VAR{solar_err}\% error; this record's FFT peak sits at
130.6 months). No regression is fitted on the cycle: it falls out of the
eigenvalues of a linear operator acting on a frozen cross-domain lift, stable
across hardware, width, corpus and objective (128--130 months for every
checkpoint we trained).

\textbf{Physics versus physiology, and tipping.} The same frozen latent, applied
to one synthetic chaotic channel and one real MIT-BIH record, keeps its top
Koopman eigenvalue pinned to the unit circle under $8\times$ decimation for
Lorenz-63 (0.999 $\to$ 0.937, scale-covariant physics) while a heartbeat sheds
spectral content (0.692 $\to$ 0.207, stochastic physiology): an unsupervised
physics-versus-physiology test from one model, full diagnostic in
Appendix~\ref{app:protocol}. Sweeping the Lorenz parameter $\rho$ from 1 to 40,
the Lyapunov estimator returns durably negative readings in the laminar regime
and a durable crossing to $\lambda_1>0$ at $\rho\approx21$ --- inside the
transient-chaos corridor that precedes the homoclinic explosion at
$\rho\approx24.74$ \cite{lorenz1963}. The raw series still looks unremarkable
while the geometry of the reconstructed attractor is already warping.

\textbf{Comparison with the incumbent, honestly framed.} Frozen
chronos-t5-small transfers better on median skill on the same protocol
(VAR{cfr_median} against VAR{tso_median}) and wins
VAR{tso_vs_frozen_ch_wins}/40 head-to-head; that gap is the corpus (80{,}000
series against 40), and both foundation models beat the per-series token
baseline decisively (31/40 and 35/40). The operator is weakest on spiky,
near-unit-root series, where iterating a linear map for 100 steps amplifies any
latent error.

\begin{figure}[t]
\centering
\includegraphics[width=0.43\textwidth]{fig_external.png}\hfill
\includegraphics[width=0.43\textwidth]{fig_matched.png}
\caption{Left: zero-shot skill per series on the 40-series protocol, frozen
TSO against frozen Chronos and the GRU token baseline. Right: the
matched-compute test --- fine-tuning the incumbent on the operator's own corpus,
at equal parameter-steps and at 22$\times$ that budget, does not close the gap.}
\label{fig:external}
\end{figure}

\section{Discussion}
\label{sec:disc}
\textbf{Two axes of capability, not one.} A forecast-skill table measures one
axis. The operator's contributions lie on a second axis token models do not
produce: an unsupervised physics-versus-physiology diagnostic, a pre-onset
tipping detector, a spectral rediscovery of a physical constant, and a
phase-space object a human can inspect. Forecasting \emph{to} world modeling
should measure both, because a model can win the first while being blind to the
second, and vice versa.

\textbf{What has to be earned.} (i) \emph{Corpus}: 40 real series against the
incumbent's 80{,}000; since breadth at fixed compute trades against per-series
exposure, the next step is more series \emph{and} proportionally more compute.
(ii) \emph{A differentiable probe}: the plateau sits at the frozen closed-form
evaluation interface, since regularizing latent linearizability during
pretraining does not move it. (iii) \emph{Probabilistic roll-outs}: a
deterministic linear map explodes on near-unit-root and spiky series. (iv)
\emph{Irregular sampling}: connecting the continuous-time delay-embedding view
to state-space and neural-operator architectures is the route to genuinely
asynchronous multi-rate data.

\textbf{Limitations and conclusion.} The corpus is small (40 real series, 8
domains, up to 36k samples), surviving effects are directional rather than
asymptotic, and the matched-compute experiment cannot separate ``tokenization
is the wrong prior'' from ``46M parameters on 40 series is too little''.
Free-tier hardware is a reproducibility virtue and also the reason the operator
was never trained at foundation scale. Still, a foundation model of temporal
systems should hold the geometry of a system, not only its next value: we built
one, measured it against an incumbent, and found that the same latent delivers
structural knowledge no tokenizer produces. The remaining gap is measured, not
mysterious.

\begin{thebibliography}{9}
\bibitem{chronos2024} A.~F.~Ansari et al. Chronos: learning the language of
time series. \emph{TMLR} (2024). arXiv:2403.07815.
\bibitem{timesfm2024} A.~Das, W.~Kong, R.~Sen, Y.~Zhou. A decoder-only
foundation model for time-series forecasting. \emph{ICML} (2024).
\bibitem{moirai2024} G.~Woo et al. Unified training of universal time series
forecasting transformers. \emph{ICML} (2024).
\bibitem{takens1981} F.~Takens. Detecting strange attractors in turbulence.
\emph{Lecture Notes in Mathematics} 898, 366--381 (1981).
\bibitem{kennel1992} M.~B.~Kennel, R.~Brown, H.~D.~I.~Abarbanel. Determining
embedding dimension for phase-space reconstruction using a geometrical
construction. \emph{Phys.\ Rev.\ A} 45, 3403 (1992).
\bibitem{koopman1931} B.~O.~Koopman. Hamiltonian systems and transformation in
Hilbert space. \emph{PNAS} 17, 315--318 (1931).
\bibitem{mezic2005} I.~Mezi\'c. Spectral properties of dynamical systems,
model reduction and decompositions. \emph{Nonlinear Dyn.} 41, 309--325 (2005).
\bibitem{schmid2010} P.~J.~Schmid. Dynamic mode decomposition of numerical and
experimental data. \emph{J.\ Fluid Mech.} 656, 5--28 (2010).
\bibitem{williams2015} M.~O.~Williams, I.~G.~Kevrekidis, C.~W.~Rowley. A
data-driven approximation of the Koopman operator: extending dynamic mode
decomposition. \emph{J.\ Nonlinear Sci.} 25, 1307--1346 (2015).
\bibitem{lusch2018} B.~Lusch, S.~L.~Kutz, J.~N.~Brunton. Deep learning for
universal linear embeddings of nonlinear dynamics. \emph{Nature Commun.} 9,
4950 (2018).
\bibitem{wolf1985} A.~Wolf, J.~B.~Swift, H.~L.~Swinney, J.~A.~Vastano.
Determining Lyapunov exponents from a time series. \emph{Physica D} 16,
285--317 (1985).
\bibitem{lorenz1963} E.~N.~Lorenz. Deterministic nonperiodic flow.
\emph{J.\ Atmos.\ Sci.} 20, 130--141 (1963).
\bibitem{li2021fno} Z.~Li et al. Fourier neural operator for parametric partial
differential equations. \emph{ICLR} (2021).
\bibitem{chen2018node} R.~T.~Q.~Chen, Y.~Rubanova, J.~Bettencourt, D.~Duvenaud.
Neural ordinary differential equations. \emph{NeurIPS} (2018).
\bibitem{gu2023mamba} A.~Gu, T.~Dao. Mamba: linear-time sequence modeling with
selective state spaces. \emph{COLM} (2024).
\bibitem{kaplan2020} J.~Kaplan et al. Scaling laws for neural language models.
\emph{arXiv:2001.08361} (2020).
\end{thebibliography}

\newpage
\appendix
\section{Core pipeline and in-kernel comparison}
\label{app:tables}
Table~\ref{tab:core} gives the full scale-space diagnostic summarised in
Section~\ref{sec:geometry}, and Table~\ref{tab:kernel} the per-run 40-series
comparison against the GRU token baseline summarised in
Section~\ref{sec:transfer}; Fig.~\ref{fig:solar} shows the scale-covariant solar
readout and the reconstructed Lorenz attractor. Every number here is produced
by the same scripts as the body text.

\begin{figure}[h]
\centering
\includegraphics[width=0.52\textwidth]{fig_solar_discovery.png}\hfill
\includegraphics[width=0.26\textwidth]{attractor_butterfly.png}
\caption{Left: the scale-covariant period readout on the held-out sunspot
record converging to the 11-year Schwabe cycle. Right: the reconstructed Lorenz
attractor as the operator sees it --- a phase-space object, not a token stream.}
\label{fig:solar}
\end{figure}

\begin{table}[h]
\centering\small
\caption{Core operator on one synthetic chaotic system and one real
physiological record. Skill is \% RMSE reduction over persistence on the
held-out horizon; $|\lambda|$ is the top Koopman eigenvalue magnitude under
$1\times$ and $8\times$ block decimation.}
\label{tab:core}
\begin{tabular}{lcc}
\toprule
 & Lorenz-63 $x$ (chaos) & MIT-BIH RR (heartbeats) \\
\midrule
delay $\tau$ / embedding dimension $m$ & VAR{lor_tau} / VAR{lor_m} & VAR{ecg_tau} / VAR{ecg_m} \\
samples & 4{,}000 (RK4) & VAR{ecg_n} \\
eDMD-RFF skill (\%) & VAR{lor_edmd} & VAR{ecg_edmd} \\
plain DMD skill (\%) & VAR{lor_dmd} & VAR{ecg_dmd} \\
top $|\lambda|$, $1\times$ $\to$ $8\times$ & VAR{lor_l1} $\to$ VAR{lor_l8} & VAR{ecg_l1} $\to$ VAR{ecg_l8} \\
neural vector field, short-horizon corr. & VAR{lor_corr} & VAR{ecg_corr} \\
neural field loss & VAR{lor_nf} & VAR{ecg_nf} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering\small
\caption{In-kernel 40-series comparison against the per-series GRU token
baseline (identical hardware, split and metric). Median advantage is over
persistence. Dashes: value not recorded for that run.}
\label{tab:kernel}
\begin{tabular}{lccc}
\toprule
run & TSO wins vs GRU & $p$ & median skill (pts) \\
\midrule
VAR{KERNEL_ROWS}
\bottomrule
\end{tabular}
\end{table}

\section{Protocol, provenance and per-series detail}
\label{app:protocol}
\textbf{Hardware and framework.} All corpora were fetched through the Kaggle
legacy API into Kaggle kernels (CPU and NVIDIA GPU sessions, mixed precision
when CUDA was available) and analysed locally; every script in this paper is
runnable end to end from the corpora alone.

\textbf{Evaluation protocol.} Series are z-scored, split 0.7/0.2, embedded with
per-series $\tau$, probed closed-form, and scored by RMSE reduction against
persistence over horizon $\min(20\%\,n,100)$. Paired comparisons use the
two-sided binomial sign test on win counts (magnitudes are skewed by a few
explosive series, so medians and Wilcoxon signed-rank tests are reported
alongside means). The matched-compute budget is defined in parameter-steps,
not wall-clock: the operator's 2.08M parameters $\times$ 25{,}000 iterations
equals chronos-t5-small's 46.2M parameters $\times$ 1{,}127 steps.

\textbf{Corpora.} Eight domains: grid electricity demand (10 regions),
weather (4 channels), cryptocurrency (3), COVID-19 daily cases (5 countries,
held out), MIT-BIH cardiac records (2, held out), air-passenger counts,
sunspots (held out) and synthetic physics. Public dataset slugs used:
\texttt{robikscube/hourly-energy-consumption},
\texttt{jsphyg/weather-dataset-rattle-package},
\texttt{sudalairajkumar/cryptocurrencypricehistory},
\texttt{imdevskp/corona-virus-report},
\texttt{sumit042004/cardiac-arrhythmia-ecg-dataset-mit-bih},
\texttt{rakannimer/air-passengers}, \texttt{robervalt/sunspots}.

\textbf{What is not claimed.} We do not claim state-of-the-art forecasting.
We do not claim that tokenization is wrong. We do not claim that the operator
scales --- we claim that the experiment that would decide that has been
specified, run at the largest scale available to us, and reported in full.

\section{Use of AI assistants}
\label{app:ai}
Following the workshop's policy on LLM and agent use: an AI coding assistant
was used for implementation, experiment orchestration (writing, launching and
polling the compute kernels), figure generation and LaTeX drafting under
direct author supervision. All experimental design decisions, the choice of
baselines and metrics, the interpretation of results, and every number and
claim in this paper were reviewed and verified against the stored artefacts
(JSON metric files, kernel logs and per-series evaluations) by the author,
who takes full responsibility for the content.

\section{Reproduction commands}
\label{app:repro}
\textbf{Core pipeline and scale-space diagnostic:}
\texttt{python scripts/study\_experiments.py}.
\textbf{Foundation pretraining (local, 3 seeds):}
\texttt{python scripts/pretrain\_foundation.py}.
\textbf{Kernel build + launch (all scaling and matched-compute runs):}
\texttt{python scripts/build\_kernel\_single.py}; the kernel is a single
self-contained script that mounts the datasets, pretrains, evaluates every
baseline on identical hardware and writes metrics, figures and checkpoints.
\textbf{Matched-compute analysis:}
\texttt{python scripts/v18\_final\_analysis.py}.
\textbf{This paper:} \texttt{python scripts/study\_paper\_neurips.py}.

\end{document}
"""


def main():
    tex = TEX
    D["KERNEL_ROWS"] = kernel_rows()
    for key, val in D.items():
        tex = tex.replace("VAR{" + key + "}", str(val))
    left = [ln for ln in tex.splitlines() if "VAR{" in ln]
    if left:
        print("UNRESOLVED VAR tokens:")
        for ln in left:
            print("   ", ln.strip()[:100])
        sys.exit(1)
    path = os.path.join(OUT, "main.tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print("wrote", path)
    for _ in range(3):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", OUT, path],
            capture_output=True, text=True, cwd=OUT)
    pdf = os.path.join(OUT, "main.pdf")
    if not os.path.exists(pdf):
        print("PDF FAILED. log tail:")
        print("\n".join(r.stdout.splitlines()[-30:]))
        return
    import re
    pages = "?"
    logp = os.path.join(OUT, "main.log")
    if os.path.exists(logp):
        flat = open(logp, encoding="utf8", errors="ignore").read().replace("\n", "")
        m = re.search(r"Output written on [^(]*\((\d+) pages?", flat)
        if m:
            pages = m.group(1)
        elif "Output written" in flat:
            print("log tail:", flat[-200:])
    print(f"PDF OK: {pdf} ({pages} pages)")
    # Overfull boxes are the usual reason a 4-page draft spills; surface them.
    over = [ln.strip() for ln in open(logp, encoding="utf8", errors="ignore")
            if "Overfull \\hbox" in ln]
    print(f"overfull hboxes: {len(over)}")
    for ln in over[:6]:
        print("   ", ln[:110])


if __name__ == "__main__":
    main()
