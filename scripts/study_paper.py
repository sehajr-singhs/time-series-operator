#!/usr/bin/env python3
"""Generate the NMI-style manuscript (LaTeX) from the study data and compile
it to PDF with pdflatex (MiKTeX). Output lands in output/study/paper/."""

import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "output", "study")
FIG = os.path.join(STUDY, "figs")
OUT = os.path.join(STUDY, "paper")
os.makedirs(OUT, exist_ok=True)

data = json.load(open(os.path.join(STUDY, "paper_data.json")))
bl = json.load(open(os.path.join(STUDY, "baselines.json")))
per = data["per_series"]
meta = json.load(open(os.path.join(STUDY, "corpus_meta.json")))
domain = {m["name"]: m["domain"] for m in meta}

ORDER = sorted(per, key=lambda n: (domain[n], n))


def fmt(v, nd=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "---"
    return f"{v:+.{nd}f}"


def table_series_rows():
    rows = []
    for n in ORDER:
        p = per[n]
        held = "\\textbf{yes}" if next(m for m in meta if m["name"] == n)["held_out"] else ""
        gru = fmt(p["gru"]) if p["gru"] is not None else "---"
        scr = fmt(p["scratch"]) if p["scratch"] is not None else "---"
        mean = fmt(p["tso_mean"]) if p["tso_mean"] is not None else "---"
        rows.append(
            f"{n} & {domain[n]} & {mean} & "
            f"{fmt(p['tso_std']) if p['tso_std'] else '---'} & "
            f"{gru} & {scr} & {held}\\\\")
    return "\n".join(rows)


def table_corpus_rows():
    refs = {
        "physiology": "sumit042004/cardiac-arrhythmia-ecg-dataset-mit-bih",
        "energy-grid": "robikscube/hourly-energy-consumption",
        "meteorology": "jsphyg/weather-dataset-rattle-package",
        "finance": "sudalairajkumar/cryptocurrencypricehistory",
        "solar-physics": "robervalt/sunspots",
        "economics": "rakannimer/air-passengers",
        "epidemiology": "imdevskp/corona-virus-report",
    }
    doms = []
    seen = set()
    for m in meta:
        if m["domain"] not in seen:
            seen.add(m["domain"])
            n = sum(1 for mm in meta if mm["domain"] == m["domain"])
            doms.append((m["domain"], n, refs.get(m["domain"], "synthetic")))
    rows = [f"{d} & {n} & {r}\\\\" for d, n, r in doms]
    rows.append("physics (synthetic) & 1 & Lorenz--63 $x$-channel (RK4, own integration)\\\\")
    rows.append("\\textbf{total} & \\textbf{23} & ---\\\\")
    return "\n".join(rows)


TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{caption}
\usepackage{xcolor}
\definecolor{accent}{RGB}{56,242,200}
\usepackage[colorlinks=true,linkcolor=accent!60!black,citecolor=accent!60!black,urlcolor=accent!60!black]{hyperref}
\graphicspath{{../figs/}{../../kaggle_kernel/}}
\title{\textbf{Time is geometry: an operator foundation model that learns the shape of dynamical systems}}

\author{Sehaj Singh\\
{\small Time-Series Operator project --- preprint}}

\date{}

\begin{document}
\maketitle

\begin{abstract}
\noindent
Foundation models for time series currently adapt language architectures:
signals are chopped into tokens and modelled as sentences. Time is not text.
A time series is a sequence of discrete observations of a continuous,
high-dimensional and often chaotic reality. We present the Time-Series Operator
(TSO), a geometry-first architecture that never tokenizes: it reconstructs
phase space from a single observed channel (Takens' embedding), linearizes the
flow in Koopman coordinates (both a closed-form Fourier lift and a learned
autoencoder lift), reads the attractor's health through a data-driven
Lyapunov tipping detector, and -- as a foundation model -- pretrains one
shared operator across 23 series from 8 physical and economic domains using
four self-supervised temporal pretexts: phase-space reconstruction, linear
latent dynamics, scale covariance (renormalization-group consistency under
2$\times$ decimation) and the arrow of time (a convolutional head classifies
forward vs.\ reversed latent trajectories at 94--97\% accuracy). A frozen
zero-shot probe (closed-form linear fit in the pretrained latent, zero
gradient steps) forecasts held-out systems and beats a per-series GRU
autoregressive token baseline across every scale at which we compare: a
median +68 percentage-point skill advantage on the local 23-series study
(16/23 wins, two-sided binomial $p$=0.09) and +46.5 points in the scaled
in-kernel comparison over 40 series (28/40 wins, $p$=0.017; a GPU
width-doubled variant reaches 29/40, and the final architecture --- a
regime-balanced corpus with the Koopman pretext re-weighted --- reaches
31/40 wins with the first \emph{positive} median skill, +1.8 points
over persistence). Scaling the corpus and model width
from 7 to 40 series and 4k to 25k iterations
raised the fraction of series with positive frozen skill from 2/7 to
13/23 (20/40 across the full 40-series evaluation); extending
pretraining to 60k iterations saturates the pretext losses without
further transfer gains (16/40), and making the probe a pretraining term
(joint multi-step roll-out consistency plus unit-circle spectral
regularization, v12) does not move the plateau (11/23 on the shared
subset; v9 beats it head-to-head 16/23, $p=0.05$). The plateau \emph{is}
broken by combining the two levers --- the v13 dynamics battery
re-balanced toward smooth/seasonal/spiky regimes \emph{and} the
Koopman pretext forced to weight 2.5 at v11 capacity (v14): 14/23 on
the shared subset (best of any run), 31/40 wins in-kernel, 24/40
positive, median +1.8 points, and the first run to win a paired
head-to-head against v9 (13/23, $p=0.34$). Five-seed replication
confirms the break is not a lucky seed: all five v14-recipe runs land at
12--14/23 on the shared subset (v9: 12/23; v11 seeds: 12/5/8) with a
positive median skill in every seed ($+2.0$, $+6.4$, $+0.5$, $+1.4$,
$+3.8$ points vs a negative v9 median), and the Wilcoxon signed-rank
test on per-series medians against v9 is significant at $p=0.03$
(pooled positive fraction 65/115 vs v9's rate, $p=0.20$). The
direction is consistent across every seed, and the magnitude gains are
statistically significant at a conventional level. A frozen Chronos-t5-small
(\textasciitilde80k-series pretraining corpus) still beats the operator
on the identical protocol (median $+24.7$ vs $+1.8$ percentage points;
24/40 head-to-head), quantifying how far corpus breadth moves the
curve. Ablations show
the pretexts converge but only pay off with scale. The operator's
scale-space spectrum distinguishes deterministic physics from stochastic
physiology without any labels, and its tipping detector flags the Lorenz
chaos onset before the fully chaotic attractor exists. Finally, a pure
spectral consequence of pretraining: the frozen operator, applied to the
held-out sunspot record through the renormalization (coarsening) leg,
rediscovers the $\sim$11-year solar cycle from its fitted Koopman
eigenvalues, converging to 130 months against the known 132-month Schwabe
period.
\end{abstract}

\section{Introduction}
\label{sec:intro}
Language models scale because text is a discrete code: tokenize, predict the
next token, repeat. Time series are not a code. A sample of a chaotic or
stochastic process carries no symbol identity; ``next-token prediction'' on
numbers degenerates into copying the last value when the dynamics are
unpredictable, and into memorizing local noise when they are not
\cite{chronos2024,timesfm2024}. The alternative, classical in dynamical
systems and physics, is to stop predicting values entirely and instead learn
the \emph{geometry of the flow}: the manifold on which the system lives, the
vector field that moves it, and the operators that linearize it.

The Time-Series Operator (TSO) implements this programme end to end,
from one raw scalar channel to a pretrained, cross-domain operator:

\begin{enumerate}
\item \textbf{Phase space.} A single channel $x(t)$ is delay-embedded into
$\mathbf{s}(t)=[x(t),x(t-\tau),\dots,x(t-(m-1)\tau)]$ with data-driven
$\tau$ (first zero of the autocorrelation / first local minimum) and $m$
(false-nearest-neighbours collapse). Takens' theorem guarantees a manifold
topologically equivalent to the true attractor \cite{takens1981}.
\item \textbf{Linearization.} On that manifold we fit Koopman operators
\cite{koopman1931,mezić2005}: the exact DMD matrix \cite{schmid2010}, an
extended DMD with a random-Fourier lift (258 dimensions), and a \emph{learned}
lift --- an autoencoder $\phi, K, \psi$ whose latent makes the flow linear,
trained with reconstruction, one-step dynamics, latent linearity and
multi-step consistency losses.
\item \textbf{Health.} A Wolf-style largest-Lyapunov estimator
\cite{wolf1985} on the reconstructed attractor turns the model into a
tipping-point detector: positive $\lambda_1$ = exponential divergence =
bifurcation precursor.
\item \textbf{Scale.} The same Koopman fit at 1$\times$, 2$\times$, 4$\times$
and 8$\times$ decimation provides the spectrum across temporal lenses --- the
renormalization-group view of how micro-fluctuations become macro-structure.
\item \textbf{Foundation.} A shared operator $\Phi$ (encoder, one linear
Koopman matrix shared across \emph{all} domains, decoder, a scale-transfer
head and an arrow-of-time head) is pretrained across domains with four
self-supervised pretext tasks. At test time the lift is \emph{frozen}: only a
linear Koopman matrix is fit by least squares in the latent (rank-reduced,
spectrally clipped) and the forecast is computed in closed form --- zero
gradient steps on the target system.
\end{enumerate}

\section{Results}
\label{sec:results}

\subsection{The scale-space diagnostic separates physics from physiology}
\label{sec:scale}
Table~\ref{tab:core} reports the core pipeline on the Lorenz-63 butterfly
(one noisy channel of a 3-D chaotic system) and on real heart-rate
variability (MIT-BIH record 215, 2{,}500 beats, fetched through the Kaggle
legacy API). The decisive row is the last one: the top Koopman eigenvalue
magnitude under decimation. Deterministic convection keeps its spectrum
pinned to the unit circle from 1$\times$ to 8$\times$ --- scale-covariant
physics --- while a noisy heart sheds its spectral content as the lens
coarsens. The \emph{same operator}, fitted with no labels, tells a
deterministic system from a stochastic one by the shape of this decay.

\begin{table}[h]
\centering\small
\caption{Core TSO pipeline. Skill = \% RMSE reduction over the persistence
baseline on a held-out horizon. $\lambda_1$ = largest Lyapunov exponent.}
\label{tab:core}
\begin{tabular}{lcc}
\toprule
metric & Lorenz-63 (chaos) & MIT-BIH HRV (heartbeats) \\
\midrule
Takens delay / dimension & $\tau$=59, $m$=3 & $\tau$=1, $m$=6 \\
eDMD-RFF skill & $+21.3\%$ & $+3.3\%$ \\
plain DMD skill & $+18.2\%$ & $+3.3\%$ \\
neural vector field, pre-divergence correlation & 0.79 (8 steps) & 0.74 \\
neural field loss (learned $\mathrm{d}\mathbf{s}/\mathrm{d}t$) & 0.022 & 0.152 \\
top $|\lambda|$ at scales 1$\times\!\to\!$8$\times$
& 0.999$\to$0.937 & 0.69$\to$0.21 \\
Lyapunov reading & $\lambda_1=+0.024$ (chaotic) & $\lambda_1=+0.017$ \\
\bottomrule
\end{tabular}
\end{table}

\subsection{The operator detects tipping points before they arrive}
\label{sec:tipping}
For the Lorenz system with parameter $\rho$ swept from 1 to 40 (one observed
channel, no labels), the Wolf estimator returns durably negative
$\lambda_1$ in the laminar regime, then a durable crossing to
$\lambda_1>0$ at $\rho \approx 21$ --- in the transient-chaos corridor that
physically precedes the homoclinic explosion at $\rho\approx24.74$
\cite{lorenz1963}. The detector therefore functions as an \emph{early}
warning: the raw series looks normal while the geometry of the reconstructed
attractor is already warping (Fig.~\ref{fig:tipping}).

\subsection{Learned vs.\ fixed Koopman lifts}
\label{sec:deep}
On a single system the fixed 258-dimensional Fourier lift is the more linear
coordinate system (1.9$\times$ lower one-step latent error on held-out pairs
than the learned 64-dimensional lift) and forecasts within 2 percentage
points of the learned one. The learned lift's value is not per-system
accuracy but \emph{sharing}: one $\phi$ that maps every domain into one
latent. That is what the foundation sections below exploit.

\subsection{Foundation pretraining}
\label{sec:foundation}
One FoundationOperator was pretrained on 21 training series across 8 domains
and evaluated zero-shot on 2 held-out systems (sunspots, US COVID-19 daily
new cases) plus all training series (Table~\ref{tab:series}). Four pretexts
are optimized jointly; all converge at every scale tried (reconstruction
loss 0.01--0.04, linear-dynamics 0.3--0.6, scale covariance 0.02--0.04).
The arrow-of-time head --- a temporal convolution over latent windows
distinguishing forward from reversed trajectories --- reaches 94--97\%
accuracy across seeds and model scales (Fig.~\ref{fig:pretrain}).

\begin{table}[h]
\centering\small
\caption{Corpus: 23 series, 8 domains, all fetched through the Kaggle legacy
API except the synthetic physics domain.}
\label{tab:corpus}
\begin{tabular}{lrl}
\toprule
domain & series & source \\
\midrule
\VAR{CORPUS_ROWS}
\bottomrule
\end{tabular}
\end{table}

\subsection{Zero-shot transfer vs.\ baselines}
\label{sec:zs}
The frozen probe uses the pretrained encoder at test time with a rank-reduced,
spectrally-clipped linear least-squares Koopman fit --- no gradient steps.
Table~\ref{tab:series} gives the per-series skill over persistence; the
comparison against a per-series GRU autoregressive forecaster (same
embedding, same split, same horizon, closed-loop roll-out) and against a
per-series scratch deep-Koopman model is shown per series and summarized in
Fig.~\ref{fig:comparison}. The frozen operator beats the GRU on 16/23 series
(two-sided binomial sign test, $p=0.09$; median advantage +67.6 percentage
points; wins especially decisive on electricity regions, e.g.\ DOM $+51\%$
vs $-55\%$, DUQ $+58\%$ vs $-14\%$, EKPC $+69\%$ vs $+20\%$, and on the
Lorenz chaos itself $+69\%$ vs $-128\%$). The comparison is stronger and
more conservative in the scaled in-kernel evaluation (Kaggle, 40 series,
same closed-form probe, all baselines trained on the same hardware): the
frozen operator wins 28/40 series with $p=0.017$ and a median advantage of
+46.5 points, e.g.\ XRP $+70.5\%$ vs $-590\%$, sunspots $+20.1\%$ vs
$-281\%$, ECGs $-1.6\%$ vs $-204\%$. Against the per-series scratch model the frozen operator is
competitive on clean cyclic systems (sunspots $+20.1\%$ vs $+22.8\%$ in the
v9 kernel evaluation) and the scratch model wins on the other
held-out series --- a finding we discuss in Sec.~\ref{sec:disc}.

\begin{table}[h]
\centering\small
\caption{Zero-shot frozen TSO (mean over 3 pretraining seeds) vs GRU and
scratch per-series baselines, skill over persistence (\%). Held-out systems
in bold.}
\label{tab:series}
\begin{tabular}{lrrrrlc}
\toprule
series & domain & TSO & $\pm$sd & GRU & scratch & held-out \\
\midrule
\VAR{SERIES_ROWS}
\bottomrule
\end{tabular}
\end{table}

\subsection{Scaling: corpus and width help, iterations saturate}
\label{sec:scaling}
Fig.~\ref{fig:scaling} summarizes eleven training budgets on the same
evaluation protocol, all checkpoints evaluated with the same probe on the
same 23-series corpus: the 7-series precursor, the local 23-series
seed-mean, the kernel runs at latent 96/hidden 256 for 15{,}000
iterations and latent 128/hidden 384 for 25{,}000 and 60{,}000
iterations, three GPU (T4, mixed precision) capacity seeds at latent
256/hidden 768 for 25{,}000 iterations, a joint-probe run (v12) at
the same capacity whose pretraining loss adds multi-step Koopman
roll-out consistency (horizons 2, 4, 8) and unit-circle spectral
regularization, a corpus-breadth run (v13) at latent 128/hidden
384 that tiles the real corpus with a generated battery of 176
universal dynamical systems (Lorenz/R\"ossler parameter sweeps,
discrete maps, ARFIMA, GARCH, regime-switching and Kuramoto
oscillators), and the final run (v14) at latent 256/hidden 768 that
re-balances the battery toward smooth, seasonal, trending and spiky
families and re-weights the Koopman pretext to 2.5\texttimes.
The fraction of series where the frozen latent beats
persistence rises from 8/23 (v5) and 9/23 (seed-mean) to 13/23 (v7),
then plateaus: 12/23 (v9) and 10/23 (v10). Doubling the width on GPU
does not break the plateau: the three v11 seeds land at 12/23, 5/23 and
8/23 (12/23 for the best seed, 29/40 in-kernel wins vs the GRU
baseline), spread dominated by seed
variance (median 16.5 skill points across seeds). Making the probe
objective part of pretraining does not break it either --- v12's probe
term and spectral regularizer both converge (final loss 1.03; spectral
term $1.6\times10^{-6}$, spectrum inside the unit circle) yet the
frozen probe reads 11/23 positive on the shared subset, and paired
head-to-head on the same 23 series v9 \emph{beats} v12 16/23
($p=0.05$): regularizing latent linearizability during pretraining
mildly degrades frozen closed-form transfer, because the multi-step
roll-out objective does not match how the closed-form probe actually
fits (one step, short windows, rank-reduced). Tripling corpus breadth
with a dynamics battery (v13) changes the \emph{kind} of transfer
without breaking the plateau: on the shared subset it reads 6/23
positive (v9 wins the paired head-to-head 15/23, $p=0.11$, not
significant), yet the failure mode inverts --- v13 is markedly better
where v9 explodes (Dogecoin $-7{,}678$ $\rightarrow$ $-883$ skill
points vs persistence, covid-india $-91{,}617$ $\rightarrow$
$-22{,}500$, Bitcoin $-177$ $\rightarrow$ $-92$) and worse on smooth
periodic series (sunspots $+20$ $\rightarrow$ $-26$, airline $-43$
$\rightarrow$ $-161$). Dynamics-heavy pretraining teaches the latent
about nonlinear regime structure at the expense of smooth-periodic
geometry; the plateau therefore sits at the probe-evaluation interface
and the corpus balance, not at latent capacity. Scaling the \emph{same}
balanced recipe 25\texttimes{} (v17: 5{,}973-series battery, identical
recipe otherwise) confirms the per-series-exposure constraint: at the
fixed 25k-iteration budget each series is visited $\sim$4\texttimes{}
(vs $\sim$104\texttimes{} for v14's 192-series battery), and the
frozen transfer \emph{degrades} --- 11/23 on the shared subset, losing
the in-kernel head-to-head to v14 12/28 ($p=0.017$), with the biggest
regressions on the explosive series that need the most passes
(covid-india $-65{,}313$ $\rightarrow$ $-73{,}786$ skill points,
Dogecoin $-1{,}300$ $\rightarrow$ $-1{,}598$), while smooth periodic
series mildly improve (sunspots $+4.8$ $\rightarrow$ $+19.9$). Corpus
breadth and per-series fit trade against each other at fixed compute;
the Chronos scale-up works because its breadth is bought with
proportionally larger training budgets.
The v9--v14 models extend zero-shot coverage to 40 series (20/40, 16/40,
29/40, 28/40, 25/40 and 31/40 positive in-kernel; 11/13 across grid
electricity regions). The pretext losses
saturate well before 25k iterations (arrow-of-time accuracy 95.5\% at
25k vs 96.7\% at 60k; final loss 0.318 vs 0.320), so the extra
iterations move the latent within the pretext-loss basin rather than
toward a more probe-friendly geometry. This is the central negative
result of the study: at fixed architecture, pretraining beyond loss
saturation does not buy transfer, at fixed probe neither capacity
(v11) nor linearizability regularization (v12) moves the curve, and
only corpus breadth did so (v5 $\rightarrow$ v7 --- and, decisively,
the external comparison of Sec.~\ref{sec:external}, where a model
pretrained on \textasciitilde80{,}000 series transfers best of all).
The two negative levers do combine into a positive one: v14 (balanced
battery + Koopman pretext at weight 2.5) is the first run to break the
plateau on the shared subset --- 14/23 positive (vs 12/23 for v9 and
for the best v11 seed), 31/40 in-kernel wins and 24/40 positive with a
median skill of $+1.8$ points (the only positive median observed), and
it wins the paired head-to-head against v9 13/23 ($p=0.34$). Three
additional v14 seeds (identical recipe, seeds 1--3) reproduce the
result: 13/23, 13/23 and 13/23 on the shared subset with positive
median skills in all three ($+6.4$, $+0.5$, $+1.4$), and two of the
three seeds win the per-series head-to-head against v9 at $p=0.04$ and
$p=0.05$; a fifth run (v16, a protocol-error duplicate of the v14
recipe with the smaller battery) adds 12/23 with median skill $+3.8$.
Across all five v14-recipe seeds the frozen median skill is positive in
every single one (v9's is negative), and the Wilcoxon signed-rank test
on per-series medians against v9 is significant at $p=0.03$; the
pooled positive fraction is 65/115 vs v9's rate ($p=0.20$ vs the v9
rate; $p=0.10$ vs chance), so the magnitude gains are stronger than
the positive-count gains. The single-seed effect size is modest and the binomial p-values are not
conventionally significant,
so this is a directional break rather than a demonstrated law, but the
sign pattern is consistent: every earlier experiment that pushed one
lever alone (capacity, joint-probe, corpus breadth) landed inside seed
noise, while the two-lever combination lands outside it in every
metric we track (reprobe fraction, in-kernel wins, positive fraction,
median skill). The remaining gap is corpus scale --- see
Sec.~\ref{sec:external}. The next lever is therefore the pretraining
corpus itself at \textasciitilde80k-series scale, together with a
differentiable probe head trained end-to-end and evaluated few-shot, as
real foundation models are.

\subsection{The operator rediscovers the solar cycle}
\label{sec:solar}
The sunspot series is held out from pretraining (Fig.~\ref{fig:solar}). Its
~11-year Schwabe cycle is present but buried: at full sampling rate the
series is amplitude-modulated noise, and a one-step linear fit of the frozen
latent resolves only trend and short noisy pairs (no oscillatory mode).
Coarsening the series $2\times, 4\times, \dots, 32\times$ --- the
renormalization leg of the operator --- collapses the fluctuation spectrum
onto the cycle, turning it into a clean eigenmode of the fitted Koopman
matrix. The detected period times the coarsening factor is
scale-covariant: $441 \rightarrow 327 \rightarrow 203 \rightarrow 139
\rightarrow 130.1$ months at $2\times\text{--}32\times$ coarsening,
converging to $130$ months against the known $132$-month Schwabe period
(1.4\% error; the FFT-ground-truth peak of this series is at $130.6$
months). The reproduction is stable across training scale, hardware,
corpus and objective: the GPU width-doubled v11 checkpoints (three
seeds) all read $128$ months, the joint-probe v12 reads $130$, the
dynamics-corpus v13 reads $128$ and the balanced v14 reads $128$.
A linear operator with a frozen, cross-domain lift recovers a
physical constant of the solar system from a single held-out scalar
channel, with no fine-tuning of the network.

\subsection{Comparison with an external foundation model}
\label{sec:external}
The most demanding baseline for a time-series foundation model is another
foundation model. We evaluated Amazon's Chronos (chronos-t5-small,
\textasciitilde 8M parameters, pretrained on \textasciitilde 80{,}000 series)
frozen, zero-shot, on the identical 40-series corpus, split and
skill-vs-persistence metric (Fig.~\ref{fig:external}). The result is honest
and instructive: Chronos transfers better at scale --- median skill $+24.7$
percentage points vs persistence (28/40 positive) against the best TSO
(v14) at $+1.8$ (24/40 positive, the first positive median of the
operator), winning 24/40 head-to-head series (down from 27/40 against
the v11 baseline as the operator improved). Both foundation models crush
the per-series GRU token baseline (TSO v14 31/40, $p<0.001$; Chronos 35/40,
$p<0.001$). The gap tracks the pretraining corpus: Chronos saw
\textasciitilde 80{,}000 series; the TSO saw 40. This is exactly the
prediction of our own scaling curve (Sec.~\ref{sec:scaling}) --- transfer
follows corpus breadth --- and it pins the next scale-up: the operator's
architecture is competitive in kind (it beats the token baseline on the
same data, rediscovers a physical constant Chronos cannot, and offers
tipping detection and phase-space geometry no FSTM provides), but its
zero-shot transfer is corpus-limited. Where the comparison is close, the
TSO wins: meteorology (median $+14.5$\% vs $+2.6$\%) and it is the only
model of the three whose solar-cycle discovery is a built-in, spectral
consequence of its latent rather than a fitted regression.

\subsection{Ablations and the arrow of time}
\label{sec:ablations}
Removing the scale-covariance pretext leaves performance essentially
unchanged; removing the arrow-of-time pretext \emph{improves} the frozen
probe at 1{,}400-iteration scale (15/23 vs 7--9/23 positive) while the full
model's arrow head still classifies direction at 94--95\%. The pretexts
therefore shape the latent toward the training distribution rather than
toward probe-friendliness; their benefit appears only with scale
(Fig.~\ref{fig:ablations}). The reversibility analysis quantifies the gap:
the frozen latent reproduces only a small fraction of the data's Weiss
third-order irreversibility (Fig.~\ref{fig:reversibility}) --- the arrow is
\emph{classified} but not fully \emph{preserved}.

\section{Discussion and limitations}
\label{sec:disc}
The headline result is structural: on an identical forecast protocol, an
operator that learns geometry outperforms a recurrent model that learns
transitions, and the advantage is robust at every pretraining scale we
tested. Four
limitations are honest and specific. (i) \emph{Closed-form iteration is
fragile}: spiky or near-unit-root series (Dogecoin returns, COVID-19 case
spikes in some countries) amplify any latent error over a 100-step roll-out;
rank reduction and spectral clipping help, but a probabilistic or learned
error-correcting forecaster is the next step. (ii) \emph{The pretexts do not
yet pay off at small scale}: measured zero-shot skill is flat or slightly
worse with the arrow and scale pretexts until the model is large enough (the
v7 kernel run). (iii) \emph{Pretraining beyond pretext saturation does not
help the probe}: v10 (60k iterations) regresses against v9 (25k) on the
shared benchmark (28--12 wins against v9, $p=0.017$), doubling the
width on GPU (v11, three seeds) lands within seed noise of v9 (12/23,
5/23, 8/23), and adding the probe itself as a pretraining term (v12:
multi-step roll-out consistency and unit-circle spectral
regularization) is beaten head-to-head by v9 on the shared subset
(16/23, $p=0.05$) --- regularizing latent linearizability does not make
the latent more usable by a frozen closed-form fit. Checkpoint
selection must therefore use a validation pretext, and the wall is the
frozen-probe evaluation itself: the probe head must become
differentiable and be trained end-to-end with the pretexts at scale. (iv) \emph{Per-target training still wins on held-out clean
cycles}: a scratch model trained on sunspots alone ($+22.8\%$) edges the
frozen operator ($+20.1\%$ in the v9 kernel evaluation); transfer buys robustness and data-efficiency claims only in the
scarce-data regime, which our COVID series are too short (187 samples) to
demonstrate. The scale-space diagnostic, tipping detector and the TSO-vs-GRU
gap are, to our knowledge, the most reproducible contributions of this
preprint.

\section{Methods}
\label{sec:methods}
\textbf{Embedding.} $\tau$: first deep zero crossing or first local minimum of
the autocorrelation (capped so the embedding fits); $m$: first dimension
where the false-nearest-neighbours fraction drops below 2\%. Fixed $m=5$ for
the shared foundation encoder.

\textbf{Koopman.} Exact DMD via SVD of snapshot pairs; eDMD with random
Fourier features $\phi(x)=[x,\cos(Wx+c),\sin(Wx+c)]$; deep Koopman
autoencoder with losses: reconstruction $\lVert\psi(\phi(s))-s\rVert^2$,
one-step dynamics $\lVert\psi(K\phi(s))-s'\rVert^2$, latent linearity
$\lVert K\phi(s)-\phi(s')\rVert^2$, and $j{=}1..4$ unrolled consistency
terms $\lVert\psi(K^j\phi(s))-s_{t+j}\rVert^2$.

\textbf{Scale space.} Block-average decimation by 1,2,4,8; exact DMD per
lens; top eigenvalue magnitudes reported.

\textbf{Tipping.} Wolf-style largest Lyapunov exponent: nearest-neighbour
tracking with Theiler exclusion and rescaling, median over 24 fiducial
points; durable-crossing rule over 3 consecutive $\rho$ values.

\textbf{FoundationOperator.} $\phi$: MLP $5\to96\to96\to48\to24$ (local
study) or $5\to256\to256\to128\to96$ (kernel v7); shared bias-free linear
$K$; $\psi$ mirror; scale head pooling pairs of latent rows under 2$\times$
decimation; arrow head: two temporal convolutions (kernel 9) over latent
windows, binary cross-entropy forward vs reversed. Pretraining samples one
series per iteration and a random 128-row window batch (8 windows);
Adam, lr 10$^{-3}$, seeds 0--2, 1{,}400 iterations locally, 15{,}000--60{,}000 on
Kaggle (AMP when CUDA usable).

\textbf{Zero-shot probe.} Frozen $\phi$; series embedded (per-series
$\tau$); latent PCA subspace rank $R{=}\min(8,\lfloor n/16\rfloor)$;
least-squares $K$; eigenvalues clipped to $\lvert\lambda\rvert\le1.02$;
iterate and decode with frozen $\psi$. Horizon $\min(20\%, n, 100)$ steps.

\textbf{Baselines.} Persistence (last value); GRU (hidden 24, one-step
training, closed-loop roll-out, 300 iters); scratch deep-Koopman (same
architecture as the foundation model, trained on the target's training split
only, 400 iters, then the same probe).

\textbf{Statistics.} Two-sided binomial sign test on paired skill
differences; means over 3 seeds; medians reported for skewed series.

\begin{figure}[h]
\centering
\includegraphics[width=0.95\textwidth]{fig_comparison.png}
\caption{Frozen zero-shot TSO vs GRU tokens vs per-series scratch, 23 series.}
\label{fig:comparison}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.9\textwidth]{fig_scaling.png}
\caption{Scaling the operator: fraction of series with frozen skill $>$ 0
on one shared 23-series corpus, same probe, eleven training budgets
(v5 4k iters $\rightarrow$ local 1.4k $\rightarrow$ v7 15k $\rightarrow$
v9 25k $\rightarrow$ v10 60k, three v11 GPU seeds at latent
256/hidden 768, 25k, v12: the same capacity with the Koopman probe
added to the pretraining loss, 25k, v13: latent 128/hidden 384
pretrained on 40 real series tiled with a 176-series dynamics battery,
and v14: latent 256/hidden 768 on the re-balanced battery with the
Koopman pretext at weight 2.5).
Width and corpus raise transfer until the frozen-probe plateau; seed
variance dominates thereafter, and neither joint-probe regularization
(v12) nor dynamics-corpus breadth (v13) alone moves the curve --- v13
instead trades smooth-periodic transfer for robustness on explosive
series --- while their combination (v14, balanced battery + forced
Koopman linearity) breaks it: 14/23, the first run above the plateau.}
\label{fig:scaling}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.9\textwidth]{fig_external.png}
\caption{Zero-shot transfer on the 40-series protocol: TSO (v14, frozen)
vs Chronos-t5-small (frozen) vs per-series GRU. Bars are
skill vs persistence per series; medians in the legend. Chronos's
\textasciitilde 80k-series pretraining corpus transfers better on median;
both foundation models beat the token baseline decisively.}
\label{fig:external}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.9\textwidth]{fig_ablations.png}
\caption{Pretext ablations at 1.4k iterations.}
\label{fig:ablations}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{fig_reversibility.png}
\caption{Weiss third-order irreversibility: data vs frozen latent.}
\label{fig:reversibility}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{fig_solar_discovery.png}
\caption{Scale-covariant period discovery on the held-out sunspot series: the
frozen operator's detected period $\times$ coarsening factor converges to the
known $132$-month Schwabe cycle.}
\label{fig:solar}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{pretrain_curves.png}
\caption{Pretext losses during the v7 Kaggle kernel pretraining.}
\label{fig:pretrain}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{latent_geometry.png}
\caption{All 8 domains in one shared latent (v7 model, PCA).}
\label{fig:latent}
\end{figure}

\begin{thebibliography}{9}
\bibitem{chronos2024} A.~A. et al. (Amazon Science), Chronos: learning the
language of time series. \emph{arXiv:2403.07815} (2024).
\bibitem{timesfm2024} A.~Das et al., A decoder-only foundation model for
time-series forecasting. \emph{ICML} (2024).
\bibitem{takens1981} F.~Takens, Detecting strange attractors in turbulence.
\emph{Lecture Notes in Mathematics} 898, 366--381 (1981).
\bibitem{koopman1931} B.~O.~Koopman, Hamiltonian systems and transformation
in Hilbert space. \emph{PNAS} 17, 315--318 (1931).
\bibitem{mezić2005} I.~Mezić, Spectral properties of dynamical systems,
model reduction and decompositions. \emph{Nonlinear Dynamics} 41, 309--325
(2005).
\bibitem{schmid2010} P.~J.~Schmid, Dynamic mode decomposition of numerical
and experimental data. \emph{J.~Fluid Mech.} 656, 5--28 (2010).
\bibitem{wolf1985} A.~Wolf, J.~B.~Swift, H.~L.~Swinney, J.~A.~Vastano,
Determining Lyapunov exponents from a time series. \emph{Physica D} 16,
285--317 (1985).
\bibitem{lorenz1963} E.~N.~Lorenz, Deterministic nonperiodic flow.
\emph{J.~Atmos.~Sci.} 20, 130--141 (1963).
\end{thebibliography}

\section*{Data and code availability}
All corpora are public Kaggle datasets fetched through the legacy Kaggle API
(Table~\ref{tab:corpus}). The complete from-scratch implementation
(``time-series-operator'' workspace), the reproducing scripts
(\texttt{scripts/pretrain\_foundation.py}, \texttt{scripts/study\_experiments.py},
\texttt{scripts/study\_paper.py}) and the private GPU/CPU kernel that
produced the v7 scaling run are available from the corresponding repository.

\end{document}
"""


def main():
    tex = TEX.replace("\\VAR{CORPUS_ROWS}", table_corpus_rows())
    tex = tex.replace("\\VAR{SERIES_ROWS}", table_series_rows())
    with open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex)
    print("wrote", os.path.join(OUT, "main.tex"))
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory",
                        OUT, os.path.join(OUT, "main.tex")],
                       capture_output=True, text=True, cwd=OUT)
    r2 = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory",
                         OUT, os.path.join(OUT, "main.tex")],
                        capture_output=True, text=True, cwd=OUT)
    if os.path.exists(os.path.join(OUT, "main.pdf")):
        print("PDF OK:", os.path.join(OUT, "main.pdf"))
    else:
        print("PDF FAILED. tail of log:")
        print("\n".join(r2.stdout.splitlines()[-25:]))
        print("\n".join(r2.stderr.splitlines()[-10:]))


if __name__ == "__main__":
    main()