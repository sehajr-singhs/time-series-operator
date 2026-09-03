"""Artwork for the TSO: phase-space geometry, Koopman spectra, forecasts.

Every figure is generated from the model's own outputs — the attractor IS the
art. Dark theme, glowing trajectories, high DPI.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BG = "#0a0d16"
FG = "#e8ecf4"
ACCENT = "#38f2c8"
ORANGE = "#ff9f45"
MAGENTA = "#e056fd"
BLUE = "#4aa8ff"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": FG,
    "axes.edgecolor": "#3a4154",
    "axes.labelcolor": FG,
    "xtick.color": "#8b93a7",
    "ytick.color": "#8b93a7",
    "grid.color": "#1c2233",
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.6,
})


def _trajectory_colors(n, cmap="turbo"):
    cm = plt.get_cmap(cmap)
    return np.array([cm(i / max(n - 1, 1)) for i in range(n)])


# --------------------------------------------------------------------------
# Individual pieces
# --------------------------------------------------------------------------

def plot_butterfly(ys, path, title="The Lorenz Butterfly — a chaotic attractor in phase space",
                   downsample=1, alpha=0.85):
    """The ground-truth 3D attractor, colored by time (the 'physics reality')."""
    y = ys[::downsample]
    n = len(y)
    fig = plt.figure(figsize=(8, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)
    colors = _trajectory_colors(n)
    for i in range(0, n - 1, max(1, n // 4000)):
        ax.plot(y[i:i + 2, 0], y[i:i + 2, 1], y[i:i + 2, 2],
                color=colors[i], alpha=alpha, linewidth=1.2)
    ax.set_title(title, fontsize=12, color=FG, pad=4)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#222838"); ax.yaxis.pane.set_edgecolor("#222838")
    ax.zaxis.pane.set_edgecolor("#222838")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction(true3d, embedded, delay, path,
                        title="Takens reconstruction: one sensor channel re-creates the butterfly"):
    """Left: ground truth (x,y,z). Right: the manifold rebuilt from x(t) alone."""
    fig = plt.figure(figsize=(13, 6), dpi=160)
    n = min(len(true3d), len(embedded))
    t3 = true3d[:n]; emb = embedded[:n]
    colors = _trajectory_colors(n)
    for i, (ax, data, lbls) in enumerate([
        (fig.add_subplot(121, projection="3d", facecolor=BG), t3, ("x", "y", "z")),
        (fig.add_subplot(122, projection="3d", facecolor=BG),
         emb, (f"x(t)", f"x(t-{delay})", f"x(t-{2 * delay})")),
    ]):
        step = max(1, n // 4000)
        for i2 in range(0, n - 1, step):
            ax.plot(data[i2:i2 + 2, 0], data[i2:i2 + 2, 1], data[i2:i2 + 2, 2],
                    color=colors[i2], alpha=0.8, linewidth=1.1)
        ax.set_title("Ground truth (x, y, z)" if i == 0 else "Rebuilt from one channel", fontsize=10)
        ax.set_xlabel(lbls[0]); ax.set_ylabel(lbls[1]); ax.set_zlabel(lbls[2])
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.grid(True)
    fig.suptitle(title, fontsize=12, color=FG)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_koopman_spectrum(model, path,
                          title="The Koopman spectrum: chaos unwrapped into rotations and decays"):
    """Eigenvalues of the linearized operator on the complex plane.

    |mu| ~ 1 -> neutral rotation (the persistent oscillation modes).
    |mu| < 1 -> decaying modes. The unit circle is the 'attractor skeleton'.
    """
    w = model["eigenvalues"]
    mag = np.abs(w)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(theta), np.sin(theta), color="#2c3450", linewidth=2, zorder=1)
    sc = ax.scatter(w.real, w.imag, c=mag, cmap="turbo", s=46,
                    edgecolors="white", linewidths=0.3, zorder=3)
    ax.axhline(0, color="#222838", linewidth=0.8)
    ax.axvline(0, color="#222838", linewidth=0.8)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|λ| — closeness to the neutral unit circle")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_forecast(result, path, title="Closed-form Koopman forecast vs reality"):
    """Truth vs the model's linearized forecast and the persistence baseline."""
    x = result["signal"]
    split = result["split"]
    pred = result["pred"]
    true = result["true"]
    start = split  # embedded index == signal index for column 0
    horizon = len(pred)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    t_full = np.arange(len(x))
    ax.plot(t_full[:start], x[:start], color=BLUE, alpha=0.35, linewidth=0.8,
            label="training (observed)")
    t_pred = start + np.arange(horizon)
    ax.plot(t_pred, true, color=ORANGE, linewidth=1.4, label="reality (held out)")
    ax.plot(t_pred, pred, color=ACCENT, linewidth=1.4, label="Koopman forecast")
    ax.plot(t_pred, np.full(horizon, x[start - 1]), color=MAGENTA, alpha=0.7,
            linewidth=1.0, linestyle="--", label="persistence baseline")
    ax.axvline(start, color="white", alpha=0.35, linestyle=":", linewidth=1)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("time (samples)"); ax.set_ylabel("normalized signal")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_scale_space(scale_out, path,
                     title="Universal scale space: the same physics through 1×, 2×, 4×, 8× lenses"):
    """Top Koopman eigenvalue magnitudes at each decimation scale."""
    fig, axes = plt.subplots(1, len(scale_out), figsize=(13, 3.6), dpi=160)
    for ax, (s, info) in zip(axes, sorted(scale_out.items())):
        w = info["eigenvalues"]
        ax.scatter(w.real, w.imag, s=18, color=ACCENT, alpha=0.8)
        th = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(th), np.sin(th), color="#2c3450", linewidth=1)
        ax.set_aspect("equal")
        ax.set_title(f"scale ×{s}  (n={info['n']})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# The masterpiece
# --------------------------------------------------------------------------

def plot_learned_field(true_manifold, learned_traj, delay, path,
                       title="The neural field learned the flow — its trajectory rides the same butterfly"):
    """Faint ground-truth attractor + the trajectory integrated from the learned vector field.

    The orbits drift apart (chaos) but the geometry — the shape — is reproduced.
    """
    fig = plt.figure(figsize=(9, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)
    n = len(true_manifold)
    step = max(1, n // 3000)
    for i in range(0, n - 1, step):
        ax.plot(true_manifold[i:i + 2, 0], true_manifold[i:i + 2, 1],
                true_manifold[i:i + 2, 2], color=BLUE, alpha=0.12, linewidth=0.7)
    m = len(learned_traj)
    cm = plt.get_cmap("magma")
    for i in range(0, m - 1, max(1, m // 1500)):
        ax.plot(learned_traj[i:i + 2, 0], learned_traj[i:i + 2, 1],
                learned_traj[i:i + 2, 2],
                color=cm(i / max(m - 1, 1)), alpha=0.95, linewidth=1.6)
    ax.scatter(learned_traj[0, 0], learned_traj[0, 1], learned_traj[0, 2],
               color="white", s=40, label="start")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("s1"); ax.set_ylabel("s2"); ax.set_zlabel("s3")
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_masterpiece(butterfly=None, embedded=None, delay=1, model=None,
                     result=None, scale_out=None, out_path="output/masterpiece.png"):
    """2×2 artwork: attractor, Takens reconstruction, Koopman spectrum, forecast."""
    fig = plt.figure(figsize=(16, 14), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                          hspace=0.28, wspace=0.16)

    # --- top-left: butterfly ---
    ax = fig.add_subplot(gs[0, 0], projection="3d", facecolor=BG)
    if butterfly is not None:
        n = len(butterfly)
        colors = _trajectory_colors(n)
        step = max(1, n // 3500)
        for i in range(0, n - 1, step):
            ax.plot(butterfly[i:i + 2, 0], butterfly[i:i + 2, 1],
                    butterfly[i:i + 2, 2], color=colors[i], alpha=0.8, linewidth=1.1)
    ax.set_title("1 · The attractor — reality, in phase space", fontsize=11, pad=2)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.grid(True)

    # --- top-right: Takens reconstruction ---
    ax = fig.add_subplot(gs[0, 1], projection="3d", facecolor=BG)
    if embedded is not None:
        n = len(embedded)
        colors = _trajectory_colors(n)
        step = max(1, n // 3500)
        for i in range(0, n - 1, step):
            ax.plot(embedded[i:i + 2, 0], embedded[i:i + 2, 1],
                    embedded[i:i + 2, 2], color=colors[i], alpha=0.8, linewidth=1.1)
    ax.set_title(f"2 · Takens lift — one channel rebuilds the shape (τ={delay})",
                 fontsize=11, pad=2)
    ax.set_xlabel("x(t)"); ax.set_ylabel(f"x(t-{delay})"); ax.set_zlabel(f"x(t-{2 * delay})")
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.grid(True)

    # --- bottom-left: Koopman spectrum ---
    ax = fig.add_subplot(gs[1, 0])
    if model is not None:
        w = model["eigenvalues"]
        mag = np.abs(w)
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), color="#2c3450", linewidth=1.6, zorder=1)
        ax.scatter(w.real, w.imag, c=mag, cmap="turbo", s=40, edgecolors="white",
                   linewidths=0.3, zorder=3)
    ax.set_aspect("equal")
    ax.set_title("3 · Koopman spectrum — the chaos, linearized", fontsize=11)
    ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)")

    # --- bottom-right: forecast ---
    ax = fig.add_subplot(gs[1, 1])
    if result is not None:
        x = result["signal"]; split = result["split"]
        pred = result["pred"]; true = result["true"]
        t_pred = split + np.arange(len(pred))
        ax.plot(np.arange(split), x[:split], color=BLUE, alpha=0.35, linewidth=0.8,
                label="observed")
        ax.plot(t_pred, true, color=ORANGE, linewidth=1.3, label="reality")
        ax.plot(t_pred, pred, color=ACCENT, linewidth=1.3, label="Koopman")
        ax.axvline(split, color="white", alpha=0.3, linestyle=":", linewidth=1)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title("4 · Closed-form forecast vs reality", fontsize=11)
    ax.set_xlabel("time"); ax.set_ylabel("normalized signal")

    fig.suptitle("THE TIME-SERIES OPERATOR — time as geometry",
                 fontsize=17, color=ACCENT, y=0.985)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Foundation-model artwork (deep Koopman, bifurcation, shared latent, zero-shot)
# --------------------------------------------------------------------------

def plot_deep_koopman_bench(true_vals, pred_rff, pred_dk, path, rff_skill,
                            dk_skill, title="Learned lift vs fixed RFF lift on the Lorenz butterfly"):
    """The same forecast task: truth, fixed Fourier lift, learned Koopman lift."""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    t = np.arange(len(true_vals))
    ax.plot(t, true_vals, color=ORANGE, linewidth=1.4, label="reality")
    ax.plot(t, pred_rff, color=MAGENTA, linewidth=1.2, alpha=0.9,
            label=f"fixed RFF lift  (skill {rff_skill:+.1f}%)")
    ax.plot(t, pred_dk, color=ACCENT, linewidth=1.6,
            label=f"learned Koopman lift  (skill {dk_skill:+.1f}%)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("forecast horizon (samples)"); ax.set_ylabel("normalized x")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_bifurcation(sweep, path, detected_rho, theoretical=24.74,
                     title="Tipping detector: the Lorenz attractor warps as ρ sweeps through chaos onset"):
    """Tipping score (from the Wolf Lyapunov estimator on one channel) vs ρ."""
    rho = np.asarray(sweep["rho"])
    score = np.asarray(sweep["tipping_score"])
    lam = np.asarray(sweep["lambda1"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=160, sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6], "hspace": 0.12})
    axes[0].plot(rho, lam, color=ACCENT, linewidth=1.6)
    axes[0].axhline(0, color=MAGENTA, linewidth=0.9, linestyle="--")
    axes[0].set_ylabel("largest Lyapunov λ₁")
    axes[0].set_title("phase-space geometry reads the tipping point before the raw signal does",
                      fontsize=10)
    axes[1].plot(rho, score, color=ORANGE, linewidth=1.8)
    axes[1].axvline(theoretical, color=BLUE, linewidth=1.1, linestyle=":",
                    label=f"theoretical chaos onset ρ≈{theoretical}")
    if np.isfinite(detected_rho):
        axes[1].axvline(detected_rho, color=MAGENTA, linewidth=1.6,
                        label=f"detected tipping ρ≈{detected_rho:.1f}")
    axes[1].set_xlabel("Lorenz parameter ρ")
    axes[1].set_ylabel("tipping score")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    axes[1].set_ylim(-0.05, 1.05)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_latent_geometry(points, labels, path,
                         title="One shared latent space: attractors from every domain, mapped to a common geometry"):
    """PCA of the pretrained Koopman latent, colored by domain."""
    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    cm = plt.get_cmap("turbo")
    uniq = sorted(set(labels))
    colors = {u: cm(i / max(len(uniq) - 1, 1)) for i, u in enumerate(uniq)}
    for u in uniq:
        m = labels == u
        ax.scatter(points[m, 0], points[m, 1], s=3, alpha=0.45, color=colors[u],
                   label=u)
    ax.legend(loc="best", frameon=False, fontsize=9, markerscale=3)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_zero_shot(true, pred, path, title, skill, baseline=None):
    """Zero-shot forecast of a never-seen system (frozen lift + closed-form K)."""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    t = np.arange(len(true))
    ax.plot(t, true, color=ORANGE, linewidth=1.3, label="reality (held out)")
    if baseline is not None:
        ax.plot(t, baseline, color="#8b93a7", linewidth=1.1, alpha=0.85,
                linestyle="--", label="persistence baseline")
    ax.plot(t, pred, color=ACCENT, linewidth=1.6, label="zero-shot TSO forecast")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(f"{title}  —  zero-shot skill {skill:+.1f}% over persistence", fontsize=11)
    ax.set_xlabel("forecast horizon (samples)"); ax.set_ylabel("normalized signal")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_solar_discovery(rows, path, known_months=132.0):
    """The operator renormalizes its way to the Schwabe cycle: detected
    period x coarsening vs the true ~11-year period, per scale."""
    det = [r for r in rows if r.get("period_months") is not None]
    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=160)
    cs = [r["coarsening"] for r in rows]
    ax.plot(cs, [known_months] * len(cs), color=ORANGE, linewidth=2.2,
            linestyle="--", label=f"known Schwabe cycle ({known_months:.0f} mo)")
    if det:
        ax.plot([r["coarsening"] for r in det], [r["period_months"] for r in det],
                color=ACCENT, marker="o", markersize=8, linewidth=1.8,
                label="operator detection (period x coarsening)")
        for r in det:
            ax.annotate(f"{r['period_months']:.0f} mo  (amp {r['amp']:.2f})",
                        (r["coarsening"], r["period_months"]),
                        textcoords="offset points", xytext=(6, 10),
                        fontsize=8, color=FG)
    ax.set_xscale("log", base=2)
    ax.set_xticks(cs)
    ax.set_xticklabels([str(c) for c in cs])
    ax.set_xlabel("coarsening factor (renormalization scale)")
    ax.set_ylabel("detected period  x  coarsening  (months)")
    ax.set_title("The operator rediscovers the 11-year solar cycle — from a "
                 "held-out series and a frozen lift", fontsize=11)
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_pretrain_curves(history, parts, path,
                         title="Pretraining: the four temporal pretexts, one loss"):
    """Total loss + per-pretext breakdown over pretraining iterations."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=160)
    it = np.arange(len(history))
    axes[0].plot(it, history, color=ACCENT, linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_title("total pretraining loss", fontsize=10)
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("loss")
    labels = {"recon": "reconstruction", "dyn": "linear dynamics",
              "scale": "scale covariance", "arrow": "arrow of time"}
    for k, v in parts.items():
        if k not in labels:
            continue
        axes[1].plot(np.arange(len(v))[::5], v[::5], linewidth=0.9, label=labels[k])
    axes[1].set_yscale("log")
    axes[1].set_title("pretext tasks", fontsize=10)
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("loss")
    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
