"""
Standardised plotting helpers for RND analysis.

Each function returns a ``matplotlib.figure.Figure`` so callers can save,
display inline, or compose further without side effects.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Consistent colour palette across all plots
COLORS = {
    "bl":   "#1f77b4",   # blue
    "svi":  "#ff7f0e",   # orange
    "ssvi": "#2ca02c",   # green
    "hist": "#d62728",   # red
    "atm":  "#7f7f7f",   # grey
    "call": "#1f77b4",
    "put":  "#ff7f0e",
}


def plot_iv_smile(options_df, S0: float, F: float, title: str = "Implied Volatility Smile"):
    """Scatter plot of the OTM put-call IV smile.

    Parameters
    ----------
    options_df : pd.DataFrame
        Output of ``compute_option_ivs`` with a ``'type'`` column.
    S0 : float
        Spot price (draws vertical reference line).
    F : float
        Forward price (draws vertical reference line).
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    calls = options_df[options_df["type"] == "call"]
    puts  = options_df[options_df["type"] == "put"]

    ax.scatter(calls["strike"], calls["iv"] * 100, c=COLORS["call"],
               s=55, zorder=3, label="OTM Calls")
    ax.scatter(puts["strike"],  puts["iv"]  * 100, c=COLORS["put"],
               s=55, zorder=3, label="OTM Puts")
    ax.axvline(S0, color=COLORS["atm"],  linestyle="--", linewidth=1,
               label=f"S₀ = {S0:.2f}")
    ax.axvline(F,  color="purple",       linestyle=":",  linewidth=1,
               label=f"F = {F:.2f}")

    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bl_fit(K_grid, rnd_bl, spline, strikes, call_equiv, S0: float, F: float, mean: float):
    """Two-panel: PCHIP call-equivalent price fit (left) and B-L density (right).

    Parameters
    ----------
    K_grid : np.ndarray
        Strike grid for the density.
    rnd_bl : np.ndarray
        Breeden-Litzenberger density.
    spline : PchipInterpolator
        Fitted PCHIP spline from ``extract_rnd_bl``.
    strikes : np.ndarray
        Market strikes used in the spline.
    call_equiv : np.ndarray
        Call-equivalent prices at *strikes*.
    S0, F, mean : float
        Reference price levels for vertical lines.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: spline fit
    ax = axes[0]
    K_fine = np.linspace(strikes.min(), strikes.max(), 500)
    ax.plot(K_fine, spline(K_fine), color=COLORS["bl"], linewidth=2,
            label="PCHIP spline")
    ax.scatter(strikes, call_equiv, color="k", s=40, zorder=5,
               label="Call-equiv. prices")
    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Call-equivalent price ($)")
    ax.set_title("Call-Equivalent Price Curve")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: density
    ax = axes[1]
    ax.plot(K_grid, rnd_bl, color=COLORS["bl"], linewidth=2, label="B-L RND")
    ax.axvline(S0,   color=COLORS["atm"], linestyle="--", linewidth=1, label=f"S₀={S0:.2f}")
    ax.axvline(F,    color="purple",      linestyle=":",  linewidth=1, label=f"F={F:.2f}")
    ax.axvline(mean, color="red",         linestyle="-.", linewidth=1, label=f"E[S_T]={mean:.2f}")
    ax.set_xlabel("Terminal price S_T ($)")
    ax.set_ylabel("Density f^Q(K)")
    ax.set_title("Breeden-Litzenberger RND")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_smile_fit(k_mkt, iv_mkt, k_fine, iv_svi, iv_ssvi=None, title: str = "Smile Fit"):
    """Market IVs vs. model fit(s) in log-moneyness space.

    Parameters
    ----------
    k_mkt : np.ndarray
        Market log-moneyness values.
    iv_mkt : np.ndarray
        Market implied vols (decimal).
    k_fine : np.ndarray
        Fine grid for model curve.
    iv_svi : np.ndarray
        SVI implied vols on *k_fine*.
    iv_ssvi : np.ndarray, optional
        SSVI implied vols on *k_fine* (overlaid if provided).
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(k_mkt, iv_mkt * 100, c="k", s=40, zorder=3, label="Market")
    ax.plot(k_fine, iv_svi * 100, color=COLORS["svi"], linewidth=2, label="SVI")
    if iv_ssvi is not None:
        ax.plot(k_fine, iv_ssvi * 100, color=COLORS["ssvi"], linewidth=2,
                linestyle="--", label="SSVI")
    ax.set_xlabel("Log-moneyness  k = log(K/F)")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_model_comparison(bl, svi, ssvi, S0: float, xlim=None):
    """Side-by-side linear and log-scale density comparison across three models.

    Parameters
    ----------
    bl, svi, ssvi : tuple[np.ndarray, np.ndarray]
        ``(K_grid, density)`` pairs for each model.
    S0 : float
        Spot price.
    xlim : tuple, optional
        x-axis limits (K_lo, K_hi).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    K_bl, rnd_bl   = bl
    K_svi, rnd_svi = svi
    K_ssvi, rnd_ssvi = ssvi

    for ax, yscale in zip(axes, ("linear", "log")):
        ax.plot(K_bl,   rnd_bl,   color=COLORS["bl"],   linewidth=2, label="Breeden-Litzenberger")
        ax.plot(K_svi,  rnd_svi,  color=COLORS["svi"],  linewidth=2, label="SVI")
        ax.plot(K_ssvi, rnd_ssvi, color=COLORS["ssvi"], linewidth=2, label="SSVI")
        ax.axvline(S0 * 0.90, color="grey", linestyle=":",  alpha=0.7, label="−10%")
        ax.axvline(S0 * 0.80, color="grey", linestyle="--", alpha=0.7, label="−20% crash")
        ax.set_yscale(yscale)
        ax.set_xlabel("Terminal price S_T ($)")
        ax.set_ylabel("Density" + (" (log)" if yscale == "log" else ""))
        ax.set_title(f"RND Model Comparison ({yscale} scale)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if xlim:
            ax.set_xlim(xlim)

    fig.tight_layout()
    return fig


def plot_q_vs_p(
    K_q, rnd_q, K_p, rnd_p, S0: float,
    metrics_q: dict, metrics_p: dict,
    vrp_add: float, vrp_ratio: float,
):
    """Three-panel Q vs. P comparison: densities, tail probabilities, VaR/CVaR.

    Parameters
    ----------
    K_q, rnd_q : np.ndarray
        Risk-neutral (Q) density.
    K_p, rnd_p : np.ndarray
        Historical (P) density.
    S0 : float
        Spot price.
    metrics_q, metrics_p : dict
        Output of ``compute_tail_metrics`` for Q and P.
    vrp_add : float
        Additive VRP (decimal).
    vrp_ratio : float
        Variance ratio.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    w = 0.35

    # Panel 1: overlaid densities
    ax = axes[0]
    ax.plot(K_q, rnd_q, color=COLORS["ssvi"], linewidth=2, label="Q (SSVI)")
    ax.plot(K_p, rnd_p, color=COLORS["hist"],  linewidth=2,
            linestyle="--", label="P (Historical)")
    ax.axvline(S0, color=COLORS["atm"], linestyle=":", alpha=0.6,
               label=f"S₀={S0:.2f}")
    ax.set_xlabel("Terminal price ($)")
    ax.set_ylabel("Density")
    ax.set_title("Risk-Neutral Q vs. Historical P")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: tail probabilities
    ax = axes[1]
    labels = ["−10%", "+10%", "−20% crash"]
    q_vals = [metrics_q["prob_down10"]  * 100,
              metrics_q["prob_up10"]    * 100,
              metrics_q["prob_crash20"] * 100]
    p_vals = [metrics_p["prob_down10"]  * 100,
              metrics_p["prob_up10"]    * 100,
              metrics_p["prob_crash20"] * 100]
    x = np.arange(len(labels))
    ax.bar(x - w / 2, q_vals, w, label="Q", color=COLORS["ssvi"], alpha=0.85)
    ax.bar(x + w / 2, p_vals, w, label="P", color=COLORS["hist"],  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Tail Probabilities: Q vs. P")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: VaR / CVaR
    ax = axes[2]
    risk_labels = ["VaR 95%", "CVaR 95%", "VaR 99%", "CVaR 99%"]
    q_risk = [metrics_q["var_95"], metrics_q["cvar_95"],
              metrics_q["var_99"], metrics_q["cvar_99"]]
    p_risk = [metrics_p["var_95"], metrics_p["cvar_95"],
              metrics_p["var_99"], metrics_p["cvar_99"]]
    x = np.arange(len(risk_labels))
    bars_q = ax.bar(x - w / 2, q_risk, w, label="Q", color=COLORS["ssvi"], alpha=0.85)
    bars_p = ax.bar(x + w / 2, p_risk, w, label="P", color=COLORS["hist"],  alpha=0.85)
    for bar in list(bars_q) + list(bars_p):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"${h:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(risk_labels, rotation=15)
    ax.set_ylabel("Dollar loss ($)")
    ax.set_title(f"VaR / CVaR: Q vs. P\nVRP = {vrp_add*100:+.2f}% ({vrp_ratio:.2f}×)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig
