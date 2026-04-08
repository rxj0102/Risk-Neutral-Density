"""
Parametric volatility surface models: SVI and SSVI.

References
----------
Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility
    parameterization with application to the valuation of volatility
    derivatives. Global Derivatives & Risk Management, Madrid.

Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
    Quantitative Finance, 14(1), 59–71.
"""
import warnings

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# SVI — Stochastic Volatility Inspired
# ---------------------------------------------------------------------------

def svi_total_var(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """SVI total-variance parametrization.

    w(k) = a + b · [ρ(k − m) + √((k − m)² + σ²)]

    Parameters
    ----------
    k : array-like
        Log-moneyness k = log(K/F).
    a : float
        Vertical translation; controls the overall level of variance.
    b : float
        Slope of the wings.
    rho : float
        Correlation-like skew parameter (|ρ| < 1).
    m : float
        Horizontal translation; shifts the minimum.
    sigma : float
        Smoothness of the vertex (σ > 0).

    Returns
    -------
    np.ndarray
        Total variance w = σ²_BS · τ ≥ 0.
    """
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def calibrate_svi(
    strikes: np.ndarray,
    ivs: np.ndarray,
    F: float,
    tau: float,
) -> np.ndarray:
    """Calibrate SVI parameters to market implied volatilities.

    Minimises the sum-of-squared errors in total-variance space, which is
    more numerically stable than fitting directly in IV space.

    Constraints
    -----------
    * a ≥ 0, b > 0               — non-negative total variance
    * |ρ| < 1                     — valid correlation
    * σ ≥ 0.05                    — prevents degenerate V-shape density artefacts

    Parameters
    ----------
    strikes : array-like
        Market strikes.
    ivs : array-like
        Market implied vols (decimal, annualised).
    F : float
        Forward price.
    tau : float
        Time-to-expiry in years.

    Returns
    -------
    np.ndarray
        Optimal SVI parameters ``[a, b, rho, m, sigma]``.
    """
    k = np.log(np.asarray(strikes, dtype=float) / F)
    w_mkt = np.asarray(ivs, dtype=float) ** 2 * tau

    atm_w = float(np.interp(0.0, np.sort(k), w_mkt[np.argsort(k)]))

    def _objective(params: np.ndarray) -> float:
        return float(np.sum((svi_total_var(k, *params) - w_mkt) ** 2))

    x0 = [atm_w * 0.5, 0.1, -0.3, 0.0, 0.15]
    bounds = [
        (1e-8, None),       # a ≥ 0
        (1e-8, None),       # b > 0
        (-0.999, 0.999),    # |ρ| < 1
        (-2.0, 2.0),        # m
        (0.05, 2.0),        # σ ≥ 0.05
    ]

    result = minimize(
        _objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-14, "gtol": 1e-12, "maxiter": 20_000},
    )
    if not result.success:
        warnings.warn(f"SVI calibration did not converge: {result.message}", RuntimeWarning)
    return result.x


# ---------------------------------------------------------------------------
# SSVI — Surface SVI (arbitrage-free)
# ---------------------------------------------------------------------------

def ssvi_total_var(
    k: np.ndarray,
    theta: float,
    phi: float,
    rho: float,
) -> np.ndarray:
    """SSVI total-variance parametrization (Gatheral & Jacquier, 2014).

    w(k) = (θ/2) · [1 + ρφk + √((φk + ρ)² + 1 − ρ²)]

    Note: w(0) = θ by construction, so θ equals the ATM total variance.

    Parameters
    ----------
    k : array-like
        Log-moneyness k = log(K/F).
    theta : float
        ATM total variance θ = σ²_ATM · τ.
    phi : float
        Curvature / skew parameter (φ > 0).
    rho : float
        Correlation parameter (|ρ| < 1).

    Returns
    -------
    np.ndarray
        Total variance w = σ²_BS · τ ≥ 0.
    """
    k = np.asarray(k, dtype=float)
    return (theta / 2) * (
        1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2)
    )


def calibrate_ssvi(
    strikes: np.ndarray,
    ivs: np.ndarray,
    F: float,
    tau: float,
) -> np.ndarray:
    """Calibrate SSVI parameters to market implied volatilities.

    θ (ATM total variance) is pinned from market data via interpolation.
    φ and ρ are jointly optimised subject to the no-butterfly-arbitrage
    constraint φ · θ ≤ 4 (Gatheral & Jacquier, 2014, Lemma 4.2).

    Parameters
    ----------
    strikes : array-like
        Market strikes.
    ivs : array-like
        Market implied vols (decimal, annualised).
    F : float
        Forward price.
    tau : float
        Time-to-expiry in years.

    Returns
    -------
    np.ndarray
        Optimal SSVI parameters ``[theta, phi, rho]``.
    """
    k = np.log(np.asarray(strikes, dtype=float) / F)
    w_mkt = np.asarray(ivs, dtype=float) ** 2 * tau

    k_sorted = np.sort(k)
    w_sorted = w_mkt[np.argsort(k)]
    theta = float(np.interp(0.0, k_sorted, w_sorted))

    # No-butterfly upper bound: φ·θ ≤ 4  (conservative; tighter if |ρ| large)
    phi_max = min(50.0, 4.0 / theta)

    def _objective(params: np.ndarray) -> float:
        phi, rho = params
        return float(np.sum((ssvi_total_var(k, theta, phi, rho) - w_mkt) ** 2))

    x0 = [min(1.0, phi_max * 0.5), -0.3]
    bounds = [(1e-8, phi_max), (-0.999, 0.999)]

    result = minimize(
        _objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-14, "gtol": 1e-12, "maxiter": 20_000},
    )
    if not result.success:
        warnings.warn(f"SSVI calibration did not converge: {result.message}", RuntimeWarning)

    phi, rho = result.x
    return np.array([theta, phi, rho])
