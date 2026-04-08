"""
Risk-neutral density (RND) extraction from option prices and parametric models.

Three methods are implemented:

1. **Breeden-Litzenberger** (model-free)
   Second derivative of the call price curve w.r.t. strike:
   f^Q(K) = e^{rτ} · ∂²C/∂K²
   A PCHIP spline is fitted to call-equivalent prices to ensure C(K) is
   monotone decreasing (arbitrage-free) before differentiation.

2. **SVI** (Gatheral 2004, parametric 5-parameter smile)
   Total variance w(k) = a + b[ρ(k−m) + √((k−m)²+σ²)] is fitted, then
   the RND is recovered via the Breeden-Litzenberger identity in
   log-moneyness space.

3. **SSVI** (Gatheral & Jacquier 2014, arbitrage-free surface)
   w(k) = (θ/2)[1 + ρφk + √((φk+ρ)²+1−ρ²)]. Same extraction procedure.

Mathematical note
-----------------
Working in log-moneyness k = log(K/F), the undiscounted (forward-measure)
call price is:

    C_fwd(k) = F · [N(d₁) − e^k · N(d₂)]
    d₁ = −k/√w + √w/2,  d₂ = d₁ − √w

The Breeden-Litzenberger identity transforms to k-space as:

    f_K(K) = ∂²C_fwd/∂K² = (1/K²) · (∂²C_fwd/∂k² − ∂C_fwd/∂k)

This formulation is numerically preferable because it avoids differentiating
on an unevenly-spaced K-grid.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

from .models import svi_total_var, ssvi_total_var


try:
    _np_trapz = np.trapezoid   # NumPy >= 2.0
except AttributeError:
    _np_trapz = np.trapz       # NumPy < 2.0


def trapezoidal(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal rule numerical integration of y over x."""
    return float(_np_trapz(y, x))


# ---------------------------------------------------------------------------
# Breeden-Litzenberger (model-free)
# ---------------------------------------------------------------------------

def extract_rnd_bl(
    strikes: np.ndarray,
    call_equiv_prices: np.ndarray,
    r: float,
    tau: float,
    n_grid: int = 3_000,
) -> tuple:
    """Extract the RND via the Breeden-Litzenberger formula.

    Put prices must first be converted to call-equivalent prices via
    put-call parity so the input curve is monotone decreasing in K.

    Parameters
    ----------
    strikes : array-like
        Market strikes, sorted ascending.
    call_equiv_prices : array-like
        Call-equivalent mid-prices at each strike.
    r : float
        Risk-free rate (decimal).
    tau : float
        Time-to-expiry in years.
    n_grid : int
        Number of evaluation points for the output density grid.

    Returns
    -------
    K_grid : np.ndarray
        Strike evaluation grid.
    rnd : np.ndarray
        Normalised risk-neutral density.
    spline : PchipInterpolator
        Fitted PCHIP spline (for diagnostic use).
    """
    strikes = np.asarray(strikes, dtype=float)
    call_equiv_prices = np.asarray(call_equiv_prices, dtype=float)

    spline = PchipInterpolator(strikes, call_equiv_prices)
    K_grid = np.linspace(strikes.min(), strikes.max(), n_grid)

    d2C_dK2 = spline.derivative(2)(K_grid)
    rnd = np.exp(r * tau) * np.maximum(d2C_dK2, 0.0)

    norm_const = trapezoidal(K_grid, rnd)
    if norm_const > 1e-12:
        rnd /= norm_const

    return K_grid, rnd, spline


# ---------------------------------------------------------------------------
# Shared helper for parametric models
# ---------------------------------------------------------------------------

def _forward_call_prices(
    k_grid: np.ndarray, w: np.ndarray, F: float
) -> np.ndarray:
    """Undiscounted Black-Scholes call prices on a log-moneyness grid.

    C_fwd(k) = F · [N(d₁) − e^k · N(d₂)]
    """
    sqrt_w = np.sqrt(np.maximum(w, 1e-12))
    d1 = -k_grid / sqrt_w + sqrt_w / 2
    d2 = d1 - sqrt_w
    return F * (norm.cdf(d1) - np.exp(k_grid) * norm.cdf(d2))


def _rnd_from_total_var(
    k_grid: np.ndarray,
    w: np.ndarray,
    F: float,
) -> tuple:
    """Compute the RND in strike space from a total-variance array.

    Uses the log-moneyness form of Breeden-Litzenberger:

        f_K(K) = (1/K²) · (∂²C_fwd/∂k² − ∂C_fwd/∂k)

    Derivation: dK = K·dk ⟹ f_K = f_k/K, and ∂²/∂K² = (1/K²)(∂²/∂k² − ∂/∂k).

    Parameters
    ----------
    k_grid : np.ndarray
        Uniformly-spaced log-moneyness grid.
    w : np.ndarray
        Total variance values on *k_grid*.
    F : float
        Forward price.

    Returns
    -------
    K_grid : np.ndarray
    rnd : np.ndarray
        Normalised density in strike (K) space.
    """
    K_grid = F * np.exp(k_grid)
    C_fwd = _forward_call_prices(k_grid, w, F)

    dC_dk = np.gradient(C_fwd, k_grid)
    d2C_dk2 = np.gradient(dC_dk, k_grid)

    # Breeden-Litzenberger in log-moneyness space
    rnd = np.maximum((d2C_dk2 - dC_dk) / K_grid**2, 0.0)

    norm_const = trapezoidal(K_grid, rnd)
    if norm_const > 1e-12:
        rnd /= norm_const

    return K_grid, rnd


# ---------------------------------------------------------------------------
# SVI density
# ---------------------------------------------------------------------------

def rnd_from_svi(
    params: np.ndarray,
    F: float,
    tau: float,
    k_range: tuple = (np.log(0.5), np.log(2.0)),
    n_grid: int = 3_000,
) -> tuple:
    """Compute the risk-neutral density from calibrated SVI parameters.

    Parameters
    ----------
    params : array-like
        SVI parameters ``[a, b, rho, m, sigma]`` from ``calibrate_svi``.
    F : float
        Forward price.
    tau : float
        Time-to-expiry in years.
    k_range : tuple
        ``(k_min, k_max)`` log-moneyness evaluation range.
    n_grid : int
        Grid resolution (default 3 000).

    Returns
    -------
    K_grid : np.ndarray
    rnd : np.ndarray
        Normalised RND in strike space.
    """
    a, b, rho, m, sigma = params
    k_grid = np.linspace(*k_range, n_grid)
    w = svi_total_var(k_grid, a, b, rho, m, sigma)
    return _rnd_from_total_var(k_grid, w, F)


# ---------------------------------------------------------------------------
# SSVI density
# ---------------------------------------------------------------------------

def rnd_from_ssvi(
    params: np.ndarray,
    F: float,
    tau: float,
    k_range: tuple = (np.log(0.5), np.log(2.0)),
    n_grid: int = 3_000,
) -> tuple:
    """Compute the risk-neutral density from calibrated SSVI parameters.

    Parameters
    ----------
    params : array-like
        SSVI parameters ``[theta, phi, rho]`` from ``calibrate_ssvi``.
    F : float
        Forward price.
    tau : float
        Time-to-expiry in years.
    k_range : tuple
        ``(k_min, k_max)`` log-moneyness evaluation range.
    n_grid : int
        Grid resolution (default 3 000).

    Returns
    -------
    K_grid : np.ndarray
    rnd : np.ndarray
        Normalised RND in strike space.
    """
    theta, phi, rho = params
    k_grid = np.linspace(*k_range, n_grid)
    w = ssvi_total_var(k_grid, theta, phi, rho)
    return _rnd_from_total_var(k_grid, w, F)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def clip_normalise(
    price_grid: np.ndarray,
    density: np.ndarray,
    lo: float,
    hi: float,
) -> tuple:
    """Clip a density to the interval [lo, hi] and re-normalise.

    Useful for overlaying densities with different support widths on the
    same plot without spurious visual area differences.

    Parameters
    ----------
    price_grid : np.ndarray
    density : np.ndarray
    lo, hi : float
        Clip bounds in price space.

    Returns
    -------
    K_clip : np.ndarray
    rnd_clip : np.ndarray
        Clipped and normalised density.
    """
    mask = (price_grid >= lo) & (price_grid <= hi)
    K_clip = price_grid[mask]
    d_clip = density[mask].copy()
    norm_const = trapezoidal(K_clip, d_clip)
    if norm_const > 1e-12:
        d_clip /= norm_const
    return K_clip, d_clip
