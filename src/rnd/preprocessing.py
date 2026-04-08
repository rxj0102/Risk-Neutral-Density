"""
Option chain preprocessing: OTM put-call stitching and Black-Scholes implied vol.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes helpers (private)
# ---------------------------------------------------------------------------

def _bs_price(
    S: float,
    K: float,
    r: float,
    q: float,
    tau: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Generalised Black-Scholes price with continuous dividend yield."""
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    if option_type == "call":
        return S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)


def _implied_vol(
    price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    tau: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes implied volatility via Brent's method.

    Returns ``np.nan`` if the price is at or below intrinsic value or if the
    root-finding bracket cannot be established.
    """
    disc_q = np.exp(-q * tau)
    disc_r = np.exp(-r * tau)
    if option_type == "call":
        intrinsic = max(S * disc_q - K * disc_r, 0.0)
    else:
        intrinsic = max(K * disc_r - S * disc_q, 0.0)

    if price <= intrinsic + 1e-8:
        return np.nan

    try:
        return brentq(
            lambda sigma: _bs_price(S, K, r, q, tau, sigma, option_type) - price,
            1e-5,
            10.0,
            xtol=1e-9,
            maxiter=200,
        )
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_options(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    F: float,
) -> pd.DataFrame:
    """Stitch OTM puts (K < F) and OTM calls (K ≥ F) for maximum liquidity.

    Using out-of-the-money options on both sides of the forward avoids the
    wide bid-ask spreads typical of deep in-the-money contracts.

    Parameters
    ----------
    calls : pd.DataFrame
        Pre-filtered calls from ``load_chain``.
    puts : pd.DataFrame
        Pre-filtered puts from ``load_chain``.
    F : float
        Forward price used to split the strike axis.

    Returns
    -------
    pd.DataFrame
        Combined OTM slice sorted by strike with columns
        ``['strike', 'mid', 'rel_spread', 'openInterest', 'type']``.
    """

    def _enrich(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
        return df[["strike", "mid", "rel_spread", "openInterest"]]

    calls_df = _enrich(calls)
    puts_df = _enrich(puts)

    otm_calls = calls_df[calls_df["strike"] >= F].copy()
    otm_calls["type"] = "call"
    otm_puts = puts_df[puts_df["strike"] < F].copy()
    otm_puts["type"] = "put"

    return (
        pd.concat([otm_puts, otm_calls])
        .sort_values("strike")
        .reset_index(drop=True)
    )


def compute_option_ivs(
    options: pd.DataFrame,
    S0: float,
    r: float,
    q: float,
    tau: float,
) -> pd.DataFrame:
    """Compute Black-Scholes implied volatilities for each OTM option.

    Parameters
    ----------
    options : pd.DataFrame
        Output of ``preprocess_options``.
    S0 : float
        Current spot price.
    r : float
        Annualised risk-free rate (decimal).
    q : float
        Annualised continuous dividend yield (decimal).
    tau : float
        Time-to-expiry in years.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``'iv'`` column; rows with invalid
        IVs (``NaN`` or non-positive) are dropped.
    """
    ivs = [
        _implied_vol(row["mid"], S0, row["strike"], r, q, tau, row["type"])
        for _, row in options.iterrows()
    ]
    result = options.copy()
    result["iv"] = ivs
    return result.dropna(subset=["iv"]).query("iv > 0").reset_index(drop=True)
