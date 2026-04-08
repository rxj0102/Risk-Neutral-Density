"""
Data acquisition: fetch option chain and market rates from yfinance.
"""
import datetime
import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def select_expiry(ticker: yf.Ticker, target_dte: int = 30) -> str:
    """Return the expiry date string closest to *target_dte* calendar days.

    Parameters
    ----------
    ticker : yf.Ticker
        Instantiated yfinance Ticker object.
    target_dte : int
        Target days-to-expiry (default 30).

    Returns
    -------
    str
        ISO-format expiry date (``'YYYY-MM-DD'``).

    Raises
    ------
    ValueError
        If no options data is available for the ticker.
    """
    today = datetime.date.today()
    expirations = ticker.options
    if not expirations:
        raise ValueError(f"No options data available for {ticker.ticker}")
    return min(
        expirations,
        key=lambda e: abs((datetime.date.fromisoformat(e) - today).days - target_dte),
    )


def compute_tau(expiry: str) -> float:
    """Compute time-to-expiry in years (calendar days / 365).

    Parameters
    ----------
    expiry : str
        ISO-format expiry date (``'YYYY-MM-DD'``).

    Returns
    -------
    float
        Time to expiry τ in years.
    """
    today = datetime.date.today()
    dte = (datetime.date.fromisoformat(expiry) - today).days
    return dte / 365.0


def fetch_risk_free_rate(default: float = 0.04) -> float:
    """Fetch the annualised 3-month T-bill rate from Yahoo Finance (``^IRX``).

    Falls back to *default* on any network or parsing error.

    Parameters
    ----------
    default : float
        Fallback rate (default 4 %).

    Returns
    -------
    float
        Annualised risk-free rate as a decimal.
    """
    try:
        irx = yf.Ticker("^IRX")
        rate_pct = irx.fast_info["lastPrice"]
        return float(rate_pct) / 100.0
    except Exception:
        logger.warning("Could not fetch ^IRX; using default r=%.2f%%", default * 100)
        return default


def fetch_dividend_yield(ticker: yf.Ticker, default: float = 0.0) -> float:
    """Extract the annualised continuous dividend yield from ticker info.

    Handles the yfinance quirk where the yield is sometimes returned as a
    percentage integer (e.g. ``42`` instead of ``0.0042``).

    Parameters
    ----------
    ticker : yf.Ticker
        Instantiated yfinance Ticker object.
    default : float
        Fallback yield (default 0).

    Returns
    -------
    float
        Annualised dividend yield as a decimal.
    """
    try:
        info = ticker.info
        q = float(info.get("dividendYield", default) or default)
        if q > 1:  # yfinance bug: value returned as integer percentage
            q /= 100.0
        return q
    except Exception:
        logger.warning(
            "Could not fetch dividend yield; using default q=%.4f", default
        )
        return default


def load_chain(
    ticker: yf.Ticker,
    expiry: str,
    min_oi: int = 100,
    max_spread: float = 0.25,
) -> tuple:
    """Fetch and liquidity-filter the option chain for a given expiry.

    Drops contracts with ``openInterest < min_oi`` or relative bid-ask
    spread ``(ask − bid) / mid > max_spread``.

    Parameters
    ----------
    ticker : yf.Ticker
        Instantiated yfinance Ticker object.
    expiry : str
        ISO-format expiry date.
    min_oi : int
        Minimum open interest (default 100).
    max_spread : float
        Maximum relative bid-ask spread (default 0.25).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Filtered ``(calls, puts)`` DataFrames.
    """
    chain = ticker.option_chain(expiry)
    calls, puts = chain.calls.copy(), chain.puts.copy()

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        mid = (df["bid"] + df["ask"]) / 2
        rel_spread = (df["ask"] - df["bid"]) / mid.replace(0, np.nan)
        mask = (df["openInterest"] >= min_oi) & (rel_spread <= max_spread)
        return df[mask].reset_index(drop=True)

    return _filter(calls), _filter(puts)
