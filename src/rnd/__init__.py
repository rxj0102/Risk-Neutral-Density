"""
rnd — Risk-Neutral Density toolkit for options analysis.

Modules
-------
data          : Option chain and rate fetching (yfinance wrappers).
preprocessing : OTM put-call stitching and Black-Scholes implied vol.
models        : SVI and SSVI volatility surface parametrizations.
density       : RND extraction — Breeden-Litzenberger, SVI, SSVI.
metrics       : Tail risk metrics, VaR/CVaR, and variance risk premium.
plotting      : Standardised figure helpers.

Typical usage
-------------
>>> import yfinance as yf
>>> from rnd import (select_expiry, compute_tau, fetch_risk_free_rate,
...                  fetch_dividend_yield, load_chain, preprocess_options,
...                  compute_option_ivs, calibrate_svi, rnd_from_svi,
...                  compute_tail_metrics)
"""
# data and preprocessing require yfinance; import conditionally so that the
# pure-math modules (models, density, metrics) remain importable in environments
# where yfinance is not installed (e.g. CI / unit tests).
try:
    from .data import (
        select_expiry,
        compute_tau,
        fetch_risk_free_rate,
        fetch_dividend_yield,
        load_chain,
    )
    from .preprocessing import preprocess_options, compute_option_ivs
except ModuleNotFoundError:
    pass
from .models import svi_total_var, calibrate_svi, ssvi_total_var, calibrate_ssvi
from .density import (
    extract_rnd_bl,
    rnd_from_svi,
    rnd_from_ssvi,
    clip_normalise,
    trapezoidal,
)
from .metrics import compute_tail_metrics, vrp_metrics
from . import plotting

__all__ = [
    # data
    "select_expiry", "compute_tau", "fetch_risk_free_rate",
    "fetch_dividend_yield", "load_chain",
    # preprocessing
    "preprocess_options", "compute_option_ivs",
    # models
    "svi_total_var", "calibrate_svi", "ssvi_total_var", "calibrate_ssvi",
    # density
    "extract_rnd_bl", "rnd_from_svi", "rnd_from_ssvi",
    "clip_normalise", "trapezoidal",
    # metrics
    "compute_tail_metrics", "vrp_metrics",
    # submodule
    "plotting",
]
