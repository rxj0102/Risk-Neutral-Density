"""
Risk metrics: tail probabilities, VaR/CVaR, distributional moments,
and the variance risk premium (VRP).
"""
import numpy as np

from .density import trapezoidal


def compute_tail_metrics(
    price_grid: np.ndarray,
    density: np.ndarray,
    S0: float,
) -> dict:
    """Compute a comprehensive set of risk metrics from a density function.

    The density is normalised internally, so it need not integrate to exactly
    one on input.

    Parameters
    ----------
    price_grid : np.ndarray
        Strike/price grid (must be sorted ascending).
    density : np.ndarray
        Probability density values on *price_grid*.
    S0 : float
        Current spot price used to compute tail-probability thresholds and
        to express VaR/CVaR as dollar losses.

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``skewness``, ``excess_kurtosis``,
        ``prob_down10``, ``prob_up10``, ``prob_crash20``,
        ``var_95``, ``cvar_95``, ``var_99``, ``cvar_99``.
        All probabilities are Q-measure; VaR/CVaR are positive dollar losses.
    """
    total = trapezoidal(price_grid, density)
    pdf = density / total

    # Build CDF via cumulative trapezoidal integration
    cdf = np.array(
        [trapezoidal(price_grid[: i + 1], pdf[: i + 1]) for i in range(len(pdf))]
    )
    cdf = np.clip(cdf, 0.0, 1.0)

    # --- Moments ---
    mean = trapezoidal(price_grid, price_grid * pdf)
    variance = trapezoidal(price_grid, (price_grid - mean) ** 2 * pdf)
    std = float(np.sqrt(max(variance, 0.0)))

    if std > 1e-12:
        skewness = trapezoidal(
            price_grid, ((price_grid - mean) / std) ** 3 * pdf
        )
        excess_kurtosis = (
            trapezoidal(price_grid, ((price_grid - mean) / std) ** 4 * pdf) - 3.0
        )
    else:
        skewness = excess_kurtosis = np.nan

    # --- Tail probabilities ---
    prob_down10 = float(np.interp(S0 * 0.90, price_grid, cdf))
    prob_up10 = float(1.0 - np.interp(S0 * 1.10, price_grid, cdf))
    prob_crash20 = float(np.interp(S0 * 0.80, price_grid, cdf))

    # --- VaR and CVaR ---
    def _var_cvar(alpha: float) -> tuple:
        """Dollar VaR and CVaR (losses) at significance level alpha."""
        var_price = float(np.interp(alpha, cdf, price_grid))
        var_loss = S0 - var_price

        tail_mask = price_grid <= var_price
        if tail_mask.sum() < 2:
            return var_loss, np.nan

        tail_prob = trapezoidal(price_grid[tail_mask], pdf[tail_mask])
        if tail_prob < 1e-12:
            return var_loss, np.nan

        cvar_price = (
            trapezoidal(price_grid[tail_mask], price_grid[tail_mask] * pdf[tail_mask])
            / tail_prob
        )
        return var_loss, S0 - cvar_price

    var_95, cvar_95 = _var_cvar(0.05)
    var_99, cvar_99 = _var_cvar(0.01)

    return {
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "prob_down10": prob_down10,
        "prob_up10": prob_up10,
        "prob_crash20": prob_crash20,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "var_99": var_99,
        "cvar_99": cvar_99,
    }


def vrp_metrics(impl_vol: float, hist_vol: float) -> dict:
    """Compute the variance risk premium (VRP).

    The VRP is the compensation demanded by variance sellers; it is positive
    when implied volatility exceeds realised volatility.

    Parameters
    ----------
    impl_vol : float
        ATM implied volatility (decimal, annualised).
    hist_vol : float
        Historical/realised volatility (decimal, annualised).

    Returns
    -------
    dict
        ``vrp_additive`` : implied − realised vol (percentage points).
        ``vrp_ratio``     : (implied / realised)² — variance ratio.
    """
    return {
        "vrp_additive": impl_vol - hist_vol,
        "vrp_ratio": (impl_vol / hist_vol) ** 2 if hist_vol > 0 else np.nan,
    }
