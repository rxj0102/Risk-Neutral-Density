"""
Unit tests for tail-risk metrics and VRP.
"""
import numpy as np
import pytest

from rnd.metrics import compute_tail_metrics, vrp_metrics


def _gaussian(K, mu, sigma):
    """Standard Gaussian density."""
    return np.exp(-((K - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


class TestComputeTailMetrics:
    def test_gaussian_mean(self):
        K = np.linspace(50, 150, 8000)
        metrics = compute_tail_metrics(K, _gaussian(K, 100, 10), S0=100.0)
        assert metrics["mean"] == pytest.approx(100.0, rel=1e-3)

    def test_gaussian_std(self):
        K = np.linspace(50, 150, 8000)
        metrics = compute_tail_metrics(K, _gaussian(K, 100, 10), S0=100.0)
        assert metrics["std"] == pytest.approx(10.0, rel=1e-2)

    def test_gaussian_zero_skewness(self):
        K = np.linspace(50, 150, 8000)
        metrics = compute_tail_metrics(K, _gaussian(K, 100, 10), S0=100.0)
        assert metrics["skewness"] == pytest.approx(0.0, abs=0.02)

    def test_gaussian_zero_excess_kurtosis(self):
        K = np.linspace(50, 150, 8000)
        metrics = compute_tail_metrics(K, _gaussian(K, 100, 10), S0=100.0)
        assert metrics["excess_kurtosis"] == pytest.approx(0.0, abs=0.05)

    def test_var_positive(self):
        """95% VaR should be a positive loss for a distribution centred at S0."""
        K = np.linspace(50, 150, 5000)
        metrics = compute_tail_metrics(K, _gaussian(K, 100, 15), S0=100.0)
        assert metrics["var_95"] > 0

    def test_cvar_geq_var(self):
        """CVaR ≥ VaR (CVaR is the expected loss beyond VaR)."""
        K = np.linspace(50, 150, 5000)
        metrics = compute_tail_metrics(K, _gaussian(K, 100, 15), S0=100.0)
        assert metrics["cvar_95"] >= metrics["var_95"] - 1e-8
        assert metrics["cvar_99"] >= metrics["var_99"] - 1e-8

    def test_var_99_geq_var_95(self):
        """99% VaR should be at least as large as 95% VaR."""
        K = np.linspace(50, 150, 5000)
        metrics = compute_tail_metrics(K, _gaussian(K, 100, 15), S0=100.0)
        assert metrics["var_99"] >= metrics["var_95"] - 1e-8

    def test_tail_probs_in_unit_interval(self):
        K = np.linspace(50, 150, 5000)
        metrics = compute_tail_metrics(K, _gaussian(K, 100, 10), S0=100.0)
        for key in ("prob_down10", "prob_up10", "prob_crash20"):
            assert 0.0 <= metrics[key] <= 1.0

    def test_unnormalised_input_ok(self):
        """The function should normalise the input density internally."""
        K = np.linspace(50, 150, 5000)
        pdf = _gaussian(K, 100, 10)
        m1 = compute_tail_metrics(K, pdf, S0=100.0)
        m2 = compute_tail_metrics(K, pdf * 5.0, S0=100.0)  # scaled up
        assert m1["mean"] == pytest.approx(m2["mean"], rel=1e-6)


class TestVRPMetrics:
    def test_additive_vrp(self):
        result = vrp_metrics(0.25, 0.20)
        assert result["vrp_additive"] == pytest.approx(0.05, rel=1e-10)

    def test_ratio_vrp(self):
        result = vrp_metrics(0.25, 0.20)
        assert result["vrp_ratio"] == pytest.approx((0.25 / 0.20) ** 2, rel=1e-10)

    def test_zero_hist_vol_returns_nan(self):
        result = vrp_metrics(0.20, 0.0)
        assert np.isnan(result["vrp_ratio"])
