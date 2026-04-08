"""
Unit tests for risk-neutral density extraction functions.
"""
import numpy as np
import pytest

from rnd.density import (
    clip_normalise,
    extract_rnd_bl,
    rnd_from_ssvi,
    rnd_from_svi,
    trapezoidal,
)
from rnd.models import svi_total_var, ssvi_total_var


# ---------------------------------------------------------------------------
# trapezoidal
# ---------------------------------------------------------------------------

class TestTrapezoidal:
    def test_constant_one(self):
        x = np.linspace(0, 1, 5000)
        assert trapezoidal(x, np.ones_like(x)) == pytest.approx(1.0, rel=1e-4)

    def test_linear_function(self):
        """∫₀¹ x dx = 0.5"""
        x = np.linspace(0, 1, 10_000)
        assert trapezoidal(x, x) == pytest.approx(0.5, rel=1e-4)


# ---------------------------------------------------------------------------
# Breeden-Litzenberger RND
# ---------------------------------------------------------------------------

class TestExtractRNDBL:
    def _call_prices(self, strikes, F, sigma, r, tau):
        """Analytical BS call prices for a flat smile."""
        from scipy.stats import norm
        k = np.log(strikes / F)
        w = sigma**2 * tau
        d1 = -k / np.sqrt(w) + np.sqrt(w) / 2
        d2 = d1 - np.sqrt(w)
        return np.exp(-r * tau) * F * (norm.cdf(d1) - np.exp(k) * norm.cdf(d2))

    def test_integrates_to_one(self):
        F, sigma, r, tau = 100.0, 0.20, 0.04, 0.25
        strikes = np.linspace(60, 160, 25)
        prices = self._call_prices(strikes, F, sigma, r, tau)
        K_grid, rnd, _ = extract_rnd_bl(strikes, prices, r, tau)
        assert trapezoidal(K_grid, rnd) == pytest.approx(1.0, abs=0.02)

    def test_mode_near_forward(self):
        """For a symmetric, near-ATM smile the density mode should be near F."""
        F, sigma, r, tau = 100.0, 0.20, 0.0, 0.25
        strikes = np.linspace(60, 160, 30)
        prices = self._call_prices(strikes, F, sigma, r, tau)
        K_grid, rnd, _ = extract_rnd_bl(strikes, prices, r, tau)
        mode_K = K_grid[np.argmax(rnd)]
        assert abs(mode_K - F) / F < 0.05


# ---------------------------------------------------------------------------
# SVI / SSVI RND
# ---------------------------------------------------------------------------

class TestRNDFromSVI:
    def test_integrates_to_one(self):
        """SVI density should integrate to approximately 1."""
        F, tau = 100.0, 0.25
        params = np.array([0.02, 0.08, -0.4, 0.0, 0.15])
        K_grid, rnd = rnd_from_svi(params, F, tau)
        assert trapezoidal(K_grid, rnd) == pytest.approx(1.0, abs=0.05)

    def test_non_negative(self):
        F, tau = 100.0, 0.25
        params = np.array([0.02, 0.08, -0.4, 0.0, 0.15])
        _, rnd = rnd_from_svi(params, F, tau)
        assert np.all(rnd >= 0)

    def test_skew_direction_with_rho(self):
        """Negative ρ produces a left-skewed smile → lower RND skewness than positive ρ."""
        from rnd.metrics import compute_tail_metrics
        F, tau = 100.0, 0.25
        params_neg = np.array([0.02, 0.08, -0.7, 0.0, 0.15])
        params_pos = np.array([0.02, 0.08,  0.7, 0.0, 0.15])
        K_grid_n, rnd_n = rnd_from_svi(params_neg, F, tau)
        K_grid_p, rnd_p = rnd_from_svi(params_pos, F, tau)
        skew_n = compute_tail_metrics(K_grid_n, rnd_n, F)["skewness"]
        skew_p = compute_tail_metrics(K_grid_p, rnd_p, F)["skewness"]
        assert skew_n < skew_p


class TestRNDFromSSVI:
    def test_integrates_to_one(self):
        F, tau = 100.0, 0.25
        params = np.array([0.04, 2.0, -0.3])
        K_grid, rnd = rnd_from_ssvi(params, F, tau)
        assert trapezoidal(K_grid, rnd) == pytest.approx(1.0, abs=0.05)

    def test_non_negative(self):
        F, tau = 100.0, 0.25
        params = np.array([0.04, 2.0, -0.3])
        _, rnd = rnd_from_ssvi(params, F, tau)
        assert np.all(rnd >= 0)


# ---------------------------------------------------------------------------
# clip_normalise
# ---------------------------------------------------------------------------

class TestClipNormalise:
    def test_bounds_respected(self):
        x = np.linspace(50, 200, 1000)
        y = np.exp(-((x - 100) ** 2) / 400)
        K_clip, _ = clip_normalise(x, y, 80, 120)
        assert K_clip.min() >= 80 - 1e-10
        assert K_clip.max() <= 120 + 1e-10

    def test_normalised(self):
        x = np.linspace(50, 200, 1000)
        y = np.exp(-((x - 100) ** 2) / 400)
        K_clip, rnd_clip = clip_normalise(x, y, 70, 140)
        assert trapezoidal(K_clip, rnd_clip) == pytest.approx(1.0, abs=1e-3)
