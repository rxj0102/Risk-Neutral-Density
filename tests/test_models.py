"""
Unit tests for SVI and SSVI volatility surface models.
"""
import numpy as np
import pytest

from rnd.models import (
    calibrate_svi,
    calibrate_ssvi,
    ssvi_total_var,
    svi_total_var,
)


# ---------------------------------------------------------------------------
# SVI
# ---------------------------------------------------------------------------

class TestSVITotalVar:
    def test_non_negative_on_grid(self):
        """Total variance must be non-negative everywhere."""
        k = np.linspace(-1.5, 1.5, 200)
        w = svi_total_var(k, a=0.02, b=0.05, rho=-0.3, m=0.0, sigma=0.15)
        assert np.all(w >= 0)

    def test_atm_value(self):
        """At k=0, w = a + b·σ (the ATM total variance)."""
        a, b, sigma = 0.04, 0.1, 0.2
        w_atm = svi_total_var(0.0, a=a, b=b, rho=-0.5, m=0.0, sigma=sigma)
        assert w_atm == pytest.approx(a + b * sigma, rel=1e-10)

    def test_symmetry_with_zero_rho_and_m(self):
        """With ρ=0, m=0, the smile is symmetric around k=0."""
        k = np.array([-0.3, -0.1, 0.1, 0.3])
        w = svi_total_var(k, a=0.02, b=0.05, rho=0.0, m=0.0, sigma=0.2)
        np.testing.assert_allclose(w[:2], w[::-1][:2], rtol=1e-10)

    def test_scalar_input(self):
        """Accepts scalar input and returns a float-like."""
        w = svi_total_var(0.0, 0.02, 0.05, -0.3, 0.0, 0.15)
        assert float(w) > 0


class TestCalibrateSVI:
    def test_roundtrip_recovery(self):
        """Calibration should closely recover synthetic market total variances."""
        F, tau = 100.0, 0.25
        true_params = np.array([0.02, 0.08, -0.4, 0.0, 0.15])
        strikes = F * np.exp(np.linspace(-0.3, 0.3, 12))
        k = np.log(strikes / F)
        w_true = svi_total_var(k, *true_params)
        ivs = np.sqrt(w_true / tau)

        fitted = calibrate_svi(strikes, ivs, F, tau)
        w_fit = svi_total_var(k, *fitted)
        np.testing.assert_allclose(w_fit, w_true, rtol=1e-3)

    def test_sigma_lower_bound(self):
        """Calibrated σ must satisfy the lower bound (σ ≥ 0.05)."""
        F, tau = 100.0, 0.25
        strikes = F * np.exp(np.linspace(-0.2, 0.2, 8))
        ivs = np.full(len(strikes), 0.20)
        params = calibrate_svi(strikes, ivs, F, tau)
        assert params[4] >= 0.05 - 1e-8  # sigma


# ---------------------------------------------------------------------------
# SSVI
# ---------------------------------------------------------------------------

class TestSSVITotalVar:
    def test_non_negative(self):
        k = np.linspace(-1.5, 1.5, 200)
        w = ssvi_total_var(k, theta=0.04, phi=2.0, rho=-0.3)
        assert np.all(w >= 0)

    def test_atm_equals_theta(self):
        """SSVI at k=0 must equal θ by construction."""
        theta = 0.05
        w_atm = ssvi_total_var(0.0, theta=theta, phi=3.0, rho=-0.4)
        assert w_atm == pytest.approx(theta, rel=1e-10)

    def test_scalar_input(self):
        w = ssvi_total_var(0.0, 0.04, 2.0, -0.3)
        assert float(w) > 0


class TestCalibrateSSVI:
    def test_roundtrip_recovery(self):
        """Calibration should closely recover synthetic SSVI market IVs."""
        F, tau = 100.0, 0.25
        theta, phi, rho = 0.04, 3.0, -0.3
        strikes = F * np.exp(np.linspace(-0.25, 0.25, 12))
        k = np.log(strikes / F)
        w_true = ssvi_total_var(k, theta, phi, rho)
        ivs = np.sqrt(w_true / tau)

        params = calibrate_ssvi(strikes, ivs, F, tau)
        w_fit = ssvi_total_var(k, *params)
        np.testing.assert_allclose(w_fit, w_true, rtol=5e-3)

    def test_butterfly_arbitrage_constraint(self):
        """Calibrated parameters must satisfy φ·θ ≤ 4 (no-butterfly condition)."""
        F, tau = 100.0, 0.25
        theta, phi, rho = 0.04, 3.0, -0.3
        strikes = F * np.exp(np.linspace(-0.25, 0.25, 12))
        k = np.log(strikes / F)
        w = ssvi_total_var(k, theta, phi, rho)
        ivs = np.sqrt(w / tau)

        params = calibrate_ssvi(strikes, ivs, F, tau)
        theta_fit, phi_fit, _ = params
        assert phi_fit * theta_fit <= 4.0 + 1e-6
