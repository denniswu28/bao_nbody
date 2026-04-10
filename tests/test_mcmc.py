"""
tests/test_mcmc.py
-------------------
Tests for the MCMC BAO template fitting.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mcmc import bao_template, log_likelihood, log_prior, run_mcmc
from pk_input import power_spectrum, power_spectrum as power_spectrum_nw, sound_horizon
from scipy.interpolate import interp1d


h, Omega_m, Omega_b, n_s, sigma8 = 0.6736, 0.3153, 0.0493, 0.9649, 0.8111


def _make_template_funcs():
    """Create P(k) interpolators for template fitting."""
    k_th = np.logspace(-3, 0, 1000)
    Pk_lin = power_spectrum(k_th, h, Omega_m, Omega_b, n_s, sigma8, z=0.0, wiggle=True)
    Pk_nw = power_spectrum(k_th, h, Omega_m, Omega_b, n_s, sigma8, z=0.0, wiggle=False)
    Pk_lin_func = interp1d(k_th, Pk_lin, bounds_error=False, fill_value=0.0)
    Pk_nw_func = interp1d(k_th, Pk_nw, bounds_error=False, fill_value=0.0)
    return Pk_lin_func, Pk_nw_func


def test_bao_template_positive():
    """BAO template should return positive P(k) for reasonable parameters."""
    Pk_lin_func, Pk_nw_func = _make_template_funcs()
    k = np.logspace(-2, -0.5, 50)
    Pk = bao_template(k, alpha=1.0, Sigma=5.0, B=1.0, Pk_lin_func=Pk_lin_func, Pk_nw_func=Pk_nw_func)
    assert np.all(Pk > 0), "BAO template has non-positive values"


def test_bao_template_alpha1_matches_input():
    """At alpha=1 and B=1, template should closely match input P(k) (modulo damping)."""
    Pk_lin_func, Pk_nw_func = _make_template_funcs()
    k = np.logspace(-2, -0.5, 50)
    Pk_template = bao_template(k, alpha=1.0, Sigma=0.0, B=1.0,
                                Pk_lin_func=Pk_lin_func, Pk_nw_func=Pk_nw_func)
    Pk_input = Pk_lin_func(k)
    np.testing.assert_allclose(Pk_template, Pk_input, rtol=1e-6)


def test_log_prior_in_range():
    """Prior should be 0 inside range, -inf outside."""
    priors = {'alpha': [0.8, 1.2], 'Sigma': [0.0, 20.0], 'B': [0.5, 2.0]}
    assert log_prior([1.0, 5.0, 1.0], priors) == 0.0
    assert log_prior([0.7, 5.0, 1.0], priors) == -np.inf  # alpha below range
    assert log_prior([1.0, 25.0, 1.0], priors) == -np.inf  # Sigma above range


def test_mcmc_recovers_alpha():
    """MCMC should recover alpha=1.0 when fitting noiseless P(k) from the same template."""
    Pk_lin_func, Pk_nw_func = _make_template_funcs()

    # Generate "data" from template with known alpha=1.0, Sigma=5.0, B=1.0
    k_data = np.linspace(0.02, 0.3, 30)
    Pk_data = bao_template(k_data, 1.0, 5.0, 1.0, Pk_lin_func, Pk_nw_func)
    Pk_err = 0.01 * Pk_data  # 1% errors

    priors = {'alpha': [0.8, 1.2], 'Sigma': [0.0, 20.0], 'B': [0.5, 2.0]}
    chain, _, accept_rate = run_mcmc(
        k_data, Pk_data, Pk_err,
        Pk_lin_func, Pk_nw_func,
        theta_init=[1.0, 5.0, 1.0],
        priors=priors,
        n_steps=5000, n_burn=1000, seed=42,
        step_size=np.array([0.001, 0.1, 0.005]),  # smaller steps for tight data
    )

    alpha_median = np.median(chain[:, 0])
    assert abs(alpha_median - 1.0) < 0.05, \
        f"MCMC alpha = {alpha_median:.4f}, expected ~1.0"
    assert 0.05 < accept_rate < 0.95, \
        f"Acceptance rate {accept_rate:.3f} out of reasonable range"
