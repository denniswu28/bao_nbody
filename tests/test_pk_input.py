"""
tests/test_pk_input.py
----------------------
Unit tests for the Eisenstein-Hu power spectrum module.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pk_input import power_spectrum, sound_horizon, transfer_function_eh


def test_sigma8_normalization():
    """P(k) should be normalized so sigma8 matches input."""
    h, Omega_m, Omega_b, n_s, sigma8 = 0.6736, 0.3153, 0.0493, 0.9649, 0.8111
    k = np.logspace(-4, 2, 2000)
    Pk = power_spectrum(k, h, Omega_m, Omega_b, n_s, sigma8)
    # Compute sigma8 from the output P(k) directly
    R = 8.0
    x = k * R
    W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    W[x < 1e-3] = 1.0
    integrand = k**2 * Pk * W**2 / (2 * np.pi**2)
    sigma8_measured = np.sqrt(np.trapezoid(integrand, k))
    assert abs(sigma8_measured - sigma8) / sigma8 < 0.01, \
        f"sigma8 mismatch: {sigma8_measured:.4f} vs {sigma8:.4f}"


def test_sound_horizon_range():
    """Sound horizon should be ~150 Mpc for Planck cosmology (EH98 fitting formula)."""
    r_s = sound_horizon(0.6736, 0.3153, 0.0493)
    assert 140 < r_s < 160, f"Unexpected r_s = {r_s:.1f} Mpc"


def test_pk_positive():
    """P(k) should be positive for all k."""
    h, Omega_m, Omega_b, n_s, sigma8 = 0.6736, 0.3153, 0.0493, 0.9649, 0.8111
    k = np.logspace(-3, 1, 200)
    Pk = power_spectrum(k, h, Omega_m, Omega_b, n_s, sigma8)
    assert np.all(Pk > 0), "P(k) has non-positive values"


def test_growth_factor_z0():
    """Growth factor should be 1 at z=0."""
    from pk_input import growth_factor
    D = growth_factor(0.0, 0.3153)
    assert abs(D - 1.0) < 1e-6, f"D(z=0) = {D}, expected 1.0"


def test_transfer_function_large_scale():
    """Transfer function should approach 1 on large scales (small k)."""
    h, Omega_m, Omega_b = 0.6736, 0.3153, 0.0493
    k_small = np.array([1e-4])
    T = transfer_function_eh(k_small, h, Omega_m, Omega_b)
    assert abs(T[0] - 1.0) < 0.05, f"T(k->0) = {T[0]:.4f}, expected ~1"
