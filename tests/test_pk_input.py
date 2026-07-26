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
    """Sound horizon should be ~101 Mpc/h (= ~150 Mpc) for Planck cosmology.

    sound_horizon() returns r_s in Mpc/h via EH98 eq. 26 fitting formula.
    The formula gives r_s_Mpc ≈ 149.8 Mpc; multiplying by h converts to
    Mpc/h (h^-1 Mpc): r_s = r_s_Mpc * h ≈ 100.9 Mpc/h for Planck 2018.
    Convention: all distances in this codebase are in Mpc/h.
    """
    r_s = sound_horizon(0.6736, 0.3153, 0.0493)
    assert 90 < r_s < 115, f"Unexpected r_s = {r_s:.1f} Mpc/h"


def test_sound_horizon_regression():
    """Regression test: EH98 eq.26 Mpc/h value must stay within 1% of 100.9 Mpc/h.

    sound_horizon() returns r_s = r_s_Mpc × h ≈ 100.9 Mpc/h where r_s_Mpc ≈ 149.8 Mpc
    is the EH98 fitting formula output and h = 0.6736 converts to Mpc/h (h⁻¹ Mpc).
    """
    r_s = sound_horizon(0.6736, 0.3153, 0.0493)
    r_s_expected = 100.92   # Mpc/h (= 149.8 Mpc * h)
    assert abs(r_s - r_s_expected) / r_s_expected < 0.01, (
        f"r_s = {r_s:.4f} Mpc/h, expected ~{r_s_expected} Mpc/h "
        f"(EH98 eq.26, Planck 2018 cosmology)"
    )


def test_sound_horizon_nowiggle_consistency():
    """transfer_function_nowiggle uses sound_horizon(h,Om,Ob)/h internally.

    The no-wiggle transfer function uses k*s (dimensionless) where k is in Mpc^-1
    (EH98 internals) and s = sound_horizon(h,Om,Ob)/h is in Mpc.  This test
    verifies that:
    1. The Mpc sound horizon is in the expected EH98 range (~149 Mpc).
    2. The no-wiggle T(k) evaluated at k matching the BAO scale is in (0, 1].
    """
    from pk_input import transfer_function_nowiggle
    h, Omega_m, Omega_b = 0.6736, 0.3153, 0.0493
    r_s_code = sound_horizon(h, Omega_m, Omega_b)  # Mpc/h
    r_s_mpc = r_s_code / h                          # Mpc (for EH98 internals)
    assert 130 < r_s_mpc < 165, (
        f"Sound horizon in Mpc = {r_s_mpc:.1f}; expected ~149 Mpc from EH98 eq.26"
    )
    # Verify T_nw is a valid transfer function: evaluate at a few wavenumbers
    k_test = np.array([0.01, 0.05, 0.1, 0.2, 0.5])  # h/Mpc
    T_nw = transfer_function_nowiggle(k_test, h, Omega_m, Omega_b)
    # Allow a tiny tolerance above 1.0 for floating-point rounding at very large scales
    _T_UPPER_TOL = 1e-9
    assert np.all(T_nw > 0) and np.all(T_nw <= 1.0 + _T_UPPER_TOL), (
        f"No-wiggle T(k) out of range (0, 1]: {T_nw}"
    )
    # T_nw should be close to 1 at large scales and decrease at small scales
    assert T_nw[0] > T_nw[-1], (
        "No-wiggle T(k) should decrease from large to small scales"
    )


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
