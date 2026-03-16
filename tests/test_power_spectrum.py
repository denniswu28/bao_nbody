"""
tests/test_power_spectrum.py
----------------------------
Tests for the P(k) estimator.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from power_spectrum import estimate_pk


def test_uniform_catalog_shotnoise():
    """
    A uniform random catalog should have P(k) ~ 0 after shot noise subtraction.
    """
    N = 32
    L = 500.0
    rng = np.random.default_rng(0)
    pos = rng.uniform(-L/2, L/2, (3, N**3))

    k, Pk, nmodes = estimate_pk(pos, N, L, n_mesh=64, subtract_shotnoise=True)

    # After shot noise subtraction, P(k) should be small relative to 1/nbar
    nbar = N**3 / L**3
    shot_noise = 1.0 / nbar
    assert np.abs(np.mean(Pk)) < 0.1 * shot_noise, \
        f"Residual P(k) after shot noise subtraction is too large: {np.mean(Pk):.2f}"


def test_pk_bins_increasing():
    """k bins should be monotonically increasing."""
    N = 32
    L = 500.0
    rng = np.random.default_rng(1)
    pos = rng.uniform(-L/2, L/2, (3, N**3))
    k, Pk, nmodes = estimate_pk(pos, N, L, n_mesh=64)
    assert np.all(np.diff(k) > 0), "k bins not monotonically increasing"


def test_nmodes_positive():
    """Number of modes per bin should be positive."""
    N = 32
    L = 500.0
    rng = np.random.default_rng(2)
    pos = rng.uniform(-L/2, L/2, (3, N**3))
    k, Pk, nmodes = estimate_pk(pos, N, L, n_mesh=64)
    assert np.all(nmodes > 0), "Some k-bins have zero modes"
