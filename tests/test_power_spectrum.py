"""
tests/test_power_spectrum.py
----------------------------
Tests for the P(k) estimator.

Shot noise and CIC deconvolution convention
-------------------------------------------
CIC assignment windows both signal and shot noise identically, so the
measured power in each mode is:

    P_measured(k) = W_CIC^2(k) * [P(k) + 1/nbar]

The correct estimator therefore:
  1. Divides by W_CIC^2(k) to deconvolve both signal and shot noise.
  2. Subtracts 1/nbar to remove the (now-unwindowed) Poisson shot noise.

Reversing this order (subtract then divide) overcorrects at high k because
W_CIC -> 0 near the Nyquist frequency, causing divergent negative values.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from power_spectrum import estimate_pk


def test_uniform_catalog_shotnoise():
    """
    A uniform random catalog should have P(k) ~ 0 after shot noise subtraction.
    Validates that CIC deconvolution is applied before shot noise subtraction
    (wrong order produces large negative P(k) at high k).
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


def test_shotnoise_no_subtraction_gives_positive_pk():
    """Without subtraction, P(k) for a uniform catalog should be close to 1/nbar > 0."""
    N = 16
    L = 300.0
    rng = np.random.default_rng(5)
    pos = rng.uniform(-L/2, L/2, (3, N**3))
    k, Pk, nmodes = estimate_pk(pos, N, L, n_mesh=32, subtract_shotnoise=False)
    nbar = N**3 / L**3
    shot_noise = 1.0 / nbar
    # All bins should be positive and within a factor of 2 of the shot noise
    assert np.all(Pk > 0), "P(k) without shot noise subtraction should be positive"
    assert np.all(Pk < 2 * shot_noise), (
        f"P(k) without subtraction is far above shot noise: max Pk={Pk.max():.1f} "
        f"vs 1/nbar={shot_noise:.1f}"
    )


def test_uniform_catalog_residual_scale():
    """After correct ordering (divide then subtract), mean |P(k)| << 1/nbar.

    Checks that the CIC deconvolution ordering is correct: incorrect ordering
    would give |mean(Pk)| >> 1/nbar due to large negative values at high k.
    """
    N = 32
    L = 500.0
    rng = np.random.default_rng(7)
    pos = rng.uniform(-L/2, L/2, (3, N**3))
    k, Pk, nmodes = estimate_pk(pos, N, L, n_mesh=64, subtract_shotnoise=True)
    nbar = N**3 / L**3
    shot_noise = 1.0 / nbar
    # mean should be near zero (< 5% of shot noise); sign should be mixed, not all negative
    mean_pk = np.mean(Pk)
    assert abs(mean_pk) < 0.05 * shot_noise, (
        f"mean P(k) = {mean_pk:.1f}, shot_noise = {shot_noise:.1f}.  "
        "Large negative mean indicates wrong CIC/shot-noise ordering."
    )


def test_pk_reproducible():
    """Same positions should give identical P(k) (determinism check)."""
    N = 16
    L = 300.0
    rng = np.random.default_rng(42)
    pos = rng.uniform(-L/2, L/2, (3, N**3))
    k1, Pk1, nm1 = estimate_pk(pos, N, L, n_mesh=32)
    k2, Pk2, nm2 = estimate_pk(pos, N, L, n_mesh=32)
    np.testing.assert_array_equal(Pk1, Pk2, err_msg="P(k) not reproducible")
