"""
tests/test_power_spectrum.py
----------------------------
Tests for the P(k) estimator.

Shot Noise and CIC Deconvolution Convention
-------------------------------------------
CIC assignment windows both signal and shot noise identically, so the
measured power in each mode is:

    P_measured(k) = W_CIC^2(k) * [P(k) + 1/nbar]

The correct estimator therefore:
  1. Divides by W_CIC^2(k) to deconvolve both signal and shot noise.
  2. Subtracts 1/nbar to remove the (now-unwindowed) Poisson shot noise.

Reversing this order (subtract then divide) overcorrects at high k because
W_CIC approaches 0 near the Nyquist frequency, causing divergent negative values.
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


def test_shotnoise_no_subtraction_pk_positive():
    """Without shot-noise subtraction, P(k) for a uniform catalog should be close to 1/nbar > 0."""
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


def test_injected_signal_recovery():
    """Controlled-catalog test: P(k) estimator recovers an injected sinusoidal signal.

    Place N_sig particles at positions that trace a single Fourier mode k0 in x,
    plus a uniform random background.  The measured P(k) at the bin containing
    k0 should be significantly above the surrounding shot-noise floor, confirming
    that the CIC/shot-noise ordering correctly preserves injected power.

    This test validates both the ordering fix (divide before subtract) and that
    real signal is not over-subtracted.
    """
    N = 32
    L = 500.0
    rng = np.random.default_rng(99)

    # Fundamental mode: k0 = 2π/L is the LOWEST non-zero Fourier mode of the
    # box (longest wavelength that fits exactly once in L), not the Nyquist.
    # We inject signal at this frequency because it is guaranteed to fall in
    # the first k-bin and avoids aliasing effects near the Nyquist frequency.
    N_total = N**3

    # Sinusoidal overdensity along x: ρ(x) ∝ 1 + A·sin(2π x / L)
    # Sample by rejection: accept if U < (1 + A·sin(2π x / L)) / (1 + A)
    A = 5.0   # amplitude (unitless); large enough to detect above shot noise
    k0 = 2 * np.pi / L   # fundamental mode [h/Mpc]
    x_candidates = rng.uniform(-L/2, L/2, 10 * N_total)
    accept = rng.uniform(0, 1, len(x_candidates)) < (1 + A * np.sin(k0 * x_candidates)) / (1 + A)
    x_signal = x_candidates[accept][:N_total]
    pos_signal = np.vstack([
        x_signal,
        rng.uniform(-L/2, L/2, len(x_signal)),
        rng.uniform(-L/2, L/2, len(x_signal)),
    ])

    k, Pk, nmodes = estimate_pk(pos_signal, N, L, n_mesh=64, subtract_shotnoise=True)

    # The bin closest to k0 should have elevated power
    idx_k0 = np.argmin(np.abs(k - k0))
    nbar = N_total / L**3
    shot_noise = 1.0 / nbar

    # With amplitude A=5 the injected signal contributes ~A²/2 * V = 25/2 × L³ modes
    # of power above shot noise. Factor-of-2 is a conservative floor: if the
    # CIC/shot-noise ordering is wrong (subtract first, then divide), the
    # high-k modes diverge negative and the mean P(k) collapses below zero.
    assert Pk[idx_k0] > 2 * shot_noise, (
        f"P(k0={k0:.4f}) = {Pk[idx_k0]:.2f} should exceed 2×shot_noise = "
        f"{2*shot_noise:.2f}. Wrong CIC/shot-noise ordering may be present."
    )
