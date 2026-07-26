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

Reference: Jing (2005), ApJ 620, 559 — CIC aliasing and power spectrum estimation.
https://doi.org/10.1086/427087
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from power_spectrum import estimate_pk, cic_window_correction_1d


def _pk_wrong_ordering(pos, L, n_mesh=64):
    """Deliberately wrong estimator: subtract shot noise BEFORE dividing by W_CIC^2.

    This overcorrects at high k where W_CIC → 0, causing divergent negative values
    near the Nyquist frequency.  Used in tests to cross-check that the correct
    ``estimate_pk`` ordering passes assertions that this wrong ordering fails.
    """
    from pm_gravity import cic_paint_vectorized
    N_particles = pos.shape[1]
    V = L**3
    dx = L / n_mesh
    nbar = N_particles / V

    delta = cic_paint_vectorized(pos, n_mesh, L)
    delta_k = np.fft.fftn(delta) / n_mesh**3

    dk = 2 * np.pi / L
    k1d = np.fft.fftfreq(n_mesh, d=1.0 / n_mesh) * dk
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    Pk_raw = np.abs(delta_k)**2 * V
    # WRONG ORDER: subtract shot noise first, then divide by W_CIC^2
    Pk_raw -= 1.0 / nbar
    W_cic = (cic_window_correction_1d(kx, dx)
             * cic_window_correction_1d(ky, dx)
             * cic_window_correction_1d(kz, dx))
    W_cic[0, 0, 0] = 1.0
    Pk_raw /= W_cic**2

    k_nyq = np.pi / dx
    k_min = dk
    n_bins = n_mesh // 2
    k_edges = np.linspace(0, k_nyq, n_bins + 1)
    k_flat = k_mag.ravel()
    Pk_flat = Pk_raw.ravel()
    bin_idx = np.digitize(k_flat, k_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    bi = bin_idx[valid]
    counts = np.bincount(bi, minlength=n_bins)
    k_sum = np.bincount(bi, weights=k_flat[valid], minlength=n_bins)
    Pk_sum = np.bincount(bi, weights=Pk_flat[valid], minlength=n_bins)
    good = counts > 0
    k_bins = k_sum[good] / counts[good]
    Pk_bins = Pk_sum[good] / counts[good]
    mask = (k_bins >= k_min) & (k_bins < 0.9 * k_nyq)
    return k_bins[mask], Pk_bins[mask]


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

    Place N_total particles whose x-positions are sampled from the density
    ρ(x) ∝ 1 + A sin(k0 x) using acceptance-rejection sampling.  With A ≤ 1
    the acceptance probability (1 + A sin(k0 x))/(1 + A) is in [0, 1]
    everywhere, guaranteeing a valid nonneg density for rejection sampling.

    Two checks distinguish the CIC/shot-noise ordering convention:
      (1) Signal recovery: P(k0) is within a justified factor of the analytic
          expectation P_analytic = (A/2)^2 × L^3 (see Jing 2005, ApJ 620, 559).
      (2) Ordering discriminator: mean P(k) for k > 0.8 × k_Nyquist must be
          close to zero for the correct estimator.  The wrong ordering
          (subtract-then-divide) causes W_CIC → 0 near Nyquist to amplify the
          subtracted shot noise to large negative values (~−1 to −10 × shot_noise
          in the high-k bins).  A cross-check using _pk_wrong_ordering() verifies
          that the wrong ordering FAILS this same assertion, proving that the test
          actually discriminates between the two conventions.

    The k cut k > 0.8 × k_Nyquist targets the highest-k bins where the CIC
    window is small and the ordering error is largest (Jing 2005).
    """
    N = 32
    L = 500.0
    rng = np.random.default_rng(99)

    # Fundamental mode: k0 = 2π/L is the lowest non-zero Fourier mode of the
    # box, guaranteed to fall in the first k-bin and well away from Nyquist.
    N_total = N**3

    # Sinusoidal overdensity along x: ρ(x) ∝ 1 + A·sin(2π x / L)
    # A = 0.5 ensures ρ ≥ 0 everywhere (valid rejection sampling).
    # A must be in (0, 1]: A=0 gives a uniform catalog (no signal to test);
    # A > 1 makes ρ negative for some x (invalid for rejection sampling).
    A = 0.5   # amplitude; 0 < A ≤ 1
    k0 = 2 * np.pi / L   # fundamental mode [h/Mpc]
    x_candidates = rng.uniform(-L/2, L/2, 10 * N_total)
    # Acceptance probability is in [0, 1] for all x when A ≤ 1
    accept = rng.uniform(0, 1, len(x_candidates)) < (1 + A * np.sin(k0 * x_candidates)) / (1 + A)
    x_signal = x_candidates[accept][:N_total]
    pos_signal = np.vstack([
        x_signal,
        rng.uniform(-L/2, L/2, len(x_signal)),
        rng.uniform(-L/2, L/2, len(x_signal)),
    ])

    k, Pk, nmodes = estimate_pk(pos_signal, N, L, n_mesh=64, subtract_shotnoise=True)

    nbar = N_total / L**3
    shot_noise = 1.0 / nbar

    # Check 1: Analytic signal amplitude within a justified tolerance.
    # Analytic P at the fundamental mode: (A/2)^2 × V = (0.25)^2 × 500^3 ≈ 7.8e6.
    # The shell-averaged bin includes approximately 18 modes for n_mesh=64, L=500
    # (6 axis-aligned ±k0 modes plus 12 face-diagonal modes with |k|=√2·k0 that
    # also fall in the first included bin).  The 2 signal modes (±k0 along x)
    # contribute ~(2/18) × P_analytic ≈ 0.11 × P_analytic after shell averaging.
    # A 5% floor (0.05 × P_analytic) provides generous margin for sampling noise.
    idx_k0 = np.argmin(np.abs(k - k0))
    analytic_Pk0 = (A / 2)**2 * L**3
    assert Pk[idx_k0] > 0.05 * analytic_Pk0, (
        f"P(k0={k0:.4f}) = {Pk[idx_k0]:.2e} is far below the analytic expectation "
        f"0.05 × (A/2)^2 × V = {0.05 * analytic_Pk0:.2e}. Injected signal not recovered."
    )

    # Check 2: Ordering discriminator at very high k (k > 0.8 × k_Nyquist).
    # At these scales, W_CIC is significantly < 1.  The correct estimator
    # (divide-then-subtract) gives P(k) ≈ 0 after shot-noise removal.
    # The wrong estimator (subtract-then-divide) amplifies the residual by 1/W_CIC^2
    # ≫ 1, driving the shell-averaged mean to ~ -1 to -3 × shot_noise.
    # A threshold of -0.5 × shot_noise is:
    #   - well above what the wrong ordering produces (≲ -1 × shot_noise), and
    #   - well below zero, so correct-ordering noise does not accidentally fail it.
    k_nyq = np.pi * 64 / L   # Nyquist frequency for n_mesh=64
    high_k_mask = k > 0.8 * k_nyq
    assert high_k_mask.any(), (
        "No k-bins found above 0.8 × k_Nyquist; check n_mesh and L parameters."
    )
    mean_high_k = np.mean(Pk[high_k_mask])
    assert mean_high_k > -0.5 * shot_noise, (
        f"Mean P(k) at k > 0.8 k_Nyq: {mean_high_k:.1f} (Mpc/h)^3 "
        f"(threshold = -0.5 × shot_noise = {-0.5 * shot_noise:.1f}). "
        "Strongly negative high-k values indicate wrong CIC/shot-noise ordering "
        "(Jing 2005, ApJ 620, 559)."
    )

    # Cross-check: the WRONG ordering must FAIL the same high-k assertion.
    # This verifies that the test actually discriminates the two conventions.
    k_wrong, Pk_wrong = _pk_wrong_ordering(pos_signal, L, n_mesh=64)
    high_k_wrong = k_wrong > 0.8 * k_nyq
    if high_k_wrong.any():
        mean_wrong_high_k = np.mean(Pk_wrong[high_k_wrong])
        assert mean_wrong_high_k < -0.5 * shot_noise, (
            f"Wrong ordering should produce mean P(k) < -0.5 × shot_noise "
            f"at k > 0.8 k_Nyq, but got {mean_wrong_high_k:.1f} "
            f"(shot_noise = {shot_noise:.1f}). "
            "The test is not correctly discriminating the ordering conventions."
        )
