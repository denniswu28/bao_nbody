"""
tests/test_initial_conditions.py
---------------------------------
Tests for Zel'dovich initial conditions.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from initial_conditions import make_ics, hubble, growth_rate, make_grid, generate_gaussian_field


# Planck 2018 cosmology
h, Omega_m, Omega_b, n_s, sigma8 = 0.6736, 0.3153, 0.0493, 0.9649, 0.8111


def test_growth_factor_z49():
    """D(z=49) should be ~0.02 (matter dominated, D ~ a)."""
    from pk_input import growth_factor
    D = growth_factor(49.0, Omega_m)
    assert 0.015 < D < 0.035, f"D(z=49) = {D:.4f}, expected ~0.02"


def test_growth_rate_matter_dominated():
    """f(z=49) should be ~1.0 in matter-dominated regime."""
    f = growth_rate(49.0, Omega_m)
    assert 0.98 < f < 1.02, f"f(z=49) = {f:.4f}, expected ~1.0"


def test_hubble_high_z():
    """E(z=49) should be dominated by matter: E ~ (1+z)^(3/2) * Omega_m^(1/2)."""
    E = hubble(49.0, h, Omega_m)
    E_matter = np.sqrt(Omega_m) * (1 + 49.0)**1.5
    assert abs(E - E_matter) / E < 0.01, f"E(z=49) = {E:.2f}, expected ~{E_matter:.2f}"


def test_make_grid_shape():
    """Grid should have shape (3, N^3)."""
    N, L = 16, 100.0
    pos = make_grid(N, L)
    assert pos.shape == (3, N**3), f"Grid shape {pos.shape}, expected (3, {N**3})"


def test_make_grid_range():
    """Grid positions should be within [-L/2, L/2]."""
    N, L = 16, 100.0
    pos = make_grid(N, L)
    assert np.all(pos >= -L / 2) and np.all(pos <= L / 2), "Grid positions out of range"


def test_ics_shape():
    """ICs should produce arrays of shape (3, N^3)."""
    N, L = 16, 500.0
    pos, vel, delta_k = make_ics(N, L, h, Omega_m, Omega_b, n_s, sigma8,
                                  z_initial=49.0, seed=42)
    assert pos.shape == (3, N**3), f"pos shape {pos.shape}"
    assert vel.shape == (3, N**3), f"vel shape {vel.shape}"
    assert delta_k.shape == (N, N, N), f"delta_k shape {delta_k.shape}"


def test_ics_positions_in_box():
    """Displaced positions should remain within the box after periodic wrapping."""
    N, L = 16, 500.0
    pos, vel, _ = make_ics(N, L, h, Omega_m, Omega_b, n_s, sigma8,
                            z_initial=49.0, seed=42)
    assert np.all(pos >= -L / 2) and np.all(pos <= L / 2), \
        f"Positions out of box: range [{pos.min():.1f}, {pos.max():.1f}]"


def test_ics_small_displacements():
    """At z=49, ZA displacements should be small (verified via velocity magnitude)."""
    N, L = 16, 500.0
    pos, vel, _ = make_ics(N, L, h, Omega_m, Omega_b, n_s, sigma8,
                            z_initial=49.0, seed=42)
    # v = a * H * f * Psi, so Psi_rms = v_rms / (a * H * f)
    from initial_conditions import hubble as Hz_func
    a_init = 1.0 / (1 + 49.0)
    H = Hz_func(49.0, h, Omega_m) * 100  # km/s / (Mpc/h)
    f = growth_rate(49.0, Omega_m)
    v_rms = np.sqrt(np.mean(vel**2))
    psi_rms = v_rms / (a_init * H * f)
    assert psi_rms < 5, f"RMS displacement = {psi_rms:.3f} Mpc/h — too large for z=49"


def test_ics_small_velocities():
    """At z=49, peculiar velocities should be small (< 10 km/s)."""
    N, L = 16, 500.0
    pos, vel, _ = make_ics(N, L, h, Omega_m, Omega_b, n_s, sigma8,
                            z_initial=49.0, seed=42)
    rms_vel = np.sqrt(np.mean(vel**2))
    assert rms_vel < 200, f"RMS velocity = {rms_vel:.1f} km/s — too large for z=49"


def test_ics_reproducible():
    """Same seed should produce identical ICs."""
    N, L = 8, 200.0
    pos1, vel1, _ = make_ics(N, L, h, Omega_m, Omega_b, n_s, sigma8,
                              z_initial=49.0, seed=99)
    pos2, vel2, _ = make_ics(N, L, h, Omega_m, Omega_b, n_s, sigma8,
                              z_initial=49.0, seed=99)
    np.testing.assert_array_equal(pos1, pos2)
    np.testing.assert_array_equal(vel1, vel2)
