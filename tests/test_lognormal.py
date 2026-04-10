"""
tests/test_lognormal.py
------------------------
Tests for the lognormal catalog generator.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lognormal import generate_lognormal_catalog, generate_lognormal_field, poisson_sample_vectorized


h, Omega_m, Omega_b, n_s, sigma8 = 0.6736, 0.3153, 0.0493, 0.9649, 0.8111


def test_lognormal_field_positive():
    """Lognormal density field: 1 + delta_LN > 0 everywhere."""
    from pk_input import power_spectrum
    from scipy.interpolate import interp1d

    N_mesh, L = 32, 500.0
    k_arr = np.logspace(-3, 1, 300)
    Pk = power_spectrum(k_arr, h, Omega_m, Omega_b, n_s, sigma8, z=0.38)
    Pk_func = interp1d(k_arr, Pk, bounds_error=False, fill_value=0.0)

    delta_LN = generate_lognormal_field(N_mesh, L, Pk_func, seed=42)
    assert np.all(1 + delta_LN > 0), "Lognormal field has negative density"


def test_lognormal_field_mean():
    """Mean of delta_LN should be ~0 (mean density = background)."""
    from pk_input import power_spectrum
    from scipy.interpolate import interp1d

    N_mesh, L = 32, 500.0
    k_arr = np.logspace(-3, 1, 300)
    Pk = power_spectrum(k_arr, h, Omega_m, Omega_b, n_s, sigma8, z=0.38)
    Pk_func = interp1d(k_arr, Pk, bounds_error=False, fill_value=0.0)

    delta_LN = generate_lognormal_field(N_mesh, L, Pk_func, seed=42)
    # Mean of lognormal field: <exp(G - sigma^2/2) - 1> = 0 analytically
    # but finite box has sample variance, so allow tolerance
    assert abs(np.mean(delta_LN)) < 0.5, \
        f"Mean of lognormal field = {np.mean(delta_LN):.4f}, expected ~0"


def test_catalog_galaxy_count():
    """Generated catalog should have roughly nbar * V galaxies."""
    N_mesh, L = 32, 500.0
    nbar = 3e-4

    pos, delta_LN = generate_lognormal_catalog(
        N_mesh, L, h, Omega_m, Omega_b, n_s, sigma8,
        nbar=nbar, b=1.5, z=0.38, seed=42)

    N_expected = nbar * L**3
    N_actual = pos.shape[1]
    assert abs(N_actual - N_expected) / N_expected < 0.3, \
        f"Galaxy count {N_actual} far from expected {N_expected:.0f}"


def test_catalog_positions_in_box():
    """Galaxy positions should be within [-L/2, L/2]."""
    N_mesh, L = 32, 500.0
    nbar = 3e-4

    pos, _ = generate_lognormal_catalog(
        N_mesh, L, h, Omega_m, Omega_b, n_s, sigma8,
        nbar=nbar, b=1.5, z=0.38, seed=42)

    assert np.all(pos >= -L / 2 - 1) and np.all(pos <= L / 2 + 1), \
        f"Galaxy positions out of box: [{pos.min():.1f}, {pos.max():.1f}]"


def test_catalog_shape():
    """Catalog should have shape (3, N_galaxies)."""
    N_mesh, L = 32, 500.0
    nbar = 3e-4

    pos, delta_LN = generate_lognormal_catalog(
        N_mesh, L, h, Omega_m, Omega_b, n_s, sigma8,
        nbar=nbar, b=1.5, z=0.38, seed=42)

    assert pos.shape[0] == 3, f"pos shape[0] = {pos.shape[0]}, expected 3"
    assert delta_LN.shape == (N_mesh, N_mesh, N_mesh)
