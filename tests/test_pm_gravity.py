"""
tests/test_pm_gravity.py
-------------------------
Tests for the PM gravity solver.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pm_gravity import cic_paint_vectorized, compute_forces, cic_interpolate, compute_particle_forces


def test_cic_mass_conservation():
    """CIC paint should conserve total mass: sum(delta) * n_bar ≈ N_particles."""
    N_mesh, L = 32, 500.0
    rng = np.random.default_rng(42)
    N_part = 16**3
    pos = rng.uniform(-L/2, L/2, (3, N_part))

    delta = cic_paint_vectorized(pos, N_mesh, L)

    # Overdensity: delta = rho/rho_bar - 1, so sum(1 + delta) * (V / N_cells) * n_bar = N_particles
    n_bar = N_part / N_mesh**3
    total_mass = np.sum(1 + delta) * n_bar
    assert abs(total_mass - N_part) < 1, \
        f"Mass not conserved: got {total_mass:.1f}, expected {N_part}"


def test_cic_uniform_zero_overdensity():
    """A uniform grid should give delta ~ 0 everywhere."""
    N, L = 16, 500.0
    dx = L / N
    x1d = np.arange(N) * dx - L / 2 + dx / 2
    xx, yy, zz = np.meshgrid(x1d, x1d, x1d, indexing='ij')
    pos_grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=0)

    # Using N_mesh = N means one particle per cell exactly
    delta = cic_paint_vectorized(pos_grid, N, L)
    assert np.max(np.abs(delta)) < 0.1, \
        f"Uniform grid overdensity too large: max|delta| = {np.max(np.abs(delta)):.4f}"


def test_forces_shape():
    """Force field should have shape (3, N_mesh, N_mesh, N_mesh)."""
    N_mesh, L = 32, 500.0
    delta = np.random.randn(N_mesh, N_mesh, N_mesh) * 0.01
    forces = compute_forces(delta, N_mesh, L, h=0.6736, Omega_m=0.3153, a=1.0)
    assert forces.shape == (3, N_mesh, N_mesh, N_mesh), f"Forces shape: {forces.shape}"


def test_forces_zero_for_uniform():
    """A uniform density field (delta=0) should give zero forces."""
    N_mesh, L = 32, 500.0
    delta = np.zeros((N_mesh, N_mesh, N_mesh))
    forces = compute_forces(delta, N_mesh, L, h=0.6736, Omega_m=0.3153, a=1.0)
    assert np.max(np.abs(forces)) < 1e-10, \
        f"Non-zero force for uniform field: max|F| = {np.max(np.abs(forces)):.2e}"


def test_particle_forces_output_shape():
    """compute_particle_forces should return (3, N_particles) forces and (N_mesh,)^3 delta."""
    N_mesh, L = 32, 500.0
    N_part = 8**3
    rng = np.random.default_rng(42)
    pos = rng.uniform(-L/2, L/2, (3, N_part))

    f_particles, delta = compute_particle_forces(pos, N_mesh, L, h=0.6736, Omega_m=0.3153, a=0.5)
    assert f_particles.shape == (3, N_part), f"f_particles shape: {f_particles.shape}"
    assert delta.shape == (N_mesh, N_mesh, N_mesh), f"delta shape: {delta.shape}"
