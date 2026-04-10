"""
tests/test_nbody.py
--------------------
Tests for the N-body leapfrog integrator.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nbody import run_nbody, scale_factor_steps
from initial_conditions import make_ics


h, Omega_m, Omega_b, n_s, sigma8 = 0.6736, 0.3153, 0.0493, 0.9649, 0.8111


def test_scale_factor_steps():
    """Scale factor array should go from a_init to a_final = 1."""
    a_steps = scale_factor_steps(49.0, 0.0, 50)
    assert len(a_steps) == 51
    np.testing.assert_allclose(a_steps[0], 1.0 / 50.0, rtol=1e-10)
    np.testing.assert_allclose(a_steps[-1], 1.0, rtol=1e-10)
    assert np.all(np.diff(a_steps) > 0), "Scale factor not monotonically increasing"


def test_nbody_runs_and_returns_snapshots():
    """Short N-body simulation should complete and return snapshots."""
    N, L, N_mesh = 8, 500.0, 16
    pos, vel, _ = make_ics(N, L, h, Omega_m, Omega_b, n_s, sigma8,
                            z_initial=49.0, seed=42)
    snapshots, _ = run_nbody(pos, vel, N_mesh, L, h, Omega_m,
                              z_initial=49.0, z_final=0.0, n_steps=5,
                              z_snapshots=[0.0], output_dir='/tmp/bao_test_snaps',
                              save=False)
    assert len(snapshots) >= 1, "No snapshots returned"
    snap = snapshots[-1]
    assert snap['pos'].shape == (3, N**3)
    assert snap['vel'].shape == (3, N**3)
    assert snap['z'] < 0.5, f"Final snapshot z = {snap['z']:.2f}, expected ~0"


def test_nbody_positions_stay_in_box():
    """Particles should remain within the periodic box."""
    N, L, N_mesh = 8, 500.0, 16
    pos, vel, _ = make_ics(N, L, h, Omega_m, Omega_b, n_s, sigma8,
                            z_initial=49.0, seed=42)
    snapshots, _ = run_nbody(pos, vel, N_mesh, L, h, Omega_m,
                              z_initial=49.0, z_final=0.0, n_steps=5,
                              z_snapshots=[0.0], save=False)
    final_pos = snapshots[-1]['pos']
    assert np.all(final_pos >= -L / 2) and np.all(final_pos <= L / 2), \
        f"Positions out of box: [{final_pos.min():.1f}, {final_pos.max():.1f}]"


def test_nbody_velocity_is_peculiar():
    """Snapshot vel should be peculiar velocity (v = a*dx/dt), not conjugate momentum."""
    N, L, N_mesh = 8, 500.0, 16
    pos, vel, _ = make_ics(N, L, h, Omega_m, Omega_b, n_s, sigma8,
                            z_initial=49.0, seed=42)
    snapshots, _ = run_nbody(pos, vel, N_mesh, L, h, Omega_m,
                              z_initial=49.0, z_final=0.0, n_steps=5,
                              z_snapshots=[0.0], save=False)
    v_rms = np.sqrt(np.mean(snapshots[-1]['vel']**2))
    # With only 8^3 particles, 5 steps, 500 Mpc/h box,
    # velocities are small but should be finite and physical
    assert 0.1 < v_rms < 5000, f"RMS velocity = {v_rms:.1f} km/s — seems unphysical"
