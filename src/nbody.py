"""
nbody.py
--------
Leapfrog (kick-drift-kick) N-body integrator with PM gravity.

The leapfrog integrator in cosmological coordinates uses the scale factor a
as the time variable. Each step:

    v_(i+1/2) = v_(i-1/2) + g(x_i) * dt    [kick]
    x_(i+1)   = x_i + v_(i+1/2) * dt        [drift]

where g is the gravitational acceleration from the PM solver, and dt is
chosen adaptively from the scale factor stepping da.

Snapshots are saved at specified redshifts for P(k) analysis and animation.
"""

import numpy as np
import os
import h5py
from tqdm import tqdm

from pm_gravity import compute_particle_forces
from initial_conditions import hubble, growth_rate


def scale_factor_steps(z_initial, z_final, n_steps):
    """
    Generate scale factor steps from z_initial to z_final.
    Linear stepping in scale factor a = 1/(1+z).
    """
    a_initial = 1 / (1 + z_initial)
    a_final   = 1 / (1 + z_final)
    return np.linspace(a_initial, a_final, n_steps + 1)


def _find_snapshot_steps(a_steps, z_snapshots):
    """
    Find which steps are closest to the requested snapshot redshifts.
    """
    snap_indices = []
    for z in z_snapshots:
        a_target = 1 / (1 + z)
        idx = np.argmin(np.abs(a_steps - a_target))
        snap_indices.append(idx)
    return sorted(set(snap_indices))


def save_snapshot(pos, vel, delta, a, step, output_dir):
    """Save a simulation snapshot to HDF5."""
    os.makedirs(output_dir, exist_ok=True)
    z = 1 / a - 1
    fname = os.path.join(output_dir, f"snap_{step:04d}_z{z:.2f}.h5")
    with h5py.File(fname, 'w') as f:
        f.create_dataset('pos', data=pos, compression='gzip')
        f.create_dataset('vel', data=vel, compression='gzip')
        f.create_dataset('delta', data=delta, compression='gzip')
        f.attrs['a'] = a
        f.attrs['z'] = z
        f.attrs['step'] = step
    return fname


def load_snapshot(fname):
    """Load a simulation snapshot from HDF5."""
    with h5py.File(fname, 'r') as f:
        pos   = f['pos'][:]
        vel   = f['vel'][:]
        delta = f['delta'][:]
        a     = f.attrs['a']
        z     = f.attrs['z']
    return pos, vel, delta, a, z


def run_nbody(pos, vel, N_mesh, L, h, Omega_m,
              z_initial=49.0, z_final=0.0, n_steps=50,
              z_snapshots=None, output_dir='outputs/snapshots',
              save=True):
    """
    Run the PM N-body simulation from z_initial to z_final.

    Parameters
    ----------
    pos : ndarray, shape (3, N^3)
        Initial particle positions in Mpc/h.
    vel : ndarray, shape (3, N^3)
        Initial particle velocities in km/s.
    N_mesh : int
        PM mesh resolution.
    L : float
        Box side length in Mpc/h.
    h, Omega_m : float
        Cosmological parameters.
    z_snapshots : list of float, optional
        Redshifts at which to save snapshots. Defaults to [z_initial, 2, 1, 0].
    output_dir : str
        Directory to save snapshots.
    save : bool
        Whether to save snapshots to disk.

    Returns
    -------
    snapshots : list of dict
        List of {'pos', 'vel', 'delta', 'a', 'z'} dicts at each snapshot.
    """
    if z_snapshots is None:
        z_snapshots = [z_initial, 5.0, 2.0, 1.0, 0.5, z_final]

    a_steps = scale_factor_steps(z_initial, z_final, n_steps)
    snap_step_indices = _find_snapshot_steps(a_steps, z_snapshots)

    snapshots = []
    snapshot_fnames = []

    pos = pos.copy()
    vel = vel.copy()

    print(f"Running N-body: {n_steps} steps, z={z_initial:.0f} -> z={z_final:.1f}")
    print(f"  N_particles = {pos.shape[1]},  N_mesh = {N_mesh},  L = {L} Mpc/h")
    print(f"  Snapshots at steps: {snap_step_indices}")

    for step in tqdm(range(n_steps), desc="N-body"):

        a = a_steps[step]
        a_next = a_steps[step + 1]
        da = a_next - a
        dt = da   # time variable is scale factor; prefactors handled in forces

        # Compute forces at current positions
        f_particles, delta = compute_particle_forces(pos, N_mesh, L, h, Omega_m, a)

        # Half-kick
        vel += 0.5 * f_particles * dt

        # Drift
        pos += vel * dt / a**2   # comoving coordinates: dx/dt = v / a^2

        # Periodic boundary conditions
        pos = pos % L - L / 2

        # Second half-kick (with forces at new positions)
        f_particles_new, delta_new = compute_particle_forces(pos, N_mesh, L, h, Omega_m, a_next)
        vel += 0.5 * f_particles_new * dt

        # Save snapshot if requested
        if (step + 1) in snap_step_indices:
            snap = {
                'pos': pos.copy(),
                'vel': vel.copy(),
                'delta': delta_new.copy(),
                'a': a_next,
                'z': 1 / a_next - 1,
                'step': step + 1,
            }
            snapshots.append(snap)

            if save:
                fname = save_snapshot(pos, vel, delta_new, a_next, step + 1, output_dir)
                snapshot_fnames.append(fname)
                print(f"  Saved snapshot: z={snap['z']:.2f} -> {fname}")

    print(f"Simulation complete. {len(snapshots)} snapshots saved.")
    return snapshots, snapshot_fnames


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from initial_conditions import make_ics

    h, Omega_m, Omega_b, n_s, sigma8 = 0.6736, 0.3153, 0.0493, 0.9649, 0.8111
    N, L = 32, 1500.0   # small N for quick test
    N_mesh = 64

    pos, vel, _ = make_ics(N, L, h, Omega_m, Omega_b, n_s, sigma8,
                            z_initial=49.0, seed=42)

    snapshots, _ = run_nbody(pos, vel, N_mesh, L, h, Omega_m,
                              z_initial=49.0, z_final=0.0, n_steps=20,
                              z_snapshots=[49.0, 2.0, 0.0],
                              output_dir='outputs/snapshots', save=True)

    # Plot density slice from last snapshot
    fig, axes = plt.subplots(1, len(snapshots), figsize=(5*len(snapshots), 4))
    for ax, snap in zip(axes, snapshots):
        im = ax.imshow(snap['delta'].sum(axis=2), cmap='inferno', origin='lower')
        ax.set_title(f"z = {snap['z']:.1f}")
        plt.colorbar(im, ax=ax, label=r'$\delta$')
    plt.tight_layout()
    plt.savefig('outputs/figures/density_evolution.png', dpi=150)
    plt.show()
