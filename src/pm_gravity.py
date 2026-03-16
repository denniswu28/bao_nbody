"""
pm_gravity.py
-------------
Particle-Mesh (PM) gravitational force computation via FFT.

Algorithm:
    1. Paint particles onto N_mesh^3 grid using CIC (Cloud-In-Cell)
    2. FFT density field -> delta(k)
    3. Solve Poisson equation: phi(k) = -delta(k) / k^2
    4. Differentiate: F_i(k) = -i * k_i * phi(k)
    5. Inverse FFT -> F_i(x)
    6. Interpolate forces back to particle positions (CIC)

The Poisson equation in comoving coordinates:
    nabla^2 phi = 4 * pi * G * rho_bar * a * delta
    -> in k-space: phi(k) = -4piG * rho_bar * a / k^2 * delta(k)

For leapfrog we fold all prefactors into the acceleration:
    a_particle = -grad(phi) / a^2
"""

import numpy as np


def cic_paint(pos, N_mesh, L):
    """
    Cloud-In-Cell (CIC) mass assignment.
    Paints particle positions onto an N_mesh^3 density grid.

    Parameters
    ----------
    pos : ndarray, shape (3, N_particles)
        Particle positions in Mpc/h, assumed in [-L/2, L/2].
    N_mesh : int
        Number of mesh cells per side.
    L : float
        Box side length in Mpc/h.

    Returns
    -------
    delta : ndarray, shape (N_mesh, N_mesh, N_mesh)
        Overdensity field (n/n_bar - 1).
    """
    N_particles = pos.shape[1]
    delta = np.zeros((N_mesh, N_mesh, N_mesh), dtype=np.float64)

    dx = L / N_mesh
    # Map positions to [0, N_mesh)
    ijk = (pos + L / 2) / dx    # float cell indices

    for p in range(N_particles):
        i0 = int(ijk[0, p]) % N_mesh
        j0 = int(ijk[1, p]) % N_mesh
        k0 = int(ijk[2, p]) % N_mesh
        i1 = (i0 + 1) % N_mesh
        j1 = (j0 + 1) % N_mesh
        k1 = (k0 + 1) % N_mesh

        dx_p = ijk[0, p] - int(ijk[0, p])
        dy_p = ijk[1, p] - int(ijk[1, p])
        dz_p = ijk[2, p] - int(ijk[2, p])

        delta[i0, j0, k0] += (1 - dx_p) * (1 - dy_p) * (1 - dz_p)
        delta[i1, j0, k0] += dx_p       * (1 - dy_p) * (1 - dz_p)
        delta[i0, j1, k0] += (1 - dx_p) * dy_p       * (1 - dz_p)
        delta[i0, j0, k1] += (1 - dx_p) * (1 - dy_p) * dz_p
        delta[i1, j1, k0] += dx_p       * dy_p       * (1 - dz_p)
        delta[i1, j0, k1] += dx_p       * (1 - dy_p) * dz_p
        delta[i0, j1, k1] += (1 - dx_p) * dy_p       * dz_p
        delta[i1, j1, k1] += dx_p       * dy_p       * dz_p

    # Convert to overdensity
    n_bar = N_particles / N_mesh**3
    delta = delta / n_bar - 1.0

    return delta


def cic_paint_vectorized(pos, N_mesh, L):
    """
    Vectorized CIC paint — much faster than the loop version for large N.
    Use this for production runs.
    """
    N_particles = pos.shape[1]
    dx = L / N_mesh

    ijk = (pos + L / 2) / dx   # (3, N_particles)

    i0 = np.floor(ijk[0]).astype(int) % N_mesh
    j0 = np.floor(ijk[1]).astype(int) % N_mesh
    k0 = np.floor(ijk[2]).astype(int) % N_mesh
    i1 = (i0 + 1) % N_mesh
    j1 = (j0 + 1) % N_mesh
    k1 = (k0 + 1) % N_mesh

    dx_p = ijk[0] - np.floor(ijk[0])
    dy_p = ijk[1] - np.floor(ijk[1])
    dz_p = ijk[2] - np.floor(ijk[2])

    delta = np.zeros((N_mesh, N_mesh, N_mesh), dtype=np.float64)

    weights = [
        ((i0, j0, k0), (1-dx_p)*(1-dy_p)*(1-dz_p)),
        ((i1, j0, k0), dx_p    *(1-dy_p)*(1-dz_p)),
        ((i0, j1, k0), (1-dx_p)*dy_p    *(1-dz_p)),
        ((i0, j0, k1), (1-dx_p)*(1-dy_p)*dz_p    ),
        ((i1, j1, k0), dx_p    *dy_p    *(1-dz_p)),
        ((i1, j0, k1), dx_p    *(1-dy_p)*dz_p    ),
        ((i0, j1, k1), (1-dx_p)*dy_p    *dz_p    ),
        ((i1, j1, k1), dx_p    *dy_p    *dz_p    ),
    ]

    for (ii, jj, kk), w in weights:
        np.add.at(delta, (ii, jj, kk), w)

    n_bar = N_particles / N_mesh**3
    delta = delta / n_bar - 1.0

    return delta


def compute_forces(delta, N_mesh, L, h, Omega_m, a):
    """
    Compute gravitational forces from density field via FFT Poisson solve.

    Parameters
    ----------
    delta : ndarray, shape (N_mesh, N_mesh, N_mesh)
        Overdensity field.
    N_mesh : int
        Mesh size per side.
    L : float
        Box side length in Mpc/h.
    h, Omega_m : float
        Cosmological parameters.
    a : float
        Scale factor.

    Returns
    -------
    forces : ndarray, shape (3, N_mesh, N_mesh, N_mesh)
        Gravitational force field (acceleration) in Mpc/h per (km/s)^2 * (Mpc/h).
    """
    dk = 2 * np.pi / L
    k1d = np.fft.fftfreq(N_mesh, d=1.0 / N_mesh) * dk
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0   # avoid division by zero

    # FFT density field
    delta_k = np.fft.fftn(delta)

    # Poisson equation: phi_k = -4piG * rho_bar * a * delta_k / k^2
    # In comoving units with H0=100h km/s/Mpc:
    # 4piG * rho_bar = 3/2 * Omega_m * H0^2 (in proper coordinates)
    # Comoving acceleration: g = -grad(phi)/a^2 = 3/2 * Omega_m * H0^2 / a * delta_k * (-ik)/k^2
    H0 = 100.0   # km/s / (Mpc/h), so H0*h in km/s/Mpc
    prefactor = 1.5 * Omega_m * H0**2 / a

    phi_k = -prefactor * delta_k / k2
    phi_k[0, 0, 0] = 0.0

    forces = []
    for ki in [kx, ky, kz]:
        F_k = -1j * ki * phi_k
        F_i = np.real(np.fft.ifftn(F_k))
        forces.append(F_i)

    return np.array(forces)


def cic_interpolate(forces, pos, N_mesh, L):
    """
    Interpolate force field back to particle positions using CIC.

    Parameters
    ----------
    forces : ndarray, shape (3, N_mesh, N_mesh, N_mesh)
        Force field on the mesh.
    pos : ndarray, shape (3, N_particles)
        Particle positions in Mpc/h.

    Returns
    -------
    f_particles : ndarray, shape (3, N_particles)
        Forces at each particle position.
    """
    N_particles = pos.shape[1]
    dx = L / N_mesh

    ijk = (pos + L / 2) / dx
    i0 = np.floor(ijk[0]).astype(int) % N_mesh
    j0 = np.floor(ijk[1]).astype(int) % N_mesh
    k0 = np.floor(ijk[2]).astype(int) % N_mesh
    i1 = (i0 + 1) % N_mesh
    j1 = (j0 + 1) % N_mesh
    k1 = (k0 + 1) % N_mesh

    dx_p = ijk[0] - np.floor(ijk[0])
    dy_p = ijk[1] - np.floor(ijk[1])
    dz_p = ijk[2] - np.floor(ijk[2])

    f_particles = (
        forces[:, i0, j0, k0] * ((1-dx_p)*(1-dy_p)*(1-dz_p)) +
        forces[:, i1, j0, k0] * (dx_p    *(1-dy_p)*(1-dz_p)) +
        forces[:, i0, j1, k0] * ((1-dx_p)*dy_p    *(1-dz_p)) +
        forces[:, i0, j0, k1] * ((1-dx_p)*(1-dy_p)*dz_p    ) +
        forces[:, i1, j1, k0] * (dx_p    *dy_p    *(1-dz_p)) +
        forces[:, i1, j0, k1] * (dx_p    *(1-dy_p)*dz_p    ) +
        forces[:, i0, j1, k1] * ((1-dx_p)*dy_p    *dz_p    ) +
        forces[:, i1, j1, k1] * (dx_p    *dy_p    *dz_p    )
    )

    return f_particles


def compute_particle_forces(pos, N_mesh, L, h, Omega_m, a):
    """
    Full PM force pipeline: paint -> FFT Poisson solve -> interpolate.

    Returns
    -------
    f_particles : ndarray, shape (3, N_particles)
        Gravitational accelerations at each particle in (km/s)^2 / (Mpc/h).
    delta : ndarray, shape (N_mesh, N_mesh, N_mesh)
        Density field (returned for diagnostics/animation).
    """
    delta = cic_paint_vectorized(pos, N_mesh, L)
    forces = compute_forces(delta, N_mesh, L, h, Omega_m, a)
    f_particles = cic_interpolate(forces, pos, N_mesh, L)
    return f_particles, delta
