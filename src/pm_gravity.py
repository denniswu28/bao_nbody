"""
pm_gravity.py
-------------
Particle-Mesh gravity solver. CIC paint, FFT Poisson, CIC interpolate to get accelerations on each particle.
"""

import numpy as np


def cic_paint(pos, N_mesh, L):
    # python-loop CIC
    N_particles = pos.shape[1]
    delta = np.zeros((N_mesh, N_mesh, N_mesh), dtype=np.float64)

    dx = L / N_mesh
    ijk = (pos + L / 2) / dx

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

    n_bar = N_particles / N_mesh**3
    delta = delta / n_bar - 1.0

    return delta


def cic_paint_vectorized(pos, N_mesh, L):
    # vectorized version
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
    dk = 2 * np.pi / L
    k1d = np.fft.fftfreq(N_mesh, d=1.0 / N_mesh) * dk
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0

    delta_k = np.fft.fftn(delta)

    # comoving poisson
    H0 = 100.0
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
    # full pipeline
    delta = cic_paint_vectorized(pos, N_mesh, L)
    forces = compute_forces(delta, N_mesh, L, h, Omega_m, a)
    f_particles = cic_interpolate(forces, pos, N_mesh, L)
    return f_particles, delta
