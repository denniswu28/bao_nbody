"""
power_spectrum.py
-----------------
FFT-based P(k) and xi(r) estimators from particle positions.
Includes CIC window deconvolution and shot-noise subtraction.
"""

import numpy as np


def cic_window_correction_1d(k_component, dx):
    # 1d CIC window
    return np.sinc(k_component * dx / (2 * np.pi))**2


def estimate_pk(pos, N, L, n_mesh=None, subtract_shotnoise=True):
    if n_mesh is None:
        n_mesh = 2 * N

    N_particles = pos.shape[1]
    V = L**3
    dx = L / n_mesh
    nbar = N_particles / V

    from pm_gravity import cic_paint_vectorized
    delta = cic_paint_vectorized(pos, n_mesh, L)

    delta_k = np.fft.fftn(delta) / n_mesh**3

    dk = 2 * np.pi / L
    k1d = np.fft.fftfreq(n_mesh, d=1.0 / n_mesh) * dk
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    # subtract shot noise BEFORE CIC deconvolve, otherwise the noise floor
    Pk_raw = np.abs(delta_k)**2 * V
    if subtract_shotnoise:
        Pk_raw -= 1.0 / nbar

    W_cic = (cic_window_correction_1d(kx, dx)
             * cic_window_correction_1d(ky, dx)
             * cic_window_correction_1d(kz, dx))
    W_cic[0, 0, 0] = 1.0
    Pk_raw /= W_cic**2

    # spherical shell binning
    k_nyq = np.pi / dx
    k_min = dk
    n_bins = int(n_mesh / 2)
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
    nmodes_bins = counts[good]

    # cut below 0.9*Nyquist
    mask = (k_bins >= k_min) & (k_bins < 0.9 * k_nyq)
    return k_bins[mask], Pk_bins[mask], nmodes_bins[mask]


def pk_from_snapshot(snap, N, L, n_mesh=None):
    return estimate_pk(snap['pos'], N, L, n_mesh=n_mesh)


def estimate_xi(pos, N, L, n_mesh=None, r_max=200.0, n_bins=50):
    # FFT route to xi(r)
    if n_mesh is None:
        n_mesh = 2 * N

    N_particles = pos.shape[1]
    V = L**3
    dx = L / n_mesh
    nbar = N_particles / V

    from pm_gravity import cic_paint_vectorized
    delta = cic_paint_vectorized(pos, n_mesh, L)

    return _xi_from_delta_grid(delta, L, n_mesh, nbar, r_max, n_bins)


def estimate_xi_from_delta(delta, L, nbar=None, r_max=200.0, n_bins=50):
    # same as estimate_xi but takes a precomputed overdensity grid
    n_mesh = delta.shape[0]
    return _xi_from_delta_grid(delta, L, n_mesh, nbar, r_max, n_bins)


def _xi_from_delta_grid(delta, L, n_mesh, nbar, r_max, n_bins):
    V = L**3
    dx = L / n_mesh

    delta_k = np.fft.fftn(delta)

    kfreq = np.fft.fftfreq(n_mesh, d=1.0 / n_mesh) * (2 * np.pi / L)
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    W_cic = (cic_window_correction_1d(kx, dx)
             * cic_window_correction_1d(ky, dx)
             * cic_window_correction_1d(kz, dx))
    W_cic[0, 0, 0] = 1.0

    # subtract shot noise before CIC deconvolution
    Pk_grid = np.abs(delta_k)**2 / n_mesh**6
    if nbar is not None:
        Pk_grid -= 1.0 / (nbar * V)
    Pk_grid /= W_cic**2
    xi_grid = np.fft.ifftn(Pk_grid).real * n_mesh**3

    x1d = np.fft.fftfreq(n_mesh, d=1.0/n_mesh) * dx
    rx, ry, rz = np.meshgrid(x1d, x1d, x1d, indexing='ij')
    r_mag = np.sqrt(rx**2 + ry**2 + rz**2)

    r_edges = np.linspace(0, r_max, n_bins + 1)

    r_flat = r_mag.ravel()
    xi_flat = xi_grid.ravel()

    bin_idx = np.digitize(r_flat, r_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    bi = bin_idx[valid]
    counts = np.bincount(bi, minlength=n_bins)
    r_sum = np.bincount(bi, weights=r_flat[valid], minlength=n_bins)
    xi_sum = np.bincount(bi, weights=xi_flat[valid], minlength=n_bins)

    good = counts > 0
    r_bins = r_sum[good] / counts[good]
    xi_bins = xi_sum[good] / counts[good]
    npairs_bins = counts[good]

    return r_bins, xi_bins, npairs_bins


def xi_from_pk(k, Pk, r_grid):
    # spherical Hankel transform
    xi = np.zeros_like(r_grid, dtype=float)
    k_max = k[-1]
    k_taper = 0.8 * k_max
    taper = np.exp(-0.5 * np.clip((k - k_taper) / (0.1 * k_max), 0, None)**2)

    for i, r in enumerate(r_grid):
        integrand = k**2 * Pk * taper * np.sinc(k * r / np.pi) / (2 * np.pi**2)
        xi[i] = np.trapezoid(integrand, k)
    return xi


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pk_input import power_spectrum

    N = 64
    L = 1500.0
    rng = np.random.default_rng(42)
    pos = rng.uniform(-L/2, L/2, (3, N**3))

    k, Pk, nmodes = estimate_pk(pos, N, L)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(k, Pk, label='Measured P(k) (uniform random)')
    ax.axhline(1 / (N**3 / L**3), color='r', ls='--', label='Expected shot noise 1/nbar')
    ax.set_xlabel(r'$k$ [$h$/Mpc]')
    ax.set_ylabel(r'$P(k)$ [$(h^{-1}$Mpc$)^3$]')
    ax.set_title('P(k) estimator test: uniform random catalog')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/figures/pk_estimator_test.png', dpi=150)
    plt.show()
