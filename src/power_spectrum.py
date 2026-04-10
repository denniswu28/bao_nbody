"""
power_spectrum.py
-----------------
FFT-based isotropic power spectrum estimator P(k) from particle positions.

Algorithm:
    1. Paint particles onto N_mesh^3 grid using CIC
    2. FFT -> delta(k)
    3. Correct for CIC window function: W(k) = sinc(k*dx/2)^2
    4. Bin |delta(k)|^2 / V in spherical k-shells
    5. Subtract shot noise 1/nbar
"""

import numpy as np


def cic_window_correction_1d(k_component, dx):
    """
    CIC window function along one axis:  sinc^2(k_i * dx / 2).

    np.sinc is normalized:  sinc(x) = sin(pi*x)/(pi*x),
    so sinc(k*dx/(2*pi))^2  gives the un-normalized sinc^2.
    """
    return np.sinc(k_component * dx / (2 * np.pi))**2


def estimate_pk(pos, N, L, n_mesh=None, subtract_shotnoise=True):
    """
    Estimate isotropic power spectrum P(k) from particle positions.

    Parameters
    ----------
    pos : ndarray, shape (3, N_particles)
        Particle positions in Mpc/h, in range [-L/2, L/2].
    N : int
        Number of particles per side (for shot noise: nbar = N^3/L^3).
    L : float
        Box side length in Mpc/h.
    n_mesh : int, optional
        Mesh resolution. Defaults to 2*N.
    subtract_shotnoise : bool
        Whether to subtract Poisson shot noise 1/nbar.

    Returns
    -------
    k_bins : ndarray
        Bin-center wavenumbers in h/Mpc.
    Pk : ndarray
        Power spectrum in (Mpc/h)^3.
    nmodes : ndarray
        Number of modes per k-bin.
    """
    if n_mesh is None:
        n_mesh = 2 * N

    N_particles = pos.shape[1]
    V = L**3
    dx = L / n_mesh
    nbar = N_particles / V

    # Paint particles onto mesh
    from pm_gravity import cic_paint_vectorized
    delta = cic_paint_vectorized(pos, n_mesh, L)

    # FFT
    delta_k = np.fft.fftn(delta) / n_mesh**3   # normalize

    # k grids
    dk = 2 * np.pi / L
    k1d = np.fft.fftfreq(n_mesh, d=1.0 / n_mesh) * dk
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    # Power |delta_k|^2 * V, corrected for CIC window (per-axis)
    # W_CIC(k) = prod sinc^2(k_i * dx/2);  divide by |W_CIC|^2 = W_CIC^2
    Pk_raw = np.abs(delta_k)**2 * V
    W_cic = (cic_window_correction_1d(kx, dx)
             * cic_window_correction_1d(ky, dx)
             * cic_window_correction_1d(kz, dx))
    W_cic[0, 0, 0] = 1.0
    Pk_raw /= W_cic**2

    # Subtract shot noise
    if subtract_shotnoise:
        Pk_raw -= 1.0 / nbar

    # Bin in spherical shells (vectorized)
    k_nyq = np.pi / dx
    k_min = dk
    n_bins = int(n_mesh / 2)
    k_edges = np.linspace(0, k_nyq, n_bins + 1)

    k_flat = k_mag.ravel()
    Pk_flat = Pk_raw.ravel()

    bin_idx = np.digitize(k_flat, k_edges) - 1  # 0-based bin index
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    bi = bin_idx[valid]
    counts = np.bincount(bi, minlength=n_bins)
    k_sum = np.bincount(bi, weights=k_flat[valid], minlength=n_bins)
    Pk_sum = np.bincount(bi, weights=Pk_flat[valid], minlength=n_bins)

    good = counts > 0
    k_bins = k_sum[good] / counts[good]
    Pk_bins = Pk_sum[good] / counts[good]
    nmodes_bins = counts[good]

    # Only return modes above k_min and below Nyquist
    mask = (k_bins >= k_min) & (k_bins < 0.9 * k_nyq)
    return k_bins[mask], Pk_bins[mask], nmodes_bins[mask]


def pk_from_snapshot(snap, N, L, n_mesh=None):
    """
    Convenience wrapper: estimate P(k) from an N-body snapshot dict.
    """
    return estimate_pk(snap['pos'], N, L, n_mesh=n_mesh)


def estimate_xi(pos, N, L, n_mesh=None, r_max=200.0, n_bins=50):
    """
    Estimate the isotropic two-point correlation function xi(r) via FFT.

    Algorithm:
        1. Paint particles to mesh via CIC → δ(x)
        2. FFT → δ(k)
        3. |δ(k)|² / W_CIC² → P_3D(k) on the grid
        4. IFFT of P_3D(k) → ξ(r) on the grid (Wiener-Khinchin theorem)
        5. Bin ξ(r) in spherical shells of |r|

    Parameters
    ----------
    pos : ndarray, shape (3, N_particles)
        Particle positions in Mpc/h, in range [-L/2, L/2].
    N : int
        Number of particles per side (for shot noise: nbar = N^3/L^3).
    L : float
        Box side length in Mpc/h.
    n_mesh : int, optional
        Mesh resolution. Defaults to 2*N.
    r_max : float
        Maximum separation in Mpc/h (should be < L/2).
    n_bins : int
        Number of radial bins.

    Returns
    -------
    r_bins : ndarray
        Bin-center separations in Mpc/h.
    xi_bins : ndarray
        Correlation function values.
    npairs : ndarray
        Number of grid-cell pairs per bin.
    """
    if n_mesh is None:
        n_mesh = 2 * N

    N_particles = pos.shape[1]
    V = L**3
    dx = L / n_mesh
    nbar = N_particles / V

    # Paint particles onto mesh
    from pm_gravity import cic_paint_vectorized
    delta = cic_paint_vectorized(pos, n_mesh, L)

    return _xi_from_delta_grid(delta, L, n_mesh, nbar, r_max, n_bins)


def estimate_xi_from_delta(delta, L, nbar=None, r_max=200.0, n_bins=50):
    """
    Estimate xi(r) from a pre-computed overdensity grid delta(x).

    Useful for reconstructed fields where delta is already available.

    Parameters
    ----------
    delta : ndarray, shape (N_mesh, N_mesh, N_mesh)
        Overdensity field.
    L : float
        Box side length in Mpc/h.
    nbar : float, optional
        Mean number density for shot noise subtraction. If None, no subtraction.
    r_max, n_bins : float, int
        Binning parameters.

    Returns
    -------
    r_bins, xi_bins, npairs : ndarrays
    """
    n_mesh = delta.shape[0]
    return _xi_from_delta_grid(delta, L, n_mesh, nbar, r_max, n_bins)


def _xi_from_delta_grid(delta, L, n_mesh, nbar, r_max, n_bins):
    """
    Core routine: compute xi(r) from a density grid via FFT.
    """
    V = L**3
    dx = L / n_mesh

    delta_k = np.fft.fftn(delta)

    # CIC window correction
    kfreq = np.fft.fftfreq(n_mesh, d=1.0 / n_mesh) * (2 * np.pi / L)
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    W_cic = (cic_window_correction_1d(kx, dx)
             * cic_window_correction_1d(ky, dx)
             * cic_window_correction_1d(kz, dx))
    W_cic[0, 0, 0] = 1.0

    # Power on the full grid, CIC-corrected
    Pk_grid = np.abs(delta_k)**2 / n_mesh**6 / W_cic**2

    # Subtract shot noise (constant in k-space)
    if nbar is not None:
        Pk_grid -= 1.0 / (nbar * V)

    # IFFT → xi(r):
    #   Pk_grid = |FFT(δ)|² / N^6 / W² = P(k) / V
    #   ξ(r_m) = (1/V) Σ_k P(k_n) e^{ik_n·r_m}
    #          = Σ_k Pk_grid[k_n] e^{2πi n·m/N}
    #          = N^3 * numpy.IFFT(Pk_grid)[m]
    xi_grid = np.fft.ifftn(Pk_grid).real * n_mesh**3

    # Build r-grid for binning
    x1d = np.fft.fftfreq(n_mesh, d=1.0/n_mesh) * dx
    rx, ry, rz = np.meshgrid(x1d, x1d, x1d, indexing='ij')
    r_mag = np.sqrt(rx**2 + ry**2 + rz**2)

    # Bin in spherical shells (vectorized)
    r_edges = np.linspace(0, r_max, n_bins + 1)

    r_flat = r_mag.ravel()
    xi_flat = xi_grid.ravel()

    bin_idx = np.digitize(r_flat, r_edges) - 1  # 0-based bin index
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
    """
    Compute xi(r) from binned P(k) via discrete spherical Hankel transform.

        xi(r) = 1/(2 pi^2) int dk k^2 P(k) sin(kr)/(kr)

    Useful for computing theory xi(r) from an analytic or tabulated P(k).
    To reduce ringing from noisy/truncated P(k), applies a Gaussian taper
    exp(-k^2 / (2 * k_max^2)) near the upper integration limit.

    Parameters
    ----------
    k : ndarray
        Wavenumbers in h/Mpc.
    Pk : ndarray
        Power spectrum in (Mpc/h)^3.
    r_grid : ndarray
        Separations in Mpc/h at which to evaluate xi(r).

    Returns
    -------
    xi : ndarray
        Correlation function at each r.
    """
    xi = np.zeros_like(r_grid, dtype=float)
    # Gaussian taper: damp the last ~20% of k range to reduce ringing
    k_max = k[-1]
    k_taper = 0.8 * k_max
    taper = np.exp(-0.5 * np.clip((k - k_taper) / (0.1 * k_max), 0, None)**2)

    for i, r in enumerate(r_grid):
        # np.sinc(x) = sin(pi*x)/(pi*x), so sinc(kr/pi) = sin(kr)/(kr)
        integrand = k**2 * Pk * taper * np.sinc(k * r / np.pi) / (2 * np.pi**2)
        xi[i] = np.trapezoid(integrand, k)
    return xi


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pk_input import power_spectrum

    # Quick test: generate random particles and check P(k) ~ const (Poisson)
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
