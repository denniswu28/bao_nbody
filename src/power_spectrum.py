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


def cic_window_correction(k, dx):
    """
    CIC mass assignment window function in Fourier space.
    W(k) = [sinc(k_x * dx/2) * sinc(k_y * dx/2) * sinc(k_z * dx/2)]^2
    For isotropic correction we use the approximate scalar form.
    """
    x = k * dx / 2
    # np.sinc is normalized: sinc(x) = sin(pi*x)/(pi*x)
    W = np.sinc(x / np.pi)**2    # un-normalized sinc
    return W


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

    # Power |delta_k|^2 * V, corrected for CIC window
    Pk_raw = np.abs(delta_k)**2 * V
    W = cic_window_correction(k_mag, dx)**2   # CIC applied twice (paint + interp)
    W[0, 0, 0] = 1.0
    Pk_raw /= W

    # Subtract shot noise
    if subtract_shotnoise:
        Pk_raw -= 1.0 / nbar

    # Bin in spherical shells
    k_nyq = np.pi / dx
    k_min = dk
    n_bins = int(n_mesh / 2)
    k_edges = np.linspace(0, k_nyq, n_bins + 1)

    k_bins = []
    Pk_bins = []
    nmodes_bins = []

    k_flat = k_mag.ravel()
    Pk_flat = Pk_raw.ravel()

    for i in range(n_bins):
        mask = (k_flat >= k_edges[i]) & (k_flat < k_edges[i+1])
        nmodes = mask.sum()
        if nmodes > 0:
            k_bins.append(k_flat[mask].mean())
            Pk_bins.append(Pk_flat[mask].mean())
            nmodes_bins.append(nmodes)

    k_bins = np.array(k_bins)
    Pk_bins = np.array(Pk_bins)
    nmodes_bins = np.array(nmodes_bins)

    # Only return modes above k_min and below Nyquist
    mask = (k_bins >= k_min) & (k_bins < 0.9 * k_nyq)
    return k_bins[mask], Pk_bins[mask], nmodes_bins[mask]


def pk_from_snapshot(snap, N, L, n_mesh=None):
    """
    Convenience wrapper: estimate P(k) from an N-body snapshot dict.
    """
    return estimate_pk(snap['pos'], N, L, n_mesh=n_mesh)


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
