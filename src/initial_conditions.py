"""
initial_conditions.py
---------------------
Zel'dovich Approximation (ZA) for 3D cosmological initial conditions.

The ZA displaces particles from a uniform grid using the linear displacement
field Psi(k), computed from the density field via the Poisson equation in
Fourier space:

    Psi(k) = -i * k_hat / k * delta(k)

Particle positions:   x = x_grid + D(z) * Psi
Particle velocities:  v = a * H(z) * f * D(z) * Psi

where f = d ln D / d ln a ~ Omega_m(z)^0.55 is the linear growth rate.

References:
    Zel'dovich (1970)
    Dolag et al. (2005) — IC generation review
"""

import numpy as np
from pk_input import power_spectrum, growth_factor


def make_grid(N, L):
    """
    Create a uniform 3D particle grid with N^3 particles in a box of side L.

    Returns
    -------
    pos : ndarray, shape (3, N^3)
        Particle positions in Mpc/h, centered at box origin [-L/2, L/2].
    """
    dx = L / N
    x = np.arange(N) * dx - L / 2 + dx / 2
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    pos = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=0)
    return pos


def _k_grids(N, L):
    """
    Compute 3D k-space grids for an N^3 box of side L.

    Returns kx, ky, kz grids and the scalar |k| grid.
    """
    dk = 2 * np.pi / L
    k1d = np.fft.fftfreq(N, d=1.0 / N) * dk    # [0, dk, 2dk, ..., -dk]
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0    # avoid division by zero; will be zeroed later
    return kx, ky, kz, k2


def generate_gaussian_field(N, L, Pk_func, seed=42):
    """
    Generate a Gaussian random density field delta(x) with power spectrum P(k).

    Parameters
    ----------
    N : int
        Grid size per side.
    L : float
        Box side length in Mpc/h.
    Pk_func : callable
        P(k) function, takes k array in h/Mpc, returns (Mpc/h)^3.
    seed : int
        Random seed.

    Returns
    -------
    delta_k : ndarray, shape (N, N, N), complex
        Density field in Fourier space.
    """
    rng = np.random.default_rng(seed)

    kx, ky, kz, k2 = _k_grids(N, L)
    k = np.sqrt(k2)
    k[0, 0, 0] = 0.0

    Pk = np.zeros_like(k)
    mask = k > 0
    Pk[mask] = Pk_func(k[mask])

    # Volume element for normalization: <|delta_k|^2> = P(k) / V
    V = L**3
    amplitude = np.sqrt(Pk / (2 * V))

    # Draw complex Gaussian modes
    re = rng.standard_normal((N, N, N))
    im = rng.standard_normal((N, N, N))
    delta_k = amplitude * (re + 1j * im)
    delta_k[0, 0, 0] = 0.0   # zero mean

    return delta_k


def displacement_field(delta_k, N, L):
    """
    Compute the 3D Zel'dovich displacement field Psi from delta_k.

    Solves: Psi(k) = -i * k / k^2 * delta(k)

    Returns
    -------
    Psi : ndarray, shape (3, N^3)
        Displacement vectors for each particle.
    """
    kx, ky, kz, k2 = _k_grids(N, L)

    Psi = []
    for ki in [kx, ky, kz]:
        Psi_k = -1j * ki / k2 * delta_k
        Psi_k[0, 0, 0] = 0.0
        Psi_i = np.real(np.fft.ifftn(Psi_k))
        Psi.append(Psi_i.ravel())

    return np.array(Psi)


def hubble(z, h, Omega_m, Omega_lambda=None):
    """
    Hubble parameter H(z) in km/s/(Mpc/h), i.e. returns H(z)/H0.
    """
    if Omega_lambda is None:
        Omega_lambda = 1 - Omega_m
    a = 1 / (1 + z)
    E2 = Omega_m / a**3 + Omega_lambda
    return np.sqrt(E2)   # H(z) = H0 * E(z), returns E(z)


def growth_rate(z, Omega_m):
    """
    Linear growth rate f = d ln D / d ln a ~ Omega_m(z)^0.55.
    """
    Omega_lambda = 1 - Omega_m
    a = 1 / (1 + z)
    Omega_m_z = Omega_m / a**3 / (Omega_m / a**3 + Omega_lambda)
    return Omega_m_z**0.55


def make_ics(N, L, h, Omega_m, Omega_b, n_s, sigma8,
             z_initial=49.0, seed=42):
    """
    Generate Zel'dovich initial conditions at redshift z_initial.

    Parameters
    ----------
    N : int
        Number of particles per side (N^3 total).
    L : float
        Box side length in Mpc/h.
    z_initial : float
        Starting redshift (typically 49 or 99).
    seed : int
        Random seed.

    Returns
    -------
    pos : ndarray, shape (3, N^3)
        Initial particle positions in Mpc/h.
    vel : ndarray, shape (3, N^3)
        Initial particle velocities in km/s.
    delta_k : ndarray, shape (N, N, N)
        Initial density field in Fourier space (for diagnostics).
    """
    print(f"Generating Zel'dovich ICs: N={N}, L={L} Mpc/h, z_init={z_initial}")

    # Growth factor at z_initial (normalized to 1 at z=0)
    D_init = growth_factor(z_initial, Omega_m)

    # P(k) at z=0, then scale to z_initial
    def Pk_func(k):
        return power_spectrum(k, h, Omega_m, Omega_b, n_s, sigma8, z=0.0)

    # Generate density field
    delta_k = generate_gaussian_field(N, L, Pk_func, seed=seed)
    delta_k_init = delta_k * D_init    # scale to z_initial

    # Displacement field
    Psi = displacement_field(delta_k_init, N, L)

    # Particle grid positions
    pos_grid = make_grid(N, L)

    # Zel'dovich displacement
    pos = pos_grid + Psi

    # Velocities: v = a * H * f * D * Psi_0
    # In units where H0=100h km/s/Mpc, a=1/(1+z)
    a_init = 1 / (1 + z_initial)
    Hz = hubble(z_initial, h, Omega_m) * 100   # H(z) in km/s/Mpc
    f = growth_rate(z_initial, Omega_m)

    # Psi is already scaled by D_init, so vel = a * H * f * Psi
    vel = a_init * Hz * f * Psi

    # Apply periodic boundary conditions
    pos = pos % L - L / 2

    print(f"  D(z_init={z_initial:.0f}) = {D_init:.4f}")
    print(f"  f(z_init={z_initial:.0f}) = {f:.4f}")
    print(f"  Position range: [{pos.min():.1f}, {pos.max():.1f}] Mpc/h")
    print(f"  Velocity range: [{vel.min():.1f}, {vel.max():.1f}] km/s")

    return pos, vel, delta_k


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from power_spectrum import estimate_pk

    h, Omega_m, Omega_b, n_s, sigma8 = 0.6736, 0.3153, 0.0493, 0.9649, 0.8111
    N, L = 64, 1500.0

    pos, vel, delta_k = make_ics(N, L, h, Omega_m, Omega_b, n_s, sigma8,
                                  z_initial=49.0, seed=42)

    # Measure P(k) from ICs and compare to input
    k_out, Pk_out, _ = estimate_pk(pos, N, L, n_mesh=128)
    k_in = np.logspace(-3, 0, 300)
    Pk_in = power_spectrum(k_in, h, Omega_m, Omega_b, n_s, sigma8, z=49.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(k_in, Pk_in, 'k--', label='Input P(k) at z=49')
    ax.loglog(k_out, Pk_out, label='Measured from ICs')
    ax.set_xlabel(r'$k$ [$h$/Mpc]')
    ax.set_ylabel(r'$P(k)$ [$(h^{-1}$Mpc$)^3$]')
    ax.set_title('Zel\'dovich ICs: Input vs Measured P(k)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/figures/ic_pk_check.png', dpi=150)
    plt.show()
