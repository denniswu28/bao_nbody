"""
lognormal.py
------------
3D lognormal galaxy catalog generator.

The lognormal model approximates the galaxy density field as:
    1 + delta_LN(x) = exp(delta_G(x) - sigma_G^2/2)

where delta_G is a Gaussian random field. The Gaussian power spectrum P_G(k)
is related to the target galaxy power spectrum P(k) via the log transform of
the correlation function:

    xi_G(r) = ln(1 + xi(r))

In practice, we convert in Fourier space iteratively or via the Fourier
transform of ln(1 + xi(r)).

Galaxy positions are then Poisson-sampled from n(x) = nbar * (1 + delta_LN(x)).

References:
    Coles & Jones (1991)
    Agrawal et al. (2017) — https://arxiv.org/abs/1706.09471
"""

import numpy as np


def pk_to_xi(k, Pk, r_grid):
    """
    Compute the correlation function xi(r) from P(k) via Hankel transform.
    Uses the 3D isotropic relation:
        xi(r) = 1/(2pi^2) * integral dk k^2 P(k) sinc(kr)
    """
    xi = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r == 0:
            integrand = k**2 * Pk / (2 * np.pi**2)
        else:
            integrand = k**2 * Pk * np.sinc(k * r / np.pi) / (2 * np.pi**2)
        xi[i] = np.trapz(integrand, k)
    return xi


def xi_to_pk(r, xi, k_grid):
    """
    Compute P(k) from xi(r) via inverse Hankel transform.
        P(k) = 4pi * integral dr r^2 xi(r) sinc(kr)
    """
    Pk = np.zeros_like(k_grid)
    for i, ki in enumerate(k_grid):
        integrand = r**2 * xi * np.sinc(ki * r / np.pi) * 4 * np.pi
        Pk[i] = np.trapz(integrand, r)
    return Pk


def galaxy_pk_to_gaussian_pk(k, Pk_galaxy, N_mesh, L):
    """
    Convert target galaxy P(k) to the Gaussian P_G(k) needed for the
    lognormal field generator.

    The relation is: xi_G(r) = ln(1 + xi(r))

    Parameters
    ----------
    k, Pk_galaxy : arrays
        Input galaxy power spectrum.
    N_mesh : int
        Mesh resolution for internal FFT operations.
    L : float
        Box size.

    Returns
    -------
    k_out, Pk_G : arrays
        Gaussian power spectrum on the same k grid.
    """
    # Compute r grid
    r_grid = np.linspace(0, L / 2, 500)
    r_grid[0] = 1e-3   # avoid r=0

    # P(k) -> xi(r)
    xi = pk_to_xi(k, Pk_galaxy, r_grid)

    # Log transform: xi_G = ln(1 + xi)
    xi_G = np.log(1 + np.clip(xi, -0.999, None))

    # xi_G(r) -> P_G(k)
    Pk_G = xi_to_pk(r_grid, xi_G, k)

    return k, np.abs(Pk_G)


def generate_lognormal_field(N_mesh, L, Pk_G_func, seed=42):
    """
    Generate a 3D lognormal density field.

    Parameters
    ----------
    N_mesh : int
        Grid resolution per side.
    L : float
        Box side length in Mpc/h.
    Pk_G_func : callable
        Gaussian power spectrum P_G(k) function.
    seed : int
        Random seed.

    Returns
    -------
    delta_LN : ndarray, shape (N_mesh, N_mesh, N_mesh)
        Lognormal overdensity field.
    """
    from initial_conditions import generate_gaussian_field

    # Generate Gaussian random field with P_G(k)
    delta_k = generate_gaussian_field(N_mesh, L, Pk_G_func, seed=seed)
    delta_G = np.real(np.fft.ifftn(delta_k)) * N_mesh**3

    # Variance of Gaussian field
    sigma_G2 = np.var(delta_G)

    # Lognormal transform
    delta_LN = np.exp(delta_G - sigma_G2 / 2) - 1.0

    return delta_LN


def poisson_sample(delta_LN, nbar, L, seed=42):
    """
    Poisson-sample galaxy positions from lognormal density field.

    Parameters
    ----------
    delta_LN : ndarray, shape (N_mesh, N_mesh, N_mesh)
        Lognormal overdensity field.
    nbar : float
        Mean galaxy number density in (h/Mpc)^3.
    L : float
        Box side length in Mpc/h.
    seed : int
        Random seed.

    Returns
    -------
    pos : ndarray, shape (3, N_galaxies)
        Galaxy positions in Mpc/h.
    """
    rng = np.random.default_rng(seed)
    N_mesh = delta_LN.shape[0]
    dV = (L / N_mesh)**3

    # Expected number of galaxies per voxel
    n_expected = nbar * dV * (1 + delta_LN)
    n_expected = np.clip(n_expected, 0, None)

    # Draw Poisson counts
    n_gals = rng.poisson(n_expected)

    # Get voxel centers
    dx = L / N_mesh
    x1d = np.arange(N_mesh) * dx - L / 2 + dx / 2
    xx, yy, zz = np.meshgrid(x1d, x1d, x1d, indexing='ij')

    # Scatter galaxies within each occupied voxel
    total = n_gals.sum()
    pos = np.zeros((3, total))
    idx = 0
    for i in range(N_mesh):
        for j in range(N_mesh):
            for k in range(N_mesh):
                n = n_gals[i, j, k]
                if n > 0:
                    pos[0, idx:idx+n] = xx[i,j,k] + rng.uniform(-dx/2, dx/2, n)
                    pos[1, idx:idx+n] = yy[i,j,k] + rng.uniform(-dx/2, dx/2, n)
                    pos[2, idx:idx+n] = zz[i,j,k] + rng.uniform(-dx/2, dx/2, n)
                    idx += n

    return pos


def poisson_sample_vectorized(delta_LN, nbar, L, seed=42):
    """
    Vectorized Poisson sampling — much faster for large N_mesh.
    """
    rng = np.random.default_rng(seed)
    N_mesh = delta_LN.shape[0]
    dV = (L / N_mesh)**3

    n_expected = nbar * dV * np.clip(1 + delta_LN, 0, None)
    n_gals = rng.poisson(n_expected)   # shape (N_mesh, N_mesh, N_mesh)

    dx = L / N_mesh
    x1d = np.arange(N_mesh) * dx - L / 2 + dx / 2
    xx, yy, zz = np.meshgrid(x1d, x1d, x1d, indexing='ij')

    # Repeat voxel centers by galaxy count
    total = n_gals.sum()
    cx = np.repeat(xx.ravel(), n_gals.ravel())
    cy = np.repeat(yy.ravel(), n_gals.ravel())
    cz = np.repeat(zz.ravel(), n_gals.ravel())

    # Add random offset within voxel
    pos = np.zeros((3, total))
    pos[0] = cx + rng.uniform(-dx/2, dx/2, total)
    pos[1] = cy + rng.uniform(-dx/2, dx/2, total)
    pos[2] = cz + rng.uniform(-dx/2, dx/2, total)

    return pos


def generate_lognormal_catalog(N_mesh, L, h, Omega_m, Omega_b, n_s, sigma8,
                                nbar, b=1.0, z=0.38, seed=42):
    """
    Full lognormal catalog generation pipeline.

    Parameters
    ----------
    N_mesh : int
        Grid resolution.
    L : float
        Box side length in Mpc/h.
    nbar : float
        Target mean galaxy number density in (h/Mpc)^3.
    b : float
        Linear galaxy bias.
    z : float
        Effective redshift.
    seed : int
        Random seed.

    Returns
    -------
    pos : ndarray, shape (3, N_galaxies)
        Galaxy positions in Mpc/h.
    delta_LN : ndarray, shape (N_mesh, N_mesh, N_mesh)
        Underlying density field.
    """
    from pk_input import power_spectrum

    print(f"Generating lognormal catalog: N_mesh={N_mesh}, L={L} Mpc/h, nbar={nbar:.1e}")

    # Galaxy power spectrum: P_gal(k) = b^2 * P_matter(k)
    k_arr = np.logspace(-3, 1, 500)
    Pk_matter = power_spectrum(k_arr, h, Omega_m, Omega_b, n_s, sigma8, z=z)
    Pk_galaxy = b**2 * Pk_matter

    # Convert to Gaussian P_G(k)
    print("  Converting galaxy P(k) -> Gaussian P_G(k)...")
    _, Pk_G = galaxy_pk_to_gaussian_pk(k_arr, Pk_galaxy, N_mesh, L)

    # Interpolate P_G to a callable
    from scipy.interpolate import interp1d
    Pk_G_interp = interp1d(k_arr, Pk_G, bounds_error=False, fill_value=0.0)

    # Generate lognormal field
    print("  Generating lognormal density field...")
    delta_LN = generate_lognormal_field(N_mesh, L, Pk_G_interp, seed=seed)

    # Poisson sample galaxies
    print("  Poisson sampling galaxy positions...")
    pos = poisson_sample_vectorized(delta_LN, nbar, L, seed=seed+1)

    print(f"  Generated {pos.shape[1]} galaxies  (expected {nbar * L**3:.0f})")
    return pos, delta_LN


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from power_spectrum import estimate_pk
    from pk_input import power_spectrum

    h, Omega_m, Omega_b, n_s, sigma8 = 0.6736, 0.3153, 0.0493, 0.9649, 0.8111
    N_mesh, L = 64, 1500.0
    nbar, b, z = 3e-4, 1.5, 0.38

    pos, delta_LN = generate_lognormal_catalog(
        N_mesh, L, h, Omega_m, Omega_b, n_s, sigma8, nbar, b=b, z=z, seed=42)

    # Measure P(k)
    k_out, Pk_out, _ = estimate_pk(pos, N_mesh, L, n_mesh=N_mesh*2)

    # Theory
    k_th = np.logspace(-3, 0, 300)
    Pk_th = b**2 * power_spectrum(k_th, h, Omega_m, Omega_b, n_s, sigma8, z=z)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].loglog(k_th, Pk_th, 'k--', label='Theory')
    axes[0].loglog(k_out, Pk_out, label='Lognormal catalog')
    axes[0].set_xlabel(r'$k$ [$h$/Mpc]')
    axes[0].set_ylabel(r'$P(k)$ [$(h^{-1}$Mpc$)^3$]')
    axes[0].set_title('Lognormal P(k) vs Theory')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].imshow(delta_LN.sum(axis=2), cmap='inferno', origin='lower')
    axes[1].set_title('Lognormal density field (projected)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    plt.tight_layout()
    plt.savefig('outputs/figures/lognormal_test.png', dpi=150)
    plt.show()
