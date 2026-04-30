"""
lognormal.py
------------
Lognormal galaxy mocks: 1+delta_LN = exp(delta_G - sigma_G^2/2).
Galaxies are Poisson-sampled from nbar*(1+delta_LN). The Gaussian P_G(k)
is obtained from the target P(k) via xi_G = ln(1+xi).
"""

import numpy as np


def pk_to_xi(k, Pk, r_grid):
    # hankel transform from P(k) to xi(r)
    xi = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r == 0:
            integrand = k**2 * Pk / (2 * np.pi**2)
        else:
            integrand = k**2 * Pk * np.sinc(k * r / np.pi) / (2 * np.pi**2)
        xi[i] = np.trapezoid(integrand, k)
    return xi


def xi_to_pk(r, xi, k_grid):
    # inverse hankel from xi(r) back to P(k)
    Pk = np.zeros_like(k_grid)
    for i, ki in enumerate(k_grid):
        integrand = r**2 * xi * np.sinc(ki * r / np.pi) * 4 * np.pi
        Pk[i] = np.trapezoid(integrand, r)
    return Pk


def galaxy_pk_to_gaussian_pk(k, Pk_galaxy, N_mesh, L):
    # P_gal(k) -> xi(r) -> xi_G = ln(1+xi) -> P_G(k)
    r_grid = np.linspace(0, L / 2, 500)
    r_grid[0] = 1e-3

    xi = pk_to_xi(k, Pk_galaxy, r_grid)
    xi_G = np.log(1 + np.clip(xi, -0.999, None))
    Pk_G = xi_to_pk(r_grid, xi_G, k)

    return k, np.abs(Pk_G)


def generate_lognormal_field(N_mesh, L, Pk_G_func, seed=42):
    from initial_conditions import generate_gaussian_field

    delta_k = generate_gaussian_field(N_mesh, L, Pk_G_func, seed=seed)
    delta_G = np.real(np.fft.ifftn(delta_k))

    sigma_G2 = np.var(delta_G)
    delta_LN = np.exp(delta_G - sigma_G2 / 2) - 1.0

    return delta_LN


def poisson_sample(delta_LN, nbar, L, seed=42):
    # python-loop version, kept for clarity
    rng = np.random.default_rng(seed)
    N_mesh = delta_LN.shape[0]
    dV = (L / N_mesh)**3

    n_expected = nbar * dV * (1 + delta_LN)
    n_expected = np.clip(n_expected, 0, None)

    n_gals = rng.poisson(n_expected)

    # voxel centers
    dx = L / N_mesh
    x1d = np.arange(N_mesh) * dx - L / 2 + dx / 2
    xx, yy, zz = np.meshgrid(x1d, x1d, x1d, indexing='ij')

    # scatter galaxies inside their voxel
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
    rng = np.random.default_rng(seed)
    N_mesh = delta_LN.shape[0]
    dV = (L / N_mesh)**3

    n_expected = nbar * dV * np.clip(1 + delta_LN, 0, None)
    n_gals = rng.poisson(n_expected)

    dx = L / N_mesh
    x1d = np.arange(N_mesh) * dx - L / 2 + dx / 2
    xx, yy, zz = np.meshgrid(x1d, x1d, x1d, indexing='ij')

    total = n_gals.sum()
    cx = np.repeat(xx.ravel(), n_gals.ravel())
    cy = np.repeat(yy.ravel(), n_gals.ravel())
    cz = np.repeat(zz.ravel(), n_gals.ravel())

    pos = np.zeros((3, total))
    pos[0] = cx + rng.uniform(-dx/2, dx/2, total)
    pos[1] = cy + rng.uniform(-dx/2, dx/2, total)
    pos[2] = cz + rng.uniform(-dx/2, dx/2, total)

    return pos


def generate_lognormal_catalog(N_mesh, L, h, Omega_m, Omega_b, n_s, sigma8,
                                nbar, b=1.0, z=0.38, seed=42):
    from pk_input import power_spectrum

    print(f"Generating lognormal catalog: N_mesh={N_mesh}, L={L} Mpc/h, nbar={nbar:.1e}")

    # galaxy power = bias squared times matter power
    k_arr = np.logspace(-3, 1, 500)
    Pk_matter = power_spectrum(k_arr, h, Omega_m, Omega_b, n_s, sigma8, z=z)
    Pk_galaxy = b**2 * Pk_matter

    print("  Converting galaxy P(k) -> Gaussian P_G(k)...")
    _, Pk_G = galaxy_pk_to_gaussian_pk(k_arr, Pk_galaxy, N_mesh, L)

    from scipy.interpolate import interp1d
    Pk_G_interp = interp1d(k_arr, Pk_G, bounds_error=False, fill_value=0.0)

    print("  Generating lognormal density field...")
    delta_LN = generate_lognormal_field(N_mesh, L, Pk_G_interp, seed=seed)

    print("  Poisson sampling galaxy positions...")
    pos = poisson_sample_vectorized(delta_LN, nbar, L, seed=seed+1)

    print(f"  Generated {pos.shape[1]} galaxies  (expected {nbar * L**3:.0f})")
    return pos, delta_LN


def generate_mock_covariance(N_mocks, N_mesh, L, h, Omega_m, Omega_b,
                              n_s, sigma8, nbar, b=1.0, z=0.38,
                              seed_start=1000, k_max=0.3):
    # build the sample covariance of P(k) from N_mocks lognormal realizations
    from pk_input import power_spectrum as pk_func
    from power_spectrum import estimate_pk
    from scipy.interpolate import interp1d

    print(f"Generating {N_mocks} lognormal mocks for covariance matrix")
    print(f"  N_mesh={N_mesh}, L={L}, nbar={nbar:.1e}, b={b}, z={z}")

    # P_G(k) is the same for every mock
    k_arr = np.logspace(-3, 1, 500)
    Pk_matter = pk_func(k_arr, h, Omega_m, Omega_b, n_s, sigma8, z=z)
    Pk_galaxy = b**2 * Pk_matter
    _, Pk_G = galaxy_pk_to_gaussian_pk(k_arr, Pk_galaxy, N_mesh, L)
    Pk_G_interp = interp1d(k_arr, Pk_G, bounds_error=False, fill_value=0.0)

    Pk_all = None
    k_bins = None

    for i in range(N_mocks):
        seed = seed_start + i

        delta_LN = generate_lognormal_field(N_mesh, L, Pk_G_interp, seed=seed)
        pos = poisson_sample_vectorized(delta_LN, nbar, L, seed=seed + N_mocks)

        k, Pk, nmodes = estimate_pk(pos, N_mesh, L, n_mesh=N_mesh)

        mask = k <= k_max
        k = k[mask]
        Pk = Pk[mask]

        if Pk_all is None:
            k_bins = k
            Pk_all = np.zeros((N_mocks, len(k_bins)))

        Pk_all[i] = Pk

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Mock {i+1}/{N_mocks} done  (N_gal={pos.shape[1]})")

    Pk_mean = np.mean(Pk_all, axis=0)
    cov = np.cov(Pk_all, rowvar=False)

    # hartlap correction (Hartlap+ 2007) for unbiased inverse covariance
    N_bins = len(k_bins)
    hartlap = (N_mocks - N_bins - 2) / (N_mocks - 1)
    print(f"\nCovariance computed: {N_bins} k-bins from {N_mocks} mocks")
    print(f"  Hartlap correction factor: {hartlap:.3f}")
    if hartlap <= 0:
        print(f"  WARNING: N_mocks={N_mocks} < N_bins+2={N_bins+2}, "
              "covariance matrix is singular!")

    return k_bins, Pk_mean, cov, Pk_all


def generate_mock_xi(N_mocks, N_mesh, L, h, Omega_m, Omega_b, n_s, sigma8,
                     nbar, b=1.0, z=0.0, seed_start=2000,
                     r_max=200.0, n_bins=60):
    # average xi(r) over N_mocks lognormal realizations
    from pk_input import power_spectrum as pk_func
    from power_spectrum import estimate_xi
    from scipy.interpolate import interp1d

    print(f"Generating {N_mocks} lognormal mocks for xi(r) averaging")
    print(f"  N_mesh={N_mesh}, L={L}, nbar={nbar:.1e}, b={b}, z={z}")

    k_arr = np.logspace(-3, 1, 500)
    Pk_matter = pk_func(k_arr, h, Omega_m, Omega_b, n_s, sigma8, z=z)
    Pk_galaxy = b**2 * Pk_matter
    _, Pk_G = galaxy_pk_to_gaussian_pk(k_arr, Pk_galaxy, N_mesh, L)
    Pk_G_interp = interp1d(k_arr, Pk_G, bounds_error=False, fill_value=0.0)

    xi_all = None
    r_bins = None

    for i in range(N_mocks):
        seed = seed_start + i

        delta_LN = generate_lognormal_field(N_mesh, L, Pk_G_interp, seed=seed)
        pos = poisson_sample_vectorized(delta_LN, nbar, L, seed=seed + N_mocks)

        r, xi, _ = estimate_xi(pos, N_mesh, L, n_mesh=N_mesh,
                               r_max=r_max, n_bins=n_bins)

        if xi_all is None:
            r_bins = r
            xi_all = np.zeros((N_mocks, len(r_bins)))

        xi_all[i] = xi

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Mock {i+1}/{N_mocks} done  (N_gal={pos.shape[1]})")

    xi_mean = np.mean(xi_all, axis=0)
    xi_std = np.std(xi_all, axis=0)

    print(f"\nxi(r) averaged: {len(r_bins)} r-bins from {N_mocks} mocks")

    return r_bins, xi_mean, xi_std, xi_all


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from power_spectrum import estimate_pk
    from pk_input import power_spectrum

    h, Omega_m, Omega_b, n_s, sigma8 = 0.6736, 0.3153, 0.0493, 0.9649, 0.8111
    N_mesh, L = 64, 1500.0
    nbar, b, z = 3e-4, 1.5, 0.38

    pos, delta_LN = generate_lognormal_catalog(
        N_mesh, L, h, Omega_m, Omega_b, n_s, sigma8, nbar, b=b, z=z, seed=42)

    # measure P(k)
    k_out, Pk_out, _ = estimate_pk(pos, N_mesh, L, n_mesh=N_mesh*2)

    # theory
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