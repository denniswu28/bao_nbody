import numpy as np
from pk_input import power_spectrum, growth_factor


def make_grid(N, L):
    dx = L / N
    x = np.arange(N) * dx - L / 2 + dx / 2
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    pos = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=0)
    return pos


def _k_grids(N, L):
    dk = 2 * np.pi / L
    k1d = np.fft.fftfreq(N, d=1.0 / N) * dk
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0   # avoid divide by zero
    return kx, ky, kz, k2


def generate_gaussian_field(N, L, Pk_func, seed=42):
    rng = np.random.default_rng(seed)

    kx, ky, kz, k2 = _k_grids(N, L)
    k = np.sqrt(k2)
    k[0, 0, 0] = 0.0

    Pk = np.zeros_like(k)
    mask = k > 0
    Pk[mask] = Pk_func(k[mask])

    # amplitude tuned so Re(IFFT) recovers the input power spectrum
    # (factor of 2 from drawing non-Hermitian modes cancels the Re() halving)
    V = L**3
    amplitude = N**3 * np.sqrt(Pk / V)

    re = rng.standard_normal((N, N, N))
    im = rng.standard_normal((N, N, N))
    delta_k = amplitude * (re + 1j * im)
    delta_k[0, 0, 0] = 0.0

    return delta_k


def displacement_field(delta_k, N, L):
    # zel'dovich displacement field from delta
    kx, ky, kz, k2 = _k_grids(N, L)

    Psi = []
    for ki in [kx, ky, kz]:
        Psi_k = -1j * ki / k2 * delta_k
        Psi_k[0, 0, 0] = 0.0
        Psi_i = np.real(np.fft.ifftn(Psi_k))
        Psi.append(Psi_i.ravel())

    return np.array(Psi)


def hubble(z, h, Omega_m, Omega_lambda=None):
    # dimensionless Hubble factor
    if Omega_lambda is None:
        Omega_lambda = 1 - Omega_m
    a = 1 / (1 + z)
    E2 = Omega_m / a**3 + Omega_lambda
    return np.sqrt(E2)


def growth_rate(z, Omega_m):
    # logarithmic growth rate, standard Omega_m(z)^0.55 approximation
    Omega_lambda = 1 - Omega_m
    a = 1 / (1 + z)
    Omega_m_z = Omega_m / a**3 / (Omega_m / a**3 + Omega_lambda)
    return Omega_m_z**0.55


def make_ics(N, L, h, Omega_m, Omega_b, n_s, sigma8,
             z_initial=49.0, seed=42):
    print(f"Generating Zel'dovich ICs: N={N}, L={L} Mpc/h, z_init={z_initial}")

    D_init = growth_factor(z_initial, Omega_m)

    def Pk_func(k):
        return power_spectrum(k, h, Omega_m, Omega_b, n_s, sigma8, z=0.0)

    delta_k = generate_gaussian_field(N, L, Pk_func, seed=seed)
    delta_k_init = delta_k * D_init

    Psi = displacement_field(delta_k_init, N, L)
    pos_grid = make_grid(N, L)
    pos = pos_grid + Psi

    # zel'dovich peculiar velocity
    a_init = 1 / (1 + z_initial)
    Hz = hubble(z_initial, h, Omega_m) * 100   # km/s/Mpc
    f = growth_rate(z_initial, Omega_m)
    vel = a_init * Hz * f * Psi

    # periodic BC
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
