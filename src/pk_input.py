"""
pk_input.py
-----------
Eisenstein & Hu (1998) fitting formulae for the matter power spectrum P(k),
including the BAO wiggle spectrum and the smooth no-wiggle reference spectrum.

References:
    Eisenstein & Hu (1998), ApJ, 496, 605
    https://arxiv.org/abs/astro-ph/9709066
"""

import numpy as np


def sound_horizon(h, Omega_m, Omega_b):
    """
    Compute the BAO sound horizon scale r_s in Mpc/h (comoving).
    Eisenstein & Hu (1998) eq. 26 fitting formula.

    The EH98 formula itself returns r_s in Mpc (~150 Mpc for Planck);
    we multiply by h here so that the returned value is in Mpc/h, matching
    the rest of the pipeline. Planck cosmology: r_s ~ 101 Mpc/h.
    """
    Omega_m_h2 = Omega_m * h**2
    Omega_b_h2 = Omega_b * h**2

    r_s_Mpc = 44.5 * np.log(9.83 / Omega_m_h2) \
              / np.sqrt(1 + 10 * Omega_b_h2**0.75)   # Mpc

    return r_s_Mpc * h    # Mpc/h


def transfer_function_eh(k, h, Omega_m, Omega_b):
    """
    Full Eisenstein & Hu (1998) transfer function with BAO wiggles.

    Parameters
    ----------
    k : array_like
        Wavenumbers in h/Mpc.
    h, Omega_m, Omega_b : float
        Cosmological parameters.

    Returns
    -------
    T : ndarray
        Transfer function T(k), dimensionless.
    """
    k = np.atleast_1d(k) * h   # convert h/Mpc → Mpc⁻¹ for internal EH98 formulae

    Omega_m_h2 = Omega_m * h**2
    Omega_b_h2 = Omega_b * h**2
    f_b = Omega_b / Omega_m    # baryon fraction

    # Redshifts
    z_eq = 2.5e4 * Omega_m_h2 * (2.725 / 2.7)**(-4)
    k_eq = 7.46e-2 * Omega_m_h2 * (2.725 / 2.7)**(-2)   # Mpc⁻¹

    b1 = 0.313 * Omega_m_h2**(-0.419) * (1 + 0.607 * Omega_m_h2**0.674)
    b2 = 0.238 * Omega_m_h2**0.223
    z_drag = 1291 * Omega_m_h2**0.251 / (1 + 0.659 * Omega_m_h2**0.828) \
             * (1 + b1 * Omega_b_h2**b2)

    R_drag = 31.5e3 * Omega_b_h2 * (2.725 / 2.7)**(-4) / z_drag
    R_eq   = 31.5e3 * Omega_b_h2 * (2.725 / 2.7)**(-4) / z_eq

    s = 2 / (3 * k_eq) * np.sqrt(6 / R_eq) \
        * np.log((np.sqrt(1 + R_drag) + np.sqrt(R_drag + R_eq)) / (1 + np.sqrt(R_eq)))

    k_silk = 1.6 * Omega_b_h2**0.52 * Omega_m_h2**0.01 \
             * (1 + (11.25 * Omega_b_h2)**(-0.824))   # Silk damping

    # CDM transfer function
    a1 = (46.9 * Omega_m_h2)**0.670 * (1 + (32.1 * Omega_m_h2)**(-0.532))
    a2 = (12.0 * Omega_m_h2)**0.424 * (1 + (45.0 * Omega_m_h2)**(-0.582))
    alpha_c = a1**(-f_b) * a2**(-f_b**3)

    bb1 = 0.944 / (1 + (458 * Omega_m_h2)**(-0.708))
    bb2 = (0.395 * Omega_m_h2)**(-0.0266)
    beta_c = 1 / (1 + bb1 * ((1 - f_b)**bb2 - 1))

    def T_tilde(k, alpha, beta):
        q = k / (13.41 * k_eq)
        C = 14.2 / alpha + 386 / (1 + 69.9 * q**1.08)
        return np.log(np.e + 1.8 * beta * q) / (np.log(np.e + 1.8 * beta * q) + C * q**2)

    f = 1 / (1 + (k * s / 5.4)**4)
    T_c = f * T_tilde(k, 1, beta_c) + (1 - f) * T_tilde(k, alpha_c, 1)

    # Baryon transfer function
    y = z_eq / (1 + z_drag)
    G = y * (-6 * np.sqrt(1 + y) + (2 + 3 * y) * np.log((np.sqrt(1 + y) + 1) / (np.sqrt(1 + y) - 1)))

    alpha_b = 2.07 * k_eq * s * (1 + R_drag)**(-3/4) * G

    beta_b  = 0.5 + f_b + (3 - 2 * f_b) * np.sqrt((17.2 * Omega_m_h2)**2 + 1)
    beta_node = 8.41 * Omega_m_h2**0.435

    s_tilde = s / (1 + (beta_node / (k * s))**3)**(1/3)

    j0 = np.sinc(k * s_tilde / np.pi)    # np.sinc uses normalized sinc
    T_b = (T_tilde(k, 1, 1) / (1 + (k * s / 5.2)**2)
           + alpha_b / (1 + (beta_b / (k * s))**3) * np.exp(-(k / k_silk)**1.4)) * j0

    T = f_b * T_b + (1 - f_b) * T_c

    return T


def transfer_function_nowiggle(k, h, Omega_m, Omega_b):
    """
    Eisenstein & Hu (1998) smooth no-wiggle transfer function.
    Used as the broad-band reference in BAO template fitting.
    """
    k = np.atleast_1d(k) * h   # convert h/Mpc → Mpc⁻¹ for internal EH98 formulae

    Omega_m_h2 = Omega_m * h**2
    Omega_b_h2 = Omega_b * h**2
    f_b = Omega_b / Omega_m

    alpha_gamma = 1 - 0.328 * np.log(431 * Omega_m_h2) * f_b \
                  + 0.38 * np.log(22.3 * Omega_m_h2) * f_b**2

    # sound_horizon() returns Mpc/h; EH98 eq. 30 uses s in Mpc (k here is in Mpc^-1)
    s = sound_horizon(h, Omega_m, Omega_b) / h   # Mpc
    gamma_eff = Omega_m * h * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43 * k * s)**4))

    q = k * (2.725 / 2.7)**2 / (gamma_eff)
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731 / (1 + 62.5 * q)
    T_nw = L0 / (L0 + C0 * q**2)

    return T_nw


def power_spectrum(k, h, Omega_m, Omega_b, n_s, sigma8, z=0.0, wiggle=True):
    """
    Full matter power spectrum P(k) using Eisenstein-Hu transfer function.

    P(k) = A * k^n_s * T(k)^2

    Normalized to sigma8 at z=0, then scaled by the linear growth factor D(z)^2.

    Parameters
    ----------
    k : array_like
        Wavenumbers in h/Mpc.
    wiggle : bool
        If True, use the full transfer function with BAO wiggles.
        If False, use the smooth no-wiggle transfer function.

    Returns
    -------
    Pk : ndarray
        Power spectrum in (Mpc/h)^3.
    """
    k = np.atleast_1d(k)

    if wiggle:
        T = transfer_function_eh(k, h, Omega_m, Omega_b)
    else:
        T = transfer_function_nowiggle(k, h, Omega_m, Omega_b)

    Pk_unnorm = k**n_s * T**2

    # Normalize to sigma8 (uses internal k array for robust integration)
    sigma8_unnorm = _compute_sigma8(None, h, Omega_m, Omega_b, n_s, wiggle=wiggle)
    A = (sigma8 / sigma8_unnorm)**2
    Pk = A * Pk_unnorm

    # Apply growth factor
    Dz = growth_factor(z, Omega_m)
    Pk *= Dz**2

    return Pk


def growth_factor(z, Omega_m, Omega_lambda=None):
    """
    Linear growth factor D(z) normalized to 1 at z=0.
    Carroll, Press & Turner (1992) approximation.
    At high z (matter dominated): D(z) ~ a = 1/(1+z).
    """
    if Omega_lambda is None:
        Omega_lambda = 1 - Omega_m

    def _D(z_):
        a_ = 1 / (1 + z_)
        E2 = Omega_m * (1 + z_)**3 + Omega_lambda
        Om_z = Omega_m * (1 + z_)**3 / E2
        Ol_z = Omega_lambda / E2
        return a_ * 2.5 * Om_z / (Om_z**(4/7) - Ol_z
               + (1 + Om_z / 2) * (1 + Ol_z / 70))

    return _D(z) / _D(0)


def _compute_sigma8(Pk_unnorm_func, h, Omega_m, Omega_b, n_s, wiggle=True):
    """
    Compute sigma8 for normalization using a top-hat window of R=8 Mpc/h.

    Uses an internal well-sampled k array so this works regardless of
    what k values the caller passes.
    """
    k_int = np.logspace(-4, 2, 2000)

    if wiggle:
        T = transfer_function_eh(k_int, h, Omega_m, Omega_b)
    else:
        T = transfer_function_nowiggle(k_int, h, Omega_m, Omega_b)

    Pk_unnorm = k_int**n_s * T**2

    R = 8.0  # Mpc/h
    x = k_int * R
    W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    W[x < 1e-3] = 1.0

    integrand = k_int**2 * Pk_unnorm * W**2 / (2 * np.pi**2)
    sigma8_sq = np.trapezoid(integrand, k_int)
    return np.sqrt(sigma8_sq)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    h, Omega_m, Omega_b, n_s, sigma8 = 0.6736, 0.3153, 0.0493, 0.9649, 0.8111

    k = np.logspace(-3, 0, 500)
    Pk_wiggle   = power_spectrum(k, h, Omega_m, Omega_b, n_s, sigma8, wiggle=True)
    Pk_nowiggle = power_spectrum(k, h, Omega_m, Omega_b, n_s, sigma8, wiggle=False)
    r_s = sound_horizon(h, Omega_m, Omega_b)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    axes[0].loglog(k, Pk_wiggle, label='With BAO wiggles')
    axes[0].loglog(k, Pk_nowiggle, '--', label='No-wiggle (smooth)')
    axes[0].set_xlabel(r'$k$ [$h$/Mpc]')
    axes[0].set_ylabel(r'$P(k)$ [$(h^{-1}$Mpc$)^3$]')
    axes[0].set_title(f'Eisenstein-Hu Power Spectrum  |  $r_s = {r_s:.1f}$ Mpc/$h$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(k, Pk_wiggle / Pk_nowiggle, label='BAO wiggles / smooth')
    axes[1].axhline(1, color='k', lw=0.8, ls='--')
    axes[1].set_xlabel(r'$k$ [$h$/Mpc]')
    axes[1].set_ylabel(r'$P(k) / P_\mathrm{nw}(k)$')
    axes[1].set_title('BAO Oscillations')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/pk_input.png', dpi=150)
    plt.show()
    print(f"Sound horizon r_s = {r_s:.2f} Mpc/h")