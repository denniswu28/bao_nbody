"""
mcmc.py
-------
Metropolis-Hastings MCMC for BAO template fitting.

Two fitting modes:

1. **Broadband-marginalized** (default, recommended):
   The model is decomposed into an oscillatory (wiggle-only) piece that
   depends on (alpha, Sigma) and an additive polynomial broadband that
   is analytically marginalized out.  The MCMC samples only (alpha, Sigma).

       P_model(k) = O_wiggle(k; alpha, Sigma)  +  sum_j  a_j * B_j(k)

   where O_wiggle = [P_lin(k/alpha) - P_nw(k/alpha)] * exp(-k^2 Sigma^2 / 2)
   and B_j(k) are broadband basis functions {(k/k_ref)^j} for j in poly_order.

   For fixed (alpha, Sigma) the polynomial coefficients {a_j} enter linearly
   and are solved in closed form; the marginalized chi-squared is:

       chi2_marg = r^T  P_perp  r
       P_perp   = C^{-1} - C^{-1} B (B^T C^{-1} B)^{-1} B^T C^{-1}

   where r = P_data - O_wiggle.  This decouples alpha from the broadband
   shape, eliminating the main systematic in the simple 3-parameter fit.

2. **Legacy 3-parameter** (backward-compatible):
   P_model = B * [O_wiggle + P_nw(k/alpha)]
   Samples (alpha, Sigma, B).  Used when broadband_marginalize=False.

The sound horizon is recovered as:  r_s = r_s_fiducial / alpha

References:
    Xu et al. (2012)      -- https://arxiv.org/abs/1202.0091
    Anderson et al. (2014) -- https://arxiv.org/abs/1312.4877
    Beutler et al. (2017)  -- https://arxiv.org/abs/1607.03149
"""

import numpy as np
import os
import corner
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def bao_template(k, alpha, Sigma, B, Pk_lin_func, Pk_nw_func):
    """
    Legacy damped BAO template (3 parameters).

    P_model = B * [(P_lin(k/a) - P_nw(k/a)) * exp(-k^2 S^2/2) + P_nw(k/a)]
    """
    k_scaled = k / alpha
    Pk_lin = Pk_lin_func(k_scaled)
    Pk_nw  = Pk_nw_func(k_scaled)

    damping = np.exp(-k**2 * Sigma**2 / 2)
    Pk_model = B * ((Pk_lin - Pk_nw) * damping + Pk_nw)

    return Pk_model


def wiggle_template(k, alpha, Sigma, Pk_lin_func, Pk_nw_func):
    """
    Oscillatory (wiggle-only) piece of the BAO template.

    O(k; alpha, Sigma) = [P_lin(k/a) - P_nw(k/a)] * exp(-k^2 S^2 / 2)
    """
    k_scaled = k / alpha
    return (Pk_lin_func(k_scaled) - Pk_nw_func(k_scaled)) * np.exp(-k**2 * Sigma**2 / 2)


def broadband_basis(k, poly_powers=(-2, -1, 0, 1, 2)):
    """
    Build the broadband polynomial basis matrix.

    B_j(k) = (k / k_ref)^j   with k_ref = 0.1 h/Mpc.

    Parameters
    ----------
    k : ndarray, shape (N,)
    poly_powers : tuple of ints
        Exponents for the broadband terms.

    Returns
    -------
    B : ndarray, shape (N, len(poly_powers))
    """
    k_ref = 0.1  # h/Mpc — keeps coefficients order-unity
    return np.column_stack([(k / k_ref)**j for j in poly_powers])


# ---------------------------------------------------------------------------
# Analytic broadband marginalisation
# ---------------------------------------------------------------------------

def _build_projector(B_basis, inv_cov):
    """
    Build the broadband-projection matrix once per fit.

    P_perp = C^{-1}  -  C^{-1} B (B^T C^{-1} B)^{-1} B^T C^{-1}

    Projects a residual vector into the subspace orthogonal to the broadband.
    """
    CiB = inv_cov @ B_basis                        # (N, p)
    BtCiB = B_basis.T @ CiB                        # (p, p)
    BtCiB_inv = np.linalg.inv(BtCiB)               # (p, p)
    projector = inv_cov - CiB @ BtCiB_inv @ CiB.T  # (N, N)
    return projector


def _build_projector_diag(B_basis, Pk_err):
    """
    Same as _build_projector but for diagonal errors (no full covariance).
    """
    w = 1.0 / Pk_err**2                            # (N,)
    W = np.diag(w)                                  # (N, N) diagonal C^{-1}
    WB = W @ B_basis                                # (N, p)
    BtWB = B_basis.T @ WB                           # (p, p)
    BtWB_inv = np.linalg.inv(BtWB)
    return W - WB @ BtWB_inv @ WB.T


def log_likelihood_marginalized(params, k_data, Pk_data, Pk_err,
                                Pk_lin_func, Pk_nw_func,
                                projector):
    """
    Broadband-marginalized Gaussian log-likelihood.

    Only (alpha, Sigma) are free; the polynomial broadband is projected out.

        log L = -0.5 * r^T  P_perp  r

    where r = P_data - O_wiggle(k; alpha, Sigma).
    """
    alpha, Sigma = params
    O = wiggle_template(k_data, alpha, Sigma, Pk_lin_func, Pk_nw_func)
    r = Pk_data - O
    return -0.5 * r @ projector @ r


def bestfit_broadband(k_data, Pk_data, alpha, Sigma,
                      Pk_lin_func, Pk_nw_func,
                      inv_cov, B_basis):
    """
    Solve for the best-fit broadband coefficients at fixed (alpha, Sigma).

    Returns
    -------
    a_fit : ndarray, shape (n_poly,)
        Best-fit polynomial coefficients.
    Pk_model : ndarray, shape (N_bins,)
        Full model P(k) = O_wiggle + B_basis @ a_fit.
    """
    O = wiggle_template(k_data, alpha, Sigma, Pk_lin_func, Pk_nw_func)
    r = Pk_data - O
    CiB = inv_cov @ B_basis
    BtCiB = B_basis.T @ CiB
    a_fit = np.linalg.solve(BtCiB, CiB.T @ r)
    Pk_model = O + B_basis @ a_fit
    return a_fit, Pk_model


# ---------------------------------------------------------------------------
# Legacy (3-parameter) likelihood
# ---------------------------------------------------------------------------

def log_likelihood(params, k_data, Pk_data, Pk_err,
                   Pk_lin_func, Pk_nw_func, inv_cov=None):
    """
    Gaussian log-likelihood for the legacy 3-parameter template.
    """
    alpha, Sigma, B = params
    Pk_model = bao_template(k_data, alpha, Sigma, B, Pk_lin_func, Pk_nw_func)
    residuals = Pk_data - Pk_model
    if inv_cov is not None:
        return -0.5 * residuals @ inv_cov @ residuals
    return -0.5 * np.sum((residuals / Pk_err)**2)


# ---------------------------------------------------------------------------
# Priors & posteriors
# ---------------------------------------------------------------------------

def log_prior(params, priors):
    """
    Flat (uniform) priors.  Works for both 2-param and 3-param modes.
    """
    names = ['alpha', 'Sigma', 'B'][:len(params)]
    for val, name in zip(params, names):
        lo, hi = priors[name]
        if not (lo < val < hi):
            return -np.inf
    return 0.0


def log_posterior_marginalized(params, k_data, Pk_data, Pk_err,
                               Pk_lin_func, Pk_nw_func,
                               priors, projector):
    lp = log_prior(params, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_marginalized(
        params, k_data, Pk_data, Pk_err,
        Pk_lin_func, Pk_nw_func, projector)


def log_posterior(params, k_data, Pk_data, Pk_err,
                  Pk_lin_func, Pk_nw_func, priors, inv_cov=None):
    lp = log_prior(params, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, k_data, Pk_data, Pk_err,
                                Pk_lin_func, Pk_nw_func, inv_cov=inv_cov)


# ---------------------------------------------------------------------------
# MCMC samplers
# ---------------------------------------------------------------------------

def run_mcmc_marginalized(k_data, Pk_data, Pk_err,
                          Pk_lin_func, Pk_nw_func,
                          theta_init, priors, projector,
                          n_steps=10000, n_burn=2000,
                          step_size=None, seed=42):
    """
    Metropolis-Hastings sampler for (alpha, Sigma) with broadband
    analytically marginalized.
    """
    rng = np.random.default_rng(seed)
    n_params = 2
    theta = np.array(theta_init[:2], dtype=float)

    if step_size is None:
        step_size = np.array([0.03, 2.0])  # alpha, Sigma — tuned for ~30% acceptance

    chain = np.zeros((n_steps, n_params))
    log_probs = np.zeros(n_steps)
    n_accept = 0

    log_prob_current = log_posterior_marginalized(
        theta, k_data, Pk_data, Pk_err,
        Pk_lin_func, Pk_nw_func, priors, projector)

    print(f"Running MCMC (broadband-marginalized): {n_steps} steps, burn-in={n_burn}")
    print(f"  Initial params: alpha={theta[0]:.3f}, Sigma={theta[1]:.2f}")
    print(f"  Initial log-posterior: {log_prob_current:.2f}")

    for i in range(n_steps):
        theta_proposal = theta + rng.normal(0, step_size[:n_params], n_params)

        log_prob_proposal = log_posterior_marginalized(
            theta_proposal, k_data, Pk_data, Pk_err,
            Pk_lin_func, Pk_nw_func, priors, projector)

        if np.log(rng.uniform()) < log_prob_proposal - log_prob_current:
            theta = theta_proposal
            log_prob_current = log_prob_proposal
            n_accept += 1

        chain[i] = theta
        log_probs[i] = log_prob_current

        if (i + 1) % 2000 == 0:
            rate = n_accept / (i + 1)
            print(f"  Step {i+1}/{n_steps}  |  accept rate = {rate:.3f}  |  "
                  f"alpha={theta[0]:.4f}, Sigma={theta[1]:.2f}")

    accept_rate = n_accept / n_steps
    chain_burned = chain[n_burn:]
    log_probs_burned = log_probs[n_burn:]

    print(f"\nMCMC complete. Acceptance rate: {accept_rate:.3f}")
    for name, vals in zip(['alpha', 'Sigma'], chain_burned.T):
        med = np.median(vals)
        lo, hi = np.percentile(vals, [16, 84])
        print(f"  {name} = {med:.4f} + {hi-med:.4f} - {med-lo:.4f}")

    return chain_burned, log_probs_burned, accept_rate


def run_mcmc(k_data, Pk_data, Pk_err,
             Pk_lin_func, Pk_nw_func,
             theta_init, priors,
             n_steps=10000, n_burn=2000,
             step_size=None, seed=42,
             inv_cov=None):
    """
    Legacy Metropolis-Hastings sampler for 3-parameter (alpha, Sigma, B) template.
    """
    rng = np.random.default_rng(seed)

    n_params = len(theta_init)
    theta = np.array(theta_init, dtype=float)

    if step_size is None:
        step_size = np.array([0.005, 0.5, 0.02])

    chain = np.zeros((n_steps, n_params))
    log_probs = np.zeros(n_steps)
    n_accept = 0

    log_prob_current = log_posterior(theta, k_data, Pk_data, Pk_err,
                                      Pk_lin_func, Pk_nw_func, priors,
                                      inv_cov=inv_cov)

    cov_label = 'full covariance' if inv_cov is not None else 'diagonal errors'
    print(f"Running MCMC (legacy): {n_steps} steps, burn-in={n_burn} ({cov_label})")
    print(f"  Initial params: alpha={theta[0]:.3f}, Sigma={theta[1]:.2f}, B={theta[2]:.3f}")
    print(f"  Initial log-posterior: {log_prob_current:.2f}")

    for i in range(n_steps):
        theta_proposal = theta + rng.normal(0, step_size, n_params)

        log_prob_proposal = log_posterior(theta_proposal, k_data, Pk_data, Pk_err,
                                           Pk_lin_func, Pk_nw_func, priors,
                                           inv_cov=inv_cov)

        log_alpha_accept = log_prob_proposal - log_prob_current
        if np.log(rng.uniform()) < log_alpha_accept:
            theta = theta_proposal
            log_prob_current = log_prob_proposal
            n_accept += 1

        chain[i] = theta
        log_probs[i] = log_prob_current

        if (i + 1) % 2000 == 0:
            rate = n_accept / (i + 1)
            print(f"  Step {i+1}/{n_steps}  |  accept rate = {rate:.3f}  |  "
                  f"alpha={theta[0]:.4f}, Sigma={theta[1]:.2f}, B={theta[2]:.3f}")

    accept_rate = n_accept / n_steps
    chain_burned = chain[n_burn:]
    log_probs_burned = log_probs[n_burn:]

    print(f"\nMCMC complete. Acceptance rate: {accept_rate:.3f}")
    for name, vals in zip(['alpha', 'Sigma', 'B'], chain_burned.T):
        med = np.median(vals)
        lo, hi = np.percentile(vals, [16, 84])
        print(f"  {name} = {med:.4f} + {hi-med:.4f} - {med-lo:.4f}")

    return chain_burned, log_probs_burned, accept_rate


def make_corner_plot(chain, labels, truths=None, fname=None):
    """
    Make a corner plot of the MCMC chain.

    Parameters
    ----------
    chain : ndarray, shape (N, 3)
    labels : list of str
    truths : list of float, optional
        True parameter values to mark.
    fname : str, optional
        Output filename.
    """
    fig = corner.corner(chain, labels=labels, truths=truths,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12})
    if fname:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved corner plot: {fname}")
    return fig


def fit_bao(k_data, Pk_data, Pk_err, h, Omega_m, Omega_b, n_s, sigma8, z=0.0,
            mcmc_config=None, label='', output_dir='outputs/mcmc',
            cov=None, hartlap_factor=1.0,
            broadband_marginalize=True, poly_powers=(-2, -1, 0, 1, 2)):
    """
    Full BAO fitting pipeline: build template, run MCMC, make plots.

    Parameters
    ----------
    k_data, Pk_data, Pk_err : arrays
        Measured power spectrum and errors.
    label : str
        Label for output files (e.g. 'nbody_z0', 'lognormal', 'recon').
    broadband_marginalize : bool
        If True (default), analytically marginalize a polynomial broadband
        and sample only (alpha, Sigma).  If False, use the legacy
        3-parameter (alpha, Sigma, B) template.
    poly_powers : tuple of ints
        Exponents for the broadband polynomial basis, e.g. (-2,-1,0,1,2).

    Returns
    -------
    chain : ndarray
        MCMC chain — shape (N, 2) if marginalized, (N, 3) if legacy.
    r_s_fit : float
        Recovered sound horizon in Mpc/h.
    """
    from pk_input import power_spectrum, sound_horizon

    if mcmc_config is None:
        mcmc_config = {
            'n_steps': 10000, 'n_burn': 2000,
            'alpha_init': 1.0, 'Sigma_init': 5.0, 'B_init': 1.0,
            'alpha_prior': [0.5, 1.5],
            'Sigma_prior': [0.0, 20.0],
            'B_prior': [0.5, 4.0],
        }

    # Build template P(k) functions
    k_th = np.logspace(-3, 0, 1000)
    Pk_lin = power_spectrum(k_th, h, Omega_m, Omega_b, n_s, sigma8, z=z, wiggle=True)
    Pk_nw  = power_spectrum(k_th, h, Omega_m, Omega_b, n_s, sigma8, z=z, wiggle=False)

    Pk_lin_func = interp1d(k_th, Pk_lin, bounds_error=False, fill_value=0.0)
    Pk_nw_func  = interp1d(k_th, Pk_nw,  bounds_error=False, fill_value=0.0)

    r_s_fid = sound_horizon(h, Omega_m, Omega_b)

    priors = {
        'alpha': mcmc_config['alpha_prior'],
        'Sigma': mcmc_config['Sigma_prior'],
        'B':     mcmc_config.get('B_prior', [0.5, 4.0]),
    }

    # Compute inverse covariance
    inv_cov = None
    if cov is not None:
        inv_cov = np.linalg.inv(cov) * hartlap_factor
        print(f"  Using full covariance matrix ({cov.shape[0]}x{cov.shape[1]}), "
              f"Hartlap = {hartlap_factor:.3f}")

    # --- Broadband-marginalized mode (recommended) ---
    if broadband_marginalize:
        B_basis = broadband_basis(k_data, poly_powers=poly_powers)
        print(f"  Broadband basis: powers = {poly_powers} "
              f"({B_basis.shape[1]} terms)")

        if inv_cov is not None:
            projector = _build_projector(B_basis, inv_cov)
        else:
            projector = _build_projector_diag(B_basis, Pk_err)

        theta_init = [mcmc_config['alpha_init'], mcmc_config['Sigma_init']]

        chain, log_probs, accept_rate = run_mcmc_marginalized(
            k_data, Pk_data, Pk_err,
            Pk_lin_func, Pk_nw_func,
            theta_init, priors, projector,
            n_steps=mcmc_config['n_steps'],
            n_burn=mcmc_config['n_burn'],
        )

        alpha_med = np.median(chain[:, 0])
        r_s_fit = r_s_fid / alpha_med
        print(f"\nRecovered r_s = {r_s_fit:.2f} Mpc/h  (fiducial: {r_s_fid:.2f} Mpc/h)")

        # Corner plot — 2 parameters
        labels_2p = [r'$\alpha$', r'$\Sigma$ [Mpc/$h$]']
        fname_corner = os.path.join(output_dir, f'corner_{label}.png')
        make_corner_plot(chain, labels_2p, truths=[1.0, None], fname=fname_corner)

        # Best-fit P(k) plot with broadband solved at the median
        alpha_bf, Sigma_bf = np.median(chain, axis=0)
        inv_cov_for_solve = inv_cov if inv_cov is not None else np.diag(1.0 / Pk_err**2)
        a_fit, Pk_bestfit = bestfit_broadband(
            k_data, Pk_data, alpha_bf, Sigma_bf,
            Pk_lin_func, Pk_nw_func, inv_cov_for_solve, B_basis)

        O_bf = wiggle_template(k_data, alpha_bf, Sigma_bf,
                               Pk_lin_func, Pk_nw_func)
        bb_bf = B_basis @ a_fit

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 2, 1]})
        axes[0].errorbar(k_data, Pk_data, yerr=Pk_err, fmt='.', alpha=0.6,
                         label='Data', zorder=1)
        axes[0].plot(k_data, Pk_bestfit, 'r-', lw=1.5,
                     label=f'Best fit (α={alpha_bf:.3f})', zorder=3)
        axes[0].plot(k_data, bb_bf, 'g--', alpha=0.6,
                     label='Broadband polynomial', zorder=2)
        axes[0].loglog(k_th, Pk_lin, 'k--', alpha=0.3, label='Linear theory')
        axes[0].set_ylabel(r'$P(k)$ [$(h^{-1}$Mpc$)^3$]')
        axes[0].legend(fontsize='small')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f'BAO fit (broadband-marg): {label}  |  '
                          f'$r_s = {r_s_fit:.1f}$ Mpc/$h$')

        # Wiggle residual panel
        axes[1].plot(k_data, Pk_data - bb_bf, 'b.', alpha=0.6,
                     label='Data − broadband')
        axes[1].plot(k_data, O_bf, 'r-', lw=1.5, label='Wiggle template')
        axes[1].axhline(0, color='gray', ls=':', alpha=0.5)
        axes[1].set_ylabel(r'$P(k) - P_{\rm BB}(k)$')
        axes[1].legend(fontsize='small')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(k_data, (Pk_data - Pk_bestfit) / Pk_err, '.', alpha=0.6)
        axes[2].axhline(0, color='r', ls='--')
        axes[2].set_xlabel(r'$k$ [$h$/Mpc]')
        axes[2].set_ylabel(r'Residual / $\sigma$')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'bestfit_{label}.png'), dpi=150)
        plt.close(fig)

        return chain, r_s_fit

    # --- Legacy 3-parameter mode ---
    theta_init = [
        mcmc_config['alpha_init'],
        mcmc_config['Sigma_init'],
        mcmc_config['B_init'],
    ]

    chain, log_probs, accept_rate = run_mcmc(
        k_data, Pk_data, Pk_err,
        Pk_lin_func, Pk_nw_func,
        theta_init, priors,
        n_steps=mcmc_config['n_steps'],
        n_burn=mcmc_config['n_burn'],
        inv_cov=inv_cov,
    )

    alpha_med = np.median(chain[:, 0])
    r_s_fit = r_s_fid / alpha_med
    print(f"\nRecovered r_s = {r_s_fit:.2f} Mpc/h  (fiducial: {r_s_fid:.2f} Mpc/h)")

    labels = [r'$\alpha$', r'$\Sigma$ [Mpc/$h$]', r'$B$']
    fname = os.path.join(output_dir, f'corner_{label}.png')
    make_corner_plot(chain, labels, truths=[1.0, None, 1.0], fname=fname)

    alpha_bf, Sigma_bf, B_bf = np.median(chain, axis=0)
    Pk_bestfit = bao_template(k_data, alpha_bf, Sigma_bf, B_bf,
                              Pk_lin_func, Pk_nw_func)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].errorbar(k_data, Pk_data, yerr=Pk_err, fmt='.', alpha=0.6, label='Data')
    axes[0].loglog(k_data, Pk_bestfit, 'r-', label=f'Best fit (α={alpha_bf:.3f})')
    axes[0].loglog(k_th, Pk_lin, 'k--', alpha=0.4, label='Theory (wiggle)')
    axes[0].loglog(k_th, Pk_nw,  'k:',  alpha=0.4, label='Theory (no-wiggle)')
    axes[0].set_ylabel(r'$P(k)$ [$(h^{-1}$Mpc$)^3$]')
    axes[0].legend(fontsize='small')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'BAO fit: {label}  |  $r_s = {r_s_fit:.1f}$ Mpc/$h$')

    axes[1].semilogx(k_data, (Pk_data - Pk_bestfit) / Pk_err, '.', alpha=0.6)
    axes[1].axhline(0, color='r', ls='--')
    axes[1].set_xlabel(r'$k$ [$h$/Mpc]')
    axes[1].set_ylabel(r'Residual / $\sigma$')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'bestfit_{label}.png'), dpi=150)
    plt.close(fig)

    return chain, r_s_fit
