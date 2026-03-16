"""
mcmc.py
-------
Metropolis-Hastings MCMC for BAO template fitting.

Fits the damped BAO template model to a measured P(k):

    P_model(k; alpha, Sigma, B) = B * [
        (P_lin(k/alpha) - P_nw(k/alpha)) * exp(-k^2 * Sigma^2 / 2)
        + P_nw(k/alpha)
    ]

Parameters:
    alpha  : BAO dilation parameter (1 = no shift, r_s_fit = r_s_fid / alpha)
    Sigma  : BAO damping scale [Mpc/h] (nonlinear smearing)
    B      : broad-band amplitude offset

The sound horizon is recovered as: r_s = r_s_fiducial / alpha

References:
    Xu et al. (2012) — https://arxiv.org/abs/1202.0091
    Anderson et al. (2014) — https://arxiv.org/abs/1312.4877
"""

import numpy as np
import os
import corner
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def bao_template(k, alpha, Sigma, B, Pk_lin_func, Pk_nw_func):
    """
    Damped BAO template model.

    Parameters
    ----------
    k : ndarray
        Wavenumbers in h/Mpc.
    alpha : float
        BAO dilation parameter.
    Sigma : float
        BAO damping scale in Mpc/h.
    B : float
        Broad-band amplitude.
    Pk_lin_func, Pk_nw_func : callables
        Interpolating functions for linear P(k) and no-wiggle P(k).

    Returns
    -------
    Pk_model : ndarray
        Model power spectrum.
    """
    k_scaled = k / alpha
    Pk_lin = Pk_lin_func(k_scaled)
    Pk_nw  = Pk_nw_func(k_scaled)

    damping = np.exp(-k**2 * Sigma**2 / 2)
    Pk_model = B * ((Pk_lin - Pk_nw) * damping + Pk_nw)

    return Pk_model


def log_likelihood(params, k_data, Pk_data, Pk_err,
                   Pk_lin_func, Pk_nw_func):
    """
    Gaussian log-likelihood:
        log L = -0.5 * sum[(P_data - P_model)^2 / sigma^2]
    """
    alpha, Sigma, B = params
    Pk_model = bao_template(k_data, alpha, Sigma, B, Pk_lin_func, Pk_nw_func)
    residuals = Pk_data - Pk_model
    return -0.5 * np.sum((residuals / Pk_err)**2)


def log_prior(params, priors):
    """
    Flat (uniform) priors on all parameters.

    priors : dict with keys 'alpha', 'Sigma', 'B', each a [min, max] list.
    """
    alpha, Sigma, B = params
    if not (priors['alpha'][0] < alpha < priors['alpha'][1]):
        return -np.inf
    if not (priors['Sigma'][0] < Sigma < priors['Sigma'][1]):
        return -np.inf
    if not (priors['B'][0] < B < priors['B'][1]):
        return -np.inf
    return 0.0


def log_posterior(params, k_data, Pk_data, Pk_err,
                  Pk_lin_func, Pk_nw_func, priors):
    lp = log_prior(params, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, k_data, Pk_data, Pk_err,
                                Pk_lin_func, Pk_nw_func)


def run_mcmc(k_data, Pk_data, Pk_err,
             Pk_lin_func, Pk_nw_func,
             theta_init, priors,
             n_steps=10000, n_burn=2000,
             step_size=None, seed=42):
    """
    Metropolis-Hastings MCMC sampler.

    Parameters
    ----------
    k_data, Pk_data, Pk_err : arrays
        Data: wavenumbers, power spectrum, and errors.
    Pk_lin_func, Pk_nw_func : callables
        Linear and no-wiggle P(k) interpolators.
    theta_init : array-like [alpha, Sigma, B]
        Starting parameter values.
    priors : dict
        Flat prior ranges for each parameter.
    n_steps : int
        Total MCMC steps.
    n_burn : int
        Burn-in steps to discard.
    step_size : array-like, optional
        Proposal step sizes for each parameter. Auto-tuned if None.
    seed : int
        Random seed.

    Returns
    -------
    chain : ndarray, shape (n_steps - n_burn, 3)
        MCMC chain after burn-in.
    log_probs : ndarray
        Log-posterior values along the chain.
    accept_rate : float
        Acceptance rate.
    """
    rng = np.random.default_rng(seed)

    n_params = len(theta_init)
    theta = np.array(theta_init, dtype=float)

    if step_size is None:
        # Default proposal widths: ~1% of prior range
        step_size = np.array([
            0.005,   # alpha
            0.5,     # Sigma [Mpc/h]
            0.02,    # B
        ])

    chain = np.zeros((n_steps, n_params))
    log_probs = np.zeros(n_steps)
    n_accept = 0

    log_prob_current = log_posterior(theta, k_data, Pk_data, Pk_err,
                                      Pk_lin_func, Pk_nw_func, priors)

    print(f"Running MCMC: {n_steps} steps, burn-in={n_burn}")
    print(f"  Initial params: alpha={theta[0]:.3f}, Sigma={theta[1]:.2f}, B={theta[2]:.3f}")
    print(f"  Initial log-posterior: {log_prob_current:.2f}")

    for i in range(n_steps):
        # Gaussian proposal
        theta_proposal = theta + rng.normal(0, step_size, n_params)

        log_prob_proposal = log_posterior(theta_proposal, k_data, Pk_data, Pk_err,
                                           Pk_lin_func, Pk_nw_func, priors)

        # Metropolis acceptance
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

    # Print parameter estimates
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
            mcmc_config=None, label='', output_dir='outputs/mcmc'):
    """
    Full BAO fitting pipeline: build template, run MCMC, make plots.

    Parameters
    ----------
    k_data, Pk_data, Pk_err : arrays
        Measured power spectrum and errors.
    label : str
        Label for output files (e.g. 'nbody_z0', 'lognormal', 'recon').

    Returns
    -------
    chain : ndarray
        MCMC chain.
    r_s_fit : float
        Recovered sound horizon in Mpc/h.
    """
    from pk_input import power_spectrum, sound_horizon

    if mcmc_config is None:
        mcmc_config = {
            'n_steps': 10000, 'n_burn': 2000,
            'alpha_init': 1.0, 'Sigma_init': 5.0, 'B_init': 1.0,
            'alpha_prior': [0.8, 1.2],
            'Sigma_prior': [0.0, 20.0],
            'B_prior': [0.5, 2.0],
        }

    # Build template P(k) functions
    k_th = np.logspace(-3, 0, 1000)
    Pk_lin = power_spectrum(k_th, h, Omega_m, Omega_b, n_s, sigma8, z=z, wiggle=True)
    Pk_nw  = power_spectrum(k_th, h, Omega_m, Omega_b, n_s, sigma8, z=z, wiggle=False)

    Pk_lin_func = interp1d(k_th, Pk_lin, bounds_error=False, fill_value=0.0)
    Pk_nw_func  = interp1d(k_th, Pk_nw,  bounds_error=False, fill_value=0.0)

    r_s_fid = sound_horizon(h, Omega_m, Omega_b)

    theta_init = [
        mcmc_config['alpha_init'],
        mcmc_config['Sigma_init'],
        mcmc_config['B_init'],
    ]
    priors = {
        'alpha': mcmc_config['alpha_prior'],
        'Sigma': mcmc_config['Sigma_prior'],
        'B':     mcmc_config['B_prior'],
    }

    chain, log_probs, accept_rate = run_mcmc(
        k_data, Pk_data, Pk_err,
        Pk_lin_func, Pk_nw_func,
        theta_init, priors,
        n_steps=mcmc_config['n_steps'],
        n_burn=mcmc_config['n_burn'],
    )

    alpha_med = np.median(chain[:, 0])
    r_s_fit = r_s_fid / alpha_med
    print(f"\nRecovered r_s = {r_s_fit:.2f} Mpc/h  (fiducial: {r_s_fid:.2f} Mpc/h)")

    # Corner plot
    labels = [r'$\alpha$', r'$\Sigma$ [Mpc/$h$]', r'$B$']
    fname = os.path.join(output_dir, f'corner_{label}.png')
    make_corner_plot(chain, labels, truths=[1.0, None, 1.0], fname=fname)

    # Best-fit P(k) plot
    alpha_bf, Sigma_bf, B_bf = np.median(chain, axis=0)
    Pk_bestfit = bao_template(k_data, alpha_bf, Sigma_bf, B_bf, Pk_lin_func, Pk_nw_func)

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

    return chain, r_s_fit
