# Codebase Overview

_Auto-generated from `src/`._

## Files

- [`compute_xi.py`](#compute-xi)
- [`initial_conditions.py`](#initial-conditions)
- [`lognormal.py`](#lognormal)
- [`main.py`](#main)
- [`mcmc.py`](#mcmc)
- [`nbody.py`](#nbody)
- [`pk_input.py`](#pk-input)
- [`pm_gravity.py`](#pm-gravity)
- [`power_spectrum.py`](#power-spectrum)
- [`run_bao_marg.py`](#run-bao-marg)
- [`utils.py`](#utils)

## Modules

### `src/compute_xi.py` (233 lines)

> compute and plot the two-point correlation function xi(r) for the

**Imports:** h5py, lognormal.generate_mock_xi, matplotlib, matplotlib.pyplot, numpy, os, pk_input.power_spectrum, pk_input.sound_horizon, pm_gravity.cic_paint_vectorized, power_spectrum.estimate_xi, power_spectrum.estimate_xi_from_delta, power_spectrum.xi_from_pk, pyrecon.IterativeFFTReconstruction, scipy.interpolate.UnivariateSpline, sys, utils.load_config


### `src/initial_conditions.py` (143 lines)

> initial_conditions.py

**Imports:** numpy, pk_input.growth_factor, pk_input.power_spectrum

**Functions:**

- `make_grid(N, L)`
- `generate_gaussian_field(N, L, Pk_func, seed=42)`
- `displacement_field(delta_k, N, L)`
- `hubble(z, h, Omega_m, Omega_lambda=None)`
- `growth_rate(z, Omega_m)`
- `make_ics(N, L, h, Omega_m, Omega_b, n_s, sigma8, z_initial=49.0, seed=42)`

**Private helpers:** `_k_grids`


### `src/lognormal.py` (281 lines)

> lognormal.py

**Imports:** numpy

**Functions:**

- `pk_to_xi(k, Pk, r_grid)`
- `xi_to_pk(r, xi, k_grid)`
- `galaxy_pk_to_gaussian_pk(k, Pk_galaxy, N_mesh, L)`
- `generate_lognormal_field(N_mesh, L, Pk_G_func, seed=42)`
- `poisson_sample(delta_LN, nbar, L, seed=42)`
- `poisson_sample_vectorized(delta_LN, nbar, L, seed=42)`
- `generate_lognormal_catalog(N_mesh, L, h, Omega_m, Omega_b, n_s, sigma8, nbar, b=1.0, z=0.38, seed=42)`
- `generate_mock_covariance(N_mocks, N_mesh, L, h, Omega_m, Omega_b, n_s, sigma8, nbar, b=1.0, z=0.38, seed_start=1000, k_max=0.3)`
- `generate_mock_xi(N_mocks, N_mesh, L, h, Omega_m, Omega_b, n_s, sigma8, nbar, b=1.0, z=0.0, seed_start=2000, r_max=200.0, n_bins=60)`


### `src/main.py` (354 lines)

> main.py

**Imports:** argparse, initial_conditions.make_ics, lognormal.generate_lognormal_catalog, mcmc.fit_bao, nbody.run_nbody, numpy, os, pk_input.power_spectrum, pk_input.sound_horizon, power_spectrum.estimate_pk, utils.load_config, utils.make_animation, utils.pk_error_gaussian, utils.plot_density_slices, utils.plot_pk_comparison

**Functions:**

- `stage_ics(cfg, cosmo, box, sim)`
- `stage_nbody(pos, vel, cfg, cosmo, box, sim, out)`
- `stage_lognormal(cfg, cosmo, box, gal, out)`
- `stage_pk(snapshots, pos_ln, cfg, cosmo, box, gal, out)`
- `stage_recon(pos_nbody_z0, cfg, cosmo, box, gal, out)`
- `stage_mcmc(pk_results, pos_recon, cfg, cosmo, box, gal, out)`
- `stage_plots(snapshots, pk_results, pos_recon, cfg, cosmo, box, gal, out)`
- `main()`


### `src/mcmc.py` (422 lines)

> mcmc.py

**Imports:** corner, matplotlib.pyplot, numpy, os, scipy.interpolate.interp1d

**Functions:**

- `bao_template(k, alpha, Sigma, B, Pk_lin_func, Pk_nw_func)`
- `wiggle_template(k, alpha, Sigma, Pk_lin_func, Pk_nw_func)`
- `broadband_basis(k, poly_powers=(-2, -1, 0, 1, 2))`
- `log_likelihood_marginalized(params, k_data, Pk_data, Pk_err, Pk_lin_func, Pk_nw_func, projector)`
- `bestfit_broadband(k_data, Pk_data, alpha, Sigma, Pk_lin_func, Pk_nw_func, inv_cov, B_basis)`
- `log_likelihood(params, k_data, Pk_data, Pk_err, Pk_lin_func, Pk_nw_func, inv_cov=None)`
- `log_prior(params, priors)`
- `log_posterior_marginalized(params, k_data, Pk_data, Pk_err, Pk_lin_func, Pk_nw_func, priors, projector)`
- `log_posterior(params, k_data, Pk_data, Pk_err, Pk_lin_func, Pk_nw_func, priors, inv_cov=None)`
- `run_mcmc_marginalized(k_data, Pk_data, Pk_err, Pk_lin_func, Pk_nw_func, theta_init, priors, projector, n_steps=10000, n_burn=2000, step_size=None, seed=42)`
- `run_mcmc(k_data, Pk_data, Pk_err, Pk_lin_func, Pk_nw_func, theta_init, priors, n_steps=10000, n_burn=2000, step_size=None, seed=42, inv_cov=None)`
- `make_corner_plot(chain, labels, truths=None, fname=None)`
- `fit_bao(k_data, Pk_data, Pk_err, h, Omega_m, Omega_b, n_s, sigma8, z=0.0, mcmc_config=None, label='', output_dir='outputs/mcmc', cov=None, hartlap_factor=1.0, broadband_marginalize=True, poly_powers=(-2, -1, 0, 1, 2))`

**Private helpers:** `_build_projector`, `_build_projector_diag`


### `src/nbody.py` (152 lines)

> nbody.py

**Imports:** h5py, initial_conditions.hubble, numpy, os, pm_gravity.compute_particle_forces, tqdm.tqdm

**Functions:**

- `scale_factor_steps(z_initial, z_final, n_steps)`
- `save_snapshot(pos, vel, delta, a, step, output_dir)`
- `load_snapshot(fname)`
- `run_nbody(pos, vel, N_mesh, L, h, Omega_m, z_initial=49.0, z_final=0.0, n_steps=50, z_snapshots=None, output_dir='outputs/snapshots', save=True)`

**Private helpers:** `_find_snapshot_steps`


### `src/pk_input.py` (194 lines)

> pk_input.py

**Imports:** numpy

**Functions:**

- `sound_horizon(h, Omega_m, Omega_b)`
- `transfer_function_eh(k, h, Omega_m, Omega_b)`
- `transfer_function_nowiggle(k, h, Omega_m, Omega_b)`
- `power_spectrum(k, h, Omega_m, Omega_b, n_s, sigma8, z=0.0, wiggle=True)`
- `growth_factor(z, Omega_m, Omega_lambda=None)`

**Private helpers:** `_compute_sigma8`


### `src/pm_gravity.py` (144 lines)

> pm_gravity.py

**Imports:** numpy

**Functions:**

- `cic_paint(pos, N_mesh, L)`
- `cic_paint_vectorized(pos, N_mesh, L)`
- `compute_forces(delta, N_mesh, L, h, Omega_m, a)`
- `cic_interpolate(forces, pos, N_mesh, L)`
- `compute_particle_forces(pos, N_mesh, L, h, Omega_m, a)`


### `src/power_spectrum.py` (174 lines)

> power_spectrum.py

**Imports:** numpy

**Functions:**

- `cic_window_correction_1d(k_component, dx)`
- `estimate_pk(pos, N, L, n_mesh=None, subtract_shotnoise=True)`
- `pk_from_snapshot(snap, N, L, n_mesh=None)`
- `estimate_xi(pos, N, L, n_mesh=None, r_max=200.0, n_bins=50)`
- `estimate_xi_from_delta(delta, L, nbar=None, r_max=200.0, n_bins=50)`
- `xi_from_pk(k, Pk, r_grid)`

**Private helpers:** `_xi_from_delta_grid`


### `src/run_bao_marg.py` (135 lines)

> run broadband-marginalized MCMC on saved pre-recon and post-recon P(k).

**Imports:** mcmc.fit_bao, numpy, os, pk_input.sound_horizon, sys, utils.load_config

**Functions:**

- `apply_cut(k, Pk, nm, kmin, kmax)`
- `thin_bins(k, Pk, nm, n_target)`
- `match_cov(k_data, k_cov, cov)`


### `src/utils.py` (178 lines)

> utils.py

**Imports:** numpy, os, scipy.integrate.quad, yaml

**Functions:**

- `comoving_distance(z, h, Omega_m, Omega_de=None, w0=-1.0, wa=0.0)`
- `angular_diameter_distance(z, h, Omega_m, **kwargs)`
- `hubble_z(z, h, Omega_m, Omega_de=None, w0=-1.0, wa=0.0)`
- `load_config(fname)`
- `plot_pk_comparison(k_list, Pk_list, labels, colors=None, title='', fname=None, k_range=(0.01, 0.4))`
- `plot_density_slices(snapshots, L, fname=None)`
- `make_animation(snapshot_dir, output_fname, L, fps=5)`
- `pk_error_gaussian(Pk, nmodes)`

