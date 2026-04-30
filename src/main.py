"""
main.py
-------
Pipeline orchestrator for the BAO N-body project.

Stages:
    ics       - Generate Zel'dovich initial conditions
    nbody     - Run PM N-body simulation
    lognormal - Generate lognormal catalog
    pk        - Measure P(k) from all snapshots
    recon     - Run BAO reconstruction (via pyrecon)
    mcmc      - Fit BAO template and recover r_s
    plots     - Make all summary figures

Usage:
    python main.py --config configs/default.yaml
    python main.py --config configs/default.yaml --stage nbody
"""

import argparse
import numpy as np
import os

from utils import load_config, plot_pk_comparison, plot_density_slices, pk_error_gaussian, make_animation
from pk_input import power_spectrum, sound_horizon
from initial_conditions import make_ics
from nbody import run_nbody
from lognormal import generate_lognormal_catalog
from power_spectrum import estimate_pk
from mcmc import fit_bao


def stage_ics(cfg, cosmo, box, sim):
    print("\n" + "="*60)
    print("STAGE: Initial Conditions (Zel'dovich Approximation)")
    print("="*60)
    pos, vel, delta_k = make_ics(
        N=box['N'], L=box['L'],
        h=cosmo['h'], Omega_m=cosmo['Omega_m'],
        Omega_b=cosmo['Omega_b'], n_s=cosmo['n_s'],
        sigma8=cosmo['sigma8'],
        z_initial=sim['z_initial'],
        seed=sim['seed'],
    )
    return pos, vel


def stage_nbody(pos, vel, cfg, cosmo, box, sim, out):
    print("\n" + "="*60)
    print("STAGE: PM N-body Simulation (Leapfrog)")
    print("="*60)
    snapshots, fnames = run_nbody(
        pos, vel,
        N_mesh=box['N_mesh'], L=box['L'],
        h=cosmo['h'], Omega_m=cosmo['Omega_m'],
        z_initial=sim['z_initial'], z_final=0.0,
        n_steps=sim['n_steps'],
        z_snapshots=sim['z_snapshots'],
        output_dir=out['snapshot_dir'],
        save=out['save_snapshots'],
    )
    return snapshots


def stage_lognormal(cfg, cosmo, box, gal, out):
    print("\n" + "="*60)
    print("STAGE: Lognormal Catalog Generation")
    print("="*60)
    pos_ln, delta_ln = generate_lognormal_catalog(
        N_mesh=box['N_mesh'], L=box['L'],
        h=cosmo['h'], Omega_m=cosmo['Omega_m'],
        Omega_b=cosmo['Omega_b'], n_s=cosmo['n_s'],
        sigma8=cosmo['sigma8'],
        nbar=gal['nbar'], b=gal['b'],
        z=cosmo['z_eff'],
        seed=cfg['lognormal']['seed'],
    )
    return pos_ln, delta_ln


def stage_pk(snapshots, pos_ln, cfg, cosmo, box, gal, out):
    print("\n" + "="*60)
    print("STAGE: Power Spectrum Estimation")
    print("="*60)

    results = {}

    # P(k) from each N-body snapshot
    results['nbody'] = []
    for snap in snapshots:
        k, Pk, nmodes = estimate_pk(snap['pos'], box['N'], box['L'],
                                     n_mesh=box['N_mesh'])
        Pk_err = pk_error_gaussian(Pk, nmodes)
        results['nbody'].append({
            'k': k, 'Pk': Pk, 'Pk_err': Pk_err, 'nmodes': nmodes,
            'z': snap['z'], 'a': snap['a'],
        })
        print(f"  N-body z={snap['z']:.2f}: {len(k)} k-bins, "
              f"P(k=0.1) = {np.interp(0.1, k, Pk):.1f} (Mpc/h)^3")

    # P(k) from lognormal catalog
    k, Pk, nmodes = estimate_pk(pos_ln, box['N_mesh'], box['L'],
                                 n_mesh=box['N_mesh']*2)
    Pk_err = pk_error_gaussian(Pk, nmodes)
    results['lognormal'] = {'k': k, 'Pk': Pk, 'Pk_err': Pk_err, 'nmodes': nmodes}
    print(f"  Lognormal: {len(k)} k-bins, "
          f"P(k=0.1) = {np.interp(0.1, k, Pk):.1f} (Mpc/h)^3")

    # Theory
    k_th = np.logspace(-3, 0, 500)
    results['theory'] = {
        'k': k_th,
        'Pk': gal['b']**2 * power_spectrum(k_th, cosmo['h'], cosmo['Omega_m'],
                                             cosmo['Omega_b'], cosmo['n_s'],
                                             cosmo['sigma8'], z=cosmo['z_eff']),
    }

    return results


def stage_recon(pos_nbody_z0, cfg, cosmo, box, gal, out):
    """
    BAO reconstruction using pyrecon (IterativeFFT, Burden et al. 2015).
    Returns the D-R reconstructed P(k) via CIC painting of displaced
    data minus displaced randoms.
    """
    print("\n" + "="*60)
    print("STAGE: BAO Reconstruction (pyrecon)")
    print("="*60)

    try:
        from pyrecon import IterativeFFTReconstruction
    except ImportError:
        print("  pyrecon not installed. Skipping reconstruction.")
        print("  Install with: pip install pyrecon")
        return None

    L = box['L']
    N_mesh = box['N_mesh']
    f = cosmo['Omega_m']**0.55   # growth rate at z=0 (approximate)
    bias = 1.0  # dark matter particles, not galaxies
    smoothing = 15.0  # Mpc/h

    Npart = pos_nbody_z0.shape[1]
    Nrand = 10 * Npart
    rng = np.random.default_rng(99)
    pos_rand = rng.uniform(0, L, (Nrand, 3))

    # Positions are in [-L/2, L/2]; shift to [0, L] for pyrecon
    pos_data = (pos_nbody_z0 + L / 2).T  # (N, 3) for pyrecon

    print(f"  f = {f:.4f}, bias = {bias}, smoothing = {smoothing} Mpc/h")
    print(f"  N_particles = {Npart}, N_randoms = {Nrand}")

    recon = IterativeFFTReconstruction(
        f=f, bias=bias,
        nmesh=N_mesh, boxsize=L, boxcenter=L/2,
        wrap=True,  # periodic box
    )
    recon.assign_data(pos_data)
    recon.assign_randoms(pos_rand)
    recon.set_density_contrast(smoothing_radius=smoothing)
    recon.run()

    # Get displaced data and randoms
    pos_data_s = recon.read_shifted_positions(pos_data, field='disp')
    pos_rand_s = recon.read_shifted_positions(pos_rand, field='disp')

    # Convert to (3, N) in [-L/2, L/2] with periodic wrapping
    from pm_gravity import cic_paint_vectorized
    pos_data_cic = (pos_data_s.T - L/2) % L - L/2
    pos_rand_cic = (pos_rand_s.T - L/2) % L - L/2

    # D-R: paint both and subtract overdensities
    delta_D = cic_paint_vectorized(pos_data_cic, N_mesh, L)
    delta_R = cic_paint_vectorized(pos_rand_cic, N_mesh, L)
    delta_rec = delta_D - delta_R

    print(f"  Reconstruction complete. delta_rec std = {np.std(delta_rec):.4f}")
    return {'pos_data': pos_data_cic, 'delta_rec': delta_rec}


def stage_mcmc(pk_results, pos_recon, cfg, cosmo, box, gal, out):
    print("\n" + "="*60)
    print("STAGE: BAO Template Fitting (MCMC)")
    print("="*60)

    mcmc_cfg = cfg['mcmc']
    chains = {}

    # Try to load lognormal covariance if available
    cov_path = os.path.join(out['mcmc_dir'], 'lognormal_covariance.npz')
    cov_data, cov_full, k_cov, N_mocks = None, None, None, 100
    if os.path.exists(cov_path):
        cov_data = np.load(cov_path)
        cov_full = cov_data['cov']
        k_cov = cov_data['k_bins']
        N_mocks = cov_data['Pk_all'].shape[0] if 'Pk_all' in cov_data else 100
        print(f"  Loaded covariance: {cov_full.shape} from {N_mocks} mocks")

    def _match_cov(k_data, k_cov, cov_full, N_mocks):
        """Select sub-matrix of cov matching k_data bins, return (cov, hartlap).

        Uses nearest-neighbor index selection but refuses to duplicate a
        covariance row (which would make cov_sub singular).  Raises if the
        data k-binning is finer than the mock k-binning.
        """
        idx = np.array([np.argmin(np.abs(k_cov - ki)) for ki in k_data])
        if len(np.unique(idx)) != len(idx):
            raise ValueError(
                f"Data k-binning ({len(k_data)} bins) is finer than "
                f"covariance k-binning ({len(k_cov)} bins); "
                "rebin the data or regenerate the covariance on a matching grid."
            )
        cov_sub = cov_full[np.ix_(idx, idx)]
        hartlap = (N_mocks - len(k_data) - 2) / (N_mocks - 1)
        if hartlap <= 0:
            raise ValueError(
                f"Hartlap factor non-positive: N_mocks={N_mocks}, "
                f"N_bins={len(k_data)}.  Need N_mocks > N_bins + 2."
            )
        return cov_sub, hartlap

    # Fit N-body final snapshot (z=0)
    pk_final = pk_results['nbody'][-1]
    mask = (pk_final['k'] > 0.02) & (pk_final['k'] < 0.3)
    cov_sub, hartlap = (None, 1.0)
    if cov_full is not None:
        cov_sub, hartlap = _match_cov(pk_final['k'][mask], k_cov, cov_full, N_mocks)
    chain, r_s = fit_bao(
        pk_final['k'][mask], pk_final['Pk'][mask], pk_final['Pk_err'][mask],
        cosmo['h'], cosmo['Omega_m'], cosmo['Omega_b'], cosmo['n_s'], cosmo['sigma8'],
        z=0.0, mcmc_config=mcmc_cfg, label='nbody_z0',
        output_dir=out['mcmc_dir'],
        cov=cov_sub, hartlap_factor=hartlap,
        broadband_marginalize=True,
    )
    chains['nbody_z0'] = {'chain': chain, 'r_s': r_s}
    print(f"  N-body z=0: r_s = {r_s:.2f} Mpc/h")

    # Fit lognormal catalog
    pk_ln = pk_results['lognormal']
    mask = (pk_ln['k'] > 0.02) & (pk_ln['k'] < 0.3)
    cov_sub, hartlap = (None, 1.0)
    if cov_full is not None:
        cov_sub, hartlap = _match_cov(pk_ln['k'][mask], k_cov, cov_full, N_mocks)
    chain, r_s = fit_bao(
        pk_ln['k'][mask], pk_ln['Pk'][mask], pk_ln['Pk_err'][mask],
        cosmo['h'], cosmo['Omega_m'], cosmo['Omega_b'], cosmo['n_s'], cosmo['sigma8'],
        z=cosmo['z_eff'], mcmc_config=mcmc_cfg, label='lognormal',
        output_dir=out['mcmc_dir'],
        cov=cov_sub, hartlap_factor=hartlap,
        broadband_marginalize=True,
    )
    chains['lognormal'] = {'chain': chain, 'r_s': r_s}
    print(f"  Lognormal: r_s = {r_s:.2f} Mpc/h")

    # Fit reconstructed catalog (if available)
    if pos_recon is not None:
        from power_spectrum import estimate_pk
        from utils import pk_error_gaussian
        k_r, Pk_r, nm_r = estimate_pk(pos_recon, box['N'], box['L'], n_mesh=box['N_mesh'])
        Pk_r_err = pk_error_gaussian(Pk_r, nm_r)
        mask = (k_r > 0.02) & (k_r < 0.3)
        cov_sub, hartlap = (None, 1.0)
        if cov_full is not None:
            cov_sub, hartlap = _match_cov(k_r[mask], k_cov, cov_full, N_mocks)
        chain, r_s = fit_bao(
            k_r[mask], Pk_r[mask], Pk_r_err[mask],
            cosmo['h'], cosmo['Omega_m'], cosmo['Omega_b'], cosmo['n_s'], cosmo['sigma8'],
            z=0.0, mcmc_config=mcmc_cfg, label='recon',
            output_dir=out['mcmc_dir'],
            cov=cov_sub, hartlap_factor=hartlap,
            broadband_marginalize=True,
        )
        chains['recon'] = {'chain': chain, 'r_s': r_s}
        print(f"  Reconstructed: r_s = {r_s:.2f} Mpc/h")

    r_s_fid = sound_horizon(cosmo['h'], cosmo['Omega_m'], cosmo['Omega_b'])
    print(f"\n  Fiducial sound horizon: r_s = {r_s_fid:.2f} Mpc/h")

    return chains


def stage_plots(snapshots, pk_results, pos_recon, cfg, cosmo, box, gal, out):
    print("\n" + "="*60)
    print("STAGE: Summary Plots")
    print("="*60)

    os.makedirs(out['figure_dir'], exist_ok=True)

    # 1. Density field slices
    fig = plot_density_slices(snapshots, box['L'],
                               fname=os.path.join(out['figure_dir'], 'density_slices.png'))
    print("  Saved: density_slices.png")

    # 2. P(k) evolution
    k_list, Pk_list, labels = [], [], []
    for snap_pk in pk_results['nbody']:
        k_list.append(snap_pk['k'])
        Pk_list.append(snap_pk['Pk'])
        labels.append(f"N-body z={snap_pk['z']:.1f}")
    k_list.append(pk_results['lognormal']['k'])
    Pk_list.append(pk_results['lognormal']['Pk'])
    labels.append('Lognormal')
    k_list.append(pk_results['theory']['k'])
    Pk_list.append(pk_results['theory']['Pk'])
    labels.append('Theory (EH)')

    plot_pk_comparison(k_list, Pk_list, labels,
                        title='BAO Signal: N-body evolution + Lognormal',
                        fname=os.path.join(out['figure_dir'], 'pk_evolution.png'))
    print("  Saved: pk_evolution.png")

    # 3. Density evolution animation
    if out.get('make_animation', True):
        make_animation(out['snapshot_dir'],
                       os.path.join(out['figure_dir'], 'density_evolution.mp4'),
                       L=box['L'],
                       fps=out.get('animation_fps', 5))
        print("  Saved: density_evolution.mp4")


def main():
    parser = argparse.ArgumentParser(description='BAO N-body Pipeline')
    parser.add_argument('--config', default='configs/default.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--stage', default='all',
                        choices=['all', 'ics', 'nbody', 'lognormal', 'pk', 'recon', 'mcmc', 'plots'],
                        help='Pipeline stage to run')
    args = parser.parse_args()

    cfg   = load_config(args.config)
    cosmo = cfg['cosmology']
    box   = cfg['box']
    sim   = cfg['simulation']
    gal   = cfg['galaxy']
    out   = cfg['output']

    os.makedirs(out['snapshot_dir'], exist_ok=True)
    os.makedirs(out['figure_dir'],   exist_ok=True)
    os.makedirs(out['mcmc_dir'],     exist_ok=True)

    r_s_fid = sound_horizon(cosmo['h'], cosmo['Omega_m'], cosmo['Omega_b'])
    print(f"\nFiducial sound horizon r_s = {r_s_fid:.2f} Mpc/h")
    print(f"Box: L={box['L']} Mpc/h, N={box['N']}^3 particles, N_mesh={box['N_mesh']}")

    stage = args.stage

    if stage in ('all', 'ics', 'nbody', 'pk', 'recon', 'mcmc', 'plots'):
        pos, vel = stage_ics(cfg, cosmo, box, sim)

    if stage in ('all', 'nbody', 'pk', 'recon', 'mcmc', 'plots'):
        snapshots = stage_nbody(pos, vel, cfg, cosmo, box, sim, out)

    if stage in ('all', 'lognormal', 'pk', 'mcmc', 'plots'):
        pos_ln, delta_ln = stage_lognormal(cfg, cosmo, box, gal, out)

    if stage in ('all', 'pk', 'mcmc', 'plots'):
        pk_results = stage_pk(snapshots, pos_ln, cfg, cosmo, box, gal, out)

    if stage in ('all', 'recon', 'mcmc', 'plots'):
        pos_nbody_z0 = snapshots[-1]['pos']
        pos_recon = stage_recon(pos_nbody_z0, cfg, cosmo, box, gal, out)

    if stage in ('all', 'mcmc'):
        chains = stage_mcmc(pk_results, pos_recon, cfg, cosmo, box, gal, out)

    if stage in ('all', 'plots'):
        stage_plots(snapshots, pk_results, pos_recon, cfg, cosmo, box, gal, out)

    print("\n Pipeline complete.")


if __name__ == "__main__":
    main()
