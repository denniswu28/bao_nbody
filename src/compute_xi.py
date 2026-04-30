"""
Compute and plot the two-point correlation function xi(r) from
N-body (pre-recon) and reconstructed (post-recon) density fields.

Shows the BAO bump at r ~ 100-110 Mpc/h in configuration space.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

from utils import load_config
from power_spectrum import estimate_xi, estimate_xi_from_delta, xi_from_pk
from pk_input import power_spectrum, sound_horizon
from lognormal import generate_mock_xi

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg = load_config(os.path.join(root, 'configs', 'default.yaml'))
cosmo = cfg['cosmology']
box = cfg['box']
fig_dir = os.path.join(root, 'outputs', 'figures')
snap_dir = os.path.join(root, 'outputs', 'snapshots')
mcmc_dir = os.path.join(root, 'outputs', 'mcmc')
os.makedirs(fig_dir, exist_ok=True)

L = box['L']
N = box['N']
N_mesh = box['N_mesh']
XI_BINS = 60
R_MAX = 200.0

# ---- Theory xi(r) via Hankel of analytic P(k) ----
print("Computing theory xi(r) ...")
k_th = np.logspace(-4, 1, 10000)
Pk_w  = power_spectrum(k_th, cosmo['h'], cosmo['Omega_m'], cosmo['Omega_b'],
                       cosmo['n_s'], cosmo['sigma8'], z=0.0, wiggle=True)
Pk_nw = power_spectrum(k_th, cosmo['h'], cosmo['Omega_m'], cosmo['Omega_b'],
                       cosmo['n_s'], cosmo['sigma8'], z=0.0, wiggle=False)

r_theory = np.linspace(1, R_MAX, 500)
xi_w  = xi_from_pk(k_th, Pk_w, r_theory)
xi_nw = xi_from_pk(k_th, Pk_nw, r_theory)

rs_fid = sound_horizon(cosmo['h'], cosmo['Omega_m'], cosmo['Omega_b'])  # Mpc/h
print(f"  r_s = {rs_fid:.2f} Mpc/h")

# ---- N-body xi(r) at z=0 (pre-recon) via FFT ----
print("Computing xi(r) from N-body z=0 snapshot ...")
snap_file = os.path.join(snap_dir, 'snap_0050_z0.00.h5')
with h5py.File(snap_file, 'r') as f:
    pos_nb = f['pos'][:]

r_pre, xi_pre, np_pre = estimate_xi(pos_nb, N, L, n_mesh=N_mesh,
                                     r_max=R_MAX, n_bins=XI_BINS)
print(f"  Done: {len(r_pre)} bins, xi(100)={np.interp(100, r_pre, xi_pre):.4f}")

# ---- Mock-averaged xi(r) from 100 lognormal mocks ----
gal = cfg['galaxy']
N_MOCKS = 100
print(f"\nAveraging xi(r) from {N_MOCKS} lognormal mocks ...")
r_mock, xi_mock_mean, xi_mock_std, xi_mock_all = generate_mock_xi(
    N_MOCKS, N_mesh, L,
    cosmo['h'], cosmo['Omega_m'], cosmo['Omega_b'],
    cosmo['n_s'], cosmo['sigma8'],
    nbar=gal['nbar'], b=gal['b'], z=cosmo['z_eff'],
    seed_start=2000, r_max=R_MAX, n_bins=XI_BINS)

# SNR of mock-averaged curve (computed after BAO amplitude is known)
m_mock_bao = (r_mock > 80) & (r_mock < 160)
noise_mock_per_bin = np.mean(xi_mock_std[m_mock_bao]) * np.mean(r_mock[m_mock_bao]**2)
print(f"  Mock-average noise per bin ~ {noise_mock_per_bin:.1f} (Mpc/h)^2")

# ---- Post-recon xi(r) via FFT of D-R density field ----
print("Running reconstruction to get delta_rec ...")
from pyrecon import IterativeFFTReconstruction
from pm_gravity import cic_paint_vectorized

f_growth = cosmo['Omega_m']**0.55
bias = 1.0
smoothing = 15.0

Npart = pos_nb.shape[1]
Nrand = 10 * Npart
rng = np.random.default_rng(99)
pos_rand = rng.uniform(0, L, (Nrand, 3))
pos_data = (pos_nb + L / 2).T  # (N,3) for pyrecon, shift to [0,L]

recon = IterativeFFTReconstruction(
    f=f_growth, bias=bias, nmesh=N_mesh, boxsize=L, boxcenter=L/2, wrap=True)
recon.assign_data(pos_data)
recon.assign_randoms(pos_rand)
recon.set_density_contrast(smoothing_radius=smoothing)
recon.run()

pos_data_s = recon.read_shifted_positions(pos_data, field='disp')
pos_rand_s = recon.read_shifted_positions(pos_rand, field='disp')

pos_data_cic = (pos_data_s.T - L/2) % L - L/2
pos_rand_cic = (pos_rand_s.T - L/2) % L - L/2

delta_D = cic_paint_vectorized(pos_data_cic, N_mesh, L)
delta_R = cic_paint_vectorized(pos_rand_cic, N_mesh, L)
delta_rec = delta_D - delta_R
print(f"  delta_rec std = {np.std(delta_rec):.4f}")

print("Computing xi(r) from reconstructed field ...")
r_rec, xi_rec, np_rec = estimate_xi_from_delta(delta_rec, L, nbar=None,
                                                r_max=R_MAX, n_bins=XI_BINS)
print(f"  Done: {len(r_rec)} bins, xi(100)={np.interp(100, r_rec, xi_rec):.4f}")

# ---- Compute BAO SNR ----
xi_diff_th = xi_w - xi_nw
m_bao = (r_theory > 80) & (r_theory < 160)
bao_amplitude = np.max(r_theory[m_bao]**2 * xi_diff_th[m_bao]) - \
                np.min(r_theory[m_bao]**2 * xi_diff_th[m_bao])
m_noise = (r_pre > 80) & (r_pre < 160)
noise = np.std(r_pre[m_noise]**2 * xi_pre[m_noise])
snr = bao_amplitude / (2 * noise)
print(f"\n  BAO signal (peak-trough in r²Δξ): {bao_amplitude:.1f} (Mpc/h)²")
print(f"  N-body noise σ(r²ξ) at BAO scales: {noise:.1f} (Mpc/h)²")
print(f"  SNR = {snr:.2f}")

# Mock-averaged SNR.
# The noise on the mean of N_MOCKS independent realizations is
# sigma_mean(r) = xi_std(r) / sqrt(N_MOCKS).  Average r^2 * sigma_mean
# across the BAO bins to get a representative noise amplitude.
sigma_mean = xi_mock_std[m_mock_bao] / np.sqrt(N_MOCKS)
noise_mock = np.mean(r_mock[m_mock_bao]**2 * sigma_mean)
snr_mock = bao_amplitude / (2 * noise_mock) if noise_mock > 0 else 0
print(f"  Mock-averaged xi noise ~ {noise_mock:.1f} (Mpc/h)^2")
print(f"  Mock-averaged xi SNR ~ {snr_mock:.1f}")

# ---- Plot ----
fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.08})

R_MIN_PLOT = 50

ax = axes[0]
# Theory
ax.plot(r_theory, r_theory**2 * xi_w, 'k-', lw=2, alpha=0.7,
        label='Linear theory (wiggles)')
ax.plot(r_theory, r_theory**2 * xi_nw, 'k--', lw=1.5, alpha=0.5,
        label='No-wiggle reference')

# N-body (FFT)
m_pre = r_pre >= R_MIN_PLOT
ax.plot(r_pre[m_pre], r_pre[m_pre]**2 * xi_pre[m_pre], 'C0o-', ms=4, lw=1,
        alpha=0.8, label=f'N-body $z=0$ ($N={N}^3$)')

# Mock-averaged (100 lognormal mocks)
m_mock = r_mock >= R_MIN_PLOT
xi_mean_err = xi_mock_std / np.sqrt(N_MOCKS)
ax.fill_between(r_mock[m_mock],
                r_mock[m_mock]**2 * (xi_mock_mean[m_mock] - xi_mean_err[m_mock]),
                r_mock[m_mock]**2 * (xi_mock_mean[m_mock] + xi_mean_err[m_mock]),
                color='C2', alpha=0.25)
ax.plot(r_mock[m_mock], r_mock[m_mock]**2 * xi_mock_mean[m_mock],
        'C2D-', ms=3, lw=1.2, alpha=0.9,
        label=f'Mean of {N_MOCKS} lognormal mocks')

# Galaxy-bias theory for reference
gal_b = gal['b']
ax.plot(r_theory, r_theory**2 * gal_b**2 * xi_w, 'C2--', lw=1, alpha=0.4,
        label=f'$b^2$ theory ($b={gal_b}$)')

# Post-recon (FFT of D-R field)
m_rec = r_rec >= R_MIN_PLOT
ax.plot(r_rec[m_rec], r_rec[m_rec]**2 * xi_rec[m_rec], 'C3s-', ms=4, lw=1,
        alpha=0.8, label='Post-recon (D$-$R)')

ax.axvline(rs_fid, color='gray', ls=':', alpha=0.5,
           label=f'$r_s$ = {rs_fid:.0f} Mpc/$h$')
ax.axhline(0, color='gray', lw=0.5, alpha=0.3)

# Annotate SNR
ax.text(0.03, 0.05,
        f'Single N-body SNR = {snr:.1f}\n'
        f'Mock-avg SNR ~ {snr_mock:.1f} ({N_MOCKS} mocks)\n'
        f'BAO signal: {bao_amplitude:.1f} (Mpc/$h$)²\n'
        f'$N={N}^3$ in $({L:.0f}$ Mpc/$h)^3$',
        transform=ax.transAxes, fontsize=8, va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.8))

ax.set_ylabel(r'$r^2 \, \xi(r)$ [(Mpc/$h$)$^2$]', fontsize=12)
ax.legend(fontsize='small', loc='upper right')
ax.grid(True, alpha=0.2)
ax.set_title('Correlation Function: BAO Bump in Configuration Space', fontsize=13)
ax.set_xlim(R_MIN_PLOT, R_MAX)

# Lower panel: wiggle-only component (theory) + measured residuals
ax2 = axes[1]
ax2.fill_between(r_theory, r_theory**2 * xi_diff_th, 0,
                 color='gray', alpha=0.2, label='Theory BAO feature')
ax2.plot(r_theory, r_theory**2 * xi_diff_th, 'k-', lw=1.5, alpha=0.6)

# Measured: subtract smooth spline to extract oscillatory residual
from scipy.interpolate import UnivariateSpline
for r_data, xi_data, color, marker, label in [
    (r_pre, xi_pre, 'C0', 'o', 'Pre-recon residual'),
    (r_mock, xi_mock_mean, 'C2', 'D', f'Mock mean ({N_MOCKS})'),
    (r_rec, xi_rec, 'C3', 's', 'Post-recon residual'),
]:
    try:
        m = r_data >= R_MIN_PLOT
        spl = UnivariateSpline(r_data[m], r_data[m]**2 * xi_data[m],
                               s=len(r_data[m]) * 5)
        ax2.plot(r_data[m], r_data[m]**2 * xi_data[m] - spl(r_data[m]),
                 color=color, marker=marker, ms=2, lw=0.8, alpha=0.6, label=label)
    except Exception:
        pass

ax2.axhline(0, color='gray', ls=':', alpha=0.5)
ax2.axvline(rs_fid, color='gray', ls=':', alpha=0.5)
ax2.set_xlabel(r'$r$ [Mpc/$h$]', fontsize=12)
ax2.set_ylabel(r'$r^2 \, \Delta\xi(r)$', fontsize=12)
ax2.legend(fontsize='x-small')
ax2.grid(True, alpha=0.2)

fig.tight_layout()
fname = os.path.join(fig_dir, 'xi_correlation_function.png')
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {fname}")

# Save data
np.savez(os.path.join(mcmc_dir, 'xi_data.npz'),
         r_pre=r_pre, xi_pre=xi_pre,
         r_rec=r_rec, xi_rec=xi_rec,
         r_mock=r_mock, xi_mock_mean=xi_mock_mean,
         xi_mock_std=xi_mock_std, xi_mock_all=xi_mock_all,
         r_theory=r_theory, xi_theory_w=xi_w, xi_theory_nw=xi_nw,
         rs_fid=rs_fid, bao_snr=snr, bao_snr_mock=snr_mock)
print("Saved xi data to outputs/mcmc/xi_data.npz")
