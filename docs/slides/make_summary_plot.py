"""
final pipeline summary figure for the PHY 305 presentation.
4-panel layout: density evolution, P(k), wiggle ratio, alpha posterior.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pk_input import power_spectrum, sound_horizon
from utils import load_config

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg = load_config(os.path.join(root, 'configs', 'default.yaml'))
cosmo = cfg['cosmology']

snap_dir = os.path.join(root, 'outputs', 'snapshots')
mcmc_dir = os.path.join(root, 'outputs', 'mcmc')
fig_dir  = os.path.join(root, 'outputs', 'figures')
os.makedirs(fig_dir, exist_ok=True)


fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, hspace=0.32, wspace=0.30)

# density field z=5.4 and z=0
ax_a1 = fig.add_subplot(gs[0, 0])

# load first and last snapshots
snap_first = sorted([f for f in os.listdir(snap_dir) if f.endswith('.h5')])[0]
snap_last  = sorted([f for f in os.listdir(snap_dir) if f.endswith('.h5')])[-1]

with h5py.File(os.path.join(snap_dir, snap_first), 'r') as f:
    delta_early = f['delta'][:]
    z_early = float(f.attrs['z'])
with h5py.File(os.path.join(snap_dir, snap_last), 'r') as f:
    delta_late = f['delta'][:]
    z_late = float(f.attrs['z'])

L = cfg['box']['L']
slab = delta_early.shape[2] // 8

# side-by-side: split the axis in two
from mpl_toolkits.axes_grid1 import make_axes_locatable

proj_early = np.log10(np.clip(delta_early[:, :, :slab].mean(axis=2) + 1, 1e-2, None))
proj_late  = np.log10(np.clip(delta_late[:, :, :slab].mean(axis=2) + 1, 1e-2, None))
vmin, vmax = -0.3, max(np.percentile(proj_late, 99.5), 1.5)

# single axis with two images pasted side by side
combined = np.concatenate([proj_early, proj_late], axis=1)
im = ax_a1.imshow(combined, cmap='inferno', origin='lower',
                  extent=[0, 2*L, 0, L], vmin=vmin, vmax=vmax, aspect='equal')
ax_a1.axvline(L, color='white', lw=1, ls='--', alpha=0.7)
ax_a1.text(L*0.5, L*0.92, f'z = {z_early:.1f}', color='white', fontsize=11,
           ha='center', va='top', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))
ax_a1.text(L*1.5, L*0.92, f'z = {z_late:.1f}', color='white', fontsize=11,
           ha='center', va='top', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))
ax_a1.set_xlabel('Mpc/$h$')
ax_a1.set_ylabel('Mpc/$h$')
ax_a1.set_title('(a) Density field evolution', fontsize=12)
plt.colorbar(im, ax=ax_a1, label=r'$\log_{10}(1+\delta)$', fraction=0.046, pad=0.04)

# P(k) evolution
ax_b = fig.add_subplot(gs[0, 1])

# load recon_pk for pre-recon P(k)
recon = np.load(os.path.join(mcmc_dir, 'recon_pk.npz'))
k_pre, Pk_pre = recon['k_pre'], recon['Pk_pre']
k_rec, Pk_rec = recon['k_rec'], recon['Pk_rec']
k_th, Pk_w, Pk_nw_arr = recon['k_th'], recon['Pk_w'], recon['Pk_nw']

ax_b.loglog(k_th, Pk_w, 'k-', alpha=0.4, lw=1.5, label='Linear theory (wiggles)')
ax_b.loglog(k_th, Pk_nw_arr, 'k--', alpha=0.3, lw=1, label='No-wiggle')
mask = k_pre > 0.01
ax_b.loglog(k_pre[mask], Pk_pre[mask], 'C0-', alpha=0.7, lw=1, label='N-body z=0')
mask = k_rec > 0.01
ax_b.loglog(k_rec[mask], Pk_rec[mask], 'C3-', alpha=0.7, lw=1, label='Post-recon')
ax_b.set_xlabel(r'$k$ [$h$/Mpc]')
ax_b.set_ylabel(r'$P(k)$ [$(h^{-1}$Mpc$)^3$]')
ax_b.set_xlim(0.01, 0.5)
ax_b.legend(fontsize='x-small', loc='lower left')
ax_b.grid(True, alpha=0.2)
ax_b.set_title('(b) Power spectrum', fontsize=12)

# BAO wiggles ratio
ax_c = fig.add_subplot(gs[1, 0])

# interpolate P_nw at data k values
from scipy.interpolate import interp1d
Pnw_func = interp1d(k_th, Pk_nw_arr, bounds_error=False, fill_value='extrapolate')
Pw_func  = interp1d(k_th, Pk_w, bounds_error=False, fill_value='extrapolate')

kmin, kmax = 0.02, 0.35
mask_pre = (recon['k_pre'] > kmin) & (recon['k_pre'] < kmax)
mask_rec = (recon['k_rec'] > kmin) & (recon['k_rec'] < kmax)

k_p = recon['k_pre'][mask_pre]
k_r = recon['k_rec'][mask_rec]
ratio_pre = recon['Pk_pre'][mask_pre] / Pnw_func(k_p)
ratio_rec = recon['Pk_rec'][mask_rec] / Pnw_func(k_r)
ratio_th  = Pw_func(k_th) / Pnw_func(k_th)

ax_c.plot(k_th, ratio_th, 'k-', alpha=0.3, lw=1.5, label='Linear theory')

# bin the ratios for clarity (running mean over groups of ~5)
def running_mean(k, y, n=5):
    pad = len(y) % n
    if pad:
        k, y = k[:len(y)-pad], y[:len(y)-pad]
    return k.reshape(-1, n).mean(axis=1), y.reshape(-1, n).mean(axis=1)

k_p_b, r_p_b = running_mean(k_p, ratio_pre, 5)
k_r_b, r_r_b = running_mean(k_r, ratio_rec, 5)
ax_c.plot(k_p_b, r_p_b, 'C0o-', ms=3, alpha=0.6, lw=0.8, label='Pre-recon (binned)')
ax_c.plot(k_r_b, r_r_b, 'C3s-', ms=3, alpha=0.6, lw=0.8, label='Post-recon (binned)')
ax_c.axhline(1, color='gray', ls=':', alpha=0.5)
ax_c.set_xlabel(r'$k$ [$h$/Mpc]')
ax_c.set_ylabel(r'$P(k) / P_{\rm nw}(k)$')
ax_c.set_xlim(kmin, kmax)
# auto-range so the binned points are visible
all_ratios = np.concatenate([r_p_b, r_r_b])
ylo = max(0.0, np.percentile(all_ratios, 2) - 0.05)
yhi = min(3.0, np.percentile(all_ratios, 98) + 0.05)
ax_c.set_ylim(ylo, yhi)
ax_c.legend(fontsize='x-small')
ax_c.grid(True, alpha=0.2)
ax_c.set_title('(c) BAO wiggles (ratio to no-wiggle)', fontsize=12)
ax_c.text(0.03, 0.05, 'Wiggles unresolved\nat $N=128^3$',
          transform=ax_c.transAxes, fontsize=8, style='italic',
          va='bottom', color='gray')

# posteriors
ax_d = fig.add_subplot(gs[1, 1])

marg = np.load(os.path.join(mcmc_dir, 'marg_mcmc_results.npz'))
chain_pre = marg['chain_pre']
chain_rec = marg['chain_rec']

# also load old 3-param results if available
old_path = os.path.join(mcmc_dir, 'recon_mcmc_results.npz')
if os.path.exists(old_path):
    old = np.load(old_path)
    if 'chain_pre' in old:
        ax_d.hist(old['chain_pre'][:, 0], bins=40, density=True, alpha=0.3,
                  color='gray', label='Old 3-param (pre)')

bins = np.linspace(0.5, 1.5, 50)
ax_d.hist(chain_pre[:, 0], bins=bins, density=True, alpha=0.5,
          color='C0', label=f'Pre-recon (marg)')
ax_d.hist(chain_rec[:, 0], bins=bins, density=True, alpha=0.5,
          color='C3', label=f'Post-recon (marg)')
ax_d.axvline(1.0, color='k', ls='--', lw=1.5, label=r'$\alpha = 1$ (fiducial)')

# annotate medians
for chain, color, name in [(chain_pre, 'C0', 'pre'), (chain_rec, 'C3', 'post')]:
    med = np.median(chain[:, 0])
    lo, hi = np.percentile(chain[:, 0], [16, 84])
    ax_d.axvline(med, color=color, ls=':', alpha=0.7)

rs_fid = sound_horizon(cosmo['h'], cosmo['Omega_m'], cosmo['Omega_b'])
ax_d.set_xlabel(r'$\alpha = r_s^{\rm fid} / r_s$', fontsize=11)
ax_d.set_ylabel('Posterior density')
ax_d.legend(fontsize='x-small')
ax_d.set_xlim(0.5, 1.5)
ax_d.grid(True, alpha=0.2)
ax_d.set_title(r'(d) BAO dilation parameter $\alpha$', fontsize=12)

# numeric annotations on alpha posteriors
med_pre = np.median(chain_pre[:, 0])
sig_pre = 0.5 * (np.percentile(chain_pre[:, 0], 84) - np.percentile(chain_pre[:, 0], 16))
med_rec = np.median(chain_rec[:, 0])
sig_rec = 0.5 * (np.percentile(chain_rec[:, 0], 84) - np.percentile(chain_rec[:, 0], 16))
ax_d.text(0.97, 0.95,
          f'Pre:  α = {med_pre:.2f} ± {sig_pre:.2f}\n'
          f'Post: α = {med_rec:.2f} ± {sig_rec:.2f}\n'
          f'$r_s^{{\\rm fid}}$ = {rs_fid:.1f} Mpc/h',
          transform=ax_d.transAxes, fontsize=9,
          va='top', ha='right',
          bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

# save
fig.suptitle('BAO Analysis Pipeline: $N$-body → Reconstruction → MCMC',
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig(os.path.join(fig_dir, 'pipeline_summary.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_dir}/pipeline_summary.png")
