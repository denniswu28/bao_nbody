"""
split the 4-panel pipeline summary into 4 standalone slide plots.
outputs into outputs/figures/: panel_density.png, panel_pk.png,
panel_wiggles.png, panel_alpha.png.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from pk_input import sound_horizon
from utils import load_config

# --------------------------------------------------------------- setup
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg = load_config(os.path.join(root, 'configs', 'default.yaml'))
cosmo = cfg['cosmology']
L = cfg['box']['L']

snap_dir = os.path.join(root, 'outputs', 'snapshots')
mcmc_dir = os.path.join(root, 'outputs', 'mcmc')
fig_dir  = os.path.join(root, 'outputs', 'figures')
os.makedirs(fig_dir, exist_ok=True)

# bigger default fonts so the slides are legible
plt.rcParams.update({
    'font.size':        16,
    'axes.labelsize':   18,
    'axes.titlesize':   18,
    'xtick.labelsize':  14,
    'ytick.labelsize':  14,
    'legend.fontsize':  14,
    'lines.linewidth':  2.0,
    'axes.linewidth':   1.2,
    'xtick.major.width':1.2,
    'ytick.major.width':1.2,
})

DPI = 250

# panel 1: density field z = early vs z = 0
snap_files = sorted([f for f in os.listdir(snap_dir) if f.endswith('.h5')])
snap_first, snap_last = snap_files[0], snap_files[-1]
with h5py.File(os.path.join(snap_dir, snap_first), 'r') as f:
    delta_early = f['delta'][:]
    z_early = float(f.attrs['z'])
with h5py.File(os.path.join(snap_dir, snap_last), 'r') as f:
    delta_late = f['delta'][:]
    z_late = float(f.attrs['z'])

slab = delta_early.shape[2] // 8
proj_early = np.log10(np.clip(delta_early[:, :, :slab].mean(axis=2) + 1, 1e-2, None))
proj_late  = np.log10(np.clip(delta_late [:, :, :slab].mean(axis=2) + 1, 1e-2, None))
vmin = -0.3
vmax = max(np.percentile(proj_late, 99.5), 1.5)

fig, ax = plt.subplots(figsize=(8, 4.2))
combined = np.concatenate([proj_early, proj_late], axis=1)
im = ax.imshow(combined, cmap='inferno', origin='lower',
               extent=[0, 2 * L, 0, L], vmin=vmin, vmax=vmax, aspect='equal')
ax.axvline(L, color='white', lw=1.5, ls='--', alpha=0.8)
ax.text(L * 0.5, L * 0.93, f'z = {z_early:.1f}', color='white', fontsize=18,
        ha='center', va='top', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', fc='black', alpha=0.55))
ax.text(L * 1.5, L * 0.93, f'z = {z_late:.1f}', color='white', fontsize=18,
        ha='center', va='top', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', fc='black', alpha=0.55))
ax.set_xlabel('Mpc / $h$')
ax.set_ylabel('Mpc / $h$')
cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label(r'$\log_{10}(1+\delta)$')
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'panel_density.png'),
            dpi=DPI, bbox_inches='tight')
plt.close(fig)

# panel 2: P(k)
recon = np.load(os.path.join(mcmc_dir, 'recon_pk.npz'))
k_pre, Pk_pre = recon['k_pre'], recon['Pk_pre']
k_rec, Pk_rec = recon['k_rec'], recon['Pk_rec']
k_th, Pk_w, Pk_nw = recon['k_th'], recon['Pk_w'], recon['Pk_nw']

fig, ax = plt.subplots(figsize=(7.5, 5.0))
ax.loglog(k_th, Pk_w,  'k-',  alpha=0.55, lw=2.0, label='Linear theory')
ax.loglog(k_th, Pk_nw, 'k--', alpha=0.35, lw=1.5, label='No-wiggle')
mask = k_pre > 0.01
ax.loglog(k_pre[mask], Pk_pre[mask], color='C0', alpha=0.85, lw=2.0,
          label='$N$-body $z=0$')
mask = k_rec > 0.01
ax.loglog(k_rec[mask], Pk_rec[mask], color='C3', alpha=0.85, lw=2.0,
          label='Post-reconstruction')
ax.set_xlabel(r'$k$ [$h$/Mpc]')
ax.set_ylabel(r'$P(k)$ [$(h^{-1}\mathrm{Mpc})^3$]')
ax.set_xlim(0.01, 0.5)
ax.legend(loc='lower left', frameon=True)
ax.grid(True, which='both', alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'panel_pk.png'),
            dpi=DPI, bbox_inches='tight')
plt.close(fig)

# panel 3: wiggle ratio
Pnw_f = interp1d(k_th, Pk_nw, bounds_error=False, fill_value='extrapolate')
Pw_f  = interp1d(k_th, Pk_w,  bounds_error=False, fill_value='extrapolate')

kmin, kmax = 0.02, 0.35
mask_p = (k_pre > kmin) & (k_pre < kmax)
mask_r = (k_rec > kmin) & (k_rec < kmax)
k_p, r_p = k_pre[mask_p], Pk_pre[mask_p] / Pnw_f(k_pre[mask_p])
k_r, r_r = k_rec[mask_r], Pk_rec[mask_r] / Pnw_f(k_rec[mask_r])


def running_mean(k, y, n=5):
    pad = len(y) % n
    if pad:
        k, y = k[:len(y) - pad], y[:len(y) - pad]
    return k.reshape(-1, n).mean(axis=1), y.reshape(-1, n).mean(axis=1)


k_pb, r_pb = running_mean(k_p, r_p, 5)
k_rb, r_rb = running_mean(k_r, r_r, 5)
ratio_th = Pw_f(k_th) / Pnw_f(k_th)

fig, ax = plt.subplots(figsize=(7.5, 5.0))
ax.plot(k_th, ratio_th, 'k-', alpha=0.45, lw=2.0, label='Linear theory')
ax.plot(k_pb, r_pb, 'o-', color='C0', ms=6, alpha=0.85, lw=1.5,
        label='Pre-reconstruction')
ax.plot(k_rb, r_rb, 's-', color='C3', ms=6, alpha=0.85, lw=1.5,
        label='Post-reconstruction')
ax.axhline(1, color='gray', ls=':', alpha=0.6, lw=1.2)
ax.set_xlabel(r'$k$ [$h$/Mpc]')
ax.set_ylabel(r'$P(k)\,/\,P_{\rm nw}(k)$')
ax.set_xlim(kmin, kmax)
all_r = np.concatenate([r_pb, r_rb])
ylo = max(0.0, np.percentile(all_r, 2) - 0.05)
yhi = min(3.0, np.percentile(all_r, 98) + 0.05)
ax.set_ylim(ylo, yhi)
ax.legend(loc='best', frameon=True)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'panel_wiggles.png'),
            dpi=DPI, bbox_inches='tight')
plt.close(fig)

# panel 4: alpha posterior
marg = np.load(os.path.join(mcmc_dir, 'marg_mcmc_results.npz'))
chain_pre = marg['chain_pre']
chain_rec = marg['chain_rec']

fig, ax = plt.subplots(figsize=(7.5, 5.0))
bins = np.linspace(0.5, 1.5, 55)
ax.hist(chain_pre[:, 0], bins=bins, density=True, alpha=0.55,
        color='C0', label='Pre-reconstruction')
ax.hist(chain_rec[:, 0], bins=bins, density=True, alpha=0.55,
        color='C3', label='Post-reconstruction')
ax.axvline(1.0, color='k', ls='--', lw=2.0, label=r'$\alpha = 1$ (fiducial)')
for chain, color in [(chain_pre, 'C0'), (chain_rec, 'C3')]:
    ax.axvline(np.median(chain[:, 0]), color=color, ls=':', lw=1.5, alpha=0.85)

rs_fid = sound_horizon(cosmo['h'], cosmo['Omega_m'], cosmo['Omega_b'])
med_p = np.median(chain_pre[:, 0])
sig_p = 0.5 * (np.percentile(chain_pre[:, 0], 84) - np.percentile(chain_pre[:, 0], 16))
med_r = np.median(chain_rec[:, 0])
sig_r = 0.5 * (np.percentile(chain_rec[:, 0], 84) - np.percentile(chain_rec[:, 0], 16))

ax.text(0.97, 0.96,
        f'Pre:  $\\alpha$ = {med_p:.3f} $\\pm$ {sig_p:.3f}\n'
        f'Post: $\\alpha$ = {med_r:.3f} $\\pm$ {sig_r:.3f}',
        transform=ax.transAxes, fontsize=14,
        va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

ax.set_xlabel(r'Ruler stretch $\alpha = r_s^{\rm fid} / r_s^{\rm measured}$')
ax.set_ylabel('Posterior density')
ax.legend(loc='upper left', frameon=True)
ax.set_xlim(0.85, 1.20)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'panel_alpha.png'),
            dpi=DPI, bbox_inches='tight')
plt.close(fig)

print("Saved:")
for name in ['panel_density', 'panel_pk', 'panel_wiggles', 'panel_alpha']:
    print(f"  {fig_dir}/{name}.png")
