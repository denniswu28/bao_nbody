"""
Run broadband-marginalized MCMC on saved pre-recon and post-recon P(k).

Loads data from outputs/mcmc/recon_pk.npz and outputs/mcmc/lognormal_covariance.npz,
then fits BAO with the broadband-marginalized template.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from mcmc import fit_bao
from utils import load_config

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg = load_config(os.path.join(root, 'configs', 'default.yaml'))
cosmo = cfg['cosmology']
mcmc_cfg = cfg['mcmc']
out_dir = os.path.join(root, 'outputs', 'mcmc')
os.makedirs(out_dir, exist_ok=True)

# ---- Load saved P(k) data ----
recon = np.load(os.path.join(out_dir, 'recon_pk.npz'))
k_pre, Pk_pre  = recon['k_pre'], recon['Pk_pre']
nm_pre          = recon['nm_pre']
k_rec, Pk_rec  = recon['k_rec'], recon['Pk_rec']
nm_rec          = recon['nm_rec']

# ---- Load lognormal covariance ----
cov_data = np.load(os.path.join(out_dir, 'lognormal_covariance.npz'))
cov_full = cov_data['cov']
k_cov    = cov_data['k_bins']
N_mocks  = int(cov_data.get('Pk_all', np.zeros((100,1))).shape[0])  # number of mocks

print(f"Loaded P(k): pre-recon {len(k_pre)} bins, post-recon {len(k_rec)} bins")
print(f"Covariance: {cov_full.shape} from {N_mocks} mocks")

# ---- k-range cut and bin thinning ----
kmin, kmax = 0.02, 0.30
N_bins_target = 30  # keep bins manageable for Hartlap correction

def apply_cut(k, Pk, nm, kmin, kmax):
    mask = (k > kmin) & (k < kmax)
    return k[mask], Pk[mask], nm[mask], mask

def thin_bins(k, Pk, nm, n_target):
    """Thin to ~n_target evenly spaced bins."""
    n = len(k)
    if n <= n_target:
        return k, Pk, nm
    step = max(1, n // n_target)
    idx = np.arange(0, n, step)
    return k[idx], Pk[idx], nm[idx]

k_pre_c, Pk_pre_c, nm_pre_c, mask_pre = apply_cut(k_pre, Pk_pre, nm_pre, kmin, kmax)
k_rec_c, Pk_rec_c, nm_rec_c, mask_rec = apply_cut(k_rec, Pk_rec, nm_rec, kmin, kmax)

# Thin bins for better Hartlap factor
k_pre_c, Pk_pre_c, nm_pre_c = thin_bins(k_pre_c, Pk_pre_c, nm_pre_c, N_bins_target)
k_rec_c, Pk_rec_c, nm_rec_c = thin_bins(k_rec_c, Pk_rec_c, nm_rec_c, N_bins_target)
print(f"After thinning: pre-recon {len(k_pre_c)} bins, post-recon {len(k_rec_c)} bins")

# Match covariance bins to data k bins
def match_cov(k_data, k_cov, cov):
    """Select sub-matrix of cov matching k_data bins."""
    idx = np.array([np.argmin(np.abs(k_cov - ki)) for ki in k_data])
    return cov[np.ix_(idx, idx)]

cov_pre = match_cov(k_pre_c, k_cov, cov_full)
cov_rec = match_cov(k_rec_c, k_cov, cov_full)

# Hartlap correction factor: (N_mocks - N_bins - 2) / (N_mocks - 1)
N_bins_pre = len(k_pre_c)
N_bins_rec = len(k_rec_c)
hartlap_pre = (N_mocks - N_bins_pre - 2) / (N_mocks - 1)
hartlap_rec = (N_mocks - N_bins_rec - 2) / (N_mocks - 1)
print(f"Hartlap factors: pre={hartlap_pre:.3f} ({N_bins_pre} bins), "
      f"rec={hartlap_rec:.3f} ({N_bins_rec} bins)")

# Diagonal errors for plotting
Pk_err_pre = np.sqrt(np.diag(cov_pre))
Pk_err_rec = np.sqrt(np.diag(cov_rec))

# ---- Increase chain length for better convergence ----
mcmc_cfg_run = dict(mcmc_cfg)
mcmc_cfg_run['n_steps'] = 20000
mcmc_cfg_run['n_burn']  = 5000

# ---- Pre-recon fit ----
print("\n" + "="*60)
print("PRE-RECON: Broadband-marginalized BAO fit")
print("="*60)
chain_pre, rs_pre = fit_bao(
    k_pre_c, Pk_pre_c, Pk_err_pre,
    cosmo['h'], cosmo['Omega_m'], cosmo['Omega_b'],
    cosmo['n_s'], cosmo['sigma8'],
    z=0.0, mcmc_config=mcmc_cfg_run,
    label='pre_recon_marg', output_dir=out_dir,
    cov=cov_pre, hartlap_factor=hartlap_pre,
    broadband_marginalize=True,
)

# ---- Post-recon fit ----
print("\n" + "="*60)
print("POST-RECON: Broadband-marginalized BAO fit")
print("="*60)
chain_rec, rs_rec = fit_bao(
    k_rec_c, Pk_rec_c, Pk_err_rec,
    cosmo['h'], cosmo['Omega_m'], cosmo['Omega_b'],
    cosmo['n_s'], cosmo['sigma8'],
    z=0.0, mcmc_config=mcmc_cfg_run,
    label='post_recon_marg', output_dir=out_dir,
    cov=cov_rec, hartlap_factor=hartlap_rec,
    broadband_marginalize=True,
)

# ---- Summary ----
from pk_input import sound_horizon
rs_fid = sound_horizon(cosmo['h'], cosmo['Omega_m'], cosmo['Omega_b'])

print("\n" + "="*60)
print("SUMMARY: Broadband-marginalized BAO fits")
print("="*60)
for name, chain, rs in [("Pre-recon", chain_pre, rs_pre),
                         ("Post-recon", chain_rec, rs_rec)]:
    a_med = np.median(chain[:, 0])
    a_lo, a_hi = np.percentile(chain[:, 0], [16, 84])
    S_med = np.median(chain[:, 1])
    S_lo, S_hi = np.percentile(chain[:, 1], [16, 84])
    print(f"  {name}:")
    print(f"    alpha = {a_med:.4f} +{a_hi-a_med:.4f} -{a_med-a_lo:.4f}")
    print(f"    Sigma = {S_med:.2f} +{S_hi-S_med:.2f} -{S_med-S_lo:.2f} Mpc/h")
    print(f"    r_s   = {rs:.2f} Mpc/h  (fid: {rs_fid:.2f})")

# Save results
np.savez(os.path.join(out_dir, 'marg_mcmc_results.npz'),
         chain_pre=chain_pre, chain_rec=chain_rec,
         rs_pre=rs_pre, rs_rec=rs_rec, rs_fid=rs_fid)
print(f"\nSaved results to {out_dir}/marg_mcmc_results.npz")
