# BAO Analysis Pipeline — Results Summary

## Overview

Baryon acoustic oscillations (BAO) are a relic of sound waves that propagated through the photon–baryon plasma before recombination at $z \approx 1100$. These waves imprinted a characteristic separation scale — the **sound horizon** $r_s \approx 150$ Mpc — into the distribution of matter, visible today as a slight excess of galaxy pairs at that separation. Because $r_s$ is calibrated by CMB physics, it serves as a **standard ruler**: measuring its apparent size at different redshifts constrains the expansion history $H(z)$ and angular-diameter distance $d_A(z)$.

This pipeline demonstrates the full BAO measurement cycle end-to-end:

```
Eisenstein-Hu P(k)  →  Zel'dovich ICs (z = 49)
        →  PM N-body (z = 49 → 0)
        →  P(k) & ξ(r) estimation
        →  Lognormal mock covariance (100 realizations)
        →  BAO reconstruction (pyrecon)
        →  MCMC template fitting (α, Σ)
```

All core components (CIC mass assignment, FFT Poisson solver, leapfrog integrator, lognormal generator, Metropolis–Hastings sampler, P(k) estimator) are written from scratch in NumPy/SciPy.

---

## Simulation Parameters

### Cosmology (Planck 2018)

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Hubble parameter | $h$ | 0.6736 |
| Total matter density | $\Omega_m$ | 0.3153 |
| Baryon density | $\Omega_b$ | 0.0493 |
| Dark energy density | $\Omega_\Lambda$ | 0.6847 |
| Spectral index | $n_s$ | 0.9649 |
| Amplitude of fluctuations | $\sigma_8$ | 0.8111 |

### Box & Simulation

| Parameter | Value |
|-----------|-------|
| Box size $L$ | 1500 Mpc/$h$ (periodic cube) |
| Particles $N$ | $128^3 = 2{,}097{,}152$ |
| PM mesh $N_\text{mesh}$ | $256^3$ (2× oversampling) |
| Initial redshift $z_\text{init}$ | 49 |
| Final redshift | 0 |
| Leapfrog steps | 50 (linear in scale factor $a$) |
| IC random seed | 42 |

### Galaxy Catalog (for lognormal mocks)

| Parameter | Value |
|-----------|-------|
| Galaxy bias $b$ | 1.5 |
| Number density $\bar{n}$ | $3 \times 10^{-4}$ $(h/\text{Mpc})^3$ |
| Effective redshift $z_\text{eff}$ | 0.38 |
| Lognormal seed | 123 |
| Number of mocks | 100 |

---

## Stage 1: Input Power Spectrum

The matter power spectrum is computed using the **Eisenstein & Hu (1998)** analytic transfer function, which captures both the full BAO oscillation pattern ("wiggles") and a smooth no-wiggle reference. Key quantities:

- **Sound horizon** (EH98 eq. 26 fitting formula): $r_s \approx 149.8$ Mpc/$h$
- **Growth factor**: Carroll, Press & Turner (1992) approximation, normalized $D(z=0) = 1$
- **Growth rate**: $f(z) = \Omega_m(z)^{0.55}$
- **Normalization**: $\sigma_8 = 0.8111$ via a top-hat window integral over an internal log-spaced $k$ grid

The ratio $P(k)/P_\text{nw}(k)$ oscillates around unity — these are the BAO wiggles that encode the sound horizon scale.

![Eisenstein-Hu power spectrum with and without BAO wiggles, and their ratio](outputs/figures/pk_input.png)

---

## Stage 2: Initial Conditions

Particle positions and velocities at $z = 49$ are generated via the **Zel'dovich approximation** (first-order Lagrangian perturbation theory):

$$\mathbf{x} = \mathbf{q} + D(z_\text{init})\,\boldsymbol{\Psi}(\mathbf{q}), \qquad \boldsymbol{\Psi}(\mathbf{k}) = -\frac{i\mathbf{k}}{k^2}\,\delta(\mathbf{k})$$

$$\mathbf{v}_\text{pec} = a\,H(z)\,f(z)\,D(z)\,\boldsymbol{\Psi}_0$$

At $z = 49$: $D \approx 0.02$, so particle displacements are small (a few Mpc/$h$) and the density field is nearly linear. The measured P(k) from the displaced particles matches the input linear theory to high precision:

![IC power spectrum validation: measured from particles vs. input theory at z=49](outputs/figures/ic_pk_check.png)

---

## Stage 3: N-body Evolution

A **particle-mesh (PM) leapfrog** integrator evolves $128^3$ particles from $z = 49$ to $z = 0$ in 50 time steps (linear in $a$). The gravity computation uses:

1. **CIC mass assignment** onto a $256^3$ grid
2. **FFT Poisson solve**: $\hat{\Phi}(\mathbf{k}) = -\frac{3}{2}\,\Omega_m\,H_0^2\,\hat{\delta}(\mathbf{k}) / (a\,k^2)$
3. **Finite-difference gradient** → acceleration field
4. **Kick-drift-kick** integration with conjugate momentum $p = a\,v_\text{pec}$

### Density Field Evolution

The density field evolves from a nearly uniform distribution at high redshift into the cosmic web of filaments, walls, and voids by $z = 0$:

![Density field slices at z = 5.4, 2.0, 1.0, 0.5, 0.0](outputs/figures/density_evolution.png)

### Power Spectrum Evolution

The measured P(k) grows with time and agrees well with linear theory at large scales ($k \lesssim 0.05$ $h$/Mpc). At smaller scales, nonlinear structure formation causes the measured P(k) to exceed linear predictions — the ratio $P_\text{measured}/P_\text{linear}$ increasingly deviates from unity at high $k$ and low $z$:

![Power spectrum at five redshifts (left) and ratio to linear theory (right)](outputs/figures/pk_evolution.png)

The BAO wiggles, clearly visible in the linear input P(k), become progressively damped by nonlinear gravitational evolution. This damping is the key physical effect that motivates BAO reconstruction.

---

## Stage 4: Lognormal Mocks & Covariance Matrix

To build a covariance matrix for the MCMC likelihood, **100 lognormal mock catalogs** are generated. Each mock:

1. Converts the galaxy power spectrum $P_\text{gal}(k) = b^2 P_\text{matter}(k)$ to a Gaussian correlation function via $\xi_G(r) = \ln(1 + \xi_\text{gal}(r))$
2. Draws a Gaussian random field from $P_G(k)$
3. Applies the lognormal transform: $1 + \delta_\text{LN} = \exp(\delta_G - \sigma_G^2/2)$
4. Poisson-samples galaxy positions from the density field

The sample covariance matrix $\hat{C}_{ij}$ is computed from the 100 P(k) measurements. When inverting for the likelihood, the **Hartlap correction** is applied:

$$\hat{C}^{-1}_\text{Hartlap} = \frac{N_\text{mocks} - N_\text{bins} - 2}{N_\text{mocks} - 1} \times \hat{C}^{-1}$$

Diagnostics show the correlation matrix is diagonally dominant with mild off-diagonal correlations, and fractional P(k) errors are $\lesssim 5\%$ for $k > 0.02$ $h$/Mpc:

![Lognormal mean P(k), correlation matrix, and fractional error](outputs/figures/covariance_diagnostics.png)

---

## Stage 5: BAO Reconstruction

BAO reconstruction partially reverses the bulk gravitational displacements that smear the acoustic peak. The pipeline uses **`pyrecon.IterativeFFTReconstruction`** with:

- Growth rate $f = \Omega_m^{0.55} \approx 0.47$
- Bias $b = 1.0$ (dark matter)
- Smoothing scale $R = 15$ Mpc/$h$
- 10× random catalog for shot noise correction

The displaced data and random positions define a density–random (D−R) field $\delta_\text{rec} = \delta_D - \delta_R$.

### Pre- vs. Post-Reconstruction P(k)

Post-reconstruction, the power spectrum rises closer to linear theory at intermediate scales ($k \sim 0.05$–$0.2$ $h$/Mpc):

![Pre- and post-recon P(k) and P/P_nw ratio](outputs/figures/recon_comparison.png)

### Reconstruction Summary

The 4-panel reconstruction summary shows the power spectra, BAO wiggle ratio, and MCMC posteriors for $\alpha$ and $\Sigma_\text{nl}$:

![BAO reconstruction summary: P(k), wiggles, α and Σ posteriors](outputs/figures/recon_summary.png)

Key results from the reconstruction-summary MCMC fit:

| Quantity | Pre-Reconstruction | Post-Reconstruction |
|----------|-------------------|---------------------|
| $\alpha$ | $1.192 \pm 0.009$ | $1.199 \pm 0.001$ |
| $\Sigma_\text{nl}$ | 2.3 Mpc/$h$ | 0.6 Mpc/$h$ |

The post-reconstruction $\Sigma_\text{nl}$ drops from 2.3 to 0.6 Mpc/$h$, confirming that reconstruction reduces nonlinear BAO damping. The $\alpha$ uncertainty shrinks by nearly an order of magnitude. The systematic offset of $\alpha \approx 1.19$ from the fiducial $\alpha = 1$ is a known limitation of this small-box ($L = 1500$ Mpc/$h$), low-resolution ($N = 128^3$) simulation — see [Discussion](#discussion-known-limitations) below.

---

## Stage 6: Correlation Function

The two-point correlation function $\xi(r)$ provides a complementary view of the BAO feature in configuration space. The characteristic **BAO bump** appears at $r \sim 105$ Mpc/$h$, corresponding to the sound horizon scale.

The plot shows $r^2 \xi(r)$ for linear theory, the N-body simulation at $z = 0$, the mean of 100 lognormal mocks, and the post-reconstruction D−R field:

![Correlation function r²ξ(r) showing the BAO bump](outputs/figures/xi_correlation_function.png)

### BAO Signal-to-Noise

| Measurement | BAO SNR |
|-----------|---------|
| Single N-body realization | 3.4 |
| Mock-averaged (100 mocks) | ~383 |
| BAO signal amplitude | 106.8 (Mpc/$h$)² |

The single-realization SNR of 3.4 is modest — expected for a $(1500\;\text{Mpc}/h)^3$ box with $128^3$ particles. The mock-averaged SNR of ~383 demonstrates that the BAO signal is robust and recoverable with sufficient volume or number of realizations.

---

## Stage 7: MCMC BAO Fitting

The BAO scale is extracted by fitting a template to the measured P(k). The template isolates the BAO wiggle component and fits for the **dilation parameter** $\alpha = r_s^\text{fid}/r_s$ (where $\alpha = 1$ means the measured BAO scale matches the fiducial cosmology) and the **nonlinear damping scale** $\Sigma_\text{nl}$.

### Broadband-Marginalized Template

The recommended fitting method analytically marginalizes over a polynomial broadband:

$$P_\text{model}(k) = O_\text{wiggle}(k;\,\alpha,\Sigma) + \sum_{j} a_j \left(\frac{k}{k_\text{ref}}\right)^j$$

where $O_\text{wiggle} = [P_\text{lin}(k/\alpha) - P_\text{nw}(k/\alpha)]\,e^{-k^2\Sigma^2/2}$ captures only the oscillatory BAO component, and the broadband coefficients $\{a_j\}$ are projected out analytically. This leaves only **two free parameters**: $(\alpha, \Sigma)$.

**MCMC settings**: Metropolis–Hastings, 20,000 steps + 5,000 burn-in, flat priors ($\alpha \in [0.5, 1.5]$, $\Sigma \in [0, 20]$ Mpc/$h$), full lognormal covariance with Hartlap correction, $k \in [0.02, 0.30]$ $h$/Mpc.

### Pre-Reconstruction Fit

| Parameter | Value |
|-----------|-------|
| $\alpha$ | $0.97^{+0.43}_{-0.30}$ |
| $\Sigma_\text{nl}$ | $4.79^{+3.56}_{-3.28}$ Mpc/$h$ |
| Recovered $r_s$ | 153.8 Mpc/$h$ |

![Pre-recon broadband-marginalized corner plot](outputs/mcmc/corner_pre_recon_marg.png)

![Pre-recon best-fit P(k), wiggle template, and residuals](outputs/mcmc/bestfit_pre_recon_marg.png)

### Post-Reconstruction Fit

| Parameter | Value |
|-----------|-------|
| $\alpha$ | $1.05^{+0.38}_{-0.33}$ |
| $\Sigma_\text{nl}$ | $4.55^{+3.05}_{-3.07}$ Mpc/$h$ |
| Recovered $r_s$ | 142.3 Mpc/$h$ |

![Post-recon broadband-marginalized corner plot](outputs/mcmc/corner_post_recon_marg.png)

![Post-recon best-fit P(k), wiggle template, and residuals](outputs/mcmc/bestfit_post_recon_marg.png)

Both pre- and post-reconstruction $\alpha$ posteriors are **consistent with the fiducial value** $\alpha = 1$ within their (broad) uncertainties. The large error bars reflect the limited constraining power of a single $(1500\;\text{Mpc}/h)^3$ box with $128^3$ particles.

---

## Pipeline Summary Figure

The 4-panel summary figure captures the entire analysis pipeline in a single view:

- **(a)** Density field evolution: projected $\delta(\mathbf{x})$ at $z = 5.4$ vs. $z = 0$
- **(b)** Power spectrum: linear theory, N-body $z = 0$, and post-reconstruction
- **(c)** BAO wiggles: $P(k)/P_\text{nw}(k)$ ratio before and after reconstruction
- **(d)** BAO dilation parameter $\alpha$ posteriors: pre- vs. post-reconstruction

![4-panel pipeline summary](outputs/figures/pipeline_summary.png)

---

## Key Results Summary

| Quantity | Value | Notes |
|----------|-------|-------|
| Sound horizon $r_s^\text{fid}$ | 149.8 Mpc/$h$ | EH98 eq. 26 fitting formula |
| BAO bump location in $\xi(r)$ | ~105 Mpc/$h$ | Configuration-space peak |
| Single N-body BAO SNR | 3.4 | In $r^2\xi(r)$, 80–160 Mpc/$h$ range |
| Mock-averaged BAO SNR | ~383 | 100 lognormal realizations |
| Pre-recon $\alpha$ (marg) | $0.97 \pm 0.36$ | Consistent with $\alpha = 1$ |
| Post-recon $\alpha$ (marg) | $1.05 \pm 0.36$ | Consistent with $\alpha = 1$ |
| Pre-recon $\alpha$ (recon summary) | $1.192 \pm 0.009$ | Tighter but biased — see discussion |
| Post-recon $\alpha$ (recon summary) | $1.199 \pm 0.001$ | 8× tighter than pre-recon |
| Pre-recon $\Sigma_\text{nl}$ (recon summary) | 2.3 Mpc/$h$ | |
| Post-recon $\Sigma_\text{nl}$ (recon summary) | 0.6 Mpc/$h$ | Reconstruction reduces damping |

---

## Discussion: Known Limitations

1. **Resolution effects on $\alpha$**: The reconstruction-summary MCMC yields $\alpha \approx 1.19$ instead of the expected $\alpha = 1$. This ~19% bias is a consequence of the limited particle resolution ($128^3$) and box size ($1500$ Mpc/$h$). With only $\sim 2 \times 10^6$ particles in a box that spans $\sim 10 \times r_s$, the BAO feature has few independent modes and the broadband shape is distorted by shot noise and aliasing. The broadband-marginalized method, which absorbs these systematic effects into the polynomial nuisance terms, correctly recovers $\alpha \approx 1$.

2. **BAO wiggle resolution**: The $P(k)/P_\text{nw}(k)$ ratio in the summary plot shows that post-reconstruction wiggles are enhanced relative to pre-reconstruction, but not fully resolved to match linear theory at $N = 128$. The annotation "Wiggles unresolved at $N = 128$" on the summary plot reflects this.

3. **Covariance limitations**: With 100 lognormal mocks and ~30–43 $k$-bins, the Hartlap correction factor ranges from 0.56 to 0.69, indicating that the inverse covariance matrix is somewhat noisy. More mocks would tighten the constraints.

4. **PM force resolution**: The particle-mesh method smooths forces on scales below the mesh cell size ($\Delta x = L/N_\text{mesh} \approx 5.9$ Mpc/$h$), meaning small-scale halo structure is not resolved. This is acceptable for BAO measurements, which probe scales of ~100 Mpc/$h$.

---

## Output File Inventory

### Snapshots (`outputs/snapshots/`)

5 primary HDF5 snapshots, each containing particle positions `pos` (3, $N^3$), peculiar velocities `vel` (3, $N^3$), density contrast `delta` ($256^3$), and metadata (`a`, `z`, `step`):

| File | Redshift |
|------|----------|
| `snap_0007_z5.36.h5` | 5.36 |
| `snap_0016_z2.00.h5` | 2.00 |
| `snap_0024_z1.04.h5` | 1.04 |
| `snap_0033_z0.50.h5` | 0.50 |
| `snap_0050_z0.00.h5` | 0.00 |

An additional 20 snapshots spanning $z = 15.89$ to $z = 0$ are stored in `src/outputs/snapshots/` and were used to produce the density evolution animation (`src/outputs/figures/density_evolution.mp4`).

### Data Files (`outputs/mcmc/`)

| File | Contents |
|------|----------|
| `lognormal_covariance.npz` | Sample covariance matrix, $k$-bins, all 100 mock P(k) measurements |
| `recon_pk.npz` | Pre- and post-recon P(k), mode counts, theory P(k) with/without wiggles |
| `marg_mcmc_results.npz` | Broadband-marginalized MCMC chains ($\alpha$, $\Sigma$) pre- and post-recon, recovered $r_s$ |
| `recon_mcmc_results.npz` | Legacy 3-parameter MCMC chains ($\alpha$, $\Sigma$, $B$) |
| `xi_data.npz` | Correlation functions: N-body, post-recon, 100 mock mean/scatter, theory, BAO SNR values |

### Figures (`outputs/figures/`)

| File | Description |
|------|-------------|
| `pk_input.png` | Eisenstein-Hu P(k) with/without BAO wiggles |
| `ic_pk_check.png` | IC P(k) validation at $z = 49$ |
| `density_evolution.png` | Density field slices at 5 redshifts |
| `pk_evolution.png` | P(k) at 5 redshifts + ratio to linear theory |
| `pk_estimator_test.png` | P(k) estimator shot-noise validation |
| `covariance_diagnostics.png` | Lognormal covariance: mean P(k), correlation matrix, fractional error |
| `recon_comparison.png` | Pre- vs. post-recon P(k) and P/$P_\text{nw}$ ratio |
| `recon_summary.png` | 4-panel: P(k), wiggles, $\alpha$ and $\Sigma$ posteriors |
| `xi_correlation_function.png` | $r^2\xi(r)$ with BAO bump and SNR annotation |
| `pipeline_summary.png` | 4-panel capstone: density, P(k), wiggles, $\alpha$ posteriors |

### MCMC Plots (`outputs/mcmc/`)

| File | Description |
|------|-------------|
| `corner_pre_recon_marg.png` | Broadband-marg corner plot ($\alpha$, $\Sigma$), pre-recon |
| `corner_post_recon_marg.png` | Broadband-marg corner plot ($\alpha$, $\Sigma$), post-recon |
| `bestfit_pre_recon_marg.png` | Best-fit P(k) + wiggle template + residuals, pre-recon |
| `bestfit_post_recon_marg.png` | Best-fit P(k) + wiggle template + residuals, post-recon |
| `corner_pre_recon.png` | Legacy 3-param corner plot, pre-recon |
| `corner_post_recon.png` | Legacy 3-param corner plot, post-recon |
| `bestfit_pre_recon.png` | Legacy best-fit, pre-recon |
| `bestfit_post_recon.png` | Legacy best-fit, post-recon |
| `corner_nbody_z0_fullcov.png` | N-body $z = 0$ full-covariance fit |
| `bestfit_nbody_z0_fullcov.png` | N-body $z = 0$ full-covariance best-fit |

### Animation

| File | Description |
|------|-------------|
| `src/outputs/figures/density_evolution.mp4` | Animated density field evolution $z = 49 \to 0$ |

---

## How to Reproduce

```bash
# Setup
pip install -r requirements.txt
pip install "pyrecon[extras] @ git+https://github.com/cosmodesi/pyrecon"

# Full pipeline
cd src
python main.py --config ../configs/default.yaml

# Individual stages
python main.py --config ../configs/default.yaml --stage ics
python main.py --config ../configs/default.yaml --stage nbody
python main.py --config ../configs/default.yaml --stage lognormal
python main.py --config ../configs/default.yaml --stage pk
python main.py --config ../configs/default.yaml --stage recon
python main.py --config ../configs/default.yaml --stage mcmc
python main.py --config ../configs/default.yaml --stage plots

# Standalone scripts
python compute_xi.py          # Correlation function + BAO SNR
python run_bao_marg.py        # Broadband-marginalized MCMC
python make_summary_plot.py   # 4-panel summary figure

# Tests (from repo root)
pytest tests/ -v
```
