# BAO Pipeline — Design Document

## Project Summary

An end-to-end Baryon Acoustic Oscillation (BAO) analysis pipeline for PHY 305. The pipeline simulates how gravity degrades the BAO signal imprinted in the early universe, applies reconstruction to partially restore it, and uses MCMC to quantify recovery at each stage.

**Scientific narrative:** Sharp BAO signal in initial conditions → gravity smears it → reconstruction partially restores it → MCMC quantifies recovery.

**Repo:** `~/bao_nbody/`  
**Environment:** `bao305` (mamba, Python 3.11)  
**Config:** `configs/default.yaml` (all runtime parameters)

---

## Architecture

```
configs/default.yaml          ← all runtime params
src/main.py                   ← pipeline orchestrator (--stage flag)
src/pk_input.py               ← Eisenstein-Hu P(k), sound horizon, growth factor
src/initial_conditions.py     ← Zel'dovich approximation ICs
src/pm_gravity.py             ← CIC mass assignment + FFT Poisson solver
src/nbody.py                  ← Leapfrog integrator + snapshot I/O
src/power_spectrum.py         ← FFT-based P(k) estimator with CIC correction
src/lognormal.py              ← 3D lognormal catalog generator
src/mcmc.py                   ← Metropolis-Hastings BAO template fitting
src/utils.py                  ← cosmology helpers, comoving distance, plotting
tests/                        ← unit tests (pytest)
```

The pipeline runs sequentially: ICs → N-body → P(k) estimation → lognormal mocks (for covariance) → reconstruction (pyrecon) → MCMC fitting. Each stage reads from `configs/default.yaml` and writes to `outputs/`.

---

## Module Status & Implementation Guide

### 1. `pk_input.py` — ✅ DONE

Eisenstein-Hu (1998) transfer function producing P(k) with BAO wiggles and a smooth no-wiggle reference. Also provides `growth_factor(z)` and `sound_horizon()`.

**Verified outputs:**
- `r_s ≈ 149.82 Mpc/h` (EH98 eq. 26 fitting formula)
- P(k) plot with visible BAO wiggles saved to `outputs/figures/pk_input.png`

**Known fixes already applied:**
- `np.trapz` → `np.trapezoid` (NumPy 2.0 rename)
- Sound horizon formula replaced with EH98 eq. 26
- Growth factor: must compute Ω_m(z) and Ω_Λ(z) using E²(z) as denominator, multiply D by scale factor `a`

**Key functions:**
- `power_spectrum(k, h, Om, Ob, ns, s8, wiggle=True)` → P(k) array
- `power_spectrum_nowiggle(k, ...)` → smooth P_nw(k)
- `growth_factor(z, Omega_m)` → D(z)/D(0), should give ~0.02 at z=49
- `growth_rate(z, Omega_m)` → f(z) ≈ Ω_m(z)^0.55
- `sound_horizon(h, Omega_m, Omega_b)` → r_s in Mpc/h

### 2. `initial_conditions.py` — ✅ DONE

Zel'dovich approximation: displace particles from a uniform grid using the linear displacement field derived from P(k).

**What exists:** Full implementation. Generates 3D Gaussian random field in Fourier space, computes displacement via ψ_i(k) = -i k_i / k² × δ(k), scales by D(z_init).

**What to verify after growth factor fix:**
- `D(z=49)` should print ~0.02 (not 1.27)
- Position displacements should be ~few Mpc/h (not hundreds)
- Round-trip test: measure P(k) from displaced particles, compare to input — BAO wiggles should be visible

**Key physics:** The Zel'dovich approximation is first-order Lagrangian perturbation theory. Particles are displaced from grid positions q to Eulerian positions x = q + D(z) × ψ(q), where ψ is the displacement field.

### 3. `pm_gravity.py` — ✅ DONE

Particle-Mesh gravity solver: CIC mass assignment → FFT → multiply by Green's function → IFFT → CIC interpolation of forces back to particles.

**What exists:** Full scaffolding with CIC paint, Poisson solve via FFT Green's function, and force interpolation.

**Implementation details to understand:**
- CIC (Cloud-in-Cell) assigns each particle's mass to the 8 nearest grid points with trilinear weights
- Poisson's equation ∇²Φ = 4πGρ becomes -k²Φ̃(k) = 4πGρ̃(k) in Fourier space
- The Green's function G(k) = -1/k² (with k=0 mode set to zero for periodic boundaries)
- Force = -∇Φ, computed via finite differences or spectral derivative

**Validation approach:** Place two point masses, check force falls off as 1/r² at large separation and flattens at grid scale.

### 4. `nbody.py` — ✅ DONE

Leapfrog (kick-drift-kick) integrator evolving particles from z_init=49 to z=0.

**What exists:** Full scaffolding with KDK leapfrog, cosmological time-stepping (scale factor as time variable), and HDF5 snapshot output.

**Implementation details:**
- Leapfrog is symplectic: kick velocities by dt/2, drift positions by dt, kick velocities by dt/2
- In cosmological context, the time variable is the scale factor `a`, and the equation of motion includes Hubble drag
- Energy conservation check: total energy (kinetic + potential) should be roughly conserved

**Validation approach:** Run short simulation, check energy conservation to ~1% over the full run. Visual check: 2D density slice should show filamentary structure forming.

### 5. `power_spectrum.py` — ✅ DONE

FFT-based P(k) estimator with CIC deconvolution and shot noise subtraction.

**Verified:** Uniform random catalog gives P(k) ≈ 0 after shot noise subtraction.

**Key function:** `estimate_pk(positions, N_particles, L_box, n_mesh)` → `(k_bins, Pk, n_modes)`

**Details:** Paints particles to mesh via CIC, FFTs, computes |δ(k)|², averages in spherical k-shells, divides by CIC window function W²(k), subtracts shot noise 1/n̄.

### 6. `lognormal.py` — ✅ DONE

Generates lognormal random fields matching a target P(k), then Poisson-samples galaxy positions.

**What exists:** Full implementation. Transforms Gaussian field to lognormal via δ_LN = exp(G - σ²/2) - 1, then Poisson-samples N_gal from the density field.

**Known fix applied:** `np.trapz` → `np.trapezoid`

**Purpose:** NOT to rediscover P(k), but to generate many independent realizations for building a covariance matrix C_ij for the P(k) estimator. This covariance feeds into the MCMC likelihood.

**Validation:** Generate ~100 mocks, compute mean P(k) — should match input. Variance across mocks gives the diagonal of the covariance matrix.

### 7. Reconstruction (pyrecon) — ✅ DONE

BAO reconstruction uses the observed galaxy overdensity to estimate the displacement field and shift galaxies back toward initial positions, sharpening the BAO peak.

**External library:** `pip install "pyrecon[extras] @ git+https://github.com/cosmodesi/pyrecon"` (the `#egg=` syntax does NOT work)

**Integration plan:**
1. Take galaxy catalog (positions) from lognormal mock or N-body snapshot
2. Call pyrecon's `IterativeFFTReconstruction` or `MultiGridReconstruction`
3. Get shifted positions back
4. Re-estimate P(k) from shifted positions
5. Compare pre- and post-reconstruction P(k)

**Expected result:** BAO peak in P(k)/P_nw(k) ratio should be sharper after reconstruction.

### 8. `mcmc.py` — ✅ DONE

Metropolis-Hastings sampler fitting a damped BAO template to measured P(k).

**What exists:** Full scaffolding. Fits parameters α (dilation, should be ~1.0) and Σ_nl (nonlinear damping scale).

**BAO template model:**
```
P_model(k) = B × [P_nw(k/α) + (P_w(k/α) - P_nw(k/α)) × exp(-k²Σ²/2)]
```
where B is a broadband amplitude, α rescales the BAO ruler, and Σ controls damping.

**Likelihood:** χ² = (P_data - P_model)ᵀ C⁻¹ (P_data - P_model), where C is the covariance from lognormal mocks.

**Validation:** Fit to input P(k) (no noise) should recover α = 1.0 ± small error. Corner plot should show well-behaved posteriors.

### 9. `main.py` — ✅ DONE

Pipeline orchestrator. Reads YAML config, runs stages sequentially or individually via `--stage` flag.

**Stages:** `ics`, `nbody`, `pk`, `lognormal`, `recon`, `mcmc`, `all`

---

## Implementation Order

This is the recommended order for working through the remaining implementation:

1. **Verify growth factor fix** — Run `python src/pk_input.py` and `python src/initial_conditions.py`. Confirm D(z=49) ≈ 0.02 and displacements are ~few Mpc/h.

2. **IC round-trip test** — After ICs generate, measure P(k) from the displaced particles. The measured P(k) at z=49 should match the input P(k) scaled by D²(z=49). BAO wiggles must be visible.

3. **PM gravity + leapfrog** — Run a short N-body from the ICs. Check energy conservation. Visualize a 2D density slice — you should see filaments and voids forming.

4. **P(k) from N-body** — Measure P(k) at multiple snapshots (z=49, 2, 1, 0). The BAO peak should progressively broaden and suppress at lower redshift.

5. **Lognormal mocks** — Generate ~50-100 realizations. Compute mean P(k) and covariance matrix. Validate that mean matches input.

6. **Reconstruction** — Apply pyrecon to a mock catalog. Compare P(k) before and after. The BAO peak should sharpen.

7. **MCMC fitting** — Fit the BAO template at each stage: ICs, post-N-body, post-reconstruction. Produce corner plots. The posterior on α should narrow after reconstruction.

8. **Animation** — Render 2D density slices at each snapshot into an MP4.

9. **Polish** — Final plots for the presentation, write-up.

---

## Key Physics Concepts

**BAO as a standard ruler:** Sound waves in the early universe traveled ~150 Mpc before recombination (z ≈ 1100), leaving a preferred separation between galaxy pairs. Since this physical scale is known from CMB physics, measuring its apparent size at different redshifts constrains the expansion history.

**Why reconstruction works:** Gravity moves galaxies away from their initial positions, smearing the sharp BAO feature. Reconstruction estimates the large-scale displacement field from the observed galaxy density and reverses it, partially restoring the original sharpness.

**Leapfrog integrator:** A symplectic integrator that conserves phase-space volume. The kick-drift-kick scheme staggers velocity and position updates, making it time-reversible and preventing artificial energy drift — critical for long N-body simulations.

**Lognormal mocks for covariance:** A single P(k) measurement has noise from the finite number of Fourier modes in each k-bin ("cosmic variance"). Running many independent lognormal realizations and measuring P(k) from each gives a sample covariance matrix, which tells the MCMC how much to trust each k-bin.

---

## Config Reference (`configs/default.yaml`)

| Section | Key params | Notes |
|---------|-----------|-------|
| `cosmology` | h=0.6736, Ω_m=0.3153, Ω_b=0.0493 | Planck 2018 |
| `box` | L=1500 Mpc/h, N=128, N_mesh=256 | N³ particles, N_mesh for PM |
| `simulation` | n_steps=50, z_init=49 | Leapfrog steps from z=49→0 |
| `lognormal` | seed=123 | Separate RNG from N-body |
| `mcmc` | n_steps=10000, n_burn=2000 | α ∈ [0.5, 1.5] prior, broadband-marginalized |

---

## Deliverables Checklist

- [x] Python codebase with all modules passing tests (40/40)
- [x] MP4 animation of density field evolution
- [x] P(k) comparison plot: initial → evolved → reconstructed
- [x] MCMC corner plots at each stage (broadband-marginalized + legacy)
- [x] Pipeline summary figure (4-panel)
- [ ] 1-2 page written report
- [ ] 8-minute presentation (~8 slides)