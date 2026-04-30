# PHY 305 Creative Project Proposal

**Dennis Wu**

---

## Introduction

One of the most striking features in the large-scale distribution of galaxies is a preferred separation of roughly 150 megaparsecs (Mpc) between galaxy pairs—a relic of sound waves that traveled through the hot plasma of the early universe before the epoch of recombination, approximately 380,000 years after the Big Bang. These Baryon Acoustic Oscillations (BAO) leave a characteristic "ring" in the galaxy distribution that acts as a standard ruler: because the physical size of this feature is calculable from well-established physics, measuring how large it appears on the sky at different cosmic epochs directly constrains the expansion history of the universe and the properties of dark energy.

Measuring the BAO scale precisely is complicated by the fact that gravitational evolution smears the originally sharp acoustic feature over cosmic time. BAO reconstruction—a technique that uses the observed galaxy field to approximately reverse this gravitational smoothing—partially restores the signal before the final statistical fit. This project builds an end-to-end computational pipeline in Python that simulates all of these stages from first principles: generating a density field with the BAO feature imprinted, evolving it forward under gravity, generating a mock galaxy catalog, applying reconstruction, and finally fitting a model to recover the acoustic scale.

---

## Goals

The primary goals of this project are:

- Implement a physically accurate 3D simulation of structure formation that captures the degradation and recovery of the BAO feature across cosmic time.
- Build a complete, modular data analysis pipeline in Python that mirrors the structure of a real cosmological survey analysis.
- Quantitatively demonstrate—through power spectrum plots and MCMC corner plots—how each pipeline stage (N-body evolution, mock catalog generation, reconstruction, and fitting) affects the recovered acoustic scale.
- Produce an animated visualization of the evolving density field that intuitively conveys how gravity erases initial structure.
- Develop practical familiarity with the core computational physics techniques covered in PHY 305 by applying them to a unified, scientifically motivated problem.

---

## Methods

The pipeline proceeds through four sequential stages, each building on the output of the previous.

**Stage 1 — N-body Simulation.** Initial conditions are generated using the Zel'dovich approximation, which displaces particles from a uniform grid according to the linear density field drawn from an input matter power spectrum. A 3D particle-mesh (PM) gravity solver then evolves the particle positions forward in time. The density field is computed on a grid via cloud-in-cell mass assignment, the gravitational potential is solved using Poisson's equation in Fourier space (exploiting FFTs for efficiency), and the resulting accelerations are applied to particles. Time integration uses the leapfrog (Verlet) algorithm—a symplectic integrator that conserves phase-space volume and is time-reversible, making it well-suited to long Hamiltonian simulations.

**Stage 2 — Lognormal Mock Catalog.** A lognormal random field is generated from the same input power spectrum to serve as a fast, statistically controlled alternative to the full N-body output. Galaxy positions are sampled from this field via Poisson sampling, producing a mock catalog whose two-point statistics match the target power spectrum by construction. The primary purpose of the mock catalog is not to rediscover the power spectrum, but to characterize the noise properties of the FFT-based estimator—specifically, to build a covariance matrix for the measured P(k) by running many independent realizations, which is essential for any downstream likelihood analysis.

**Stage 3 — BAO Reconstruction.** Reconstruction is applied to the mock catalog using the pyrecon library (cosmodesi). The algorithm estimates the large-scale displacement field from the observed galaxy overdensity, then shifts galaxies back toward their initial positions to sharpen the acoustic peak. The power spectrum is re-estimated before and after reconstruction to quantify the improvement in BAO signal-to-noise.

**Stage 4 — MCMC Parameter Fitting.** A damped BAO template model is fit to the measured power spectrum using a Metropolis-Hastings MCMC sampler written from scratch in NumPy. The sampler explores the posterior distribution of the acoustic scale dilation parameter α and the sound horizon r_s. The likelihood is evaluated against the covariance matrix estimated from the lognormal mocks. Corner plots are produced at each pipeline stage to track how well the acoustic scale is recovered before and after reconstruction.

---

## Numerical Methods

This project directly exercises the following techniques from PHY 305:

- **Symplectic ODE Integration (Leapfrog).** The leapfrog integrator advances particle positions and velocities in a staggered, time-symmetric fashion. Unlike standard Euler or Runge-Kutta methods, it exactly conserves a modified Hamiltonian, preventing artificial energy drift over long simulations. Energy conservation is verified as a numerical accuracy check.

- **Fast Fourier Transforms (FFT).** FFTs are used in two places: (1) in the PM gravity solver, to efficiently solve Poisson's equation by multiplying the density field by the Green's function in Fourier space and transforming back; and (2) in the power spectrum estimator, to compute the galaxy overdensity field and measure P(k) as the squared amplitude of its Fourier modes, averaged over shells of constant wavenumber k.

- **Metropolis-Hastings MCMC.** A custom Markov chain sampler proposes steps from a Gaussian proposal distribution, accepts or rejects them based on the log-likelihood ratio, and converges to the posterior distribution of the model parameters. Convergence is assessed via chain autocorrelation and acceptance rate diagnostics.

- **Numerical Integration.** The comoving distance as a function of redshift is computed by numerically integrating 1/E(z), where E(z) = H(z)/H₀ is the dimensionless Hubble parameter. This integral is evaluated using Simpson's rule and is used to convert between cosmological units throughout the pipeline. The linear growth factor D(z) is similarly computed by numerical quadrature.

---

## Relationship to Existing BAO Research

I am currently involved in BAO research using SPHEREx satellite data, which uses production-grade tools, full-sky lognormal mocks with realistic survey geometry, and analysis frameworks developed by large collaborations. This PHY 305 project is distinct from that research in scope, tooling, and purpose.

In my research, I use existing pipelines and libraries (e.g., cosmodesi, pypower, velocileptors) to analyze mocks generated by dedicated survey simulation infrastructure. The science questions concern real dark energy parameters w₀ and wₐ and require rigorous treatment of survey window functions, fiber incompleteness, and photometric systematics. That work is beyond the scope of this project and involves collaborators and shared codebases—none of which are reproduced here.

This project, by contrast, builds every component from scratch or from basic scientific Python libraries (NumPy, SciPy, Matplotlib), with the explicit goal of understanding the physics of each step. The N-body simulation, PM gravity solver, lognormal generator, P(k) estimator, and MCMC sampler are all written and verified independently, in a simplified 3D box geometry without survey effects. The pyrecon library is used for reconstruction because implementing a full reconstruction algorithm from scratch would exceed the scope of a semester project, but all surrounding pipeline stages are original code. The project is designed to build intuition for BAO physics and computational methods, not to produce research-grade constraints.

---

## Expected Results

The pipeline is expected to reproduce several well-known results from BAO physics in a controlled, simplified setting. The initial power spectrum from Zel'dovich initial conditions should display a clear BAO peak near k ~ 0.06 h/Mpc. After N-body evolution, the BAO peak will be broadened and suppressed relative to the initial conditions, reflecting nonlinear gravitational smearing. After reconstruction, the peak sharpness should partially recover, with the acoustic scale recovered to better precision. The MCMC fits at each stage will quantify this progression: the posterior on α is expected to be broad before reconstruction and narrower afterward, demonstrating the utility of the reconstruction step.

The lognormal mock realizations will be used to build a covariance matrix for P(k); the resulting error bars on the power spectrum should be consistent across mocks, validating the estimator. The MP4 animation of the density field will visually show particles clustering along filaments and into halos as the simulation progresses, providing an intuitive illustration of structure formation.

---

## Deliverables

The final project will consist of the following:

- A Python codebase with modular components for each pipeline stage (N-body simulation, lognormal mock generation, P(k) estimation, reconstruction, and MCMC fitting), organized with a YAML configuration system and unit tests for key functions.
- An MP4 animation of the 3D density field evolving under gravity, rendered as projected 2D slices at each simulation timestep.
- Power spectrum comparison plots showing P(k) before N-body evolution, after evolution, and after BAO reconstruction, with error bars from lognormal mocks.
- MCMC corner plots at each stage showing the posterior distributions on the acoustic scale parameter α and sound horizon r_s, quantifying how reconstruction improves parameter recovery.
- A one-to-two page written report summarizing the physics, methods, results, and lessons learned.
- An 8-minute presentation covering the motivation, pipeline design, and key results.
