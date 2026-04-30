# Speaker Script — *Measuring the Universe with Sound Waves*

Target runtime: ~6 minutes for slides 1–10. Bracketed cues are stage
directions, not spoken.

---

## Slide 1 — Title

> The goal of this project was to build, from scratch, a small
> end-to-end cosmology pipeline — a simulation of a patch of the
> universe, the statistics to analyze it, and a parameter fit at
> the end. The physics target is the baryon acoustic oscillation
> ruler: a sound wave imprinted on matter in the early universe.

---

## Slide 2 — A sound wave left a ruler in the sky

> Before atoms formed, the universe was a hot photon-baryon plasma
> with sound speed close to c over root-three. Overdensities
> launched spherical pressure waves; at recombination, photons
> decoupled and those waves froze in place at a comoving radius of
> about 150 megaparsecs. We see them today as a bump in the galaxy
> correlation function near 100 h-inverse Mpc, equivalently as
> wiggles in P(k). Because the CMB pins the absolute size, this
> works as a standard ruler — its apparent size at any redshift
> gives us the expansion history and dark-energy constraints.

---

## Slide 3 — Why measuring it is hard

> Three obstacles. One: gravity blurs the shell by several Mpc as
> galaxies fall toward overdensities — fixed by reconstruction,
> which estimates bulk flows and moves galaxies back. Two: the
> wiggles are a few-percent ripple on a huge smooth spectrum, and
> finite resolution distorts that smooth part — fixed by
> marginalizing the broadband as a nuisance. Three: one universe
> gives one noisy measurement — fixed by generating 100 fast mock
> universes for a covariance matrix.

---

## Slide 4 — The pipeline

> Left to right: linear P(k) as input, Zel'dovich initial conditions
> at z=49, particle-mesh N-body down to z=0, then a P(k) estimator
> and reconstruction on the particle catalog. In parallel, 100
> lognormal mocks build the covariance, and Metropolis-Hastings
> MCMC fits the BAO template. The box is 1.5 Gpc/h, 128 cubed
> particles on a 256 cubed force grid. Everything is from scratch
> except reconstruction, which uses `pyrecon`.

---

## Slide 5 — How the N-body works

> Direct pair summation is N-squared and infeasible. So I use a
> particle-mesh scheme: cloud-in-cell deposit onto a grid, FFT,
> divide by k-squared to solve Poisson, FFT back, finite-difference
> gradient, interpolate forces back to particles. N-squared becomes
> N-log-N.
>
> For time stepping I use leapfrog — a symplectic integrator, so
> energy errors stay bounded instead of drifting. Kick-drift-kick,
> stepping in the scale factor rather than physical time so each
> step covers a comparable fraction of cosmic history.

---

## Slide 6 — Error bars and the fit

> For the covariance: lognormal mocks are a cheap non-Gaussian
> approximation — draw a Gaussian field with the right P(k),
> exponentiate, Poisson-sample. Run that 100 times and take the
> covariance of the resulting P(k) measurements.
>
> The fit has two physical parameters: alpha, the ruler stretch,
> which equals 1 if the simulated distance scale matches the
> expected length; and Sigma-nl, the BAO smearing scale. On top of
> those, three smooth-background coefficients — but because they
> enter linearly, I integrate them out of the Gaussian likelihood
> in closed form. MCMC then only explores two dimensions, which
> insulates alpha from broadband distortion.

---

## Slide 7 — Four views of the result

> Panel (a): density slabs at z=5.4 and z=0. The cosmic web grows
> entirely from gravitational amplification of tiny initial
> fluctuations.
>
> Panel (b): the power spectrum. At large scales the simulation
> tracks linear theory to within a percent; at smaller scales it
> exceeds linear theory — that's nonlinear clustering — until the
> mesh resolution cuts off near Nyquist.
>
> Panel (c): P(k) divided by a no-wiggle reference, so only the
> BAO signature remains. Post-reconstruction in red tracks linear
> theory more cleanly than pre-reconstruction in blue.
>
> Panel (d): the MCMC posterior on alpha. Red is visibly narrower
> than blue. That's the money plot.

---

## Slide 8 — The ruler

> Left, the correlation function — Fourier transform of P(k). The
> bump near 100 h-inverse Mpc is the frozen sound shell in real
> space. Signal-to-noise is about 3.4 in one box, 80 averaged over
> all 100 mocks.
>
> Right, the numbers. Pre-recon: alpha = 1.027 ± 0.015. Post-recon:
> 1.032 ± 0.010. Both consistent with 1 at the few-percent level —
> the input distance scale is recovered. The error bar tightens by
> a factor of 1.6 and Sigma-nl drops from 4.8 to 3.0 h-inverse Mpc,
> exactly what reconstruction is meant to deliver. For context,
> DESI and Euclid reach alpha at the 0.3-percent level on hundreds
> of cubic Gpc of volume.

---

## Slide 9 — Validation

> Five checks. A null test on uniform random particles returns
> zero P(k) after shot-noise subtraction. At z=49 the ICs
> reproduce linear theory times the growth factor squared. Large
> scales match linear theory; small scales overshoot in the
> expected nonlinear way. Skipping broadband marginalization gives
> alpha = 1.19 — a 19-percent bias — which collapses to 1 once
> marginalization is on, on the exact same data. The 100 mocks
> reproduce the input P(k) on average, and `pytest` covers every
> module.

---

## Slide 10 — Takeaways

> A from-scratch pipeline recovers the BAO ruler to roughly one
> percent from a 1.5 Gpc box. Reconstruction tightens the error
> bar by 1.6×; broadband marginalization removes a 19-percent
> bias. Natural extensions: a higher-resolution tree-PM solver,
> redshift-space effects with realistic survey masks, and fitting
> alpha parallel and perpendicular separately to disentangle
> Hubble from angular-diameter distance.
>
> Happy to take questions.

---

## Backup — Covariance diagnostics

> Left: fractional error per k-bin, larger at small k because of
> finite-volume mode counts. Middle: correlation matrix — small
> off-diagonals, so bins are nearly independent. Right: mean of
> the 100 mocks matches the theory input.

---

## Backup — Why marginalization matters

> The template is a wiggle piece carrying the physics plus a
> smooth polynomial background carrying nuisance effects like
> resolution and bias. The three smooth coefficients enter
> linearly, so a Woodbury-style identity integrates them out
> analytically. On the same data, joint fitting gives alpha ≈
> 1.19; analytical marginalization gives alpha = 1.

---

*End of script. ~6 minutes main, ~30 seconds per backup if prompted.*
