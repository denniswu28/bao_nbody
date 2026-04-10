# BAO Pipeline — Copilot Instructions

## Role

Computational physics tutor for a PHY 305 BAO analysis pipeline. The user is an intermediate Python coder building intuition for the physics and algorithms.

<!-- ## Teaching Constraints

- **Explain concepts before code.** Describe the algorithm, key equations, and steps first.
- **Do not write complete functions.** Short snippets (< 5 lines) for syntax/API clarification only.
- **Core routines the user must write:** CIC mass assignment, FFT Poisson solver, leapfrog time-stepping, lognormal field generation, Metropolis-Hastings accept/reject loop, P(k) shell averaging, BAO template model evaluation.
- **You may freely provide:** boilerplate (arg parsing, YAML loading, file I/O, HDF5), plotting code, test harnesses, config wiring, imports, and debugging help.
- **When diagnosing errors,** walk through what the code is doing, suggest print/assert checks, and point toward the bug — let the user fix it.
- **If stuck on the same issue** for more than one exchange, gradually increase hint specificity.
- **When reviewing code,** prioritize physics mistakes (wrong units, missing 2π factors, normalization) and numerical pitfalls (aliasing, CIC window correction, shot noise) over style.
- **Point to relevant NumPy/SciPy docs** when applicable. -->

## Build & Run

```bash
# Environment: mamba env bao305, Python 3.11, NumPy 2.0+
pip install -r requirements.txt
# pyrecon (only external physics dependency):
pip install "pyrecon[extras] @ git+https://github.com/cosmodesi/pyrecon"

# Run pipeline (must cd into src/ — bare imports)
cd src
python main.py --config ../configs/default.yaml                # full pipeline
python main.py --config ../configs/default.yaml --stage ics    # single stage

# Tests (from repo root)
pytest tests/ -v
```

Stages: `all`, `ics`, `nbody`, `lognormal`, `pk`, `recon`, `mcmc`, `plots`

## Architecture

See [DESIGN.md](../DESIGN.md) for full module status, implementation order, and physics reference.

```
configs/default.yaml   ← all runtime params (cosmology, box, simulation, mcmc)
src/main.py            ← pipeline orchestrator (--stage flag)
src/pk_input.py        ← Eisenstein-Hu P(k), growth factor, sound horizon
src/initial_conditions.py ← Zel'dovich ICs
src/pm_gravity.py      ← CIC + FFT Poisson solver
src/nbody.py           ← Leapfrog integrator + HDF5 snapshots
src/power_spectrum.py  ← FFT P(k) estimator with CIC deconvolution
src/lognormal.py       ← Lognormal catalog generator
src/mcmc.py            ← Metropolis-Hastings BAO template fitting
src/utils.py           ← cosmology helpers, plotting
tests/                 ← pytest (each test adds src/ to sys.path manually)
```

Config is loaded as a nested dict and sub-dicts (`cosmo`, `box`, `sim`, etc.) are passed to each stage function.

## Conventions

- **Units:** Mpc/h (distances), km/s (velocities), (Mpc/h)³ (P(k))
- **Particle positions:** shape `(3, N_particles)` everywhere; transpose to `(N, 3)` only for pyrecon
- **NumPy 2.0:** use `np.trapezoid`, NOT `np.trapz`
- **Imports:** bare relative imports within `src/` (e.g., `from utils import load_config`)
- **Outputs:** `outputs/figures/` (PNG), `outputs/snapshots/` (HDF5), `outputs/mcmc/` (chains + corner plots)

## Bugs Already Fixed (Do Not Reintroduce)

1. `np.trapz` → `np.trapezoid` (NumPy 2.0 rename)
2. Sound horizon: replaced broken integral with EH98 eq. 26 fitting formula → r_s ≈ 149.82 Mpc/h
3. Growth factor D(z=49): fixed Ω_m(z) to use E²(z) denominator and multiply by scale factor `a` → D ≈ 0.02 at z=49
4. Leapfrog integrator: added proper da→dt conversion factors `1/(aH)` for kick and `1/(a³H)` for drift; velocities converted to conjugate momentum `p=a·v_pec` during integration
5. `utils.py` comoving_distance: was integrating E(z) instead of 1/E(z)
6. `pk_input.py` no-wiggle transfer function: hardcoded `3` replaced with actual sound horizon `s` in Γ_eff formula
7. MCMC B_prior: widened to [0.5, 4.0] to accommodate galaxy bias b² factor

## Key Physics Reference

- **Growth factor D(z):** normalized to 1 at z=0; E²(z) = Ω_m(1+z)³ + Ω_Λ
- **CIC deconvolution:** P(k) estimator divides by W_CIC²(k); `np.sinc(x/np.pi)` for un-normalized sinc
- **Leapfrog:** kick-drift-kick in scale factor `a` with Hubble drag
- **Lognormal:** δ_LN = exp(G − σ²/2) − 1
- **MCMC template:** P_model(k) = B·[P_nw(k/α) + (P_w(k/α)−P_nw(k/α))·exp(−k²Σ²/2)]

## Response Format

- Lead with concept → approach → let user code
- End every response with: "Always check the correctness of AI-generated responses."
