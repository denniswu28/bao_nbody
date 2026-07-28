# BAO N-body Pipeline

[![CI](https://github.com/denniswu28/bao_nbody/actions/workflows/ci.yml/badge.svg)](https://github.com/denniswu28/bao_nbody/actions/workflows/ci.yml)

An end-to-end Baryon Acoustic Oscillation (BAO) simulation and analysis pipeline
built for PHY 305. The BAO scale is a "standard ruler" imprinted by sound waves
in the early universe (~150 Mpc in physical units, the same scale expressed as
~101 Mpc/h in the h-scaled units used throughout this pipeline). This project simulates how
gravity smears the signal over cosmic time and how BAO reconstruction partially
restores it.

**Educational scope**: results are from a small-box ($128^3$, 1500 Mpc/$h$)
simulation and are not suitable for research-grade cosmological constraints.
See `docs/RESULTS.md` for a detailed discussion of limitations.

## Pipeline Overview

```
Eisenstein-Hu P(k)
       │
       ▼
Zel'dovich Initial Conditions   ──► Lognormal Catalog (static mock)
       │                                      │
       ▼                                      │
PM N-body (leapfrog)                          │
       │                                      │
       ▼                                      ▼
Snapshots at z=49,2,1,0          P(k) Estimator (CIC + FFT)
       │                                      │
       ▼                                      ▼
Pyrecon (reconstruction)       Metropolis-Hastings MCMC
                                               │
                                               ▼
                                   BAO scale α, corner plots
```

## Modules

| File | Description |
|------|-------------|
| `src/pk_input.py` | Eisenstein-Hu P(k), sound horizon (Mpc/h), growth factor |
| `src/initial_conditions.py` | Zel'dovich approximation for 3D ICs |
| `src/pm_gravity.py` | Particle-mesh Poisson solver via FFT |
| `src/nbody.py` | Leapfrog integrator + snapshot management |
| `src/power_spectrum.py` | FFT-based P(k) estimator with CIC deconvolution |
| `src/lognormal.py` | 3D lognormal catalog generator |
| `src/mcmc.py` | Metropolis-Hastings BAO template fitting |
| `src/utils.py` | Shared utilities (cosmology, CIC, plotting helpers) |

## Installation

```bash
git clone https://github.com/denniswu28/bao_nbody.git
cd bao_nbody
pip install -r requirements.txt
```

The reconstruction stage requires `pyrecon`. **The `all`, `recon`, and
`mcmc` stages fail with an error if pyrecon is not installed** — they do not
silently skip. Supported on **Linux only** (macOS/Windows untested).
PyPI release 0.3.0 has a broken sdist — install from the pinned
source commit that includes numpy>=2 compatibility:

```bash
pip install "pyrecon @ git+https://github.com/cosmodesi/pyrecon@7d1e6c24598a05134c5958d109d9bcc7136ff83d"
# or with optional extras (NUFFT, etc.):
pip install "pyrecon[extras] @ git+https://github.com/cosmodesi/pyrecon@7d1e6c24598a05134c5958d109d9bcc7136ff83d"
```

> **NEEDS_REVIEW — transitive dependency gap:** `pyrecon@7d1e6c24` installs
> `pmesh @ git+https://github.com/MP-Gadget/pmesh` without a pinned commit,
> so full transitive reproducibility is not guaranteed.

## Usage

Run the full pipeline (requires pyrecon):

```bash
python src/main.py --config configs/default.yaml
```

Run individual stages:

```bash
python src/main.py --config configs/default.yaml --stage ics
python src/main.py --config configs/default.yaml --stage nbody
python src/main.py --config configs/default.yaml --stage lognormal
python src/main.py --config configs/default.yaml --stage pk
python src/main.py --config configs/default.yaml --stage recon
python src/main.py --config configs/default.yaml --stage mcmc
```

All commands above are run from the **repository root**; outputs are written
to `outputs/` relative to the root.  `main.py` resolves output paths via the
script's own location (`src/`), so the working directory does not affect where
files land.

Quick smoke test (lightweight config, requires pyrecon):

```bash
python src/main.py --config configs/ci_lightweight.yaml --stage all
```

## Testing

```bash
pytest tests/ -v
```

All tests are deterministic (fixed seeds) and run without network access.
The reconstruction stage itself has no unit tests (reconstruction correctness
is validated end-to-end via the CI lightweight pipeline smoke test).
The stage interface tests in `tests/test_stage_interfaces.py` verify that
`stage_recon` returns `None` when pyrecon is absent and that `stage_mcmc`
handles that sentinel correctly without pyrecon installed.

## Conventions and Units

- **Distances**: Mpc/$h$ throughout (box, P(k), r_s output of `sound_horizon()`)
- **Wave-numbers**: $h$/Mpc
- **P(k)**: $(h^{-1}$Mpc$)^3$
- **Positions**: shape `(3, N_particles)` everywhere
- **Sound horizon**: `sound_horizon()` returns ~100.9 Mpc/$h$ (= ~149.8 Mpc from EH98 eq. 26)
- **Seeds**: `simulation.seed=42`, `lognormal.seed=123` for reproducibility

## P(k) Estimator Limitations

`estimate_pk()` applies CIC deconvolution and restricts returned bins to
`k < 0.9 × k_Nyquist`. This cut reduces but does not eliminate near-Nyquist
aliasing and systematic bias from the finite CIC window. Results near the
Nyquist frequency should be treated with caution regardless of the cut.

## Configuration

| Config | Description |
|--------|-------------|
| `configs/default.yaml` | Full scientific run ($128^3$, $N_\text{mesh}=256$, 1500 Mpc/$h$, 50 steps) |
| `configs/ci_lightweight.yaml` | CI smoke test ($32^3$, $N_\text{mesh}=64$, 500 Mpc/$h$, 5 steps) |

## Results

See `docs/RESULTS.md` for a full description of all outputs. Key figures:

- `outputs/figures/pk_evolution.png` — P(k) at each snapshot showing BAO damping
- `outputs/figures/pk_comparison.png` — N-body vs lognormal vs reconstructed P(k)
- `outputs/mcmc/corner_*.png` — MCMC corner plots at each stage

## References and Attribution

See `CITATION.md` for full citations. Key references:

- Eisenstein & Hu (1998) — P(k) transfer function and sound-horizon formula
- Eisenstein et al. (2007) — BAO reconstruction
- Carroll, Press & Turner (1992) — growth factor approximation
- pyrecon: https://github.com/cosmodesi/pyrecon (BSD-3-Clause)
- LogNormalSimulations reference: https://github.com/Wide-Angle-Team/LogNormalSimulations

## License

MIT License — see `LICENSE`. Third-party components retain their own licenses;
see `CITATION.md` for details.
