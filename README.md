# BAO N-body Pipeline

[![CI](https://github.com/denniswu28/bao_nbody/actions/workflows/ci.yml/badge.svg)](https://github.com/denniswu28/bao_nbody/actions/workflows/ci.yml)

An end-to-end Baryon Acoustic Oscillation (BAO) simulation and analysis pipeline
built for PHY 305. The BAO scale (~150 Mpc, ~101 Mpc/h) is a "standard ruler"
imprinted by sound waves in the early universe. This project simulates how
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

The reconstruction stage requires `pyrecon` (optional; pipeline skips
gracefully if absent). PyPI release 0.3.0 has a broken sdist — install
from source:

```bash
pip install "pyrecon @ git+https://github.com/cosmodesi/pyrecon"
# or with optional extras (NUFFT, etc.):
pip install "pyrecon[extras] @ git+https://github.com/cosmodesi/pyrecon"
```

## Usage

Run the full pipeline (requires ~20 min and ~4 GB RAM for default config):

```bash
cd src
python main.py --config ../configs/default.yaml
```

Run individual stages:

```bash
python main.py --config ../configs/default.yaml --stage ics
python main.py --config ../configs/default.yaml --stage nbody
python main.py --config ../configs/default.yaml --stage lognormal
python main.py --config ../configs/default.yaml --stage pk
python main.py --config ../configs/default.yaml --stage recon
python main.py --config ../configs/default.yaml --stage mcmc
```

Quick smoke test (lightweight config, ~30 seconds):

```bash
cd src
python main.py --config ../configs/ci_lightweight.yaml --stage ics
```

## Testing

```bash
pytest tests/ -v
```

All tests are deterministic (fixed seeds) and run without network access.
Reconstruction tests are skipped if pyrecon is not installed.

## Conventions and Units

- **Distances**: Mpc/$h$ throughout (box, P(k), r_s output of `sound_horizon()`)
- **Wave-numbers**: $h$/Mpc
- **P(k)**: $(h^{-1}$Mpc$)^3$
- **Positions**: shape `(3, N_particles)` everywhere
- **Sound horizon**: `sound_horizon()` returns ~100.9 Mpc/$h$ (= ~149.8 Mpc from EH98 eq. 26)
- **Seeds**: `simulation.seed=42`, `lognormal.seed=123` for reproducibility

## Configuration

| Config | Description | Resources |
|--------|-------------|-----------|
| `configs/default.yaml` | Full scientific run ($128^3$, 1500 Mpc/$h$, 50 steps) | ~20 min, ~4 GB |
| `configs/ci_lightweight.yaml` | CI smoke test ($16^3$, 500 Mpc/$h$, 5 steps) | ~30 s, < 500 MB |

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
