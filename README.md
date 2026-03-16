# BAO N-body Pipeline

A 3D Baryon Acoustic Oscillation (BAO) simulation and analysis pipeline built for PHY 305.

The BAO scale (~150 Mpc/h) is a "standard ruler" imprinted in the early universe that appears
as a preferred galaxy separation in large-scale structure. This project simulates how gravity
smears out this signal over cosmic time, and how BAO reconstruction partially restores it.

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
       ▼                                      │
Snapshots at z=∞,2,1,0                        │
       │                                      │
       ▼                                      ▼
  Pyrecon (reconstruction)       ◄────  P(k) Estimator
       │                                      │
       ▼                                      ▼
  Recovered P(k)              Metropolis-Hastings MCMC
                                              │
                                              ▼
                                  BAO scale α, corner plots
```

## Modules

| File | Description |
|------|-------------|
| `src/pk_input.py` | Eisenstein-Hu P(k) and no-wiggle spectrum |
| `src/initial_conditions.py` | Zel'dovich approximation for 3D ICs |
| `src/pm_gravity.py` | Particle-mesh Poisson solver via FFT |
| `src/nbody.py` | Leapfrog integrator + snapshot management |
| `src/power_spectrum.py` | FFT-based P(k) estimator with CIC correction |
| `src/lognormal.py` | 3D lognormal catalog generator |
| `src/mcmc.py` | Metropolis-Hastings BAO template fitting |
| `src/utils.py` | Shared utilities (cosmology, CIC, plotting helpers) |

## Installation

```bash
git clone https://github.com/<your-username>/bao_nbody.git
cd bao_nbody
pip install -r requirements.txt
```

For reconstruction:
```bash
pip install pyrecon
```

## Usage

Run the full pipeline:
```bash
python src/main.py --config configs/default.yaml
```

Run individual stages:
```bash
python src/main.py --config configs/default.yaml --stage ics
python src/main.py --config configs/default.yaml --stage nbody
python src/main.py --config configs/default.yaml --stage lognormal
python src/main.py --config configs/default.yaml --stage recon
python src/main.py --config configs/default.yaml --stage mcmc
```

## Results

The pipeline produces:

- `outputs/figures/pk_evolution.png` — P(k) at each snapshot showing BAO damping
- `outputs/figures/pk_comparison.png` — N-body vs lognormal vs reconstructed P(k)
- `outputs/figures/density_slices.png` — 2D slices of the density field at each snapshot
- `outputs/mcmc/corner_nbody.png` — MCMC corner plot for N-body final snapshot
- `outputs/mcmc/corner_lognormal.png` — MCMC corner plot for lognormal catalog
- `outputs/mcmc/corner_recon.png` — MCMC corner plot after reconstruction

## References

- Eisenstein & Hu (1998) — P(k) fitting formula
- Eisenstein et al. (2007) — BAO reconstruction
- Pyrecon: https://github.com/cosmodesi/pyrecon
- LogNormalSimulations (reference pipeline): https://github.com/Wide-Angle-Team/LogNormalSimulations
