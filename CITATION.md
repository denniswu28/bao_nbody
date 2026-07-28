# Citations and Attributions

This project was developed as a course project for PHY 305.

**Authorship:** The core simulation modules, MCMC implementation, and
scientific validation described in `docs/RESULTS.md` were written by
Dennis Wu.  CI/CD infrastructure (`.github/workflows/ci.yml`,
`configs/ci_lightweight.yaml`), regression tests, and documentation
revisions in the `copilot/make-bao-nbody-reproducible` branch were
produced with GitHub Copilot coding-agent assistance and are pending
Dennis Wu's review before being considered validated.  The MIT license
in `LICENSE` covers Dennis Wu's original contributions; third-party
library licenses are listed separately below.

Third-party libraries, papers, and upstream code are cited below.

---

## Papers

### Eisenstein & Hu (1998)
Power spectrum transfer function and sound-horizon fitting formula used in
`src/pk_input.py`:

> Eisenstein, D. J., & Hu, W. (1998). Power Spectra for Cold Dark Matter and
> Its Variants. *The Astrophysical Journal*, 496(2), 605–614.
> https://doi.org/10.1086/305424

Equations used:
- Transfer function: EH98 §§ 2–3 (full and no-wiggle)
- Sound horizon: EH98 Eq. 26 (fitting formula; result in Mpc, converted to
  Mpc/h by multiplying by h)
- No-wiggle transfer function: EH98 Eq. 29–31

### Carroll, Press & Turner (1992)
Approximate growth factor formula used in `src/pk_input.py`:

> Carroll, S. M., Press, W. H., & Turner, E. L. (1992). The cosmological
> constant. *Annual Review of Astronomy and Astrophysics*, 30, 499–542.
> https://doi.org/10.1146/annurev.aa.30.090192.002435

### Eisenstein et al. (2007)
BAO reconstruction methodology:

> Eisenstein, D. J., Seo, H.-J., Sirko, E., & Spergel, D. N. (2007).
> Improving Cosmological Distance Measurements by Reconstruction of the
> Baryon Acoustic Peak. *The Astrophysical Journal*, 664(2), 675–679.
> https://doi.org/10.1086/518712

### Hartlap, Simon & Schneider (2007)
Bias correction for inverted sample covariance matrices:

> Hartlap, J., Simon, P., & Schneider, P. (2007). Why your model parameter
> confidences might be too optimistic. *Astronomy & Astrophysics*, 464(1),
> 399–404. https://doi.org/10.1051/0004-6361:20066170

---

## Software

### pyrecon
BAO reconstruction library (optional dependency, used in `src/main.py`
`stage_recon`):

> cosmodesi developers, cosmodesi/pyrecon. GitHub.
> https://github.com/cosmodesi/pyrecon
>
> License: BSD-3-Clause (see https://github.com/cosmodesi/pyrecon/blob/main/LICENSE)
>
> Install: `pip install "pyrecon @ git+https://github.com/cosmodesi/pyrecon@7d1e6c24598a05134c5958d109d9bcc7136ff83d"`
> Pinned commit: `7d1e6c24` (2026-03-26) — includes numpy>=2 copy=False compatibility fix.
> Supported on Linux only; Windows and macOS are untested.

### LogNormalSimulations (reference)
The lognormal mock generation approach in `src/lognormal.py` follows the
methodology described in:

> Wide-Angle-Team/LogNormalSimulations. GitHub.
> https://github.com/Wide-Angle-Team/LogNormalSimulations

### NumPy, SciPy, Matplotlib, corner
Standard scientific Python stack. See their respective documentation for
licenses (BSD-family).

---

## Cosmological Parameters

Planck 2018 results used as defaults in `configs/default.yaml`:

> Planck Collaboration (2020). Planck 2018 results. VI. Cosmological
> parameters. *Astronomy & Astrophysics*, 641, A6.
> https://doi.org/10.1051/0004-6361/201833910

---

## Educational Scope

This project is an **educational demonstration** developed for a course
assignment. Results should not be used for research-grade cosmological
constraints. See `docs/RESULTS.md` for a detailed discussion of known
limitations.
