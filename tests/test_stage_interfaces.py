"""
tests/test_stage_interfaces.py
-------------------------------
Regression tests for the inter-stage data contracts in src/main.py.

These tests verify that the return values from pipeline stages match the
expected interfaces consumed by downstream stages.  They guard against the
bug where stage_recon() returned a dict but stage_mcmc() passed it directly
to estimate_pk() which expects a positional array.
"""

import numpy as np
import sys, os
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import main as pipeline


def test_stage_recon_returns_none_without_pyrecon(monkeypatch):
    """stage_recon should return None when pyrecon is not installed.

    stage_mcmc and stage_plots both guard on `pos_recon is not None`,
    so this None sentinel must be returned (not raise) when pyrecon absent.
    """
    # Setting sys.modules['pyrecon'] = None causes `from pyrecon import ...`
    # to raise ImportError ("import of pyrecon halted; None in sys.modules"),
    # which is exactly what stage_recon's try/except ImportError catches.
    monkeypatch.setitem(sys.modules, 'pyrecon', None)

    fake_pos = np.zeros((3, 8))  # 2^3 particles, shape (3, N)
    cfg = {}
    cosmo = {'Omega_m': 0.3153}
    box = {'L': 500.0, 'N_mesh': 4}
    gal = {}
    out = {}

    result = pipeline.stage_recon(fake_pos, cfg, cosmo, box, gal, out)

    assert result is None, (
        f"stage_recon should return None when pyrecon is absent, got {type(result)}"
    )


def test_stage_mcmc_accepts_none_recon(tmp_path):
    """stage_mcmc should handle pos_recon=None without raising.

    When reconstruction is unavailable (pos_recon is None), stage_mcmc
    should skip the reconstructed-catalog fit and return chains for
    the other catalogs only.

    This is a unit-level smoke test: we pass minimal synthetic P(k) data
    (not a physically accurate catalog) and verify the interface works.
    """
    h, Omega_m, Omega_b, n_s, sigma8 = 0.6736, 0.3153, 0.0493, 0.9649, 0.8111
    k = np.linspace(0.05, 0.25, 10)
    Pk = 5000 * np.ones_like(k)
    Pk_err = 500 * np.ones_like(k)

    pk_results = {
        'nbody': [{'k': k, 'Pk': Pk, 'Pk_err': Pk_err, 'z': 0.0, 'a': 1.0}],
        'lognormal': {'k': k, 'Pk': Pk, 'Pk_err': Pk_err},
    }
    mcmc_cfg = {
        'n_steps': 10, 'n_burn': 2,
        'alpha_init': 1.0, 'Sigma_init': 5.0, 'B_init': 1.0,
        'alpha_prior': [0.5, 1.5], 'Sigma_prior': [0.0, 20.0], 'B_prior': [0.5, 4.0],
    }
    cfg = {'covariance': {'cov_file': None, 'N_mocks': 0}, 'mcmc': mcmc_cfg}
    cosmo = {
        'h': h, 'Omega_m': Omega_m, 'Omega_b': Omega_b,
        'n_s': n_s, 'sigma8': sigma8, 'z_eff': 0.38,
    }
    box = {'N': 4, 'L': 500.0, 'N_mesh': 8}
    gal = {'b': 1.5}
    out = {'mcmc_dir': str(tmp_path)}

    # Must not raise; pos_recon=None means recon fit is skipped
    chains = pipeline.stage_mcmc(pk_results, None, cfg, cosmo, box, gal, out)

    assert isinstance(chains, dict), "stage_mcmc should return a dict of chains"
    assert 'recon' not in chains, "recon chain should be absent when pos_recon=None"


def test_stage_mcmc_uses_pos_data_array(monkeypatch, tmp_path):
    """Behavioral: stage_mcmc must extract the pos_data ndarray from pos_recon dict.

    This is a regression test for the bug where stage_mcmc passed the entire
    pos_recon dict to estimate_pk(), which expects a (3, N) positional array.
    Passing a dict would cause an AttributeError on .shape inside estimate_pk.

    Strategy: spy on power_spectrum.estimate_pk via monkeypatch.  The spy records
    what type was passed as the positional argument.  We then assert it received
    an ndarray (not a dict), proving that stage_mcmc extracted pos_data correctly.
    """
    import power_spectrum as ps_mod

    received_types = []

    def spy_estimate_pk(pos, N, L, **kwargs):
        received_types.append(type(pos).__name__)
        k = np.linspace(0.02, 0.3, 5)
        # Return plausible P(k) ≈ 5000 (Mpc/h)^3 and nmodes ≈ 100 so that
        # downstream code (pk_error_gaussian, masking) does not raise.
        return k, 5000 * np.ones(5), 100 * np.ones(5)

    # Patch estimate_pk on the module object so the local import inside
    # stage_mcmc (`from power_spectrum import estimate_pk`) picks up the spy.
    monkeypatch.setattr(ps_mod, 'estimate_pk', spy_estimate_pk)

    # Also patch fit_bao to avoid expensive MCMC computation.
    def fake_fit_bao(*args, **kwargs):
        return np.zeros((5, 2)), 100.0

    monkeypatch.setattr(pipeline, 'fit_bao', fake_fit_bao)

    h, Omega_m, Omega_b, n_s, sigma8 = 0.6736, 0.3153, 0.0493, 0.9649, 0.8111
    k = np.linspace(0.05, 0.25, 10)
    Pk = 5000 * np.ones_like(k)
    Pk_err = 500 * np.ones_like(k)
    pk_results = {
        'nbody': [{'k': k, 'Pk': Pk, 'Pk_err': Pk_err, 'z': 0.0, 'a': 1.0}],
        'lognormal': {'k': k, 'Pk': Pk, 'Pk_err': Pk_err},
    }
    mcmc_cfg = {
        'n_steps': 5, 'n_burn': 1,
        'alpha_init': 1.0, 'Sigma_init': 5.0, 'B_init': 1.0,
        'alpha_prior': [0.5, 1.5], 'Sigma_prior': [0.0, 20.0], 'B_prior': [0.5, 4.0],
    }
    cfg = {'covariance': {'cov_file': None, 'N_mocks': 0}, 'mcmc': mcmc_cfg}
    cosmo = {
        'h': h, 'Omega_m': Omega_m, 'Omega_b': Omega_b,
        'n_s': n_s, 'sigma8': sigma8, 'z_eff': 0.38,
    }
    box = {'N': 4, 'L': 500.0, 'N_mesh': 8}
    gal = {'b': 1.5}
    out = {'mcmc_dir': str(tmp_path)}

    # pos_recon is a dict (as returned by stage_recon when pyrecon is installed)
    fake_pos_array = np.zeros((3, 10))
    pos_recon = {'pos_data': fake_pos_array, 'delta_rec': np.zeros((4, 4, 4))}

    chains = pipeline.stage_mcmc(pk_results, pos_recon, cfg, cosmo, box, gal, out)

    assert isinstance(chains, dict), "stage_mcmc should return a dict"
    assert 'recon' in chains, "recon chain should be present when pos_recon is a dict"

    # Verify estimate_pk was called exactly once (for the recon catalog)
    # and received an ndarray, not a dict.
    assert len(received_types) == 1, (
        f"Expected 1 estimate_pk call (recon path), got {len(received_types)}"
    )
    assert received_types[0] == 'ndarray', (
        f"estimate_pk received type '{received_types[0]}', expected 'ndarray'. "
        "stage_mcmc must extract pos_recon['pos_data'] before calling estimate_pk; "
        "passing the raw dict causes AttributeError on .shape."
    )


def test_stage_plots_does_not_call_reconstruction(monkeypatch, tmp_path):
    """Behavioral: --stage plots dispatch must not invoke stage_recon.

    stage_plots() does not use pos_recon; dispatching to plots must not trigger
    a reconstruction call (which would fail closed if pyrecon is absent).
    """
    # Write a minimal config with absolute output paths.
    cfg_data = {
        'cosmology': {'h': 0.6736, 'Omega_m': 0.3153, 'Omega_b': 0.0493,
                      'n_s': 0.9649, 'sigma8': 0.8111, 'z_eff': 0.38},
        'box': {'L': 500.0, 'N': 8, 'N_mesh': 16},
        'galaxy': {'b': 1.5, 'nbar': 3.0e-4},
        'simulation': {'n_steps': 1, 'z_initial': 49.0,
                       'z_snapshots': [0.0], 'seed': 42},
        'lognormal': {'seed': 123},
        'mcmc': {'n_steps': 5, 'n_burn': 1, 'alpha_init': 1.0,
                 'Sigma_init': 5.0, 'B_init': 1.0,
                 'alpha_prior': [0.5, 1.5], 'Sigma_prior': [0.0, 20.0],
                 'B_prior': [0.5, 4.0]},
        'output': {
            'snapshot_dir': str(tmp_path / 'snap'),
            'figure_dir': str(tmp_path / 'fig'),
            'mcmc_dir': str(tmp_path / 'mcmc'),
            'save_snapshots': False,
            'make_animation': False,
            'animation_fps': 5,
        },
    }
    config_path = tmp_path / 'test_cfg.yaml'
    config_path.write_text(yaml.dump(cfg_data))

    k_stub = np.array([0.1])
    pk_stub = {
        'nbody': [{'k': k_stub, 'Pk': np.array([1000.0]),
                   'Pk_err': np.array([100.0]), 'z': 0.0, 'a': 1.0}],
        'lognormal': {'k': k_stub, 'Pk': np.array([1000.0]),
                      'Pk_err': np.array([100.0])},
        'theory': {'k': k_stub, 'Pk': np.array([1000.0])},
    }
    snapshots_stub = [{'pos': np.zeros((3, 8)), 'vel': np.zeros((3, 8)),
                       'z': 0.0, 'a': 1.0}]

    recon_called = []

    monkeypatch.setattr(pipeline, 'stage_ics',
                        lambda *a, **kw: (np.zeros((3, 8)), np.zeros((3, 8))))
    monkeypatch.setattr(pipeline, 'stage_nbody',
                        lambda *a, **kw: snapshots_stub)
    monkeypatch.setattr(pipeline, 'stage_lognormal',
                        lambda *a, **kw: (np.zeros((3, 8)), np.zeros((4, 4, 4))))
    monkeypatch.setattr(pipeline, 'stage_pk',
                        lambda *a, **kw: pk_stub)
    monkeypatch.setattr(pipeline, 'stage_recon',
                        lambda *a, **kw: recon_called.append(True) or None)
    monkeypatch.setattr(pipeline, 'stage_plots',
                        lambda *a, **kw: None)

    monkeypatch.setattr(sys, 'argv',
                        ['main.py', '--config', str(config_path), '--stage', 'plots'])
    pipeline.main()

    assert len(recon_called) == 0, (
        "stage_recon must not be called for '--stage plots'; "
        "stage_plots() does not use pos_recon."
    )
