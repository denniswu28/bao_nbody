"""
tests/test_utils.py
-------------------
Tests for cosmology utilities.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import comoving_distance, hubble_z


def test_comoving_distance_z0():
    """Comoving distance at z=0 should be 0."""
    chi = comoving_distance(0.0, h=0.6736, Omega_m=0.3153)
    assert abs(chi) < 1e-6, f"chi(z=0) = {chi}"


def test_comoving_distance_increasing():
    """Comoving distance should increase with redshift."""
    h, Omega_m = 0.6736, 0.3153
    z_arr = [0.1, 0.5, 1.0, 2.0]
    chi_arr = [comoving_distance(z, h, Omega_m) for z in z_arr]
    assert all(chi_arr[i] < chi_arr[i+1] for i in range(len(chi_arr)-1)), \
        "Comoving distance not monotonically increasing"


def test_hubble_z0():
    """H(z=0) should equal H0 = 100*h km/s/Mpc."""
    h, Omega_m = 0.6736, 0.3153
    Hz = hubble_z(0.0, h, Omega_m)
    assert abs(Hz - 100.0) < 1e-3, f"H(z=0) = {Hz:.4f}, expected 100.0"


def test_comoving_distance_planck():
    """Comoving distance to z=0.38 should be ~1100 Mpc/h for Planck cosmology."""
    chi = comoving_distance(0.38, h=0.6736, Omega_m=0.3153)
    assert 900 < chi < 1300, f"chi(z=0.38) = {chi:.1f} Mpc/h, out of expected range"
