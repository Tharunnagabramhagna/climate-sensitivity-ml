"""
Tests for Physical Climate Modeling (DOECLIM / Hector SCM Engine).
Verifies energy conservation, numerical stability, time-step convergence, and simulation output files.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from simulate_hector_2xco2 import (
    DOECLIMModel,
    get_abrupt_2xco2_forcing,
    get_1pct_co2_forcing,
    verify_energy_conservation_and_convergence
)

HECTOR_CSV = os.path.join(PROJECT_ROOT, "hector_outputs", "hector_2xco2_ensemble_summary.csv")


def test_hector_ensemble_summary_exists():
    """Verify that Phase 7 simulation summary CSV was generated with correct shape."""
    assert os.path.exists(HECTOR_CSV), f"Missing Hector summary file: {HECTOR_CSV}"
    df = pd.read_csv(HECTOR_CSV)
    
    # 501 rows (Year 0 through Year 500)
    assert len(df) == 501, f"Expected 501 simulation years, got {len(df)}"
    assert "Year" in df.columns
    assert "Median_Temp" in df.columns
    assert "Deterministic_Central" in df.columns
    assert "ML_Ridge_Roll" in df.columns


def test_doeclim_energy_conservation_and_asymptote():
    """Verify DOECLIM implicit solver produces stable, non-negative, bounded warming."""
    ecs = 3.0
    f2x = 3.93
    model = DOECLIMModel(ecs=ecs, diffusivity=2.3, f_2x=f2x)
    
    # 100-year abrupt 2xCO2 run
    f_abrupt = get_abrupt_2xco2_forcing(n_years=100, f_2x=f2x)
    res = model.run_scenario(f_abrupt, n_years=100)
    
    temps = res["temp"]
    imbalances = res["imbalance"]
    ohc = res["ohc_zj"]
    
    # Monotonic surface warming under constant positive step forcing
    diffs = np.diff(temps)
    assert (diffs >= 0).all(), "Warming trajectory must be non-decreasing under constant step forcing"
    
    # Initial warming is zero
    assert temps[0] == 0.0
    
    # Year 100 warming must be positive but less than asymptotic equilibrium ECS (3.0°C)
    assert 1.0 < temps[100] < ecs, f"Year 100 warming ({temps[100]:.2f}°C) violates physical bounds"
    
    # Top-of-atmosphere radiative imbalance must be positive and decaying toward zero
    assert imbalances[0] == f2x
    assert imbalances[100] < imbalances[0]
    assert imbalances[100] > 0.0
    
    # Cumulative Ocean Heat Content must be strictly increasing
    assert (np.diff(ohc) > 0).all(), "Cumulative ocean heat uptake must be strictly monotonic"


def test_transient_climate_response_less_than_ecs():
    """Verify TCR < ECS relationship (fundamental physical climate constraint)."""
    ecs = 3.0
    f2x = 3.93
    model = DOECLIMModel(ecs=ecs, diffusivity=2.3, f_2x=f2x)
    
    f_1pct = get_1pct_co2_forcing(n_years=70, f_2x=f2x)
    res = model.run_scenario(f_1pct, n_years=70)
    
    tcr = res["temp"][70]
    # In physics, TCR / ECS is typically between 0.40 and 0.65 due to ocean heat uptake
    assert 0.8 < tcr < 2.0, f"Diagnosed TCR ({tcr:.2f}°C) is outside physical range"
    assert tcr < ecs, f"TCR ({tcr:.2f}°C) must be strictly less than ECS ({ecs:.2f}°C)"


def test_doeclim_energy_conservation_precision():
    """
    Verify independent integration of cumulative TOA energy uptake
    matches internal ocean heat content change with relative error < 1e-10%.
    """
    check = verify_energy_conservation_and_convergence(n_years=100, verbose=False)
    assert check["energy_rel_err_pct"] < 1e-10, (
        f"Energy conservation error ({check['energy_rel_err_pct']:.4e}%) exceeds tolerance 1e-10%"
    )


def test_doeclim_time_step_convergence():
    """
    Verify numerical convergence: surface warming at Year 100 with refined time steps
    (dt = 0.5 yr, dt = 0.1 yr) differs from standard dt = 1.0 yr by less than 0.01°C.
    """
    check = verify_energy_conservation_and_convergence(n_years=100, verbose=False)
    assert check["diff_05_1"] < 0.01, f"dt=0.5 difference ({check['diff_05_1']:.4f}°C) exceeds 0.01°C"
    assert check["diff_01_1"] < 0.01, f"dt=0.1 difference ({check['diff_01_1']:.4f}°C) exceeds 0.01°C"
