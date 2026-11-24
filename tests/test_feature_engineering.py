"""
Tests for Feature Engineering, Leakage Prevention, and Partitioning.
Verifies strictly past lags, chronological splits, and feature schema.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from run_pipeline import load_and_prepare_data
from src.features import build_climate_features, build_step_feature_vector, FEATURE_COLS, TARGET_COL


def test_feature_matrix_integrity():
    """Verify clean dataset output from load_and_prepare_data."""
    df = load_and_prepare_data()
    
    # 814 months - 13 initialization lag months = 801 valid rows (1959-04 to 2025-12)
    assert len(df) == 801, f"Expected 801 rows after lag initialization, got {len(df)}"
    assert df["date"].iloc[0] == "1959-04"
    assert df["date"].iloc[-1] == "2025-12"


def test_required_feature_columns():
    """Verify all 9 predictive features are correctly generated without NaNs."""
    df = load_and_prepare_data()
    expected_features = [
        "temp_lag1", "temp_lag2", "temp_lag3", "temp_lag12",
        "ln_co2_lag1", "co2_diff1", "co2_diff12",
        "sin_month", "cos_month"
    ]
    for col in expected_features:
        assert col in df.columns, f"Missing feature: {col}"
        assert df[col].isna().sum() == 0, f"Found NaN in feature: {col}"


def test_zero_future_leakage():
    """Verify that features only reference past observations."""
    df = load_and_prepare_data()
    
    # temp_lag1 at row i must match target_temp at row i-1
    assert np.allclose(df["temp_lag1"].iloc[1:].values, df["target_temp"].iloc[:-1].values)
    
    # ln_co2_lag1 must match log of co2_lag1
    assert np.allclose(df["ln_co2_lag1"].values, np.log(df["co2_lag1"].values))
    
    # Seasonal harmonics must be strictly bounded in [-1, 1]
    assert df["sin_month"].between(-1.0, 1.0).all()
    assert df["cos_month"].between(-1.0, 1.0).all()


def test_chronological_splits():
    """Verify partition date windows and counts match experimental specification."""
    df = load_and_prepare_data()
    
    train_mask = df["year"] <= 2005
    val_mask = (df["year"] >= 2006) & (df["year"] <= 2015)
    test_mask = df["year"] >= 2016
    
    assert train_mask.sum() == 561, f"Expected 561 training months, got {train_mask.sum()}"
    assert val_mask.sum() == 120, f"Expected 120 validation months, got {val_mask.sum()}"
    assert test_mask.sum() == 120, f"Expected 120 test months, got {test_mask.sum()}"
    
    # No temporal overlap
    assert df.loc[train_mask, "year"].max() < df.loc[val_mask, "year"].min()
    assert df.loc[val_mask, "year"].max() < df.loc[test_mask, "year"].min()


def test_centralized_feature_builder_consistency():
    """Verify src.features produces identical output and respects FEATURE_COLS order."""
    df_raw = pd.read_csv(os.path.join(PROJECT_ROOT, "climate_paired.csv"), comment="#")
    df_feat = build_climate_features(df_raw, dropna=True)
    df_pipe = load_and_prepare_data()

    assert len(df_feat) == 801
    assert df_feat.shape == df_pipe.shape
    for col in FEATURE_COLS + [TARGET_COL]:
        assert np.allclose(df_feat[col].values, df_pipe[col].values)


def test_step_feature_vector_builder():
    """Verify build_step_feature_vector creates correct shape and raises on short history."""
    temp_hist = [0.1 * i for i in range(12)]
    co2_hist = [350.0 + i for i in range(13)]
    vec = build_step_feature_vector(temp_hist, co2_hist, month=5)

    assert vec.shape == (1, 9)
    # Check lag 1 temp and CO2 match last elements
    assert np.isclose(vec[0, 0], temp_hist[-1])
    assert np.isclose(vec[0, 4], np.log(co2_hist[-1]))

    # Test error handling on insufficient history
    with pytest.raises(ValueError):
        build_step_feature_vector(temp_hist[:11], co2_hist, month=5)
    with pytest.raises(ValueError):
        build_step_feature_vector(temp_hist, co2_hist[:12], month=5)


