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

