"""
Tests for Model Serialization, Joblib Persistence, and Inference Output.
Verifies bundle contents, deserialization, deterministic predictions, and 2026 forecast intervals.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from predict import load_model, predict_from_features, compute_prediction_intervals

MODEL_PATH = os.path.join(PROJECT_ROOT, "modeling_outputs", "ridge_climate_model.joblib")
PROJECTIONS_2026_CSV = os.path.join(PROJECT_ROOT, "modeling_outputs", "projected_anomalies_2026.csv")


def test_model_artifact_exists():
    """Verify serialized joblib bundle is present on disk."""
    assert os.path.exists(MODEL_PATH), f"Serialized model not found at {MODEL_PATH}"


def test_model_bundle_contents():
    """Verify metadata and schema inside the joblib bundle."""
    bundle = load_model(MODEL_PATH)
    
    assert isinstance(bundle, dict), "Model bundle is not a dictionary"
    required_keys = {
        "model", "model_type", "hyperparameters", "feature_names",
        "target_name", "target_baseline", "training_window",
        "test_window", "test_metrics", "test_metrics_1step",
        "test_metrics_recursive_10yr", "validation_residual_std",
        "coefficients", "intercept"
    }
    assert required_keys.issubset(set(bundle.keys())), "Missing required keys in model bundle"
    
    # Check hyperparameters and metrics
    assert bundle["model_type"] == "Ridge Regression"
    assert bundle["hyperparameters"]["alpha"] == 0.01
    assert bundle["test_metrics"]["R2"] > 0.60, f"Saved R2 is unexpectedly low: {bundle['test_metrics']['R2']}"
    assert bundle["test_metrics"]["RMSE"] < 0.12, f"Saved RMSE is unexpectedly high: {bundle['test_metrics']['RMSE']}"
    assert bundle["validation_residual_std"] > 0.08, "Validation residual std unexpectedly small"
    assert len(bundle["feature_names"]) == 9


def test_inference_prediction_shape_and_range():
    """Verify that deserialized model generates valid anomaly predictions."""
    bundle = load_model(MODEL_PATH)
    
    # Synthetic realistic feature vector (9 inputs)
    # Lags ~ +1.0°C, CO2 ~ 420 ppm (ln ~ 6.04), sin/cos
    test_features = np.array([[
        1.05, 0.98, 0.95, 0.88,
        np.log(420.0), 0.5, 2.5,
        0.5, 0.866
    ]])
    
    pred = predict_from_features(test_features, bundle)
    
    assert isinstance(pred, np.ndarray), "Prediction is not a numpy array"
    assert pred.shape == (1,), f"Unexpected prediction shape: {pred.shape}"
    
    # Realistic anomaly range for current conditions (+0.5°C to +2.0°C)
    val = float(pred[0])
    assert 0.5 <= val <= 2.0, f"Predicted anomaly out of plausible physical bounds: {val}"


def test_prediction_interval_propagation():
    """Verify that autoregressive forecast error variance accumulates monotonically."""
    bundle = load_model(MODEL_PATH)
    sigma_h = compute_prediction_intervals(bundle, n_steps=9)
    
    assert len(sigma_h) == 9
    assert (sigma_h > 0).all(), "Standard errors must be strictly positive"
    
    # Monotonically non-decreasing uncertainty
    diffs = np.diff(sigma_h)
    assert (diffs >= 0).all(), "Forecast uncertainty must not shrink with longer forecast horizon"
    
    # Step 1 uncertainty matches validation sigma (~0.111°C)
    assert np.isclose(sigma_h[0], bundle["validation_residual_std"], atol=1e-4)
    
    # Step 9 uncertainty is larger due to AR memory compounding (~0.142°C)
    assert sigma_h[-1] > sigma_h[0]


def test_projected_anomalies_2026_file():
    """Verify that projected_anomalies_2026.csv exists and has correct columns and labeling."""
    assert os.path.exists(PROJECTIONS_2026_CSV), f"Missing projections file: {PROJECTIONS_2026_CSV}"
    df = pd.read_csv(PROJECTIONS_2026_CSV)
    
    assert len(df) == 9, f"Expected 9 monthly projections (2026-01 to 2026-09), got {len(df)}"
    expected_cols = [
        "Date", "NOAA_Measured_CO2_ppm", "Forecast_Temp_Anomaly_C",
        "CI_Lower_95", "CI_Upper_95", "Baseline", "Projection_Type"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"
        
    assert (df["CI_Lower_95"] < df["Forecast_Temp_Anomaly_C"]).all()
    assert (df["Forecast_Temp_Anomaly_C"] < df["CI_Upper_95"]).all()
    assert (df["Projection_Type"] == "Statistical Projection (Unobserved)").all()
