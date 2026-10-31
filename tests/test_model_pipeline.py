"""
Tests for Model Evaluation Protocols and Pipeline Execution.
Verifies Protocol A (1-step teacher forced) vs Protocol B (10-year recursive rollout),
baseline models (Naive Persistence, Seasonal Persistence), and comparative metrics.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

COMPARISON_CSV = os.path.join(PROJECT_ROOT, "modeling_outputs", "evaluation_protocols_comparison.csv")
BUNDLE_PATH = os.path.join(PROJECT_ROOT, "modeling_outputs", "ridge_climate_model.joblib")


def test_comparison_csv_exists_and_structure():
    """Verify evaluation_protocols_comparison.csv exists and contains all required models."""
    assert os.path.exists(COMPARISON_CSV), f"Missing comparison CSV: {COMPARISON_CSV}"
    df = pd.read_csv(COMPARISON_CSV)
    
    # 6 rows: 3 models x 2 protocols
    assert len(df) == 6, f"Expected 6 entries (3 models x 2 protocols), got {len(df)}"
    expected_cols = ["Model", "Protocol", "RMSE", "MAE", "R2"]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"
    
    models = set(df["Model"].values)
    assert any("Ridge" in m for m in models)
    assert any("Naive Persistence" in m for m in models)
    assert any("Seasonal Persistence" in m for m in models)
    
    protocols = set(df["Protocol"].values)
    assert "1-Step Teacher-Forced" in protocols
    assert "10-Yr Recursive Rollout" in protocols


def test_protocol_a_teacher_forced_hierarchy():
    """
    Verify Protocol A results:
    Ridge achieves lowest RMSE (~0.1124°C) and highest R2 (~0.6532),
    outperforming Naive Persistence (~0.1179°C, R2 ~0.6186) and
    Seasonal Persistence (~0.2082°C, R2 ~ -0.1898).
    """
    df = pd.read_csv(COMPARISON_CSV)
    df_proto_a = df[df["Protocol"] == "1-Step Teacher-Forced"].set_index("Model")
    
    ridge_row = [idx for idx in df_proto_a.index if "Ridge" in idx][0]
    naive_row = [idx for idx in df_proto_a.index if "Naive Persistence" in idx][0]
    season_row = [idx for idx in df_proto_a.index if "Seasonal Persistence" in idx][0]
    
    r_rmse = float(df_proto_a.loc[ridge_row, "RMSE"])
    n_rmse = float(df_proto_a.loc[naive_row, "RMSE"])
    s_rmse = float(df_proto_a.loc[season_row, "RMSE"])
    
    assert r_rmse < n_rmse < s_rmse, f"Unexpected RMSE ordering: {r_rmse}, {n_rmse}, {s_rmse}"
    
    r_r2 = float(df_proto_a.loc[ridge_row, "R2"])
    n_r2 = float(df_proto_a.loc[naive_row, "R2"])
    s_r2 = float(df_proto_a.loc[season_row, "R2"])
    
    assert r_r2 > n_r2 > s_r2, f"Unexpected R2 ordering: {r_r2}, {n_r2}, {s_r2}"
    assert r_r2 > 0.60, f"Ridge Protocol A R2 unexpectedly low: {r_r2}"


def test_protocol_b_recursive_rollout_hierarchy():
    """
    Verify Protocol B results:
    10-year autonomous recursive rollout exhibits error accumulation:
    Ridge RMSE increases to ~0.1754°C and R2 drops to ~0.1555,
    but retains superior performance over Naive Persistence (R2 ~ -0.6517)
    and Seasonal Persistence (R2 ~ -0.6951).
    """
    df = pd.read_csv(COMPARISON_CSV)
    df_proto_b = df[df["Protocol"] == "10-Yr Recursive Rollout"].set_index("Model")
    df_proto_a = df[df["Protocol"] == "1-Step Teacher-Forced"].set_index("Model")
    
    ridge_row = [idx for idx in df_proto_b.index if "Ridge" in idx][0]
    naive_row = [idx for idx in df_proto_b.index if "Naive Persistence" in idx][0]
    season_row = [idx for idx in df_proto_b.index if "Seasonal Persistence" in idx][0]
    
    r_rmse = float(df_proto_b.loc[ridge_row, "RMSE"])
    n_rmse = float(df_proto_b.loc[naive_row, "RMSE"])
    s_rmse = float(df_proto_b.loc[season_row, "RMSE"])
    
    assert r_rmse < n_rmse, f"Ridge recursive RMSE ({r_rmse}) must be lower than Naive ({n_rmse})"
    assert r_rmse < s_rmse, f"Ridge recursive RMSE ({r_rmse}) must be lower than Seasonal ({s_rmse})"
    
    r_r2 = float(df_proto_b.loc[ridge_row, "R2"])
    n_r2 = float(df_proto_b.loc[naive_row, "R2"])
    s_r2 = float(df_proto_b.loc[season_row, "R2"])
    
    assert r_r2 > 0.0, f"Ridge recursive R2 should remain positive: {r_r2}"
    assert n_r2 < 0.0, f"Naive persistence recursive R2 should be negative: {n_r2}"
    assert s_r2 < 0.0, f"Seasonal persistence recursive R2 should be negative: {s_r2}"
    
    # Error compounding check: Protocol B RMSE > Protocol A RMSE
    r_a_rmse = float(df_proto_a.loc[ridge_row, "RMSE"])
    assert r_rmse > r_a_rmse, "Recursive rollout RMSE must exceed teacher-forced 1-step RMSE"
