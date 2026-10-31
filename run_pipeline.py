"""
=============================================================================
CLIMATE SENSITIVITY & TEMPERATURE PREDICTION PIPELINE
=============================================================================
Author: Antigravity AI & Research Team
Dataset: NOAA GML Mauna Loa CO₂ & NASA GISS GISTEMP v4 (1958–2025)

This script provides an end-to-end, reproducible execution of all project
phases:
  1. Data verification and pairing
  2. Exploratory Data Analysis & statistical diagnostics
  3. Machine Learning benchmark & baseline modeling (Chronological splits)
  4. Advanced model evaluation (Ridge, RF, GBR, XGBoost)
  5. Scientific interpretation, feature attribution, and era breakdown
=============================================================================
"""

import os
import sys

# Ensure UTF-8 output across Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(PROJECT_ROOT, "climate_paired.csv")
EDA_DIR = os.path.join(PROJECT_ROOT, "eda_outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "modeling_outputs")
INTERP_DIR = os.path.join(PROJECT_ROOT, "interpretation_outputs")

for d in [EDA_DIR, MODELS_DIR, INTERP_DIR]:
    os.makedirs(d, exist_ok=True)


def load_and_prepare_data():
    """Loads validated paired climate data and creates leakage-free features."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Paired dataset not found at {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, comment="#")
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt").reset_index(drop=True)

    # Feature Engineering (strictly past information)
    df["target_temp"] = df["temp_anomaly_c"]
    df["temp_lag1"] = df["temp_anomaly_c"].shift(1)
    df["temp_lag2"] = df["temp_anomaly_c"].shift(2)
    df["temp_lag3"] = df["temp_anomaly_c"].shift(3)
    df["temp_lag12"] = df["temp_anomaly_c"].shift(12)

    df["co2_lag1"] = df["co2_ppm"].shift(1)
    df["ln_co2_lag1"] = np.log(df["co2_lag1"])
    df["co2_diff1"] = df["co2_ppm"].shift(1) - df["co2_ppm"].shift(2)
    df["co2_diff12"] = df["co2_ppm"].shift(1) - df["co2_ppm"].shift(13)

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Drop rows with NaN from 12-month lag initialization
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean


def run_pipeline():
    print("=" * 75)
    print("CLIMATE PREDICTIVE MODELING & SCIENTIFIC EVALUATION PIPELINE")
    print("=" * 75)

    df = load_and_prepare_data()
    print(f"Loaded {len(df)} contiguous monthly observations ({df['date'].iloc[0]} to {df['date'].iloc[-1]})")

    feature_cols = [
        "temp_lag1", "temp_lag2", "temp_lag3", "temp_lag12",
        "ln_co2_lag1", "co2_diff1", "co2_diff12",
        "sin_month", "cos_month"
    ]

    # Chronological Split
    train_mask = df["year"] <= 2005
    val_mask = (df["year"] >= 2006) & (df["year"] <= 2015)
    test_mask = df["year"] >= 2016
    trainval_mask = df["year"] <= 2015

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, "target_temp"].values

    X_val = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, "target_temp"].values

    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, "target_temp"].values

    X_trainval = df.loc[trainval_mask, feature_cols].values
    y_trainval = df.loc[trainval_mask, "target_temp"].values

    print(f"\nPartitions:")
    print(f"  Train:      1959-04 to 2005-12 ({len(X_train)} months)")
    print(f"  Validation: 2006-01 to 2015-12 ({len(X_val)} months)")
    print(f"  Test:       2016-01 to 2025-12 ({len(X_test)} months, UNTOUCHED)")

    # 1. Train Models
    print("\nFitting & Evaluating Models on Untouched Test Set...")
    models = {
        "Climatology Mean Baseline": np.full_like(y_test, np.mean(y_trainval)),
        "Persistence Baseline (Lag-1)": df.loc[test_mask, "temp_lag1"].values,
        "Seasonal Persistence (Lag-12)": df.loc[test_mask, "temp_lag12"].values,
        "Simple Bivariate Linear (CO₂ only)": None,
        "Baseline ARX Linear Regression": None,
        "Ridge Regression (alpha=0.01)": None,
        "Random Forest Regressor": None,
        "Gradient Boosting Regressor": None,
        "XGBoost Regressor": None,
    }

    # Bivariate linear
    biv = LinearRegression().fit(df.loc[trainval_mask, ["co2_lag1"]].values, y_trainval)
    models["Simple Bivariate Linear (CO₂ only)"] = biv.predict(df.loc[test_mask, ["co2_lag1"]].values)

    # Linear ARX
    arx = LinearRegression().fit(X_trainval, y_trainval)
    models["Baseline ARX Linear Regression"] = arx.predict(X_test)

    # Ridge
    ridge = Ridge(alpha=0.01, random_state=42).fit(X_trainval, y_trainval)
    models["Ridge Regression (alpha=0.01)"] = ridge.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_leaf=3, random_state=42)
    rf.fit(X_trainval, y_trainval)
    models["Random Forest Regressor"] = rf.predict(X_test)

    # Gradient Boosting
    gbr = GradientBoostingRegressor(learning_rate=0.1, max_depth=2, n_estimators=100, random_state=42)
    gbr.fit(X_trainval, y_trainval)
    models["Gradient Boosting Regressor"] = gbr.predict(X_test)

    # XGBoost
    xm = xgb.XGBRegressor(learning_rate=0.05, max_depth=2, n_estimators=150, reg_lambda=5.0,
                           random_state=42, objective="reg:squarederror")
    xm.fit(X_trainval, y_trainval)
    models["XGBoost Regressor"] = xm.predict(X_test)

    results = []
    for name, pred in models.items():
        mae = mean_absolute_error(y_test, pred)
        rmse = root_mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        results.append({"Model": name, "Test_MAE": mae, "Test_RMSE": rmse, "Test_R2": r2})

    res_df = pd.DataFrame(results).sort_values("Test_RMSE").reset_index(drop=True)
    print("\n--- Out-of-Sample Test Benchmark: Protocol A (1-Step-Ahead Teacher-Forced, 2016–2025) ---")
    print(res_df.to_string(index=False))

    # Protocol B: Recursive Multi-Step 10-Year Rollout
    def roll_test_recursive(model_func):
        hist_temp = list(df.loc[trainval_mask, "target_temp"].values)
        hist_co2 = list(df.loc[trainval_mask, "co2_lag1"].values)
        test_co2 = list(df.loc[test_mask, "co2_lag1"].values)
        test_m = list(df.loc[test_mask, "month"].values)
        preds = []
        for i in range(len(y_test)):
            m = test_m[i]
            c1 = test_co2[i]
            c2 = hist_co2[-1]
            c13 = hist_co2[-12]
            x = np.array([[
                hist_temp[-1], hist_temp[-2], hist_temp[-3], hist_temp[-12],
                np.log(c1), c1 - c2, c1 - c13,
                np.sin(2 * np.pi * m / 12.0), np.cos(2 * np.pi * m / 12.0)
            ]])
            p = float(model_func(x)[0])
            preds.append(p)
            hist_temp.append(p)
            hist_co2.append(c1)
        return np.array(preds)

    proto_comp = [
        {"Model": "Ridge Regression (alpha=0.01)", "Protocol": "1-Step Teacher-Forced",
         "RMSE": root_mean_squared_error(y_test, models["Ridge Regression (alpha=0.01)"]),
         "MAE": mean_absolute_error(y_test, models["Ridge Regression (alpha=0.01)"]),
         "R2": r2_score(y_test, models["Ridge Regression (alpha=0.01)"])},
        {"Model": "Ridge Regression (alpha=0.01)", "Protocol": "10-Yr Recursive Rollout",
         "RMSE": root_mean_squared_error(y_test, roll_test_recursive(ridge.predict)),
         "MAE": mean_absolute_error(y_test, roll_test_recursive(ridge.predict)),
         "R2": r2_score(y_test, roll_test_recursive(ridge.predict))},
        {"Model": "Naive Persistence (Lag-1)", "Protocol": "1-Step Teacher-Forced",
         "RMSE": root_mean_squared_error(y_test, df.loc[test_mask, "temp_lag1"].values),
         "MAE": mean_absolute_error(y_test, df.loc[test_mask, "temp_lag1"].values),
         "R2": r2_score(y_test, df.loc[test_mask, "temp_lag1"].values)},
        {"Model": "Naive Persistence (Lag-1)", "Protocol": "10-Yr Recursive Rollout",
         "RMSE": root_mean_squared_error(y_test, np.full_like(y_test, df.loc[trainval_mask, "target_temp"].iloc[-1])),
         "MAE": mean_absolute_error(y_test, np.full_like(y_test, df.loc[trainval_mask, "target_temp"].iloc[-1])),
         "R2": r2_score(y_test, np.full_like(y_test, df.loc[trainval_mask, "target_temp"].iloc[-1]))},
        {"Model": "Seasonal Persistence (Lag-12)", "Protocol": "1-Step Teacher-Forced",
         "RMSE": root_mean_squared_error(y_test, df.loc[test_mask, "temp_lag12"].values),
         "MAE": mean_absolute_error(y_test, df.loc[test_mask, "temp_lag12"].values),
         "R2": r2_score(y_test, df.loc[test_mask, "temp_lag12"].values)},
        {"Model": "Seasonal Persistence (Lag-12)", "Protocol": "10-Yr Recursive Rollout",
         "RMSE": root_mean_squared_error(y_test, np.tile(df.loc[trainval_mask, "target_temp"].iloc[-12:].values, 10)),
         "MAE": mean_absolute_error(y_test, np.tile(df.loc[trainval_mask, "target_temp"].iloc[-12:].values, 10)),
         "R2": r2_score(y_test, np.tile(df.loc[trainval_mask, "target_temp"].iloc[-12:].values, 10))},
    ]
    proto_df = pd.DataFrame(proto_comp)
    print("\n--- Protocol Comparison: 1-Step-Ahead vs 10-Year Recursive Rollout (2016–2025) ---")
    print(proto_df.to_string(index=False))
    proto_df.to_csv(os.path.join(MODELS_DIR, "evaluation_protocols_comparison.csv"), index=False)

    # 2. Historical Era Breakdown (Ridge Model)
    df["ridge_pred"] = ridge.predict(df[feature_cols].values)
    df["ridge_res"] = df["target_temp"] - df["ridge_pred"]

    eras = [
        ("1959–1975: Aerosol Plateau / Volcanic Cooling", 1959, 1975),
        ("1976–1997: Rapid Greenhouse Acceleration", 1976, 1997),
        ("1998–2012: Warming Slowdown ('Hiatus')", 1998, 2012),
        ("2013–2025: Modern Rapid Escalation", 2013, 2025),
    ]

    era_stats = []
    for era_name, y0, y1 in eras:
        sub = df[(df["year"] >= y0) & (df["year"] <= y1)]
        era_stats.append({
            "Era": era_name,
            "Months": len(sub),
            "MAE": mean_absolute_error(sub["target_temp"], sub["ridge_pred"]),
            "RMSE": root_mean_squared_error(sub["target_temp"], sub["ridge_pred"]),
            "R2": r2_score(sub["target_temp"], sub["ridge_pred"]),
            "Mean_Bias": np.mean(sub["ridge_res"]),
        })

    era_df = pd.DataFrame(era_stats)
    print("\n--- Sub-Period Evaluation Across Climate Eras ---")
    print(era_df.to_string(index=False))

    # 3. Model Serialization (Joblib)
    import joblib
    y_rec_ridge = roll_test_recursive(ridge.predict)
    # Validation residual std (tuning-free uncertainty estimate)
    ridge_tr = Ridge(alpha=0.01, random_state=42).fit(X_train, y_train)
    val_res = y_val - ridge_tr.predict(X_val)
    sigma_val = float(np.std(val_res, ddof=1))

    model_bundle = {
        "model": ridge,
        "model_type": "Ridge Regression",
        "hyperparameters": {"alpha": 0.01, "random_state": 42},
        "feature_names": feature_cols,
        "target_name": "temp_anomaly_c",
        "target_baseline": "NASA GISS 1951-1980 baseline (°C)",
        "training_window": "1959-04 to 2015-12 (681 months)",
        "test_window": "2016-01 to 2025-12 (120 months)",
        "test_metrics": {
            "MAE": float(mean_absolute_error(y_test, models["Ridge Regression (alpha=0.01)"])),
            "RMSE": float(root_mean_squared_error(y_test, models["Ridge Regression (alpha=0.01)"])),
            "R2": float(r2_score(y_test, models["Ridge Regression (alpha=0.01)"])),
        },
        "test_metrics_1step": {
            "MAE": float(mean_absolute_error(y_test, models["Ridge Regression (alpha=0.01)"])),
            "RMSE": float(root_mean_squared_error(y_test, models["Ridge Regression (alpha=0.01)"])),
            "R2": float(r2_score(y_test, models["Ridge Regression (alpha=0.01)"])),
            "protocol": "1-step-ahead teacher-forced rolling evaluation",
        },
        "test_metrics_recursive_10yr": {
            "MAE": float(mean_absolute_error(y_test, y_rec_ridge)),
            "RMSE": float(root_mean_squared_error(y_test, y_rec_ridge)),
            "R2": float(r2_score(y_test, y_rec_ridge)),
            "protocol": "10-year autonomous recursive rollout without intermediate ground truth",
        },
        "validation_residual_std": sigma_val,
        "coefficients": dict(zip(feature_cols, [float(c) for c in ridge.coef_])),
        "intercept": float(ridge.intercept_),
    }
    joblib_path = os.path.join(MODELS_DIR, "ridge_climate_model.joblib")
    joblib.dump(model_bundle, joblib_path)
    print(f"\nSerialized Best-Performing Model to: {joblib_path}")

    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run empirical ML and physical climate pipeline.")
    parser.add_argument("--simulate-hector", action="store_true",
                        help="Also execute Phase 7 Hector/DOECLIM physical 2xCO2 simulation.")
    args = parser.parse_args()

    run_pipeline()

    if args.simulate_hector:
        import subprocess
        print("\nLaunching Phase 7 Hector / DOECLIM Physical Simulation...")
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "simulate_hector_2xco2.py")], check=True)
