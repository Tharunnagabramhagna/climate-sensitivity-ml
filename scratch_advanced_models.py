"""
Phase 5: Advanced Model Investigation
Evaluates Ridge Regression, Random Forest, Gradient Boosting, and XGBoost.
Hyperparameter tuning strictly on Train + Validation sets.
Evaluation on untouched Test period (2016–2025).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

PROJECT = r"c:\Users\bharg\OneDrive\Desktop\ml_model"
DATA_FILE = os.path.join(PROJECT, "climate_paired.csv")
MODELS_DIR = os.path.join(PROJECT, "modeling_outputs")
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE, comment="#")
df["date_dt"] = pd.to_datetime(df["date"])
df = df.sort_values("date_dt").reset_index(drop=True)

# -------------------------------------------------------------
# Input Features
# -------------------------------------------------------------
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

df_model = df.dropna().reset_index(drop=True)

# Splits
train_df = df_model[df_model["year"] <= 2005]
val_df = df_model[(df_model["year"] >= 2006) & (df_model["year"] <= 2015)]
test_df = df_model[df_model["year"] >= 2016]

feature_cols = [
    "temp_lag1", "temp_lag2", "temp_lag3", "temp_lag12",
    "ln_co2_lag1", "co2_diff1", "co2_diff12",
    "sin_month", "cos_month"
]

X_train = train_df[feature_cols].values
y_train = train_df["target_temp"].values

X_val = val_df[feature_cols].values
y_val = val_df["target_temp"].values

X_test = test_df[feature_cols].values
y_test = test_df["target_temp"].values

test_dates = test_df["date_dt"]

def eval_pred(y_true, y_pred, name):
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

# -------------------------------------------------------------
# 1. Ridge Regression (Hyperparameter Tuning on Val)
# -------------------------------------------------------------
print("Tuning Ridge Regression...")
best_ridge_alpha = None
best_ridge_val_rmse = float("inf")
ridge_candidates = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]

for alpha in ridge_candidates:
    m = Ridge(alpha=alpha, random_state=42)
    m.fit(X_train, y_train)
    val_pred = m.predict(X_val)
    val_rmse = root_mean_squared_error(y_val, val_pred)
    if val_rmse < best_ridge_val_rmse:
        best_ridge_val_rmse = val_rmse
        best_ridge_alpha = alpha

print(f"  Best Ridge Alpha: {best_ridge_alpha} (Val RMSE: {best_ridge_val_rmse:.5f})")
# Fit on Train + Val with best hyperparameter before final test evaluation
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

ridge_best = Ridge(alpha=best_ridge_alpha, random_state=42)
ridge_best.fit(X_trainval, y_trainval)
ridge_test_pred = ridge_best.predict(X_test)

# -------------------------------------------------------------
# 2. Random Forest Regressor (Tuning on Val)
# -------------------------------------------------------------
print("Tuning Random Forest...")
best_rf_params = None
best_rf_val_rmse = float("inf")

for n_est in [50, 100, 200]:
    for max_d in [3, 5, 8, None]:
        for min_leaf in [1, 3, 5]:
            rf = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, min_samples_leaf=min_leaf, random_state=42)
            rf.fit(X_train, y_train)
            val_rmse = root_mean_squared_error(y_val, rf.predict(X_val))
            if val_rmse < best_rf_val_rmse:
                best_rf_val_rmse = val_rmse
                best_rf_params = (n_est, max_d, min_leaf)

print(f"  Best RF Params: n_est={best_rf_params[0]}, max_depth={best_rf_params[1]}, min_leaf={best_rf_params[2]} (Val RMSE: {best_rf_val_rmse:.5f})")
rf_best = RandomForestRegressor(n_estimators=best_rf_params[0], max_depth=best_rf_params[1],
                                min_samples_leaf=best_rf_params[2], random_state=42)
rf_best.fit(X_trainval, y_trainval)
rf_test_pred = rf_best.predict(X_test)

# -------------------------------------------------------------
# 3. Gradient Boosting Regressor (Tuning on Val)
# -------------------------------------------------------------
print("Tuning Gradient Boosting Regressor...")
best_gbr_params = None
best_gbr_val_rmse = float("inf")

for lr in [0.01, 0.03, 0.05, 0.1]:
    for max_d in [2, 3, 4]:
        for n_est in [50, 100, 150]:
            gbr = GradientBoostingRegressor(learning_rate=lr, max_depth=max_d, n_estimators=n_est, random_state=42)
            gbr.fit(X_train, y_train)
            val_rmse = root_mean_squared_error(y_val, gbr.predict(X_val))
            if val_rmse < best_gbr_val_rmse:
                best_gbr_val_rmse = val_rmse
                best_gbr_params = (lr, max_d, n_est)

print(f"  Best GBR Params: lr={best_gbr_params[0]}, max_depth={best_gbr_params[1]}, n_est={best_gbr_params[2]} (Val RMSE: {best_gbr_val_rmse:.5f})")
gbr_best = GradientBoostingRegressor(learning_rate=best_gbr_params[0], max_depth=best_gbr_params[1],
                                    n_estimators=best_gbr_params[2], random_state=42)
gbr_best.fit(X_trainval, y_trainval)
gbr_test_pred = gbr_best.predict(X_test)

# -------------------------------------------------------------
# 4. XGBoost Regressor (Tuning on Val)
# -------------------------------------------------------------
print("Tuning XGBoost Regressor...")
best_xgb_params = None
best_xgb_val_rmse = float("inf")

for lr in [0.01, 0.03, 0.05, 0.1]:
    for max_d in [2, 3, 4]:
        for n_est in [50, 100, 150]:
            for reg_l in [0.1, 1.0, 5.0]:
                xm = xgb.XGBRegressor(learning_rate=lr, max_depth=max_d, n_estimators=n_est,
                                      reg_lambda=reg_l, random_state=42, objective="reg:squarederror")
                xm.fit(X_train, y_train)
                val_rmse = root_mean_squared_error(y_val, xm.predict(X_val))
                if val_rmse < best_xgb_val_rmse:
                    best_xgb_val_rmse = val_rmse
                    best_xgb_params = (lr, max_d, n_est, reg_l)

print(f"  Best XGB Params: lr={best_xgb_params[0]}, max_depth={best_xgb_params[1]}, n_est={best_xgb_params[2]}, lambda={best_xgb_params[3]} (Val RMSE: {best_xgb_val_rmse:.5f})")
xgb_best = xgb.XGBRegressor(learning_rate=best_xgb_params[0], max_depth=best_xgb_params[1],
                           n_estimators=best_xgb_params[2], reg_lambda=best_xgb_params[3],
                           random_state=42, objective="reg:squarederror")
xgb_best.fit(X_trainval, y_trainval)
xgb_test_pred = xgb_best.predict(X_test)

# Also refit Baseline ARX on Train+Val for fair comparison
from sklearn.linear_model import LinearRegression
lr_baseline = LinearRegression()
lr_baseline.fit(X_trainval, y_trainval)
baseline_test_pred = lr_baseline.predict(X_test)

# Persistence on test
pers_test_pred = test_df["temp_lag1"].values

# -------------------------------------------------------------
# Full Comparative Performance Table on Untouched Test Set
# -------------------------------------------------------------
comparison_results = [
    eval_pred(y_test, pers_test_pred, "Persistence Baseline (Lag-1)"),
    eval_pred(y_test, baseline_test_pred, "Baseline ARX Linear Regression"),
    eval_pred(y_test, ridge_test_pred, f"Ridge Regression (alpha={best_ridge_alpha})"),
    eval_pred(y_test, rf_test_pred, "Random Forest Regressor"),
    eval_pred(y_test, gbr_test_pred, "Gradient Boosting Regressor"),
    eval_pred(y_test, xgb_test_pred, "XGBoost Regressor"),
]

comp_df = pd.DataFrame(comparison_results)
print("\n" + "=" * 70)
print("PHASE 5: UNTOUCHED TEST SET EVALUATION (2016–2025)")
print("=" * 70)
print(comp_df.to_string(index=False))

comp_df.to_csv(os.path.join(MODELS_DIR, "advanced_models_test_comparison.csv"), index=False)

# -------------------------------------------------------------
# Feature Importance for Tree Models & Coefficients for Linear Models
# -------------------------------------------------------------
print("\n--- Model Feature Importance / Weights ---")
feat_summary = pd.DataFrame({"Feature": feature_cols})
feat_summary["Ridge_Coef"] = ridge_best.coef_
feat_summary["RF_Importance"] = rf_best.feature_importances_
feat_summary["GBR_Importance"] = gbr_best.feature_importances_
feat_summary["XGB_Importance"] = xgb_best.feature_importances_
print(feat_summary.to_string(index=False))
feat_summary.to_csv(os.path.join(MODELS_DIR, "feature_importance_summary.csv"), index=False)

# -------------------------------------------------------------
# Comparative Plots
# -------------------------------------------------------------
# 1. Test Predictions Comparison
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
ax.plot(test_dates, y_test, color="black", linewidth=2.2, label="Actual Observed Anomaly (GISTEMP)")
ax.plot(test_dates, ridge_test_pred, color="#1f77b4", linewidth=1.8, label="Ridge Regression (alpha=5)")
ax.plot(test_dates, baseline_test_pred, color="#2ca02c", linestyle="--", linewidth=1.6, label="ARX Linear Baseline")
ax.plot(test_dates, xgb_test_pred, color="#9467bd", linestyle="-.", linewidth=1.6, label="XGBoost Regressor")
ax.plot(test_dates, rf_test_pred, color="#d62728", linestyle=":", linewidth=1.6, label="Random Forest")

ax.set_title("Advanced Models vs. Baselines: Untouched Test Period Forecasts (2016–2025)", fontsize=13, fontweight="bold")
ax.set_xlabel("Year (Monthly Evaluation)")
ax.set_ylabel("Global Temperature Anomaly (°C)")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper left", framealpha=0.9)

note_str = ("Scientific Observation:\n"
            "• Ridge and Linear ARX capture the unprecedented record heat (2023–24).\n"
            "• Tree ensembles (RF, GBR) suffer from bounded extrapolation on non-stationary data.")
ax.text(0.48, 0.12, note_str, transform=ax.transAxes, fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="#fff9db", edgecolor="#e6b800", alpha=0.9))

fig.text(0.99, 0.01, "Chronological Evaluation: Train+Val (1959–2015) -> Test (2016–2025)",
         ha="right", fontsize=8, color="#555555", style="italic")
p_comp_path = os.path.join(MODELS_DIR, "advanced_models_test_forecasts.png")
fig.savefig(p_comp_path)
plt.close(fig)
print(f"\nGenerated: {p_comp_path}")

# 2. Residual Comparison Plot
fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150, sharex=True, sharey=True)

models_eval = [
    ("Ridge Regression", y_test - ridge_test_pred, "#1f77b4", axes[0, 0]),
    ("Baseline ARX Linear", y_test - baseline_test_pred, "#2ca02c", axes[0, 1]),
    ("Random Forest", y_test - rf_test_pred, "#d62728", axes[1, 0]),
    ("XGBoost Regressor", y_test - xgb_test_pred, "#9467bd", axes[1, 1]),
]

for name, res, col, ax_sub in models_eval:
    ax_sub.plot(test_dates, res, color=col, linewidth=1.2)
    ax_sub.axhline(0, color="black", linestyle="--", linewidth=0.8)
    mae_val = mean_absolute_error(y_test, y_test - res)
    rmse_val = root_mean_squared_error(y_test, y_test - res)
    ax_sub.set_title(f"{name} (MAE: {mae_val:.3f}, RMSE: {rmse_val:.3f})", fontsize=11, fontweight="bold")
    ax_sub.grid(True, linestyle="--", alpha=0.6)
    ax_sub.set_ylabel("Residual (°C)")

axes[1, 0].set_xlabel("Year (2016–2025)")
axes[1, 1].set_xlabel("Year (2016–2025)")
fig.suptitle("Comparison of Residual Time Series Across Models on Untouched Test Set", fontsize=13, fontweight="bold")
p_res_comp_path = os.path.join(MODELS_DIR, "advanced_models_residuals.png")
fig.savefig(p_res_comp_path)
plt.close(fig)
print(f"Generated: {p_res_comp_path}")

print("Phase 5 script execution complete.")

