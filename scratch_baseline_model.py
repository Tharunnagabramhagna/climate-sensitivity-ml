"""
Phase 4: ML Problem Formulation and Baseline Modelling
Evaluates persistence, climatological mean, simple linear, and ARX linear regression.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

PROJECT = r"c:\Users\bharg\OneDrive\Desktop\ml_model"
DATA_FILE = os.path.join(PROJECT, "climate_paired.csv")
MODELS_DIR = os.path.join(PROJECT, "modeling_outputs")
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE, comment="#")
df["date_dt"] = pd.to_datetime(df["date"])
df = df.sort_values("date_dt").reset_index(drop=True)

# -------------------------------------------------------------
# Feature Engineering (Strictly past information, no leakage)
# -------------------------------------------------------------
# Target: Temperature Anomaly at month t
df["target_temp"] = df["temp_anomaly_c"]

# Lags
df["temp_lag1"] = df["temp_anomaly_c"].shift(1)
df["temp_lag2"] = df["temp_anomaly_c"].shift(2)
df["temp_lag12"] = df["temp_anomaly_c"].shift(12)

# CO2 features (known at t-1)
df["co2_lag1"] = df["co2_ppm"].shift(1)
df["ln_co2_lag1"] = np.log(df["co2_lag1"])
df["co2_diff1"] = df["co2_ppm"].shift(1) - df["co2_ppm"].shift(2)
df["co2_diff12"] = df["co2_ppm"].shift(1) - df["co2_ppm"].shift(13)

# Seasonal harmonics (known calendar feature)
df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

# Drop initial rows with NaN from 12-month lag
df_model = df.dropna().reset_index(drop=True)
print(f"Dataset after lagging (13 rows dropped): {len(df_model)} rows from {df_model['date'].iloc[0]} to {df_model['date'].iloc[-1]}")

# -------------------------------------------------------------
# Chronological Split
# Train: 1959-04 to 2005-12
# Val:   2006-01 to 2015-12
# Test:  2016-01 to 2025-12
# -------------------------------------------------------------
train_mask = df_model["year"] <= 2005
val_mask = (df_model["year"] >= 2006) & (df_model["year"] <= 2015)
test_mask = df_model["year"] >= 2016

train_df = df_model[train_mask]
val_df = df_model[val_mask]
test_df = df_model[test_mask]

print(f"Train period: {train_df['date'].iloc[0]} to {train_df['date'].iloc[-1]} ({len(train_df)} months)")
print(f"Val period:   {val_df['date'].iloc[0]} to {val_df['date'].iloc[-1]} ({len(val_df)} months)")
print(f"Test period:  {test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]} ({len(test_df)} months)")

y_train = train_df["target_temp"].values
y_val = val_df["target_temp"].values
y_test = test_df["target_temp"].values

# -------------------------------------------------------------
# Baseline 1: Historical Mean (Climatology)
# -------------------------------------------------------------
mean_pred_val = np.full_like(y_val, np.mean(y_train))
mean_pred_test = np.full_like(y_test, np.mean(y_train))

# -------------------------------------------------------------
# Baseline 2: Persistence (T_t = T_{t-1})
# -------------------------------------------------------------
pers_pred_val = val_df["temp_lag1"].values
pers_pred_test = test_df["temp_lag1"].values

# -------------------------------------------------------------
# Baseline 3: Simple Bivariate Linear Regression (CO2 only)
# -------------------------------------------------------------
X_biv_train = train_df[["co2_lag1"]].values
X_biv_val = val_df[["co2_lag1"]].values
X_biv_test = test_df[["co2_lag1"]].values

lr_biv = LinearRegression()
lr_biv.fit(X_biv_train, y_train)
biv_pred_val = lr_biv.predict(X_biv_val)
biv_pred_test = lr_biv.predict(X_biv_test)

# -------------------------------------------------------------
# Baseline 4: Multivariate Dynamic Linear Regression (ARX)
# Features: temp_lag1, temp_lag2, temp_lag12, ln_co2_lag1, co2_diff1, co2_diff12, sin_month, cos_month
# -------------------------------------------------------------
feature_cols = [
    "temp_lag1", "temp_lag2", "temp_lag12",
    "ln_co2_lag1", "co2_diff1", "co2_diff12",
    "sin_month", "cos_month"
]
X_arx_train = train_df[feature_cols].values
X_arx_val = val_df[feature_cols].values
X_arx_test = test_df[feature_cols].values

lr_arx = LinearRegression()
lr_arx.fit(X_arx_train, y_train)
arx_pred_val = lr_arx.predict(X_arx_val)
arx_pred_test = lr_arx.predict(X_arx_test)

# -------------------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------------------
def get_metrics(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

val_metrics = [
    get_metrics(y_val, mean_pred_val, "Climatology Mean Baseline"),
    get_metrics(y_val, pers_pred_val, "Persistence Baseline (Lag-1)"),
    get_metrics(y_val, biv_pred_val, "Bivariate Linear (CO₂ only)"),
    get_metrics(y_val, arx_pred_val, "Dynamic ARX Linear Regression"),
]

test_metrics = [
    get_metrics(y_test, mean_pred_test, "Climatology Mean Baseline"),
    get_metrics(y_test, pers_pred_test, "Persistence Baseline (Lag-1)"),
    get_metrics(y_test, biv_pred_test, "Bivariate Linear (CO₂ only)"),
    get_metrics(y_test, arx_pred_test, "Dynamic ARX Linear Regression"),
]

val_metrics_df = pd.DataFrame(val_metrics)
test_metrics_df = pd.DataFrame(test_metrics)

print("\n--- Validation Period Performance (2006–2015) ---")
print(val_metrics_df.to_string(index=False))

print("\n--- Test Period Performance (2016–2025) ---")
print(test_metrics_df.to_string(index=False))

# Coefficients of ARX Model
print("\n--- ARX Linear Regression Coefficients ---")
for col, coef in zip(feature_cols, lr_arx.coef_):
    print(f"  {col:<15}: {coef:+.6f}")
print(f"  {'Intercept':<15}: {lr_arx.intercept_:+.6f}")

# -------------------------------------------------------------
# Diagnostics & Plots
# -------------------------------------------------------------
# 1. Actual vs Predicted on Test Set
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
test_dates = test_df["date_dt"]
ax.plot(test_dates, y_test, color="black", linewidth=2.0, label="Actual Observed Anomaly (GISTEMP)")
ax.plot(test_dates, pers_pred_test, color="#ff7f0e", linestyle=":", linewidth=1.5, label="Persistence Baseline (Lag-1)")
ax.plot(test_dates, biv_pred_test, color="#1f77b4", linestyle="--", linewidth=1.5, label="Bivariate CO₂ Linear")
ax.plot(test_dates, arx_pred_test, color="#d62728", linewidth=2.0, label="Dynamic ARX Linear Regression")

ax.set_title("Phase 4 Baseline Models: Out-of-Sample Test Forecasts (2016–2025)", fontsize=13, fontweight="bold")
ax.set_xlabel("Year (Monthly Test Evaluation)")
ax.set_ylabel("Global Temperature Anomaly (°C)")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper left")
fig.text(0.99, 0.01, "Chronological Test Evaluation (Zero Future Information Leakage)",
         ha="right", fontsize=8, color="#555555", style="italic")
p_test_path = os.path.join(MODELS_DIR, "baseline_test_predictions.png")
fig.savefig(p_test_path)
plt.close(fig)
print(f"\nGenerated: {p_test_path}")

# 2. Residual Analysis for ARX Linear Model
arx_res_test = y_test - arx_pred_test
fig, (ax_res, ax_acf) = plt.subplots(2, 1, figsize=(9, 6), dpi=150)

ax_res.plot(test_dates, arx_res_test, color="#d62728", linewidth=1.2)
ax_res.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax_res.set_title("Residual Series on Test Period: e_t = Actual - ARX Predicted", fontsize=11, fontweight="bold")
ax_res.set_ylabel("Residual (°C)")
ax_res.grid(True, linestyle="--", alpha=0.6)

# Residual ACF
n_res = len(arx_res_test)
c0_res = np.sum((arx_res_test - np.mean(arx_res_test))**2) / n_res
res_acf = []
for l in range(25):
    if l == 0:
        res_acf.append(1.0)
    else:
        c_l = np.sum((arx_res_test[:-l] - np.mean(arx_res_test)) * (arx_res_test[l:] - np.mean(arx_res_test))) / n_res
        res_acf.append(c_l / c0_res)

lags_res = np.arange(25)
conf_res = 1.96 / np.sqrt(n_res)
ax_acf.vlines(lags_res, 0, res_acf, color="#d62728", linewidth=1.5)
ax_acf.scatter(lags_res, res_acf, color="#d62728", s=15)
ax_acf.axhline(0, color="black", linewidth=0.8)
ax_acf.axhline(conf_res, color="blue", linestyle="--", linewidth=0.9, alpha=0.7, label="95% Confidence Band")
ax_acf.axhline(-conf_res, color="blue", linestyle="--", linewidth=0.9, alpha=0.7)
ax_acf.set_title("Residual Autocorrelation Function (ACF)", fontsize=11, fontweight="bold")
ax_acf.set_xlabel("Lag (Months)")
ax_acf.set_ylabel("Autocorrelation")
ax_acf.grid(True, linestyle="--", alpha=0.6)
ax_acf.legend(loc="upper right")

fig.suptitle("Diagnostic Residual Inspection: Dynamic ARX Linear Baseline", fontsize=13, fontweight="bold")
fig.text(0.99, 0.01, f"Test Residual Mean = {np.mean(arx_res_test):+.3f}°C | Std = {np.std(arx_res_test, ddof=1):.3f}°C",
         ha="right", fontsize=8, color="#555555", style="italic")
p_res_path = os.path.join(MODELS_DIR, "baseline_residual_diagnostics.png")
fig.savefig(p_res_path)
plt.close(fig)
print(f"Generated: {p_res_path}")

# Save baseline results table
val_metrics_df.to_csv(os.path.join(MODELS_DIR, "baseline_val_metrics.csv"), index=False)
test_metrics_df.to_csv(os.path.join(MODELS_DIR, "baseline_test_metrics.csv"), index=False)
print("Saved baseline validation and test metrics.")

