"""
Phase 3: Exploratory Data Analysis (EDA) on Paired Climate Dataset
Generates high-resolution figures and statistical tables.
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT = r"c:\Users\bharg\OneDrive\Desktop\ml_model"
DATA_FILE = os.path.join(PROJECT, "climate_paired.csv")
EDA_DIR = os.path.join(PROJECT, "eda_outputs")
os.makedirs(EDA_DIR, exist_ok=True)

print("Loading paired dataset...")
df = pd.read_csv(DATA_FILE, comment="#")
df["date_dt"] = pd.to_datetime(df["date"])
df = df.sort_values("date_dt").reset_index(drop=True)
print(f"Loaded {len(df)} rows from {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

# 1. Basic descriptive statistics
co2 = df["co2_ppm"].values
temp = df["temp_anomaly_c"].values
dates = df["date_dt"].values

stats = {
    "Variable": ["CO2 Concentration (ppm)", "Temperature Anomaly (deg C)"],
    "Count": [len(co2), len(temp)],
    "Mean": [np.mean(co2), np.mean(temp)],
    "Std Dev": [np.std(co2, ddof=1), np.std(temp, ddof=1)],
    "Min": [np.min(co2), np.min(temp)],
    "25% (Q1)": [np.percentile(co2, 25), np.percentile(temp, 25)],
    "Median": [np.median(co2), np.median(temp)],
    "75% (Q3)": [np.percentile(co2, 75), np.percentile(temp, 75)],
    "Max": [np.max(co2), np.max(temp)],
    "IQR": [np.percentile(co2, 75) - np.percentile(co2, 25), np.percentile(temp, 75) - np.percentile(temp, 25)],
}
stats_df = pd.DataFrame(stats)
print("\n--- Summary Statistics ---")
print(stats_df.to_string(index=False))

# 2. Correlations
# Log CO2
log_co2 = np.log(co2)
# First differences (month over month)
d_co2 = np.diff(co2)
d_temp = np.diff(temp)
# 12-month differences (year over year)
d12_co2 = co2[12:] - co2[:-12]
d12_temp = temp[12:] - temp[:-12]

# Linear detrending
t = np.arange(len(co2))
p_co2 = np.polyfit(t, co2, 1)
co2_detrended = co2 - np.polyval(p_co2, t)

p_temp = np.polyfit(t, temp, 1)
temp_detrended = temp - np.polyval(p_temp, t)

# Quadratic detrending
p2_co2 = np.polyfit(t, co2, 2)
co2_quad_detrended = co2 - np.polyval(p2_co2, t)

p2_temp = np.polyfit(t, temp, 2)
temp_quad_detrended = temp - np.polyval(p2_temp, t)

def calc_pearson(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    return np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))

def calc_spearman(x, y):
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    return calc_pearson(rx, ry)

corr_results = [
    ("Raw Level: CO2 vs Temp Anomaly", calc_pearson(co2, temp), calc_spearman(co2, temp)),
    ("Radiative Forcing Proxy: ln(CO2) vs Temp Anomaly", calc_pearson(log_co2, temp), calc_spearman(log_co2, temp)),
    ("1-Month Difference: Delta CO2 vs Delta Temp", calc_pearson(d_co2, d_temp), calc_spearman(d_co2, d_temp)),
    ("12-Month Difference (YoY): Delta_12 CO2 vs Delta_12 Temp", calc_pearson(d12_co2, d12_temp), calc_spearman(d12_co2, d12_temp)),
    ("Linearly Detrended: Residual CO2 vs Residual Temp", calc_pearson(co2_detrended, temp_detrended), calc_spearman(co2_detrended, temp_detrended)),
    ("Quadratic Detrended: Residual CO2 vs Residual Temp", calc_pearson(co2_quad_detrended, temp_quad_detrended), calc_spearman(co2_quad_detrended, temp_quad_detrended)),
]

print("\n--- Correlation Analysis & Robustness Checks ---")
print(f"{'Specification':<55} {'Pearson r':>10} {'Spearman rho':>12}")
print("-" * 80)
for spec, r_p, r_s in corr_results:
    print(f"{spec:<55} {r_p:>10.4f} {r_s:>12.4f}")

# 3. Autocorrelation (ACF) up to lag 36
def calc_acf(series, max_lag=36):
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    c0 = np.sum((series - mean)**2) / n
    acf_vals = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            c_k = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / n
            acf_vals.append(c_k / c0)
    return np.array(acf_vals)

acf_co2 = calc_acf(co2, 36)
acf_temp = calc_acf(temp, 36)

print("\n--- Autocorrelation (ACF) Key Lags ---")
for lag in [1, 2, 3, 6, 12, 24, 36]:
    print(f"Lag {lag:2d}: CO2 ACF = {acf_co2[lag]:.4f}, Temp Anomaly ACF = {acf_temp[lag]:.4f}")

# 4. Outliers and extreme events
temp_mean, temp_std = np.mean(temp), np.std(temp, ddof=1)
temp_z = (temp - temp_mean) / temp_std
top_warm = df.iloc[np.argsort(temp)[-10:]][["date", "co2_ppm", "temp_anomaly_c"]].iloc[::-1]
top_cold = df.iloc[np.argsort(temp)[:10]][["date", "co2_ppm", "temp_anomaly_c"]]

print("\n--- Top 5 Warmest Months in Paired History ---")
print(top_warm.head(5).to_string(index=False))

print("\n--- Top 5 Coldest Months in Paired History ---")
print(top_cold.head(5).to_string(index=False))

# Check for largest residuals from quadratic trend
temp_res_z = (temp_quad_detrended - np.mean(temp_quad_detrended)) / np.std(temp_quad_detrended, ddof=1)
large_pos_spikes = df[temp_res_z > 2.5][["date", "co2_ppm", "temp_anomaly_c"]]
print(f"\nUnusual positive temperature excursions (>2.5 sigma from trend): {len(large_pos_spikes)} months")
print(large_pos_spikes.to_string(index=False))

# Save metrics table
summary_table_file = os.path.join(EDA_DIR, "eda_metrics_summary.csv")
pd.DataFrame(corr_results, columns=["Specification", "Pearson_r", "Spearman_rho"]).to_csv(summary_table_file, index=False)
print(f"\nSaved summary metrics to {summary_table_file}")

