"""
src/evaluate_2026.py - 2026 Retrospective Conditional Evaluation
=============================================================================
Retrospective conditional evaluation using observed 2026 atmospheric CO2 forcing.
Compares the frozen Ridge ARX 2026 monthly projections against actual overlapping
observations.

Scientific Methodology:
  1. Actual 2026 CO2 measurements from NOAA Mauna Loa were supplied to the model.
  2. Temperature predictions were generated recursively without future temperature
     observations (Protocol B style recursive rollout for temperature lags).
  3. Evaluates error metrics (RMSE, MAE, nominal coverage) strictly out-of-sample.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
MODELING_DIR = BASE_DIR / "modeling_outputs"
PROJECTIONS_FILE = MODELING_DIR / "projected_anomalies_2026.csv"
OUTPUT_EVAL_CSV = MODELING_DIR / "forecast_2026_evaluation.csv"
OUTPUT_EVAL_PLOT = MODELING_DIR / "forecast_2026_vs_actual.png"
TEMP_RAW_FILE = BASE_DIR / "data" / "raw" / "gistemp_monthly.csv"
NASA_UPDATED_RAW_FILE = BASE_DIR / "data" / "updated" / "gistemp_glb_latest.csv"


def evaluate_2026_forecast():
    """Evaluate 2026 projections against available observed temperatures."""
    if not PROJECTIONS_FILE.exists():
        print(f"[ERROR] Projections file not found: {PROJECTIONS_FILE}")
        return False

    df_proj = pd.read_csv(PROJECTIONS_FILE)
    
    # In gistemp_monthly.csv, two distinct series are present:
    # 1. GCAG (NOAA NCEI Global Climate at a Glance, 1901-2000 baseline) through 2026-07
    # 2. GISTEMP (NASA GISS GISTEMP v4, 1951-1980 baseline) through 2025-12
    # Note: Differences between GCAG and GISTEMP are due to different agencies, methodologies,
    # and baseline periods (1901-2000 vs 1951-1980), NOT historical revisions within NASA GISTEMP.
    df_temp_raw = pd.read_csv(TEMP_RAW_FILE)
    df_gcag_2026 = df_temp_raw[(df_temp_raw["Source"] == "GCAG") & (df_temp_raw["Year"].str.startswith("2026"))].copy()
    df_gcag_2026 = df_gcag_2026.rename(columns={"Year": "Date", "Mean": "Observed_Temp_Anomaly_GCAG"})

    # Check for downloaded NASA GISTEMP v4 file on disk
    gistemp_obs = {}
    if NASA_UPDATED_RAW_FILE.exists():
        try:
            from src.data_updater import parse_gistemp_raw
            df_nasa = parse_gistemp_raw(NASA_UPDATED_RAW_FILE)
            for _, r in df_nasa.iterrows():
                gistemp_obs[str(r["date"])] = float(r["temp_anomaly_c"])
            print(f"[OK] Loaded {len(gistemp_obs)} verified NASA GISTEMP observations from {NASA_UPDATED_RAW_FILE.name}")
        except Exception as err:
            print(f"[WARNING] Could not parse {NASA_UPDATED_RAW_FILE.name}: {err}")
    else:
        print(f"[STATUS: NOT VERIFIED] Official NASA GISTEMP file not found on disk at:")
        print(f"                       {NASA_UPDATED_RAW_FILE}")
        print("                       2026 NASA ground-truth comparison remains unverified until")
        print("                       the user manually downloads GLB.Ts+dSST.csv to data/updated/.")

    eval_rows = []
    for _, row in df_proj.iterrows():
        dt = row["Date"]
        pred = row["Forecast_Temp_Anomaly_C"]
        ci_low = row["CI_Lower_95"]
        ci_high = row["CI_Upper_95"]
        co2 = row["NOAA_Measured_CO2_ppm"]
        
        # GCAG obs from local file
        gcag_match = df_gcag_2026[df_gcag_2026["Date"] == dt]
        gcag_val = float(gcag_match["Observed_Temp_Anomaly_GCAG"].iloc[0]) if len(gcag_match) > 0 else np.nan
        
        # NASA GISTEMP obs from verified file (or NaN if missing)
        giss_val = gistemp_obs.get(dt, np.nan)
        is_verified = not np.isnan(giss_val)
        
        eval_rows.append({
            "Date": dt,
            "NOAA_Measured_CO2_ppm": co2,
            "Forecast_Temp_Anomaly_C": pred,
            "CI_Lower_95": ci_low,
            "CI_Upper_95": ci_high,
            "Observed_GISTEMP_C": giss_val,
            "Observed_GCAG_C": gcag_val,
            "Error_vs_GISTEMP": round(pred - giss_val, 4) if is_verified else np.nan,
            "Inside_95_CI": bool(ci_low <= giss_val <= ci_high) if is_verified else False,
            "Evaluation_Status": "Verified NASA Observation on Disk" if is_verified else "NOT VERIFIED: Raw NASA file missing from data/updated/"
        })

    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(OUTPUT_EVAL_CSV, index=False)
    print(f"[OK] Saved 2026 forecast evaluation to {OUTPUT_EVAL_CSV.relative_to(BASE_DIR)}")

    # Calculate metrics for months with NASA GISTEMP observations
    valid_eval = df_eval.dropna(subset=["Observed_GISTEMP_C"])
    if len(valid_eval) > 0:
        rmse_2026 = np.sqrt(np.mean((valid_eval["Forecast_Temp_Anomaly_C"] - valid_eval["Observed_GISTEMP_C"])**2))
        mae_2026 = np.mean(np.abs(valid_eval["Forecast_Temp_Anomaly_C"] - valid_eval["Observed_GISTEMP_C"]))
        ci_coverage = (valid_eval["Inside_95_CI"].sum() / len(valid_eval)) * 100.0
        print(f"2026 Out-of-Sample Performance Summary ({len(valid_eval)} evaluated months):")
        print(f"  • RMSE: {rmse_2026:.4f} °C")
        print(f"  • MAE:  {mae_2026:.4f} °C")
        print(f"  • 95% CI Empirical Coverage: {ci_coverage:.1f}%")

    # Generate comparison visualization
    plt.figure(figsize=(10, 5), dpi=300)
    dates = df_eval["Date"].values
    x_indices = np.arange(len(dates))
    
    plt.plot(x_indices, df_eval["Forecast_Temp_Anomaly_C"], "b-o", label="Ridge ARX Forecast (Conditional)", linewidth=2)
    plt.fill_between(x_indices, df_eval["CI_Lower_95"], df_eval["CI_Upper_95"], color="blue", alpha=0.15, label="95% Prediction Interval (±1.96σ(h))")
    
    if not valid_eval.empty:
        plt.plot(x_indices[:len(valid_eval)], valid_eval["Observed_GISTEMP_C"], "r--s", label="Observed NASA GISTEMP v4", linewidth=2)
        plt.plot(x_indices[:len(valid_eval)], valid_eval["Observed_GCAG_C"], "g:^", label="Observed NOAA GCAG", linewidth=1.5)

    plt.xticks(x_indices, dates, rotation=30, ha="right")
    plt.xlabel("Calendar Month (2026)", fontsize=11, fontweight="bold")
    plt.ylabel("Global Temperature Anomaly (°C)", fontsize=11, fontweight="bold")
    plt.title("2026 Retrospective Conditional Evaluation: Ridge ARX Projection vs. Actual Observations", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_EVAL_PLOT)
    plt.close()
    print(f"[OK] Saved 2026 evaluation figure to {OUTPUT_EVAL_PLOT.relative_to(BASE_DIR)}")
    return True


if __name__ == "__main__":
    evaluate_2026_forecast()
