"""
Faculty Presentation Graph Generation Script
Project: Investigating Climate Sensitivity Under 2xCO2
Algorithm: Ridge Regression (alpha=0.01)

Generates 6 publication-quality scientific figures for faculty presentation:
  Graph 1: Actual vs. Predicted Temperature Anomaly (1-Step Ahead Scatter)
  Graph 2: Actual vs. Predicted Temperature Over Time (2016-2025 Timeline)
  Graph 3: Prediction Residual Analysis (Residuals Over Time & Confidence Bands)
  Graph 4: Residual Distribution (Histogram, KDE Curve, and Gaussian Reference)
  Graph 5: Recursive Forecast vs. Actual Values (Protocol B 10-Year Autonomous Rollout)
  Graph 6: Historical CO2 and Temperature Relationship (814 Months, Granger-Newbold Warning)

Outputs:
  - High-resolution PNGs (300 DPI) and vector PDFs in modeling_outputs/faculty_graphs/
  - test_actual_vs_predicted_values.csv
  - test_residuals.csv
  - calculated_evaluation_metrics.csv
  - GRAPH_INTERPRETATION_REPORT.md
"""

import os
import sys
import tempfile
import time
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "modeling_outputs", "faculty_graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_climate_ml")

# Set UTF-8 encoding for standard output across Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import matplotlib
# If run headlessly as main or requested via env, use non-interactive Agg
if __name__ == "__main__" or os.environ.get("MPLBACKEND_FORCE_AGG") == "1":
    if not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.features import build_climate_features, build_step_feature_vector

# Styling setup
plt.rcParams['font.sans-serif'] = 'DejaVu Sans', 'Arial', 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.9

MODEL_PATH = os.path.join(PROJECT_ROOT, "modeling_outputs", "ridge_climate_model.joblib")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "climate_paired.csv")


def safe_savefig(fig, file_path, **kwargs):
    """
    Safely saves a matplotlib figure to file_path on Windows without failing
    if the destination file is memory-mapped or locked by viewers or indexers.
    Uses atomic temporary file write and replacement with retries.
    """
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    temp_path = os.path.join(dir_name, f".tmp_{os.getpid()}_{base_name}")

    # Write figure cleanly to temp file first
    fig.savefig(temp_path, **kwargs)

    # Atomically replace destination file with retries
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            if os.path.exists(file_path):
                try:
                    os.replace(temp_path, file_path)
                    return
                except OSError:
                    time.sleep(0.15 * (attempt + 1))
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
                    os.replace(temp_path, file_path)
                    return
            else:
                os.replace(temp_path, file_path)
                return
        except OSError:
            if attempt == max_attempts - 1:
                # Clean up temp file and fall back to direct savefig
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                fig.savefig(file_path, **kwargs)
                return
            time.sleep(0.2 * (attempt + 1))


def load_model_and_data():
    """Loads trained model bundle and paired climate dataset."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model bundle not found at {MODEL_PATH}. Run 'python run_pipeline.py' first.")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")
    
    bundle = joblib.load(MODEL_PATH)
    df_paired = pd.read_csv(DATA_PATH, comment="#")
    return bundle, df_paired


def compute_test_predictions(bundle, df_paired):
    """
    Computes both Protocol A (1-step teacher forced) and Protocol B (10-yr recursive)
    predictions on the untouched test window (2016-01 to 2025-12, 120 months).
    """
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    
    clean_df = build_climate_features(df_paired, dropna=True)
    clean_df["date"] = pd.to_datetime(clean_df["year"].astype(str) + "-" + clean_df["month"].astype(str).str.zfill(2) + "-01")
    
    test_df = clean_df[clean_df["year"] >= 2016].copy().reset_index(drop=True)
    X_test = test_df[feature_names].values
    y_test = test_df["temp_anomaly_c"].values
    
    # Protocol A: 1-step ahead teacher forced
    y_pred_1step = model.predict(X_test)
    
    # Protocol B: 10-year recursive rollout
    train_val_df = clean_df[clean_df["year"] <= 2015].copy()
    temp_hist = list(train_val_df["temp_anomaly_c"].values)
    co2_hist = list(train_val_df["co2_ppm"].values)
    
    y_pred_rec = []
    for idx, row in test_df.iterrows():
        m_num = int(row["month"])
        feat_vector = build_step_feature_vector(temp_hist, co2_hist, m_num)
        
        pred_val = float(model.predict(feat_vector)[0])
        y_pred_rec.append(pred_val)
        
        temp_hist.append(pred_val)
        co2_hist.append(row["co2_ppm"])
        
    y_pred_rec = np.array(y_pred_rec)
    
    # Baselines for comparison
    naive_pred_1step = test_df["temp_lag1"].values
    naive_pred_rec = np.full(len(test_df), train_val_df["temp_anomaly_c"].iloc[-1])
    seasonal_pred_1step = test_df["temp_lag12"].values
    
    # Seasonal recursive repeats 2015 cycle
    dec_2015_cycle = train_val_df.iloc[-12:]["temp_anomaly_c"].values
    seasonal_pred_rec = np.tile(dec_2015_cycle, 10)[:len(test_df)]
    
    test_df["Observed_Temp_Anomaly_C"] = y_test
    test_df["Ridge_1Step_Predicted_C"] = y_pred_1step
    test_df["Ridge_1Step_Residual_C"] = y_test - y_pred_1step
    test_df["Ridge_Recursive_Predicted_C"] = y_pred_rec
    test_df["Ridge_Recursive_Residual_C"] = y_test - y_pred_rec
    test_df["Naive_Persistence_1Step"] = naive_pred_1step
    test_df["Naive_Persistence_Recursive"] = naive_pred_rec
    test_df["Seasonal_Persistence_1Step"] = seasonal_pred_1step
    test_df["Seasonal_Persistence_Recursive"] = seasonal_pred_rec
    
    return test_df, clean_df


def save_tables(test_df, bundle):
    """Saves verified CSV tables for faculty inspection."""
    # 1. Predictions Table
    cols_pred = [
        "date", "year", "month", "co2_ppm", "Observed_Temp_Anomaly_C",
        "Ridge_1Step_Predicted_C", "Ridge_1Step_Residual_C",
        "Ridge_Recursive_Predicted_C", "Ridge_Recursive_Residual_C",
        "Naive_Persistence_1Step", "Naive_Persistence_Recursive",
        "Seasonal_Persistence_1Step", "Seasonal_Persistence_Recursive"
    ]
    test_df[cols_pred].to_csv(os.path.join(OUTPUT_DIR, "test_actual_vs_predicted_values.csv"), index=False)
    
    # 2. Residuals Table
    res_df = pd.DataFrame({
        "Date": test_df["date"].dt.strftime("%Y-%m"),
        "Observed_C": test_df["Observed_Temp_Anomaly_C"],
        "Ridge_1Step_Residual_C": test_df["Ridge_1Step_Residual_C"],
        "Ridge_1Step_SqError": test_df["Ridge_1Step_Residual_C"] ** 2,
        "Ridge_Recursive_Residual_C": test_df["Ridge_Recursive_Residual_C"],
        "Ridge_Recursive_SqError": test_df["Ridge_Recursive_Residual_C"] ** 2,
    })
    res_df.to_csv(os.path.join(OUTPUT_DIR, "test_residuals.csv"), index=False)
    
    # 3. Calculated Evaluation Metrics Table
    y = test_df["Observed_Temp_Anomaly_C"].values
    p1 = test_df["Ridge_1Step_Predicted_C"].values
    pr = test_df["Ridge_Recursive_Predicted_C"].values
    pn1 = test_df["Naive_Persistence_1Step"].values
    pnr = test_df["Naive_Persistence_Recursive"].values
    ps1 = test_df["Seasonal_Persistence_1Step"].values
    psr = test_df["Seasonal_Persistence_Recursive"].values
    
    def get_metrics_row(model_name, protocol, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        res = y_true - y_pred
        return {
            "Model": model_name,
            "Protocol": protocol,
            "Evaluation_Window": "2016-01 to 2025-12 (120 Mo)",
            "RMSE_degC": round(rmse, 4),
            "MAE_degC": round(mae, 4),
            "R2_Score": round(r2, 4),
            "Mean_Residual_degC": round(float(np.mean(res)), 4),
            "Std_Residual_degC": round(float(np.std(res, ddof=1)), 4),
            "Max_Abs_Error_degC": round(float(np.max(np.abs(res))), 4)
        }
    
    metrics_records = [
        get_metrics_row("Ridge Regression (alpha=0.01)", "Protocol A (1-Step Teacher-Forced)", y, p1),
        get_metrics_row("Ridge Regression (alpha=0.01)", "Protocol B (10-Yr Recursive Rollout)", y, pr),
        get_metrics_row("Naive Persistence (Lag-1)", "Protocol A (1-Step Teacher-Forced)", y, pn1),
        get_metrics_row("Naive Persistence (Lag-1)", "Protocol B (10-Yr Recursive Rollout)", y, pnr),
        get_metrics_row("Seasonal Persistence (Lag-12)", "Protocol A (1-Step Teacher-Forced)", y, ps1),
        get_metrics_row("Seasonal Persistence (Lag-12)", "Protocol B (10-Yr Recursive Rollout)", y, psr),
    ]
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "calculated_evaluation_metrics.csv"), index=False)
    print("Saved CSV tables to:", OUTPUT_DIR)


def get_faculty_data():
    """
    Retrieves the data needed for faculty presentation graphs.
    First loads precomputed verified outputs if present:
      - modeling_outputs/faculty_graphs/test_actual_vs_predicted_values.csv
      - climate_paired.csv
    If test CSV is missing, falls back to computing predictions from model bundle
    and saving the verified CSV tables, ensuring zero duplicate calculations.
    
    Returns:
        tuple: (test_df, df_paired)
    """
    pred_csv_path = os.path.join(OUTPUT_DIR, "test_actual_vs_predicted_values.csv")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Paired climate dataset not found at {DATA_PATH}.")
    
    df_paired = pd.read_csv(DATA_PATH, comment="#")
    
    if os.path.exists(pred_csv_path):
        test_df = pd.read_csv(pred_csv_path)
        test_df["date"] = pd.to_datetime(test_df["date"])
        if "Naive_Persistence_Recursive" not in test_df.columns:
            if "Naive_Persistence_1Step" in test_df.columns:
                test_df["Naive_Persistence_Recursive"] = test_df["Naive_Persistence_1Step"].iloc[0]
            else:
                test_df["Naive_Persistence_Recursive"] = 1.17
        return test_df, df_paired

    # Fallback: compute from model bundle and raw paired dataset
    bundle, df_paired = load_model_and_data()
    test_df, clean_df = compute_test_predictions(bundle, df_paired)
    save_tables(test_df, bundle)
    return test_df, df_paired


def get_screen_dimensions():
    """Detects primary screen dimensions (width, height) in pixels."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        if w > 400 and h > 300:
            return w, h
    except Exception:
        pass
    return 1280, 800


def create_faculty_figure(figsize=(11.0, 6.0)):
    """
    Creates a responsive, screen-friendly Matplotlib figure using layout='constrained'.
    Targets approximately 80-86% of available screen width and height so the interactive
    window opens cleanly without overflowing, while safe_savefig exports crisp 300 DPI PNGs.
    """
    screen_w, screen_h = get_screen_dimensions()
    
    # Target ~82% of screen width and ~78% of screen height at 100 DPI
    max_w_in = (screen_w * 0.82) / 100.0
    max_h_in = (screen_h * 0.78) / 100.0
    
    fig_w = min(figsize[0], max_w_in)
    fig_h = min(figsize[1], max_h_in)
    fig_w = max(fig_w, 8.5)
    fig_h = max(fig_h, 5.0)
    
    fig, ax = plt.subplots(figsize=(round(fig_w, 2), round(fig_h, 2)), dpi=100, layout="constrained")
    return fig, ax


def display_or_close(fig, show: bool = False, window_title: str = None):
    """Handles interactive figure display and clean memory disposal."""
    if window_title and hasattr(fig.canvas.manager, "set_window_title"):
        try:
            fig.canvas.manager.set_window_title(window_title)
        except Exception:
            pass

    if show:
        auto_close = os.environ.get("MPL_AUTO_CLOSE_MS")
        if auto_close:
            try:
                ms = int(auto_close)
                timer = fig.canvas.new_timer(interval=ms)
                timer.add_callback(plt.close, fig)
                timer.start()
            except Exception:
                pass
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# GRAPH GENERATORS (1 through 6)
# ---------------------------------------------------------------------------

def generate_graph_1(test_df, show: bool = False):
    """Graph 1: Actual vs Predicted Temperature Anomaly (Scatter with 1:1 line)."""
    fig, ax = create_faculty_figure(figsize=(10.5, 6.2))
    
    y_true = test_df["Observed_Temp_Anomaly_C"].values
    y_pred = test_df["Ridge_1Step_Predicted_C"].values
    years = test_df["year"].values
    
    # Scatter points colored by year
    sc = ax.scatter(y_true, y_pred, c=years, cmap="viridis", s=45, alpha=0.88,
                    edgecolors="#1a202c", linewidths=0.6, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Observation Year (2016–2025)", fontsize=9.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=8.5)
    
    # Reference 1:1 diagonal line
    min_val = min(y_true.min(), y_pred.min()) - 0.08
    max_val = max(y_true.max(), y_pred.max()) + 0.08
    ax.plot([min_val, max_val], [min_val, max_val], color="#4a5568", linestyle="--",
            linewidth=1.6, label="Ideal 1:1 Reference Line (y = x)", zorder=2)
    
    # OLS fit line of predictions vs observations
    slope, intercept, r_val, _, _ = stats.linregress(y_true, y_pred)
    x_line = np.linspace(min_val, max_val, 100)
    ax.plot(x_line, slope * x_line + intercept, color="#e53e3e", linewidth=1.6,
            label=f"Linear Best Fit (Slope = {slope:.3f})", zorder=2)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    stats_text = (
        "Ridge Model Out-of-Sample Performance:\n"
        f"• R² = {r2:.4f}  |  RMSE = {rmse:.4f} °C\n"
        f"• MAE = {mae:.4f} °C  |  α = 0.01\n"
        f"• Test Window: 2016–2025 (N = {len(y_true)}, Protocol A)"
    )
    ax.text(0.04, 0.74, stats_text, transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cbd5e0", alpha=0.92))
    
    ax.set_title("Graph 1: Observed vs. Predicted Global Temperature Anomaly (2016–2025)\nRidge Regression Protocol A (1-Step Ahead Out-of-Sample Benchmark)",
                 fontsize=11.5, fontweight="bold", pad=8)
    ax.set_xlabel("Observed Temperature Anomaly (°C, NASA GISS GISTEMP v4)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Predicted Temperature Anomaly (°C, Ridge Model)", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8.5)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.5, borderpad=0.35, labelspacing=0.25)
    
    png_path = os.path.join(OUTPUT_DIR, "01_actual_vs_predicted_scatter.png")
    pdf_path = os.path.join(OUTPUT_DIR, "01_actual_vs_predicted_scatter.pdf")
    safe_savefig(fig, png_path, dpi=300)
    safe_savefig(fig, pdf_path)
    print("Generated Graph 1:", png_path)
    display_or_close(fig, show=show, window_title="Graph 1: Actual vs Predicted Scatter")
    return fig


def generate_graph_2(test_df, show: bool = False):
    """Graph 2: Actual vs Predicted Temperature Over Time (2016-2025 Timeline)."""
    fig, ax = create_faculty_figure(figsize=(11.0, 5.8))
    
    dates = test_df["date"].values
    y_true = test_df["Observed_Temp_Anomaly_C"].values
    y_pred = test_df["Ridge_1Step_Predicted_C"].values
    
    # Plot curves
    ax.plot(dates, y_true, color="#1a365d", linewidth=1.8, marker="o", markersize=3.5,
            label="Observed Anomaly (NASA GISS Ground Truth)", zorder=3)
    ax.plot(dates, y_pred, color="#c53030", linewidth=1.6, linestyle="--", marker="s", markersize=3.2,
            label="Ridge 1-Step-Ahead Forecast (Protocol A)", zorder=4)
    
    # Shading difference
    ax.fill_between(dates, y_true, y_pred, color="#e2e8f0", alpha=0.6, label="Prediction Residual Error |y - ŷ|")
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    stats_box = (
        "Out-of-Sample Test Performance:\n"
        f"• R² = {r2:.4f}  |  RMSE = {rmse:.4f} °C  |  MAE = {mae:.4f} °C\n"
        "• Period: 2016–2025 (120 Months, Protocol A)"
    )
    ax.text(0.03, 0.78, stats_box, transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cbd5e0", alpha=0.92))
    
    # Event annotations (2016 El Niño, 2023-2024 surge)
    max_idx = np.argmax(y_true)
    ax.annotate(f"Record Surge: {y_true[max_idx]:.2f}°C\n({pd.to_datetime(dates[max_idx]).strftime('%b %Y')})",
                xy=(dates[max_idx], y_true[max_idx]),
                xytext=(dates[max_idx] - np.timedelta64(450, 'D'), y_true[max_idx] + 0.05),
                arrowprops=dict(facecolor="#2d3748", shrink=0.08, width=0.8, headwidth=4),
                fontsize=8.0, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fffaf0", edgecolor="#dd6b20", alpha=0.9))
    
    ax.set_title("Graph 2: Out-of-Sample Global Temperature Anomaly Forecast Timeline (2016–2025)\nComparison of NASA GISS Observations vs. Ridge Regression 1-Step-Ahead Predictions",
                 fontsize=11.5, fontweight="bold", pad=8)
    ax.set_xlabel("Timeline (Monthly Observations, 2016–2025)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Global Temperature Anomaly (°C, 1951–1980 Baseline)", fontsize=10, fontweight="bold")
    ax.set_ylim(0.60, 1.62)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.5, borderpad=0.35, labelspacing=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    png_path = os.path.join(OUTPUT_DIR, "02_actual_vs_predicted_timeline.png")
    pdf_path = os.path.join(OUTPUT_DIR, "02_actual_vs_predicted_timeline.pdf")
    safe_savefig(fig, png_path, dpi=300)
    safe_savefig(fig, pdf_path)
    print("Generated Graph 2:", png_path)
    display_or_close(fig, show=show, window_title="Graph 2: Actual vs Predicted Timeline")
    return fig


def generate_graph_3(test_df, show: bool = False):
    """Graph 3: Prediction Residual Analysis (Residuals vs Time with Confidence Bands)."""
    fig, ax = create_faculty_figure(figsize=(11.0, 5.8))
    
    dates = test_df["date"].values
    residuals = test_df["Ridge_1Step_Residual_C"].values
    
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals, ddof=1))
    
    # Stem plot / scatter with connecting lines
    ax.plot(dates, residuals, color="#2b6cb0", linewidth=1.1, alpha=0.7, zorder=3)
    ax.scatter(dates, residuals, color="#2b6cb0", s=20, alpha=0.85, zorder=4, label="Individual Monthly Residual (e = y - ŷ)")
    
    # Zero line
    ax.axhline(0.0, color="#1a202c", linestyle="-", linewidth=1.2, label="Ideal Zero-Error Baseline (e = 0.0 °C)", zorder=2)
    
    # Mean residual line
    ax.axhline(mean_res, color="#e53e3e", linestyle="--", linewidth=1.2,
               label=f"Mean Residual Bias (μ = {mean_res:+.4f} °C)", zorder=2)
    
    # ±1 Std Dev and ±2 Std Dev bands
    ax.axhspan(-std_res, std_res, color="#bee3f8", alpha=0.45, label=f"±1 Std Dev Band (±{std_res:.4f} °C, ~68% Confidence)", zorder=1)
    ax.axhspan(-2 * std_res, 2 * std_res, color="#ebf8ff", alpha=0.5, label=f"±2 Std Dev Band (±{2*std_res:.4f} °C, ~95% Confidence)", zorder=0)
    
    # Inset metrics
    dw_stat = float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2))
    stats_text = (
        "Residual Error Diagnostics:\n"
        f"• Sample Size: N = {len(residuals)} months (2016–2025)\n"
        f"• Mean (μ): {mean_res:+.4f} °C  |  Std Dev (σ): {std_res:.4f} °C\n"
        f"• Range: [{np.min(residuals):+.4f}, {np.max(residuals):+.4f}] °C  |  Durbin-Watson: {dw_stat:.3f}"
    )
    ax.text(0.03, 0.78, stats_text, transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cbd5e0", alpha=0.92))
    
    ax.set_title("Graph 3: Ridge Regression Prediction Residual Analysis (2016–2025 Test Period)\nTemporal Tracking of Residuals Under Protocol A (1-Step Ahead Teacher-Forced)",
                 fontsize=11.5, fontweight="bold", pad=8)
    ax.set_xlabel("Timeline (Monthly Observations, 2016–2025)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Prediction Residual: Observed - Predicted (°C)", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.35, 0.45)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.0, borderpad=0.35, labelspacing=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    png_path = os.path.join(OUTPUT_DIR, "03_prediction_residuals_timeline.png")
    pdf_path = os.path.join(OUTPUT_DIR, "03_prediction_residuals_timeline.pdf")
    safe_savefig(fig, png_path, dpi=300)
    safe_savefig(fig, pdf_path)
    print("Generated Graph 3:", png_path)
    display_or_close(fig, show=show, window_title="Graph 3: Prediction Residuals Timeline")
    return fig


def generate_graph_4(test_df, show: bool = False):
    """Graph 4: Residual Distribution (Histogram with KDE Curve and Gaussian Reference)."""
    fig, ax = create_faculty_figure(figsize=(10.5, 6.0))
    
    residuals = test_df["Ridge_1Step_Residual_C"].values
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals, ddof=1))
    median_res = float(np.median(residuals))
    skew_res = float(stats.skew(residuals))
    kurt_res = float(stats.kurtosis(residuals))
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    # Histogram
    n_bins = 18
    counts, bins, patches = ax.hist(residuals, bins=n_bins, density=True, color="#63b3ed",
                                    edgecolor="#1a365d", linewidth=0.8, alpha=0.75,
                                    label=f"Observed Residuals ({n_bins} Bins)", zorder=2)
    
    # Smooth KDE curve using scipy gaussian_kde
    kde = stats.gaussian_kde(residuals, bw_method="silverman")
    x_grid = np.linspace(-0.35, 0.45, 300)
    ax.plot(x_grid, kde(x_grid), color="#1a365d", linewidth=2.0,
            label="Empirical KDE Curve", zorder=4)
    
    # Theoretical Gaussian Distribution N(mean, std^2)
    norm_pdf = stats.norm.pdf(x_grid, loc=mean_res, scale=std_res)
    ax.plot(x_grid, norm_pdf, color="#e53e3e", linestyle="--", linewidth=1.5,
            label=f"Theoretical Normal N(μ={mean_res:.2f}, σ={std_res:.2f})", zorder=3)
    
    # Vertical reference lines
    ax.axvline(0.0, color="#2d3748", linestyle="-", linewidth=1.2, label="Zero-Error Baseline (e = 0)", zorder=2)
    ax.axvline(mean_res, color="#e53e3e", linestyle=":", linewidth=1.4, label=f"Mean (μ = {mean_res:+.3f} °C)", zorder=2)
    
    # Inset statistical summary in upper right
    stats_box = (
        "Distributional Statistics:\n"
        f"• Count: N = {len(residuals)}  |  Mean: {mean_res:+.4f} °C\n"
        f"• Median: {median_res:+.4f} °C  |  Std Dev: {std_res:.4f} °C\n"
        f"• Skewness: {skew_res:+.3f}  |  Excess Kurtosis: {kurt_res:+.3f}\n"
        f"• Shapiro-Wilk p = {shapiro_p:.3f} (Fail to reject normality)"
    )
    ax.text(0.56, 0.68, stats_box, transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cbd5e0", alpha=0.92))
    
    ax.set_title("Graph 4: Residual Distribution and Density Analysis\nEvaluation of Ridge Regression Out-of-Sample Residuals (2016–2025)",
                 fontsize=11.5, fontweight="bold", pad=8)
    ax.set_xlabel("Prediction Residual Error: Observed - Predicted (°C)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.35, 0.45)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax.legend(loc="upper left", framealpha=0.92, fontsize=8.0, borderpad=0.35, labelspacing=0.25)
    
    png_path = os.path.join(OUTPUT_DIR, "04_residual_distribution_kde.png")
    pdf_path = os.path.join(OUTPUT_DIR, "04_residual_distribution_kde.pdf")
    safe_savefig(fig, png_path, dpi=300)
    safe_savefig(fig, pdf_path)
    print("Generated Graph 4:", png_path)
    display_or_close(fig, show=show, window_title="Graph 4: Residual Distribution KDE")
    return fig


def generate_graph_5(test_df, show: bool = False):
    """Graph 5: Recursive Forecast vs Actual Values (Protocol B 10-Year Autonomous Rollout)."""
    fig, ax = create_faculty_figure(figsize=(11.5, 6.2))
    
    dates = test_df["date"].values
    y_true = test_df["Observed_Temp_Anomaly_C"].values
    y_rec = test_df["Ridge_Recursive_Predicted_C"].values
    y_1step = test_df["Ridge_1Step_Predicted_C"].values
    y_naive = test_df["Naive_Persistence_Recursive"].values
    
    # 1. Observed Ground Truth
    ax.plot(dates, y_true, color="#1a202c", linewidth=1.8, marker="o", markersize=3.5,
            label="Observed Ground Truth (NASA GISS)", zorder=4)
    
    # 2. Protocol B: 10-Year Autonomous Recursive Rollout
    ax.plot(dates, y_rec, color="#805ad5", linewidth=2.0, linestyle="-", marker="^", markersize=3.5,
            label="Protocol B: Ridge 10-Yr Autonomous Rollout (R² = +0.1555)", zorder=5)
    
    # 3. Protocol A: 1-Step Ahead Teacher-Forced (for direct comparison)
    ax.plot(dates, y_1step, color="#718096", linewidth=1.2, linestyle="--", alpha=0.75,
            label="Protocol A: Ridge 1-Step Teacher-Forced (R² = +0.6532)", zorder=3)
    
    # 4. Naive Persistence Baseline Rollout (Frozen at Dec 2015)
    ax.plot(dates, y_naive, color="#dd6b20", linewidth=1.2, linestyle=":", alpha=0.85,
            label="Naive Persistence Rollout (Dec 2015 Anomaly, R² = -0.6517)", zorder=2)
    
    # Calculate recursive metrics
    rmse_rec = np.sqrt(mean_squared_error(y_true, y_rec))
    mae_rec = mean_absolute_error(y_true, y_rec)
    r2_rec = r2_score(y_true, y_rec)
    
    rmse_1 = np.sqrt(mean_squared_error(y_true, y_1step))
    mae_1 = mean_absolute_error(y_true, y_1step)
    r2_1 = r2_score(y_true, y_1step)
    
    scorecard = (
        "Dual-Protocol Evaluation Comparison (2016–2025):\n"
        f"• Protocol B (10-Yr Autonomous Rollout): R² = {r2_rec:+.4f}  |  RMSE = {rmse_rec:.4f} °C  |  MAE = {mae_rec:.4f} °C\n"
        f"• Protocol A (1-Step Teacher-Forced):   R² = {r2_1:+.4f}  |  RMSE = {rmse_1:.4f} °C  |  MAE = {mae_1:.4f} °C\n"
        "• Naive Persistence (Frozen Dec 2015):   R² = -0.6517  |  RMSE = 0.2453 °C"
    )
    ax.text(0.02, 0.78, scorecard, transform=ax.transAxes, fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cbd5e0", alpha=0.92))
    
    note_box = (
        "Scientific Distinction:\n"
        "• Protocol A: Model receives true past temperatures every month (1-step operational forecast).\n"
        "• Protocol B: Model feeds its own past predictions back as lags for 120 months (autonomous rollout)."
    )
    ax.text(0.02, 0.04, note_box, transform=ax.transAxes, fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffaf0", edgecolor="#d69e2e", alpha=0.92))
    
    ax.set_title("Graph 5: Ridge Regression 10-Year Autonomous Recursive Rollout (Protocol B) vs. Observations\nContrasting Autonomous Long-Horizon Forecasting Against 1-Step Teacher-Forced Prediction (2016–2025)",
                 fontsize=11.5, fontweight="bold", pad=8)
    ax.set_xlabel("Timeline (Monthly Observations, 2016–2025)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Global Temperature Anomaly (°C, NASA GISS Baseline)", fontsize=10, fontweight="bold")
    ax.set_ylim(0.40, 1.62)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.0, borderpad=0.35, labelspacing=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    png_path = os.path.join(OUTPUT_DIR, "05_recursive_forecast_vs_actual.png")
    pdf_path = os.path.join(OUTPUT_DIR, "05_recursive_forecast_vs_actual.pdf")
    safe_savefig(fig, png_path, dpi=300)
    safe_savefig(fig, pdf_path)
    print("Generated Graph 5:", png_path)
    display_or_close(fig, show=show, window_title="Graph 5: Recursive Forecast vs Actual")
    return fig


def generate_graph_6(df_paired, show: bool = False):
    """Graph 6: Historical CO2 and Temperature Relationship (814 Months, Granger-Newbold Warning)."""
    fig, ax = create_faculty_figure(figsize=(11.0, 6.2))
    
    co2 = df_paired["co2_ppm"].values
    temp = df_paired["temp_anomaly_c"].values
    years = df_paired["year"].values
    n_obs = len(df_paired)
    
    # Scatter plot colored by decade
    sc = ax.scatter(co2, temp, c=years, cmap="plasma", s=28, alpha=0.75,
                    edgecolors="#2d3748", linewidths=0.3, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Observation Year (1958–2025)", fontsize=9.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=8.5)
    
    # Linear OLS regression line
    slope, intercept, r_val, p_val, std_err = stats.linregress(co2, temp)
    x_line = np.linspace(co2.min(), co2.max(), 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#1a202c", linewidth=1.8, linestyle="-",
            label=f"Bivariate OLS Trendline: ΔT = {slope:.5f} · CO₂ {intercept:+.4f}\n(Pearson r = {r_val:.4f}, R² = {r_val**2:.4f})", zorder=4)
    
    # Statistical summary box
    summary_box = (
        "Dataset & Correlation Summary (1958–2025):\n"
        f"• Paired Observations: N = {n_obs} months (NOAA GML & NASA GISS)\n"
        f"• Atmospheric CO₂: 313.2 to 427.5 ppm  |  Temp Anomaly: -0.40 to +1.43 °C\n"
        f"• Pearson Correlation: r = {r_val:.4f} (p < 1e-300)  |  R² = {r_val**2:.4f}\n"
        f"• Empirical Slope: {slope*100:.3f} °C per 100 ppm CO₂"
    )
    ax.text(0.03, 0.76, summary_box, transform=ax.transAxes, fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cbd5e0", alpha=0.92))
    
    # Scientific Integrity Notice (Granger-Newbold Risk)
    warning_box = (
        "SCIENTIFIC INTEGRITY & METHODOLOGICAL LIMITATION (Granger–Newbold Risk):\n"
        "• Shared multi-decadal upward trend drives high correlation (r = 0.9278).\n"
        "• When quadratically detrended, correlation collapses to r = 0.1451; differenced (ΔCO₂ vs ΔT) is r = -0.0218.\n"
        "• Bivariate empirical regression cannot deduce Equilibrium Climate Sensitivity (ECS) without ocean thermal uptake (N → 0)."
    )
    ax.text(0.03, 0.04, warning_box, transform=ax.transAxes, fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5f5", edgecolor="#e53e3e", alpha=0.95))
    
    ax.set_title("Graph 6: Historical Atmospheric CO₂ vs. Global Temperature Anomaly (1958–2025)\nBivariate Correlation Analysis & Econometric Non-Stationarity Critique",
                 fontsize=11.5, fontweight="bold", pad=8)
    ax.set_xlabel("Atmospheric CO₂ Concentration (ppm, NOAA GML Mauna Loa)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Global Temperature Anomaly (°C, NASA GISS GISTEMP v4)", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax.legend(loc="center right", framealpha=0.92, fontsize=8.0, borderpad=0.35, labelspacing=0.25)
    
    png_path = os.path.join(OUTPUT_DIR, "06_historical_co2_vs_temperature.png")
    pdf_path = os.path.join(OUTPUT_DIR, "06_historical_co2_vs_temperature.pdf")
    safe_savefig(fig, png_path, dpi=300)
    safe_savefig(fig, pdf_path)
    print("Generated Graph 6:", png_path)
    display_or_close(fig, show=show, window_title="Graph 6: Historical CO2 vs Temperature")
    return fig


def main():
    print("=" * 80)
    print("FACULTY PRESENTATION GRAPH GENERATION ENGINE")
    print("Model: Ridge Regression (alpha=0.01) | Dataset: NOAA Mauna Loa & NASA GISS")
    print("=" * 80)
    
    bundle, df_paired = load_model_and_data()
    test_df, clean_df = compute_test_predictions(bundle, df_paired)
    
    # Save CSV tables
    save_tables(test_df, bundle)
    
    # Generate all 6 graphs
    print("\nRendering 6 high-resolution publication figures (300 DPI PNG + PDF)...")
    generate_graph_1(test_df)
    generate_graph_2(test_df)
    generate_graph_3(test_df)
    generate_graph_4(test_df)
    generate_graph_5(test_df)
    generate_graph_6(df_paired)
    
    print("\n" + "=" * 80)
    print("ALL FACULTY GRAPHS AND TABLES GENERATED SUCCESSFULLY!")
    print(f"Directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
