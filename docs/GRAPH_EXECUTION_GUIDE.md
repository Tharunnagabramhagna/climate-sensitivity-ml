# Climate Sensitivity Under 2×CO₂: Interactive Graph Execution Guide

This guide explains how to view, interact with, and export the six faculty presentation graphs for the **Investigating Climate Sensitivity Under 2×CO₂** research project.

---

## 1. Quick Start Overview

You can open and inspect any of the six faculty graphs in two ways:
1. **Interactive Menu / CLI Launcher** (`run_graph.py`) — Convenient numbered menu or shortcut flag.
2. **Direct Script Execution** (`modeling_outputs/faculty_graphs/0X_...py`) — Standard Python script execution opening a real, interactive Matplotlib window.

Each execution displays the figure in an interactive desktop window (zoom, pan, inspect data coordinates, save) and automatically updates high-resolution publication assets (**300 DPI PNG** and vector **PDF**).

---

## 2. Launching via the Interactive Menu (`run_graph.py`)

Run the launcher from the project root without arguments:

```bash
python run_graph.py
```

You will see the interactive menu:

```text
CLIMATE SENSITIVITY PROJECT — GRAPH MENU

1. Actual vs Predicted Scatter
2. Actual vs Predicted Timeline
3. Prediction Residuals Timeline
4. Residual Distribution KDE
5. Recursive Forecast vs Actual
6. Historical CO₂ vs Temperature

Enter graph number (1–6):
```

Type a number from `1` to `6` and press **Enter**. The interactive Matplotlib window will immediately open.

---

## 3. Fast Command-Line Shortcuts (`run_graph.py <number>`)

You can directly launch any graph by supplying its number as a command-line argument:

| Command | Graph Name | Description |
| :--- | :--- | :--- |
| `python run_graph.py 1` | **Actual vs Predicted Scatter** | Scatter plot with 1:1 line & linear best-fit |
| `python run_graph.py 2` | **Actual vs Predicted Timeline** | Monthly 2016–2025 timeline with residual error shading |
| `python run_graph.py 3` | **Prediction Residuals Timeline** | Residuals tracking with mean bias & $\pm 1\sigma, \pm 2\sigma$ bands |
| `python run_graph.py 4` | **Residual Distribution KDE** | Error density histogram with KDE and theoretical Gaussian reference |
| `python run_graph.py 5` | **Recursive Forecast vs Actual** | 10-year autonomous recursive rollout (Protocol B) comparison |
| `python run_graph.py 6` | **Historical CO₂ vs Temperature** | 814-month bivariate correlation (1958–2025) & Granger-Newbold critique |

---

## 4. Running Individual Graph Scripts Directly

Each graph script is located in `modeling_outputs/faculty_graphs/` and can be executed independently from any terminal directory:

```bash
# Graph 1: Actual vs. Predicted Scatter
python modeling_outputs/faculty_graphs/01_actual_vs_predicted_scatter.py

# Graph 2: Actual vs. Predicted Timeline
python modeling_outputs/faculty_graphs/02_actual_vs_predicted_timeline.py

# Graph 3: Prediction Residuals Timeline
python modeling_outputs/faculty_graphs/03_prediction_residuals_timeline.py

# Graph 4: Residual Distribution KDE
python modeling_outputs/faculty_graphs/04_residual_distribution_kde.py

# Graph 5: Recursive Forecast vs. Actual
python modeling_outputs/faculty_graphs/05_recursive_forecast_vs_actual.py

# Graph 6: Historical CO2 vs. Temperature
python modeling_outputs/faculty_graphs/06_historical_co2_vs_temperature.py
```

---

## 5. Data Sources Used by Each Graph

All calculations reuse verified project data and precomputed outputs, ensuring instant loading without redundant retraining or data modification:

| Graph Script | Primary Data Source | Key Scientific Columns / Metrics |
| :--- | :--- | :--- |
| `01_actual_vs_predicted_scatter.py` | `modeling_outputs/faculty_graphs/test_actual_vs_predicted_values.csv` | `Observed_Temp_Anomaly_C`, `Ridge_1Step_Predicted_C`, $R^2 = 0.6532$, $\text{RMSE} = 0.1124^\circ\text{C}$ |
| `02_actual_vs_predicted_timeline.py` | `modeling_outputs/faculty_graphs/test_actual_vs_predicted_values.csv` | `date`, `Observed_Temp_Anomaly_C`, `Ridge_1Step_Predicted_C`, Error shading, Surge peak annotations |
| `03_prediction_residuals_timeline.py` | `modeling_outputs/faculty_graphs/test_actual_vs_predicted_values.csv` | `date`, `Ridge_1Step_Residual_C`, $\mu = +0.0310^\circ\text{C}$, $s_{\text{test}} = 0.1085^\circ\text{C}$, Durbin-Watson stat |
| `04_residual_distribution_kde.py` | `modeling_outputs/faculty_graphs/test_actual_vs_predicted_values.csv` | `Ridge_1Step_Residual_C`, Empirical KDE, Normal PDF $N(\mu, s^2)$, Shapiro-Wilk test ($p > 0.05$) |
| `05_recursive_forecast_vs_actual.py` | `modeling_outputs/faculty_graphs/test_actual_vs_predicted_values.csv` | `date`, `Observed_Temp_Anomaly_C`, `Ridge_Recursive_Predicted_C`, `Naive_Persistence_Recursive` |
| `06_historical_co2_vs_temperature.py` | `data/processed/climate_paired.csv` (814 contiguous monthly records) | `co2_ppm` (NOAA GML), `temp_anomaly_c` (NASA GISS), Pearson $r = 0.9278$, Detrended $r = 0.1451$ |

---

## 6. Where Are Output Files Saved?

Whenever a graph script runs (interactively or via the automated pipeline), it saves two publication-grade copies into `modeling_outputs/faculty_graphs/`:

* **High-Resolution PNG** (`300 DPI` raster format, suitable for slides, web display, and reports).
* **Vector PDF** (`.pdf` format, fully scalable vector graphics for LaTeX papers and faculty submission).

| File Name (PNG & PDF) | Dimensions | Purpose |
| :--- | :--- | :--- |
| `01_actual_vs_predicted_scatter.png / .pdf` | $8.5 \times 7.5\text{ in}$ | Model validation scatter with 1:1 reference line |
| `02_actual_vs_predicted_timeline.png / .pdf` | $12.0 \times 6.0\text{ in}$ | Out-of-sample monthly forecast timeline (2016–2025) |
| `03_prediction_residuals_timeline.png / .pdf` | $12.0 \times 5.8\text{ in}$ | Residual error temporal tracking with $\pm 1\sigma, \pm 2\sigma$ bands |
| `04_residual_distribution_kde.png / .pdf` | $9.0 \times 6.2\text{ in}$ | Residual density analysis & Gaussian symmetry test |
| `05_recursive_forecast_vs_actual.png / .pdf` | $12.0 \times 6.2\text{ in}$ | Protocol B 10-year autonomous recursive rollout vs. actual |
| `06_historical_co2_vs_temperature.png / .pdf` | $9.5 \times 7.2\text{ in}$ | 814-month bivariate correlation & Granger-Newbold warning |

---

## 7. Interactive Matplotlib Controls

When the Matplotlib window opens on your screen, you can use the toolbar at the bottom:
* **Magnifying Glass (Zoom)**: Click and drag a rectangle over any time period to zoom in on specific months.
* **Crossed Arrows (Pan)**: Click and drag to slide the timeline left or right.
* **Home Icon**: Reset the view to original full dimensions.
* **Disk Icon (Save)**: Save a custom screenshot or alternative file format (SVG, EPS, TIFF).
* **Coordinate Readout**: Move your mouse over any data point to inspect exact dates, temperature anomalies, and CO₂ concentrations in the bottom-right corner.
* **Closing the Window**: Click the standard window **X** button (or press `Alt + F4` / `Ctrl + W`) to close the window and return to your terminal prompt.

---

## 8. Troubleshooting

### Issue 1: "UserWarning: FigureCanvasAgg is non-interactive" or Window Does Not Appear
* **Cause**: The Matplotlib backend was set to `Agg` (headless mode).
* **Fix**: Ensure you run the standalone script directly (`python modeling_outputs/faculty_graphs/01_actual_vs_predicted_scatter.py`) or via `python run_graph.py 1`. The scripts automatically use the native interactive `TkAgg` backend on Windows.

### Issue 2: "Required dataset or model file missing"
* **Cause**: The model pipeline has not been executed yet on this machine.
* **Fix**: Run the automated pipeline stage once to train the model and generate baseline CSV tables:
  ```bash
  python main.py --pipeline
  ```

### Issue 3: Automated Workflow Running in Headless Mode
* The master automated workflow (`python main.py --all` or `python main.py --graphs`) remains 100% headless. It generates all 28 project visualizations without opening blocking GUI windows, making it fully compatible with headless servers, CI/CD runners, and background automation.
* To launch the interactive HTML gallery in your default browser at any time:
  ```bash
  python main.py --gallery
  ```
