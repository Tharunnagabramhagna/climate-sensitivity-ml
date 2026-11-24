# PROJECT OUTPUTS GUIDE

**Project:** Investigating Climate Sensitivity Under 2×CO₂  
**Algorithms & Models:** Ridge Regression ($\alpha = 0.01$ ARX) & DOECLIM 1D Ocean Heat Diffusion  
**Data Sources:** NOAA Global Monitoring Laboratory (Mauna Loa CO₂) & NASA GISS (GISTEMP v4)

---

## 1. Directory Structure Overview

The repository is structured into clear, modular directories separating observational data, model artifacts, simulation results, tabular predictions, and visual outputs:

```
c:\climate_sensitivity_ml\
├── [observational data]        # Raw observational data and paired baseline CSV (root)
├── modeling_outputs/           # Serialized model, tabular metrics, predictions, and faculty graphs
│   ├── ridge_climate_model.joblib  # Production trained Ridge ARX model bundle
│   └── faculty_graphs/         # 6 core publication figures (PNG + vector PDF)
├── hector_outputs/             # DOECLIM physical climate model simulation outputs & graphs
├── eda_outputs/                # Exploratory data analysis graphs (10 figures)
├── interpretation_outputs/     # Model interpretability, feature importance & attribution graphs (4 figures)
├── src/                        # Centralized source code (features.py)
├── tests/                      # Automated test suite (unit, regression, physics, CLI)
└── [documentation files]       # Methodology, baseline, validation, reproduction guides
```

---

## 2. Observational Dataset Files (Repository Root)

All observational datasets cover monthly observational records and are preserved in their original format at the project root.

### 2.1 `co2_mm_mlo.csv`
- **Source:** NOAA Global Monitoring Laboratory, Mauna Loa Observatory, Hawaii.
- **Header Rows:** 56 lines of NOAA metadata comments prefixed with `#`.
- **Row Count:** 814 monthly observation rows (1958-03 to 2025-12).
- **Columns:**
  - `year`: Observation year (integer, 1958–2025).
  - `month`: Observation month (integer, 1–12).
  - `decimal_date`: Decimal year representation (float, e.g., 1958.2027).
  - `monthly_average`: Monthly mean atmospheric $\text{CO}_2$ mole fraction in dry air (ppm). Missing values flagged as -99.99.
  - `deseasonalized`: Seasonally adjusted $\text{CO}_2$ concentration (ppm).
  - `days`: Number of days with valid measurements in the month.
  - `std_dev`: Standard deviation of daily measurements (ppm).
  - `uncertainty`: Analytical uncertainty (ppm).
- **Role:** Primary forcing variable representing global atmospheric $\text{CO}_2$ concentrations.

### 2.2 `gistemp_monthly.csv`
- **Source:** Composite global temperature anomaly file containing two distinct series:
  1. `Source == 'GCAG'`: NOAA NCEI Global Component of Climate at a Glance (1850-01 to 2026-07, 2,119 records; 1901–2000 20th-century baseline).
  2. `Source == 'GISTEMP'`: NASA Goddard Institute for Space Studies (GISS) Surface Temperature Analysis (GISTEMP v4, 1880-01 to 2025-12, 1,752 records; 1951–1980 baseline).
- **Columns:** `Source`, `Year` (YYYY-MM), `Mean` (temperature anomaly in °C).
- **Role:** Primary historical target variable. The benchmark dataset `climate_paired.csv` filters strictly for `Source == 'GISTEMP'`.

### 2.3 `climate_paired.csv`
- **Generation:** Produced by merging NOAA $\text{CO}_2$ (`monthly_average`) with NASA GISTEMP monthly anomalies on chronological `(year, month)`.
- **Row Count:** 814 rows (1958-03 to 2025-12).
- **Columns:**
  - `date`: First day of observation month (`YYYY-MM-DD`).
  - `year`: Calendar year (1958–2025).
  - `month`: Month of year (1–12).
  - `co2_ppm`: Monthly atmospheric $\text{CO}_2$ concentration (ppm, range 315.71 to 427.60).
  - `temp_anomaly_c`: Global surface temperature anomaly ($^\circ\text{C}$, range -0.37 to +1.64).
- **Role:** Standard paired dataset feeding the central feature engineering pipeline `src.features.build_climate_features()`. Frozen historical benchmark.

### 2.4 Updated Observational Datasets (`data/updated/`)
- **`data/updated/gistemp_glb_latest.csv`:**
  - **Primary Scientific Origin:** NASA Goddard Institute for Space Studies (GISS), GISTEMP v4 (Lenssen et al., 2019, 2024).
  - **Redistribution Mirror:** Green Decades Climate Mirror (tidy `date,value` schema).
  - **Row Count:** 1,762 monthly records (1880-01 to 2026-10).
  - **Baseline & Units:** 1951–1980 base period, degrees Celsius anomaly (°C).
- **`data/updated/co2_mm_mlo_latest.csv`:**
  - **Source:** NOAA Global Monitoring Laboratory, Mauna Loa Observatory (824 rows, 1958-03 to 2026-10).
- **`data/updated/climate_paired_updated.csv`:**
  - **Generation:** Inner join of updated NOAA $\text{CO}_2$ and NASA GISTEMP observations through October 2026 (824 rows). Includes full provenance metadata in header comments. Historical benchmark `climate_paired.csv` remains strictly isolated.

---

## 3. Serialized Model Artifacts (`modeling_outputs/`)

### 3.1 `modeling_outputs/ridge_climate_model.joblib`
- **Algorithm:** L2-Regularized Auto-Regressive with Exogenous Inputs Linear Model (Ridge Regression).
- **Framework:** `scikit-learn` (`sklearn.linear_model.Ridge`).
- **Hyperparameters:**
  - `alpha`: $0.01$ (L2 penalty coefficient, selected via validation grid search).
  - `fit_intercept`: `True` (intercept fitted: $\beta_0 = -0.015243^\circ\text{C}$).
  - `solver`: `auto` (Cholesky decomposition).
- **Feature Schema (9 Input Features):**
  1. `temp_lag1`: Previous month's temperature anomaly ($T_{t-1}$).
  2. `temp_lag2`: Temperature anomaly lagged 2 months ($T_{t-2}$).
  3. `temp_lag3`: Temperature anomaly lagged 3 months ($T_{t-3}$).
  4. `temp_lag12`: Temperature anomaly lagged 12 months ($T_{t-12}$, capturing seasonal memory).
  5. `ln_co2_lag1`: Log-transformed $\text{CO}_2$ lagged 1 month ($\ln(\text{CO}_{2, t-1})$).
  6. `co2_diff1`: 1-month difference in $\text{CO}_2$ ($\Delta \text{CO}_{2, t} = \text{CO}_{2,t} - \text{CO}_{2,t-1}$).
  7. `co2_diff12`: 12-month annual growth rate of $\text{CO}_2$ ($\text{CO}_{2,t} - \text{CO}_{2,t-12}$).
  8. `sin_month`: Harmonic seasonal encoding $\sin(2\pi \cdot \text{month} / 12)$.
  9. `cos_month`: Harmonic seasonal encoding $\cos(2\pi \cdot \text{month} / 12)$.
- **Target Variable:** `temp_anomaly_c` at month $t$.
- **Training Set:** Chronological refit on combined Train+Val (`trainval`, 681 samples from 1959-04 to 2015-12). Zero test set leakage.
- **Serialization Format:** Python `joblib` dictionary bundle with model, metadata, coefficients, and validation error variance $\sigma_{\text{val}} = 0.11109^\circ\text{C}$.
- **Generation Command:** `python main.py --pipeline` or `python run_pipeline.py`.

---

## 4. Modeling & Simulation Tabular Outputs

### 4.1 `modeling_outputs/evaluation_protocols_comparison.csv`
- **Generated By:** `run_pipeline.py` / `main.py --pipeline`.
- **Description:** Side-by-side comparison of Protocol A (1-step teacher forced) vs Protocol B (10-year recursive rollout) across Ridge Regression and Persistence baselines.
- **Columns:** `Model`, `Protocol`, `RMSE`, `MAE`, `R2`.

### 4.2 `modeling_outputs/advanced_models_test_comparison.csv`
- **Generated By:** `run_pipeline.py` / `main.py --pipeline`.
- **Description:** Out-of-sample benchmark across candidate model architectures on the untouched test set (2016–2025).
- **Columns:** `Model`, `Test_MAE`, `Test_RMSE`, `Test_R2`.

### 4.3 `modeling_outputs/projected_anomalies_2026.csv`
- **Generated By:** `predict.py` / `main.py --predict`.
- **Description:** Monthly projections for calendar year 2026 under observed NOAA Mauna Loa $\text{CO}_2$ with nominal 95% prediction intervals derived from validation residuals via autoregressive error propagation.
- **Columns:** `date`, `noaa_measured_co2_ppm`, `forecast_temp_anomaly_c`, `ci_lower_95`, `ci_upper_95`, `status`.

### 4.4 `modeling_outputs/forecast_2026_evaluation.csv`
- **Generated By:** `src/evaluate_2026.py` / `main.py --predict`.
- **Description:** Retrospective conditional evaluation using observed 2026 atmospheric CO₂ forcing, comparing Ridge ARX projections against observed NASA GISTEMP v4 and NOAA GCAG surface temperature measurements.
- **Columns:** `Date`, `NOAA_Measured_CO2_ppm`, `Forecast_Temp_Anomaly_C`, `CI_Lower_95`, `CI_Upper_95`, `Observed_GISTEMP_C`, `Observed_GCAG_C`, `Error_vs_GISTEMP`, `Inside_95_CI`, `Evaluation_Status`.

### 4.5 `hector_outputs/hector_2xco2_ensemble_summary.csv`
- **Generated By:** `simulate_hector_2xco2.py` / `main.py --simulate`.
- **Description:** Summary metrics across deterministic and 5,000-member Monte Carlo ensemble runs for abrupt 2×CO₂ forcing ($3.71\,\text{W/m}^2$).
- **Key Columns:** `metric`, `deterministic_value`, `ensemble_median`, `ensemble_p05`, `ensemble_p95`.

### 4.5 `modeling_outputs/faculty_graphs/calculated_evaluation_metrics.csv`
- **Generated By:** `generate_faculty_graphs.py` / `main.py --graphs`.
- **Description:** Evaluated statistical metrics on the out-of-sample test set (2016–2025) comparing Ridge ARX against baselines.
- **Columns:** `model`, `test_rmse`, `test_mae`, `test_r2`, `mean_residual`, `residual_std`.

### 4.6 `modeling_outputs/faculty_graphs/test_actual_vs_predicted_values.csv`
- **Generated By:** `generate_faculty_graphs.py` / `main.py --graphs`.
- **Description:** Monthly actual observed temperature vs Protocol A predicted values and 95% uncertainty intervals for test period (2016–2025).
- **Columns:** `date`, `actual_temp_anomaly`, `predicted_temp_anomaly`, `ci_lower_95`, `ci_upper_95`.

---

## 5. Complete Visualizations Catalog (28 Unique Graphs / 34 Files)

All 28 unique graphs are generated through `python main.py --graphs` or `python generate_all_graphs.py`.

### Group A: Faculty Presentation Visualizations (6 PNGs + 6 PDFs in `modeling_outputs/faculty_graphs/`)
1. **`01_actual_vs_predicted_scatter.png` / `.pdf`**
   - *Title:* Out-of-Sample Test Set: Actual vs. Predicted Temperature Anomalies (2016–2025).
   - *Script:* `generate_faculty_graphs.py`.
   - *Description:* Scatter plot comparing observed NASA GISTEMP anomalies with Protocol A 1-step teacher-forced Ridge predictions. Features 1:1 identity line, regression line, and metrics ($R^2 = 0.6532$, $\text{RMSE} = 0.1124^\circ\text{C}$).
2. **`02_actual_vs_predicted_timeline.png` / `.pdf`**
   - *Title:* Test Period Timeline: Actual vs. Model Predictions with 95% Confidence Intervals.
   - *Script:* `generate_faculty_graphs.py`.
   - *Description:* Chronological trajectory (2016–2025) showing monthly observations tracking model projections within $\pm 1.96 \sigma_{\text{val}}$ prediction intervals.
3. **`03_prediction_residuals_timeline.png` / `.pdf`**
   - *Title:* Chronological Model Residuals: Out-of-Sample Test Evaluation.
   - *Script:* `generate_faculty_graphs.py`.
   - *Description:* Time series of prediction errors ($e_t = T_t - \hat{T}_t$) demonstrating stationarity, zero mean drift, and homoscedastic variance across the 120-month test window.
4. **`04_residual_distribution_kde.png` / `.pdf`**
   - *Title:* Residual Distribution & Normality Diagnostics.
   - *Script:* `generate_faculty_graphs.py`.
   - *Description:* Histogram and Kernel Density Estimation (KDE) curve overlaid with a theoretical normal Gaussian distribution ($N(0, \sigma^2)$) and Q-Q diagnostic plot.
5. **`05_recursive_forecast_vs_actual.png` / `.pdf`**
   - *Title:* Protocol B: Autonomous 10-Year Recursive Forecast vs. Actuals (2016–2025).
   - *Script:* `generate_faculty_graphs.py`.
   - *Description:* 120-month recursive forecast where autoregressive temperature lags are fed autonomously without observing actual post-2015 temperatures. Shows expanding analytical uncertainty cone.
6. **`06_historical_co2_vs_temperature.png` / `.pdf`**
   - *Title:* Full Observational Record (1958–2025): Atmospheric CO₂ and Global Temperature Anomaly.
   - *Script:* `generate_faculty_graphs.py`.
   - *Description:* Dual-axis historical timeline showing NOAA Mauna Loa $\text{CO}_2$ concentration against NASA GISTEMP anomaly, with train, validation, and test partitions demarcated.

### Group B: Simple Climate Model Visualizations (4 PNGs in `hector_outputs/`)
7. **`01_hector_probabilistic_plume_500yr.png`**: 500-year probabilistic temperature plume under abrupt 2×CO₂ from 5,000-member Monte Carlo ensemble with 5th, 50th, and 95th percentiles.
8. **`02_physical_scm_vs_empirical_ml_2xco2.png`**: Long-term response comparison under step forcing contrasting DOECLIM multi-century ocean thermal diffusion with empirical ML rapid stabilization.
9. **`03_ocean_heat_and_planetary_imbalance.png`**: Coupled evolution of downward ocean heat uptake flux $N(t)$ and declining top-of-atmosphere radiative planetary energy imbalance over 500 years.
10. **`04_transient_climate_response_1pct.png`**: Transient Climate Response (TCR) simulation under idealized 1%/year $\text{CO}_2$ compounding to doubling at Year 70.
- *Script:* `simulate_hector_2xco2.py`.

### Group C: Exploratory Data Analysis Visualizations (10 PNGs in `eda_outputs/`)
11. **`01_co2_over_time.png`**: Full Mauna Loa $\text{CO}_2$ record showing Keeling curve seasonal oscillation and upward secular trend.
12. **`02_temp_anomaly_over_time.png`**: Full monthly temperature anomaly time series from 1880/1958 to 2025.
13. **`03_co2_distribution.png`**: Probability density and distribution histogram of $\text{CO}_2$ measurements.
14. **`04_temp_distribution.png`**: Probability density and distribution histogram of temperature anomalies.
15. **`05_scatter_co2_vs_temp.png`**: Direct bivariate scatter plot of $\text{CO}_2$ vs temperature anomaly with linear regression fit.
16. **`06_correlation_robustness.png`**: Detrending analysis and differenced correlation verifying spurious regression risk.
17. **`07_rolling_averages.png`**: Multi-scale 12-month, 36-month, and 60-month smoothing curves revealing decadal trends.
18. **`08_decadal_trends.png`**: Decade-by-decade rate of temperature increase and $\text{CO}_2$ accumulation.
19. **`09_autocorrelation_acf.png`**: Autocorrelation (ACF) and Partial Autocorrelation (PACF) profiles.
20. **`10_unusual_observations_events.png`**: Highlight of major volcanic eruptions (Pinatubo, El Chichón) and strong El Niño events (1998, 2016, 2023–2024).
- *Script:* `scratch_make_plots.py`.

### Group D: Model Interpretability Visualizations (4 PNGs in `interpretation_outputs/`)
21. **`01_feature_importance_attribution.png`**: Standardized regression coefficient magnitudes and feature importance rankings.
22. **`02_shap_temporal_decomposition.png`**: Temporal attribution decomposing predictions into autoregressive memory vs radiative forcing.
23. **`03_era_performance_comparison.png`**: Model performance evaluation across distinct historical climate eras (1959–1975, 1976–1997, 1998–2012, 2013–2025).
24. **`04_overfitting_comparison.png`**: Train vs validation vs test error comparisons demonstrating absence of overfitting.
- *Script:* `scratch_interpretation.py`.

### Group E: Model Diagnostics Visualizations (4 PNGs in `modeling_outputs/`)
25. **`baseline_test_predictions.png`**: Baseline model out-of-sample forecast comparisons against test observations.
26. **`baseline_residual_diagnostics.png`**: Diagnostic residual plots for naive and seasonal persistence baselines.
27. **`advanced_models_test_forecasts.png`**: Forecast comparison of non-linear tree models (RF, GBDT, XGBoost) showing test boundary clipping.
28. **`advanced_models_residuals.png`**: Error distribution and residual diagnostics for tree-based ensemble models.
- *Script:* `scratch_baseline_model.py` & `scratch_advanced_models.py`.

---

## 6. How to Regenerate All Outputs

To regenerate the entire project cleanly from scratch:

```powershell
# Run the complete end-to-end pipeline in sequence:
python main.py --all

# Or run individual stages on demand:
python main.py --pipeline    # Re-train Ridge ARX model and generate metrics
python main.py --predict     # Re-generate 10-year forecast & 2026 projections
python main.py --simulate    # Re-run DOECLIM 1D ocean heat diffusion simulation
python main.py --graphs      # Re-generate all 28 visualizations (34 files)
python main.py --test        # Re-run automated test suite (pytest)
```

To view all generated visualizations interactively in any web browser without an external server:
```powershell
start GRAPH_INDEX.html
```
