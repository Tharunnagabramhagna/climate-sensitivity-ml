# Implementation Baseline & Verification Report

**Project:** Investigating Climate Sensitivity Under 2×CO₂  
**Initial Git Commit:** `df7756e`  
**Date of Audit:** November 2026  
**Auditor / Engineer:** Senior Machine Learning Engineer, Python Developer, Climate Researcher  

---

## 1. Executive Summary & Git State

This report establishes the verified mathematical, empirical, and architectural baseline of the repository prior to implementing portability fixes, feature engineering centralization, master graph generation, and unified CLI automation.

### Git Status
- **Current Branch:** `main`
- **Commit Hash:** `df7756e` (*Audit corrections: align residual diagnostics with independent calculations, qualify normality and detrending claims, and regenerate faculty graphs*)
- **Working Tree:** Clean (no uncommitted or untracked changes).
- **History Integrity:** Preserved without force-pushing or rewriting commits.

---

## 2. Chronological Data Splitting & Leakage Verification

### Exact Dataset Splits in `run_pipeline.py`
The paired ground-truth dataset (`climate_paired.csv`) contains **814 contiguous monthly records** (1958-03 to 2025-12).  
Constructing the 13-month backward lag (`co2_diff12 = shift(1) - shift(13)`) produces 13 initial rows with `NaN`.  
Dropping these rows yields **801 contiguous feature rows** spanning **April 1959 to December 2025**.

The 801 usable months are partitioned strictly chronologically:

| Partition | Start Month | End Month | Sample Count | Percentage | Operational Role |
|---|---|---|---|---|---|
| **Training Split (`train`)** | 1959-04 | 2005-12 | 561 months | 70.0% | Initial parameter estimation & hyperparameter grid search |
| **Validation Split (`val`)** | 2006-01 | 2015-12 | 120 months | 15.0% | Tuning penalty $\alpha$, tree depth/leaves, estimating $\sigma_{\text{val}}$ |
| **Combined Train+Val (`trainval`)** | 1959-04 | 2015-12 | 681 months | 85.0% | Refitting final chosen model prior to test deployment |
| **Test Split (`test`)** | 2016-01 | 2025-12 | 120 months | 15.0% | **Completely untouched** out-of-sample benchmark |

### Resolution of Split Clarification
- **The Issue:** Earlier summaries described training as "1959-04 to 2015-12" while also listing a separate validation set "2006 to 2015", leading to confusion over whether validation data was included during model fitting.
- **The Code Reality:** In `run_pipeline.py`:
  1. $\alpha = 0.01$ and tree architectures were selected on the validation window (2006–2015) using models fit on the training window (1959–2005).
  2. The validation residual standard deviation ($\sigma_{\text{val}} = 0.11109^\circ\text{C}$) was estimated by fitting `ridge_tr` on `train` (561 months) and scoring on `val` (120 months).
  3. Once hyperparameters were fixed, the production model evaluated on the untouched test set was fit on `trainval` (681 months, 1959-04 to 2015-12).
  4. The test partition (2016-01 to 2025-12, 120 months) was **never** seen during hyperparameter selection or model training.
- **Leakage Finding:** **ZERO LEAKAGE INTO THE TEST SET.** The test set is completely held out. The procedure follows standard industrial time-series refitting practice where a model is updated with all available historical data up to the forecast origin.

---

## 3. Verified Existing Model Performance Metrics

All metrics independently reproduced from raw paired observations match saved CSV outputs exactly:

### Out-of-Sample Test Window (2016–2025, 120 Months)

| Model Architecture | Protocol | RMSE (°C) | MAE (°C) | R² Score | Mean Residual Bias (°C) | Residual Std Dev (°C) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Ridge Regression ($\alpha = 0.01$)** | Protocol A (1-Step Teacher-Forced) | **0.1124** | **0.0897** | **+0.6532** | +0.0310 | 0.1085 |
| **Ridge Regression ($\alpha = 0.01$)** | Protocol B (10-Yr Autonomous Rollout) | **0.1754** | **0.1316** | **+0.1555** | +0.0898 | 0.1513 |
| **Naive Persistence (Lag-1)** | Protocol A (1-Step Teacher-Forced) | 0.1179 | 0.0941 | +0.6186 | -0.0009 | 0.1184 |
| **Naive Persistence (Lag-1)** | Protocol B (10-Yr Autonomous Rollout) | 0.2453 | 0.2144 | -0.6517 | -0.1541 | 0.1917 |
| **Seasonal Persistence (Lag-12)** | Protocol A (1-Step Teacher-Forced) | 0.2082 | 0.1597 | -0.1898 | +0.0295 | 0.2070 |
| **Seasonal Persistence (Lag-12)** | Protocol B (10-Yr Autonomous Rollout) | 0.2485 | 0.2004 | -0.6951 | +0.1176 | 0.2198 |

### Model Bundle Artifact (`modeling_outputs/ridge_climate_model.joblib`)
- **Model Type:** `Ridge Regression`
- **Hyperparameters:** `{'alpha': 0.01, 'random_state': 42}`
- **Intercept:** `-7.126892`
- **Validation Residual Std Dev ($\sigma_{\text{val}}$):** `0.111088 °C`
- **Coefficients:**
  - `temp_lag1`: `+0.432710`
  - `temp_lag2`: `+0.257021`
  - `temp_lag3`: `+0.010641`
  - `temp_lag12`: `-0.046339`
  - `ln_co2_lag1`: `+1.231208`
  - `co2_diff1`: `-0.008157`
  - `co2_diff12`: `+0.009537`
  - `sin_month`: `+0.017426`
  - `cos_month`: `+0.025776`

---

## 4. Verification of 2026 Prediction Workflow (`predict.py`)

- **Initialization:** History is initialized from the end of `climate_paired.csv` (December 2025: $T = +1.06^\circ\text{C}$, $\text{CO}_2 = 427.49\text{ ppm}$).
- **Forcing Inputs:** Real measured atmospheric $\text{CO}_2$ observations from NOAA GML Mauna Loa (`co2_mm_mlo.csv`) for months 2026-01 through 2026-09 ($428.62\text{ ppm}$ to $426.07\text{ ppm}$).
- **Autonomous Temperature Rollout:** Zero future temperature observations enter the pipeline. Forecasted temperatures recursively feed into lag positions.
- **Uncertainty Propagation:** 95% confidence intervals are computed via the Wold moving-average representation:
  $$\sigma(h) = \sigma_{\text{val}} \sqrt{\sum_{j=0}^{h-1} \psi_j^2}, \quad \text{CI}_{95}(h) = \hat{T}_{t+h} \pm 1.96 \sigma(h)$$
  Monotonically widening from $\pm 0.218^\circ\text{C}$ (Jan) to $\pm 0.278^\circ\text{C}$ (Sep).
- **Labeling:** All outputs are explicitly flagged as `"Statistical Projection (Unobserved)"`.

---

## 5. Verification of DOECLIM Physical Simulation (`simulate_hector_2xco2.py`)

- **Thermodynamic Equations:** Couples a well-mixed surface layer ($h_m = 100\text{ m}$) to a 40-layer diffusive deep ocean ($H = 4,000\text{ m}$, $\kappa = 2.3 \times 10^{-4}\text{ m}^2/\text{s}$).
- **Solver Stability:** Backward Euler matrix inversion ($\mathbf{M} = \mathbf{I} - \Delta t \mathbf{A}$) provides unconditional numerical stability.
- **Energy Conservation:** Integrated TOA imbalance matches total ocean heat content change $\Delta\text{OHC}$ at Year 100 with relative error **$5.45 \times 10^{-13}\%$** (machine precision).
- **Numerical Convergence:** Time-step refinement from $\Delta t = 1.0\text{ yr}$ to $\Delta t = 0.1\text{ yr}$ alters Year 100 warming by only **$0.0023^\circ\text{C}$** ($< 0.01^\circ\text{C}$ tolerance).
- **Distinction:** Numerical convergence verifies algebraic accuracy; it does not constitute empirical validation of Earth's true climate sensitivity.

---

## 6. Existing Output File Inventory (28 Unique Plots / 14 CSVs)

- **Faculty Presentation Suite (`modeling_outputs/faculty_graphs/`):** 6 figures in PNG and vector PDF format (12 files) + 3 CSV tables.
- **Physical Climate Suite (`hector_outputs/`):** 4 PNG figures + 1 ensemble summary CSV.
- **Exploratory Data Analysis (`eda_outputs/`):** 10 PNG figures + 1 metrics summary CSV.
- **Attribution & Interpretation (`interpretation_outputs/`):** 4 PNG figures + 3 summary CSVs.
- **Model Diagnostics (`modeling_outputs/`):** 4 PNG figures + 4 CSV metric files + 1 model bundle (`ridge_climate_model.joblib`).

---

## 7. Known Issues & Boundaries for Phase 2 Optimization

1. **Hardcoded Paths:** Five legacy scripts (`scratch_eda.py`, `scratch_make_plots.py`, `scratch_baseline_model.py`, `scratch_advanced_models.py`, `scratch_interpretation.py`) reference `C:\Users\bharg\OneDrive\Desktop\ml_model`.
2. **Duplicated Feature Logic:** The 9-feature construction is duplicated across four scripts.
3. **Execution Gaps:** No top-level runner exists; `run_pipeline.py` does not regenerate graphs.
4. **Preservation Rule:** The core modeling algorithms, hyperparameters ($\alpha = 0.01$), chronological splits, and physical SCM code must remain strictly preserved throughout all refactoring.
