# Investigating Climate Sensitivity Under 2×CO₂

**Empirical Time-Series Machine Learning and 1D Ocean-Diffusion Physical Climate Modeling (1958–2026)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 34 Passed](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## 1. Introduction

**Investigating Climate Sensitivity Under 2×CO₂** examines the empirical and thermodynamic relationship between atmospheric carbon dioxide ($\text{CO}_2$) concentrations and global mean surface temperature anomalies from 1958 through 2026.

A central distinction underpins this study: **ordinary temperature prediction** forecasts short-term monthly surface temperature variations driven strongly by recent thermal persistence, whereas **climate sensitivity investigation** asks how much global surface warming eventually occurs under sustained radiative forcing—specifically Equilibrium Climate Sensitivity (ECS), the steady-state warming following a doubling of atmospheric $\text{CO}_2$ ($560\text{ ppm}$) once the deep ocean equilibrates. Because short historical records conflate transient warming, ocean heat uptake, internal climate variability, and non-stationary trends, statistical machine learning alone cannot establish ECS. This repository pairs strictly leakage-free empirical modeling (**Ridge Regression**) with an energy-conserving 1D ocean-diffusion Simple Climate Model (**DOECLIM**) to evaluate where empirical forecasting succeeds, where it breaks down, and why physical energy-balance constraints are indispensable for multi-century projections.

---

## 2. Key Features

- **Historical Climate Dataset Processing:** Chronological alignment and integrity verification of monthly NOAA Mauna Loa $\text{CO}_2$ and NASA GISTEMP surface temperature anomaly series (814 contiguous months, 1958–2025 benchmark; extended through October 2026 in isolated `data/updated/`).
- **Regularized Empirical Modeling:** L2-regularized Ridge Regression ($\alpha = 0.01$) built on a centralized, strictly causal 9-feature autoregressive exogenous (ARX) pipeline.
- **Dual Evaluation Protocols:** Contrasts **Protocol A** (1-step-ahead teacher-forced evaluation) against **Protocol B** (10-year autonomous recursive rollout without intermediate observed temperatures).
- **DOECLIM Physical Climate Simulation:** Python 1D ocean-diffusion energy-balance solver implementing idealized `abrupt-2xCO2` and `1pctCO2` experiments deterministically and over a 5,000-member Monte Carlo prior ensemble.
- **Statistical & Residual Diagnostics:** Non-stationarity detrending/differencing checks, era-stratified attribution, normality diagnostics, and Wold-decomposition nominal prediction intervals.
- **Six Standalone Desktop Visualizations:** Native interactive Matplotlib (`tkagg`) figure scripts (`run_graph.py [1-6]`) exporting high-resolution PNG and vector PDF assets, plus an offline HTML gallery (`GRAPH_INDEX.html`) spanning 28 visualizations.
- **Reproducible End-to-End Pipeline:** Single-command CLI orchestration (`main.py --all`) backed by 34 automated unit, preprocessing, serialization, physics, and CLI tests.

---

## 3. Research Objectives

This project addresses four core research questions:

1. **Short-Term vs. Long-Horizon Empirical Skill:** How accurately can an ARX machine learning model predict monthly global temperature anomalies on an untouched 10-year test window (2016–2025) under 1-step teacher forcing compared to autonomous 10-year recursive rollout?
2. **Non-Stationarity & Spurious Correlation Risk:** To what extent does the high raw bivariate correlation between $\text{CO}_2$ and temperature ($r = 0.9278$) reflect shared secular trends (Granger–Newbold spurious regression risk) rather than direct month-to-month coupling?
3. **Extrapolation Behavior Under Unprecedented Warming:** Why do regularized parametric linear models (Ridge Regression) generalize more reliably than bounded non-linear tree ensembles (Random Forest, Gradient Boosting, XGBoost) during modern warming regimes?
4. **Statistical Forecasting vs. Equilibrium Climate Sensitivity:** Why does an empirical ARX regression plateau within ~18 months under constant $560\text{ ppm}$ forcing instead of capturing multi-century ocean thermal adjustment simulated by a physical energy-balance model?

---

## 4. Datasets and Sources

All datasets are cleanly isolated under `data/` with strict separation between the frozen historical benchmark (`data/raw/`, `data/processed/`) and the extended observational archive (`data/updated/`).

| Role | Path | Source & Provenance | Coverage | Schema & Notes |
| :--- | :--- | :--- | :---: | :--- |
| **Raw $\text{CO}_2$** | [`data/raw/co2_mm_mlo.csv`](data/raw/co2_mm_mlo.csv) | NOAA Global Monitoring Laboratory (Mauna Loa Observatory) | 1958-03 to 2026-09 | Monthly dry-air mole fraction (`ppm`), WMO X2019 scale |
| **Raw Temperature** | [`data/raw/gistemp_monthly.csv`](data/raw/gistemp_monthly.csv) | Composite CSV containing NOAA NCEI `GCAG` (1901–2000 baseline) and NASA GISS `GISTEMP` v4 (1951–1980 baseline) | 1880-01 to 2026-07 | Benchmark filters strictly for `Source == 'GISTEMP'` (ending 2025-12) |
| **Frozen Benchmark** | [`data/processed/climate_paired.csv`](data/processed/climate_paired.csv) | Exact calendar `(year, month)` inner join of NOAA $\text{CO}_2$ and NASA GISTEMP | 1958-03 to 2025-12 | 814 contiguous monthly rows; 0 missing months, 0 duplicates |
| **Updated $\text{CO}_2$** | [`data/updated/co2_mm_mlo_latest.csv`](data/updated/co2_mm_mlo_latest.csv) | NOAA GML Mauna Loa Observatory | 1958-03 to 2026-10 | 824 paired monthly observations |
| **Updated Temperature** | [`data/updated/gistemp_glb_latest.csv`](data/updated/gistemp_glb_latest.csv) | NASA GISS GISTEMP v4 data obtained via the **Green Decades** third-party mirror (`date,value` tidy format; official NASA server timed out during update) | 1880-01 to 2026-10 | 1,762 monthly rows; 1951–1980 baseline ($^\circ\text{C}$); original NASA GISS attribution preserved |
| **Updated Paired** | [`data/updated/climate_paired_updated.csv`](data/updated/climate_paired_updated.csv) | Generated by [`src/data_updater.py`](src/data_updater.py) joining updated NOAA $\text{CO}_2$ and Green Decades GISTEMP v4 mirror | 1958-03 to 2026-10 | 824 rows with embedded provenance metadata headers |

- **Benchmark Isolation:** The updated 2026 files in `data/updated/` are never mixed into historical training or test evaluation; `data/processed/climate_paired.csv` remains untouched.

---

## 5. Methodology

### 5.1 Preprocessing & Feature Engineering (`src/features.py`)
After an inner join on `(year, month)` without synthetic interpolation, `build_climate_features()` constructs **9 strictly causal input features** for target $T_t$ (`temp_anomaly_c`) using only observations available at or before month $t - 1$:
- **Autoregressive Temperature Lags (4):** `temp_lag1` ($T_{t-1}$), `temp_lag2` ($T_{t-2}$), `temp_lag3` ($T_{t-3}$), `temp_lag12` ($T_{t-12}$).
- **Exogenous $\text{CO}_2$ Forcing & Differentials (3):** `ln_co2_lag1` ($\ln \text{CO}_{2, t-1}$), `co2_diff1` ($\text{CO}_{2,t-1} - \text{CO}_{2,t-2}$), `co2_diff12` ($\text{CO}_{2,t-1} - \text{CO}_{2,t-13}$).
- **Seasonal Harmonics (2):** `sin_month` ($\sin(2\pi m / 12)$), `cos_month` ($\cos(2\pi m / 12)$).

Computing the 13-month lag (`co2_diff12`) drops the first 13 rows (`1958-03` to `1959-03`), leaving **801 contiguous feature rows** (`1959-04` to `2025-12`).

### 5.2 Chronological Splits & Ridge Regression (`src/run_pipeline.py`)
- **Train (`1959-04` to `2005-12`, 561 months):** Initial parameter fitting during hyperparameter selection.
- **Validation (`2006-01` to `2015-12`, 120 months):** Selected L2 penalty `alpha = 0.01` via lowest validation RMSE and computed the zero-leakage residual standard deviation ($\sigma_{\text{val}} = 0.1111^\circ\text{C}$).
- **Train+Val Refit (`1959-04` to `2015-12`, 681 months):** Once hyperparameters were locked, `Ridge(alpha=0.01, random_state=42)` was refit on all 681 pre-2016 months prior to evaluation and serialization (`modeling_outputs/ridge_climate_model.joblib`).
- **Test (`2016-01` to `2025-12`, 120 months):** Strictly untouched out-of-sample evaluation window.

### 5.3 Evaluation Protocols & Physical Simulation Scope
- **Protocol A (1-Step Teacher-Forced):** Predicts $T_t$ using the observed ground-truth temperature lags from $t - 1, t - 2, t - 3, t - 12$.
- **Protocol B (10-Year Recursive Rollout):** Initializes history at December 2015 and predicts 120 consecutive months (`2016-01` to `2025-12`) by feeding prior model predictions $\hat{T}_{t-k}$ back into the autoregressive lag slots while supplying observed atmospheric $\text{CO}_2$.
- **Important Methodological Boundary:** Ridge Regression empirically predicts short- and medium-term temperature anomalies based on statistical autocorrelation and historical forcing correlation. It contains no planetary energy-balance equations or ocean heat capacity terms and **does not independently estimate Equilibrium Climate Sensitivity (ECS)**. Multi-century $2\times\text{CO}_2$ behavior is investigated separately using DOECLIM in [`src/simulate_hector_2xco2.py`](src/simulate_hector_2xco2.py).

---

## 6. Model Evaluation and Results

Evaluated strictly out-of-sample on the untouched **January 2016 – December 2025** test period (120 months):

| Metric | Protocol A (1-Step Teacher-Forced) | Protocol B (10-Year Recursive Rollout) |
| :--- | :---: | :---: |
| **RMSE** | **0.1124°C** | **0.1754°C** |
| **MAE** | **0.0897°C** | **0.1316°C** |
| **$R^2$** | **0.6532** | **0.1555** |

### Why Recursive Predictions Have Lower Skill
Under Protocol A, supplying observed $T_{t-1}$ resets forecast error every month; high monthly persistence ($\rho_1 \approx 0.785$) makes 1-step prediction structurally much easier. Under Protocol B, errors from each month propagate recursively through the autoregressive coefficients ($\phi_1 \approx +0.4327, \phi_2 \approx +0.2570$) without intermediate ground-truth resets, leading to variance damping, drift from unmodeled internal modes (such as El Niño / La Niña events), and higher cumulative error.

### Baseline Comparison Context
To contextualize short-term autocorrelation versus genuine multi-year trend skill, models are also compared against **Naive Persistence (Lag-1)**:
- **Protocol A Definition ($\hat{T}_t = T_{t-1}^{\text{obs}}$):** Achieves $\text{RMSE} = 0.1179^\circ\text{C}$, $\text{MAE} = 0.0941^\circ\text{C}$, $R^2 = +0.6186$. Ridge reduces RMSE by $4.7\%$ over persistence.
- **Protocol B Definition ($\hat{T}_{t} = T_{\text{2015-12}}^{\text{obs}}$ held constant across all 120 test months):** Collapses to $\text{RMSE} = 0.2453^\circ\text{C}$, $\text{MAE} = 0.2144^\circ\text{C}$, $R^2 = -0.6517$, whereas Ridge retains positive explanatory skill ($R^2 = +0.1555$) by tracking $\ln\text{CO}_2$ forcing.
- **Tree Ensembles (Random Forest, Gradient Boosting, XGBoost):** Under Protocol A, all tested tree models record negative out-of-sample $R^2$ ($-0.1164$ to $-0.3643$) because decision trees cannot extrapolate beyond historical training target bounds ($< +1.1^\circ\text{C}$ pre-2016) into the warmer 2016–2025 decade.

---

## 7. Physical Climate Simulation (DOECLIM)

[`src/simulate_hector_2xco2.py`](src/simulate_hector_2xco2.py) implements a native Python **DOECLIM** (Diffusion Ocean Energy Balance Climate Model) 1D thermodynamic solver coupling a $100\text{ m}$ mixed surface layer to a 40-layer ($4,000\text{ m}$) diffusive deep ocean solved via Backward Euler matrix inversion ($\Delta t = 1.0\text{ yr}$):

- **Numerical Verification:** Integrated top-of-atmosphere (TOA) energy uptake matches cumulative ocean heat content change ($\Delta\text{OHC}$) with a relative residual of **$5.45 \times 10^{-13}\%$** at Year 100, and time-step refinement ($\Delta t = 1.0\text{ yr}$ vs. $0.1\text{ yr}$) converges within **$0.0023^\circ\text{C}$**.
- **Idealized $2\times\text{CO}_2$ Experiments:**
  - `abrupt-2xCO2` (instantaneous doubling from $280\text{ ppm}$ to $560\text{ ppm}$ integrated for 500 years)
  - `1pctCO2` (1%/year compounding $\text{CO}_2$ increase reaching doubling at Year 70 to diagnose Transient Climate Response, TCR)
- **Monte Carlo Prior Propagation ($N = 5,000$):** Samples assumed IPCC AR6 parameter bounds:
  - `ECS`: Skewed normal (`a=2.5, loc=2.2, scale=1.1`, clipped `[1.4, 7.0] °C`; median $\approx 3.00^\circ\text{C}$)
  - Ocean diffusivity $\kappa$: Log-normal (`mean=ln(2.3), sigma=0.30`, clipped `[0.6, 5.0] cm²/s`)
  - $2\times\text{CO}_2$ forcing $F_{2\times}$: Normal (`mean=3.93, sd=0.40`, clipped `[2.8, 5.2] W/m²`)
  - Mixed-layer depth $h_m$: Uniform (`60` to `140 m`)
- **Simulation Result & Limitations:** Under these assumed parameter priors, median warming reaches **$0.73^\circ\text{C}$ at Year 10**, **$1.58^\circ\text{C}$ at Year 70**, **$1.90^\circ\text{C}$ at Year 150**, and **$\approx 2.32^\circ\text{C}$ at Year 500** (`5th–95th percentile: [1.66°C, 3.14°C]`; Central `ECS=3.0°C` deterministic run reaches `2.38°C`), remaining below full asymptotic ECS because deep-ocean heat diffusion takes longer than 500 years to fully equilibrate. **This output reflects forward propagation of assumed priors through an idealized 1D box-diffusion model and is not an independently established empirical real-world ECS estimate.**
- **Execution & Outputs:** Run via `python main.py --simulate` or directly with `python src/simulate_hector_2xco2.py`. Saves [`hector_outputs/hector_2xco2_ensemble_summary.csv`](hector_outputs/hector_2xco2_ensemble_summary.csv), the annotated 500-year `abrupt-2xCO2` plume figure ([`PNG`](hector_outputs/01_hector_probabilistic_plume_500yr.png) \| [`PDF`](hector_outputs/01_hector_probabilistic_plume_500yr.pdf)), and companion diagnostics (`02..04.png`).
- **Contrast vs. Ridge Regression:** Under constant $560\text{ ppm}$ forcing, empirical Ridge Regression lacks deep-ocean thermal capacity and converges to a static fixed point ($+1.92^\circ\text{C}$) within ~18 months, whereas DOECLIM explicitly models energy-conserving vertical ocean heat uptake over 500+ years.

---

## 8. Visualization Gallery

All 28 project plots (35 files including vector PDFs) can be browsed offline in [`GRAPH_INDEX.html`](GRAPH_INDEX.html) (`python main.py --gallery`). The six primary faculty presentation visualizations support interactive desktop inspection (`python run_graph.py [1-6]`):

| # | Standalone Script | PNG / PDF Outputs | Scientific Purpose |
| :---: | :--- | :--- | :--- |
| **1** | [`01_actual_vs_predicted_scatter.py`](modeling_outputs/faculty_graphs/01_actual_vs_predicted_scatter.py) | [`PNG`](modeling_outputs/faculty_graphs/01_actual_vs_predicted_scatter.png) \| [`PDF`](modeling_outputs/faculty_graphs/01_actual_vs_predicted_scatter.pdf) | Out-of-sample test scatter (2016–2025) comparing observed NASA GISTEMP anomalies against Protocol A Ridge predictions with 1:1 diagonal reference and OLS fit. |
| **2** | [`02_actual_vs_predicted_timeline.py`](modeling_outputs/faculty_graphs/02_actual_vs_predicted_timeline.py) | [`PNG`](modeling_outputs/faculty_graphs/02_actual_vs_predicted_timeline.png) \| [`PDF`](modeling_outputs/faculty_graphs/02_actual_vs_predicted_timeline.pdf) | Chronological monthly test timeline (2016–2025) tracking seasonal oscillation, 2016 El Niño, and 2023–2024 warmth surge. |
| **3** | [`03_prediction_residuals_timeline.py`](modeling_outputs/faculty_graphs/03_prediction_residuals_timeline.py) | [`PNG`](modeling_outputs/faculty_graphs/03_prediction_residuals_timeline.png) \| [`PDF`](modeling_outputs/faculty_graphs/03_prediction_residuals_timeline.pdf) | Temporal error diagnostics ($e_t = T_t - \hat{T}_t$) showing mean bias ($+0.0310^\circ\text{C}$), $\pm 1 s_{\text{test}}$, $\pm 2 s_{\text{test}}$ bands, and residual autocorrelation behavior. |
| **4** | [`04_residual_distribution_kde.py`](modeling_outputs/faculty_graphs/04_residual_distribution_kde.py) | [`PNG`](modeling_outputs/faculty_graphs/04_residual_distribution_kde.png) \| [`PDF`](modeling_outputs/faculty_graphs/04_residual_distribution_kde.pdf) | Normalized histogram and empirical KDE curve compared against a theoretical Gaussian reference and Shapiro–Wilk test. |
| **5** | [`05_recursive_forecast_vs_actual.py`](modeling_outputs/faculty_graphs/05_recursive_forecast_vs_actual.py) | [`PNG`](modeling_outputs/faculty_graphs/05_recursive_forecast_vs_actual.png) \| [`PDF`](modeling_outputs/faculty_graphs/05_recursive_forecast_vs_actual.pdf) | Protocol B 10-year autonomous recursive rollout (2016–2025) contrasted against Protocol A teacher-forced forecasts and collapsed Naive Persistence. |
| **6** | [`06_historical_co2_vs_temperature.py`](modeling_outputs/faculty_graphs/06_historical_co2_vs_temperature.py) | [`PNG`](modeling_outputs/faculty_graphs/06_historical_co2_vs_temperature.png) \| [`PDF`](modeling_outputs/faculty_graphs/06_historical_co2_vs_temperature.pdf) | Full 814-month historical bivariate record (1958–2025) with explicit annotation of the Granger–Newbold spurious correlation warning. |

---

## 9. Project Structure

```text
climate_sensitivity_ml/
├── .gitignore                                  # Cache, venv, temp & OS exclusions
├── GRAPH_INDEX.html                            # Offline browser gallery for all 28 plots
├── README.md                                   # Primary project documentation
├── main.py                                     # Unified CLI workflow orchestrator
├── pytest.ini                                  # Pytest configuration
├── requirements.txt                            # Pinned Python dependencies
├── run_graph.py                                # Interactive faculty graph menu (1–6)
│
├── data/
│   ├── processed/
│   │   └── climate_paired.csv                  # Frozen 814-month paired benchmark (1958–2025)
│   ├── raw/
│   │   ├── co2_mm_mlo.csv                      # Raw NOAA Mauna Loa CO₂ record
│   │   └── gistemp_monthly.csv                 # Raw composite NOAA GCAG + NASA GISTEMP record
│   └── updated/
│       ├── climate_paired_updated.csv          # Extended paired dataset (through 2026-10)
│       ├── co2_mm_mlo_latest.csv               # Updated NOAA CO₂ record
│       └── gistemp_glb_latest.csv              # Updated GISTEMP v4 (Green Decades mirror)
│
├── docs/
│   ├── CODE_OPTIMIZATION_REPORT.md             # Code optimization & DOECLIM equivalence audit
│   ├── FINAL_WORKING_MODEL_VERIFICATION_REPORT.md # Scientific & dataset provenance audit
│   ├── GRAPH_EXECUTION_GUIDE.md                # Desktop Matplotlib launcher instructions
│   ├── MODEL_LIMITATIONS.md                    # Scientific critique & failure modes
│   ├── OUTPUTS_GUIDE.md                        # Output manifest & column schema dictionary
│   ├── REPRODUCIBILITY_GUIDE.md                # Step-by-step reproduction guide
│   └── SCIENTIFIC_METHODOLOGY.md               # Mathematical & physical formulations
│
├── eda_outputs/                                # 10 EDA plots + eda_metrics_summary.csv
├── hector_outputs/                             # 4 DOECLIM plots (PNG + Year-500 PDF) & summary CSV
├── interpretation_outputs/                     # 4 attribution/era plots + 3 summary CSVs
│
├── modeling_outputs/
│   ├── ridge_climate_model.joblib              # Serialized Ridge bundle (alpha=0.01, metrics, coefs)
│   ├── evaluation_protocols_comparison.csv     # Protocol A vs. B benchmark table
│   ├── advanced_models_test_comparison.csv     # Multi-model test benchmark
│   ├── projected_anomalies_2026.csv            # 2026 conditional projections with 95% nominal CIs
│   ├── forecast_2026_evaluation.csv            # 2026 evaluation against observed temperature
│   ├── baseline_test_metrics.csv / baseline_val_metrics.csv
│   ├── feature_importance_summary.csv
│   ├── advanced_models_residuals.png / advanced_models_test_forecasts.png
│   ├── baseline_residual_diagnostics.png / baseline_test_predictions.png
│   ├── forecast_2026_vs_actual.png
│   └── faculty_graphs/                         # 6 standalone .py scripts, 6 PNGs, 6 PDFs, 3 CSVs & report
│
├── scripts/
│   ├── __init__.py
│   ├── generate_all_graphs.py                  # Master 28-plot generator & file verifier
│   ├── generate_baseline_diagnostics.py        # Group E: Baseline diagnostic plots
│   ├── generate_eda_plots.py                   # Group C: 10 EDA figures
│   ├── generate_eda_stats.py                   # Phase 3 statistical & correlation summary script
│   ├── generate_faculty_graphs.py              # Group A: Faculty graph generation core
│   ├── generate_interpretation_plots.py        # Group D: Feature attribution & era plots
│   └── generate_model_comparison.py            # Group E: Multi-model benchmark plots
│
├── src/
│   ├── __init__.py
│   ├── data_updater.py                         # Official NOAA/NASA updater & validator
│   ├── evaluate_2026.py                        # Retrospective conditional 2026 evaluation
│   ├── features.py                             # Centralized anti-leakage ARX feature builder
│   ├── predict.py                              # Model inference & 2026 interval propagation
│   ├── run_pipeline.py                         # End-to-end training, evaluation & serialization
│   └── simulate_hector_2xco2.py                # DOECLIM 1D ocean-diffusion physical SCM
│
└── tests/
    ├── __init__.py
    ├── test_cli.py
    ├── test_data_preprocessing.py
    ├── test_feature_engineering.py
    ├── test_model_pipeline.py
    ├── test_model_serialization.py
    └── test_physical_simulation.py
```

---

## 10. Installation and Setup

### Prerequisites
- Python **3.10+** (verified on Python 3.14 on Windows)
- Git

### Windows PowerShell Setup
```powershell
# 1. Clone the repository
git clone https://github.com/Tharunnagabramhagna/climate-sensitivity-ml.git
cd climate_sensitivity_ml

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

*(On Linux/macOS, activate with `source .venv/bin/activate`.)*

---

## 11. Execution Guide

Run project workflows from the repository root using [`main.py`](main.py) or [`run_graph.py`](run_graph.py):

| Command | Action & Purpose |
| :--- | :--- |
| `python main.py --all` | Executes all 5 stages sequentially: Pipeline $\to$ Predict $\to$ Simulate $\to$ Graphs $\to$ Test. |
| `python main.py --pipeline` | Runs [`src/run_pipeline.py`](src/run_pipeline.py) to train Ridge ($\alpha=0.01$), compute Protocol A/B metrics, and save the `.joblib` bundle. |
| `python main.py --predict` | Runs [`src/predict.py`](src/predict.py) and [`src/evaluate_2026.py`](src/evaluate_2026.py) to generate 2026 conditional forecasts and evaluate them against observed data. |
| `python main.py --simulate` | Runs [`src/simulate_hector_2xco2.py`](src/simulate_hector_2xco2.py) to verify DOECLIM energy conservation/convergence and run the 500-year 5,000-member simulation. |
| `python main.py --graphs` | Runs [`scripts/generate_all_graphs.py`](scripts/generate_all_graphs.py) to regenerate and verify all 28 visualization concepts (35 PNG/PDF files). |
| `python main.py --gallery` | Opens [`GRAPH_INDEX.html`](GRAPH_INDEX.html) in the default web browser for offline visual inspection. |
| `python main.py --test` | Executes the automated pytest verification suite (`pytest -v tests/`). |
| `python run_graph.py 1` | Launches Graph 1 (`01_actual_vs_predicted_scatter.py`) in a native interactive desktop Matplotlib window (accepts `1`–`6`, or omit the number for an interactive menu). |

---

## 12. Testing and Reproducibility

Execute the full automated verification suite directly with:
```powershell
python -m pytest -v tests/
```

The test suite comprises **34 automated tests** across six modules covering CLI routing and HTML gallery links (`test_cli.py`), dataset schema and date continuity (`test_data_preprocessing.py`), lag safety and feature consistency (`test_feature_engineering.py`), Protocol A/B metric hierarchies (`test_model_pipeline.py`), joblib serialization and uncertainty propagation (`test_model_serialization.py`), and DOECLIM energy conservation, time-step convergence, and diagnostic-bypass numerical equivalence (`test_physical_simulation.py`).

---

## 13. Scientific Limitations

1. **Correlation vs. Causation:** High raw correlation between $\text{CO}_2$ and global temperature ($r = 0.9278$) drops sharply under quadratic detrending ($r = 0.1451$) and 1-month first-differencing ($r = -0.0218$). Raw correlation reflects shared non-stationary upward drift and must not be misinterpreted as direct monthly causality.
2. **Historical Trends & Non-Stationarity:** Both series behave as $I(1)$ non-stationary processes. Models fit on recent decades cannot disentangle multi-decadal greenhouse forcing from internal oceanic modes (ENSO, PDO, AMO) or intermittent volcanic/aerosol forcing.
3. **Limited Long-Term Recursive Skill:** While Ridge maintains positive $R^2$ over 10 years ($+0.1555$), recursive error compounding doubles its RMSE relative to 1-step teacher forcing and leaves multi-year internal climate variability unpredicted.
4. **Simplified Physical Model Assumptions:** DOECLIM approximates the entire global ocean as a globally uniform 1D vertical heat-diffusion column ($4,000\text{ m}$ depth, constant vertical diffusivity $\kappa$ per draw) with fixed climate feedback parameter $\lambda$, omitting spatial circulation changes, state-dependent feedbacks, and interactive carbon-cycle coupling.
5. **Uncertainty in Climate Sensitivity & Priors:** The 5,000-member Monte Carlo ensemble propagates assumed IPCC AR6 prior bounds rather than deriving an empirical Bayesian posterior conditioned on historical observations.
6. **Observational Data & Nominal Intervals:** The model excludes explicit aerosol, solar, volcanic, and ENSO indices; 2026 forecast intervals ($\pm 1.96\sigma_h$) are uncalibrated nominal analytical bounds assuming i.i.d. Gaussian innovations despite observed residual autocorrelation ($DW \approx 1.53$). See [`docs/MODEL_LIMITATIONS.md`](docs/MODEL_LIMITATIONS.md) for full details.

---

## 14. Future Scope

- **Independent Dataset Cross-Validation:** Benchmark model generalization against HadCRUT5, Berkeley Earth, and ERA5 reanalysis datasets across pre-1958 and regional latitude bands.
- **Additional Radiative Forcings & Feedbacks:** Incorporate explicit volcanic stratospheric aerosol optical depth, anthropogenic aerosol forcing, solar irradiance, and ENSO indices (e.g., ONI/MEI) into both ARX and energy-balance formulations.
- **Bayesian Posterior Uncertainty Calibration:** Condition DOECLIM's parameter sampling (ECS, $\kappa$, aerosol forcing) against observed historical ocean heat content and surface warming records via Markov Chain Monte Carlo (MCMC) rather than open-loop prior propagation.
- **Coupled Earth System Emulators:** Extend the 1D ocean-diffusion solver with coupled carbon-cycle reservoirs and interactive non-$\text{CO}_2$ greenhouse gas forcing pathways (e.g., $\text{CH}_4$, $\text{N}_2\text{O}$).

---

## 15. Documentation References

- [`docs/SCIENTIFIC_METHODOLOGY.md`](docs/SCIENTIFIC_METHODOLOGY.md) — Mathematical formulations, ARX derivations, and DOECLIM thermodynamics.
- [`docs/MODEL_LIMITATIONS.md`](docs/MODEL_LIMITATIONS.md) — Methodological critique, spurious regression warnings, and failure modes.
- [`docs/OUTPUTS_GUIDE.md`](docs/OUTPUTS_GUIDE.md) — Complete output directory manifest and CSV/model schema dictionary.
- [`docs/REPRODUCIBILITY_GUIDE.md`](docs/REPRODUCIBILITY_GUIDE.md) — Environment setup and stage-by-stage reproduction walkthrough.
- [`docs/GRAPH_EXECUTION_GUIDE.md`](docs/GRAPH_EXECUTION_GUIDE.md) — Usage guide for standalone interactive desktop graphs.
- [`docs/FINAL_WORKING_MODEL_VERIFICATION_REPORT.md`](docs/FINAL_WORKING_MODEL_VERIFICATION_REPORT.md) — Audit of Protocol A/B metrics, GCAG vs. GISTEMP disambiguation, and Green Decades dataset provenance.
- [`docs/CODE_OPTIMIZATION_REPORT.md`](docs/CODE_OPTIMIZATION_REPORT.md) — Code refactoring record and DOECLIM numerical equivalence proof.

---

## 16. Acknowledgements and Citations

- **NOAA Global Monitoring Laboratory (GML):** Lan, X., Tans, P., & Thoning, K. W. *Trends in Atmospheric Carbon Dioxide* (Mauna Loa Observatory monthly record). [https://gml.noaa.gov/ccgg/trends/](https://gml.noaa.gov/ccgg/trends/)
- **NASA Goddard Institute for Space Studies (GISS):** GISTEMP Team (2026) and Lenssen et al. (2019), *GISS Surface Temperature Analysis (GISTEMP), version 4*. [https://data.giss.nasa.gov/gistemp/](https://data.giss.nasa.gov/gistemp/)
- **Green Decades Climate Data Mirror:** Third-party redistribution mirror providing updated monthly GISTEMP v4 values used in `data/updated/gistemp_glb_latest.csv`.
- **NOAA NCEI:** Global Component of Climate at a Glance (GCAG) surface temperature series present in the raw composite temperature archive.
- **DOECLIM & Hector Simple Climate Model:** Kriegler, E. (2005), *Imprecise probability analysis of greenhouse effect and climate change* (DOECLIM formulation); Hartin, C. A., et al. (2015), *Hector v1.0*, Geoscientific Model Development, 8(4), 939–955.
- **Econometrics & Radiative Forcing:** Granger, C. W., & Newbold, P. (1974), *Spurious regressions in econometrics*, Journal of Econometrics, 2(2), 111–120; Myhre, G., et al. (1998), *New estimates of radiative forcing due to well mixed greenhouse gases*, Geophysical Research Letters, 25(14), 2715–2718.
- **IPCC Sixth Assessment Report (AR6 WGI):** Forster, P., et al. (2021), *The Earth's Energy Budget, Climate Feedbacks, and Climate Sensitivity*, Cambridge University Press.
