# Project Baseline & Technical Checkpoint

**Project:** Investigating Climate Sensitivity Under 2×CO₂  
**Repository Path:** `c:\Users\bharg\OneDrive\Desktop\ml_model`  
**Checkpoint Date:** October 31, 2026  
**Auditor & Mentor:** Senior ML Engineer, Climate Scientist & Scientific Software Developer  

---

## 1. Baseline State & Backup Confirmation

A complete, immutable backup of the project before beginning technical and scientific improvements has been created at:
`c:\Users\bharg\OneDrive\Desktop\ml_model\backup_baseline\`

### Verified File & Artifact Inventory

| Category | File Path | Size | Description / State |
| :--- | :--- | :---: | :--- |
| **Raw Dataset** | `co2_mm_mlo.csv` | 38,776 B | 823 monthly observations (1958-03 to 2026-09) from NOAA GML Mauna Loa. |
| **Raw Dataset** | `gistemp_monthly.csv` | 81,109 B | 1,752 GISTEMP monthly rows (1880-01 to 2025-12) & 2,119 GCAG rows (1850-01 to 2026-07). |
| **Paired Dataset** | `climate_paired.csv` | 23,427 B | 814 contiguous monthly pairs (1958-03 to 2025-12). Zero gaps, zero duplicates. |
| **Core Pipeline** | `run_pipeline.py` | 9,383 B | Master empirical pipeline (Phases 2–6). Trains, evaluates, and serializes Ridge model. |
| **Inference Script** | `predict.py` | 6,063 B | Standalone inference script. Loads model from disk; performs 1-step and 2026 rollouts. |
| **Physical SCM** | `simulate_hector_2xco2.py` | 24,178 B | Python DOECLIM 1D diffusion energy balance solver (500 years, 5,000 Monte Carlo runs). |
| **Model Bundle** | `modeling_outputs/ridge_climate_model.joblib` | 1,232 B | Serialized Ridge model bundle ($\alpha=0.01$) with 9 feature names and training metadata. |
| **Forecast Table** | `modeling_outputs/projected_anomalies_2026.csv` | 664 B | 2026 multi-month forward projection driven by measured NOAA CO₂. |
| **Dependencies** | `requirements.txt` | 110 B | Pinned versions: numpy 2.4.4, pandas 3.0.2, scipy 1.18.1, scikit-learn 1.9.1, xgboost 3.4.1, matplotlib 3.11.2, joblib 1.6.0. |
| **Tests** | `tests/` | 11.0 KB | 14 automated pytest cases across 4 modules. Current status: 14/14 passing. |
| **Documentation** | `README.md` | 17.5 KB | Project documentation and reproduction instructions. |

---

## 2. Established Baseline Metrics & Methodological Findings

### A. Machine Learning Test Performance (2016–2025 Untouched Decade)
1. **Teacher-Forced Evaluation (1-Step-Ahead with Observed Historical Lags):**
   - $\text{RMSE} = 0.1124^\circ\text{C}$
   - $\text{MAE} = 0.0897^\circ\text{C}$
   - $R^2 = +0.6532$
   - Outperforms Naive Persistence ($\text{RMSE} = 0.1179^\circ\text{C}$, $R^2 = +0.6186$).
2. **Autonomous Recursive Rollout (10-Year Multi-Step without Intermediate Ground Truth):**
   - $\text{RMSE} = 0.1754^\circ\text{C}$
   - $\text{MAE} = 0.1316^\circ\text{C}$
   - $R^2 = +0.1555$
   - Demonstrates that error compounds across multi-year closed-loop forecasting.

### B. 2026 Forecast Baseline
- 2026 temperature outputs are statistical autoregressive projections, not observed measurements.
- Projected slight cooling ($+1.093^\circ\text{C}$ in Jan $\to +0.981^\circ\text{C}$ in Sep 2026) is a statistical artifact of:
  - Autoregressive mean-reversion following the 2023–2024 El Niño warmth spike ($\sum \beta_{\text{AR}} \approx 0.654$).
  - Negative summer phase of seasonal harmonic regressors ($\sin/\cos$).

### C. Physical Climate Model Baseline
- The physical simulation is a **custom native Python implementation of the DOECLIM 1D diffusion energy balance equations**, representing the thermodynamic core of Hector v3. It is not the compiled C++ Hector binary.
- Numerical solver: Backward-Euler implicit matrix inversion solver $(\mathbf{I} - \Delta t \mathbf{A})^{-1}$, unconditionally stable for $\Delta t = 1.0\text{ yr}$.
- Energy conservation: Verified to floating-point precision ($< 3 \times 10^{-12}\%$ relative error over 500 years).

### D. Probabilistic Sensitivity Baseline
- The 5,000-member Monte Carlo simulation is a **forward prior propagation** of assumed IPCC AR6 parameter bounds, not a Bayesian observationally constrained posterior.

---

## 3. Git Status & Checkpoint

- Working tree is clean aside from untracked files and local environment.
- Backup folder `backup_baseline/` is established.
- Ready to proceed with incremental improvements across Phases 2 through 10.

