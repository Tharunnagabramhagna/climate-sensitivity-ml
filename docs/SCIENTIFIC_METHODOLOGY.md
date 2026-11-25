# Scientific Methodology & Technical Formulation

## Investigating Climate Sensitivity Under 2×CO₂: Empirical Machine Learning & Physical Energy-Balance Simulation

---

### 1. Research Objectives & Overview

This research examines the mathematical, statistical, and thermodynamic relationships between atmospheric carbon dioxide ($\text{CO}_2$) concentrations and global mean surface temperature anomalies. The investigation addresses four central scientific questions:

1. **Predictive Capability:** Can monthly global mean temperature anomalies be reliably predicted from historical atmospheric $\text{CO}_2$ observations and autoregressive memory using machine learning?
2. **Econometric Integrity:** Does the high empirical correlation ($r > 0.92$) between $\text{CO}_2$ and temperature reflect true causal predictive coupling or a non-stationary spurious drift (Granger–Newbold risk)?
3. **Model Architecture Comparison:** Why do regularized linear parametric models (Ridge Regression) outperform complex non-linear tree ensembles (Random Forest, XGBoost) on unprecedented climate warming regimes?
4. **Physical Thermodynamic Constraint:** Why is an energy-conserving physical Simple Climate Model (DOECLIM 1D ocean diffusion model) required to simulate multi-century equilibrium warming under $2\times\text{CO}_2$, and why does empirical machine learning fundamentally fail at this task?

---

### 2. Observational Data Provenance & Harmonization

#### 2.1 NOAA Global Monitoring Laboratory (GML) Mauna Loa $\text{CO}_2$
- **Measurement Archive:** In situ continuous non-dispersive infrared gas analyzer measurements at the Mauna Loa Observatory, Hawaii ($19.536^\circ\text{N}, 155.576^\circ\text{W}$, elevation $3397\text{ m}$).
- **Physical Variable:** Monthly mean dry-air mole fraction of carbon dioxide in units of parts per million ($\text{ppm} = \mu\text{mol CO}_2 / \text{mol dry air}$).
- **Data File:** `co2_mm_mlo.csv` (1958-03 through 2026-09, 823 raw monthly observations).
- **Quality Control & Flags:** Early records from the Scripps Institution of Oceanography (SIO, 1958–1974) contain metadata sentinel flags (`ndays = -1`, `sdev = -9.99`) denoting unrecorded daily sub-counts. The primary physical measurement column (`average`) is fully populated and calibrated to the WMO $\text{CO}_2$ X2019 mole fraction scale.

#### 2.2 NASA Goddard Institute for Space Studies (GISS) GISTEMP v4
- **Measurement Archive:** Surface temperature anomaly analysis combining land-surface air temperatures from the Global Historical Climatology Network-Monthly (GHCNm v4) and sea-surface temperatures from NOAA Extended Reconstructed Sea Surface Temperature (ERSST v5).
- **Physical Variable:** Global monthly mean surface temperature anomaly ($\Delta T$, $^\circ\text{C}$) relative to the fixed 1951–1980 base period.
- **Spatial Weighting:** $1200\text{ km}$ spatial smoothing radius with latitude-longitude gridded weighting.
- **Data File:** `gistemp_monthly.csv` (1880-01 through 2025-12, 1,752 monthly records).

#### 2.3 Paired Ground-Truth Dataset (`climate_paired.csv`)
- **Harmonization Procedure:** Exact calendar year-and-month inner join:
  $$\mathcal{D} = \{ (t, C_t, T_t) \mid \text{Year}(t) \ge 1958, \, \text{Month}(t) \in [1, 12], \, t \in \text{NOAA} \cap \text{NASA} \}$$
- **Span:** March 1958 through December 2025 ($814$ contiguous monthly observations).
- **Completeness:** Zero missing values, zero synthetic interpolations, zero duplicate timestamps.
- **2026 Data Note:** NOAA Mauna Loa $\text{CO}_2$ extends through September 2026 ($9$ months beyond NASA temperature records). These 2026 $\text{CO}_2$ observations serve as the exogenous forcing driver for out-of-sample forward statistical projections. Temperature outputs for 2026 are **statistical projections**, NOT observations.

---

### 3. Exploratory Data Analysis & Econometric Diagnostics

#### 3.1 Non-Stationarity & Spurious Correlation (The Granger–Newbold Trap)
In time-series econometrics, regressions between non-stationary integrated processes yield spuriously elevated correlation coefficients and inflated $t$-statistics:
$$C_t \sim I(1) \quad \text{and} \quad T_t \sim I(1)$$
To test for spurious correlation, we compare bivariate relationships across three mathematical representations:

1. **Raw Levels:**
   $$r(C_t, T_t) = +0.9278, \quad \rho_{\text{Spearman}} = +0.9227$$
   Both series exhibit sustained upward secular trends over 1958–2025.
2. **Quadratic Detrending:**
   Subtracting the deterministic secular acceleration ($y_t - \hat{y}_{\text{quad}}$):
   $$r(\tilde{C}_t, \tilde{T}_t) = +0.1451$$
   Removing the shared drift collapses empirical correlation by $84.4\%$.
3. **Month-over-Month First Differences:**
   $$\Delta C_t = C_t - C_{t-1}, \quad \Delta T_t = T_t - T_{t-1} \implies r(\Delta C_t, \Delta T_t) = -0.0218$$
   The high raw correlation is driven by the shared decadal trend, not by immediate monthly coupling.

#### 3.2 Autocorrelation & Memory
The temperature anomaly series exhibits high persistence ($r_{\text{lag-1}} = 0.785$, $r_{\text{lag-2}} = 0.692$, $r_{\text{lag-12}} = 0.541$). This demonstrates that atmospheric and mixed-layer ocean thermal inertia dominates monthly temperature variations, mandating autoregressive memory in any predictive model.

---

### 4. Feature Engineering & Strict Anti-Leakage Design

#### 4.1 Feature Set Specification
To predict $T_t$ (global temperature anomaly at month $t$), all features are constructed strictly from information available at or before month $t-1$:

| Feature | Mathematical Definition | Physical / Statistical Meaning |
|---|---|---|
| `temp_lag1` | $T_{t-1}$ | Immediate 1-month thermal memory / mixed-layer persistence |
| `temp_lag2` | $T_{t-2}$ | 2-month autoregressive lag |
| `temp_lag3` | $T_{t-3}$ | 3-month autoregressive lag (ENSO sub-seasonal persistence) |
| `temp_lag12` | $T_{t-12}$ | Annual lag (inter-annual memory / seasonal persistence) |
| `ln_co2_lag1` | $\ln(C_{t-1})$ | Logarithmic radiative forcing proxy ($\Delta F = \alpha \ln(C/C_0)$) |
| `co2_diff1` | $C_{t-1} - C_{t-2}$ | Monthly rate of change in atmospheric $\text{CO}_2$ |
| `co2_diff12` | $C_{t-1} - C_{t-13}$ | Annual net growth rate of $\text{CO}_2$ (removing annual cycle) |
| `sin_month` | $\sin(2\pi m_t / 12)$ | Seasonal harmonic component (annual cycle phase) |
| `cos_month` | $\cos(2\pi m_t / 12)$ | Seasonal harmonic component (annual cycle amplitude) |

- **Initialization Offset:** Because `co2_diff12` requires a 13-month lag, the first 13 months of the 814-month dataset (1958-03 to 1959-03) initialize lags. The usable feature matrix comprises **801 rows** (1959-04 to 2025-12).
- **Future Leakage Check:** $\text{Target}_t = T_t$. Feature `temp_lag1` matches $T_{t-1}$ for all $t$. Max future leakage index = 0.

#### 4.2 Chronological Partitioning
To guarantee realistic out-of-sample evaluation without temporal leakage:
- **Training Set (1959-04 to 2005-12, 561 months, 70.0%):** Historical calibration covering cooler mid-century baseline through early modern warming.
- **Validation Set (2006-01 to 2015-12, 120 months, 15.0%):** Independent hyperparameter tuning ($\alpha$ for Ridge, depth/leaves for trees) and estimation of residual standard error $\sigma_{\text{val}}$.
- **Test Set (2016-01 to 2025-12, 120 months, 15.0%):** Completely untouched out-of-sample test window covering the warmest decade in the instrumental record.

---

### 5. Empirical Machine Learning Formulation

#### 5.1 Regularized Linear Autoregressive Exogenous Model (Ridge Regression)
The Ridge Regression model optimizes the penalized sum of squared residuals:
$$\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^N \left( y_i - \beta_0 - \sum_{j=1}^P \beta_j X_{ij} \right)^2 + \alpha \sum_{j=1}^P \beta_j^2 \right\}$$
- **L2 Regularization Parameter:** Tuned over $\alpha \in [10^{-4}, 10^3]$ on the 2006–2015 validation set; optimal $\alpha = 0.01$.
- **Fitted Coefficients:**
  - $\beta(\text{temp\_lag1}) = +0.43271$
  - $\beta(\text{temp\_lag2}) = +0.25702$
  - $\beta(\text{temp\_lag3}) = +0.01064$
  - $\beta(\text{temp\_lag12}) = -0.04634$
  - $\beta(\text{ln\_co2\_lag1}) = +1.23121$
  - $\beta(\text{co2\_diff1}) = -0.00816$
  - $\beta(\text{co2\_diff12}) = +0.00954$
  - $\beta(\text{sin\_month}) = +0.01743$
  - $\beta(\text{cos\_month}) = +0.02578$
  - $\beta_0 (\text{Intercept}) = -7.12689$
- **Characteristic Equation & Stability:**
  $$1 - \phi_1 z - \phi_2 z^2 - \phi_3 z^3 - \phi_{12} z^{12} = 0$$
  All characteristic roots lie strictly outside the unit circle $|z| > 1$, guaranteeing that the autoregressive system is stationary and unconditionally mean-reverting.

#### 5.2 Tree Ensemble Failure Modes (Random Forest, XGBoost, GBDT)
Decision tree models partition feature space into orthogonal hyperplanes:
$$\hat{f}_{\text{tree}}(\mathbf{x}) = \sum_{m=1}^M c_m \mathbb{I}(\mathbf{x} \in R_m)$$
where $c_m$ is the sample mean of training targets in leaf region $R_m$.
- **Extrapolation Barrier:** If a test feature $\mathbf{x}_{\text{test}}$ falls outside the convex hull of the training set (e.g., $C_t > 410\text{ ppm}$, $T_t > 1.1^\circ\text{C}$), the tree assigns it to the boundary leaf. It cannot output a value greater than $\max(y_{\text{train}}) \approx 0.95^\circ\text{C}$.
- **Result:** Tree models fail catastrophically on the 2016–2025 test set, yielding negative $R^2$ scores ($-0.12$ to $-0.36$), while regularized linear models extrapolate continuous slopes gracefully.

---

### 6. Dual-Protocol Evaluation Framework

To evaluate forecasting skill without ambiguity, we employ two complementary evaluation protocols:

#### Protocol A: 1-Step-Ahead Teacher-Forced Forecasting
At every month $t$, true observed values $T_{t-1}, T_{t-2}, \dots$ and $C_{t-1}$ are provided:
$$\hat{T}_t = f(T_{t-1}^{\text{obs}}, T_{t-2}^{\text{obs}}, \dots, C_{t-1}^{\text{obs}})$$
- Measures pure one-month conditional expectation.
- Prevents error compounding.

#### Protocol B: 10-Year Autonomous Recursive Rollout
Starting at January 2016, only the historical observations prior to 2016-01 are provided. For all subsequent 120 months:
$$\hat{T}_t = f(\hat{T}_{t-1}, \hat{T}_{t-2}, \dots, C_{t-1}^{\text{obs}})$$
Predicted values are fed back recursively into the lag slots.
- Exogenous $\text{CO}_2$ is observed, but temperature memory is 100% autonomous.
- Measures long-term cumulative trajectory stability and error growth.

#### Comparative Baseline Models:
1. **Naive Persistence (Lag-1):** $\hat{T}_t = T_{t-1}$. In recursive mode, this freezes at $T_{\text{Dec 2015}} = 1.06^\circ\text{C}$ for all 120 months.
2. **Seasonal Persistence (Lag-12):** $\hat{T}_t = T_{t-12}$. In recursive mode, this repeats the 2015 seasonal cycle for 10 years.
3. **Climatology Historical Mean:** $\hat{T}_t = \bar{T}_{\text{train}} = +0.279^\circ\text{C}$.

---

### 7. Autoregressive Prediction Interval Formulation for 2026

#### 7.1 Wold Moving-Average Representation
For an $\text{AR}(p)$ process:
$$T_t = \sum_{k=1}^p \phi_k T_{t-k} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$
Its moving-average representation is:
$$T_t = \sum_{j=0}^\infty \psi_j \epsilon_{t-j}$$
where the impulse response weights $\psi_j$ satisfy:
$$\psi_0 = 1, \quad \psi_j = \sum_{k=1}^{\min(j, p)} \phi_k \psi_{j-k} \quad (j \ge 1)$$

#### 7.2 Cumulative Horizon Uncertainty
The $h$-step-ahead forecast error variance under recursive rollout is:
$$\text{Var}[e_t(h)] = \sigma_{\text{val}}^2 \sum_{j=0}^{h-1} \psi_j^2 \implies \sigma(h) = \sigma_{\text{val}} \sqrt{\sum_{j=0}^{h-1} \psi_j^2}$$
where $\sigma_{\text{val}} = 0.11109^\circ\text{C}$ is derived strictly from the 2006–2015 validation set residuals (zero test leakage).
The symmetric 95% confidence interval is:
$$\hat{T}_{t+h} \pm 1.96 \sigma(h)$$
For horizons $h = 1$ to $9$ (Jan–Sep 2026), $\sigma(h)$ expands monotonically:
- $h = 1$ (Jan 2026): $\sigma(1) = 0.1111^\circ\text{C} \implies \pm 0.218^\circ\text{C}$
- $h = 9$ (Sep 2026): $\sigma(9) = 0.1420^\circ\text{C} \implies \pm 0.278^\circ\text{C}$

---

### 8. Physical Climate Simulation: DOECLIM 1D Ocean Diffusion Model

To model multi-century climate sensitivity under $2\times\text{CO}_2$ ($560\text{ ppm}$), we implemented the **DOECLIM** (Diffusion Ocean Energy Balance Climate Model) framework matching the thermodynamic ocean core of **Hector v3.2.0**.

#### 8.1 Governing Partial Differential Equations
The model couples a well-mixed surface ocean/atmosphere layer to a 1D vertically diffusing deep-ocean column:

1. **Surface Mixed Layer ($z = 0$ to $h_m = 100\text{ m}$):**
   $$C_m \frac{dT_m}{dt} = F(t) - \lambda_{\text{fb}} T_m - \gamma (T_m - T_{d,0})$$
   where $C_m = \rho c_p h_m = 4.18 \times 10^8\text{ J/m}^2/\text{K}$, $\lambda_{\text{fb}} = \frac{F_{2\times}}{\text{ECS}}$ is the net radiative feedback parameter, and $\gamma = \frac{\rho c_p \kappa}{0.5 \Delta z}$ is the diffusive exchange coefficient at the mixed-layer base.
2. **Deep Ocean Interior ($k = 0, \dots, N_d-1$ layers, $H = 4000\text{ m}$, $\Delta z = 100\text{ m}$):**
   $$C_d \frac{\partial T_d}{\partial t} = \frac{\partial}{\partial z} \left( \kappa \rho c_p \frac{\partial T_d}{\partial z} \right)$$
   where $C_d = \rho c_p \Delta z = 4.18 \times 10^8\text{ J/m}^2/\text{K}$, and $\kappa$ is the vertical eddy diffusivity ($2.3 \times 10^{-4}\text{ m}^2/\text{s}$).
3. **Bottom Boundary Condition ($z = 4000\text{ m}$):**
   $$\left. \frac{\partial T_d}{\partial z} \right|_{z = -H} = 0 \quad \text{(Insulated abyssal boundary)}$$

#### 8.2 Implicit Matrix Discretization & Unconditional Stability
To prevent numerical stiffness and CFL time-step instability, the $(N_d + 1) \times (N_d + 1)$ system matrix $\mathbf{A}$ is solved using backward Euler implicit time-stepping:
$$\frac{\mathbf{T}^{n+1} - \mathbf{T}^n}{\Delta t} = \mathbf{A} \mathbf{T}^{n+1} + \mathbf{b}^n \implies (\mathbf{I} - \Delta t \mathbf{A}) \mathbf{T}^{n+1} = \mathbf{T}^n + \Delta t \mathbf{b}^n$$
$$\mathbf{T}^{n+1} = \mathbf{M}^{-1} (\mathbf{T}^n + \Delta t \mathbf{b}^n), \quad \text{where } \mathbf{M} = \mathbf{I} - \Delta t \mathbf{A}$$

#### 8.3 Energy Conservation Residual Proof
By conservation of energy, the cumulative integral of top-of-atmosphere (TOA) net radiative imbalance $N(t) = F(t) - \lambda_{\text{fb}} T_m(t)$ multiplied by ocean surface area ($A_{\text{ocean}} = 3.61 \times 10^{14}\text{ m}^2$) must equal the total ocean heat content change $\Delta\text{OHC}(t)$:
$$E_{\text{in}}(t) = \int_0^t N(t') dt' \times A_{\text{ocean}}$$
$$\Delta\text{OHC}(t) = \left( h_m T_m(t) + \Delta z \sum_{k=0}^{N_d-1} T_{d,k}(t) \right) \rho c_p A_{\text{ocean}}$$
- **Relative Conservation Residual:**
  $$\epsilon = \frac{|E_{\text{in}}(100) - \Delta\text{OHC}(100)|}{\Delta\text{OHC}(100)} \times 100\% = 5.45 \times 10^{-13}\%$$
  Energy is conserved to floating-point machine precision.

#### 8.4 Numerical Time-Step Convergence
Comparing surface temperature at Year 100 under `abrupt-2xCO2` across refined time-steps:
- $\Delta t = 1.0\text{ yr}$: $T_{100} = 1.78356^\circ\text{C}$
- $\Delta t = 0.5\text{ yr}$: $T_{100} = 1.78483^\circ\text{C}$ ($\Delta = 0.00128^\circ\text{C}$)
- $\Delta t = 0.1\text{ yr}$: $T_{100} = 1.78585^\circ\text{C}$ ($\Delta = 0.00230^\circ\text{C}$)
The solution converges with discrepancy $< 0.003^\circ\text{C}$, well within the $0.01^\circ\text{C}$ convergence tolerance.

#### 8.5 Prior Uncertainty Forward Propagation (Latin Hypercube Ensemble)
The 5,000-member Monte Carlo ensemble samples prior probability distributions matching IPCC AR6 Working Group I:
1. **ECS Prior:** Skewed normal distribution ($\text{mode} = 3.0^\circ\text{C}$, 5th–95th: $[1.93^\circ\text{C}, 4.36^\circ\text{C}]$).
2. **Vertical Diffusivity $\kappa$:** Log-normal distribution ($\text{median} = 2.31\text{ cm}^2/\text{s}$, 5th–95th: $[1.42, 3.76]$).
3. **Forcing $F_{2\times}$:** Normal distribution ($\mu = 3.94\text{ W/m}^2$, $\sigma = 0.40$).
4. **Mixed Layer Depth $h_m$:** Uniform distribution ($[60, 140]\text{ m}$).

**Scientific Note:** This ensemble constitutes **forward propagation of assumed parameter priors**, quantifying how uncertainty in physical Earth system parameters propagates forward into multi-century warming projections. It is NOT an empirical Bayesian posterior calibration against historical 1850–2025 observations.

