# Academic Presentation & Graph Interpretation Report

## Model Evaluation: Ridge Regression Under 2×CO₂ Climate Sensitivity Investigation

**Author / Candidate:** Machine Learning Engineer & Scientific Data Visualization Specialist  
**Evaluation Window:** 2016-01 to 2025-12 (120 Months Out-of-Sample Test Set)  
**Algorithm Audited:** Ridge Regression ($\alpha = 0.01$, L2-regularized ARX)  
**Output Directory:** `modeling_outputs/faculty_graphs/`  

---

### Executive Summary & Faculty Scorecard

This report accompanies the six publication-quality figures generated for faculty review. The project evaluates whether atmospheric $\text{CO}_2$ observations from **NOAA GML (Mauna Loa Observatory)** paired with surface temperature anomalies from **NASA GISS (GISTEMP v4)** can reliably predict global temperature anomalies using supervised machine learning, and critically benchmarks empirical statistical learning against physical simple climate models (DOECLIM).

```
+-----------------------------------------------------------------------------------------------------------------+
|                                     VERIFIED MODEL PERFORMANCE SCORECARD                                       |
+-------------------------------+---------------------------------------+---------------------------------------+
| Metric                        | Protocol A: 1-Step Ahead              | Protocol B: 10-Year Autonomous        |
|                               | (Teacher-Forced)                      | (Recursive Rollout)                   |
+-------------------------------+---------------------------------------+---------------------------------------+
| Root Mean Squared Error (RMSE)| 0.1124 °C                             | 0.1754 °C                             |
| Mean Absolute Error (MAE)     | 0.0897 °C                             | 0.1316 °C                             |
| Coefficient of Determination  | +0.6532 (65.3% variance explained)    | +0.1555 (Retains positive skill)      |
| Mean Residual Bias (μ)        | +0.0310 °C                            | +0.0898 °C                            |
| Residual Standard Deviation(σ)| 0.1085 °C                             | 0.1513 °C                             |
| Benchmark: Naive Persistence  | RMSE = 0.1179 °C | R² = +0.6186       | RMSE = 0.2453 °C | R² = -0.6517       |
| Benchmark: Seasonal Persist.  | RMSE = 0.2082 °C | R² = -0.1898       | RMSE = 0.2485 °C | R² = -0.6951       |
+-------------------------------+---------------------------------------+---------------------------------------+
```

---

### Graph 1: Actual vs. Predicted Temperature Anomaly (Scatter Analysis)

- **File Locations:**
  - PNG (300 DPI): `modeling_outputs/faculty_graphs/01_actual_vs_predicted_scatter.png`
  - PDF (Vector): `modeling_outputs/faculty_graphs/01_actual_vs_predicted_scatter.pdf`
- **What the Graph Represents:**
  A scatter plot comparing actual NASA GISS observed monthly global temperature anomalies ($X$-axis) against Ridge Regression predicted anomalies ($Y$-axis) across the 120-month test set (2016–2025). Points are colored chronologically from 2016 (purple) to 2025 (yellow). It includes the theoretical ideal 1:1 line ($y = x$) and the empirical linear regression best fit.
- **Dataset Used:**
  Out-of-sample test split of `climate_paired.csv` covering 2016-01 to 2025-12 ($N = 120$ months). The model was trained strictly on data prior to 2016 (1959–2005 training, 2006–2015 validation), ensuring zero test leakage.
- **Model / Method Used:**
  Ridge Regression ($\alpha = 0.01$) under **Protocol A (1-step teacher-forced)**, utilizing 9 features (4 temperature autoregressive lags, 3 $\text{CO}_2$ concentration and differential regressors, and 2 monthly seasonal harmonic terms).
- **What the Results Indicate:**
  - Predictions cluster tightly around the 1:1 diagonal line across the entire range of observed anomalies ($+0.75^\circ\text{C}$ to $+1.43^\circ\text{C}$).
  - Test $R^2 = 0.6532$ confirms that $65.3\%$ of the monthly variance in the warmest decade in recorded human history is captured by the model.
  - $\text{RMSE} = 0.1124^\circ\text{C}$ and $\text{MAE} = 0.0897^\circ\text{C}$ indicate sub-tenth-degree average forecast accuracy on monthly global anomalies.
- **Scientific Limitations That Apply:**
  1. *Teacher-Forcing Dependency:* Protocol A provides the true observed temperature of month $t-1$ to predict month $t$. Because temperature anomalies have high monthly autocorrelation ($\rho_1 \approx 0.785$), naive persistence alone achieves $R^2 = 0.6186$. Ridge's marginal advantage over naive persistence is $+0.0346$ in $R^2$.
  2. *Conflation of Short-Term Memory with Long-Term Forcing:* The high performance is primarily driven by autoregressive thermal memory ($\beta_{\text{lag1}} = 0.433, \beta_{\text{lag2}} = 0.257$), not by the immediate direct forcing of $\text{CO}_2$.

---

### Graph 2: Actual vs. Predicted Temperature Over Time (2016–2025 Timeline)

- **File Locations:**
  - PNG (300 DPI): `modeling_outputs/faculty_graphs/02_actual_vs_predicted_timeline.png`
  - PDF (Vector): `modeling_outputs/faculty_graphs/02_actual_vs_predicted_timeline.pdf`
- **What the Graph Represents:**
  A continuous 10-year monthly time-series comparison showing observed temperature anomalies (solid navy line with filled circles) versus Ridge Regression 1-step-ahead forecasts (dashed crimson line with open squares). Shaded gray regions quantify point-by-point forecast discrepancies.
- **Dataset Used:**
  Untouched test window (2016-01 to 2025-12, 120 months) from `climate_paired.csv`.
- **Model / Method Used:**
  Ridge Regression ($\alpha = 0.01$), 1-step-ahead rolling prediction using historical feature lags.
- **What the Results Indicate:**
  - The Ridge model faithfully tracks major inter-annual and decadal climate swings throughout 2016–2025, including the 2015–2016 super El Niño peak, the 2017–2018 relative plateau, the 2020–2022 triple-dip La Niña cooling, and the extreme 2023–2024 global warming spike ($+1.43^\circ\text{C}$).
  - The model does not exhibit phase lag or high-frequency numerical ringing, proving that the regularization penalty ($\alpha = 0.01$) successfully suppresses noise amplification.
- **Scientific Limitations That Apply:**
  1. *Peak Under-Prediction During Extreme Anomalies:* During the record-breaking September 2023–March 2024 anomaly spike, observed temperatures rose faster than model predictions by $\sim 0.25^\circ\text{C}$ to $0.35^\circ\text{C}$. This discrepancy occurs because unmodeled non-CO₂ forcings occurred simultaneously (e.g., the January 2022 Hunga Tonga stratospheric water vapor injection, IMO 2020 low-sulfur shipping fuel regulations reducing aerosol reflectance, and the solar cycle maximum).
  2. *Omission of Internal Modes:* The model treats ENSO events as exogenous autoregressive innovations rather than simulating coupled ocean-atmosphere wave dynamics.

---

### Graph 3: Prediction Residual Analysis (Residuals Timeline & Confidence Bands)

- **File Locations:**
  - PNG (300 DPI): `modeling_outputs/faculty_graphs/03_prediction_residuals_timeline.png`
  - PDF (Vector): `modeling_outputs/faculty_graphs/03_prediction_residuals_timeline.pdf`
- **What the Graph Represents:**
  A temporal tracking plot of prediction residuals ($e_t = T_t^{\text{obs}} - \hat{T}_t^{\text{pred}}$) across the 120-month test period, displaying the zero-error reference line, sample mean bias ($\mu_e = +0.0310^\circ\text{C}$), $\pm 1\sigma$ band ($\pm 0.1085^\circ\text{C}$, $68\%$ coverage), and $\pm 2\sigma$ band ($\pm 0.2170^\circ\text{C}$, $95\%$ coverage).
- **Dataset Used:**
  Residual series computed from test set predictions ($N = 120$, 2016-01 to 2025-12).
- **Model / Method Used:**
  Linear residual analysis and Durbin-Watson diagnostic evaluation.
- **What the Results Indicate:**
  - Homoscedasticity: The variance of residuals remains stable across the 10-year span without expanding trumpet shapes or variance explosion.
  - Coverage: $68.3\%$ of points fall within $\pm 1\sigma$, and $95.8\%$ fall within $\pm 2\sigma$, adhering closely to theoretical Gaussian expectations.
  - Durbin-Watson statistic $d = 1.621$ demonstrates that the autoregressive feature matrix successfully removes the vast majority of serial correlation from raw temperature series ($d_{\text{raw}} = 0.43$).
- **Scientific Limitations That Apply:**
  - *Small Modern Positive Bias:* The mean residual is slightly positive ($\mu_e = +0.0310^\circ\text{C}$), indicating that the test decade (2016–2025) was slightly warmer than predicted based on the 1959–2015 historical training relationship. This reflects accelerating warming rates and unmodeled aerosol reductions in recent years.

---

### Graph 4: Residual Distribution (Histogram, KDE Curve, and Gaussian Reference)

- **File Locations:**
  - PNG (300 DPI): `modeling_outputs/faculty_graphs/04_residual_distribution_kde.png`
  - PDF (Vector): `modeling_outputs/faculty_graphs/04_residual_distribution_kde.pdf`
- **What the Graph Represents:**
  The empirical probability distribution of out-of-sample prediction residuals, showing a normalized frequency histogram (18 bins), an empirical Kernel Density Estimation (KDE) curve using Silverman's rule, and a fitted theoretical Gaussian curve $\mathcal{N}(\mu, \sigma^2)$.
- **Dataset Used:**
  Test set residuals ($N = 120$).
- **Model / Method Used:**
  Non-parametric kernel density estimation (`scipy.stats.gaussian_kde`), parametric normal fitting, and the Shapiro-Wilk statistical test of normality.
- **What the Results Indicate:**
  - Mean ($\mu = +0.0310^\circ\text{C}$) and Median ($+0.0248^\circ\text{C}$) are within $0.03^\circ\text{C}$ of zero error.
  - Skewness ($+0.230$) is near zero, indicating a balanced distribution between over- and under-predictions.
  - Excess Kurtosis ($-0.070$) is nearly zero, proving absence of heavy-tailed catastrophic outliers.
  - The Shapiro-Wilk test yields $p = 0.428$ ($p > 0.05$), failing to reject the null hypothesis of normality. This confirms that the linear Ridge model's residuals satisfy classical Gauss-Markov assumptions.
- **Scientific Limitations That Apply:**
  - Normality holds for monthly aggregate global averages because spatial integration across thousands of weather stations invokes the Central Limit Theorem. This smooth Gaussian error behavior cannot be assumed for regional or daily extreme temperature projections.

---

### Graph 5: Recursive Forecast vs. Actual Values (Protocol B 10-Year Autonomous Rollout)

- **File Locations:**
  - PNG (300 DPI): `modeling_outputs/faculty_graphs/05_recursive_forecast_vs_actual.png`
  - PDF (Vector): `modeling_outputs/faculty_graphs/05_recursive_forecast_vs_actual.pdf`
- **What the Graph Represents:**
  A true out-of-sample multi-year stress test comparing actual observations against an autonomous 10-year recursive rollout (Protocol B, purple line with triangles). It contrasts Protocol B against Protocol A (1-step teacher-forced, dashed gray line) and the Naive Persistence rollout (amber dotted line, frozen at the December 2015 value).
- **Dataset Used:**
  Test window 2016-01 to 2025-12 ($120$ months). Lags are initialized exclusively from data prior to 2016-01.
- **Model / Method Used:**
  Autonomous recursive autoregression. At each monthly step, the model's own predicted temperature is fed back into `temp_lag1`, `temp_lag2`, `temp_lag3`, and `temp_lag12`, while exogenous measured $\text{CO}_2$ drives the forcing terms.
- **What the Results Indicate:**
  - Protocol B achieves $\text{RMSE} = 0.1754^\circ\text{C}$, $\text{MAE} = 0.1316^\circ\text{C}$, and $R^2 = +0.1555$.
  - While error compounds relative to 1-step teacher forcing ($R^2$ decreases from $+0.6532$ to $+0.1555$), Ridge maintains **positive predictive skill over the entire decade**, successfully tracking the upward multi-year secular warming trend ($+0.9^\circ\text{C} \to +1.2^\circ\text{C}$).
  - In stark contrast, Naive Persistence collapses to **$R^2 = -0.6517$** ($\text{RMSE} = 0.2453^\circ\text{C}$), and Seasonal Persistence collapses to **$R^2 = -0.6951$** ($\text{RMSE} = 0.2485^\circ\text{C}$).
- **Scientific Limitations That Apply:**
  1. *Variance Damping:* Under recursive rollout, autoregressive memory acts as a low-pass filter, damping monthly variance and smoothing out sub-seasonal ENSO extremes.
  2. *Accumulation of Drift:* Because errors in $\hat{T}_{t-1}$ are fed into $\hat{T}_t$, any slight structural bias accumulates over time, raising the mean residual to $+0.0898^\circ\text{C}$ by year 10.

---

### Graph 6: Historical CO₂ and Temperature Relationship (814 Months & Granger-Newbold Warning)

- **File Locations:**
  - PNG (300 DPI): `modeling_outputs/faculty_graphs/06_historical_co2_vs_temperature.png`
  - PDF (Vector): `modeling_outputs/faculty_graphs/06_historical_co2_vs_temperature.pdf`
- **What the Graph Represents:**
  A scatter plot of all 814 contiguous monthly observations of atmospheric $\text{CO}_2$ concentration ($X$-axis, ppm) versus global temperature anomaly ($Y$-axis, $^\circ\text{C}$) from March 1958 through December 2025, colored by decade from the 1950s (dark purple) to the 2020s (bright yellow), with an ordinary least squares (OLS) linear trendline.
- **Dataset Used:**
  Full 814-month historical paired ground-truth dataset (`climate_paired.csv`). Zero missing data, zero synthetic interpolations.
- **Model / Method Used:**
  Bivariate Pearson correlation analysis, linear OLS regression, and econometric non-stationarity diagnostics.
- **What the Results Indicate:**
  - The raw correlation is very high: $r = +0.9278$ ($p < 10^{-300}$), $R^2 = 0.8608$.
  - The empirical slope is $+0.01077^\circ\text{C} / \text{ppm}$, indicating an average historical warming of $\sim 1.08^\circ\text{C}$ for every $100\text{ ppm}$ increase in atmospheric $\text{CO}_2$.
- **Scientific Limitations That Apply (Essential Faculty Defense Topic):**
  1. *The Granger–Newbold Spurious Correlation Risk:* Both $\text{CO}_2$ and global temperature are non-stationary time series with persistent upward secular drifts ($I(1)$ processes). As proven by Granger & Newbold (1974), regressing two trending non-stationary series yields spuriously high $r$ and $R^2$ values that do not reflect true causal coupling.
  2. *Detrending and Differencing Collapse:* When the common quadratic acceleration trend is subtracted, correlation drops from $r = 0.9278$ to **$r = 0.1451$**. When evaluated on month-over-month first differences ($\Delta \text{CO}_2$ vs $\Delta T$), the correlation is **$r = -0.0218$** (effectively zero).
  3. *Empirical ML Cannot Deduce ECS:* Equilibrium Climate Sensitivity (ECS) represents the multi-century surface warming after doubled $\text{CO}_2$ ($560\text{ ppm}$) once deep-ocean overturning circulation reaches thermal equilibrium ($N \to 0$). Regressions on 67 years of transient historical data conflate transient climate response (TCR), ocean heat uptake, aerosol masking, and internal oscillations. Simulating ECS requires an energy-conserving physical model (such as the DOECLIM model developed in Phase 7).

---

### Summary Checklist for Faculty Presentation

When presenting these graphs to the faculty committee, emphasize:
1. **Model Distinction:** The ML algorithm is **Ridge Regression** with L2 regularization ($\alpha = 0.01$), not an unconstrained polynomial or deep neural network prone to overfitting.
2. **Protocol Integrity:** Always clearly differentiate **Protocol A (1-Step Teacher-Forced: $R^2 = 0.6532$)** from **Protocol B (10-Year Autonomous Recursive Rollout: $R^2 = 0.1555$)**. Never present 1-step skill as 10-year forecasting capability.
3. **Tree Failure:** Explain why decision tree models (Random Forest, XGBoost) fail ($R^2 < 0$) on unprecedented warming regimes due to bounded leaf partitions, whereas Ridge extrapolates continuous parametric slopes.
4. **Physical Boundary:** Make clear that empirical ML is used for short-term statistical forecasting, while the **DOECLIM 1D ocean diffusion model** (Phase 7) is strictly required to simulate multi-century $2\times\text{CO}_2$ equilibrium climate sensitivity.

