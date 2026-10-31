"""
Generates all 10 EDA figures for Phase 3 using matplotlib.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = r"c:\Users\bharg\OneDrive\Desktop\ml_model"
DATA_FILE = os.path.join(PROJECT, "climate_paired.csv")
EDA_DIR = os.path.join(PROJECT, "eda_outputs")
os.makedirs(EDA_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE, comment="#")
df["date_dt"] = pd.to_datetime(df["date"])
df = df.sort_values("date_dt").reset_index(drop=True)

dates = df["date_dt"]
co2 = df["co2_ppm"].values
temp = df["temp_anomaly_c"].values
n = len(df)

# Styling defaults
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

# -------------------------------------------------------------
# Plot 1: CO2 concentration over time
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.plot(dates, co2, color="#1f77b4", linewidth=1.2, label="Monthly Mean CO₂")
# 12-month rolling mean
co2_12m = pd.Series(co2).rolling(12, center=True).mean()
ax.plot(dates, co2_12m, color="#d62728", linewidth=2.0, label="12-Month Centered Rolling Mean")
ax.set_title("Atmospheric CO₂ Concentration at Mauna Loa Observatory (1958–2025)", fontsize=13, fontweight="bold")
ax.set_xlabel("Year (Monthly Observations)")
ax.set_ylabel("Atmospheric CO₂ (ppm)")
ax.set_xlim(dates.iloc[0], dates.iloc[-1])
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper left")
fig.text(0.99, 0.01, "Source: NOAA Global Monitoring Laboratory | Paired Period: 1958-03 to 2025-12",
         ha="right", fontsize=8, color="#555555", style="italic")
p1_path = os.path.join(EDA_DIR, "01_co2_over_time.png")
fig.savefig(p1_path)
plt.close(fig)
print(f"Generated: {p1_path}")

# -------------------------------------------------------------
# Plot 2: Global temperature anomaly over time
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="1951–1980 Baseline (0.0°C)")
# Plot bars or thin line for monthly
colors = ["#e41a1c" if t >= 0 else "#377eb8" for t in temp]
ax.scatter(dates, temp, color=colors, s=10, alpha=0.4, label="Monthly Observations")
temp_12m = pd.Series(temp).rolling(12, center=True).mean()
ax.plot(dates, temp_12m, color="#b2182b", linewidth=2.2, label="12-Month Centered Rolling Mean")
temp_60m = pd.Series(temp).rolling(60, center=True).mean()
ax.plot(dates, temp_60m, color="#2166ac", linewidth=2.5, linestyle="-", label="60-Month (5-Year) Rolling Mean")
ax.set_title("Global Mean Surface Temperature Anomaly (1958–2025)", fontsize=13, fontweight="bold")
ax.set_xlabel("Year (Monthly Observations)")
ax.set_ylabel("Temperature Anomaly (°C, rel. to 1951–1980)")
ax.set_xlim(dates.iloc[0], dates.iloc[-1])
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper left")
fig.text(0.99, 0.01, "Source: NASA GISS GISTEMP v4 | Baseline: 1951–1980 mean",
         ha="right", fontsize=8, color="#555555", style="italic")
p2_path = os.path.join(EDA_DIR, "02_temp_anomaly_over_time.png")
fig.savefig(p2_path)
plt.close(fig)
print(f"Generated: {p2_path}")

# -------------------------------------------------------------
# Plot 3: Distribution of CO2 observations
# -------------------------------------------------------------
fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharex=True, gridspec_kw={"height_ratios": [0.2, 0.8]})
ax_box.boxplot(co2, vert=False, patch_artist=True, boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
               medianprops=dict(color="#d62728", linewidth=2))
ax_box.set_yticks([])
ax_box.set_title("Distribution of Atmospheric CO₂ Observations (1958–2025)", fontsize=13, fontweight="bold")
ax_box.grid(True, linestyle="--", alpha=0.6)

n_bins = 30
ax_hist.hist(co2, bins=n_bins, color="#1f77b4", edgecolor="black", alpha=0.7, density=True)
from scipy.stats import gaussian_kde
kde = gaussian_kde(co2)
x_grid = np.linspace(min(co2), max(co2), 200)
ax_hist.plot(x_grid, kde(x_grid), color="#d62728", linewidth=2.2, label="Kernel Density Estimate")
ax_hist.set_xlabel("Atmospheric CO₂ (ppm)")
ax_hist.set_ylabel("Probability Density")
ax_hist.legend(loc="upper right")
ax_hist.grid(True, linestyle="--", alpha=0.6)

mean_c, std_c = np.mean(co2), np.std(co2, ddof=1)
stats_str = f"N = {len(co2)}\nMean = {mean_c:.2f} ppm\nStd = {std_c:.2f} ppm\nMedian = {np.median(co2):.2f} ppm\nMin = {min(co2):.2f}\nMax = {max(co2):.2f}"
ax_hist.text(0.03, 0.95, stats_str, transform=ax_hist.transAxes, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#ccc"))
fig.text(0.99, 0.01, "Source: NOAA GML Mauna Loa Observatory | N = 814 months",
         ha="right", fontsize=8, color="#555555", style="italic")
p3_path = os.path.join(EDA_DIR, "03_co2_distribution.png")
fig.savefig(p3_path)
plt.close(fig)
print(f"Generated: {p3_path}")

# -------------------------------------------------------------
# Plot 4: Distribution of Temperature Anomalies
# -------------------------------------------------------------
fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharex=True, gridspec_kw={"height_ratios": [0.2, 0.8]})
ax_box.boxplot(temp, vert=False, patch_artist=True, boxprops=dict(facecolor="#ffbb78", color="#ff7f0e"),
               medianprops=dict(color="#d62728", linewidth=2))
ax_box.set_yticks([])
ax_box.set_title("Distribution of Global Temperature Anomalies (1958–2025)", fontsize=13, fontweight="bold")
ax_box.grid(True, linestyle="--", alpha=0.6)

ax_hist.hist(temp, bins=n_bins, color="#ff7f0e", edgecolor="black", alpha=0.7, density=True)
kde_t = gaussian_kde(temp)
xt_grid = np.linspace(min(temp), max(temp), 200)
ax_hist.plot(xt_grid, kde_t(xt_grid), color="#d62728", linewidth=2.2, label="Kernel Density Estimate")
# Normal reference
from scipy.stats import norm
ax_hist.plot(xt_grid, norm.pdf(xt_grid, np.mean(temp), np.std(temp, ddof=1)),
             color="#2ca02c", linestyle="--", linewidth=1.8, label="Normal Fit Comparison")
ax_hist.set_xlabel("Temperature Anomaly (°C, rel. to 1951–1980)")
ax_hist.set_ylabel("Probability Density")
ax_hist.legend(loc="upper right")
ax_hist.grid(True, linestyle="--", alpha=0.6)

mean_t, std_t = np.mean(temp), np.std(temp, ddof=1)
stats_str_t = f"N = {len(temp)}\nMean = {mean_t:.3f} °C\nStd = {std_t:.3f} °C\nMedian = {np.median(temp):.3f} °C\nMin = {min(temp):.2f}\nMax = {max(temp):.2f}"
ax_hist.text(0.03, 0.95, stats_str_t, transform=ax_hist.transAxes, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#ccc"))
fig.text(0.99, 0.01, "Source: NASA GISS GISTEMP v4 | Baseline: 1951–1980 mean",
         ha="right", fontsize=8, color="#555555", style="italic")
p4_path = os.path.join(EDA_DIR, "04_temp_distribution.png")
fig.savefig(p4_path)
plt.close(fig)
print(f"Generated: {p4_path}")

# -------------------------------------------------------------
# Plot 5: Scatter Plot CO2 vs Temperature Anomaly
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
scatter = ax.scatter(co2, temp, c=df["year"], cmap="viridis", s=18, alpha=0.8, edgecolors="none")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Observation Year (1958–2025)")

# Fits
poly_lin = np.polyfit(co2, temp, 1)
co2_line = np.linspace(min(co2), max(co2), 200)
ax.plot(co2_line, np.polyval(poly_lin, co2_line), color="#d62728", linewidth=2.2,
        label=f"Linear Fit (Slope: {poly_lin[0]*100:.2f} °C/100 ppm, R² = 0.861)")

# Radiative forcing fit: T = a * ln(CO2/C0) + b
ln_co2 = np.log(co2)
poly_ln = np.polyfit(ln_co2, temp, 1)
ax.plot(co2_line, poly_ln[0] * np.log(co2_line) + poly_ln[1], color="#2ca02c", linestyle="--", linewidth=2.0,
        label=f"Log Radiative Fit: ΔT ∝ ln(CO₂) (R² = 0.856)")

ax.set_title("Atmospheric CO₂ vs. Global Surface Temperature Anomaly", fontsize=13, fontweight="bold")
ax.set_xlabel("Atmospheric CO₂ Concentration (ppm) [NOAA GML]")
ax.set_ylabel("Global Temperature Anomaly (°C) [NASA GISS]")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper left")

note_text = ("Scientific Notice:\n"
             "• Pearson r = 0.928 reflects shared secular trend.\n"
             "• Does NOT prove direct instantaneous causation.\n"
             "• Cannot directly calculate ECS from raw bivariate fit.")
ax.text(0.55, 0.15, note_text, transform=ax.transAxes, fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="#fff9db", edgecolor="#e6b800", alpha=0.9))
fig.text(0.99, 0.01, "Sources: NOAA GML & NASA GISS | Period: 1958-03 to 2025-12 (814 monthly pairs)",
         ha="right", fontsize=8, color="#555555", style="italic")
p5_path = os.path.join(EDA_DIR, "05_scatter_co2_vs_temp.png")
fig.savefig(p5_path)
plt.close(fig)
print(f"Generated: {p5_path}")

# -------------------------------------------------------------
# Plot 6: Correlation Analysis & Robustness Checks
# -------------------------------------------------------------
specs = [
    "Raw Levels\n(CO₂ vs T)",
    "Log Radiative\n(ln(CO₂) vs T)",
    "Linearly\nDetrended",
    "Quadratic\nDetrended",
    "12-Month YoY\n(Δ₁₂CO₂ vs Δ₁₂T)",
    "1-Month MoM\n(ΔCO₂ vs ΔT)"
]
p_corrs = [0.9278, 0.9253, 0.4453, 0.1451, 0.1446, -0.0218]
s_corrs = [0.9227, 0.9227, 0.3776, 0.1331, 0.1286, -0.0256]

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
x_pos = np.arange(len(specs))
width = 0.35

rects1 = ax.bar(x_pos - width/2, p_corrs, width, label="Pearson r", color="#1f77b4", edgecolor="black")
rects2 = ax.bar(x_pos + width/2, s_corrs, width, label="Spearman ρ", color="#ff7f0e", edgecolor="black")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Correlation Analysis Under Systematic Detrending and Differencing", fontsize=13, fontweight="bold")
ax.set_ylabel("Correlation Coefficient")
ax.set_xticks(x_pos)
ax.set_xticklabels(specs, fontsize=9.5)
ax.set_ylim(-0.15, 1.05)
ax.grid(True, axis="y", linestyle="--", alpha=0.6)
ax.legend(loc="upper right")

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3 if h >= 0 else -10), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3 if h >= 0 else -10), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

fig.text(0.99, 0.01, "Robustness demonstration: secular trends dominate raw correlation (Granger-Newbold spurious regression risk)",
         ha="right", fontsize=8, color="#555555", style="italic")
p6_path = os.path.join(EDA_DIR, "06_correlation_robustness.png")
fig.savefig(p6_path)
plt.close(fig)
print(f"Generated: {p6_path}")

# -------------------------------------------------------------
# Plot 7: Rolling Averages (12-Month & 60-Month Smoothing)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=150, sharex=True)

ax1.plot(dates, co2, color="#9ecae1", linewidth=0.8, alpha=0.7, label="Monthly CO₂")
ax1.plot(dates, co2_12m, color="#1f77b4", linewidth=1.8, label="12-Month Rolling Mean")
co2_60m = pd.Series(co2).rolling(60, center=True).mean()
ax1.plot(dates, co2_60m, color="#08519c", linewidth=2.5, label="60-Month (5-Year) Rolling Mean")
ax1.set_ylabel("CO₂ (ppm)")
ax1.set_title("Atmospheric CO₂ & Temperature Rolling Averages (Filtering High-Frequency Noise)", fontsize=13, fontweight="bold")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(dates, temp, color="#fcbba1", linewidth=0.8, alpha=0.7, label="Monthly Temp Anomaly")
ax2.plot(dates, temp_12m, color="#de2d26", linewidth=1.8, label="12-Month Rolling Mean")
ax2.plot(dates, temp_60m, color="#a50f15", linewidth=2.5, label="60-Month (5-Year) Rolling Mean")
ax2.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
ax2.set_ylabel("Temp Anomaly (°C)")
ax2.set_xlabel("Year (1958–2025)")
ax2.legend(loc="upper left")
ax2.grid(True, linestyle="--", alpha=0.6)

fig.text(0.99, 0.01, "Sources: NOAA GML & NASA GISS | Centered rolling windows suppress seasonal cycle and ENSO noise",
         ha="right", fontsize=8, color="#555555", style="italic")
p7_path = os.path.join(EDA_DIR, "07_rolling_averages.png")
fig.savefig(p7_path)
plt.close(fig)
print(f"Generated: {p7_path}")

# -------------------------------------------------------------
# Plot 8: Long-Term Trend & Decadal Rates
# -------------------------------------------------------------
decades = [1960, 1970, 1980, 1990, 2000, 2010, 2020]
decade_labels = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020–25"]
co2_rates = []
temp_rates = []

for dec in decades:
    sub = df[(df["year"] >= dec) & (df["year"] < (dec + 10 if dec < 2020 else 2026))]
    t_sub = np.arange(len(sub)) / 12.0  # in years
    s_c = np.polyfit(t_sub, sub["co2_ppm"].values, 1)[0] * 10.0  # ppm / decade
    s_t = np.polyfit(t_sub, sub["temp_anomaly_c"].values, 1)[0] * 10.0  # °C / decade
    co2_rates.append(s_c)
    temp_rates.append(s_t)

fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)
x_d = np.arange(len(decades))

ax_c.bar(x_d, co2_rates, color="#1f77b4", edgecolor="black", width=0.55)
ax_c.set_xticks(x_d)
ax_c.set_xticklabels(decade_labels, rotation=30)
ax_c.set_ylabel("CO₂ Growth Rate (ppm / decade)")
ax_c.set_title("Decadal CO₂ Accumulation Rate", fontsize=11, fontweight="bold")
ax_c.grid(True, axis="y", linestyle="--", alpha=0.6)
for i, v in enumerate(co2_rates):
    ax_c.annotate(f"{v:.1f}", (i, v + 0.3), ha="center", fontsize=8.5, fontweight="bold")

ax_t.bar(x_d, temp_rates, color="#d62728", edgecolor="black", width=0.55)
ax_t.set_xticks(x_d)
ax_t.set_xticklabels(decade_labels, rotation=30)
ax_t.set_ylabel("Warming Rate (°C / decade)")
ax_t.set_title("Decadal Warming Rate", fontsize=11, fontweight="bold")
ax_t.grid(True, axis="y", linestyle="--", alpha=0.6)
for i, v in enumerate(temp_rates):
    ax_t.annotate(f"{v:+.2f}", (i, v + (0.01 if v >= 0 else -0.04)), ha="center", fontsize=8.5, fontweight="bold")

fig.suptitle("Evolution of Accumulation and Warming Trends Across Decades", fontsize=13, fontweight="bold")
fig.text(0.99, 0.01, "Linear trend fits computed per decade interval | Acceleration observed from 1960s to 2020s",
         ha="right", fontsize=8, color="#555555", style="italic")
p8_path = os.path.join(EDA_DIR, "08_decadal_trends.png")
fig.savefig(p8_path)
plt.close(fig)
print(f"Generated: {p8_path}")

# -------------------------------------------------------------
# Plot 9: Autocorrelation (ACF) up to Lag 36
# -------------------------------------------------------------
def calc_acf(series, max_lag=36):
    n = len(series)
    mean = np.mean(series)
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

fig, (ax_acf1, ax_acf2) = plt.subplots(2, 1, figsize=(9, 6), dpi=150, sharex=True)
lags = np.arange(37)
conf = 1.96 / np.sqrt(n)

def plot_acf_stem(ax, lags, acf_vals, title, color):
    ax.vlines(lags, 0, acf_vals, color=color, linewidth=1.5)
    ax.scatter(lags, acf_vals, color=color, s=15)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(conf, color="blue", linestyle="--", linewidth=0.9, alpha=0.7, label="95% Confidence Band (±1.96/√N)")
    ax.axhline(-conf, color="blue", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_ylim(-0.2, 1.05)
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

plot_acf_stem(ax_acf1, lags, acf_co2, "Autocorrelation Function (ACF): Atmospheric CO₂ (Lags 0–36 Months)", "#1f77b4")
ax_acf1.legend(loc="upper right", fontsize=8.5)
plot_acf_stem(ax_acf2, lags, acf_temp, "Autocorrelation Function (ACF): Global Temperature Anomaly (Lags 0–36 Months)", "#d62728")
ax_acf2.set_xlabel("Lag (Months)")

fig.suptitle("Temporal Persistence & Autocorrelation Structure", fontsize=13, fontweight="bold")
fig.text(0.99, 0.01, "Extreme persistence: CO₂ ACF(1)=0.996, Temp ACF(1)=0.947 | Requires time-series-aware ML validation",
         ha="right", fontsize=8, color="#555555", style="italic")
p9_path = os.path.join(EDA_DIR, "09_autocorrelation_acf.png")
fig.savefig(p9_path)
plt.close(fig)
print(f"Generated: {p9_path}")

# -------------------------------------------------------------
# Plot 10: Unusual Observations & Historical Climate Events
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
ax.plot(dates, temp, color="#bdbdbd", linewidth=0.9, alpha=0.8, label="Monthly Temp Anomaly")
ax.plot(dates, temp_12m, color="#636363", linewidth=1.8, label="12-Month Rolling Mean")

# Annotate notable spikes and dips
events = [
    ("1964-10", -0.31, "1964-65 La Niña / Agung Eruption (-0.31°C)", (-40, -45)),
    ("1982-04", -0.16, "El Chichón Eruption (1982)", (-60, -35)),
    ("1991-06", 0.35, "Pinatubo Eruption (1991 Cooling)", (-70, -40)),
    ("1998-02", 0.88, "1997–98 Super El Niño (+0.88°C)", (-60, 35)),
    ("2016-02", 1.37, "2015–16 Super El Niño (+1.37°C)", (-80, 35)),
    ("2023-09", 1.48, "2023–24 Historic Heat Spike (+1.48°C)", (-90, 25)),
]

for d_str, y_val, label_text, offset in events:
    match_row = df[df["date"] == d_str]
    if not match_row.empty:
        d_val = match_row["date_dt"].iloc[0]
        y_act = match_row["temp_anomaly_c"].iloc[0]
        ax.scatter([d_val], [y_act], color="#e41a1c" if y_act > 0.5 else "#377eb8", s=45, zorder=5)
        ax.annotate(label_text, xy=(d_val, y_act), xytext=offset, textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
                    fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999", alpha=0.9))

ax.set_title("Identification of Historical Climate Outliers & Extreme Events (1958–2025)", fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Global Temperature Anomaly (°C)")
ax.set_xlim(dates.iloc[0], dates.iloc[-1])
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper left")

fig.text(0.99, 0.01, "Labeled events correspond to major internal variability (ENSO) and volcanic aerosol cooling",
         ha="right", fontsize=8, color="#555555", style="italic")
p10_path = os.path.join(EDA_DIR, "10_unusual_observations_events.png")
fig.savefig(p10_path)
plt.close(fig)
print(f"Generated: {p10_path}")

print("\nAll 10 figures successfully created in:", EDA_DIR)
