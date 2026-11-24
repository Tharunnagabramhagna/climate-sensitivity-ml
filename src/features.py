"""
Centralized Feature Engineering Module for Climate Sensitivity Modeling.

Provides strictly leakage-free autoregressive and exogenous (ARX) feature
construction across training, prediction, and presentation pipelines.
"""

from typing import List, Tuple, Union
import numpy as np
import pandas as pd

# The 9 canonical ARX features in exact model order
FEATURE_COLS: List[str] = [
    "temp_lag1",
    "temp_lag2",
    "temp_lag3",
    "temp_lag12",
    "ln_co2_lag1",
    "co2_diff1",
    "co2_diff12",
    "sin_month",
    "cos_month",
]

TARGET_COL: str = "target_temp"


def build_climate_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """
    Constructs the 9 canonical ARX features strictly from past observations.

    Parameters:
        df: Input DataFrame containing at least 'month', 'co2_ppm', 'temp_anomaly_c'
            and preferably 'year' and 'date'.
        dropna: If True, drops initial rows with NaN resulting from 13-month lag
                initialization (drops 13 rows from 814 -> 801 valid rows).

    Returns:
        DataFrame with feature columns and target_temp aligned.
    """
    out = df.copy()

    # Ensure chronological order
    if "date" in out.columns:
        out["date_dt"] = pd.to_datetime(out["date"])
        out = out.sort_values("date_dt").reset_index(drop=True)
    elif "year" in out.columns and "month" in out.columns:
        out = out.sort_values(["year", "month"]).reset_index(drop=True)

    # Target variable: temperature anomaly at month t
    out[TARGET_COL] = out["temp_anomaly_c"]

    # Autoregressive temperature lags (strictly past: t-1, t-2, t-3, t-12)
    out["temp_lag1"] = out["temp_anomaly_c"].shift(1)
    out["temp_lag2"] = out["temp_anomaly_c"].shift(2)
    out["temp_lag3"] = out["temp_anomaly_c"].shift(3)
    out["temp_lag12"] = out["temp_anomaly_c"].shift(12)

    # Exogenous atmospheric CO2 features (strictly past: known at t-1)
    out["co2_lag1"] = out["co2_ppm"].shift(1)
    out["ln_co2_lag1"] = np.log(out["co2_lag1"])
    out["co2_diff1"] = out["co2_ppm"].shift(1) - out["co2_ppm"].shift(2)
    out["co2_diff12"] = out["co2_ppm"].shift(1) - out["co2_ppm"].shift(13)

    # Calendar harmonics (deterministic seasonal cycle phase and amplitude)
    out["sin_month"] = np.sin(2.0 * np.pi * out["month"] / 12.0)
    out["cos_month"] = np.cos(2.0 * np.pi * out["month"] / 12.0)

    if dropna:
        out = out.dropna().reset_index(drop=True)

    return out


def build_step_feature_vector(
    temp_hist: Union[List[float], np.ndarray],
    co2_hist: Union[List[float], np.ndarray],
    month: int,
) -> np.ndarray:
    """
    Constructs a single 2D feature vector of shape (1, 9) for 1-step or recursive inference.

    Parameters:
        temp_hist: Sequence of historical temperatures (must have at least 12 elements).
                   temp_hist[-1] is T_{t-1}, temp_hist[-2] is T_{t-2}, etc.
        co2_hist: Sequence of historical CO2 values (must have at least 13 elements).
                  co2_hist[-1] is C_{t-1}, co2_hist[-2] is C_{t-2}, etc.
        month: Calendar month integer (1 to 12).

    Returns:
        2D numpy array of shape (1, 9) matching FEATURE_COLS order.
    """
    if len(temp_hist) < 12:
        raise ValueError(f"temp_hist requires at least 12 observations, got {len(temp_hist)}")
    if len(co2_hist) < 13:
        raise ValueError(f"co2_hist requires at least 13 observations, got {len(co2_hist)}")

    t1 = float(temp_hist[-1])
    t2 = float(temp_hist[-2])
    t3 = float(temp_hist[-3])
    t12 = float(temp_hist[-12])

    c1 = float(co2_hist[-1])
    c2 = float(co2_hist[-2])
    c13 = float(co2_hist[-13])

    sin_m = np.sin(2.0 * np.pi * month / 12.0)
    cos_m = np.cos(2.0 * np.pi * month / 12.0)

    return np.array([[t1, t2, t3, t12, np.log(c1), c1 - c2, c1 - c13, sin_m, cos_m]])
