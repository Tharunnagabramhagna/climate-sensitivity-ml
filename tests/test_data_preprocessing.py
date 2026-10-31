"""
Tests for Data Loading, Integrity, and Preprocessing.
Verifies dataset sources, date continuity, zero gaps, and valid physical bounds.
"""

import os
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRED_FILE = os.path.join(PROJECT_ROOT, "climate_paired.csv")
CO2_RAW = os.path.join(PROJECT_ROOT, "co2_mm_mlo.csv")
TEMP_RAW = os.path.join(PROJECT_ROOT, "gistemp_monthly.csv")


def test_raw_files_exist():
    """Verify raw source files are present."""
    assert os.path.exists(CO2_RAW), f"Missing raw CO2 file at {CO2_RAW}"
    assert os.path.exists(TEMP_RAW), f"Missing raw GISTEMP file at {TEMP_RAW}"


def test_paired_dataset_structure():
    """Verify paired dataset existence, row count, and column schema."""
    assert os.path.exists(PAIRED_FILE), f"Missing paired file at {PAIRED_FILE}"
    df = pd.read_csv(PAIRED_FILE, comment="#")
    
    assert len(df) == 814, f"Expected 814 contiguous months, found {len(df)}"
    expected_cols = {"year", "month", "date", "co2_ppm", "temp_anomaly_c"}
    assert expected_cols.issubset(set(df.columns)), f"Missing columns in paired dataset"


def test_no_missing_or_duplicate_dates():
    """Verify zero NaN values, zero duplicate timestamps, and continuous monthly cadence."""
    df = pd.read_csv(PAIRED_FILE, comment="#")
    
    # Zero missing values
    assert df["co2_ppm"].isna().sum() == 0, "Found NaN in co2_ppm"
    assert df["temp_anomaly_c"].isna().sum() == 0, "Found NaN in temp_anomaly_c"
    
    # Zero duplicate dates
    assert df["date"].duplicated().sum() == 0, "Found duplicate date entries"
    
    # Exact date bounds
    assert df["date"].iloc[0] == "1958-03", f"Unexpected start date: {df['date'].iloc[0]}"
    assert df["date"].iloc[-1] == "2025-12", f"Unexpected end date: {df['date'].iloc[-1]}"
    
    # Monotonic 1-month increment
    df["dt"] = pd.to_datetime(df["date"])
    diffs = df["dt"].diff().dropna()
    # Month lengths vary between 28 and 31 days
    assert (diffs >= pd.Timedelta(days=28)).all(), "Time jump smaller than 28 days"
    assert (diffs <= pd.Timedelta(days=31)).all(), "Time gap greater than 31 days (missing month!)"


def test_physical_variable_ranges():
    """Verify physical ranges match realistic instrumental bounds."""
    df = pd.read_csv(PAIRED_FILE, comment="#")
    
    # Atmospheric CO2 bounds (Mauna Loa record: ~315 ppm in 1958 to ~430 ppm in 2025)
    assert df["co2_ppm"].min() >= 310.0, f"CO2 concentration too low: {df['co2_ppm'].min()}"
    assert df["co2_ppm"].max() <= 440.0, f"CO2 concentration too high: {df['co2_ppm'].max()}"
    
    # Temperature anomaly bounds (-0.6°C to +1.6°C relative to 1951-1980 baseline)
    assert df["temp_anomaly_c"].min() >= -0.60, f"Temp anomaly unreasonably low: {df['temp_anomaly_c'].min()}"
    assert df["temp_anomaly_c"].max() <= 1.60, f"Temp anomaly unreasonably high: {df['temp_anomaly_c'].max()}"

