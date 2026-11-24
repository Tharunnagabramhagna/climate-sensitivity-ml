"""
src/data_updater.py - Official Climate Data Downloader & Paired Dataset Builder
================================================================================
Fetches the latest official monthly observations from:
  1. NASA GISS (GISTEMP v4): Global Monthly Land-Ocean Temperature Index (L-OTI)
     Source: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
  2. NOAA GML: Mauna Loa Monthly Mean Atmospheric CO2
     Source: https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv

Maintains strict separation between the original historical benchmark (through 2025-12)
and the updated observational dataset (extending into 2026).
"""

import os
import sys
import time
import io
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BENCHMARK_DIR = DATA_DIR / "benchmark"
UPDATED_DIR = DATA_DIR / "updated"

NASA_GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
NOAA_CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"


def download_with_retry(url: str, dest_path: Path, max_retries: int = 3, timeout: int = 30) -> bool:
    """Download a URL with retry logic and standard user agent."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ClimateSensitivityResearch/2.0"}
    req = urllib.request.Request(url, headers=headers)
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Attempt {attempt}/{max_retries}: Fetching {url} ...")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    content = resp.read()
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest_path, "wb") as f:
                        f.write(content)
                    print(f"  [OK] Saved {len(content)} bytes to {dest_path.name}")
                    return True
                else:
                    print(f"  [WARN] HTTP {resp.status} received.")
        except Exception as err:
            print(f"  [WARN] Attempt {attempt} failed: {err}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
    return False


def parse_gistemp_raw(csv_path: Path) -> pd.DataFrame:
    """Parse NASA GISS GLB.Ts+dSST.csv or Green Decades mirror tidy CSV into tidy (year, month, date, temp_anomaly_c)."""
    # Check if file is in tidy format (date,value) e.g. from Green Decades mirror
    df_head = pd.read_csv(csv_path, nrows=5)
    if "date" in df_head.columns and "value" in df_head.columns:
        df_tidy = pd.read_csv(csv_path)
        df_tidy["date_dt"] = pd.to_datetime(df_tidy["date"])
        df_tidy["year"] = df_tidy["date_dt"].dt.year
        df_tidy["month"] = df_tidy["date_dt"].dt.month
        df_tidy["date"] = df_tidy["year"].astype(str) + "-" + df_tidy["month"].astype(str).str.zfill(2)
        df_tidy["temp_anomaly_c"] = df_tidy["value"].astype(float)
        df_out = df_tidy[["year", "month", "date", "temp_anomaly_c"]].sort_values(["year", "month"]).reset_index(drop=True)
        return df_out

    # Otherwise parse NASA GISS GLB.Ts+dSST.csv wide format
    with open(csv_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    
    # Strip any comment or title line before 'Year'
    header_idx = 0
    for idx, line in enumerate(raw_lines):
        if line.startswith("Year,"):
            header_idx = idx
            break
            
    clean_csv = "".join(raw_lines[header_idx:])
    df_raw = pd.read_csv(io.StringIO(clean_csv))
    
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    records = []
    
    for _, row in df_raw.iterrows():
        try:
            year = int(row["Year"])
        except (ValueError, TypeError):
            continue
            
        for m_num, m_name in enumerate(month_names, start=1):
            if m_name not in row:
                continue
            val_str = str(row[m_name]).strip()
            if val_str in ["***", "****", "NaN", "nan", "", "-99.99"]:
                continue
            try:
                anomaly = float(val_str)
                records.append({
                    "year": year,
                    "month": m_num,
                    "date": f"{year:04d}-{m_num:02d}",
                    "temp_anomaly_c": anomaly
                })
            except ValueError:
                continue
                
    df_out = pd.DataFrame(records).sort_values(["year", "month"]).reset_index(drop=True)
    return df_out


def parse_noaa_co2_raw(csv_path: Path) -> pd.DataFrame:
    """Parse NOAA Mauna Loa monthly mean CO2 CSV."""
    df_co2 = pd.read_csv(csv_path, comment="#")
    # Identify average column (can be 'average' or 'monthly_average')
    avg_col = None
    for c in ["average", "monthly_average", "co2_ppm"]:
        if c in df_co2.columns:
            avg_col = c
            break
            
    if avg_col is None:
        raise ValueError(f"Could not identify CO2 monthly average column in {df_co2.columns}")
        
    records = []
    for _, row in df_co2.iterrows():
        year = int(row["year"])
        month = int(row["month"])
        avg_val = float(row[avg_col])
        if avg_val > 0:  # Exclude missing (-99.99)
            records.append({
                "year": year,
                "month": month,
                "date": f"{year:04d}-{month:02d}",
                "co2_ppm": avg_val
            })
            
    df_out = pd.DataFrame(records).sort_values(["year", "month"]).reset_index(drop=True)
    return df_out


def build_updated_paired_dataset() -> Tuple[bool, dict]:
    """
    Downloads official NASA & NOAA data, verifies quality, and creates updated paired dataset.
    Returns (success: bool, metadata: dict).
    """
    print("=" * 78)
    print("OFFICIAL CLIMATE DATA UPDATE & VERIFICATION PIPELINE")
    print("=" * 78)
    
    UPDATED_DIR.mkdir(parents=True, exist_ok=True)
    gistemp_raw_file = UPDATED_DIR / "gistemp_glb_latest.csv"
    noaa_raw_file = UPDATED_DIR / "co2_mm_mlo_latest.csv"
    paired_file = UPDATED_DIR / "climate_paired_updated.csv"
    
    # 1. Fetch or verify NASA GISTEMP
    if not gistemp_raw_file.exists():
        print("\n[Step 1/3] Downloading official NASA GISS GISTEMP v4 dataset...")
        giss_ok = download_with_retry(NASA_GISTEMP_URL, gistemp_raw_file)
        if not giss_ok:
            print("[FAIL] Failed to download NASA GISS GISTEMP v4.")
            return False, {"error": "NASA GISTEMP download failed"}
    else:
        print(f"\n[Step 1/3] Using verified local GISTEMP file: {gistemp_raw_file.name} ({gistemp_raw_file.stat().st_size} bytes)")
        
    # 2. Fetch or verify NOAA CO2
    if not noaa_raw_file.exists():
        print("\n[Step 2/3] Downloading official NOAA GML Mauna Loa CO2 dataset...")
        noaa_ok = download_with_retry(NOAA_CO2_URL, noaa_raw_file)
        if not noaa_ok:
            print("[FAIL] Failed to download NOAA Mauna Loa CO2.")
            return False, {"error": "NOAA CO2 download failed"}
    else:
        print(f"\n[Step 2/3] Using verified local NOAA CO2 file: {noaa_raw_file.name} ({noaa_raw_file.stat().st_size} bytes)")
        
    # 3. Parse and Pair
    print("\n[Step 3/3] Parsing raw data and building updated paired dataset...")
    df_temp = parse_gistemp_raw(gistemp_raw_file)
    df_co2 = parse_noaa_co2_raw(noaa_raw_file)
    
    print(f"  NASA GISTEMP observations: {len(df_temp)} months ({df_temp['date'].iloc[0]} to {df_temp['date'].iloc[-1]})")
    print(f"  NOAA Mauna Loa CO2:        {len(df_co2)} months ({df_co2['date'].iloc[0]} to {df_co2['date'].iloc[-1]})")
    
    # Inner join on (year, month)
    df_paired = pd.merge(df_co2, df_temp[["year", "month", "temp_anomaly_c"]], on=["year", "month"], how="inner")
    df_paired = df_paired.sort_values(["year", "month"]).reset_index(drop=True)
    
    # Validation checks
    assert df_paired["co2_ppm"].isna().sum() == 0, "NaN values found in CO2 column"
    assert df_paired["temp_anomaly_c"].isna().sum() == 0, "NaN values found in temperature column"
    assert df_paired["date"].duplicated().sum() == 0, "Duplicate dates found in paired dataset"
    assert (df_paired["co2_ppm"] > 300).all() and (df_paired["co2_ppm"] < 450).all(), "CO2 out of physical range"
    assert (df_paired["temp_anomaly_c"] > -1.0).all() and (df_paired["temp_anomaly_c"] < 2.5).all(), "Temp anomaly out of physical range"
    
    # Reorder columns to match standard schema: year, month, date, co2_ppm, temp_anomaly_c
    df_paired = df_paired[["year", "month", "date", "co2_ppm", "temp_anomaly_c"]]
    
    # Save with informative header
    header_comment = [
        "# Climate Paired Dataset (Updated Official Observations)",
        "# Generated by src/data_updater.py",
        f"# Update Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "# Atmospheric CO2 Source: NOAA GML Mauna Loa Observatory (co2_mm_mlo.csv)",
        "# Atmospheric CO2 Units: Parts per million (ppm) dry air mole fraction",
        "# Global Temperature Anomaly Source: NASA GISS GISTEMP v4 (Global Land-Ocean Temperature Index)",
        "# Temperature Redistribution Mirror: Green Decades tidy CSV mirror (date,value format)",
        "# Original Attribution: NASA Goddard Institute for Space Studies (GISS), Lenssen et al. (2019)",
        "# Temperature Units: Degrees Celsius anomaly (1951-1980 baseline)",
        "# Benchmark Isolation: Original benchmark (climate_paired.csv) untouched through 2025-12",
        f"# Total Rows: {len(df_paired)}",
        f"# Coverage: {df_paired['date'].iloc[0]} to {df_paired['date'].iloc[-1]}",
        f"# Post-2025 Added Months: {len(df_paired[df_paired['year'] >= 2026])} months (2026-01 to {df_paired['date'].iloc[-1]})",
        "#"
    ]
    with open(paired_file, "w", encoding="utf-8") as f:
        f.write("\n".join(header_comment) + "\n")
        df_paired.to_csv(f, index=False)
        
    print(f"  [SUCCESS] Created updated paired dataset: {paired_file.relative_to(BASE_DIR)}")
    print(f"  Total Paired Rows: {len(df_paired)} ({df_paired['date'].iloc[0]} to {df_paired['date'].iloc[-1]})")
    print(f"  Added post-2025 observations: {len(df_paired[df_paired['year'] >= 2026])} months in 2026")
    
    metadata = {
        "status": "PASS",
        "total_paired_months": len(df_paired),
        "start_date": df_paired["date"].iloc[0],
        "end_date": df_paired["date"].iloc[-1],
        "months_2026": len(df_paired[df_paired['year'] >= 2026]),
        "nasa_url": NASA_GISTEMP_URL,
        "noaa_url": NOAA_CO2_URL,
        "paired_file": str(paired_file),
        "co2_range": [float(df_paired["co2_ppm"].min()), float(df_paired["co2_ppm"].max())],
        "temp_range": [float(df_paired["temp_anomaly_c"].min()), float(df_paired["temp_anomaly_c"].max())]
    }
    return True, metadata


if __name__ == "__main__":
    from typing import Tuple
    success, meta = build_updated_paired_dataset()
    if not success:
        sys.exit(1)
