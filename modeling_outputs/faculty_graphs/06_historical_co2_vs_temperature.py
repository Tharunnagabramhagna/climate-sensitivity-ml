"""
06_historical_co2_vs_temperature.py
=============================================================================
Climate Sensitivity Research Project: Investigating Climate Sensitivity Under 2xCO2
Algorithm: Ridge Regression (alpha=0.01)

Graph 6: Historical Atmospheric CO2 vs. Global Temperature Anomaly (1958–2025)
- Bivariate correlation across 814 months of NOAA GML and NASA GISS records
- Econometric critique: Granger-Newbold non-stationary spurious drift warning
- Displays in an interactive Matplotlib window (plt.show()) and saves PNG + PDF

Usage:
    python modeling_outputs/faculty_graphs/06_historical_co2_vs_temperature.py
=============================================================================
"""

import os
import sys

# Resolve project root dynamically regardless of current working directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main():
    try:
        from scripts.generate_faculty_graphs import generate_graph_6, get_faculty_data
    except ImportError as err:
        print(f"[ERROR] Failed to import project graphing module: {err}", file=sys.stderr)
        print("Ensure you are running from within the climate_sensitivity_ml repository.", file=sys.stderr)
        sys.exit(1)

    try:
        _, df_paired = get_faculty_data()
    except FileNotFoundError as err:
        print(f"[ERROR] Required dataset file missing:\n  {err}", file=sys.stderr)
        print("[HINT] Run 'python main.py --pipeline' to prepare datasets.", file=sys.stderr)
        sys.exit(1)

    print("=" * 78)
    print("CLIMATE SENSITIVITY PROJECT — GRAPH 6: HISTORICAL CO2 VS TEMPERATURE")
    print("=" * 78)
    print("Long-Term Bivariate Record: 814 Monthly Observations (1958–2025)")
    print("Opening interactive Matplotlib window... (Close window to return)")
    print("-" * 78)

    generate_graph_6(df_paired, show=True)
    print("[OK] Interactive window closed. High-resolution PNG and PDF updated.")


if __name__ == "__main__":
    main()
