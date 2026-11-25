"""
02_actual_vs_predicted_timeline.py
=============================================================================
Climate Sensitivity Research Project: Investigating Climate Sensitivity Under 2xCO2
Algorithm: Ridge Regression (alpha=0.01)

Graph 2: Out-of-Sample Global Temperature Anomaly Forecast Timeline (2016–2025)
- Monthly comparison of NASA GISS observations vs. Ridge 1-step predictions
- Residual error shading |y - y_hat| and historical surge annotations
- Displays in an interactive Matplotlib window (plt.show()) and saves PNG + PDF

Usage:
    python modeling_outputs/faculty_graphs/02_actual_vs_predicted_timeline.py
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
        from scripts.generate_faculty_graphs import generate_graph_2, get_faculty_data
    except ImportError as err:
        print(f"[ERROR] Failed to import project graphing module: {err}", file=sys.stderr)
        print("Ensure you are running from within the climate_sensitivity_ml repository.", file=sys.stderr)
        sys.exit(1)

    try:
        test_df, _ = get_faculty_data()
    except FileNotFoundError as err:
        print(f"[ERROR] Required dataset or model file missing:\n  {err}", file=sys.stderr)
        print("[HINT] Run 'python main.py --pipeline' to generate required model artifacts.", file=sys.stderr)
        sys.exit(1)

    print("=" * 78)
    print("CLIMATE SENSITIVITY PROJECT — GRAPH 2: ACTUAL VS PREDICTED TIMELINE")
    print("=" * 78)
    print("Timeline: 120 Monthly Observations (2016-01 to 2025-12)")
    print("Opening interactive Matplotlib window... (Close window to return)")
    print("-" * 78)

    generate_graph_2(test_df, show=True)
    print("[OK] Interactive window closed. High-resolution PNG and PDF updated.")


if __name__ == "__main__":
    main()
