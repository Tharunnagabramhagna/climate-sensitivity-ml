"""
03_prediction_residuals_timeline.py
=============================================================================
Climate Sensitivity Research Project: Investigating Climate Sensitivity Under 2xCO2
Algorithm: Ridge Regression (alpha=0.01)

Graph 3: Ridge Regression Prediction Residual Analysis (2016–2025 Test Period)
- Temporal residual error tracking with mean bias and +/-1sigma, +/-2sigma confidence bands
- Durbin-Watson autocorrelation diagnostic
- Displays in an interactive Matplotlib window (plt.show()) and saves PNG + PDF

Usage:
    python modeling_outputs/faculty_graphs/03_prediction_residuals_timeline.py
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
        from scripts.generate_faculty_graphs import generate_graph_3, get_faculty_data
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
    print("CLIMATE SENSITIVITY PROJECT — GRAPH 3: PREDICTION RESIDUALS TIMELINE")
    print("=" * 78)
    print("Diagnostics: Mean Residual, Standard Deviation, and Confidence Bands")
    print("Opening interactive Matplotlib window... (Close window to return)")
    print("-" * 78)

    generate_graph_3(test_df, show=True)
    print("[OK] Interactive window closed. High-resolution PNG and PDF updated.")


if __name__ == "__main__":
    main()
