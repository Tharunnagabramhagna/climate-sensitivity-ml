"""
04_residual_distribution_kde.py
=============================================================================
Climate Sensitivity Research Project: Investigating Climate Sensitivity Under 2xCO2
Algorithm: Ridge Regression (alpha=0.01)

Graph 4: Residual Distribution and Density Analysis (2016–2025)
- Histogram with empirical Kernel Density Estimate (KDE) and theoretical normal reference
- Skewness, excess kurtosis, and Shapiro-Wilk Gaussianity testing
- Displays in an interactive Matplotlib window (plt.show()) and saves PNG + PDF

Usage:
    python modeling_outputs/faculty_graphs/04_residual_distribution_kde.py
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
        from scripts.generate_faculty_graphs import generate_graph_4, get_faculty_data
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
    print("CLIMATE SENSITIVITY PROJECT — GRAPH 4: RESIDUAL DISTRIBUTION KDE")
    print("=" * 78)
    print("Density Analysis: Empirical KDE vs. Theoretical Normal Reference")
    print("Opening interactive Matplotlib window... (Close window to return)")
    print("-" * 78)

    generate_graph_4(test_df, show=True)
    print("[OK] Interactive window closed. High-resolution PNG and PDF updated.")


if __name__ == "__main__":
    main()
