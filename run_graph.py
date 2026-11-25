"""
run_graph.py
=============================================================================
Climate Sensitivity Research Project: Investigating Climate Sensitivity Under 2xCO2
Interactive Graph Launcher & Menu

Launches any of the 6 faculty presentation graphs in an interactive Matplotlib
window (using TkAgg / native GUI backend).

Usage:
    python run_graph.py        # Displays interactive selection menu (1-6)
    python run_graph.py 1      # Directly opens Graph 1 (Scatter)
    python run_graph.py 2      # Directly opens Graph 2 (Timeline)
    python run_graph.py 3      # Directly opens Graph 3 (Residuals Timeline)
    python run_graph.py 4      # Directly opens Graph 4 (Residual Distribution KDE)
    python run_graph.py 5      # Directly opens Graph 5 (10-Yr Recursive Forecast)
    python run_graph.py 6      # Directly opens Graph 6 (Historical CO2 vs Temp)
=============================================================================
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "modeling_outputs", "faculty_graphs")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

GRAPH_OPTIONS = {
    "1": {
        "title": "Actual vs Predicted Scatter",
        "script": "01_actual_vs_predicted_scatter.py",
        "description": "Observed vs. predicted global temperature anomaly scatter plot with 1:1 diagonal line."
    },
    "2": {
        "title": "Actual vs Predicted Timeline",
        "script": "02_actual_vs_predicted_timeline.py",
        "description": "Monthly timeline (2016-2025) comparing NASA GISS observations with Ridge predictions."
    },
    "3": {
        "title": "Prediction Residuals Timeline",
        "script": "03_prediction_residuals_timeline.py",
        "description": "Temporal residuals timeline with mean bias and +/-1sigma, +/-2sigma confidence bands."
    },
    "4": {
        "title": "Residual Distribution KDE",
        "script": "04_residual_distribution_kde.py",
        "description": "Histogram, Kernel Density Estimate (KDE), and theoretical normal distribution."
    },
    "5": {
        "title": "Recursive Forecast vs Actual",
        "script": "05_recursive_forecast_vs_actual.py",
        "description": "10-Year autonomous multi-step recursive rollout (Protocol B) vs. actual observations."
    },
    "6": {
        "title": "Historical CO₂ vs Temperature",
        "script": "06_historical_co2_vs_temperature.py",
        "description": "Bivariate correlation across 814 months (1958-2025) with Granger-Newbold critique."
    }
}


def display_menu():
    """Prints the exact project graph menu required by specification."""
    print("CLIMATE SENSITIVITY PROJECT — GRAPH MENU\n")
    for key in sorted(GRAPH_OPTIONS.keys()):
        print(f"{key}. {GRAPH_OPTIONS[key]['title']}")
    print()


def launch_graph(choice):
    """
    Launches the selected graph script in an independent process so that
    its Matplotlib interactive window opens cleanly.
    """
    choice = str(choice).strip()
    if choice not in GRAPH_OPTIONS:
        print(f"\n[ERROR] Invalid graph selection '{choice}'.", file=sys.stderr)
        print("Please enter a valid graph number between 1 and 6.", file=sys.stderr)
        print("Example: 'python run_graph.py 1' for Actual vs Predicted Scatter.\n", file=sys.stderr)
        return False

    script_name = GRAPH_OPTIONS[choice]["script"]
    script_path = os.path.join(SCRIPTS_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"[ERROR] Graph script not found at {script_path}", file=sys.stderr)
        return False

    print(f"\n>>> Launching Graph {choice}: {GRAPH_OPTIONS[choice]['title']}...")
    try:
        proc = subprocess.run([sys.executable, script_path], check=False)
        return proc.returncode == 0
    except KeyboardInterrupt:
        print("\n[INFO] Graph display interrupted by user.")
        return True
    except Exception as exc:
        print(f"[ERROR] Failed to execute graph script: {exc}", file=sys.stderr)
        return False


def main():
    # If a command line argument was passed (e.g. 'python run_graph.py 1')
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
        success = launch_graph(choice)
        sys.exit(0 if success else 1)

    # Interactive prompt mode
    display_menu()
    try:
        user_input = input("Enter graph number (1–6): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

    if not user_input:
        print("[ERROR] No selection entered. Exiting.", file=sys.stderr)
        sys.exit(1)

    success = launch_graph(user_input)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
