"""
main.py - Unified Command-Line Interface for Climate Sensitivity ML Project
===========================================================================
Investigating Climate Sensitivity Under 2xCO2
Ridge Regression (alpha=0.01) ARX Model & DOECLIM Physical Simulation

Usage:
    python main.py --help
    python main.py --pipeline
    python main.py --predict
    python main.py --simulate
    python main.py --graphs
    python main.py --test
    python main.py --all
"""

import sys
import os
import time
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODELING_DIR = BASE_DIR / "modeling_outputs"
MODEL_FILE = MODELING_DIR / "ridge_climate_model.joblib"
DATA_DIR = BASE_DIR / "data"
DATASET_FILE = DATA_DIR / "processed" / "climate_paired.csv"
CO2_FILE = DATA_DIR / "raw" / "co2_mm_mlo.csv"
TEMP_FILE = DATA_DIR / "raw" / "gistemp_monthly.csv"


DATASET_METADATA = {
    "climate_paired.csv": {
        "desc": "Chronologically joined monthly dataset merging NOAA CO2 with NASA GISTEMP temperature anomalies (1958-03 to 2025-12).",
        "why": "Central tabular dataset feeding feature engineering pipeline (src/features.py) for all modeling protocols.",
        "url": "Generated internally by joining NOAA GML Mauna Loa CO2 and NASA GISTEMP observations.",
        "expected_location": "data/processed/climate_paired.csv",
        "verify": "Contains 814 monthly rows from 1958-03 to 2025-12 with columns: year, month, date, co2_ppm, temp_anomaly_c."
    },
    "co2_mm_mlo.csv": {
        "desc": "NOAA Global Monitoring Laboratory (GML) Mauna Loa monthly mean atmospheric CO2 mole fraction in dry air.",
        "why": "Primary exogenous radiative forcing input for regression models, diff features, and 2026 conditional forecasting.",
        "url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv",
        "expected_location": "data/raw/co2_mm_mlo.csv",
        "verify": "Contains comment header lines starting with '#' followed by columns: year, month, decimal date, average, deseasonalized."
    },
    "gistemp_monthly.csv": {
        "desc": "NASA GISS Surface Temperature Analysis (GISTEMP v4) monthly global surface temperature anomaly record.",
        "why": "Primary ground-truth target variable (temp_anomaly_c) relative to the 1951-1980 baseline.",
        "url": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
        "expected_location": "data/raw/gistemp_monthly.csv",
        "verify": "Contains monthly temperature anomalies in degrees Celsius. Check for 'GISTEMP' and 'GCAG' source identifiers."
    }
}


def check_raw_datasets():
    """Verify raw datasets exist before running pipeline or stages with detailed Phase 8 failure diagnostics."""
    missing = []
    for path, name in [
        (DATASET_FILE, "climate_paired.csv"),
        (CO2_FILE, "co2_mm_mlo.csv"),
        (TEMP_FILE, "gistemp_monthly.csv")
    ]:
        if not path.is_file():
            missing.append((name, path))

    if missing:
        print("\n" + "!" * 78)
        print("DATASET VALIDATION ERROR: Required input dataset file(s) are missing!")
        print("!" * 78)
        for name, path in missing:
            meta = DATASET_METADATA.get(name, {})
            print(f"\nMISSING FILE: {name}")
            print(f"  • Path Checked:        {path}")
            print(f"  • Data Description:    {meta.get('desc', 'N/A')}")
            print(f"  • Why It Is Required:  {meta.get('why', 'N/A')}")
            print(f"  • Official URL:        {meta.get('url', 'N/A')}")
            print(f"  • Expected Location:   {meta.get('expected_location', 'N/A')}")
            print(f"  • Verification Check:  {meta.get('verify', 'N/A')}")

        print("\nRECOVERY INSTRUCTIONS:")
        print("  1. Restore missing historical benchmark files into data/raw/ and data/processed/ from Git history if deleted.")
        print("  2. To attempt fetching fresh official observations into data/updated/, run:")
        print("     python main.py --update-data")
        print("  3. For manual download, follow the official URLs above and save files into data/raw/.")
        print("!" * 78 + "\n")
        return False
    return True


def run_stage(stage_name, cmd_args, description, check_artifacts_before=None, check_artifacts_after=None):
    """
    Execute a project stage as a subprocess using sys.executable.
    Returns (status: bool, elapsed_seconds: float, key_outputs: list).
    """
    print("=" * 78)
    print(f"STAGE: {stage_name}")
    print(f"Description: {description}")
    print(f"Executing: {' '.join(str(c) for c in cmd_args)}")
    print(f"Start time:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 78)

    # Check pre-requisite artifacts if specified
    if check_artifacts_before:
        for art_path, error_msg in check_artifacts_before:
            if not art_path.exists():
                print(f"[ERROR] Prerequisite missing: {art_path}")
                print(f"  -> {error_msg}")
                return False, 0.0, []

    start_time = time.time()
    try:
        # Run command with unbuffered Python output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(cmd_args, cwd=str(BASE_DIR), env=env)
        elapsed = time.time() - start_time
        success = (result.returncode == 0)
    except Exception as exc:
        elapsed = time.time() - start_time
        print(f"[ERROR] Subprocess invocation failed: {exc}")
        return False, elapsed, []

    print("-" * 78)
    status_str = "PASS" if success else f"FAIL (exit code {result.returncode})"
    print(f"Stage status: {status_str}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

    found_outputs = []
    if check_artifacts_after:
        print("Key generated outputs:")
        for out_path, desc in check_artifacts_after:
            if out_path.exists():
                size_kb = out_path.stat().st_size / 1024.0
                print(f"  [OK] {out_path.relative_to(BASE_DIR)} ({size_kb:.1f} KB) - {desc}")
                found_outputs.append(str(out_path.relative_to(BASE_DIR)))
            else:
                print(f"  [MISSING] {out_path.relative_to(BASE_DIR)} - {desc}")

    print("=" * 78 + "\n")
    return success, elapsed, found_outputs


def execute_pipeline():
    if not check_raw_datasets():
        return False, 0.0, []
    cmd = [sys.executable, str(BASE_DIR / "src" / "run_pipeline.py")]
    outputs = [
        (MODEL_FILE, "Trained Ridge Climate Model Artifact (.joblib)"),
        (MODELING_DIR / "evaluation_protocols_comparison.csv", "Protocol Comparison CSV (Protocol A vs B)"),
        (MODELING_DIR / "advanced_models_test_comparison.csv", "Test Evaluation Comparison Table CSV"),
    ]
    return run_stage("Pipeline Training & Evaluation", cmd, "Train Ridge ARX, evaluate Protocol A & B, save model", check_artifacts_after=outputs)


def execute_predict():
    if not check_raw_datasets():
        return False, 0.0, []
    prereqs = [
        (MODEL_FILE, "Trained model missing. Please run 'python main.py --pipeline' first.")
    ]
    cmd = [sys.executable, str(BASE_DIR / "src" / "predict.py")]
    outputs = [
        (MODELING_DIR / "projected_anomalies_2026.csv", "2026 Projections with Analytical Error Bounds CSV"),
    ]
    success, elapsed, found_outputs = run_stage(
        "Autonomous Forecasting & 2026 Projections",
        cmd,
        "Generate 2026 predictions and recursive uncertainty intervals",
        check_artifacts_before=prereqs,
        check_artifacts_after=outputs
    )
    if success:
        eval_script = BASE_DIR / "src" / "evaluate_2026.py"
        if eval_script.exists():
            print("\nEvaluating 2026 projections against available observations...")
            subprocess.run([sys.executable, str(eval_script)], cwd=str(BASE_DIR))
            eval_csv = MODELING_DIR / "forecast_2026_evaluation.csv"
            eval_plot = MODELING_DIR / "forecast_2026_vs_actual.png"
            if eval_csv.exists():
                found_outputs.append(str(eval_csv.relative_to(BASE_DIR)))
            if eval_plot.exists():
                found_outputs.append(str(eval_plot.relative_to(BASE_DIR)))
    return success, elapsed, found_outputs


def execute_update_data():
    updater_script = BASE_DIR / "src" / "data_updater.py"
    cmd = [sys.executable, str(updater_script)]
    outputs = [
        (BASE_DIR / "data" / "updated" / "co2_mm_mlo_latest.csv", "Latest Official NOAA Mauna Loa CO2 CSV"),
    ]
    return run_stage("Official Climate Data Update", cmd, "Fetch latest official NOAA and NASA observations into data/updated/", check_artifacts_after=outputs)


def execute_simulate():
    cmd = [sys.executable, str(BASE_DIR / "src" / "simulate_hector_2xco2.py")]
    outputs = [
        (BASE_DIR / "hector_outputs" / "hector_2xco2_ensemble_summary.csv", "DOECLIM 2xCO2 Ensemble Summary CSV"),
        (BASE_DIR / "hector_outputs" / "01_hector_probabilistic_plume_500yr.png", "DOECLIM 500-Yr Simulation Plume PNG"),
        (BASE_DIR / "hector_outputs" / "01_hector_probabilistic_plume_500yr.pdf", "DOECLIM 500-Yr Simulation Plume PDF"),
    ]
    return run_stage("DOECLIM Physical Climate Simulation", cmd, "1D Ocean Heat Diffusion & Energy Balance Simulation under 2xCO2", check_artifacts_after=outputs)


def open_gallery():
    """
    Conveniently opens GRAPH_INDEX.html in the default web browser.
    Guarded so headless environments or automated runs do not crash or stall.
    """
    gallery_file = BASE_DIR / "GRAPH_INDEX.html"
    if not gallery_file.exists():
        print(f"[ERROR] Graph gallery file not found: {gallery_file}")
        return False
    print(f"\nOpening visual graph gallery: {gallery_file}")
    try:
        import webbrowser
        opened = webbrowser.open(gallery_file.resolve().as_uri())
        if opened:
            print("[OK] Interactive graph gallery opened in your default browser.")
        else:
            print("[INFO] Web browser could not be launched directly (headless/no desktop environment).")
            print(f"       You can view the gallery directly: {gallery_file.resolve().as_uri()}")
    except Exception as exc:
        print(f"[INFO] Automatic browser launch skipped ({exc}).")
        print(f"       You can view the gallery directly: {gallery_file.resolve().as_uri()}")
    return True


def execute_graphs(open_gallery_after=False, skip_scm=False):
    cmd = [sys.executable, str(BASE_DIR / "scripts" / "generate_all_graphs.py")]
    if skip_scm:
        cmd.append("--skip-scm")
    faculty_dir = MODELING_DIR / "faculty_graphs"
    outputs = [
        (faculty_dir / "01_actual_vs_predicted_scatter.png", "Faculty Fig 1 (Scatter PNG)"),
        (faculty_dir / "01_actual_vs_predicted_scatter.pdf", "Faculty Fig 1 (Scatter PDF)"),
        (faculty_dir / "02_actual_vs_predicted_timeline.png", "Faculty Fig 2 (Timeline PNG)"),
        (faculty_dir / "02_actual_vs_predicted_timeline.pdf", "Faculty Fig 2 (Timeline PDF)"),
        (faculty_dir / "03_prediction_residuals_timeline.png", "Faculty Fig 3 (Residuals PNG)"),
        (faculty_dir / "03_prediction_residuals_timeline.pdf", "Faculty Fig 3 (Residuals PDF)"),
        (faculty_dir / "04_residual_distribution_kde.png", "Faculty Fig 4 (KDE Distribution PNG)"),
        (faculty_dir / "04_residual_distribution_kde.pdf", "Faculty Fig 4 (KDE Distribution PDF)"),
        (faculty_dir / "05_recursive_forecast_vs_actual.png", "Faculty Fig 5 (10-Yr Recursive PNG)"),
        (faculty_dir / "05_recursive_forecast_vs_actual.pdf", "Faculty Fig 5 (10-Yr Recursive PDF)"),
        (faculty_dir / "06_historical_co2_vs_temperature.png", "Faculty Fig 6 (CO2 vs Temp PNG)"),
        (faculty_dir / "06_historical_co2_vs_temperature.pdf", "Faculty Fig 6 (CO2 vs Temp PDF)"),
    ]
    success, elapsed, found_outputs = run_stage(
        "Generate All 28 Project Visualizations",
        cmd,
        "Regenerate Groups A, B, C, D, E (28 PNGs + 7 PDFs)",
        check_artifacts_after=outputs
    )
    if success:
        print("To view the interactive graph gallery, open GRAPH_INDEX.html or run:")
        print("    python main.py --gallery\n")
        if open_gallery_after:
            open_gallery()
    return success, elapsed, found_outputs


def execute_test():
    cmd = [sys.executable, "-m", "pytest", "-v", "tests/"]
    return run_stage("Automated Test Suite", cmd, "Run all pytest test cases across unit, integration, and physics suites")


def execute_all():
    print("#" * 78)
    print("STARTING COMPLETE END-TO-END WORKFLOW (--all)")
    print("Stages: Pipeline -> Predict -> Simulate -> Graphs -> Test")
    print("#" * 78 + "\n")

    stages = [
        ("Pipeline", execute_pipeline),
        ("Predict", execute_predict),
        ("Simulate", execute_simulate),
        ("Graphs", lambda: execute_graphs(skip_scm=True)),
        ("Test", execute_test),
    ]

    results = []
    overall_start = time.time()
    all_passed = True

    for name, func in stages:
        success, elapsed, outputs = func()
        results.append({
            "stage": name,
            "status": "PASS" if success else "FAIL",
            "elapsed": elapsed,
            "outputs": len(outputs)
        })
        if not success:
            all_passed = False
            print(f"[STOP] Workflow halted due to failure in stage: {name}")
            break

    total_time = time.time() - overall_start

    # Print Summary Table
    print("\n" + "=" * 78)
    print("PROJECT WORKFLOW EXECUTION SUMMARY")
    print("=" * 78)
    header = f"{'Stage Name':<20} | {'Status':<10} | {'Elapsed (s)':<12} | {'Key Outputs':<12}"
    print(header)
    print("-" * 78)
    for res in results:
        line = f"{res['stage']:<20} | {res['status']:<10} | {res['elapsed']:<12.2f} | {res['outputs']:<12}"
        print(line)
    print("-" * 78)
    print(f"Overall Result: {'SUCCESS - ALL STAGES PASSED' if all_passed else 'FAILED'}")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print("=" * 78 + "\n")

    return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for Climate Sensitivity Under 2xCO2 ML & Physics Research Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pipeline    Train Ridge ARX model and evaluate Protocol A
  python main.py --predict     Run 10-year autonomous recursive forecast & 2026 predictions
  python main.py --simulate    Run DOECLIM 1D ocean heat diffusion simulation
  python main.py --graphs      Regenerate all 28 visualizations (35 files)
  python main.py --gallery     Open interactive graph gallery in web browser
  python main.py --test        Run automated pytest suite
  python main.py --update-data Fetch latest official observations (NOAA/NASA)
  python main.py --all         Execute complete end-to-end workflow in sequence
"""
    )

    parser.add_argument("--pipeline", action="store_true", help="Run full training, validation, testing pipeline (src/run_pipeline.py)")
    parser.add_argument("--predict", action="store_true", help="Generate autonomous 10-yr forecast and 2026 projections (src/predict.py)")
    parser.add_argument("--simulate", action="store_true", help="Run DOECLIM physical climate simulation (src/simulate_hector_2xco2.py)")
    parser.add_argument("--graphs", action="store_true", help="Regenerate all 28 project visualizations (scripts/generate_all_graphs.py)")
    parser.add_argument("--gallery", action="store_true", help="Open the interactive graph gallery (GRAPH_INDEX.html) in your browser")
    parser.add_argument("--test", action="store_true", help="Run full automated test suite (pytest -v tests/)")
    parser.add_argument("--update-data", action="store_true", help="Fetch latest official NOAA and NASA observations into data/updated/")
    parser.add_argument("--all", action="store_true", help="Execute complete workflow in sequence (Pipeline -> Predict -> Simulate -> Graphs -> Test)")

    args = parser.parse_args()

    # If no arguments provided, show help and exit cleanly
    if not (args.pipeline or args.predict or args.simulate or args.graphs or args.gallery or args.test or args.update_data or args.all):
        parser.print_help()
        sys.exit(0)

    if args.update_data:
        success, _, _ = execute_update_data()
        sys.exit(0 if success else 1)

    if args.all:
        ret = execute_all()
        if ret == 0 and args.gallery:
            open_gallery()
        sys.exit(ret)

    if args.pipeline:
        success, _, _ = execute_pipeline()
        sys.exit(0 if success else 1)

    if args.predict:
        success, _, _ = execute_predict()
        sys.exit(0 if success else 1)

    if args.simulate:
        success, _, _ = execute_simulate()
        sys.exit(0 if success else 1)

    if args.graphs:
        success, _, _ = execute_graphs(open_gallery_after=args.gallery)
        sys.exit(0 if success else 1)

    if args.gallery:
        ok = open_gallery()
        sys.exit(0 if ok else 1)

    if args.test:
        success, _, _ = execute_test()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
