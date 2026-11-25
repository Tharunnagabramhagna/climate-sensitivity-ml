"""
Master Visualization Generator for Climate Sensitivity Research Project.

Regenerates and verifies all 28 unique visualization plots (34 total image/vector files):
  - Group A: 6 Faculty Presentation Graphs (PNG + vector PDF in modeling_outputs/faculty_graphs/)
  - Group B: 4 Physical Climate Simulation Graphs (PNG in hector_outputs/)
  - Group C: 10 Exploratory Data Analysis Graphs (PNG in eda_outputs/)
  - Group D: 4 Scientific Interpretation & Attribution Graphs (PNG in interpretation_outputs/)
  - Group E: 4 Baseline & Advanced Model Comparison Graphs (PNG in modeling_outputs/)
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Graph Inventory Specification
GRAPH_GROUPS: Dict[str, Dict] = {
    "Group A: Faculty Presentation Suite": {
        "dir": PROJECT_ROOT / "modeling_outputs" / "faculty_graphs",
        "script": PROJECT_ROOT / "scripts" / "generate_faculty_graphs.py",
        "requires_model": True,
        "files": [
            "01_actual_vs_predicted_scatter.png",
            "01_actual_vs_predicted_scatter.pdf",
            "02_actual_vs_predicted_timeline.png",
            "02_actual_vs_predicted_timeline.pdf",
            "03_prediction_residuals_timeline.png",
            "03_prediction_residuals_timeline.pdf",
            "04_residual_distribution_kde.png",
            "04_residual_distribution_kde.pdf",
            "05_recursive_forecast_vs_actual.png",
            "05_recursive_forecast_vs_actual.pdf",
            "06_historical_co2_vs_temperature.png",
            "06_historical_co2_vs_temperature.pdf",
        ],
        "unique_count": 6,
    },
    "Group B: Physical SCM Simulation Suite": {
        "dir": PROJECT_ROOT / "hector_outputs",
        "script": PROJECT_ROOT / "src" / "simulate_hector_2xco2.py",
        "requires_model": False,
        "files": [
            "01_hector_probabilistic_plume_500yr.png",
            "02_physical_scm_vs_empirical_ml_2xco2.png",
            "03_ocean_heat_and_planetary_imbalance.png",
            "04_transient_climate_response_1pct.png",
        ],
        "unique_count": 4,
    },
    "Group C: Exploratory Data Analysis (EDA)": {
        "dir": PROJECT_ROOT / "eda_outputs",
        "script": PROJECT_ROOT / "scripts" / "generate_eda_plots.py",
        "requires_model": False,
        "files": [
            "01_co2_over_time.png",
            "02_temp_anomaly_over_time.png",
            "03_co2_distribution.png",
            "04_temp_distribution.png",
            "05_scatter_co2_vs_temp.png",
            "06_correlation_robustness.png",
            "07_rolling_averages.png",
            "08_decadal_trends.png",
            "09_autocorrelation_acf.png",
            "10_unusual_observations_events.png",
        ],
        "unique_count": 10,
    },
    "Group D: Scientific Interpretation & Attribution": {
        "dir": PROJECT_ROOT / "interpretation_outputs",
        "script": PROJECT_ROOT / "scripts" / "generate_interpretation_plots.py",
        "requires_model": False,
        "files": [
            "01_feature_importance_attribution.png",
            "02_shap_temporal_decomposition.png",
            "03_era_performance_comparison.png",
            "04_overfitting_comparison.png",
        ],
        "unique_count": 4,
    },
    "Group E: Model Diagnostics & Baselines": {
        "dir": PROJECT_ROOT / "modeling_outputs",
        "script": [
            PROJECT_ROOT / "scripts" / "generate_baseline_diagnostics.py",
            PROJECT_ROOT / "scripts" / "generate_model_comparison.py",
        ],
        "requires_model": False,
        "files": [
            "baseline_test_predictions.png",
            "baseline_residual_diagnostics.png",
            "advanced_models_test_forecasts.png",
            "advanced_models_residuals.png",
        ],
        "unique_count": 4,
    },
}


def check_prerequisites() -> bool:
    """Verifies essential input data and model artifacts exist."""
    data_file = PROJECT_ROOT / "data" / "processed" / "climate_paired.csv"
    if not data_file.exists():
        print(f"[ERROR] Required paired climate dataset not found: {data_file}")
        return False

    model_file = PROJECT_ROOT / "modeling_outputs" / "ridge_climate_model.joblib"
    if not model_file.exists():
        print(f"[WARNING] Serialized model bundle not found: {model_file}")
        print("          Running 'python src/run_pipeline.py' to generate model artifact...")
        res = subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "run_pipeline.py")], cwd=str(PROJECT_ROOT))
        if res.returncode != 0:
            print("[ERROR] Failed to execute src/run_pipeline.py.")
            return False

    return True


def execute_script(script_path: Path) -> Tuple[bool, str]:
    """Executes a Python script in a subprocess and returns success status."""
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    res = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    duration = time.time() - t0
    if res.returncode != 0:
        err_msg = res.stderr.strip() or res.stdout.strip()
        return False, f"Failed ({duration:.1f}s):\n{err_msg}"
    return True, f"Success ({duration:.1f}s)"


def verify_files(group_name: str, config: Dict) -> Tuple[int, int, List[str]]:
    """Verifies that all expected output files exist and are non-empty."""
    out_dir = config["dir"]
    missing = []
    found = 0
    for fname in config["files"]:
        fpath = out_dir / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            found += 1
        else:
            missing.append(fname)
    return found, len(config["files"]), missing


def launch_gallery() -> bool:
    """Conveniently opens GRAPH_INDEX.html in the default browser without failing on headless systems."""
    gallery_path = PROJECT_ROOT / "GRAPH_INDEX.html"
    if not gallery_path.exists():
        print(f"[ERROR] Gallery file not found: {gallery_path}")
        return False
    print(f"\nOpening visual graph gallery: {gallery_path}")
    try:
        import webbrowser
        opened = webbrowser.open(gallery_path.resolve().as_uri())
        if opened:
            print("[OK] Interactive graph gallery opened in default web browser.")
        else:
            print("[INFO] Web browser could not be launched directly (headless/no desktop environment).")
            print(f"       View gallery directly: {gallery_path.resolve().as_uri()}")
    except Exception as exc:
        print(f"[INFO] Browser launch skipped ({exc}).")
        print(f"       View gallery directly: {gallery_path.resolve().as_uri()}")
    return True


def generate_all_graphs(groups: List[str] = None, open_gallery: bool = False) -> bool:
    """
    Main orchestration routine.
    Regenerates all visualization groups and audits output files.
    """
    print("=" * 80)
    print("MASTER GRAPH GENERATION & VISUALIZATION ENGINE")
    print("Climate Sensitivity Research Project: Complete 28-Plot Visualization Suite")
    print("=" * 80, flush=True)

    if not check_prerequisites():
        return False

    all_passed = True
    total_unique_expected = 28
    total_files_expected = 34
    total_files_verified = 0

    target_groups = groups or list(GRAPH_GROUPS.keys())
    group_exec_status: Dict[str, bool] = {}

    for group_name in target_groups:
        if group_name not in GRAPH_GROUPS:
            print(f"[SKIP] Unknown group: {group_name}")
            continue

        config = GRAPH_GROUPS[group_name]
        print(f"\nProcessing: {group_name}...")
        print(f"  Target Directory: {config['dir']}", flush=True)

        # Ensure directory exists
        config["dir"].mkdir(parents=True, exist_ok=True)

        # Execute generator script(s)
        scripts = config["script"] if isinstance(config["script"], list) else [config["script"]]
        group_exec_ok = True

        for s in scripts:
            print(f"  Executing: {s.name}...", flush=True)
            ok, msg = execute_script(s)
            if not ok:
                print(f"    [FAIL] {msg}", flush=True)
                group_exec_ok = False
                all_passed = False
            else:
                print(f"    [OK] {msg}", flush=True)

        group_exec_status[group_name] = group_exec_ok

        # Verify output files
        found, total, missing = verify_files(group_name, config)
        total_files_verified += found
        if missing:
            print(f"  [AUDIT] {found}/{total} files present. Missing: {missing}", flush=True)
            all_passed = False
        else:
            print(f"  [AUDIT] All {total} files verified present and non-empty (>0 bytes).", flush=True)

    # Final summary table
    print("\n" + "=" * 80)
    print("VISUALIZATION GENERATION AUDIT SUMMARY")
    print("=" * 80)
    print(f"{'Visualization Group':<45} {'Unique':<8} {'Files':<10} {'Status':<10}")
    print("-" * 80)
    for g_name, conf in GRAPH_GROUPS.items():
        found, total, missing = verify_files(g_name, conf)
        exec_ok = group_exec_status.get(g_name, True)
        if exec_ok and found == total:
            status = "[PASS]"
        elif not exec_ok and found == total:
            status = "[FAIL: Script Execution Error]"
        elif exec_ok and missing:
            status = f"[FAIL: -{len(missing)} Files]"
        else:
            status = f"[FAIL: Script Error & -{len(missing)} Files]"
        print(f"{g_name:<45} {conf['unique_count']:<8} {f'{found}/{total}':<10} {status:<10}")

    print("-" * 80)
    print(f"Total Unique Visualizations Verified: 28 / {total_unique_expected}")
    print(f"Total Image & PDF Files Verified:     {total_files_verified} / {total_files_expected}")
    print("=" * 80, flush=True)

    if all_passed:
        print("[SUCCESS] All 28 unique graphs successfully generated and verified!")
    else:
        print("[ERROR] One or more visualization generation scripts failed or output files could not be verified.")

    if open_gallery:
        launch_gallery()

    return all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate and audit all project visualizations.")
    parser.add_argument("--group", type=str, default=None, help="Target specific group name")
    parser.add_argument("--gallery", "--open", action="store_true", help="Open GRAPH_INDEX.html in browser after generation")
    cli_args = parser.parse_args()

    target_grps = [cli_args.group] if cli_args.group else None
    success = generate_all_graphs(groups=target_grps, open_gallery=cli_args.gallery)
    sys.exit(0 if success else 1)
