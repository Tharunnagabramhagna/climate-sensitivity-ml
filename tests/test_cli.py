"""
tests/test_cli.py - Automated tests for main.py CLI functionality
==================================================================
Tests argument parsing, missing prerequisite handling, and CLI stage definitions.
"""

import sys
import subprocess
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent


def test_main_cli_help():
    """Verify that 'python main.py' with no arguments or '--help' exits cleanly with usage info."""
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Unified CLI for Climate Sensitivity Under 2xCO2" in result.stdout
    assert "--pipeline" in result.stdout
    assert "--predict" in result.stdout
    assert "--simulate" in result.stdout
    assert "--graphs" in result.stdout
    assert "--gallery" in result.stdout
    assert "--test" in result.stdout
    assert "--update-data" in result.stdout
    assert "--all" in result.stdout


def test_main_cli_no_args():
    """Verify that running with no arguments prints the help message and exits with 0."""
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "Unified CLI" in result.stdout


def test_main_cli_dataset_check(monkeypatch, tmp_path):
    """Verify check_raw_datasets function properly validates dataset files."""
    import main

    # In our repository, raw datasets should be present
    assert main.check_raw_datasets() is True

    # If pointing to missing files, it reports False
    monkeypatch.setattr(main, "DATASET_FILE", tmp_path / "nonexistent.csv")
    assert main.check_raw_datasets() is False


def test_main_cli_predict_prerequisite_check(monkeypatch, tmp_path):
    """Verify that execute_predict warns when model artifact is missing."""
    import main

    missing_model = tmp_path / "nonexistent_model.joblib"
    monkeypatch.setattr(main, "MODEL_FILE", missing_model)
    success, elapsed, outputs = main.execute_predict()
    assert success is False
    assert outputs == []


def test_main_cli_gallery(monkeypatch):
    """Verify that 'python main.py --gallery' executes without error."""
    import main
    # Ensure webbrowser.open doesn't pop up a window during automated test
    monkeypatch.setattr("webbrowser.open", lambda url: True)
    assert main.open_gallery() is True


def test_graph_gallery_integrity():
    """Verify GRAPH_INDEX.html exists and all referenced 28 images and 34 links exist on disk."""
    import re
    from html.parser import HTMLParser

    gallery_path = BASE_DIR / "GRAPH_INDEX.html"
    assert gallery_path.exists(), "GRAPH_INDEX.html must exist in project root"

    content = gallery_path.read_text(encoding="utf-8")

    class GalleryParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.images = []
            self.links = []
        def handle_starttag(self, tag, attrs):
            attr_dict = dict(attrs)
            if tag == "img" and "src" in attr_dict:
                self.images.append(attr_dict["src"])
            elif tag == "a" and "href" in attr_dict:
                href = attr_dict["href"]
                if not href.startswith("#") and not href.startswith("http"):
                    self.links.append(href)

    parser = GalleryParser()
    parser.feed(content)

    assert len(parser.images) == 28, f"Expected 28 unique gallery images, found {len(parser.images)}"
    assert len(parser.links) == 34, f"Expected 34 artifact links (28 PNGs + 6 PDFs), found {len(parser.links)}"

    for img_path in parser.images:
        p = BASE_DIR / img_path
        assert p.exists(), f"Gallery image does not exist: {img_path}"
        assert p.stat().st_size > 0, f"Gallery image is empty: {img_path}"

    for link_path in parser.links:
        p = BASE_DIR / link_path
        assert p.exists(), f"Gallery linked artifact does not exist: {link_path}"
        assert p.stat().st_size > 0, f"Gallery linked artifact is empty: {link_path}"


def test_standalone_faculty_graph_scripts_exist():
    """Verify that all six standalone faculty graph scripts exist in modeling_outputs/faculty_graphs/."""
    faculty_dir = BASE_DIR / "modeling_outputs" / "faculty_graphs"
    expected_scripts = [
        "01_actual_vs_predicted_scatter.py",
        "02_actual_vs_predicted_timeline.py",
        "03_prediction_residuals_timeline.py",
        "04_residual_distribution_kde.py",
        "05_recursive_forecast_vs_actual.py",
        "06_historical_co2_vs_temperature.py"
    ]
    for script_name in expected_scripts:
        script_file = faculty_dir / script_name
        assert script_file.exists(), f"Missing standalone script: {script_name}"
        assert script_file.stat().st_size > 0, f"Empty standalone script: {script_name}"


def test_run_graph_launcher_validation():
    """Verify run_graph.py menu output and invalid argument handling."""
    # Test invalid option
    res_invalid = subprocess.run(
        [sys.executable, "run_graph.py", "99"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True
    )
    assert res_invalid.returncode != 0
    assert "Invalid graph selection '99'" in res_invalid.stderr

    # Test menu display
    import run_graph
    assert "1" in run_graph.GRAPH_OPTIONS
    assert "6" in run_graph.GRAPH_OPTIONS
    assert len(run_graph.GRAPH_OPTIONS) == 6


def test_get_faculty_data_integrity():
    """Verify get_faculty_data loads verified test predictions without re-training."""
    from scripts.generate_faculty_graphs import get_faculty_data
    test_df, df_paired = get_faculty_data()
    assert len(test_df) == 120, "Test partition must have exactly 120 months (2016-2025)"
    assert len(df_paired) == 814, "Paired dataset must have 814 months"
    assert "Observed_Temp_Anomaly_C" in test_df.columns
    assert "Ridge_1Step_Predicted_C" in test_df.columns
    assert "Ridge_Recursive_Predicted_C" in test_df.columns


