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
