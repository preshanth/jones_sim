"""Integration tests for calibration validation scripts.

Tests that validation scripts execute successfully and pass validation checks.
These are slower tests that create MS files and run full calibration pipelines.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Find scripts directory
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

# Check if CASA is available
try:
    from importlib.util import find_spec

    CASA_AVAILABLE = (
        find_spec("casatools") is not None and find_spec("casatasks") is not None
    )
except ImportError:
    CASA_AVAILABLE = False


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not CASA_AVAILABLE, reason="CASA tools not available")
class TestCalibrationValidation:
    """Integration tests for validation scripts."""

    def test_bandpass_validation_no_noise_map(self, tmp_path):
        """Test bandpass validation with no noise and MAP optimization."""
        os.chdir(tmp_path)

        script = SCRIPTS_DIR / "validate_bandpass_recovery.py"
        if not script.exists():
            pytest.skip(f"Script not found: {script}")

        cmd = [
            sys.executable,
            str(script),
            "--msname",
            "test_bandpass.ms",
            "--n_channels",
            "32",
            "--no_noise",
            "--map",
            "--seed",
            "44",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        print("STDOUT:", result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)

        assert result.returncode == 0, "Bandpass validation failed"
        assert "VALIDATION PASSED" in result.stdout or "✓" in result.stdout


@pytest.mark.fast
class TestConfigValidation:
    """Fast tests for JSON config loading."""

    def test_all_test_configs_load(self):
        """Test that all test configs load without errors."""
        from jones_sim import JonesConfig

        configs_dir = Path(__file__).parent.parent / "configs"
        test_configs = list(configs_dir.glob("test_*.json"))

        assert len(test_configs) > 0, "No test configs found"

        for config_file in test_configs:
            # Should not raise
            config = JonesConfig(str(config_file))
            assert "jones_chain" in config.config
            assert "effects" in config.config
            print(f"✓ {config_file.name} loaded successfully")

    def test_config_creates_simulator(self):
        """Test that configs can create simulators."""
        from jones_sim import JonesConfig

        configs_dir = Path(__file__).parent.parent / "configs"
        config_file = configs_dir / "test_k_delays.json"

        if not config_file.exists():
            pytest.skip(f"Config not found: {config_file}")

        config = JonesConfig(str(config_file))
        sim = config.create_simulator(n_antennas=27)

        assert sim is not None
        assert len(sim.effects) > 0


@pytest.mark.fast
class TestPlottingEnhanced:
    """Fast tests for enhanced plotting utilities."""

    def test_plot_imports(self):
        """Test that all plotting functions import correctly."""
        from jones_sim.plotting_enhanced import (
            create_validation_dashboard,
            plot_bandpass_comparison,
            plot_error_histogram,
            plot_leakage_dterms,
            plot_three_way_comparison,
            plot_time_series_gains,
        )

        # All should be callable
        assert callable(plot_bandpass_comparison)
        assert callable(plot_three_way_comparison)
        assert callable(plot_leakage_dterms)
        assert callable(plot_time_series_gains)
        assert callable(plot_error_histogram)
        assert callable(create_validation_dashboard)

    def test_three_way_comparison_plot(self, tmp_path):
        """Test three-way comparison plotting."""
        import numpy as np

        from jones_sim.plotting_enhanced import plot_three_way_comparison

        truth = np.array([1.0, 1.2, 0.9, 1.1, 1.05])
        casa = truth + np.random.normal(0, 0.01, 5)
        recovered = truth + np.random.normal(0, 0.005, 5)

        output_file = tmp_path / "test_comparison.html"

        p1, p2 = plot_three_way_comparison(
            truth,
            casa,
            recovered,
            output_file_path=str(output_file),
        )

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_error_histogram_plot(self, tmp_path):
        """Test error histogram plotting."""
        import numpy as np

        from jones_sim.plotting_enhanced import plot_error_histogram

        errors = {
            "CASA": np.random.normal(0, 0.01, 100),
            "Recovered": np.random.normal(0, 0.005, 100),
        }

        output_file = tmp_path / "test_histogram.html"

        plot_error_histogram(errors, output_file_path=str(output_file))

        assert output_file.exists()
        assert output_file.stat().st_size > 0
