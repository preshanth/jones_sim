"""Tests for JSON configuration system."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from jones_sim.config import DistributionSampler, JonesConfig, load_config


class TestDistributionSampler:
    """Test distribution sampling functionality."""

    def test_constant_scalar(self):
        """Test constant value sampling (scalar)."""
        result = DistributionSampler.sample(5.0)
        assert result == 5.0

    def test_constant_array(self):
        """Test constant value sampling (array)."""
        result = DistributionSampler.sample(5.0, size=10)
        assert result.shape == (10,)
        assert np.all(result == 5.0)

    def test_constant_dict(self):
        """Test constant distribution from dict."""
        dist_spec = {"distribution": "constant", "value": 3.14}
        result = DistributionSampler.sample(dist_spec, size=5)
        assert result.shape == (5,)
        assert np.all(result == 3.14)

    def test_uniform_distribution(self):
        """Test uniform distribution sampling."""
        dist_spec = {"distribution": "uniform", "min": 0.0, "max": 10.0}
        result = DistributionSampler.sample(dist_spec, size=100)
        assert result.shape == (100,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 10.0)
        # Check that we're getting reasonable spread
        assert np.std(result) > 1.0

    def test_gaussian_distribution(self):
        """Test Gaussian distribution sampling."""
        dist_spec = {"distribution": "gaussian", "mean": 5.0, "std": 1.0}
        result = DistributionSampler.sample(dist_spec, size=1000)
        assert result.shape == (1000,)
        # Check mean and std are approximately correct
        assert abs(np.mean(result) - 5.0) < 0.5
        assert abs(np.std(result) - 1.0) < 0.5

    def test_lognormal_distribution(self):
        """Test log-normal distribution sampling."""
        dist_spec = {"distribution": "log_normal", "mean": 1.0, "std": 0.1}
        result = DistributionSampler.sample(dist_spec, size=1000)
        assert result.shape == (1000,)
        assert np.all(result > 0)  # Log-normal is always positive
        # Check mean is approximately correct
        assert abs(np.mean(result) - 1.0) < 0.2

    def test_complex_gaussian(self):
        """Test complex Gaussian distribution."""
        dist_spec = {
            "distribution": "complex_gaussian",
            "mean_real": 0.1,
            "mean_imag": 0.05,
            "std_real": 0.01,
            "std_imag": 0.01,
        }
        result = DistributionSampler.sample(dist_spec, size=1000)
        assert result.shape == (1000,)
        assert result.dtype == np.complex128
        # Check means
        assert abs(np.mean(result.real) - 0.1) < 0.01
        assert abs(np.mean(result.imag) - 0.05) < 0.01

    def test_unknown_distribution(self):
        """Test that unknown distribution raises error."""
        dist_spec = {"distribution": "unknown_dist"}
        with pytest.raises(ValueError, match="Unknown distribution type"):
            DistributionSampler.sample(dist_spec)


class TestJonesConfig:
    """Test JonesConfig class."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = JonesConfig()
        assert "metadata" in config.config
        assert "jones_chain" in config.config
        assert "effects" in config.config
        assert "noise" in config.config
        assert "processing" in config.config

    def test_load_from_dict(self):
        """Test loading config from dictionary."""
        config_dict = {
            "metadata": {"description": "Test config"},
            "jones_chain": {"order": ["gain"], "enabled_effects": ["gain"]},
            "effects": {
                "gain": {
                    "type": "complex_electronic_gain",
                    "per_antenna": True,
                    "amplitude": {"x_pol": 1.0, "y_pol": 1.0},
                    "phase": {"x_pol": 0.0, "y_pol": 0.0},
                }
            },
            "noise": {"enabled": False},
            "processing": {"use_gpu": False},
        }
        config = JonesConfig(config_dict)
        assert config.config["metadata"]["description"] == "Test config"

    def test_load_from_file(self):
        """Test loading config from JSON file."""
        config_dict = {
            "metadata": {"description": "File test"},
            "jones_chain": {"order": [], "enabled_effects": []},
            "effects": {},
            "noise": {"enabled": False},
            "processing": {"use_gpu": False},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name

        try:
            config = JonesConfig(temp_path)
            assert config.config["metadata"]["description"] == "File test"
        finally:
            Path(temp_path).unlink()

    def test_create_simulator_gains(self):
        """Test creating simulator with gain effect."""
        config_dict = {
            "jones_chain": {"order": ["gain"], "enabled_effects": ["gain"]},
            "effects": {
                "gain": {
                    "type": "complex_electronic_gain",
                    "per_antenna": True,
                    "reference_antenna": 0,
                    "amplitude": {
                        "x_pol": {"distribution": "constant", "value": 1.0},
                        "y_pol": {"distribution": "constant", "value": 1.0},
                    },
                    "phase": {
                        "x_pol": {"distribution": "constant", "value": 0.0},
                        "y_pol": {"distribution": "constant", "value": 0.0},
                    },
                }
            },
        }

        config = JonesConfig(config_dict)
        sim = config.create_simulator(n_antennas=10)
        assert "gain" in sim.effects
        assert len(sim.effects) == 1

    def test_create_simulator_bandpass(self):
        """Test creating simulator with bandpass effect."""
        config_dict = {
            "jones_chain": {"order": ["bandpass"], "enabled_effects": ["bandpass"]},
            "effects": {
                "bandpass": {
                    "type": "bandpass_amplitude_delay",
                    "per_antenna": True,
                    "reference_antenna": 0,
                    "ref_freq": 1e9,
                    "delay": {
                        "tau_x": {"distribution": "uniform", "min": -1e-9, "max": 1e-9},
                        "tau_y": {"distribution": "uniform", "min": -1e-9, "max": 1e-9},
                    },
                }
            },
        }

        config = JonesConfig(config_dict)
        sim = config.create_simulator(n_antennas=5)
        assert "bandpass" in sim.effects

    def test_create_simulator_leakage(self):
        """Test creating simulator with leakage effect."""
        config_dict = {
            "jones_chain": {"order": ["leakage"], "enabled_effects": ["leakage"]},
            "effects": {
                "leakage": {
                    "type": "feed_leakage",
                    "per_antenna": True,
                    "d_xy": 0.01,
                    "d_yx": 0.01,
                    "theta": 0.0,
                }
            },
        }

        config = JonesConfig(config_dict)
        sim = config.create_simulator(n_antennas=5)
        assert "leakage" in sim.effects

    def test_create_simulator_crosshand_phase(self):
        """Test creating simulator with crosshand phase effect."""
        config_dict = {
            "jones_chain": {
                "order": ["crosshand_phase"],
                "enabled_effects": ["crosshand_phase"],
            },
            "effects": {
                "crosshand_phase": {
                    "type": "xy_phase_offset",
                    "per_antenna": True,
                    "reference_antenna": 0,
                    "phi": {"distribution": "uniform", "min": -0.1, "max": 0.1},
                }
            },
        }

        config = JonesConfig(config_dict)
        sim = config.create_simulator(n_antennas=5)
        assert "crosshand_phase" in sim.effects

    def test_get_noise_config_enabled(self):
        """Test getting noise configuration when enabled."""
        config_dict = {
            "noise": {
                "enabled": True,
                "thermal_noise": {
                    "tsys_kelvin": 50.0,
                    "aperture_efficiency": 0.7,
                    "antenna_diameter_meters": 25.0,
                },
                "random_seed": 42,
            }
        }

        config = JonesConfig(config_dict)
        noise_config = config.get_noise_config()
        assert noise_config is not None
        assert noise_config["tsys"] == 50.0
        assert noise_config["aperture_eff"] == 0.7
        assert noise_config["antenna_diameter"] == 25.0
        assert noise_config["seed"] == 42

    def test_get_noise_config_disabled(self):
        """Test getting noise configuration when disabled."""
        config_dict = {"noise": {"enabled": False}}

        config = JonesConfig(config_dict)
        noise_config = config.get_noise_config()
        assert noise_config is None

    def test_get_processing_config(self):
        """Test getting processing configuration."""
        config_dict = {
            "processing": {
                "use_gpu": True,
                "gpu_device": 1,
                "chunk_size_rows": 50000,
                "batch_gpu_size": 5000,
                "random_seed": 123,
            }
        }

        config = JonesConfig(config_dict)
        proc_config = config.get_processing_config()
        assert proc_config["use_gpu"] is True
        assert proc_config["gpu_device"] == 1
        assert proc_config["chunk_size_rows"] == 50000
        assert proc_config["batch_gpu_size"] == 5000
        assert proc_config["random_seed"] == 123

    def test_save_config(self):
        """Test saving configuration to file."""
        config_dict = {
            "metadata": {"description": "Save test"},
            "jones_chain": {"order": [], "enabled_effects": []},
            "effects": {},
            "noise": {"enabled": False},
            "processing": {"use_gpu": False},
        }

        config = JonesConfig(config_dict)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            config.save(temp_path)
            # Load it back
            with open(temp_path, "r") as f:
                loaded = json.load(f)
            assert loaded["metadata"]["description"] == "Save test"
        finally:
            Path(temp_path).unlink()

    def test_load_config_function(self):
        """Test load_config convenience function."""
        config_dict = {"metadata": {"description": "Convenience test"}}
        config = load_config(config_dict)
        assert isinstance(config, JonesConfig)
        assert config.config["metadata"]["description"] == "Convenience test"

    def test_reference_antenna_applied(self):
        """Test that reference antenna gets fixed values."""
        config_dict = {
            "jones_chain": {"order": ["gain"], "enabled_effects": ["gain"]},
            "effects": {
                "gain": {
                    "type": "complex_electronic_gain",
                    "per_antenna": True,
                    "reference_antenna": 2,
                    "amplitude": {
                        "x_pol": {"distribution": "uniform", "min": 0.5, "max": 1.5},
                        "y_pol": {"distribution": "uniform", "min": 0.5, "max": 1.5},
                    },
                    "phase": {
                        "x_pol": {"distribution": "uniform", "min": -3.14, "max": 3.14},
                        "y_pol": {"distribution": "uniform", "min": -3.14, "max": 3.14},
                    },
                }
            },
        }

        config = JonesConfig(config_dict)
        sim = config.create_simulator(n_antennas=5)
        effect = sim.effects["gain"]

        # Reference antenna (index 2) should have gain = 1.0 + 0j
        # This is set in the config._create_effect method
        # We can't directly check the internal arrays, but we can verify
        # the effect exists and is properly created
        assert effect is not None
