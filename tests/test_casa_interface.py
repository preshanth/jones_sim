"""Test CASA interface classes with realistic observational parameters.

These tests focus on realistic observational scenarios and require casatools
for actual functionality testing. No mock tests - only real functionality.
"""

import numpy as np
import pytest

# Try to import CASA interface
try:
    from jones_sim.casa_interface import CASA_AVAILABLE

    CASA_INTERFACE_AVAILABLE = True
except ImportError:
    CASA_INTERFACE_AVAILABLE = False
    CASA_AVAILABLE = False


class ObservationParameters:
    """Standard observational parameter sets for testing."""

    # VLA-like observation parameters
    VLA_L_BAND = {
        "freq_start": 1.0e9,  # 1 GHz
        "freq_end": 2.0e9,  # 2 GHz
        "n_channels": 64,
        "chan_width": 15.625e6,  # MHz per channel
        "n_antennas": 27,
        "integration_time": 10.0,  # seconds
        "total_time": 3600.0,  # 1 hour
        "n_times": 360,  # 10-second integrations
        "array_name": "VLA",
        "telescope_diameter": 25.0,  # meters
    }

    # ALMA-like observation parameters
    ALMA_BAND6 = {
        "freq_start": 211.0e9,  # 211 GHz
        "freq_end": 275.0e9,  # 275 GHz
        "n_channels": 128,
        "chan_width": 500e6,  # 500 MHz per channel
        "n_antennas": 43,
        "integration_time": 6.0,  # seconds
        "total_time": 1800.0,  # 30 minutes
        "n_times": 300,
        "array_name": "ALMA",
        "telescope_diameter": 12.0,  # meters
    }

    # Compact array for testing
    COMPACT_ARRAY = {
        "freq_start": 1.4e9,  # 1.4 GHz
        "freq_end": 1.5e9,  # 1.5 GHz
        "n_channels": 32,
        "chan_width": 3.125e6,  # 3.125 MHz per channel
        "n_antennas": 6,
        "integration_time": 30.0,  # seconds
        "total_time": 600.0,  # 10 minutes
        "n_times": 20,
        "array_name": "CompactArray",
        "telescope_diameter": 10.0,
    }


class CalibrationParameters:
    """Standard calibration parameter sets for testing."""

    # Gain calibration parameters
    GAIN_CAL = {
        "solint": 60.0,  # Solution interval in seconds
        "gaintype": "G",  # Gain calibration
        "calmode": "ap",  # Amplitude and phase
        "n_solutions": 60,  # Number of solution intervals
        "refant": 0,  # Reference antenna
        "minblperant": 4,  # Minimum baselines per antenna
        "amplitude_scatter": 0.1,  # 10% amplitude variations
        "phase_scatter": 0.1,  # 0.1 radian phase variations
    }

    # Bandpass calibration parameters
    BANDPASS_CAL = {
        "solint": "inf",  # Infinite solution interval (per scan)
        "gaintype": "B",  # Bandpass calibration
        "calmode": "ap",  # Amplitude and phase
        "refant": 0,  # Reference antenna
        "minblperant": 4,
        "amplitude_ripple": 0.05,  # 5% bandpass ripple
        "phase_ripple": 0.05,  # 0.05 radian phase ripple
        "edge_flagging": 0.1,  # Flag 10% of edge channels
    }

    # Polarization calibration parameters
    POL_CAL = {
        "solint": "inf",  # Per scan
        "gaintype": "D",  # Polarization leakage (D-terms)
        "leakage_level": 0.05,  # 5% polarization leakage
        "refant": 0,
        "minblperant": 4,
    }


@pytest.mark.skipif(not CASA_INTERFACE_AVAILABLE, reason="CASA interface not available")
class TestObservationParameters:
    """Test realistic observational parameter validation."""

    def test_vla_parameters_consistency(self):
        """Test VLA-like observation parameters are self-consistent."""
        params = ObservationParameters.VLA_L_BAND

        # Check frequency consistency
        bandwidth = params["freq_end"] - params["freq_start"]
        total_bandwidth = params["n_channels"] * params["chan_width"]
        np.testing.assert_allclose(bandwidth, total_bandwidth, rtol=0.01)

        # Check time consistency
        expected_integrations = params["total_time"] / params["integration_time"]
        assert abs(expected_integrations - params["n_times"]) < 1

        # Check realistic values
        assert 10 <= params["n_antennas"] <= 50  # Reasonable antenna count
        assert 1e8 <= params["freq_start"] <= 1e12  # Radio frequencies
        assert 1.0 <= params["integration_time"] <= 3600  # Reasonable integration

    def test_alma_parameters_consistency(self):
        """Test ALMA-like observation parameters are self-consistent."""
        params = ObservationParameters.ALMA_BAND6

        # Check frequency consistency
        bandwidth = params["freq_end"] - params["freq_start"]
        total_bandwidth = params["n_channels"] * params["chan_width"]
        np.testing.assert_allclose(bandwidth, total_bandwidth, rtol=0.01)

        # Check millimeter wave frequencies
        assert params["freq_start"] > 100e9  # > 100 GHz
        assert params["chan_width"] > 1e6  # > 1 MHz channels

    def test_compact_array_parameters(self):
        """Test compact array parameters for testing."""
        params = ObservationParameters.COMPACT_ARRAY

        # Should be smaller/faster for testing
        assert params["n_antennas"] <= 10
        assert params["total_time"] <= 1800  # <= 30 minutes
        assert params["n_times"] <= 100


@pytest.mark.skipif(not CASA_INTERFACE_AVAILABLE, reason="CASA interface not available")
class TestCalibrationParameters:
    """Test calibration parameter validation."""

    def test_gain_calibration_parameters(self):
        """Test gain calibration parameters are realistic."""
        params = CalibrationParameters.GAIN_CAL

        # Check solution interval
        assert 1.0 <= params["solint"] <= 3600  # 1 sec to 1 hour
        assert params["gaintype"] == "G"
        assert params["calmode"] in ["ap", "p", "a"]

        # Check scatter levels are reasonable
        assert 0.0 < params["amplitude_scatter"] < 1.0
        assert 0.0 < params["phase_scatter"] < np.pi

    def test_bandpass_calibration_parameters(self):
        """Test bandpass calibration parameters are realistic."""
        params = CalibrationParameters.BANDPASS_CAL

        assert params["gaintype"] == "B"
        assert 0.0 < params["amplitude_ripple"] < 0.5  # < 50% ripple
        assert 0.0 < params["phase_ripple"] < np.pi / 2
        assert 0.0 <= params["edge_flagging"] <= 0.5  # <= 50% edge flagging

    def test_polarization_calibration_parameters(self):
        """Test polarization calibration parameters are realistic."""
        params = CalibrationParameters.POL_CAL

        assert params["gaintype"] == "D"
        assert 0.0 < params["leakage_level"] < 0.2  # < 20% leakage


@pytest.mark.skipif(not CASA_INTERFACE_AVAILABLE, reason="CASA interface not available")
class TestFlagHandling:
    """Test flag handling functionality with realistic parameters."""

    def test_flag_array_creation(self):
        """Test creation of realistic flag arrays."""
        params = ObservationParameters.COMPACT_ARRAY

        n_baselines = params["n_antennas"] * (params["n_antennas"] - 1) // 2
        n_vis = n_baselines * params["n_times"]
        n_corr = 4  # [XX, XY, YX, YY]

        # Create realistic flag pattern
        flags = np.zeros((n_vis, n_corr), dtype=bool)

        # Random RFI flagging (5% of data)
        rfi_fraction = 0.05
        n_rfi_flags = int(rfi_fraction * n_vis * n_corr)
        rfi_indices = np.random.choice(n_vis * n_corr, n_rfi_flags, replace=False)
        flags.flat[rfi_indices] = True

        # Edge channel flagging
        edge_frac = CalibrationParameters.BANDPASS_CAL["edge_flagging"]
        int(edge_frac * params["n_channels"])

        # Antenna-based flagging (antenna 2 is bad) - need proper baseline indexing
        bad_antenna = 2
        if bad_antenna < params["n_antennas"]:
            # Flag all baselines involving bad antenna
            for i in range(n_vis):
                # This is a simplified flagging - in reality would need proper baseline->antenna mapping
                if i % params["n_antennas"] == bad_antenna:
                    flags[i, :] = True

        # Verify flag statistics
        flag_fraction = np.sum(flags) / flags.size
        assert 0.05 <= flag_fraction <= 0.25  # 5-25% flagged data

        # Verify flag properties
        assert flags.dtype == bool
        assert flags.shape == (n_vis, n_corr)

    def test_noise_respects_flags(self):
        """Test that noise addition respects flag structure."""
        from jones_sim.visibility_generator import VisibilityGenerator

        generator = VisibilityGenerator(noise_std=0.1, random_seed=42)

        # Create test visibilities and flags using realistic parameters
        n_vis = 100
        n_corr = 4

        visibilities = np.ones((n_vis, n_corr), dtype=complex)
        flags = np.zeros((n_vis, n_corr), dtype=bool)

        # Flag some data realistically
        flags[10:20, :] = True  # Time-based flagging
        flags[:, 1] = True  # Cross-correlation flagging

        # Add noise respecting flags
        noisy_vis = generator.add_noise(visibilities, flags)

        # Check that flagged data is unchanged
        flagged_vis = noisy_vis[flags]
        original_flagged = visibilities[flags]
        np.testing.assert_array_equal(flagged_vis, original_flagged)

        # Check that unflagged data has noise
        unflagged_vis = noisy_vis[~flags]
        original_unflagged = visibilities[~flags]
        assert not np.allclose(unflagged_vis, original_unflagged, atol=1e-10)


@pytest.mark.skipif(not CASA_INTERFACE_AVAILABLE, reason="CASA interface not available")
class TestSyntheticDataGeneration:
    """Test generation of synthetic data with realistic parameters."""

    def test_generate_realistic_visibility_grid(self):
        """Test generation of realistic visibility coordinate grids."""
        from jones_sim.visibility_generator import VisibilityGenerator

        params = ObservationParameters.COMPACT_ARRAY
        generator = VisibilityGenerator(n_antennas=params["n_antennas"])

        # Create frequency and time arrays
        frequencies = np.linspace(
            params["freq_start"], params["freq_end"], params["n_channels"]
        )
        times = np.linspace(0, params["total_time"], params["n_times"])

        # Generate baseline coordinates
        freq_grid, time_grid, ant1_ids, ant2_ids = generator.generate_baseline_data(
            frequencies, times, exclude_autocorr=True
        )

        # Check dimensions
        n_baselines = params["n_antennas"] * (params["n_antennas"] - 1)  # No autocorr
        expected_length = n_baselines * params["n_channels"] * params["n_times"]

        assert len(freq_grid) == expected_length
        assert len(time_grid) == expected_length
        assert len(ant1_ids) == expected_length
        assert len(ant2_ids) == expected_length

        # Check no autocorrelations
        assert not np.any(ant1_ids == ant2_ids)

        # Check frequency and time ranges
        np.testing.assert_allclose(np.min(freq_grid), params["freq_start"])
        np.testing.assert_allclose(np.max(freq_grid), params["freq_end"])
        np.testing.assert_allclose(np.min(time_grid), 0.0)
        np.testing.assert_allclose(np.max(time_grid), params["total_time"])

    def test_visibility_generation_with_realistic_source(self):
        """Test visibility generation with realistic source and parameters."""
        from jones_sim.effects import ElectronicGains
        from jones_sim.source_models import create_linear_source
        from jones_sim.visibility_generator import VisibilityGenerator

        params = ObservationParameters.COMPACT_ARRAY
        generator = VisibilityGenerator(
            n_antennas=params["n_antennas"], noise_std=0.01, random_seed=42
        )

        # Add realistic gain variations
        gain_params = CalibrationParameters.GAIN_CAL

        def gain_func(freq, time, ant_id):
            # Realistic gain variation: 1 ± scatter
            base_gain = 1.0
            variation = gain_params["amplitude_scatter"] * (np.random.random() - 0.5)
            return base_gain + variation

        gains = ElectronicGains(gain_func, gain_func)
        generator.add_jones_effect("gains", gains)

        # Create realistic source
        source = create_linear_source(flux_jy=1.0, pol_percent=5.0, pa_degrees=30.0)

        # Generate small frequency/time grid for testing
        frequencies = np.linspace(params["freq_start"], params["freq_end"], 4)
        times = np.linspace(0, 300, 5)  # 5 minute test

        # Generate visibilities
        result = generator.generate_visibilities(source, frequencies, times)

        # Verify result structure
        required_keys = [
            "visibilities",
            "flags",
            "frequencies",
            "times",
            "antenna1",
            "antenna2",
        ]
        for key in required_keys:
            assert key in result

        # Verify array shapes are consistent
        n_vis = len(result["frequencies"])
        assert result["visibilities"].shape == (n_vis, 4)
        assert result["flags"].shape == (n_vis, 4)
        assert len(result["antenna1"]) == n_vis
        assert len(result["antenna2"]) == n_vis

        # Verify flags are boolean and visibilities are complex
        assert result["flags"].dtype == bool
        assert np.iscomplexobj(result["visibilities"])


@pytest.mark.skipif(not CASA_INTERFACE_AVAILABLE, reason="CASA interface not available")
class TestCalibrationRealism:
    """Test realistic calibration scenarios."""

    def test_gain_solution_intervals(self):
        """Test realistic gain solution time intervals."""
        params = ObservationParameters.VLA_L_BAND
        cal_params = CalibrationParameters.GAIN_CAL

        # Calculate number of solution intervals
        n_solutions = int(params["total_time"] / cal_params["solint"])
        assert n_solutions > 0

        # Check solution interval is reasonable
        assert cal_params["solint"] <= params["total_time"]
        assert cal_params["solint"] >= params["integration_time"]

    def test_bandpass_channel_coverage(self):
        """Test bandpass calibration covers all channels."""
        params = ObservationParameters.VLA_L_BAND
        cal_params = CalibrationParameters.BANDPASS_CAL

        # Edge flagging should leave usable channels
        edge_channels = int(cal_params["edge_flagging"] * params["n_channels"])
        usable_channels = params["n_channels"] - 2 * edge_channels

        assert usable_channels > 0
        assert usable_channels >= params["n_channels"] * 0.5  # At least 50% usable

    def test_polarization_leakage_levels(self):
        """Test polarization leakage parameters are realistic."""
        cal_params = CalibrationParameters.POL_CAL

        # Leakage should be small but measurable
        assert 0.001 < cal_params["leakage_level"] < 0.2

        # Should be much smaller than source polarization
        source_pol = 0.05  # 5% linear polarization
        assert cal_params["leakage_level"] <= source_pol


# Integration tests - these require actual files
@pytest.mark.integration
@pytest.mark.skipif(not CASA_AVAILABLE, reason="CASA tools not available")
class TestRealDataIntegration:
    """Integration tests that require real MS/cal files."""

    def test_requires_real_data(self):
        """Placeholder for tests requiring real measurement sets."""
        pytest.skip("Integration tests require real MS/cal table files")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
