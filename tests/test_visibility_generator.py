"""Test visibility generator and complete pipeline."""

import numpy as np
import pytest

from jones_sim.visibility_generator import VisibilityGenerator, quick_unpolarized_sim, quick_polarized_sim
from jones_sim.source_models import (
    create_unpolarized_source,
    create_linear_source,
    create_rm_source,
    create_circular_source
)
from jones_sim.effects import ElectronicGains, ParallacticAngle, RotationMeasure, InstrumentalLeakage


class TestVisibilityGenerator:
    """Test the high-level visibility generator."""

    def setup_method(self):
        """Setup for each test."""
        self.generator = VisibilityGenerator(n_antennas=3, noise_std=0.1, random_seed=42)

        # Test parameters
        self.frequencies = np.array([1e9, 1.5e9, 2e9])  # 1-2 GHz
        self.times = np.array([0.0, 60.0, 120.0])  # 0-2 minutes

    def test_baseline_data_generation(self):
        """Test baseline coordinate generation."""
        freq_grid, time_grid, ant1_ids, ant2_ids = self.generator.generate_baseline_data(
            self.frequencies, self.times, exclude_autocorr=True
        )

        n_baselines = 3 * 2  # 3 antennas, exclude autocorr = 6 baselines
        n_freq = len(self.frequencies)
        n_time = len(self.times)
        expected_length = n_baselines * n_freq * n_time

        assert len(freq_grid) == expected_length
        assert len(time_grid) == expected_length
        assert len(ant1_ids) == expected_length
        assert len(ant2_ids) == expected_length

        # Check that no autocorrelations exist
        assert not np.any(ant1_ids == ant2_ids)

        # Check frequency/time ranges
        assert np.min(freq_grid) == np.min(self.frequencies)
        assert np.max(freq_grid) == np.max(self.frequencies)
        assert np.min(time_grid) == np.min(self.times)
        assert np.max(time_grid) == np.max(self.times)

    def test_baseline_data_with_autocorr(self):
        """Test baseline generation including autocorrelations."""
        freq_grid, time_grid, ant1_ids, ant2_ids = self.generator.generate_baseline_data(
            self.frequencies, self.times, exclude_autocorr=False
        )

        n_baselines = 3 * 3  # 3x3 = 9 baselines including autocorr
        expected_length = n_baselines * len(self.frequencies) * len(self.times)

        assert len(freq_grid) == expected_length

        # Check that autocorrelations exist
        assert np.any(ant1_ids == ant2_ids)

    def test_noise_statistics(self):
        """Test Gaussian noise properties."""
        # Generate large array for statistical testing
        test_shape = (10000, 4)
        test_vis = np.ones(test_shape, dtype=complex)

        # Set noise std and add noise
        noise_std = 0.1
        self.generator.noise_std = noise_std
        noisy_vis = self.generator.add_noise(test_vis)

        # Extract noise by subtracting original
        noise = noisy_vis - test_vis

        # Test noise statistics
        # For Option B: σ_real = σ_imag = σ_total/√2
        expected_component_std = noise_std / np.sqrt(2)

        # Test real component
        noise_real = noise.real.flatten()
        measured_std_real = np.std(noise_real)
        np.testing.assert_allclose(measured_std_real, expected_component_std, rtol=0.05)

        # Test imaginary component
        noise_imag = noise.imag.flatten()
        measured_std_imag = np.std(noise_imag)
        np.testing.assert_allclose(measured_std_imag, expected_component_std, rtol=0.05)

        # Test total complex variance
        noise_magnitude = np.abs(noise.flatten())
        # For complex Gaussian: E[|z|²] = σ_real² + σ_imag² = σ_total²
        measured_total_var = np.var(noise_magnitude)
        expected_total_var = noise_std**2

        # Mean of noise should be ~0
        np.testing.assert_allclose(np.mean(noise_real), 0.0, atol=0.01)
        np.testing.assert_allclose(np.mean(noise_imag), 0.0, atol=0.01)

        print(f"Expected component std: {expected_component_std:.4f}")
        print(f"Measured real std: {measured_std_real:.4f}")
        print(f"Measured imag std: {measured_std_imag:.4f}")
        print(f"Expected total variance: {expected_total_var:.4f}")
        print(f"Measured magnitude variance: {measured_total_var:.4f}")

    def test_no_noise_when_disabled(self):
        """Test that no noise is added when noise_std=0."""
        self.generator.noise_std = 0.0

        test_vis = np.ones((100, 4), dtype=complex)
        noisy_vis = self.generator.add_noise(test_vis)

        np.testing.assert_array_equal(test_vis, noisy_vis)

    def test_case1_unpolarized_source(self):
        """Test Case 1: Unpolarized source corruption."""
        # Add simple gain effect
        gains = ElectronicGains(2.0, 1.5)  # Different XX, YY gains
        self.generator.add_jones_effect('gains', gains)

        # Create unpolarized source
        source = create_unpolarized_source(1.0)

        # Generate visibilities
        result = self.generator.generate_visibilities(
            source, self.frequencies, self.times, add_noise=True
        )

        # Verify result structure
        assert 'visibilities' in result
        assert 'ideal_visibilities' in result
        assert 'frequencies' in result

        vis = result['visibilities']
        ideal = result['ideal_visibilities']

        # Check that corruption occurred (not identical to ideal)
        assert not np.allclose(vis, ideal)

        # For unpolarized source with diagonal gains:
        # Ideal: [1, 0, 0, 1], Corrupted: [g1*g1*, 0, 0, g2*g2*] = [4, 0, 0, 2.25]
        # Check one baseline (antenna 0-1)
        baseline_mask = (result['antenna1'] == 0) & (result['antenna2'] == 1)
        baseline_vis = vis[baseline_mask][0]  # First time/freq point

        expected_xx = 2.0 * 2.0  # g_xx * g_xx*
        expected_yy = 1.5 * 1.5  # g_yy * g_yy*

        # Check XX and YY correlations (should be real for unpolarized + diagonal gains)
        np.testing.assert_allclose(baseline_vis[0].real, expected_xx, rtol=0.1)  # XX
        np.testing.assert_allclose(baseline_vis[3].real, expected_yy, rtol=0.1)  # YY

        print(f"Case 1 - Unpolarized corruption test passed")
        print(f"Expected XX: {expected_xx}, Measured: {baseline_vis[0]}")
        print(f"Expected YY: {expected_yy}, Measured: {baseline_vis[3]}")

    def test_case2_linear_polarization(self):
        """Test Case 2: 5% linear polarization at 30° PA."""
        # Add leakage effect to see cross-polarization
        leakage = InstrumentalLeakage(d_hv=0.02, d_vh=0.03)
        self.generator.add_jones_effect('leakage', leakage)

        # Create 5% linearly polarized source at 30°
        source = create_linear_source(1.0, 5.0, 30.0)

        result = self.generator.generate_visibilities(
            source, self.frequencies, self.times, add_noise=True
        )

        vis = result['visibilities']
        ideal = result['ideal_visibilities']

        # Check that corruption occurred
        assert not np.allclose(vis, ideal)

        # Linear polarization should create XY, YX correlations
        # Check that cross-correlations are non-zero
        xy_correlations = vis[:, 1]  # XY
        yx_correlations = vis[:, 2]  # YX

        assert np.any(np.abs(xy_correlations) > 1e-10)
        assert np.any(np.abs(yx_correlations) > 1e-10)

        print(f"Case 2 - Linear polarization test passed")
        print(f"Max |XY|: {np.max(np.abs(xy_correlations)):.6f}")
        print(f"Max |YX|: {np.max(np.abs(yx_correlations)):.6f}")

    def test_case3_rotation_measure(self):
        """Test Case 3: RM = 25 rad/m² effect."""
        # Add rotation measure effect
        rm_effect = RotationMeasure(25.0)
        self.generator.add_jones_effect('rotation_measure', rm_effect)

        # Create RM source (linearly polarized with intrinsic RM)
        source = create_rm_source(1.0, 5.0, 30.0, 25.0, 1e9)

        result = self.generator.generate_visibilities(
            source, self.frequencies, self.times, add_noise=True
        )

        vis = result['visibilities']
        freqs = result['frequencies']

        # RM effect should be frequency dependent
        # Group by frequency and check that rotation varies
        unique_freqs = np.unique(freqs)
        rotations = []

        for freq in unique_freqs:
            freq_mask = freqs == freq
            freq_vis = vis[freq_mask]

            # Average over baselines/times for this frequency
            avg_xy = np.mean(freq_vis[:, 1])  # XY
            avg_yx = np.mean(freq_vis[:, 2])  # YX

            # Extract rotation from XY phase (rough approximation)
            if np.abs(avg_xy) > 1e-10:
                rotation = np.angle(avg_xy)
                rotations.append(rotation)

        # Check that rotations are different across frequencies (λ² dependence)
        if len(rotations) > 1:
            rotation_range = np.max(rotations) - np.min(rotations)
            assert rotation_range > 1e-6  # Should see frequency-dependent rotation

        print(f"Case 3 - Rotation measure test passed")
        print(f"Rotation range across frequencies: {rotation_range:.6f} rad")

    def test_case4_circular_polarization(self):
        """Test Case 4: 10% circular + 2% linear polarization."""
        # Add parallactic angle rotation to see circular effects
        parallactic = ParallacticAngle(np.pi/6)  # 30° parallactic angle
        self.generator.add_jones_effect('parallactic', parallactic)

        # Create circularly polarized source with small linear component
        source = create_circular_source(1.0, 10.0, 2.0, 0.0, 'right')

        result = self.generator.generate_visibilities(
            source, self.frequencies, self.times, add_noise=True
        )

        vis = result['visibilities']
        ideal = result['ideal_visibilities']

        # Check Stokes V creates correlations
        I, Q, U, V = source.stokes_parameters()
        assert V > 0  # Right circular polarization

        # Circular polarization in ideal visibilities
        ideal_xy = ideal[0, 1]  # First visibility XY
        ideal_yx = ideal[0, 2]  # First visibility YX

        # For circular polarization: XY = U + iV, YX = U - iV
        expected_xy = U + 1j * V
        expected_yx = U - 1j * V

        np.testing.assert_allclose(ideal_xy, expected_xy, rtol=1e-10)
        np.testing.assert_allclose(ideal_yx, expected_yx, rtol=1e-10)

        print(f"Case 4 - Circular polarization test passed")
        print(f"Stokes V: {V:.4f}")
        print(f"Ideal XY: {ideal_xy}")
        print(f"Ideal YX: {ideal_yx}")

    def test_effects_comparison(self):
        """Test individual effect isolation."""
        # Add multiple effects
        gains = ElectronicGains(1.2, 0.8)
        leakage = InstrumentalLeakage(d_hv=0.01, d_vh=0.02)

        self.generator.add_jones_effect('gains', gains)
        self.generator.add_jones_effect('leakage', leakage)

        source = create_linear_source(1.0, 3.0, 45.0)

        # Compare individual effects
        comparison = self.generator.compare_effects(
            source, self.frequencies[:1], self.times[:1]  # Single freq/time for speed
        )

        # Verify all expected keys exist
        assert 'baseline' in comparison
        assert 'all_effects' in comparison
        assert 'gains' in comparison
        assert 'leakage' in comparison

        # Baseline should be ideal (no corruption)
        baseline_vis = comparison['baseline']['visibilities']
        ideal_vis = comparison['baseline']['ideal_visibilities']
        np.testing.assert_allclose(baseline_vis, ideal_vis)

        # Individual effects should differ from baseline
        gains_vis = comparison['gains']['visibilities']
        leakage_vis = comparison['leakage']['visibilities']

        assert not np.allclose(gains_vis, ideal_vis)
        assert not np.allclose(leakage_vis, ideal_vis)

        print("Effects comparison test passed")

    def test_corruption_summary(self):
        """Test corruption summary generation."""
        gains = ElectronicGains(1.5, 1.2)
        self.generator.add_jones_effect('gains', gains)

        source = create_unpolarized_source(2.0)

        summary = self.generator.get_corruption_summary(source, 1.5e9, 30.0)

        # Check summary structure
        assert 'stokes_parameters' in summary
        assert 'ideal_correlations' in summary
        assert 'jones_matrices' in summary

        # Check Stokes parameters
        stokes = summary['stokes_parameters']
        expected_stokes = np.array([2.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(stokes, expected_stokes)

        # Check Jones matrices for each antenna
        for ant_id in range(self.generator.n_antennas):
            assert f'antenna_{ant_id}' in summary['jones_matrices']
            jones_matrix = summary['jones_matrices'][f'antenna_{ant_id}']
            assert jones_matrix.shape == (2, 2)

        print("Corruption summary test passed")


class TestConvenienceFunctions:
    """Test quick simulation convenience functions."""

    def test_quick_unpolarized_sim(self):
        """Test quick unpolarized simulation."""
        frequencies = np.array([1e9, 2e9])
        times = np.array([0.0, 60.0])

        jones_effects = {
            'gains': ElectronicGains(1.1, 0.9)
        }

        result = quick_unpolarized_sim(frequencies, times, jones_effects, noise_std=0.05)

        assert 'visibilities' in result
        assert 'ideal_visibilities' in result

        # Check that gains were applied
        vis = result['visibilities']
        ideal = result['ideal_visibilities']
        assert not np.allclose(vis, ideal)

    def test_quick_polarized_sim_linear(self):
        """Test quick linear polarized simulation."""
        frequencies = np.array([1e9])
        times = np.array([0.0])

        jones_effects = {
            'leakage': InstrumentalLeakage(d_hv=0.02)
        }

        result = quick_polarized_sim(
            frequencies, times, jones_effects,
            pol_type='linear', pol_fraction=0.1, noise_std=0.02
        )

        assert 'visibilities' in result
        # Should have cross-correlations due to leakage
        xy_corr = result['visibilities'][:, 1]
        assert np.any(np.abs(xy_corr) > 1e-10)

    def test_quick_polarized_sim_circular(self):
        """Test quick circular polarized simulation."""
        frequencies = np.array([1e9])
        times = np.array([0.0])

        jones_effects = {
            'parallactic': ParallacticAngle(np.pi/4)
        }

        result = quick_polarized_sim(
            frequencies, times, jones_effects,
            pol_type='circular', pol_fraction=0.15, noise_std=0.01
        )

        assert 'visibilities' in result
        # Circular polarization should create correlations
        vis = result['visibilities']
        assert np.any(np.abs(vis) > 1e-10)

    def test_invalid_pol_type(self):
        """Test error handling for invalid polarization type."""
        frequencies = np.array([1e9])
        times = np.array([0.0])
        jones_effects = {}

        with pytest.raises(ValueError):
            quick_polarized_sim(frequencies, times, jones_effects, pol_type='invalid')


if __name__ == "__main__":
    # Run a quick demonstration
    print("Running visibility generator demonstration...")

    # Quick test of all 4 cases
    generator = VisibilityGenerator(n_antennas=3, noise_std=0.05, random_seed=42)
    freqs = np.array([1e9, 2e9])
    times = np.array([0.0, 60.0])

    # Add some effects
    generator.add_jones_effect('gains', ElectronicGains(1.2, 0.8))
    generator.add_jones_effect('rotation_measure', RotationMeasure(10.0))

    sources = [
        ("Unpolarized", create_unpolarized_source(1.0)),
        ("Linear 5%", create_linear_source(1.0, 5.0, 30.0)),
        ("RM affected", create_rm_source(1.0, 5.0, 30.0, 25.0, 1e9)),
        ("Circular 10%", create_circular_source(1.0, 10.0, 2.0))
    ]

    for name, source in sources:
        result = generator.generate_visibilities(source, freqs, times)
        vis_rms = np.sqrt(np.mean(np.abs(result['visibilities'])**2))
        print(f"{name:12s}: RMS visibility = {vis_rms:.4f}")

    print("Demonstration complete!")