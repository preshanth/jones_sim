"""Test Jones simulator coordinator."""

import numpy as np
import pytest

from jones_sim.simulator import JonesSimulator
from jones_sim.effects import ParallacticAngle, ElectronicGains, InstrumentalLeakage


class TestJonesSimulator:
    """Test the Jones simulator coordinator."""

    def setup_method(self):
        """Setup for each test."""
        self.simulator = JonesSimulator()

    def test_empty_simulator(self):
        """Test simulator with no effects returns identity."""
        matrix = self.simulator.compute_jones_matrix(1e9, 0.0, 0)
        expected = np.eye(2, dtype=complex)

        np.testing.assert_allclose(matrix, expected)

    def test_single_effect(self):
        """Test simulator with single effect."""
        # Add simple gains
        gains = ElectronicGains(2.0, 1.5)
        self.simulator.add_effect('gains', gains)

        matrix = self.simulator.compute_jones_matrix(1e9, 0.0, 0)
        expected = np.array([[2.0, 0], [0, 1.5]], dtype=complex)

        np.testing.assert_allclose(matrix, expected)

    def test_multiple_effects_order(self):
        """Test that effects are applied in correct order."""
        # Add effects in non-standard order to test ordering
        gains = ElectronicGains(2.0, 1.0)
        parallactic = ParallacticAngle(np.pi/2)  # 90 degrees

        # Add in reverse order
        self.simulator.add_effect('gains', gains)
        self.simulator.add_effect('parallactic', parallactic)

        matrix = self.simulator.compute_jones_matrix(1e9, 0.0, 0)

        # Should be P * G (parallactic first in standard order)
        P = np.array([[0, 1], [-1, 0]], dtype=complex)  # 90° rotation
        G = np.array([[2, 0], [0, 1]], dtype=complex)   # Gains
        expected = P @ G

        np.testing.assert_allclose(matrix, expected, atol=1e-15)

    def test_effect_management(self):
        """Test adding/removing effects."""
        gains = ElectronicGains(1.5, 1.2)
        self.simulator.add_effect('gains', gains)

        assert 'gains' in self.simulator.list_effects()

        self.simulator.remove_effect('gains')
        assert 'gains' not in self.simulator.list_effects()

    def test_clear_effects(self):
        """Test clearing all effects."""
        gains = ElectronicGains(2.0, 1.0)
        leakage = InstrumentalLeakage(0.05, 0.03)

        self.simulator.add_effect('gains', gains)
        self.simulator.add_effect('leakage', leakage)

        assert len(self.simulator.list_effects()) == 2

        self.simulator.clear_effects()
        assert len(self.simulator.list_effects()) == 0

    def test_visibility_corruption_identity(self):
        """Test visibility corruption with identity (no effects)."""
        # Test data: 4 visibilities, 4 correlations each
        ideal_vis = np.array([
            [1, 0, 0, 1],      # Unpolarized source
            [1, 0.5, 0.5, 1],  # Partially polarized
            [0, 1, 0, 0],      # Pure XY
            [0, 0, 1, 0]       # Pure YX
        ], dtype=complex)

        freqs = np.array([1e9, 1.1e9, 1.2e9, 1.3e9])
        times = np.array([0.0, 1.0, 2.0, 3.0])
        ant1 = np.array([0, 0, 1, 1])
        ant2 = np.array([1, 2, 2, 0])

        corrupted = self.simulator.corrupt_visibilities(ideal_vis, freqs, times, ant1, ant2)

        # Should be unchanged (identity corruption)
        np.testing.assert_allclose(corrupted, ideal_vis)

    def test_visibility_corruption_gains(self):
        """Test visibility corruption with simple gains."""
        gains = ElectronicGains(2.0, 1.5)
        self.simulator.add_effect('gains', gains)

        # Simple test: single unpolarized source
        ideal_vis = np.array([[1, 0, 0, 1]], dtype=complex)
        freqs = np.array([1e9])
        times = np.array([0.0])
        ant1 = np.array([0])
        ant2 = np.array([1])

        corrupted = self.simulator.corrupt_visibilities(ideal_vis, freqs, times, ant1, ant2)

        # Both antennas have same gains: (2,1.5)
        # XX correlation: 2 * 2* = 4
        # YY correlation: 1.5 * 1.5* = 2.25
        # XY, YX remain 0 for unpolarized source with diagonal gains
        expected = np.array([[4.0, 0, 0, 2.25]], dtype=complex)

        np.testing.assert_allclose(corrupted, expected)

    def test_visibility_corruption_different_antennas(self):
        """Test corruption with different parameters per antenna."""
        def gain_func(freq, time, ant_id):
            # Different gains per antenna
            if ant_id == 0:
                return 2.0 + 0j
            else:
                return 1.0 + 0j

        gains = ElectronicGains(gain_func, gain_func)
        self.simulator.add_effect('gains', gains)

        ideal_vis = np.array([[1, 0, 0, 1]], dtype=complex)
        freqs = np.array([1e9])
        times = np.array([0.0])
        ant1 = np.array([0])  # Antenna 0: gain=2
        ant2 = np.array([1])  # Antenna 1: gain=1

        corrupted = self.simulator.corrupt_visibilities(ideal_vis, freqs, times, ant1, ant2)

        # XX: 2 * 1* = 2
        # YY: 2 * 1* = 2
        expected = np.array([[2.0, 0, 0, 2.0]], dtype=complex)

        np.testing.assert_allclose(corrupted, expected)