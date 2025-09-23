"""Test source model classes."""

import numpy as np
import pytest

from jones_sim.source_models import (
    UnpolarizedSource,
    LinearPolarizedSource,
    RMSource,
    CircularPolarizedSource,
    stokes_to_linear,
    create_unpolarized_source,
    create_linear_source,
    create_rm_source,
    create_circular_source
)


class TestStokesToLinear:
    """Test Stokes to linear correlation conversion."""

    def test_unpolarized_conversion(self):
        """Test conversion for unpolarized source."""
        I, Q, U, V = 1.0, 0.0, 0.0, 0.0
        correlations = stokes_to_linear(I, Q, U, V)

        expected = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex)
        np.testing.assert_allclose(correlations, expected)

    def test_linear_polarization_conversion(self):
        """Test conversion for linearly polarized source."""
        # 100% linear polarization at PA=0 (Q=1, U=0)
        I, Q, U, V = 1.0, 1.0, 0.0, 0.0
        correlations = stokes_to_linear(I, Q, U, V)

        # XX = I + Q = 2, YY = I - Q = 0, XY = YX = U ± iV = 0
        expected = np.array([2.0, 0.0, 0.0, 0.0], dtype=complex)
        np.testing.assert_allclose(correlations, expected)

    def test_circular_polarization_conversion(self):
        """Test conversion for circularly polarized source."""
        # 100% right circular polarization (V=1)
        I, Q, U, V = 1.0, 0.0, 0.0, 1.0
        correlations = stokes_to_linear(I, Q, U, V)

        # XX = YY = I = 1, XY = U + iV = i, YX = U - iV = -i
        expected = np.array([1.0, 1.0j, -1.0j, 1.0], dtype=complex)
        np.testing.assert_allclose(correlations, expected)


class TestUnpolarizedSource:
    """Test unpolarized source model."""

    def test_basic_properties(self):
        """Test basic source properties."""
        flux = 2.5
        source = UnpolarizedSource(flux)

        I, Q, U, V = source.stokes_parameters()
        assert I == flux
        assert Q == 0.0
        assert U == 0.0
        assert V == 0.0

    def test_linear_correlations(self):
        """Test linear correlation output."""
        source = UnpolarizedSource(1.0)
        correlations = source.linear_correlations()

        expected = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex)
        np.testing.assert_allclose(correlations, expected)


class TestLinearPolarizedSource:
    """Test linearly polarized source model."""

    def test_basic_properties(self):
        """Test basic source properties."""
        flux = 1.0
        pol_frac = 0.1
        pa = np.pi / 4  # 45 degrees

        source = LinearPolarizedSource(flux, pol_frac, pa)

        I, Q, U, V = source.stokes_parameters()
        assert I == flux
        assert V == 0.0  # No circular polarization

        # Check Q, U for 45° PA
        expected_pol = flux * pol_frac
        expected_Q = expected_pol * np.cos(2 * pa)  # cos(π/2) = 0
        expected_U = expected_pol * np.sin(2 * pa)  # sin(π/2) = 1

        np.testing.assert_allclose(Q, expected_Q, atol=1e-15)
        np.testing.assert_allclose(U, expected_U, atol=1e-10)

    def test_position_angle_zero(self):
        """Test PA=0 gives pure Q polarization."""
        source = LinearPolarizedSource(1.0, 0.1, 0.0)
        I, Q, U, V = source.stokes_parameters()

        assert I == 1.0
        np.testing.assert_allclose(Q, 0.1, atol=1e-15)
        np.testing.assert_allclose(U, 0.0, atol=1e-15)
        assert V == 0.0

    def test_invalid_polarization_fraction(self):
        """Test error handling for invalid polarization fraction."""
        with pytest.raises(ValueError):
            LinearPolarizedSource(1.0, 1.5, 0.0)  # > 1

        with pytest.raises(ValueError):
            LinearPolarizedSource(1.0, -0.1, 0.0)  # < 0


class TestRMSource:
    """Test rotation measure affected source model."""

    def test_zero_rm(self):
        """Test that zero RM gives same result as LinearPolarizedSource."""
        flux = 1.0
        pol_frac = 0.1
        pa = np.pi / 6  # 30 degrees
        freq = 1e9

        # Compare with LinearPolarizedSource
        linear_source = LinearPolarizedSource(flux, pol_frac, pa)
        rm_source = RMSource(flux, pol_frac, pa, 0.0, freq)

        linear_stokes = linear_source.stokes_parameters()
        rm_stokes = rm_source.stokes_parameters()

        np.testing.assert_allclose(linear_stokes, rm_stokes, atol=1e-15)

    def test_rm_rotation(self):
        """Test RM rotation effect."""
        flux = 1.0
        pol_frac = 1.0  # 100% for easier testing
        intrinsic_pa = 0.0  # Start with Q polarization
        rm = 10.0  # rad/m²
        freq = 1e9  # 1 GHz

        source = RMSource(flux, pol_frac, intrinsic_pa, rm, freq)
        I, Q, U, V = source.stokes_parameters()

        # Calculate expected rotation
        c = 299792458.0
        lambda_sq = (c / freq) ** 2
        rotation = rm * lambda_sq
        observed_pa = intrinsic_pa + rotation

        # Expected Stokes parameters
        expected_Q = pol_frac * np.cos(2 * observed_pa)
        expected_U = pol_frac * np.sin(2 * observed_pa)

        assert I == flux
        np.testing.assert_allclose(Q, expected_Q, rtol=1e-12)
        np.testing.assert_allclose(U, expected_U, rtol=1e-12)
        assert V == 0.0

    def test_frequency_dependence(self):
        """Test that RM rotation scales with λ²."""
        source_params = (1.0, 1.0, 0.0, 10.0)  # flux, pol_frac, PA, RM

        freq1 = 1e9
        freq2 = 2e9  # Double frequency

        source1 = RMSource(*source_params, freq1)
        source2 = RMSource(*source_params, freq2)

        _, Q1, U1, _ = source1.stokes_parameters()
        _, Q2, U2, _ = source2.stokes_parameters()

        # Rotation should scale as 1/freq² (λ² dependence)
        # At 2x frequency, rotation should be 1/4 as much
        rotation1 = np.arctan2(U1, Q1) / 2  # Divide by 2 for PA
        rotation2 = np.arctan2(U2, Q2) / 2

        expected_ratio = (freq2 / freq1) ** 2  # Should be 4
        actual_ratio = rotation1 / rotation2

        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-10)


class TestCircularPolarizedSource:
    """Test circularly polarized source model."""

    def test_pure_circular(self):
        """Test pure circular polarization."""
        flux = 1.0
        circ_frac = 0.2

        # Right-handed
        source_rcp = CircularPolarizedSource(flux, circ_frac, 0.0, 0.0, 'right')
        I, Q, U, V = source_rcp.stokes_parameters()

        assert I == flux
        assert Q == 0.0
        assert U == 0.0
        assert V == circ_frac * flux

        # Left-handed
        source_lcp = CircularPolarizedSource(flux, circ_frac, 0.0, 0.0, 'left')
        I, Q, U, V = source_lcp.stokes_parameters()

        assert I == flux
        assert Q == 0.0
        assert U == 0.0
        assert V == -circ_frac * flux

    def test_mixed_polarization(self):
        """Test circular + linear polarization."""
        flux = 1.0
        circ_frac = 0.1
        lin_frac = 0.05
        pa = np.pi / 4  # 45 degrees

        source = CircularPolarizedSource(flux, circ_frac, lin_frac, pa, 'right')
        I, Q, U, V = source.stokes_parameters()

        assert I == flux

        # Linear component
        expected_Q = lin_frac * flux * np.cos(2 * pa)
        expected_U = lin_frac * flux * np.sin(2 * pa)

        # Circular component
        expected_V = circ_frac * flux

        np.testing.assert_allclose(Q, expected_Q, atol=1e-15)
        np.testing.assert_allclose(U, expected_U, atol=1e-10)
        assert V == expected_V

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        # Circular fraction > 1
        with pytest.raises(ValueError):
            CircularPolarizedSource(1.0, 1.5, 0.0)

        # Linear fraction > 1
        with pytest.raises(ValueError):
            CircularPolarizedSource(1.0, 0.1, 1.5)

        # Total polarization > 1
        with pytest.raises(ValueError):
            CircularPolarizedSource(1.0, 0.8, 0.3)  # 0.8 + 0.3 > 1

        # Invalid handedness
        with pytest.raises(ValueError):
            CircularPolarizedSource(1.0, 0.1, 0.0, 0.0, 'invalid')


class TestConvenienceFunctions:
    """Test convenience functions for source creation."""

    def test_create_unpolarized_source(self):
        """Test unpolarized source creation."""
        source = create_unpolarized_source(2.0)
        I, Q, U, V = source.stokes_parameters()

        assert I == 2.0
        assert Q == 0.0
        assert U == 0.0
        assert V == 0.0

    def test_create_linear_source(self):
        """Test linear source creation with percentage/degrees."""
        source = create_linear_source(1.0, 10.0, 45.0)  # 10%, 45°

        I, Q, U, V = source.stokes_parameters()
        assert I == 1.0
        assert V == 0.0

        # Check conversion from percentage and degrees
        expected_pol = 0.1 * I
        expected_Q = expected_pol * np.cos(2 * np.pi / 4)  # cos(π/2) = 0
        expected_U = expected_pol * np.sin(2 * np.pi / 4)  # sin(π/2) = 1

        np.testing.assert_allclose(Q, expected_Q, atol=1e-15)
        np.testing.assert_allclose(U, expected_U, atol=1e-10)

    def test_create_rm_source(self):
        """Test RM source creation."""
        source = create_rm_source(1.0, 5.0, 30.0, 25.0, 1e9)

        I, Q, U, V = source.stokes_parameters()
        assert I == 1.0
        assert V == 0.0
        # Q, U should be non-zero due to RM rotation
        assert abs(Q) + abs(U) > 0

    def test_create_circular_source(self):
        """Test circular source creation."""
        source = create_circular_source(1.0, 15.0, 3.0, 0.0, 'left')

        I, Q, U, V = source.stokes_parameters()
        assert I == 1.0
        np.testing.assert_allclose(Q, 0.03, atol=1e-15)  # 3% linear at PA=0
        np.testing.assert_allclose(U, 0.0, atol=1e-15)
        np.testing.assert_allclose(V, -0.15, atol=1e-15)  # 15% left circular