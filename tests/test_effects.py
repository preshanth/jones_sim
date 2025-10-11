"""Test individual Jones matrix effect classes."""

import numpy as np

from jones_sim.effects import (
    BandpassDelay,
    CrosshandPhase,
    ElectronicGains,
    InstrumentalLeakage,
    ParallacticAngle,
    RLDelayDifference,
    RotationMeasure,
)


class TestParallacticAngle:
    """Test parallactic angle effect."""

    def test_scalar_angle(self):
        """Test with scalar parallactic angle."""
        psi = np.pi / 4  # 45 degrees
        effect = ParallacticAngle(psi)

        matrix = effect.jones_matrix(1e9, 0.0, 0)

        expected = np.array(
            [[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]], dtype=complex
        )

        np.testing.assert_allclose(matrix, expected)

    def test_array_angles(self):
        """Test with array of angles per antenna."""
        angles = np.array([0, np.pi / 2, np.pi])
        effect = ParallacticAngle(angles)

        # Test antenna 1 (90 degrees)
        matrix = effect.jones_matrix(1e9, 0.0, 1)
        expected = np.array([[0, 1], [-1, 0]], dtype=complex)

        np.testing.assert_allclose(matrix, expected, atol=1e-15)

    def test_callable_angle(self):
        """Test with callable angle function."""

        def angle_func(time, ant_id):
            return time * ant_id * np.pi / 180  # Simple function

        effect = ParallacticAngle(angle_func)
        matrix = effect.jones_matrix(1e9, 30.0, 2)  # 60 degrees

        expected_angle = 60 * np.pi / 180
        expected = np.array(
            [
                [np.cos(expected_angle), np.sin(expected_angle)],
                [-np.sin(expected_angle), np.cos(expected_angle)],
            ],
            dtype=complex,
        )

        np.testing.assert_allclose(matrix, expected)


class TestElectronicGains:
    """Test electronic gains effect."""

    def test_scalar_gains(self):
        """Test with scalar complex gains."""
        g_xx = 2.0 + 0.5j
        g_yy = 1.5 - 0.3j
        effect = ElectronicGains(g_xx, g_yy)

        matrix = effect.jones_matrix(1e9, 0.0, 0)
        expected = np.array([[g_xx, 0], [0, g_yy]], dtype=complex)

        np.testing.assert_allclose(matrix, expected)

    def test_array_gains(self):
        """Test with array of gains per antenna."""
        g_xx = np.array([1 + 0j, 2 + 0.5j, 0.8 - 0.2j])
        g_yy = np.array([1 + 0j, 1.5 - 0.3j, 1.2 + 0.1j])
        effect = ElectronicGains(g_xx, g_yy)

        matrix = effect.jones_matrix(1e9, 0.0, 2)
        expected = np.array([[0.8 - 0.2j, 0], [0, 1.2 + 0.1j]], dtype=complex)

        np.testing.assert_allclose(matrix, expected)

    def test_real_gains(self):
        """Test with real gains (should be converted to complex)."""
        effect = ElectronicGains(2.0, 1.5)

        matrix = effect.jones_matrix(1e9, 0.0, 0)
        expected = np.array([[2.0 + 0j, 0], [0, 1.5 + 0j]], dtype=complex)

        np.testing.assert_allclose(matrix, expected)


class TestInstrumentalLeakage:
    """Test instrumental leakage effect."""

    def test_leakage_only(self):
        """Test with only leakage terms (no misalignment)."""
        d_hv = 0.05 + 0.02j
        d_vh = 0.03 - 0.01j
        effect = InstrumentalLeakage(d_hv, d_vh, theta=0.0)

        matrix = effect.jones_matrix(1e9, 0.0, 0)
        expected = np.array([[1, d_hv], [d_vh, 1]], dtype=complex)

        np.testing.assert_allclose(matrix, expected)

    def test_misalignment_only(self):
        """Test with only misalignment (no leakage)."""
        theta = 0.1  # radians
        effect = InstrumentalLeakage(d_hv=0.0, d_vh=0.0, theta=theta)

        matrix = effect.jones_matrix(1e9, 0.0, 0)
        expected = np.array([[1, np.tan(theta)], [-np.tan(theta), 1]], dtype=complex)

        np.testing.assert_allclose(matrix, expected)

    def test_combined_leakage_misalignment(self):
        """Test combined leakage and misalignment."""
        d_hv = 0.05j
        d_vh = 0.03j
        theta = 0.1
        effect = InstrumentalLeakage(d_hv, d_vh, theta)

        matrix = effect.jones_matrix(1e9, 0.0, 0)

        T_hv = d_hv + np.tan(theta)
        T_vh = d_vh - np.tan(theta)
        expected = np.array([[1, T_hv], [T_vh, 1]], dtype=complex)

        np.testing.assert_allclose(matrix, expected)


class TestBandpassDelay:
    """Test bandpass delay effect."""

    def test_no_delay_at_reference(self):
        """Test that delay is zero at reference frequency."""
        tau_xx = 1e-9  # 1 ns delay
        tau_yy = 2e-9  # 2 ns delay
        ref_freq = 1e9
        effect = BandpassDelay(tau_xx, tau_yy, ref_freq)

        matrix = effect.jones_matrix(ref_freq, 0.0, 0)
        expected = np.eye(2, dtype=complex)

        np.testing.assert_allclose(matrix, expected)

    def test_frequency_dependence(self):
        """Test frequency-dependent phase."""
        tau_xx = 1e-9  # 1 ns
        tau_yy = 0.0  # No delay
        ref_freq = 1e9
        freq = 1.1e9  # 100 MHz offset
        effect = BandpassDelay(tau_xx, tau_yy, ref_freq)

        matrix = effect.jones_matrix(freq, 0.0, 0)

        expected_phase = 2j * np.pi * tau_xx * (freq - ref_freq)
        expected = np.array([[np.exp(expected_phase), 0], [0, 1]], dtype=complex)

        np.testing.assert_allclose(matrix, expected)


class TestRLDelayDifference:
    """Test R/L delay difference effect."""

    def test_no_delay_at_reference(self):
        """Test identity matrix at reference frequency."""
        delta_tau = 1e-9  # 1 ns R/L delay
        ref_freq = 1e9
        effect = RLDelayDifference(delta_tau, ref_freq)

        matrix = effect.jones_matrix(ref_freq, 0.0, 0)
        expected = np.eye(2, dtype=complex)

        np.testing.assert_allclose(matrix, expected)

    def test_frequency_dependence(self):
        """Test R/L delay matrix form."""
        delta_tau = 1e-9  # 1 ns
        ref_freq = 1e9
        freq = 1.1e9
        effect = RLDelayDifference(delta_tau, ref_freq)

        matrix = effect.jones_matrix(freq, 0.0, 0)

        delta_theta = 2 * np.pi * delta_tau * (freq - ref_freq)
        cos_half = np.cos(delta_theta / 2)
        sin_half = np.sin(delta_theta / 2)

        expected = np.array(
            [[cos_half, 1j * sin_half], [-1j * sin_half, cos_half]], dtype=complex
        )

        np.testing.assert_allclose(matrix, expected)

    def test_matrix_properties(self):
        """Test that R/L delay matrix is unitary."""
        delta_tau = 5e-9
        ref_freq = 1e9
        freq = 1.5e9
        effect = RLDelayDifference(delta_tau, ref_freq)

        matrix = effect.jones_matrix(freq, 0.0, 0)

        # Check unitarity: J @ J† = I
        identity = matrix @ matrix.conj().T
        np.testing.assert_allclose(identity, np.eye(2), atol=1e-15)


class TestRotationMeasure:
    """Test rotation measure (Faraday rotation) effect."""

    def test_zero_rm(self):
        """Test identity matrix for zero rotation measure."""
        effect = RotationMeasure(rotation_measure=0.0)

        matrix = effect.jones_matrix(1e9, 0.0, 0)
        expected = np.eye(2, dtype=complex)

        np.testing.assert_allclose(matrix, expected)

    def test_known_rm_value(self):
        """Test with known RM value and frequency."""
        rm_value = 25.0  # rad/m²
        freq = 1e9  # 1 GHz
        effect = RotationMeasure(rm_value)

        matrix = effect.jones_matrix(freq, 0.0, 0)

        # Calculate expected rotation angle: RM * λ²
        c = 299792458.0  # m/s
        lambda_sq = (c / freq) ** 2
        expected_angle = rm_value * lambda_sq

        expected = np.array(
            [
                [np.cos(expected_angle), -np.sin(expected_angle)],
                [np.sin(expected_angle), np.cos(expected_angle)],
            ],
            dtype=complex,
        )

        np.testing.assert_allclose(matrix, expected, rtol=1e-12)

    def test_rotation_matrix_properties(self):
        """Test that result is a proper rotation matrix."""
        effect = RotationMeasure(10.0)

        matrix = effect.jones_matrix(1e9, 0.0, 0)

        # Rotation matrices are orthogonal: R @ R.H = I
        identity = matrix @ matrix.conj().T
        np.testing.assert_allclose(identity, np.eye(2), atol=1e-15)

        # Determinant should be 1
        assert np.abs(np.linalg.det(matrix) - 1.0) < 1e-15

    def test_frequency_dependence(self):
        """Test λ² frequency dependence."""
        rm_value = 10.0
        effect = RotationMeasure(rm_value)
        c = 299792458.0  # Speed of light

        freqs = np.array([0.5e9, 1.0e9, 2.0e9])  # 0.5, 1, 2 GHz
        expected_angles = []

        # Calculate expected angles directly
        for freq in freqs:
            lambda_sq = (c / freq) ** 2
            expected_angle = rm_value * lambda_sq
            expected_angles.append(expected_angle)

        # Verify λ² scaling: angle should scale as 1/freq²
        ratio_01 = expected_angles[0] / expected_angles[1]
        expected_ratio_01 = (freqs[1] / freqs[0]) ** 2

        ratio_12 = expected_angles[1] / expected_angles[2]
        expected_ratio_12 = (freqs[2] / freqs[1]) ** 2

        np.testing.assert_allclose(ratio_01, expected_ratio_01, rtol=1e-10)
        np.testing.assert_allclose(ratio_12, expected_ratio_12, rtol=1e-10)

        # Also verify the matrices match expected rotation matrices
        for freq, expected_angle in zip(freqs, expected_angles):
            matrix = effect.jones_matrix(freq, 0.0, 0)
            expected_matrix = np.array(
                [
                    [np.cos(expected_angle), -np.sin(expected_angle)],
                    [np.sin(expected_angle), np.cos(expected_angle)],
                ],
                dtype=complex,
            )
            np.testing.assert_allclose(matrix, expected_matrix, rtol=1e-12)

    def test_array_rm_values(self):
        """Test with array of RM values per antenna."""
        rm_values = np.array([0.0, 10.0, 25.0])
        effect = RotationMeasure(rm_values)

        # Test antenna 0: should be identity
        matrix_0 = effect.jones_matrix(1e9, 0.0, 0)
        np.testing.assert_allclose(matrix_0, np.eye(2), atol=1e-15)

        # Test antenna 1: 10 rad/m²
        matrix_1 = effect.jones_matrix(1e9, 0.0, 1)
        c = 299792458.0
        lambda_sq = (c / 1e9) ** 2
        expected_angle_1 = 10.0 * lambda_sq
        expected_1 = np.array(
            [
                [np.cos(expected_angle_1), -np.sin(expected_angle_1)],
                [np.sin(expected_angle_1), np.cos(expected_angle_1)],
            ],
            dtype=complex,
        )
        np.testing.assert_allclose(matrix_1, expected_1, rtol=1e-12)


class TestCrosshandPhase:
    """Test cross-hand phase effect."""

    def test_zero_phase(self):
        """Test identity matrix for zero phase."""
        effect = CrosshandPhase(phi=0.0)

        matrix = effect.jones_matrix(1e9, 0.0, 0)
        expected = np.eye(2, dtype=complex)

        np.testing.assert_allclose(matrix, expected)

    def test_nonzero_phase(self):
        """Test cross-hand phase matrix."""
        phi = np.pi / 3  # 60 degrees
        effect = CrosshandPhase(phi)

        matrix = effect.jones_matrix(1e9, 0.0, 0)
        expected = np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)

        np.testing.assert_allclose(matrix, expected)

    def test_array_phases(self):
        """Test with array of phases per antenna."""
        phases = np.array([0, np.pi / 2, np.pi])
        effect = CrosshandPhase(phases)

        matrix = effect.jones_matrix(1e9, 0.0, 2)  # Antenna 2: π phase
        expected = np.array([[1, 0], [0, -1]], dtype=complex)

        np.testing.assert_allclose(matrix, expected, atol=1e-15)
