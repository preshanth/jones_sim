"""Individual Jones matrix effect classes for numerical forward modeling."""

from typing import Callable, Union

import numpy as np


class ParallacticAngle:
    """Parallactic angle rotation effect."""

    def __init__(self, angles: Union[float, np.ndarray, Callable]):
        """Initialize with parallactic angles.

        Args:
            angles: Scalar, array, or callable(time, antenna_id) -> angle in radians
        """
        self.angles = angles

    def jones_matrix(self, freq: float, time: float, antenna_id: int) -> np.ndarray:
        """Generate 2x2 parallactic angle rotation matrix."""
        if callable(self.angles):
            psi = self.angles(time, antenna_id)
        elif isinstance(self.angles, np.ndarray):
            psi = self.angles[antenna_id]
        else:
            psi = self.angles

        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        return np.array([[cos_psi, sin_psi], [-sin_psi, cos_psi]], dtype=complex)


class ElectronicGains:
    """Electronic gains (amplitude and phase)."""

    def __init__(
        self,
        g_xx: Union[complex, np.ndarray, Callable],
        g_yy: Union[complex, np.ndarray, Callable],
    ):
        """Initialize with complex gains.

        Args:
            g_xx: XX polarization gains
            g_yy: YY polarization gains
            Each can be scalar, array, or callable(freq, time, antenna_id) -> complex
        """
        self.g_xx = g_xx
        self.g_yy = g_yy

    def jones_matrix(self, freq: float, time: float, antenna_id: int) -> np.ndarray:
        """Generate 2x2 diagonal gains matrix."""
        g_xx = self._get_value(self.g_xx, freq, time, antenna_id)
        g_yy = self._get_value(self.g_yy, freq, time, antenna_id)

        return np.array([[g_xx, 0], [0, g_yy]], dtype=complex)

    def _get_value(self, param, freq: float, time: float, antenna_id: int) -> complex:
        if callable(param):
            return complex(param(freq, time, antenna_id))
        elif isinstance(param, np.ndarray):
            return complex(param[antenna_id])
        else:
            return complex(param)


class InstrumentalLeakage:
    """Instrumental polarization leakage and misalignment."""

    def __init__(
        self,
        d_hv: Union[complex, np.ndarray, Callable] = 0.0,
        d_vh: Union[complex, np.ndarray, Callable] = 0.0,
        theta: Union[float, np.ndarray, Callable] = 0.0,
    ):
        """Initialize leakage parameters.

        Args:
            d_hv: H->V leakage (complex)
            d_vh: V->H leakage (complex)
            theta: Misalignment angle in radians
        """
        self.d_hv = d_hv
        self.d_vh = d_vh
        self.theta = theta

    def jones_matrix(self, freq: float, time: float, antenna_id: int) -> np.ndarray:
        """Generate 2x2 leakage matrix."""
        d_hv = self._get_value(self.d_hv, freq, time, antenna_id)
        d_vh = self._get_value(self.d_vh, freq, time, antenna_id)
        theta = self._get_value(self.theta, freq, time, antenna_id)

        # Combined leakage and misalignment
        T_hv = d_hv + np.tan(theta)
        T_vh = d_vh - np.tan(theta)

        return np.array([[1, T_hv], [T_vh, 1]], dtype=complex)

    def _get_value(self, param, freq: float, time: float, antenna_id: int):
        if callable(param):
            return param(freq, time, antenna_id)
        elif isinstance(param, np.ndarray):
            return param[antenna_id]
        else:
            return param


class BandpassDelay:
    """Bandpass and delay effects."""

    def __init__(
        self,
        tau_xx: Union[float, np.ndarray, Callable] = 0.0,
        tau_yy: Union[float, np.ndarray, Callable] = 0.0,
        ref_freq: float = 1e9,
    ):
        """Initialize delay parameters.

        Args:
            tau_xx: XX delay in seconds
            tau_yy: YY delay in seconds
            ref_freq: Reference frequency in Hz
        """
        self.tau_xx = tau_xx
        self.tau_yy = tau_yy
        self.ref_freq = ref_freq

    def jones_matrix(self, freq: float, time: float, antenna_id: int) -> np.ndarray:
        """Generate 2x2 frequency-dependent delay matrix."""
        tau_xx = self._get_value(self.tau_xx, freq, time, antenna_id)
        tau_yy = self._get_value(self.tau_yy, freq, time, antenna_id)

        phase_xx = 2j * np.pi * tau_xx * (freq - self.ref_freq)
        phase_yy = 2j * np.pi * tau_yy * (freq - self.ref_freq)

        return np.array([[np.exp(phase_xx), 0], [0, np.exp(phase_yy)]], dtype=complex)

    def _get_value(self, param, freq: float, time: float, antenna_id: int):
        if callable(param):
            return param(freq, time, antenna_id)
        elif isinstance(param, np.ndarray):
            return param[antenna_id]
        else:
            return param


class RLDelayDifference:
    """R/L delay difference effect."""

    def __init__(
        self, delta_tau: Union[float, np.ndarray, Callable] = 0.0, ref_freq: float = 1e9
    ):
        """Initialize R/L delay difference.

        Args:
            delta_tau: R-L delay difference in seconds
            ref_freq: Reference frequency in Hz
        """
        self.delta_tau = delta_tau
        self.ref_freq = ref_freq

    def jones_matrix(self, freq: float, time: float, antenna_id: int) -> np.ndarray:
        """Generate 2x2 R/L delay matrix."""
        delta_tau = self._get_value(self.delta_tau, freq, time, antenna_id)

        delta_theta = 2 * np.pi * delta_tau * (freq - self.ref_freq)
        cos_half = np.cos(delta_theta / 2)
        sin_half = np.sin(delta_theta / 2)

        return np.array(
            [[cos_half, 1j * sin_half], [-1j * sin_half, cos_half]], dtype=complex
        )

    def _get_value(self, param, freq: float, time: float, antenna_id: int):
        if callable(param):
            return param(freq, time, antenna_id)
        elif isinstance(param, np.ndarray):
            return param[antenna_id]
        else:
            return param


class RotationMeasure:
    """Faraday rotation due to rotation measure."""

    def __init__(self, rotation_measure: Union[float, np.ndarray, Callable] = 0.0):
        """Initialize rotation measure effect.

        Args:
            rotation_measure: Rotation measure in rad/m² (scalar, array, or callable)
        """
        self.rotation_measure = rotation_measure
        self.c = 299792458.0  # Speed of light in m/s

    def jones_matrix(self, freq: float, time: float, antenna_id: int) -> np.ndarray:
        """Generate 2x2 Faraday rotation matrix.

        The rotation angle is RM * λ² where λ = c/ν.
        """
        rm = self._get_value(self.rotation_measure, freq, time, antenna_id)

        # Calculate wavelength squared: λ² = (c/ν)²
        lambda_sq = (self.c / freq) ** 2

        # Rotation angle = RM * λ²
        angle = rm * lambda_sq

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        return np.array(
            [[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=complex
        )

    def _get_value(self, param, freq: float, time: float, antenna_id: int):
        if callable(param):
            return param(freq, time, antenna_id)
        elif isinstance(param, np.ndarray):
            return param[antenna_id]
        else:
            return param


class CrosshandPhase:
    """Cross-hand phase offset effect."""

    def __init__(self, phi: Union[float, np.ndarray, Callable] = 0.0):
        """Initialize cross-hand phase.

        Args:
            phi: Phase offset in radians
        """
        self.phi = phi

    def jones_matrix(self, freq: float, time: float, antenna_id: int) -> np.ndarray:
        """Generate 2x2 cross-hand phase matrix."""
        phi = self._get_value(self.phi, freq, time, antenna_id)

        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)

    def _get_value(self, param, freq: float, time: float, antenna_id: int):
        if callable(param):
            return param(freq, time, antenna_id)
        elif isinstance(param, np.ndarray):
            return param[antenna_id]
        else:
            return param
