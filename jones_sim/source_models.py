"""Source models for radio interferometry simulations.

This module provides classes for different polarization states and handles
conversion from Stokes parameters to linear correlation format [XX, XY, YX, YY].
"""

import numpy as np
from typing import Union, Tuple
from abc import ABC, abstractmethod


class SourceModel(ABC):
    """Base class for all source models."""

    @abstractmethod
    def stokes_parameters(self) -> Tuple[float, float, float, float]:
        """Return Stokes parameters [I, Q, U, V]."""
        pass

    @abstractmethod
    def linear_correlations(self) -> np.ndarray:
        """Return linear correlations [XX, XY, YX, YY]."""
        pass


def stokes_to_linear(I: float, Q: float, U: float, V: float) -> np.ndarray:
    """Convert Stokes parameters to linear correlations.

    Conversion from Stokes [I, Q, U, V] to linear correlations [XX, XY, YX, YY].
    Standard radio astronomy convention for XY linear feeds.

    Args:
        I, Q, U, V: Stokes parameters

    Returns:
        Array of linear correlations [XX, XY, YX, YY]
    """
    XX = I + Q
    YY = I - Q
    XY = U + 1j * V
    YX = U - 1j * V

    return np.array([XX, XY, YX, YY], dtype=complex)


class UnpolarizedSource(SourceModel):
    """Unpolarized point source with Stokes I only."""

    def __init__(self, flux_density: float):
        """Initialize unpolarized source.

        Args:
            flux_density: Total intensity (Stokes I) in Jy
        """
        self.flux_density = flux_density

    def stokes_parameters(self) -> Tuple[float, float, float, float]:
        """Return Stokes parameters [I, Q, U, V]."""
        return (self.flux_density, 0.0, 0.0, 0.0)

    def linear_correlations(self) -> np.ndarray:
        """Return linear correlations [XX, XY, YX, YY]."""
        I, Q, U, V = self.stokes_parameters()
        return stokes_to_linear(I, Q, U, V)


class LinearPolarizedSource(SourceModel):
    """Linearly polarized point source."""

    def __init__(self,
                 flux_density: float,
                 polarization_fraction: float,
                 position_angle: float):
        """Initialize linearly polarized source.

        Args:
            flux_density: Total intensity (Stokes I) in Jy
            polarization_fraction: Linear polarization fraction (0-1)
            position_angle: Position angle in radians (measured from north toward east)
        """
        self.flux_density = flux_density
        self.pol_frac = polarization_fraction
        self.pa = position_angle

        if not 0 <= polarization_fraction <= 1:
            raise ValueError("Polarization fraction must be between 0 and 1")

    def stokes_parameters(self) -> Tuple[float, float, float, float]:
        """Return Stokes parameters [I, Q, U, V]."""
        I = self.flux_density

        # Linear polarization magnitude
        pol_intensity = I * self.pol_frac

        # Convert to Q, U using position angle
        # Q = p * cos(2*PA), U = p * sin(2*PA)
        Q = pol_intensity * np.cos(2 * self.pa)
        U = pol_intensity * np.sin(2 * self.pa)
        V = 0.0  # No circular polarization

        return (I, Q, U, V)

    def linear_correlations(self) -> np.ndarray:
        """Return linear correlations [XX, XY, YX, YY]."""
        I, Q, U, V = self.stokes_parameters()
        return stokes_to_linear(I, Q, U, V)


class RMSource(SourceModel):
    """Linearly polarized source affected by rotation measure.

    Note: This applies the RM rotation directly to the source Stokes parameters,
    representing intrinsic source RM (e.g., from source environment).
    Propagation RM should be handled via the RotationMeasure Jones matrix.
    """

    def __init__(self,
                 flux_density: float,
                 polarization_fraction: float,
                 intrinsic_position_angle: float,
                 rotation_measure: float,
                 frequency: float):
        """Initialize RM-affected linearly polarized source.

        Args:
            flux_density: Total intensity (Stokes I) in Jy
            polarization_fraction: Linear polarization fraction (0-1)
            intrinsic_position_angle: Intrinsic PA in radians (before RM rotation)
            rotation_measure: Rotation measure in rad/m²
            frequency: Observing frequency in Hz
        """
        self.flux_density = flux_density
        self.pol_frac = polarization_fraction
        self.intrinsic_pa = intrinsic_position_angle
        self.rm = rotation_measure
        self.frequency = frequency

        if not 0 <= polarization_fraction <= 1:
            raise ValueError("Polarization fraction must be between 0 and 1")

    def stokes_parameters(self) -> Tuple[float, float, float, float]:
        """Return Stokes parameters [I, Q, U, V] after RM rotation."""
        I = self.flux_density

        # Calculate RM rotation: Δχ = RM * λ²
        c = 299792458.0  # m/s
        wavelength = c / self.frequency  # meters
        rm_rotation = self.rm * wavelength**2  # radians

        # Observed position angle after RM rotation
        observed_pa = self.intrinsic_pa + rm_rotation

        # Linear polarization magnitude
        pol_intensity = I * self.pol_frac

        # Convert to Q, U using observed position angle
        Q = pol_intensity * np.cos(2 * observed_pa)
        U = pol_intensity * np.sin(2 * observed_pa)
        V = 0.0  # No circular polarization

        return (I, Q, U, V)

    def linear_correlations(self) -> np.ndarray:
        """Return linear correlations [XX, XY, YX, YY]."""
        I, Q, U, V = self.stokes_parameters()
        return stokes_to_linear(I, Q, U, V)


class CircularPolarizedSource(SourceModel):
    """Circularly polarized source with optional linear component."""

    def __init__(self,
                 flux_density: float,
                 circular_fraction: float,
                 linear_fraction: float = 0.0,
                 position_angle: float = 0.0,
                 handedness: str = 'right'):
        """Initialize circularly polarized source.

        Args:
            flux_density: Total intensity (Stokes I) in Jy
            circular_fraction: Circular polarization fraction (0-1)
            linear_fraction: Linear polarization fraction (0-1)
            position_angle: Linear polarization PA in radians (if linear_fraction > 0)
            handedness: 'right' or 'left' for circular polarization sign
        """
        self.flux_density = flux_density
        self.circ_frac = circular_fraction
        self.lin_frac = linear_fraction
        self.pa = position_angle
        self.handedness = handedness.lower()

        if not 0 <= circular_fraction <= 1:
            raise ValueError("Circular polarization fraction must be between 0 and 1")
        if not 0 <= linear_fraction <= 1:
            raise ValueError("Linear polarization fraction must be between 0 and 1")
        if circular_fraction + linear_fraction > 1:
            raise ValueError("Total polarization fraction cannot exceed 1")
        if handedness not in ['right', 'left']:
            raise ValueError("Handedness must be 'right' or 'left'")

    def stokes_parameters(self) -> Tuple[float, float, float, float]:
        """Return Stokes parameters [I, Q, U, V]."""
        I = self.flux_density

        # Linear polarization component
        lin_intensity = I * self.lin_frac
        Q = lin_intensity * np.cos(2 * self.pa)
        U = lin_intensity * np.sin(2 * self.pa)

        # Circular polarization component
        circ_intensity = I * self.circ_frac
        if self.handedness == 'right':
            V = circ_intensity  # Right-handed (positive V)
        else:
            V = -circ_intensity  # Left-handed (negative V)

        return (I, Q, U, V)

    def linear_correlations(self) -> np.ndarray:
        """Return linear correlations [XX, XY, YX, YY]."""
        I, Q, U, V = self.stokes_parameters()
        return stokes_to_linear(I, Q, U, V)


# Convenience functions for common source configurations

def create_unpolarized_source(flux_jy: float = 1.0) -> UnpolarizedSource:
    """Create standard unpolarized calibrator source."""
    return UnpolarizedSource(flux_jy)


def create_linear_source(flux_jy: float = 1.0,
                        pol_percent: float = 5.0,
                        pa_degrees: float = 30.0) -> LinearPolarizedSource:
    """Create linearly polarized source with percentage and degrees."""
    return LinearPolarizedSource(
        flux_jy,
        pol_percent / 100.0,
        np.radians(pa_degrees)
    )


def create_rm_source(flux_jy: float = 1.0,
                    pol_percent: float = 5.0,
                    pa_degrees: float = 30.0,
                    rm_rad_per_m2: float = 25.0,
                    freq_hz: float = 1e9) -> RMSource:
    """Create RM-affected linearly polarized source."""
    return RMSource(
        flux_jy,
        pol_percent / 100.0,
        np.radians(pa_degrees),
        rm_rad_per_m2,
        freq_hz
    )


def create_circular_source(flux_jy: float = 1.0,
                          circ_percent: float = 10.0,
                          lin_percent: float = 2.0,
                          pa_degrees: float = 0.0,
                          handedness: str = 'right') -> CircularPolarizedSource:
    """Create circularly polarized source with small linear component."""
    return CircularPolarizedSource(
        flux_jy,
        circ_percent / 100.0,
        lin_percent / 100.0,
        np.radians(pa_degrees),
        handedness
    )