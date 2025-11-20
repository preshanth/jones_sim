"""Solvable calibration effects for Bayesian inference.

Each effect implements:
- sample_params(): NumPyro sampling statements for inference
- apply(): Forward model (corrupt visibilities)
- apply_inverse(): Correction (divide out effect)

Effects are applied in Jones chain order:
P (Parallactic) → D (Leakage) → G (Gain) → B (Bandpass) → K (Delay) → F (Faraday)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist


class SolvableEffect(ABC):
    """Base class for solvable calibration effects."""

    @abstractmethod
    def sample_params(
        self, n_antennas: int, casa_values: Any, prior_config: Dict[str, Any], **kwargs
    ) -> jnp.ndarray:
        """Sample parameters using NumPyro.

        Args:
            n_antennas: Number of antennas
            casa_values: CASA solution as prior center
            prior_config: Prior configuration (bounds, std, etc.)
            **kwargs: Additional data (frequencies, times, etc.)

        Returns:
            Sampled parameters as JAX array
        """
        pass

    @abstractmethod
    def apply(
        self,
        vis: jnp.ndarray,
        params: jnp.ndarray,
        ant1: jnp.ndarray,
        ant2: jnp.ndarray,
        freqs: jnp.ndarray,
        times: Optional[jnp.ndarray] = None,
        n_pol: int = 4,
    ) -> jnp.ndarray:
        """Apply effect to visibilities (forward/corruption).

        Args:
            vis: Visibilities, shape (n_row, n_chan, n_pol)
            params: Effect parameters
            ant1: Antenna 1 indices
            ant2: Antenna 2 indices
            freqs: Frequencies in Hz
            times: Times (for time-dependent effects)
            n_pol: Number of polarizations

        Returns:
            Corrupted visibilities
        """
        pass

    @abstractmethod
    def apply_inverse(
        self,
        vis: np.ndarray,
        params: np.ndarray,
        ant1: np.ndarray,
        ant2: np.ndarray,
        freqs: np.ndarray,
        times: Optional[np.ndarray] = None,
        n_pol: int = 4,
    ) -> np.ndarray:
        """Apply inverse (correction) to visibilities.

        Args:
            vis: Visibilities, shape (n_pol, n_chan, n_row)
            params: Effect parameters
            ant1: Antenna 1 indices
            ant2: Antenna 2 indices
            freqs: Frequencies in Hz
            times: Times (for time-dependent effects)
            n_pol: Number of polarizations

        Returns:
            Corrected visibilities
        """
        pass

    def get_default_prior_config(self) -> Dict[str, Any]:
        """Return default prior configuration."""
        return {}

    @abstractmethod
    def extract_solution(
        self,
        trace: Dict[str, np.ndarray],
        n_antennas: int,
        casa_values: Any,
    ) -> np.ndarray:
        """Extract solution from trace.

        Args:
            trace: MCMC/MAP trace dictionary
            n_antennas: Number of antennas
            casa_values: CASA values (for reference antenna)

        Returns:
            Solution array
        """
        pass


class DelayEffect(SolvableEffect):
    """K - Delay effect.

    Applies phase rotation proportional to frequency:
    phase = 2π * (τ1 - τ2) * ν
    """

    def get_default_prior_config(self) -> Dict[str, Any]:
        return {"bound_ns": 1.0}

    def extract_solution(self, trace, n_antennas, casa_values):
        delays_free = np.mean(trace["delays_free"], axis=0)
        delays = np.zeros(n_antennas)
        delays[0] = casa_values[0]
        delays[1:] = delays_free
        return delays

    def sample_params(
        self,
        n_antennas: int,
        casa_values: jnp.ndarray,
        prior_config: Dict[str, Any],
        **kwargs,
    ) -> jnp.ndarray:
        bound_ns = prior_config.get("bound_ns", 1.0)
        bound_sec = bound_ns * 1e-9

        # Sample free antennas (ref ant fixed)
        delays_free = numpyro.sample(
            "delays_free",
            dist.Uniform(
                casa_values[1:] - bound_sec,
                casa_values[1:] + bound_sec,
            ),
        )

        # Concatenate with fixed reference antenna
        delays = jnp.concatenate([casa_values[:1], delays_free])
        return delays

    def apply(
        self,
        vis: jnp.ndarray,
        params: jnp.ndarray,
        ant1: jnp.ndarray,
        ant2: jnp.ndarray,
        freqs: jnp.ndarray,
        times: Optional[jnp.ndarray] = None,
        n_pol: int = 4,
    ) -> jnp.ndarray:
        # params = delays in seconds, shape (n_antennas,)
        delay_diff = params[ant1] - params[ant2]  # (n_row,)
        phase = 2 * jnp.pi * delay_diff[:, None] * freqs[None, :]  # (n_row, n_chan)
        # Apply same rotation to all pols
        return vis * jnp.exp(1j * phase)[:, :, None]

    def apply_inverse(
        self,
        vis: np.ndarray,
        params: np.ndarray,
        ant1: np.ndarray,
        ant2: np.ndarray,
        freqs: np.ndarray,
        times: Optional[np.ndarray] = None,
        n_pol: int = 4,
    ) -> np.ndarray:
        # vis shape: (n_pol, n_chan, n_row)
        delay_diff = params[ant1] - params[ant2]  # (n_row,)
        phase = 2 * np.pi * delay_diff[:, None] * freqs[None, :]  # (n_row, n_chan)
        correction = np.exp(-1j * phase)  # (n_row, n_chan)
        return vis * correction.T[None, :, :]  # broadcast to (n_pol, n_chan, n_row)


class GainEffect(SolvableEffect):
    """G - Electronic gain effect.

    Complex gains per antenna and polarization.
    """

    def get_default_prior_config(self) -> Dict[str, Any]:
        return {"prior_std": 0.1, "calmode": "ap"}

    def extract_solution(self, trace, n_antennas, casa_values):
        amp = np.mean(trace["gain_amp"], axis=0)
        phase = np.mean(trace["gain_phase"], axis=0)
        gains = np.ones((n_antennas, 2), dtype=complex)
        gains[0] = casa_values[0]
        gains[1:] = amp * np.exp(1j * phase)
        return gains

    def sample_params(
        self,
        n_antennas: int,
        casa_values: jnp.ndarray,
        prior_config: Dict[str, Any],
        **kwargs,
    ) -> jnp.ndarray:
        # casa_values shape: (n_antennas, 2) complex
        prior_std = prior_config.get("prior_std", 0.1)
        calmode = prior_config.get("calmode", "ap")

        # Amplitude (log-normal)
        if "a" in calmode:
            amp = numpyro.sample(
                "gain_amp",
                dist.LogNormal(
                    jnp.log(jnp.abs(casa_values[1:, :])),
                    prior_std,
                ),
            )
        else:
            amp = jnp.abs(casa_values[1:, :])

        # Phase
        if "p" in calmode:
            phase = numpyro.sample(
                "gain_phase",
                dist.Normal(
                    jnp.angle(casa_values[1:, :]),
                    prior_std,
                ),
            )
        else:
            phase = jnp.angle(casa_values[1:, :])

        # Combine and add reference antenna
        gains_free = amp * jnp.exp(1j * phase)
        gains = jnp.concatenate([casa_values[:1, :], gains_free], axis=0)
        return gains

    def apply(
        self,
        vis: jnp.ndarray,
        params: jnp.ndarray,
        ant1: jnp.ndarray,
        ant2: jnp.ndarray,
        freqs: jnp.ndarray,
        times: Optional[jnp.ndarray] = None,
        n_pol: int = 4,
    ) -> jnp.ndarray:
        # params = gains, shape (n_antennas, 2) complex
        g1 = params[ant1]  # (n_row, 2)
        g2 = params[ant2]  # (n_row, 2)

        if n_pol == 4:
            gf = jnp.stack(
                [
                    g1[:, 0] * jnp.conj(g2[:, 0]),  # XX or RR
                    g1[:, 0] * jnp.conj(g2[:, 1]),  # XY or RL
                    g1[:, 1] * jnp.conj(g2[:, 0]),  # YX or LR
                    g1[:, 1] * jnp.conj(g2[:, 1]),  # YY or LL
                ],
                axis=1,
            )  # (n_row, 4)
        else:
            gf = jnp.stack(
                [
                    g1[:, 0] * jnp.conj(g2[:, 0]),
                    g1[:, 1] * jnp.conj(g2[:, 1]),
                ],
                axis=1,
            )  # (n_row, 2)

        # Broadcast to all channels: (n_row, n_chan, n_pol)
        return vis * gf[:, None, :]

    def apply_inverse(
        self,
        vis: np.ndarray,
        params: np.ndarray,
        ant1: np.ndarray,
        ant2: np.ndarray,
        freqs: np.ndarray,
        times: Optional[np.ndarray] = None,
        n_pol: int = 4,
    ) -> np.ndarray:
        # vis shape: (n_pol, n_chan, n_row)
        g1 = params[ant1]  # (n_row, 2)
        g2 = params[ant2]  # (n_row, 2)

        if n_pol == 4:
            gf = np.stack(
                [
                    g1[:, 0] * np.conj(g2[:, 0]),
                    g1[:, 0] * np.conj(g2[:, 1]),
                    g1[:, 1] * np.conj(g2[:, 0]),
                    g1[:, 1] * np.conj(g2[:, 1]),
                ],
                axis=1,
            )  # (n_row, 4)
        else:
            gf = np.stack(
                [
                    g1[:, 0] * np.conj(g2[:, 0]),
                    g1[:, 1] * np.conj(g2[:, 1]),
                ],
                axis=1,
            )  # (n_row, 2)

        # Divide out gains
        return vis / gf.T[:, None, :]  # (n_pol, n_chan, n_row)


class BandpassEffect(SolvableEffect):
    """B - Bandpass effect.

    Frequency-dependent complex gains per antenna and polarization.
    """

    def get_default_prior_config(self) -> Dict[str, Any]:
        return {"prior_std": 0.05, "calmode": "ap"}

    def extract_solution(self, trace, n_antennas, casa_values):
        amp = np.mean(trace["bandpass_amp"], axis=0)
        phase = np.mean(trace["bandpass_phase"], axis=0)
        n_chan = amp.shape[1]
        bp = np.ones((n_antennas, n_chan, 2), dtype=complex)
        bp[0] = casa_values[0]
        bp[1:] = amp * np.exp(1j * phase)
        return bp

    def sample_params(
        self,
        n_antennas: int,
        casa_values: jnp.ndarray,
        prior_config: Dict[str, Any],
        **kwargs,
    ) -> jnp.ndarray:
        # casa_values shape: (n_antennas, n_chan, 2) complex
        prior_std = prior_config.get("prior_std", 0.05)
        calmode = prior_config.get("calmode", "ap")

        # Amplitude
        if "a" in calmode:
            amp = numpyro.sample(
                "bandpass_amp",
                dist.LogNormal(
                    jnp.log(jnp.abs(casa_values[1:, :, :])),
                    prior_std,
                ),
            )
        else:
            amp = jnp.abs(casa_values[1:, :, :])

        # Phase
        if "p" in calmode:
            phase = numpyro.sample(
                "bandpass_phase",
                dist.Normal(
                    jnp.angle(casa_values[1:, :, :]),
                    prior_std,
                ),
            )
        else:
            phase = jnp.angle(casa_values[1:, :, :])

        # Combine
        bp_free = amp * jnp.exp(1j * phase)
        bandpass = jnp.concatenate([casa_values[:1, :, :], bp_free], axis=0)
        return bandpass

    def apply(
        self,
        vis: jnp.ndarray,
        params: jnp.ndarray,
        ant1: jnp.ndarray,
        ant2: jnp.ndarray,
        freqs: jnp.ndarray,
        times: Optional[jnp.ndarray] = None,
        n_pol: int = 4,
    ) -> jnp.ndarray:
        # params = bandpass, shape (n_antennas, n_chan, 2) complex
        b1 = params[ant1]  # (n_row, n_chan, 2)
        b2 = params[ant2]  # (n_row, n_chan, 2)

        if n_pol == 4:
            bf = jnp.stack(
                [
                    b1[:, :, 0] * jnp.conj(b2[:, :, 0]),
                    b1[:, :, 0] * jnp.conj(b2[:, :, 1]),
                    b1[:, :, 1] * jnp.conj(b2[:, :, 0]),
                    b1[:, :, 1] * jnp.conj(b2[:, :, 1]),
                ],
                axis=2,
            )  # (n_row, n_chan, 4)
        else:
            bf = jnp.stack(
                [
                    b1[:, :, 0] * jnp.conj(b2[:, :, 0]),
                    b1[:, :, 1] * jnp.conj(b2[:, :, 1]),
                ],
                axis=2,
            )  # (n_row, n_chan, 2)

        return vis * bf

    def apply_inverse(
        self,
        vis: np.ndarray,
        params: np.ndarray,
        ant1: np.ndarray,
        ant2: np.ndarray,
        freqs: np.ndarray,
        times: Optional[np.ndarray] = None,
        n_pol: int = 4,
    ) -> np.ndarray:
        # vis shape: (n_pol, n_chan, n_row)
        b1 = params[ant1]  # (n_row, n_chan, 2)
        b2 = params[ant2]  # (n_row, n_chan, 2)

        if n_pol == 4:
            bf = np.stack(
                [
                    b1[:, :, 0] * np.conj(b2[:, :, 0]),
                    b1[:, :, 0] * np.conj(b2[:, :, 1]),
                    b1[:, :, 1] * np.conj(b2[:, :, 0]),
                    b1[:, :, 1] * np.conj(b2[:, :, 1]),
                ],
                axis=2,
            )  # (n_row, n_chan, 4)
        else:
            bf = np.stack(
                [
                    b1[:, :, 0] * np.conj(b2[:, :, 0]),
                    b1[:, :, 1] * np.conj(b2[:, :, 1]),
                ],
                axis=2,
            )  # (n_row, n_chan, 2)

        return vis / bf.transpose(2, 1, 0)  # (n_pol, n_chan, n_row)


class ParallacticAngleEffect(SolvableEffect):
    """P - Parallactic angle rotation.

    Time-dependent rotation of polarization basis.
    Not typically solved for - computed from geometry.
    """

    def get_default_prior_config(self) -> Dict[str, Any]:
        return {}

    def extract_solution(self, trace, n_antennas, casa_values):
        # P is computed, not sampled - just return casa_values
        return casa_values

    def sample_params(
        self,
        n_antennas: int,
        casa_values: jnp.ndarray,
        prior_config: Dict[str, Any],
        **kwargs,
    ) -> jnp.ndarray:
        # P is computed, not sampled
        # casa_values contains precomputed angles: (n_antennas, n_times)
        return casa_values

    def apply(
        self,
        vis: jnp.ndarray,
        params: jnp.ndarray,
        ant1: jnp.ndarray,
        ant2: jnp.ndarray,
        freqs: jnp.ndarray,
        times: Optional[jnp.ndarray] = None,
        n_pol: int = 4,
    ) -> jnp.ndarray:
        # params = parallactic angles, shape (n_antennas, n_times) or (n_antennas,)
        # For now, assume single time (time-averaged)
        if params.ndim == 1:
            pa1 = params[ant1]  # (n_row,)
            pa2 = params[ant2]  # (n_row,)
        else:
            # Would need time index mapping
            raise NotImplementedError("Time-dependent P not yet implemented")

        # Rotation matrix for linear feeds
        # P = [[cos(pa), -sin(pa)], [sin(pa), cos(pa)]]
        c1, s1 = jnp.cos(pa1), jnp.sin(pa1)
        c2, s2 = jnp.cos(pa2), jnp.sin(pa2)

        if n_pol == 4:
            # P1 ⊗ P2* applied to [XX, XY, YX, YY]
            p_matrix = jnp.stack(
                [
                    c1 * c2 + s1 * s2,  # XX
                    -c1 * s2 + s1 * c2,  # XY
                    -s1 * c2 + c1 * s2,  # YX
                    s1 * s2 + c1 * c2,  # YY
                ],
                axis=1,
            )  # (n_row, 4)
            return vis * p_matrix[:, None, :]
        else:
            # Simplified for 2 pols
            return vis  # TODO: implement 2-pol case

    def apply_inverse(
        self,
        vis: np.ndarray,
        params: np.ndarray,
        ant1: np.ndarray,
        ant2: np.ndarray,
        freqs: np.ndarray,
        times: Optional[np.ndarray] = None,
        n_pol: int = 4,
    ) -> np.ndarray:
        # Inverse of rotation is rotation by -angle
        if params.ndim == 1:
            pa1 = -params[ant1]
            pa2 = -params[ant2]
        else:
            raise NotImplementedError("Time-dependent P not yet implemented")

        c1, s1 = np.cos(pa1), np.sin(pa1)
        c2, s2 = np.cos(pa2), np.sin(pa2)

        if n_pol == 4:
            p_matrix = np.stack(
                [
                    c1 * c2 + s1 * s2,
                    -c1 * s2 + s1 * c2,
                    -s1 * c2 + c1 * s2,
                    s1 * s2 + c1 * c2,
                ],
                axis=1,
            )  # (n_row, 4)
            return vis * p_matrix.T[:, None, :]
        else:
            return vis


class LeakageEffect(SolvableEffect):
    """D - Polarization leakage (D-terms).

    Cross-polarization leakage between feeds.
    """

    def get_default_prior_config(self) -> Dict[str, Any]:
        return {"prior_std": 0.05}

    def extract_solution(self, trace, n_antennas, casa_values):
        d_real = np.mean(trace["leakage_real"], axis=0)
        d_imag = np.mean(trace["leakage_imag"], axis=0)
        d_terms = np.zeros((n_antennas, 2), dtype=complex)
        d_terms[0] = casa_values[0]
        d_terms[1:] = d_real + 1j * d_imag
        return d_terms

    def sample_params(
        self,
        n_antennas: int,
        casa_values: jnp.ndarray,
        prior_config: Dict[str, Any],
        **kwargs,
    ) -> jnp.ndarray:
        # casa_values shape: (n_antennas, 2) complex - Dx, Dy per antenna
        prior_std = prior_config.get("prior_std", 0.05)

        # D-terms are typically small complex numbers
        d_real = numpyro.sample(
            "leakage_real",
            dist.Normal(
                jnp.real(casa_values[1:, :]),
                prior_std,
            ),
        )
        d_imag = numpyro.sample(
            "leakage_imag",
            dist.Normal(
                jnp.imag(casa_values[1:, :]),
                prior_std,
            ),
        )

        d_free = d_real + 1j * d_imag
        d_terms = jnp.concatenate([casa_values[:1, :], d_free], axis=0)
        return d_terms

    def apply(
        self,
        vis: jnp.ndarray,
        params: jnp.ndarray,
        ant1: jnp.ndarray,
        ant2: jnp.ndarray,
        freqs: jnp.ndarray,
        times: Optional[jnp.ndarray] = None,
        n_pol: int = 4,
    ) -> jnp.ndarray:
        # params = D-terms, shape (n_antennas, 2) complex
        # D = [[1, Dx], [Dy, 1]]
        d1 = params[ant1]  # (n_row, 2) - [Dx, Dy]
        d2 = params[ant2]

        if n_pol == 4:
            # D1 ⊗ D2* for [XX, XY, YX, YY]
            # Simplified: assume small D-terms, first order
            df = jnp.stack(
                [
                    jnp.ones(len(ant1)),  # XX: ~1
                    jnp.conj(d2[:, 0]),  # XY: Dx2*
                    d1[:, 1],  # YX: Dy1
                    jnp.ones(len(ant1)) + d1[:, 1] * jnp.conj(d2[:, 0]),  # YY: ~1
                ],
                axis=1,
            )
            return vis * df[:, None, :]
        else:
            return vis

    def apply_inverse(
        self,
        vis: np.ndarray,
        params: np.ndarray,
        ant1: np.ndarray,
        ant2: np.ndarray,
        freqs: np.ndarray,
        times: Optional[np.ndarray] = None,
        n_pol: int = 4,
    ) -> np.ndarray:
        d1 = params[ant1]
        d2 = params[ant2]

        if n_pol == 4:
            df = np.stack(
                [
                    np.ones(len(ant1)),
                    np.conj(d2[:, 0]),
                    d1[:, 1],
                    np.ones(len(ant1)) + d1[:, 1] * np.conj(d2[:, 0]),
                ],
                axis=1,
            )
            return vis / df.T[:, None, :]
        else:
            return vis


class FaradayEffect(SolvableEffect):
    """F - Ionospheric Faraday rotation.

    Frequency-dependent polarization rotation due to ionosphere.
    RM = rotation measure in rad/m^2
    """

    def get_default_prior_config(self) -> Dict[str, Any]:
        return {"prior_std": 0.1}  # rad/m^2

    def extract_solution(self, trace, n_antennas, casa_values):
        rm_free = np.mean(trace["faraday_rm"], axis=0)
        rm = np.zeros(n_antennas)
        rm[0] = casa_values[0]
        rm[1:] = rm_free
        return rm

    def sample_params(
        self,
        n_antennas: int,
        casa_values: jnp.ndarray,
        prior_config: Dict[str, Any],
        **kwargs,
    ) -> jnp.ndarray:
        # casa_values = RM per antenna, shape (n_antennas,)
        prior_std = prior_config.get("prior_std", 0.1)

        rm_free = numpyro.sample(
            "faraday_rm",
            dist.Normal(
                casa_values[1:],
                prior_std,
            ),
        )

        rm = jnp.concatenate([casa_values[:1], rm_free])
        return rm

    def apply(
        self,
        vis: jnp.ndarray,
        params: jnp.ndarray,
        ant1: jnp.ndarray,
        ant2: jnp.ndarray,
        freqs: jnp.ndarray,
        times: Optional[jnp.ndarray] = None,
        n_pol: int = 4,
    ) -> jnp.ndarray:
        # params = RM, shape (n_antennas,)
        # Faraday rotation angle = RM * λ^2 = RM * (c/ν)^2
        c = 299792458.0  # speed of light
        wavelength_sq = (c / freqs) ** 2  # (n_chan,)

        rm_diff = params[ant1] - params[ant2]  # (n_row,)
        angle = rm_diff[:, None] * wavelength_sq[None, :]  # (n_row, n_chan)

        # Rotation in polarization space
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)

        if n_pol == 4:
            # Rotation of [XX, XY, YX, YY]
            # This is a simplification - full treatment needs Mueller matrix
            f_matrix = jnp.stack(
                [
                    cos_a**2,
                    cos_a * sin_a,
                    -cos_a * sin_a,
                    sin_a**2,
                ],
                axis=2,
            )  # (n_row, n_chan, 4)
            return vis * f_matrix
        else:
            return vis

    def apply_inverse(
        self,
        vis: np.ndarray,
        params: np.ndarray,
        ant1: np.ndarray,
        ant2: np.ndarray,
        freqs: np.ndarray,
        times: Optional[np.ndarray] = None,
        n_pol: int = 4,
    ) -> np.ndarray:
        c = 299792458.0
        wavelength_sq = (c / freqs) ** 2

        rm_diff = params[ant1] - params[ant2]
        angle = -rm_diff[:, None] * wavelength_sq[None, :]  # negative for inverse

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        if n_pol == 4:
            f_matrix = np.stack(
                [
                    cos_a**2,
                    cos_a * sin_a,
                    -cos_a * sin_a,
                    sin_a**2,
                ],
                axis=2,
            )
            return vis * f_matrix.transpose(2, 1, 0)
        else:
            return vis


# Effect registry
EFFECT_REGISTRY = {
    "K": DelayEffect,
    "G": GainEffect,
    "B": BandpassEffect,
    "P": ParallacticAngleEffect,
    "D": LeakageEffect,
    "F": FaradayEffect,
}


def get_effect(name: str) -> SolvableEffect:
    """Get effect instance by name."""
    if name not in EFFECT_REGISTRY:
        raise ValueError(
            f"Unknown effect: {name}. Available: {list(EFFECT_REGISTRY.keys())}"
        )
    return EFFECT_REGISTRY[name]()
