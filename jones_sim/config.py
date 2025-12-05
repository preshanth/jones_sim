"""JSON configuration system for Jones simulator.

Supports loading simulation configurations from JSON files including:
- Effect chains and ordering
- Parameter distributions
- Noise settings
- GPU/processing options
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .effects import (
    BandpassDelay,
    CrosshandPhase,
    ElectronicGains,
    InstrumentalLeakage,
    ParallacticAngle,
    RLDelayDifference,
    RotationMeasure,
)
from .simulator import JonesSimulator


class DistributionSampler:
    """Sample values from distribution specifications."""

    @staticmethod
    def sample(
        dist_spec: Union[float, Dict[str, Any]], size: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        """Sample from a distribution specification.

        Args:
            dist_spec: Distribution specification (scalar or dict)
            size: Number of samples (None for scalar)

        Returns:
            Sampled value(s)
        """
        # Scalar value
        if isinstance(dist_spec, (int, float)):
            if size is None:
                return float(dist_spec)
            return np.full(size, dist_spec)

        # Distribution specification
        dist_type = dist_spec.get("distribution", "constant")

        if dist_type == "constant":
            value = dist_spec.get("value", 0.0)
            if size is None:
                return float(value)
            return np.full(size, value)

        elif dist_type == "uniform":
            min_val = dist_spec["min"]
            max_val = dist_spec["max"]
            if size is None:
                return np.random.uniform(min_val, max_val)
            return np.random.uniform(min_val, max_val, size)

        elif dist_type == "gaussian" or dist_type == "normal":
            mean = dist_spec.get("mean", 0.0)
            std = dist_spec.get("std", 1.0)
            if size is None:
                return np.random.normal(mean, std)
            return np.random.normal(mean, std, size)

        elif dist_type == "log_normal":
            mean = dist_spec.get("mean", 1.0)
            std = dist_spec.get("std", 0.1)
            # Convert to log-space parameters
            mu = np.log(mean)
            sigma = std / mean  # Approximate for small std
            if size is None:
                return np.random.lognormal(mu, sigma)
            return np.random.lognormal(mu, sigma, size)

        elif dist_type == "complex_gaussian":
            mean_real = dist_spec.get("mean_real", 0.0)
            mean_imag = dist_spec.get("mean_imag", 0.0)
            std_real = dist_spec.get("std_real", 1.0)
            std_imag = dist_spec.get("std_imag", 1.0)

            if size is None:
                real_part = np.random.normal(mean_real, std_real)
                imag_part = np.random.normal(mean_imag, std_imag)
            else:
                real_part = np.random.normal(mean_real, std_real, size)
                imag_part = np.random.normal(mean_imag, std_imag, size)

            return real_part + 1j * imag_part

        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")


class JonesConfig:
    """Configuration loader and parser for Jones simulator."""

    def __init__(self, config_path: Optional[Union[str, Path, Dict]] = None):
        """Initialize from config file or dictionary.

        Args:
            config_path: Path to JSON config file or config dictionary
        """
        if config_path is None:
            self.config = self._default_config()
        elif isinstance(config_path, dict):
            self.config = config_path
        else:
            with open(config_path, "r") as f:
                self.config = json.load(f)

        # Set random seed if specified
        processing_config = self.config.get("processing", {})
        if "random_seed" in processing_config:
            np.random.seed(processing_config["random_seed"])

    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            "metadata": {
                "description": "Default Jones configuration",
                "version": "1.0",
            },
            "jones_chain": {"order": [], "enabled_effects": []},
            "effects": {},
            "noise": {"enabled": False},
            "processing": {"use_gpu": False, "chunk_size_rows": 100000},
        }

    def create_simulator(self, n_antennas: int) -> JonesSimulator:
        """Create and configure JonesSimulator from config.

        Args:
            n_antennas: Number of antennas in the array

        Returns:
            Configured JonesSimulator instance
        """
        sim = JonesSimulator()

        # Get enabled effects
        jones_chain = self.config.get("jones_chain", {})
        enabled = set(jones_chain.get("enabled_effects", []))
        effect_order = jones_chain.get("order", [])

        # Process effects in order
        for effect_name in effect_order:
            if effect_name not in enabled:
                continue

            effect_config = self.config["effects"].get(effect_name, {})
            effect_instance = self._create_effect(
                effect_name, effect_config, n_antennas
            )

            if effect_instance is not None:
                sim.add_effect(effect_name, effect_instance)

        return sim

    def _create_effect(
        self, effect_name: str, effect_config: Dict, n_antennas: int
    ) -> Any:
        """Create effect instance from configuration.

        Args:
            effect_name: Name of the effect
            effect_config: Effect configuration dictionary
            n_antennas: Number of antennas

        Returns:
            Effect instance or None
        """
        effect_type = effect_config.get("type", "")
        per_antenna = effect_config.get("per_antenna", False)

        # Determine sample size
        size = n_antennas if per_antenna else None

        # Create effect based on type
        if effect_type == "parallactic_rotation":
            # Parallactic angles usually computed from MS or external
            angles = effect_config.get("angles", 0.0)
            return ParallacticAngle(angles=angles)

        elif effect_type == "feed_leakage":
            d_xy = DistributionSampler.sample(effect_config.get("d_xy", 0.0), size)
            d_yx = DistributionSampler.sample(effect_config.get("d_yx", 0.0), size)
            theta = DistributionSampler.sample(effect_config.get("theta", 0.0), size)

            # Map to d_hv, d_vh for InstrumentalLeakage
            return InstrumentalLeakage(d_hv=d_xy, d_vh=d_yx, theta=theta)

        elif effect_type == "xy_phase_offset":
            phi = DistributionSampler.sample(effect_config.get("phi", 0.0), size)

            # Handle reference antenna
            if per_antenna and size is not None:
                ref_ant = effect_config.get("reference_antenna", 0)
                if isinstance(phi, np.ndarray):
                    phi[ref_ant] = 0.0

            return CrosshandPhase(phi=phi)

        elif effect_type == "xy_differential_delay":
            delta_tau = DistributionSampler.sample(
                effect_config.get("delta_tau", 0.0), size
            )

            # Handle reference antenna
            if per_antenna and size is not None:
                ref_ant = effect_config.get("reference_antenna", 0)
                if isinstance(delta_tau, np.ndarray):
                    delta_tau[ref_ant] = 0.0

            ref_freq = effect_config.get("ref_freq", 1e9)
            return RLDelayDifference(delta_tau=delta_tau, ref_freq=ref_freq)

        elif effect_type == "bandpass_amplitude_delay":
            # Sample delays
            delay_config = effect_config.get("delay", {})
            tau_x = DistributionSampler.sample(delay_config.get("tau_x", 0.0), size)
            tau_y = DistributionSampler.sample(delay_config.get("tau_y", 0.0), size)

            # Handle reference antenna
            if per_antenna and size is not None:
                ref_ant = effect_config.get("reference_antenna", 0)
                if isinstance(tau_x, np.ndarray):
                    tau_x[ref_ant] = 0.0
                if isinstance(tau_y, np.ndarray):
                    tau_y[ref_ant] = 0.0

            ref_freq = effect_config.get("ref_freq", 0.0)

            # TODO: Implement amplitude modulation (trapezoidal, ripple, etc.)
            # For now, just return delay effect
            return BandpassDelay(tau_xx=tau_x, tau_yy=tau_y, ref_freq=ref_freq)

        elif effect_type == "complex_electronic_gain":
            amp_config = effect_config.get("amplitude", {})
            phase_config = effect_config.get("phase", {})

            # Sample amplitudes
            amp_x = DistributionSampler.sample(amp_config.get("x_pol", 1.0), size)
            amp_y = DistributionSampler.sample(amp_config.get("y_pol", 1.0), size)

            # Sample phases
            phase_x = DistributionSampler.sample(phase_config.get("x_pol", 0.0), size)
            phase_y = DistributionSampler.sample(phase_config.get("y_pol", 0.0), size)

            # Combine to complex gains
            g_xx = amp_x * np.exp(1j * phase_x)
            g_yy = amp_y * np.exp(1j * phase_y)

            # Handle reference antenna
            if per_antenna and size is not None:
                ref_ant = effect_config.get("reference_antenna", 0)
                if isinstance(g_xx, np.ndarray):
                    g_xx[ref_ant] = 1.0 + 0j
                if isinstance(g_yy, np.ndarray):
                    g_yy[ref_ant] = 1.0 + 0j

            return ElectronicGains(g_xx=g_xx, g_yy=g_yy)

        elif effect_type == "rotation_measure":
            rm = DistributionSampler.sample(
                effect_config.get("rotation_measure", 0.0), size
            )
            return RotationMeasure(rotation_measure=rm)

        else:
            print(f"Warning: Unknown effect type '{effect_type}' for '{effect_name}'")
            return None

    def get_noise_config(self) -> Optional[Dict]:
        """Get noise configuration.

        Returns:
            Noise parameters dict or None if disabled
        """
        noise_config = self.config.get("noise", {})
        if not noise_config.get("enabled", False):
            return None

        thermal = noise_config.get("thermal_noise", {})
        return {
            "tsys": thermal.get("tsys_kelvin", 50.0),
            "aperture_eff": thermal.get("aperture_efficiency", 0.7),
            "antenna_diameter": thermal.get("antenna_diameter_meters", 25.0),
            "seed": noise_config.get("random_seed"),
        }

    def get_processing_config(self) -> Dict:
        """Get processing configuration.

        Returns:
            Processing parameters dict
        """
        proc = self.config.get("processing", {})
        return {
            "use_gpu": proc.get("use_gpu", False),
            "chunk_size_rows": proc.get("chunk_size_rows", 100000),
            "batch_gpu_size": proc.get("batch_gpu_size", 10000),
            "random_seed": proc.get("random_seed"),
            "gpu_device": proc.get("gpu_device", 0),  # Which GPU to use
        }

    def save(self, output_path: Union[str, Path]) -> None:
        """Save configuration to JSON file.

        Args:
            output_path: Output file path
        """
        with open(output_path, "w") as f:
            json.dump(self.config, f, indent=2)


def load_config(config_path: Union[str, Path, Dict]) -> JonesConfig:
    """Load configuration from file or dictionary.

    Args:
        config_path: Path to JSON file or config dictionary

    Returns:
        JonesConfig instance
    """
    return JonesConfig(config_path)
