"""Configuration parser for Jones matrix corruption.

Loads JSON config and generates all Jones parameters from distributions.
"""

import json
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path


class ConfigParser:
    """Parse Jones corruption config and generate parameters."""

    def __init__(self, config_path: str, random_seed: Optional[int] = None):
        """Initialize parser.

        Args:
            config_path: Path to JSON config file
            random_seed: Random seed for reproducibility
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Set random seed from config if not provided
        if random_seed is None:
            random_seed = self.config.get("processing", {}).get("random_seed", None)

        if random_seed is not None:
            np.random.seed(random_seed)
            print(f" Random seed set to: {random_seed}")

    def _load_config(self) -> Dict:
        """Load and validate JSON config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = json.load(f)

        print(f" Loaded config: {self.config_path}")

        # Validate required fields
        self._validate_config(config)

        return config

    def _validate_config(self, config: Dict):
        """Validate config structure."""
        required_fields = ["jones_chain", "effects", "processing"]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: '{field}'")

        # Validate jones_chain
        if "order" not in config["jones_chain"]:
            raise ValueError("jones_chain must have 'order' field")
        if "enabled_effects" not in config["jones_chain"]:
            raise ValueError("jones_chain must have 'enabled_effects' field")

        # Check that enabled effects exist in order
        order = config["jones_chain"]["order"]
        enabled = config["jones_chain"]["enabled_effects"]

        for effect in enabled:
            if effect not in order:
                raise ValueError(f"Enabled effect '{effect}' not in chain order")
            if effect not in config["effects"]:
                raise ValueError(
                    f"Enabled effect '{effect}' not defined in effects section"
                )

        print(f" Config validation passed")

    def get_chain_order(self) -> list:
        """Get Jones chain order."""
        return self.config["jones_chain"]["order"]

    def get_enabled_effects(self) -> list:
        """Get list of enabled effects."""
        return self.config["jones_chain"]["enabled_effects"]

    def get_processing_config(self) -> Dict:
        """Get processing configuration."""
        return self.config.get("processing", {})

    def get_noise_config(self) -> Dict:
        """Get noise configuration."""
        return self.config.get("noise", {})

    def generate_all_parameters(
        self, n_antennas: int, n_channels_per_spw: Dict[int, int]
    ) -> Dict:
        """Generate all Jones parameters from config.

        Args:
            n_antennas: Number of antennas
            n_channels_per_spw: Dict mapping SPW ID to number of channels

        Returns:
            Dictionary with all parameters for each effect
        """
        print(f"\n{'=' * 70}")
        print(f"GENERATING JONES PARAMETERS")
        print(f"{'=' * 70}")
        print(f"Antennas: {n_antennas}")
        print(f"SPWs: {list(n_channels_per_spw.keys())}")

        params = {}
        enabled = self.get_enabled_effects()

        # Generate parameters for each enabled effect
        if "gain" in enabled:
            params["gain"] = self._generate_gain_parameters(n_antennas)

        if "bandpass" in enabled:
            params["bandpass"] = self._generate_bandpass_parameters(
                n_antennas, n_channels_per_spw
            )

        if "xy_delay" in enabled:
            params["xy_delay"] = self._generate_xy_delay_parameters(n_antennas)

        if "crosshand_phase" in enabled:
            params["crosshand_phase"] = self._generate_crosshand_phase_parameters(
                n_antennas
            )

        if "leakage" in enabled:
            params["leakage"] = self._generate_leakage_parameters(n_antennas)

        if "parallactic" in enabled:
            print(f"\n Parallactic: Will be computed from MS metadata")
            params["parallactic"] = {"computed_from_ms": True}

        return params

    def _generate_gain_parameters(self, n_antennas: int) -> Dict:
        """Generate complex gain parameters."""
        print(f"\n--- GAIN ---")

        gain_config = self.config["effects"]["gain"]

        # X polarization amplitude
        amp_x = self._sample_distribution(gain_config["amplitude"]["x_pol"], n_antennas)

        # Y polarization amplitude
        amp_y = self._sample_distribution(gain_config["amplitude"]["y_pol"], n_antennas)

        # X polarization phase
        phase_x = self._sample_distribution(gain_config["phase"]["x_pol"], n_antennas)

        # Y polarization phase
        phase_y = self._sample_distribution(gain_config["phase"]["y_pol"], n_antennas)

        print(f"  Amplitude X: mean={np.mean(amp_x):.3f}, std={np.std(amp_x):.3f}")
        print(f"  Amplitude Y: mean={np.mean(amp_y):.3f}, std={np.std(amp_y):.3f}")
        print(
            f"  Phase X: mean={np.mean(phase_x):.3f} rad, std={np.std(phase_x):.3f} rad"
        )
        print(
            f"  Phase Y: mean={np.mean(phase_y):.3f} rad, std={np.std(phase_y):.3f} rad"
        )

        return {
            "amplitude_x": amp_x,
            "amplitude_y": amp_y,
            "phase_x": phase_x,
            "phase_y": phase_y,
        }

    def _generate_bandpass_parameters(
        self, n_antennas: int, n_channels_per_spw: Dict[int, int]
    ) -> Dict:
        """Generate bandpass parameters."""
        print(f"\n--- BANDPASS ---")

        bandpass_config = self.config["effects"]["bandpass"]

        # Delay parameters (scalar per antenna, per pol)
        tau_x = self._sample_distribution(bandpass_config["delay"]["tau_x"], n_antennas)

        tau_y = self._sample_distribution(bandpass_config["delay"]["tau_y"], n_antennas)

        print(
            f"  Delay X: mean={np.mean(tau_x) * 1e9:.3f} ns, std={np.std(tau_x) * 1e9:.3f} ns"
        )
        print(
            f"  Delay Y: mean={np.mean(tau_y) * 1e9:.3f} ns, std={np.std(tau_y) * 1e9:.3f} ns"
        )

        # Amplitude parameters (per antenna, per pol, per channel, per SPW)
        amplitude_x_per_spw = {}
        amplitude_y_per_spw = {}

        for spw_id, n_channels in n_channels_per_spw.items():
            print(f"  SPW {spw_id}: {n_channels} channels")

            # Generate amplitude response for each antenna
            amp_x_spw = np.zeros((n_antennas, n_channels))
            amp_y_spw = np.zeros((n_antennas, n_channels))

            for ant in range(n_antennas):
                amp_x_spw[ant, :] = self._generate_bandpass_amplitude(
                    n_channels, bandpass_config["amplitude"]["x_pol"]
                )
                amp_y_spw[ant, :] = self._generate_bandpass_amplitude(
                    n_channels, bandpass_config["amplitude"]["y_pol"]
                )

            amplitude_x_per_spw[spw_id] = amp_x_spw
            amplitude_y_per_spw[spw_id] = amp_y_spw

            print(
                f"    X-pol: mean={np.mean(amp_x_spw):.3f}, std={np.std(amp_x_spw):.3f}"
            )
            print(
                f"    Y-pol: mean={np.mean(amp_y_spw):.3f}, std={np.std(amp_y_spw):.3f}"
            )

        return {
            "tau_x": tau_x,
            "tau_y": tau_y,
            "amplitude_x": amplitude_x_per_spw,
            "amplitude_y": amplitude_y_per_spw,
        }

    def _generate_bandpass_amplitude(
        self, n_channels: int, amp_config: Dict
    ) -> np.ndarray:
        """Generate bandpass amplitude response for one antenna/pol.

        Args:
            n_channels: Number of frequency channels
            amp_config: Amplitude configuration dict

        Returns:
            Bandpass amplitude (real, positive, length n_channels)
        """
        model = amp_config.get("model", "flat")

        if model == "flat":
            return np.ones(n_channels)

        elif model == "trapezoidal":
            edge_fraction = amp_config.get("edge_fraction", 0.1)
            edge_rolloff = amp_config.get("edge_rolloff", 0.8)
            ripple_amplitude = amp_config.get("passband_ripple_amplitude", 0.0)

            return self._generate_trapezoidal_bandpass(
                n_channels, edge_fraction, edge_rolloff, ripple_amplitude
            )

        else:
            raise ValueError(f"Unknown bandpass model: {model}")

    def _generate_trapezoidal_bandpass(
        self,
        n_channels: int,
        edge_fraction: float,
        edge_rolloff: float,
        ripple_amplitude: float,
    ) -> np.ndarray:
        """Generate trapezoidal bandpass with optional ripple.

        Args:
            n_channels: Number of channels
            edge_fraction: Fraction of band in edge rolloff (each side)
            edge_rolloff: Amplitude at band edges (e.g., 0.8)
            ripple_amplitude: Amplitude of passband ripples

        Returns:
            Bandpass amplitude response
        """
        edge_channels = int(n_channels * edge_fraction)

        # Build base trapezoid
        bandpass = np.ones(n_channels)

        if edge_channels > 0:
            # Left edge: smooth transition from edge_rolloff to 1.0
            left_edge = np.linspace(edge_rolloff, 1.0, edge_channels)
            bandpass[:edge_channels] = left_edge

            # Right edge: smooth transition from 1.0 to edge_rolloff
            right_edge = np.linspace(1.0, edge_rolloff, edge_channels)
            bandpass[-edge_channels:] = right_edge

        # Add passband ripples if requested
        if ripple_amplitude > 0 and edge_channels < n_channels // 2:
            # Random phase for this antenna
            phase = np.random.uniform(0, 2 * np.pi)

            # Sinusoidal ripple (period ~10 channels)
            ripple_freq = 2 * np.pi / 10
            ripple = ripple_amplitude * np.sin(
                ripple_freq * np.arange(n_channels) + phase
            )

            # Apply ripple only in passband
            passband_start = edge_channels
            passband_end = n_channels - edge_channels
            bandpass[passband_start:passband_end] *= (
                1.0 + ripple[passband_start:passband_end]
            )

        return bandpass

    def _generate_xy_delay_parameters(self, n_antennas: int) -> Dict:
        """Generate XY differential delay parameters."""
        print(f"\n--- XY DELAY ---")

        xy_delay_config = self.config["effects"]["xy_delay"]
        ref_antenna = xy_delay_config.get("reference_antenna", 0)

        delta_tau = self._sample_distribution(xy_delay_config["delta_tau"], n_antennas)

        # Reference antenna has zero delay
        delta_tau[ref_antenna] = 0.0

        print(f"  Reference antenna: {ref_antenna}")
        print(
            f"  Δτ: mean={np.mean(delta_tau) * 1e9:.3f} ns, std={np.std(delta_tau) * 1e9:.3f} ns"
        )
        print(f"  Δτ[{ref_antenna}] = {delta_tau[ref_antenna]:.3e} s (reference)")

        return {
            "delta_tau": delta_tau,
            "reference_antenna": ref_antenna,
        }

    def _generate_crosshand_phase_parameters(self, n_antennas: int) -> Dict:
        """Generate cross-hand phase parameters."""
        print(f"\n--- CROSSHAND PHASE ---")

        crosshand_config = self.config["effects"]["crosshand_phase"]
        ref_antenna = crosshand_config.get("reference_antenna", 0)

        phi = self._sample_distribution(crosshand_config["phi"], n_antennas)

        # Reference antenna has zero phase
        phi[ref_antenna] = 0.0

        print(f"  Reference antenna: {ref_antenna}")
        print(f"  φ: mean={np.mean(phi):.3f} rad, std={np.std(phi):.3f} rad")
        print(f"  φ[{ref_antenna}] = {phi[ref_antenna]:.3f} rad (reference)")

        return {
            "phi": phi,
            "reference_antenna": ref_antenna,
        }

    def _generate_leakage_parameters(self, n_antennas: int) -> Dict:
        """Generate leakage (d-term) parameters."""
        print(f"\n--- LEAKAGE ---")

        leakage_config = self.config["effects"]["leakage"]

        # d_xy: X->Y leakage
        d_xy = self._sample_complex_distribution(leakage_config["d_xy"], n_antennas)

        # d_yx: Y->X leakage
        d_yx = self._sample_complex_distribution(leakage_config["d_yx"], n_antennas)

        print(
            f"  d_xy: |d|_mean={np.mean(np.abs(d_xy)):.4f}, |d|_max={np.max(np.abs(d_xy)):.4f}"
        )
        print(
            f"  d_yx: |d|_mean={np.mean(np.abs(d_yx)):.4f}, |d|_max={np.max(np.abs(d_yx)):.4f}"
        )

        return {
            "d_xy": d_xy,
            "d_yx": d_yx,
        }

    def _sample_distribution(self, param_config: Dict, size: int) -> np.ndarray:
        """Sample from a distribution specification.

        Args:
            param_config: Parameter config with distribution info
            size: Number of samples

        Returns:
            Array of sampled values
        """
        # Check for explicit values first
        if "values" in param_config and param_config["values"] is not None:
            values = np.array(param_config["values"])
            if len(values) != size:
                raise ValueError(
                    f"Explicit values length {len(values)} != required size {size}"
                )
            return values

        distribution = param_config.get("distribution", "constant")

        if distribution == "constant":
            value = param_config.get("value", 1.0)
            return np.full(size, value)

        elif distribution == "uniform":
            min_val = param_config["min"]
            max_val = param_config["max"]
            return np.random.uniform(min_val, max_val, size)

        elif distribution == "gaussian" or distribution == "normal":
            mean = param_config["mean"]
            std = param_config["std"]
            return np.random.normal(mean, std, size)

        elif distribution == "log_normal":
            mean = param_config["mean"]
            std = param_config["std"]
            # For log-normal: if we want mean=μ, use log(μ) - σ²/2
            mu_log = np.log(mean) - 0.5 * std**2
            return np.random.lognormal(mu_log, std, size)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def _sample_complex_distribution(self, param_config: Dict, size: int) -> np.ndarray:
        """Sample complex values from distribution.

        Args:
            param_config: Config with real/imag distributions
            size: Number of samples

        Returns:
            Array of complex values
        """
        # Check for explicit values first
        if "values" in param_config and param_config["values"] is not None:
            values = np.array(param_config["values"], dtype=complex)
            if len(values) != size:
                raise ValueError(
                    f"Explicit values length {len(values)} != required size {size}"
                )
            return values

        distribution = param_config.get("distribution", "complex_gaussian")

        if distribution == "complex_gaussian":
            # Sample real and imaginary parts independently
            mean_real = param_config.get("mean_real", 0.0)
            mean_imag = param_config.get("mean_imag", 0.0)
            std_real = param_config.get("std_real", 0.01)
            std_imag = param_config.get("std_imag", 0.01)

            real_part = np.random.normal(mean_real, std_real, size)
            imag_part = np.random.normal(mean_imag, std_imag, size)

            return real_part + 1j * imag_part

        elif distribution == "amplitude_phase":
            # Sample amplitude and phase separately
            amp_mean = param_config.get("amplitude_mean", 0.01)
            amp_std = param_config.get("amplitude_std", 0.005)

            # Rayleigh distribution for amplitude
            amplitudes = np.random.rayleigh(amp_mean, size)

            # Uniform phase
            phases = np.random.uniform(-np.pi, np.pi, size)

            return amplitudes * np.exp(1j * phases)

        else:
            raise ValueError(f"Unknown complex distribution: {distribution}")

    def print_summary(self):
        """Print config summary."""
        print(f"\n{'=' * 70}")
        print(f"CONFIG SUMMARY")
        print(f"{'=' * 70}")
        print(f"Chain order: {self.get_chain_order()}")
        print(f"Enabled effects: {self.get_enabled_effects()}")

        proc = self.get_processing_config()
        print(f"\nProcessing:")
        print(f"  GPU: {proc.get('use_gpu', False)}")
        print(f"  Chunk size: {proc.get('chunk_size_rows', 100000):,} rows")
        print(f"  Random seed: {proc.get('random_seed', 'None')}")

        noise = self.get_noise_config()
        if noise.get("enabled", False):
            print(f"\nNoise:")
            thermal = noise.get("thermal_noise", {})
            print(f"  Tsys: {thermal.get('tsys_kelvin', 'N/A')} K")
            print(f"  Aperture efficiency: {thermal.get('aperture_efficiency', 'N/A')}")
            print(
                f"  Antenna diameter: {thermal.get('antenna_diameter_meters', 'N/A')} m"
            )
        print(f"{'=' * 70}\n")


def load_config(config_path: str, random_seed: Optional[int] = None) -> ConfigParser:
    """Convenience function to load config.

    Args:
        config_path: Path to JSON config file
        random_seed: Optional random seed

    Returns:
        ConfigParser instance
    """
    return ConfigParser(config_path, random_seed)
