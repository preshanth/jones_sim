"""Unified PyMC sampler for complete Jones matrix chains with JSON configuration."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .simulator import JonesSimulator


class JonesMCSampler:
    """Unified Monte Carlo sampler for complete Jones matrix chains.

    Samples all effects simultaneously across (antenna, time, frequency) space
    with proper correlations and dependencies.
    """

    def __init__(self, config: Union[Dict, str, Path]):
        """Initialize sampler from configuration.

        Args:
            config: Dictionary, JSON string, or path to JSON file with configuration
        """
        if isinstance(config, (str, Path)):
            with open(config) as f:
                self.config = json.load(f)
        else:
            self.config = config

        # Extract grid parameters
        self.n_antennas = self.config["grid"]["n_antennas"]
        self.n_times = self.config["grid"]["n_times"]
        self.n_freqs = self.config["grid"]["n_frequencies"]

        # Time and frequency grids
        time_config = self.config["grid"]["time"]
        self.times = np.linspace(time_config["start"], time_config["end"], self.n_times)

        freq_config = self.config["grid"]["frequency"]
        self.frequencies = np.linspace(
            freq_config["start"], freq_config["end"], self.n_freqs
        )

        # Initialize components
        self.model = None
        self.trace = None
        self.jones_simulator = JonesSimulator()
        self.logger = logging.getLogger(__name__)

    def build_unified_model(self) -> pm.Model:
        """Build unified PyMC model for all enabled effects [Confidence: 85% - Evidence: PyMC patterns].

        Returns:
            Complete PyMC model with all effect parameters
        """

        with pm.Model() as model:

            # === ELECTRONIC GAINS (time-varying) ===
            if "gains" in self.config["effects"]:
                gains_config = self.config["effects"]["gains"]

                # Base gain amplitudes (log-normal for positivity)
                base_amp_xx = pm.LogNormal(
                    "base_amp_xx",
                    mu=np.log(gains_config["base_amplitude"]),
                    sigma=gains_config["amplitude_std"],
                    shape=self.n_antennas,
                )
                base_amp_yy = pm.LogNormal(
                    "base_amp_yy",
                    mu=np.log(gains_config["base_amplitude"] * 0.95),
                    sigma=gains_config["amplitude_std"],
                    shape=self.n_antennas,
                )

                # Thermal drift parameters
                thermal_amp = pm.Normal(
                    "thermal_amplitude",
                    mu=0.0,
                    sigma=gains_config["thermal_amplitude"],
                    shape=self.n_antennas,
                )
                thermal_timescale = gains_config["thermal_timescale"]

                # Phase drift rates
                phase_drift_xx = pm.Normal(
                    "phase_drift_xx",
                    mu=0.0,
                    sigma=gains_config["phase_drift_std"],
                    shape=self.n_antennas,
                )
                phase_drift_yy = pm.Normal(
                    "phase_drift_yy",
                    mu=0.0,
                    sigma=gains_config["phase_drift_std"],
                    shape=self.n_antennas,
                )

                # Time-dependent gains
                time_tensor = pt.as_tensor_variable(self.times)

                # Thermal modulation
                thermal_mod = thermal_amp[:, None] * pt.sin(
                    2 * np.pi * time_tensor[None, :] / thermal_timescale
                )

                # Amplitude evolution
                amp_xx_time = base_amp_xx[:, None] + thermal_mod
                amp_yy_time = (
                    base_amp_yy[:, None] + thermal_mod * 0.8
                )  # Correlated but different

                # Phase evolution
                phase_xx_time = phase_drift_xx[:, None] * time_tensor[None, :]
                phase_yy_time = phase_drift_yy[:, None] * time_tensor[None, :]

                # Complex gains: shape (n_antennas, n_times)
                pm.Deterministic("gains_xx", amp_xx_time * pt.exp(1j * phase_xx_time))
                pm.Deterministic("gains_yy", amp_yy_time * pt.exp(1j * phase_yy_time))

            # === BANDPASS (frequency-varying) ===
            if "bandpass" in self.config["effects"]:
                bp_config = self.config["effects"]["bandpass"]

                # Cable delays (linear phase slope)
                cable_delay = pm.Normal(
                    "cable_delay",
                    mu=0.0,
                    sigma=bp_config["delay_std"],
                    shape=self.n_antennas,
                )

                # Jagged amplitude per channel
                jagged_amp_xx = pm.Normal(
                    "jagged_amp_xx",
                    mu=0.0,
                    sigma=bp_config["jagged_amplitude"],
                    shape=(self.n_antennas, self.n_freqs),
                )
                jagged_amp_yy = pm.Normal(
                    "jagged_amp_yy",
                    mu=0.0,
                    sigma=bp_config["jagged_amplitude"],
                    shape=(self.n_antennas, self.n_freqs),
                )

                # Frequency-dependent phase
                freq_tensor = pt.as_tensor_variable(self.frequencies)
                ref_freq = self.frequencies[self.n_freqs // 2]

                # Cable delay phase
                cable_phase = (
                    2 * np.pi * cable_delay[:, None] * (freq_tensor[None, :] - ref_freq)
                )

                # Total bandpass amplitude (log-space for positivity)
                pm.Deterministic("bandpass_amp_xx", pt.exp(jagged_amp_xx))
                pm.Deterministic("bandpass_amp_yy", pt.exp(jagged_amp_yy))

                # Total bandpass phase
                pm.Deterministic("bandpass_phase_xx", cable_phase)
                pm.Deterministic("bandpass_phase_yy", cable_phase * 1.1)

            # === INSTRUMENTAL LEAKAGE (static) ===
            if "leakage" in self.config["effects"]:
                leak_config = self.config["effects"]["leakage"]

                # Complex leakage terms (small, antenna-dependent)
                d_hv_real = pm.Normal(
                    "d_hv_real",
                    mu=0.0,
                    sigma=leak_config["amplitude"],
                    shape=self.n_antennas,
                )
                d_hv_imag = pm.Normal(
                    "d_hv_imag",
                    mu=0.0,
                    sigma=leak_config["amplitude"],
                    shape=self.n_antennas,
                )
                d_vh_real = pm.Normal(
                    "d_vh_real",
                    mu=0.0,
                    sigma=leak_config["amplitude"],
                    shape=self.n_antennas,
                )
                d_vh_imag = pm.Normal(
                    "d_vh_imag",
                    mu=0.0,
                    sigma=leak_config["amplitude"],
                    shape=self.n_antennas,
                )

                pm.Deterministic("leakage_hv", d_hv_real + 1j * d_hv_imag)
                pm.Deterministic("leakage_vh", d_vh_real + 1j * d_vh_imag)

            # === PARALLACTIC ANGLE (time-varying, deterministic) ===
            if "parallactic" in self.config["effects"]:
                para_config = self.config["effects"]["parallactic"]

                # Parallactic angle evolution (deterministic celestial mechanics)
                para_rate = (
                    para_config["rate_deg_per_hour"] * np.pi / 180 / 3600
                )  # rad/sec
                para_angles = para_rate * self.times  # Simple linear model
                pm.Deterministic(
                    "parallactic_angles", pt.as_tensor_variable(para_angles)
                )

        self.model = model
        return model

    def sample(
        self,
        draws: int = 1000,
        tune: int = 500,
        chains: int = 2,
        target_accept: float = 0.9,
    ) -> az.InferenceData:
        """Sample from unified model [Confidence: 95% - Evidence: Standard PyMC pattern].

        Args:
            draws: Posterior samples per chain
            tune: Tuning samples
            chains: Number of MCMC chains
            target_accept: Target acceptance rate

        Returns:
            ArviZ InferenceData with all samples
        """
        if self.model is None:
            raise ValueError("Must build model first using build_unified_model()")

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=True,
            )

        return self.trace

    def compute_jones_matrices(self, sample_idx: Optional[int] = None) -> np.ndarray:
        """Compute Jones matrices from sampled parameters [Confidence: 80% - Evidence: Matrix multiplication logic].

        Args:
            sample_idx: Specific sample index, or None for all samples

        Returns:
            Jones matrices: shape (n_samples, n_antennas, n_times, n_freqs, 2, 2)
            or (n_antennas, n_times, n_freqs, 2, 2) if sample_idx specified
        """
        if self.trace is None:
            raise ValueError("Must sample first")

        # Extract relevant samples
        if sample_idx is not None:
            # Single sample
            n_samples = 1
            slice(sample_idx, sample_idx + 1)
        else:
            # All samples
            chain_samples = (
                self.trace.posterior.dims["draw"] * self.trace.posterior.dims["chain"]
            )
            n_samples = chain_samples
            slice(None)

        # Initialize result array
        if sample_idx is not None:
            jones_shape = (self.n_antennas, self.n_times, self.n_freqs, 2, 2)
        else:
            jones_shape = (n_samples, self.n_antennas, self.n_times, self.n_freqs, 2, 2)

        jones_matrices = np.zeros(jones_shape, dtype=complex)

        # For each sample, antenna, time, frequency combination
        sample_range = [sample_idx] if sample_idx is not None else range(n_samples)

        for s_idx in sample_range:
            for ant in range(self.n_antennas):
                for t_idx in range(self.n_times):
                    for f_idx in range(self.n_freqs):
                        # Start with identity
                        J = np.eye(2, dtype=complex)

                        # Apply effects in standard order
                        if "parallactic" in self.config["effects"]:
                            psi = self.trace.posterior[
                                "parallactic_angles"
                            ].values.flatten()[t_idx]
                            P = np.array(
                                [
                                    [np.cos(psi), np.sin(psi)],
                                    [-np.sin(psi), np.cos(psi)],
                                ]
                            )
                            J = J @ P

                        if "leakage" in self.config["effects"]:
                            hv = self.trace.posterior["leakage_hv"].values.flatten()[
                                ant
                            ]
                            vh = self.trace.posterior["leakage_vh"].values.flatten()[
                                ant
                            ]
                            X = np.array([[1, hv], [vh, 1]])
                            J = J @ X

                        if "gains" in self.config["effects"]:
                            g_xx = self.trace.posterior["gains_xx"].values.flatten()[
                                ant * self.n_times + t_idx
                            ]
                            g_yy = self.trace.posterior["gains_yy"].values.flatten()[
                                ant * self.n_times + t_idx
                            ]
                            G = np.array([[g_xx, 0], [0, g_yy]])
                            J = J @ G

                        if "bandpass" in self.config["effects"]:
                            amp_xx = self.trace.posterior[
                                "bandpass_amp_xx"
                            ].values.flatten()[ant * self.n_freqs + f_idx]
                            amp_yy = self.trace.posterior[
                                "bandpass_amp_yy"
                            ].values.flatten()[ant * self.n_freqs + f_idx]
                            phase_xx = self.trace.posterior[
                                "bandpass_phase_xx"
                            ].values.flatten()[ant * self.n_freqs + f_idx]
                            phase_yy = self.trace.posterior[
                                "bandpass_phase_yy"
                            ].values.flatten()[ant * self.n_freqs + f_idx]

                            B = np.array(
                                [
                                    [amp_xx * np.exp(1j * phase_xx), 0],
                                    [0, amp_yy * np.exp(1j * phase_yy)],
                                ]
                            )
                            J = J @ B

                        # Store result
                        if sample_idx is not None:
                            jones_matrices[ant, t_idx, f_idx] = J
                        else:
                            jones_matrices[s_idx, ant, t_idx, f_idx] = J

        return jones_matrices

    @classmethod
    def from_json_file(cls, config_path: Union[str, Path]) -> "JonesMCSampler":
        """Create sampler from JSON configuration file [Confidence: 95% - Evidence: Standard factory pattern].

        Args:
            config_path: Path to JSON configuration file

        Returns:
            Configured sampler instance
        """
        return cls(config_path)


def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary [Confidence: 90% - Evidence: Reasonable parameter ranges].

    Returns:
        Default configuration for Jones MC sampling
    """
    return {
        "grid": {
            "n_antennas": 4,
            "n_times": 30,
            "n_frequencies": 64,
            "time": {"start": 0.0, "end": 7200.0, "units": "seconds"},
            "frequency": {"start": 1.3e9, "end": 1.5e9, "units": "Hz"},
        },
        "effects": {
            "gains": {
                "base_amplitude": 1.0,
                "amplitude_std": 0.02,
                "thermal_amplitude": 0.01,
                "thermal_timescale": 3600.0,
                "phase_drift_std": 1e-5,
            },
            "bandpass": {"delay_std": 1e-9, "jagged_amplitude": 0.05},
            "leakage": {"amplitude": 0.001},
            "parallactic": {"rate_deg_per_hour": 15.0},
        },
        "sampling": {"draws": 1000, "tune": 500, "chains": 2, "target_accept": 0.9},
    }


def main():
    """Command-line interface for Jones MC sampling [Confidence: 85% - Evidence: Standard CLI patterns]."""
    parser = argparse.ArgumentParser(description="Jones Matrix Monte Carlo Sampler")
    parser.add_argument("--config", type=str, help="JSON configuration file")
    parser.add_argument(
        "--output", type=str, default="jones_samples.nc", help="Output file for samples"
    )
    parser.add_argument(
        "--create-default-config",
        type=str,
        help="Create default config file at specified path",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate interactive plots dashboard"
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default="jones_dashboard.html",
        help="Output file for plots dashboard",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.create_default_config:
        config = create_default_config()
        with open(args.create_default_config, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Created default configuration: {args.create_default_config}")
        return

    if not args.config:
        print("Error: Must specify --config or --create-default-config")
        return

    print(f"Loading configuration from: {args.config}")
    sampler = JonesMCSampler.from_json_file(args.config)

    print("Building unified model...")
    sampler.build_unified_model()

    # Extract sampling parameters
    sampling_config = sampler.config["sampling"]
    print(
        f"Running MCMC sampling: {sampling_config['draws']} draws, {sampling_config['chains']} chains"
    )

    trace = sampler.sample(
        draws=sampling_config["draws"],
        tune=sampling_config["tune"],
        chains=sampling_config["chains"],
        target_accept=sampling_config["target_accept"],
    )

    print(f"Sampling complete. Saving to: {args.output}")
    trace.to_netcdf(args.output)

    print("Summary statistics:")
    print(az.summary(trace))

    # Generate plots if requested
    if args.plot:
        print("Generating interactive dashboard...")
        from .unified_plotter import JonesPlotter

        plotter = JonesPlotter(sampler)
        summaries = plotter.create_comprehensive_dashboard(args.plot_output)

        print(f"Dashboard saved to: {args.plot_output}")
        print(f"Summary saved to: {args.plot_output.replace('.html', '_summary.json')}")

        # Print key summary info
        print("\nKey Effect Summaries:")
        for effect_name, summary in summaries.items():
            print(f"  {effect_name}: {summary.get('type', 'unknown')} effect")
            if "error" in summary:
                print(f"    Error: {summary['error']}")
            elif effect_name == "gains":
                print(
                    f"    Base amplitude XX: {summary.get('base_amplitude_xx_mean', 0):.3f}"
                )
                print(
                    f"    Thermal timescale: {summary.get('thermal_timescale', 0)/3600:.1f} hours"
                )
            elif effect_name == "bandpass":
                print(
                    f"    Frequency range: {summary.get('frequency_range_mhz', 0):.1f} MHz"
                )
                print(f"    Channels: {summary.get('n_channels', 0)}")

    print("Analysis complete!")


if __name__ == "__main__":
    main()
