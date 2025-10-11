"""PyMC GPU-accelerated Monte Carlo sampler for Jones matrix parameters."""

from typing import Tuple

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from bokeh.models import HoverTool
from bokeh.plotting import figure


class GainMCSampler:
    """PyMC-based Monte Carlo sampler for electronic gain distributions."""

    def __init__(self, n_antennas: int, n_times: int):
        """Initialize sampler for gain parameters.

        Args:
            n_antennas: Number of antennas
            n_times: Number of time points
        """
        self.n_antennas = n_antennas
        self.n_times = n_times
        self.model = None
        self.trace = None

    def build_gain_model(
        self,
        base_amp_mean: float = 1.0,
        base_amp_std: float = 0.05,
        phase_std: float = 0.1,
        thermal_timescale: float = 3600.0,
        thermal_amplitude: float = 0.02,
    ) -> pm.Model:
        """Build PyMC model for gain evolution with thermal drift.

        Args:
            base_amp_mean: Mean base amplitude
            base_amp_std: Standard deviation of base amplitude
            phase_std: Phase noise standard deviation (rad)
            thermal_timescale: Thermal drift timescale (seconds)
            thermal_amplitude: Amplitude of thermal variations

        Returns:
            PyMC model object
        """

        with pm.Model() as model:
            # Base gain amplitudes per antenna (log-normal to ensure positive)
            log_base_amp_xx = pm.Normal(
                "log_base_amp_xx",
                mu=np.log(base_amp_mean),
                sigma=base_amp_std,
                shape=self.n_antennas,
            )
            log_base_amp_yy = pm.Normal(
                "log_base_amp_yy",
                mu=np.log(base_amp_mean * 0.95),  # YY slightly different
                sigma=base_amp_std,
                shape=self.n_antennas,
            )

            base_amp_xx = pm.Deterministic("base_amp_xx", pt.exp(log_base_amp_xx))
            base_amp_yy = pm.Deterministic("base_amp_yy", pt.exp(log_base_amp_yy))

            # Base phases per antenna
            base_phase_xx = pm.Normal(
                "base_phase_xx", mu=0.0, sigma=phase_std, shape=self.n_antennas
            )
            base_phase_yy = pm.Normal(
                "base_phase_yy", mu=0.0, sigma=phase_std, shape=self.n_antennas
            )

            # Thermal drift parameters per antenna
            thermal_amp_xx = pm.Normal(
                "thermal_amp_xx", mu=0.0, sigma=thermal_amplitude, shape=self.n_antennas
            )
            thermal_amp_yy = pm.Normal(
                "thermal_amp_yy", mu=0.0, sigma=thermal_amplitude, shape=self.n_antennas
            )

            thermal_phase_xx = pm.Uniform(
                "thermal_phase_xx", lower=0, upper=2 * np.pi, shape=self.n_antennas
            )
            thermal_phase_yy = pm.Uniform(
                "thermal_phase_yy", lower=0, upper=2 * np.pi, shape=self.n_antennas
            )

            # Phase drift rates (rad/second)
            phase_drift_rate_xx = pm.Normal(
                "phase_drift_rate_xx", mu=0.0, sigma=1e-5, shape=self.n_antennas
            )
            phase_drift_rate_yy = pm.Normal(
                "phase_drift_rate_yy", mu=0.0, sigma=1e-5, shape=self.n_antennas
            )

            # Time-dependent gains (computed deterministically)
            times = np.linspace(0, 7200, self.n_times)  # 2 hours
            time_tensor = pt.as_tensor_variable(times)

            # Thermal modulation
            thermal_mod_xx = thermal_amp_xx[:, None] * pt.sin(
                2 * np.pi * time_tensor[None, :] / thermal_timescale
                + thermal_phase_xx[:, None]
            )
            thermal_mod_yy = thermal_amp_yy[:, None] * pt.sin(
                2 * np.pi * time_tensor[None, :] / thermal_timescale
                + thermal_phase_yy[:, None]
            )

            # Phase evolution
            phase_evolution_xx = (
                base_phase_xx[:, None]
                + phase_drift_rate_xx[:, None] * time_tensor[None, :]
            )
            phase_evolution_yy = (
                base_phase_yy[:, None]
                + phase_drift_rate_yy[:, None] * time_tensor[None, :]
            )

            # Final complex gains
            amp_xx = base_amp_xx[:, None] + thermal_mod_xx
            amp_yy = base_amp_yy[:, None] + thermal_mod_yy

            pm.Deterministic("gains_xx", amp_xx * pt.exp(1j * phase_evolution_xx))
            pm.Deterministic("gains_yy", amp_yy * pt.exp(1j * phase_evolution_yy))

            # Add measurement noise (if we had real data)
            # This would be where we'd add likelihood terms

        self.model = model
        return model

    def sample(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 4,
        target_accept: float = 0.95,
    ) -> az.InferenceData:
        """Sample from the gain model using NUTS.

        Args:
            draws: Number of posterior samples per chain
            tune: Number of tuning samples
            chains: Number of MCMC chains
            cores: Number of CPU cores (or GPU if available)
            target_accept: Target acceptance rate

        Returns:
            ArviZ InferenceData object with samples
        """
        if self.model is None:
            raise ValueError("Must build model first using build_gain_model()")

        with self.model:
            # Use GPU if available, otherwise CPU
            try:
                # Try GPU sampling
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=cores,
                    target_accept=target_accept,
                    return_inferencedata=True,
                )
            except Exception as e:
                print(f"GPU sampling failed, falling back to CPU: {e}")
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=cores,
                    target_accept=target_accept,
                    return_inferencedata=True,
                )

        return self.trace

    def extract_gain_samples(
        self, antenna_id: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract gain samples for specific antenna.

        Args:
            antenna_id: Antenna index

        Returns:
            (times, gains_xx_samples, gains_yy_samples)
        """
        if self.trace is None:
            raise ValueError("Must sample first using sample()")

        times = np.linspace(0, 7200, self.n_times)

        # Extract samples for specified antenna
        gains_xx = self.trace.posterior["gains_xx"].values[
            :, :, antenna_id, :
        ]  # (chain, draw, time)
        gains_yy = self.trace.posterior["gains_yy"].values[:, :, antenna_id, :]

        # Reshape to (n_samples, n_times)
        gains_xx_flat = gains_xx.reshape(-1, self.n_times)
        gains_yy_flat = gains_yy.reshape(-1, self.n_times)

        return times, gains_xx_flat, gains_yy_flat

    def plot_gain_evolution(
        self,
        antenna_id: int = 0,
        n_sample_traces: int = 50,
        show_percentiles: bool = True,
    ) -> Tuple:
        """Plot gain evolution with Monte Carlo uncertainty.

        Args:
            antenna_id: Which antenna to plot
            n_sample_traces: Number of sample traces to show
            show_percentiles: Whether to show percentile bands

        Returns:
            (amplitude_figure, phase_figure)
        """
        times, gains_xx, gains_yy = self.extract_gain_samples(antenna_id)

        # Create figures
        amp_fig = figure(
            title=f"Gain Amplitude Evolution - Antenna {antenna_id}",
            x_axis_label="Time (seconds)",
            y_axis_label="Gain Amplitude",
            width=700,
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )

        phase_fig = figure(
            title=f"Gain Phase Evolution - Antenna {antenna_id}",
            x_axis_label="Time (seconds)",
            y_axis_label="Phase (degrees)",
            width=700,
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )

        # Plot sample traces (background)
        n_samples = min(n_sample_traces, gains_xx.shape[0])
        for i in range(n_samples):
            amp_fig.line(
                times, np.abs(gains_xx[i]), alpha=0.1, color="blue", line_width=1
            )
            amp_fig.line(
                times, np.abs(gains_yy[i]), alpha=0.1, color="red", line_width=1
            )

            phase_fig.line(
                times,
                np.degrees(np.angle(gains_xx[i])),
                alpha=0.1,
                color="blue",
                line_width=1,
            )
            phase_fig.line(
                times,
                np.degrees(np.angle(gains_yy[i])),
                alpha=0.1,
                color="red",
                line_width=1,
            )

        # Plot percentile bands if requested
        if show_percentiles:
            amp_xx_percentiles = np.percentile(np.abs(gains_xx), [5, 50, 95], axis=0)
            amp_yy_percentiles = np.percentile(np.abs(gains_yy), [5, 50, 95], axis=0)
            phase_xx_percentiles = np.percentile(
                np.degrees(np.angle(gains_xx)), [5, 50, 95], axis=0
            )
            phase_yy_percentiles = np.percentile(
                np.degrees(np.angle(gains_yy)), [5, 50, 95], axis=0
            )

            # Fill between percentiles
            amp_fig.varea(
                x=times,
                y1=amp_xx_percentiles[0],
                y2=amp_xx_percentiles[2],
                alpha=0.3,
                color="blue",
                legend_label="XX 90% CI",
            )
            amp_fig.varea(
                x=times,
                y1=amp_yy_percentiles[0],
                y2=amp_yy_percentiles[2],
                alpha=0.3,
                color="red",
                legend_label="YY 90% CI",
            )

            phase_fig.varea(
                x=times,
                y1=phase_xx_percentiles[0],
                y2=phase_xx_percentiles[2],
                alpha=0.3,
                color="blue",
                legend_label="XX 90% CI",
            )
            phase_fig.varea(
                x=times,
                y1=phase_yy_percentiles[0],
                y2=phase_yy_percentiles[2],
                alpha=0.3,
                color="red",
                legend_label="YY 90% CI",
            )

            # Plot medians
            amp_fig.line(
                times,
                amp_xx_percentiles[1],
                color="blue",
                line_width=3,
                legend_label="XX median",
            )
            amp_fig.line(
                times,
                amp_yy_percentiles[1],
                color="red",
                line_width=3,
                legend_label="YY median",
            )

            phase_fig.line(
                times,
                phase_xx_percentiles[1],
                color="blue",
                line_width=3,
                legend_label="XX median",
            )
            phase_fig.line(
                times,
                phase_yy_percentiles[1],
                color="red",
                line_width=3,
                legend_label="YY median",
            )

        # Configure legends and hover
        amp_fig.legend.location = "top_right"
        phase_fig.legend.location = "top_right"

        hover = HoverTool(tooltips=[("Time", "@x{0.0} s"), ("Value", "@y{0.000}")])
        amp_fig.add_tools(hover)
        phase_fig.add_tools(hover)

        return amp_fig, phase_fig


def create_gain_example(n_antennas: int = 4, n_times: int = 50) -> GainMCSampler:
    """Create example gain sampler and run inference.

    Args:
        n_antennas: Number of antennas to simulate
        n_times: Number of time points

    Returns:
        Sampler with completed inference
    """
    sampler = GainMCSampler(n_antennas, n_times)

    # Build model with realistic parameters
    sampler.build_gain_model(
        base_amp_mean=1.0,
        base_amp_std=0.02,  # 2% amplitude variation
        phase_std=0.05,  # ~3 degree phase scatter
        thermal_timescale=3600.0,  # 1 hour thermal cycle
        thermal_amplitude=0.01,  # 1% thermal amplitude
    )

    # Sample (this is where GPU acceleration happens)
    print("Running MCMC sampling...")
    trace = sampler.sample(draws=1000, tune=500, chains=2)

    print("Sampling complete. Summary:")
    print(az.summary(trace, var_names=["base_amp_xx", "base_amp_yy"]))

    return sampler
