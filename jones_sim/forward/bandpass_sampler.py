"""PyMC GPU-accelerated Monte Carlo sampler for bandpass effects."""

from typing import Tuple

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from bokeh.models import HoverTool
from bokeh.plotting import figure


class BandpassMCSampler:
    """PyMC-based Monte Carlo sampler for bandpass response distributions."""

    def __init__(self, n_antennas: int, n_channels: int):
        """Initialize sampler for bandpass parameters.

        Args:
            n_antennas: Number of antennas
            n_channels: Number of frequency channels
        """
        self.n_antennas = n_antennas
        self.n_channels = n_channels
        self.model = None
        self.trace = None

    def build_bandpass_model(
        self,
        smooth_amp_scale: float = 0.1,
        jagged_amp_scale: float = 0.05,
        smooth_phase_scale: float = 0.2,
        jagged_phase_scale: float = 0.1,
        correlation_length: float = 5.0,
    ) -> pm.Model:
        """Build PyMC model for realistic bandpass response with smooth + jagged components.

        Args:
            smooth_amp_scale: Scale for smooth amplitude variations
            jagged_amp_scale: Scale for jagged amplitude variations
            smooth_phase_scale: Scale for smooth phase variations (radians)
            jagged_phase_scale: Scale for jagged phase variations (radians)
            correlation_length: Correlation length for GP (in channel units)

        Returns:
            PyMC model object
        """

        with pm.Model() as model:
            # Channel indices
            channels = np.arange(self.n_channels, dtype=float)
            pt.as_tensor_variable(channels)

            # === SMOOTH COMPONENTS (Gaussian Process) ===

            # GP for smooth amplitude variations
            smooth_amp_lengthscale = pm.Gamma(
                "smooth_amp_lengthscale",
                alpha=2,
                beta=1 / correlation_length,
                shape=self.n_antennas,
            )
            smooth_amp_variance = pm.HalfNormal(
                "smooth_amp_variance", sigma=smooth_amp_scale, shape=self.n_antennas
            )

            # GP for smooth phase variations
            smooth_phase_lengthscale = pm.Gamma(
                "smooth_phase_lengthscale",
                alpha=2,
                beta=1 / correlation_length,
                shape=self.n_antennas,
            )
            smooth_phase_variance = pm.HalfNormal(
                "smooth_phase_variance", sigma=smooth_phase_scale, shape=self.n_antennas
            )

            # Create GPs for each antenna
            smooth_log_amp = []
            smooth_phase = []

            for ant in range(self.n_antennas):
                # Smooth amplitude component (in log space for positivity)
                cov_amp = smooth_amp_variance[ant] * pm.gp.cov.ExpQuad(
                    1, ls=smooth_amp_lengthscale[ant]
                )
                gp_amp = pm.gp.Marginal(cov_func=cov_amp)
                smooth_log_amp_ant = gp_amp.marginal_likelihood(
                    f"smooth_log_amp_{ant}",
                    X=channels[:, None],
                    y=np.zeros(self.n_channels),
                )
                smooth_log_amp.append(smooth_log_amp_ant)

                # Smooth phase component
                cov_phase = smooth_phase_variance[ant] * pm.gp.cov.ExpQuad(
                    1, ls=smooth_phase_lengthscale[ant]
                )
                gp_phase = pm.gp.Marginal(cov_func=cov_phase)
                smooth_phase_ant = gp_phase.marginal_likelihood(
                    f"smooth_phase_{ant}",
                    X=channels[:, None],
                    y=np.zeros(self.n_channels),
                )
                smooth_phase.append(smooth_phase_ant)

            # Stack into tensors
            smooth_log_amp = pt.stack(smooth_log_amp)  # (n_antennas, n_channels)
            smooth_phase = pt.stack(smooth_phase)

            # === JAGGED COMPONENTS (Independent per channel) ===

            # Jagged amplitude variations (log-normal)
            jagged_log_amp = pm.Normal(
                "jagged_log_amp",
                mu=0.0,
                sigma=jagged_amp_scale,
                shape=(self.n_antennas, self.n_channels),
            )

            # Jagged phase variations
            jagged_phase = pm.Normal(
                "jagged_phase",
                mu=0.0,
                sigma=jagged_phase_scale,
                shape=(self.n_antennas, self.n_channels),
            )

            # === CABLE DELAY COMPONENT (Smooth phase slope) ===

            # Cable delays per antenna (creates linear phase vs frequency)
            cable_delay = pm.Normal(
                "cable_delay",
                mu=0.0,
                sigma=1e-9,  # nanosecond scale
                shape=self.n_antennas,
            )

            # Reference frequency (middle of band)
            ref_freq = 1.4e9  # 1.4 GHz
            freq_step = 1e6  # 1 MHz channels
            frequencies = ref_freq + (channels - self.n_channels / 2) * freq_step

            # Cable delay phase: 2π * τ * (ν - ν_ref)
            cable_phase = (
                2 * np.pi * cable_delay[:, None] * (frequencies[None, :] - ref_freq)
            )

            # === COMBINE COMPONENTS ===

            # Total log amplitude
            total_log_amp = smooth_log_amp + jagged_log_amp
            total_amplitude = pm.Deterministic(
                "bandpass_amplitude", pt.exp(total_log_amp)
            )

            # Total phase
            total_phase = pm.Deterministic(
                "bandpass_phase", smooth_phase + jagged_phase + cable_phase
            )

            # Complex bandpass response
            pm.Deterministic(
                "bandpass_response", total_amplitude * pt.exp(1j * total_phase)
            )

        self.model = model
        return model

    def sample(
        self,
        draws: int = 1000,
        tune: int = 500,
        chains: int = 2,
        cores: int = 2,
        target_accept: float = 0.9,
    ) -> az.InferenceData:
        """Sample from the bandpass model using NUTS.

        Args:
            draws: Number of posterior samples per chain
            tune: Number of tuning samples
            chains: Number of MCMC chains
            cores: Number of CPU cores
            target_accept: Target acceptance rate

        Returns:
            ArviZ InferenceData object with samples
        """
        if self.model is None:
            raise ValueError("Must build model first using build_bandpass_model()")

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                return_inferencedata=True,
            )

        return self.trace

    def extract_bandpass_samples(
        self, antenna_id: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract bandpass samples for specific antenna.

        Args:
            antenna_id: Antenna index

        Returns:
            (frequencies, amplitude_samples, phase_samples)
        """
        if self.trace is None:
            raise ValueError("Must sample first using sample()")

        # Frequency grid
        ref_freq = 1.4e9
        freq_step = 1e6
        channels = np.arange(self.n_channels)
        frequencies = ref_freq + (channels - self.n_channels / 2) * freq_step

        # Extract samples
        amplitudes = self.trace.posterior["bandpass_amplitude"].values[
            :, :, antenna_id, :
        ]
        phases = self.trace.posterior["bandpass_phase"].values[:, :, antenna_id, :]

        # Reshape to (n_samples, n_channels)
        amp_flat = amplitudes.reshape(-1, self.n_channels)
        phase_flat = phases.reshape(-1, self.n_channels)

        return frequencies, amp_flat, phase_flat

    def plot_bandpass_response(
        self,
        antenna_id: int = 0,
        n_sample_traces: int = 30,
        show_percentiles: bool = True,
    ) -> Tuple:
        """Plot bandpass response with Monte Carlo uncertainty.

        Args:
            antenna_id: Which antenna to plot
            n_sample_traces: Number of sample traces to show
            show_percentiles: Whether to show percentile bands

        Returns:
            (amplitude_figure, phase_figure)
        """
        frequencies, amplitudes, phases = self.extract_bandpass_samples(antenna_id)

        # Convert to GHz for display
        freq_ghz = frequencies / 1e9

        # Create figures
        amp_fig = figure(
            title=f"Bandpass Amplitude Response - Antenna {antenna_id}",
            x_axis_label="Frequency (GHz)",
            y_axis_label="Amplitude",
            width=700,
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )

        phase_fig = figure(
            title=f"Bandpass Phase Response - Antenna {antenna_id}",
            x_axis_label="Frequency (GHz)",
            y_axis_label="Phase (degrees)",
            width=700,
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )

        # Plot sample traces (background)
        n_samples = min(n_sample_traces, amplitudes.shape[0])
        for i in range(n_samples):
            amp_fig.line(freq_ghz, amplitudes[i], alpha=0.1, color="blue", line_width=1)
            phase_fig.line(
                freq_ghz, np.degrees(phases[i]), alpha=0.1, color="blue", line_width=1
            )

        # Plot percentile bands if requested
        if show_percentiles:
            amp_percentiles = np.percentile(amplitudes, [5, 50, 95], axis=0)
            phase_percentiles = np.percentile(np.degrees(phases), [5, 50, 95], axis=0)

            # Fill between percentiles
            amp_fig.varea(
                x=freq_ghz,
                y1=amp_percentiles[0],
                y2=amp_percentiles[2],
                alpha=0.3,
                color="blue",
                legend_label="90% CI",
            )
            phase_fig.varea(
                x=freq_ghz,
                y1=phase_percentiles[0],
                y2=phase_percentiles[2],
                alpha=0.3,
                color="blue",
                legend_label="90% CI",
            )

            # Plot medians
            amp_fig.line(
                freq_ghz,
                amp_percentiles[1],
                color="darkblue",
                line_width=3,
                legend_label="Median",
            )
            phase_fig.line(
                freq_ghz,
                phase_percentiles[1],
                color="darkblue",
                line_width=3,
                legend_label="Median",
            )

        # Configure legends and hover
        amp_fig.legend.location = "top_right"
        phase_fig.legend.location = "top_right"

        hover = HoverTool(
            tooltips=[("Frequency", "@x{0.000} GHz"), ("Value", "@y{0.000}")]
        )
        amp_fig.add_tools(hover)
        phase_fig.add_tools(hover)

        return amp_fig, phase_fig


def create_bandpass_example(
    n_antennas: int = 3, n_channels: int = 64
) -> BandpassMCSampler:
    """Create example bandpass sampler and run inference.

    Args:
        n_antennas: Number of antennas to simulate
        n_channels: Number of frequency channels

    Returns:
        Sampler with completed inference
    """
    sampler = BandpassMCSampler(n_antennas, n_channels)

    # Build model with realistic parameters
    sampler.build_bandpass_model(
        smooth_amp_scale=0.05,  # 5% smooth amplitude variations
        jagged_amp_scale=0.02,  # 2% jagged variations
        smooth_phase_scale=0.1,  # ~6 degree smooth phase variations
        jagged_phase_scale=0.05,  # ~3 degree jagged variations
        correlation_length=8.0,  # 8-channel correlation length
    )

    # Sample
    print("Running MCMC sampling for bandpass...")
    trace = sampler.sample(draws=800, tune=400, chains=2)

    print("Bandpass sampling complete!")
    print(az.summary(trace, var_names=["smooth_amp_variance", "cable_delay"]))

    return sampler
