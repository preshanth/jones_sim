"""Unified plotting and analysis for Jones matrix Monte Carlo results."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import arviz as az
import numpy as np
from bokeh.io import output_file
from bokeh.layouts import column, gridplot
from bokeh.models import Div, HoverTool, TabPanel, Tabs
from bokeh.plotting import figure, save

from .unified_sampler import JonesMCSampler


class JonesPlotter:
    """Unified plotter for Jones matrix Monte Carlo results with individual effect panels [Confidence: 85% - Evidence: Bokeh plotting patterns]."""

    def __init__(
        self, sampler: JonesMCSampler, logger: Optional[logging.Logger] = None
    ):
        """Initialize plotter with sampler results.

        Args:
            sampler: Sampler instance with completed trace
            logger: Optional logger for progress tracking
        """
        self.sampler = sampler
        self.logger = logger or logging.getLogger(__name__)
        self.width = 500
        self.height = 350
        self.tools = "pan,wheel_zoom,box_zoom,reset,save"

    def create_effect_summary(self) -> Dict[str, Dict[str, Any]]:
        """Generate summary statistics for each enabled effect [Confidence: 90% - Evidence: ArviZ summary patterns].

        Returns:
            Dictionary with summary stats for each effect
        """
        if self.sampler.trace is None:
            raise ValueError("No sampling results available")

        summaries = {}
        self.logger.info("Generating effect summaries...")

        for effect_name in self.sampler.config["effects"].keys():
            self.logger.info(f"Analyzing effect: {effect_name}")
            effect_summary = {}

            if effect_name == "gains":
                # Extract gain statistics
                try:
                    base_amp_summary = az.summary(
                        self.sampler.trace, var_names=["base_amp_xx", "base_amp_yy"]
                    )
                    thermal_summary = az.summary(
                        self.sampler.trace, var_names=["thermal_amplitude"]
                    )

                    effect_summary.update(
                        {
                            "type": "time_varying",
                            "base_amplitude_xx_mean": float(
                                base_amp_summary.loc["base_amp_xx[0]", "mean"]
                            ),
                            "base_amplitude_xx_std": float(
                                base_amp_summary.loc["base_amp_xx[0]", "sd"]
                            ),
                            "base_amplitude_yy_mean": float(
                                base_amp_summary.loc["base_amp_yy[0]", "mean"]
                            ),
                            "base_amplitude_yy_std": float(
                                base_amp_summary.loc["base_amp_yy[0]", "sd"]
                            ),
                            "thermal_amplitude_mean": float(
                                thermal_summary.loc["thermal_amplitude[0]", "mean"]
                            ),
                            "thermal_timescale": self.sampler.config["effects"][
                                "gains"
                            ]["thermal_timescale"],
                            "n_antennas_affected": self.sampler.n_antennas,
                            "time_range_hours": (
                                self.sampler.times[-1] - self.sampler.times[0]
                            )
                            / 3600,
                        }
                    )

                except Exception as e:
                    self.logger.warning(f"Could not extract gains summary: {e}")
                    effect_summary = {"type": "time_varying", "error": str(e)}

            elif effect_name == "bandpass":
                effect_summary.update(
                    {
                        "type": "frequency_varying",
                        "n_channels": self.sampler.n_freqs,
                        "frequency_range_mhz": (
                            self.sampler.frequencies[-1] - self.sampler.frequencies[0]
                        )
                        / 1e6,
                        "jagged_amplitude": self.sampler.config["effects"]["bandpass"][
                            "jagged_amplitude"
                        ],
                        "delay_std_ns": self.sampler.config["effects"]["bandpass"][
                            "delay_std"
                        ]
                        * 1e9,
                        "n_antennas_affected": self.sampler.n_antennas,
                    }
                )

            elif effect_name == "leakage":
                try:
                    leakage_summary = az.summary(
                        self.sampler.trace, var_names=["leakage_hv", "leakage_vh"]
                    )
                    effect_summary.update(
                        {
                            "type": "static",
                            "hv_leakage_mean_magnitude": float(
                                np.abs(leakage_summary.loc["leakage_hv[0]", "mean"])
                            ),
                            "vh_leakage_mean_magnitude": float(
                                np.abs(leakage_summary.loc["leakage_vh[0]", "mean"])
                            ),
                            "configuration_amplitude": self.sampler.config["effects"][
                                "leakage"
                            ]["amplitude"],
                            "n_antennas_affected": self.sampler.n_antennas,
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Could not extract leakage summary: {e}")
                    effect_summary = {"type": "static", "error": str(e)}

            elif effect_name == "parallactic":
                effect_summary.update(
                    {
                        "type": "time_varying_deterministic",
                        "rate_deg_per_hour": self.sampler.config["effects"][
                            "parallactic"
                        ]["rate_deg_per_hour"],
                        "total_rotation_deg": self.sampler.config["effects"][
                            "parallactic"
                        ]["rate_deg_per_hour"]
                        * (self.sampler.times[-1] - self.sampler.times[0])
                        / 3600,
                        "time_range_hours": (
                            self.sampler.times[-1] - self.sampler.times[0]
                        )
                        / 3600,
                    }
                )

            summaries[effect_name] = effect_summary
            self.logger.info(f"Effect {effect_name} summary completed")

        return summaries

    def plot_gains_vs_time(self, antenna_id: int = 0, n_traces: int = 30) -> Tuple:
        """Plot gain evolution over time [Confidence: 80% - Evidence: Time series plotting]."""
        if "gains" not in self.sampler.config["effects"]:
            return None, None

        self.logger.info(f"Plotting gains vs time for antenna {antenna_id}")

        # Extract gain samples
        gains_xx = self.sampler.trace.posterior["gains_xx"].values
        gains_yy = self.sampler.trace.posterior["gains_yy"].values

        # Reshape and select antenna
        gains_xx_flat = gains_xx.reshape(
            -1, self.sampler.n_antennas, self.sampler.n_times
        )
        gains_yy_flat = gains_yy.reshape(
            -1, self.sampler.n_antennas, self.sampler.n_times
        )

        gains_xx_ant = gains_xx_flat[:, antenna_id, :]
        gains_yy_ant = gains_yy_flat[:, antenna_id, :]

        # Time axis in hours
        times_hours = self.sampler.times / 3600

        # Create figures
        amp_fig = figure(
            title=f"Gain Amplitude vs Time - Antenna {antenna_id}",
            x_axis_label="Time (hours)",
            y_axis_label="Gain Amplitude",
            width=self.width,
            height=self.height,
            tools=self.tools,
        )

        phase_fig = figure(
            title=f"Gain Phase vs Time - Antenna {antenna_id}",
            x_axis_label="Time (hours)",
            y_axis_label="Phase (degrees)",
            width=self.width,
            height=self.height,
            tools=self.tools,
        )

        # Plot sample traces
        n_samples = min(n_traces, gains_xx_ant.shape[0])
        for i in range(n_samples):
            amp_fig.line(
                times_hours,
                np.abs(gains_xx_ant[i]),
                alpha=0.1,
                color="blue",
                line_width=1,
            )
            amp_fig.line(
                times_hours,
                np.abs(gains_yy_ant[i]),
                alpha=0.1,
                color="red",
                line_width=1,
            )

            phase_fig.line(
                times_hours,
                np.degrees(np.angle(gains_xx_ant[i])),
                alpha=0.1,
                color="blue",
                line_width=1,
            )
            phase_fig.line(
                times_hours,
                np.degrees(np.angle(gains_yy_ant[i])),
                alpha=0.1,
                color="red",
                line_width=1,
            )

        # Plot percentiles
        amp_xx_perc = np.percentile(np.abs(gains_xx_ant), [5, 50, 95], axis=0)
        amp_yy_perc = np.percentile(np.abs(gains_yy_ant), [5, 50, 95], axis=0)
        phase_xx_perc = np.percentile(
            np.degrees(np.angle(gains_xx_ant)), [5, 50, 95], axis=0
        )
        phase_yy_perc = np.percentile(
            np.degrees(np.angle(gains_yy_ant)), [5, 50, 95], axis=0
        )

        # Confidence intervals
        amp_fig.varea(
            x=times_hours,
            y1=amp_xx_perc[0],
            y2=amp_xx_perc[2],
            alpha=0.3,
            color="blue",
            legend_label="XX 90% CI",
        )
        amp_fig.varea(
            x=times_hours,
            y1=amp_yy_perc[0],
            y2=amp_yy_perc[2],
            alpha=0.3,
            color="red",
            legend_label="YY 90% CI",
        )

        phase_fig.varea(
            x=times_hours,
            y1=phase_xx_perc[0],
            y2=phase_xx_perc[2],
            alpha=0.3,
            color="blue",
            legend_label="XX 90% CI",
        )
        phase_fig.varea(
            x=times_hours,
            y1=phase_yy_perc[0],
            y2=phase_yy_perc[2],
            alpha=0.3,
            color="red",
            legend_label="YY 90% CI",
        )

        # Medians
        amp_fig.line(
            times_hours,
            amp_xx_perc[1],
            color="darkblue",
            line_width=3,
            legend_label="XX median",
        )
        amp_fig.line(
            times_hours,
            amp_yy_perc[1],
            color="darkred",
            line_width=3,
            legend_label="YY median",
        )

        phase_fig.line(
            times_hours,
            phase_xx_perc[1],
            color="darkblue",
            line_width=3,
            legend_label="XX median",
        )
        phase_fig.line(
            times_hours,
            phase_yy_perc[1],
            color="darkred",
            line_width=3,
            legend_label="YY median",
        )

        # Configure
        for fig in [amp_fig, phase_fig]:
            fig.legend.location = "top_right"
            fig.add_tools(
                HoverTool(tooltips=[("Time", "@x{0.00} hr"), ("Value", "@y{0.000}")])
            )

        return amp_fig, phase_fig

    def plot_bandpass_vs_frequency(
        self, antenna_id: int = 0, n_traces: int = 30
    ) -> Tuple:
        """Plot bandpass response vs frequency [Confidence: 80% - Evidence: Frequency domain plotting]."""
        if "bandpass" not in self.sampler.config["effects"]:
            return None, None

        self.logger.info(f"Plotting bandpass vs frequency for antenna {antenna_id}")

        # Extract bandpass samples
        bp_amp_xx = self.sampler.trace.posterior["bandpass_amp_xx"].values
        bp_amp_yy = self.sampler.trace.posterior["bandpass_amp_yy"].values
        bp_phase_xx = self.sampler.trace.posterior["bandpass_phase_xx"].values
        bp_phase_yy = self.sampler.trace.posterior["bandpass_phase_yy"].values

        # Reshape and select antenna
        bp_amp_xx_flat = bp_amp_xx.reshape(
            -1, self.sampler.n_antennas, self.sampler.n_freqs
        )
        bp_amp_yy_flat = bp_amp_yy.reshape(
            -1, self.sampler.n_antennas, self.sampler.n_freqs
        )
        bp_phase_xx_flat = bp_phase_xx.reshape(
            -1, self.sampler.n_antennas, self.sampler.n_freqs
        )
        bp_phase_yy_flat = bp_phase_yy.reshape(
            -1, self.sampler.n_antennas, self.sampler.n_freqs
        )

        amp_xx_ant = bp_amp_xx_flat[:, antenna_id, :]
        amp_yy_ant = bp_amp_yy_flat[:, antenna_id, :]
        phase_xx_ant = bp_phase_xx_flat[:, antenna_id, :]
        phase_yy_ant = bp_phase_yy_flat[:, antenna_id, :]

        # Frequency axis in GHz
        freqs_ghz = self.sampler.frequencies / 1e9

        # Create figures
        amp_fig = figure(
            title=f"Bandpass Amplitude vs Frequency - Antenna {antenna_id}",
            x_axis_label="Frequency (GHz)",
            y_axis_label="Amplitude",
            width=self.width,
            height=self.height,
            tools=self.tools,
        )

        phase_fig = figure(
            title=f"Bandpass Phase vs Frequency - Antenna {antenna_id}",
            x_axis_label="Frequency (GHz)",
            y_axis_label="Phase (degrees)",
            width=self.width,
            height=self.height,
            tools=self.tools,
        )

        # Plot sample traces
        n_samples = min(n_traces, amp_xx_ant.shape[0])
        for i in range(n_samples):
            amp_fig.line(
                freqs_ghz, amp_xx_ant[i], alpha=0.1, color="blue", line_width=1
            )
            amp_fig.line(freqs_ghz, amp_yy_ant[i], alpha=0.1, color="red", line_width=1)

            phase_fig.line(
                freqs_ghz,
                np.degrees(phase_xx_ant[i]),
                alpha=0.1,
                color="blue",
                line_width=1,
            )
            phase_fig.line(
                freqs_ghz,
                np.degrees(phase_yy_ant[i]),
                alpha=0.1,
                color="red",
                line_width=1,
            )

        # Plot percentiles
        amp_xx_perc = np.percentile(amp_xx_ant, [5, 50, 95], axis=0)
        amp_yy_perc = np.percentile(amp_yy_ant, [5, 50, 95], axis=0)
        phase_xx_perc = np.percentile(np.degrees(phase_xx_ant), [5, 50, 95], axis=0)
        phase_yy_perc = np.percentile(np.degrees(phase_yy_ant), [5, 50, 95], axis=0)

        # Confidence intervals and medians
        amp_fig.varea(
            x=freqs_ghz,
            y1=amp_xx_perc[0],
            y2=amp_xx_perc[2],
            alpha=0.3,
            color="blue",
            legend_label="XX 90% CI",
        )
        amp_fig.varea(
            x=freqs_ghz,
            y1=amp_yy_perc[0],
            y2=amp_yy_perc[2],
            alpha=0.3,
            color="red",
            legend_label="YY 90% CI",
        )

        phase_fig.varea(
            x=freqs_ghz,
            y1=phase_xx_perc[0],
            y2=phase_xx_perc[2],
            alpha=0.3,
            color="blue",
            legend_label="XX 90% CI",
        )
        phase_fig.varea(
            x=freqs_ghz,
            y1=phase_yy_perc[0],
            y2=phase_yy_perc[2],
            alpha=0.3,
            color="red",
            legend_label="YY 90% CI",
        )

        amp_fig.line(
            freqs_ghz,
            amp_xx_perc[1],
            color="darkblue",
            line_width=3,
            legend_label="XX median",
        )
        amp_fig.line(
            freqs_ghz,
            amp_yy_perc[1],
            color="darkred",
            line_width=3,
            legend_label="YY median",
        )

        phase_fig.line(
            freqs_ghz,
            phase_xx_perc[1],
            color="darkblue",
            line_width=3,
            legend_label="XX median",
        )
        phase_fig.line(
            freqs_ghz,
            phase_yy_perc[1],
            color="darkred",
            line_width=3,
            legend_label="YY median",
        )

        # Configure
        for fig in [amp_fig, phase_fig]:
            fig.legend.location = "top_right"
            fig.add_tools(
                HoverTool(
                    tooltips=[("Frequency", "@x{0.000} GHz"), ("Value", "@y{0.000}")]
                )
            )

        return amp_fig, phase_fig

    def plot_individual_jones_matrices(
        self,
        antenna_id: int = 0,
        time_idx: int = 0,
        freq_idx: int = 0,
        n_samples: int = 50,
    ) -> figure:
        """Plot individual Jones matrix elements as complex scatter plots [Confidence: 75% - Evidence: Complex plotting complexity]."""
        self.logger.info(
            f"Plotting individual Jones matrices for ant={antenna_id}, t={time_idx}, f={freq_idx}"
        )

        try:
            # This is complex and may fail - need to extract Jones matrices properly
            jones_samples = self.sampler.compute_jones_matrices()

            if jones_samples.ndim == 5:  # All samples
                jones_at_point = jones_samples[
                    :n_samples, antenna_id, time_idx, freq_idx, :, :
                ]
            else:  # Single sample
                self.logger.warning(
                    "Only single sample available for Jones matrix plotting"
                )
                return None

        except Exception as e:
            self.logger.error(f"Could not compute Jones matrices: {e}")
            return None

        # Create figure
        fig = figure(
            title=f"Jones Matrix Elements - Ant {antenna_id}, t={time_idx}, f={freq_idx}",
            x_axis_label="Real Part",
            y_axis_label="Imaginary Part",
            width=self.width,
            height=self.height,
            tools=self.tools,
        )

        # Plot each matrix element
        element_names = ["J11", "J12", "J21", "J22"]
        colors = ["blue", "red", "green", "orange"]

        for i in range(2):
            for j in range(2):
                element_idx = i * 2 + j
                element_data = jones_at_point[:, i, j]

                fig.circle(
                    element_data.real,
                    element_data.imag,
                    size=8,
                    alpha=0.6,
                    color=colors[element_idx],
                    legend_label=element_names[element_idx],
                )

        fig.legend.location = "top_right"
        fig.add_tools(
            HoverTool(tooltips=[("Real", "@x{0.000}"), ("Imag", "@y{0.000}")])
        )

        return fig

    def create_comprehensive_dashboard(
        self, output_file_name: str = "jones_dashboard.html"
    ) -> None:
        """Create comprehensive dashboard with all effects and summaries [Confidence: 85% - Evidence: Bokeh layout patterns]."""
        self.logger.info("Creating comprehensive Jones matrix dashboard...")

        # Generate summaries
        summaries = self.create_effect_summary()

        # Create summary text
        summary_html = "<h2>Jones Matrix Chain Summary</h2>"
        for effect_name, summary in summaries.items():
            summary_html += f"<h3>{effect_name.title()} Effect</h3>"
            summary_html += f"<p><b>Type:</b> {summary.get('type', 'unknown')}</p>"

            if effect_name == "gains":
                summary_html += f"<p><b>Base amplitude (XX):</b> {summary.get('base_amplitude_xx_mean', 0):.3f} ± {summary.get('base_amplitude_xx_std', 0):.3f}</p>"
                summary_html += f"<p><b>Thermal timescale:</b> {summary.get('thermal_timescale', 0) / 3600:.1f} hours</p>"
            elif effect_name == "bandpass":
                summary_html += f"<p><b>Frequency range:</b> {summary.get('frequency_range_mhz', 0):.1f} MHz</p>"
                summary_html += (
                    f"<p><b>Channels:</b> {summary.get('n_channels', 0)}</p>"
                )
            elif effect_name == "leakage":
                summary_html += f"<p><b>HV leakage magnitude:</b> {summary.get('hv_leakage_mean_magnitude', 0):.4f}</p>"

        summary_div = Div(text=summary_html, width=800)

        # Create individual effect plots
        tabs = []

        # Gains plots
        if "gains" in self.sampler.config["effects"]:
            gain_plots = []
            for ant_id in range(min(2, self.sampler.n_antennas)):  # First 2 antennas
                amp_fig, phase_fig = self.plot_gains_vs_time(ant_id)
                if amp_fig and phase_fig:
                    gain_plots.extend([amp_fig, phase_fig])

            if gain_plots:
                gains_layout = gridplot(
                    [gain_plots[i : i + 2] for i in range(0, len(gain_plots), 2)]
                )
                gains_panel = TabPanel(child=gains_layout, title="Gains vs Time")
                tabs.append(gains_panel)

        # Bandpass plots
        if "bandpass" in self.sampler.config["effects"]:
            bp_plots = []
            for ant_id in range(min(2, self.sampler.n_antennas)):
                amp_fig, phase_fig = self.plot_bandpass_vs_frequency(ant_id)
                if amp_fig and phase_fig:
                    bp_plots.extend([amp_fig, phase_fig])

            if bp_plots:
                bp_layout = gridplot(
                    [bp_plots[i : i + 2] for i in range(0, len(bp_plots), 2)]
                )
                bp_panel = TabPanel(child=bp_layout, title="Bandpass vs Frequency")
                tabs.append(bp_panel)

        # Individual Jones matrices
        jones_fig = self.plot_individual_jones_matrices(
            antenna_id=0, time_idx=0, freq_idx=0
        )
        if jones_fig:
            jones_panel = TabPanel(child=jones_fig, title="Jones Matrix Elements")
            tabs.append(jones_panel)

        # Combine everything
        if tabs:
            tabs_widget = Tabs(tabs=tabs)
            final_layout = column(summary_div, tabs_widget)
        else:
            final_layout = summary_div

        # Save dashboard
        output_file(output_file_name)
        save(final_layout)

        self.logger.info(f"Dashboard saved to: {Path(output_file_name).absolute()}")

        # Save summary as JSON
        summary_file = output_file_name.replace(".html", "_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summaries, f, indent=2)
        self.logger.info(f"Summary saved to: {Path(summary_file).absolute()}")

        return summaries


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging for Jones analysis [Confidence: 95% - Evidence: Standard logging pattern]."""
    logger = logging.getLogger("jones_analysis")
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
