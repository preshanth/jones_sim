"""Bokeh plotting routines for Jones matrix effects visualization."""

from typing import List, Optional

import numpy as np
from bokeh.io import output_file
from bokeh.layouts import gridplot
from bokeh.models import LinearColorMapper
from bokeh.palettes import RdYlBu11, Viridis256
from bokeh.plotting import figure, save, show

from .simulator import JonesSimulator


class JonesPlotter:
    """Interactive plotting of Jones matrix effects using Bokeh."""

    def __init__(self, width: int = 400, height: int = 300):
        """Initialize plotter with default figure dimensions."""
        self.width = width
        self.height = height
        self.tools = "pan,wheel_zoom,box_zoom,reset,save"

    def plot_jones_matrix_elements(
        self,
        jones_matrices: np.ndarray,
        frequencies: np.ndarray,
        times: np.ndarray,
        antenna_id: int = 0,
        title_prefix: str = "Jones Matrix",
    ) -> List:
        """Plot the 4 complex elements of Jones matrices vs frequency and time.

        Args:
            jones_matrices: Shape (n_times, n_freqs, 2, 2) complex array
            frequencies: Frequency grid in Hz
            times: Time grid in seconds
            antenna_id: Antenna ID for title
            title_prefix: Prefix for plot titles

        Returns:
            List of bokeh figure objects
        """
        figures = []
        element_names = ["J₁₁", "J₁₂", "J₂₁", "J₂₂"]

        # Create 2D meshgrids for plotting
        freq_mesh, time_mesh = np.meshgrid(frequencies, times)

        for i in range(2):
            for j in range(2):
                # Extract matrix element
                element = jones_matrices[:, :, i, j]

                # Amplitude plot
                amp_fig = figure(
                    title=f"{title_prefix} |{element_names[i * 2 + j]}| Ant {antenna_id}",
                    x_axis_label="Frequency (Hz)",
                    y_axis_label="Time (s)",
                    width=self.width,
                    height=self.height,
                    tools=self.tools,
                )

                # Use image plot for 2D data
                amp_data = np.abs(element)
                color_mapper = LinearColorMapper(
                    palette=Viridis256, low=amp_data.min(), high=amp_data.max()
                )

                amp_fig.image(
                    image=[amp_data],
                    x=frequencies.min(),
                    y=times.min(),
                    dw=frequencies.max() - frequencies.min(),
                    dh=times.max() - times.min(),
                    color_mapper=color_mapper,
                )

                # Phase plot
                phase_fig = figure(
                    title=f"{title_prefix} ∠{element_names[i * 2 + j]} Ant {antenna_id}",
                    x_axis_label="Frequency (Hz)",
                    y_axis_label="Time (s)",
                    width=self.width,
                    height=self.height,
                    tools=self.tools,
                )

                phase_data = np.angle(element)
                phase_mapper = LinearColorMapper(
                    palette=RdYlBu11, low=-np.pi, high=np.pi
                )

                phase_fig.image(
                    image=[phase_data],
                    x=frequencies.min(),
                    y=times.min(),
                    dw=frequencies.max() - frequencies.min(),
                    dh=times.max() - times.min(),
                    color_mapper=phase_mapper,
                )

                figures.extend([amp_fig, phase_fig])

        return figures

    def plot_effect_vs_frequency(
        self,
        effect_instance,
        frequencies: np.ndarray,
        time: float = 0.0,
        antenna_id: int = 0,
    ) -> List:
        """Plot single effect vs frequency.

        Args:
            effect_instance: Instance of an effect class
            frequencies: Frequency array in Hz
            time: Fixed time value
            antenna_id: Fixed antenna ID

        Returns:
            List of bokeh figures
        """
        # Compute Jones matrices across frequency
        jones_freq = np.zeros((len(frequencies), 2, 2), dtype=complex)
        for i, freq in enumerate(frequencies):
            jones_freq[i] = effect_instance.jones_matrix(freq, time, antenna_id)

        figures = []
        element_names = ["J₁₁", "J₁₂", "J₂₁", "J₂₂"]

        for i in range(2):
            for j in range(2):
                element = jones_freq[:, i, j]

                # Amplitude vs frequency
                amp_fig = figure(
                    title=f"|{element_names[i * 2 + j]}| vs Frequency",
                    x_axis_label="Frequency (Hz)",
                    y_axis_label="Amplitude",
                    width=self.width,
                    height=self.height,
                    tools=self.tools,
                )
                amp_fig.line(frequencies, np.abs(element), line_width=2, color="blue")
                amp_fig.circle(
                    frequencies, np.abs(element), size=4, color="blue", alpha=0.6
                )

                # Phase vs frequency
                phase_fig = figure(
                    title=f"∠{element_names[i * 2 + j]} vs Frequency",
                    x_axis_label="Frequency (Hz)",
                    y_axis_label="Phase (rad)",
                    width=self.width,
                    height=self.height,
                    tools=self.tools,
                )
                phase_fig.line(
                    frequencies, np.angle(element), line_width=2, color="red"
                )
                phase_fig.circle(
                    frequencies, np.angle(element), size=4, color="red", alpha=0.6
                )

                figures.extend([amp_fig, phase_fig])

        return figures

    def plot_effect_vs_time(
        self,
        effect_instance,
        times: np.ndarray,
        frequency: float = 1e9,
        antenna_id: int = 0,
    ) -> List:
        """Plot single effect vs time.

        Args:
            effect_instance: Instance of an effect class
            times: Time array in seconds
            frequency: Fixed frequency value
            antenna_id: Fixed antenna ID

        Returns:
            List of bokeh figures
        """
        # Compute Jones matrices across time
        jones_time = np.zeros((len(times), 2, 2), dtype=complex)
        for i, time in enumerate(times):
            jones_time[i] = effect_instance.jones_matrix(frequency, time, antenna_id)

        figures = []
        element_names = ["J₁₁", "J₁₂", "J₂₁", "J₂₂"]

        for i in range(2):
            for j in range(2):
                element = jones_time[:, i, j]

                # Amplitude vs time
                amp_fig = figure(
                    title=f"|{element_names[i * 2 + j]}| vs Time",
                    x_axis_label="Time (s)",
                    y_axis_label="Amplitude",
                    width=self.width,
                    height=self.height,
                    tools=self.tools,
                )
                amp_fig.line(times, np.abs(element), line_width=2, color="green")
                amp_fig.circle(times, np.abs(element), size=4, color="green", alpha=0.6)

                # Phase vs time
                phase_fig = figure(
                    title=f"∠{element_names[i * 2 + j]} vs Time",
                    x_axis_label="Time (s)",
                    y_axis_label="Phase (rad)",
                    width=self.width,
                    height=self.height,
                    tools=self.tools,
                )
                phase_fig.line(times, np.angle(element), line_width=2, color="orange")
                phase_fig.circle(
                    times, np.angle(element), size=4, color="orange", alpha=0.6
                )

                figures.extend([amp_fig, phase_fig])

        return figures

    def plot_visibility_corruption(
        self,
        ideal_visibilities: np.ndarray,
        corrupted_visibilities: np.ndarray,
        frequencies: np.ndarray,
        baseline_names: Optional[List[str]] = None,
    ) -> List:
        """Plot ideal vs corrupted visibilities.

        Args:
            ideal_visibilities: Shape (N, 4) for [XX, XY, YX, YY]
            corrupted_visibilities: Shape (N, 4) corrupted visibilities
            frequencies: Frequency array for each visibility
            baseline_names: Optional baseline labels

        Returns:
            List of bokeh figures
        """
        correlation_names = ["XX", "XY", "YX", "YY"]
        figures = []

        for corr_idx, corr_name in enumerate(correlation_names):
            ideal = ideal_visibilities[:, corr_idx]
            corrupted = corrupted_visibilities[:, corr_idx]

            # Amplitude comparison
            amp_fig = figure(
                title=f"{corr_name} Amplitude: Ideal vs Corrupted",
                x_axis_label="Frequency (Hz)",
                y_axis_label="Amplitude",
                width=self.width,
                height=self.height,
                tools=self.tools,
            )

            amp_fig.line(
                frequencies,
                np.abs(ideal),
                legend_label="Ideal",
                line_width=2,
                color="blue",
            )
            amp_fig.line(
                frequencies,
                np.abs(corrupted),
                legend_label="Corrupted",
                line_width=2,
                color="red",
                line_dash="dashed",
            )
            amp_fig.legend.location = "top_right"

            # Phase comparison
            phase_fig = figure(
                title=f"{corr_name} Phase: Ideal vs Corrupted",
                x_axis_label="Frequency (Hz)",
                y_axis_label="Phase (rad)",
                width=self.width,
                height=self.height,
                tools=self.tools,
            )

            phase_fig.line(
                frequencies,
                np.angle(ideal),
                legend_label="Ideal",
                line_width=2,
                color="blue",
            )
            phase_fig.line(
                frequencies,
                np.angle(corrupted),
                legend_label="Corrupted",
                line_width=2,
                color="red",
                line_dash="dashed",
            )
            phase_fig.legend.location = "top_right"

            figures.extend([amp_fig, phase_fig])

        return figures

    def create_dashboard(
        self,
        simulator: JonesSimulator,
        frequencies: np.ndarray,
        times: np.ndarray,
        antenna_ids: List[int] = [0],
        save_path: Optional[str] = None,
    ) -> None:
        """Create comprehensive dashboard of all effects.

        Args:
            simulator: JonesSimulator instance with effects
            frequencies: Frequency grid
            times: Time grid
            antenna_ids: List of antennas to plot
            save_path: Optional path to save HTML file
        """
        all_figures = []

        for ant_id in antenna_ids:
            # Compute total Jones matrix
            jones_total = np.zeros((len(times), len(frequencies), 2, 2), dtype=complex)
            for t_idx, time in enumerate(times):
                for f_idx, freq in enumerate(frequencies):
                    jones_total[t_idx, f_idx] = simulator.compute_jones_matrix(
                        freq, time, ant_id
                    )

            # Plot total Jones matrix
            total_figs = self.plot_jones_matrix_elements(
                jones_total, frequencies, times, ant_id, "Total Jones"
            )

            # Plot individual effects
            for effect_name in simulator.list_effects():
                effect_instance = simulator.effects[effect_name]

                # Effect vs frequency
                freq_figs = self.plot_effect_vs_frequency(
                    effect_instance, frequencies, times[0], ant_id
                )

                # Effect vs time (if times vary)
                if len(times) > 1:
                    time_figs = self.plot_effect_vs_time(
                        effect_instance, times, frequencies[0], ant_id
                    )
                    all_figures.extend(time_figs)

                all_figures.extend(freq_figs)

            all_figures.extend(total_figs)

        # Layout in grid
        n_cols = 4
        grid_figures = []
        for i in range(0, len(all_figures), n_cols):
            grid_figures.append(all_figures[i : i + n_cols])

        layout = gridplot(grid_figures)

        if save_path:
            output_file(save_path)
            save(layout)
        else:
            show(layout)
