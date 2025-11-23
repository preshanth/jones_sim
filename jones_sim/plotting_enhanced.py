"""Enhanced plotting utilities for comprehensive calibration validation.

Provides specialized plots for:
- Bandpass (amplitude/phase vs frequency)
- Leakage (D-terms)
- Time-varying gains
- Multi-way comparisons (Truth vs CASA vs Recovered)
- Error distributions
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from bokeh.layouts import column, gridplot, row
from bokeh.models import HoverTool, Span
from bokeh.plotting import figure, output_file, save
from bokeh.palettes import Category10_10


def plot_bandpass_comparison(
    freqs: np.ndarray,
    truth_bp: Optional[np.ndarray] = None,
    casa_bp: Optional[np.ndarray] = None,
    recovered_bp: Optional[np.ndarray] = None,
    antenna_idx: int = 0,
    pol_idx: int = 0,
    output_file_path: Optional[str] = None,
) -> figure:
    """Plot bandpass amplitude and phase vs frequency.

    Args:
        freqs: Frequencies in Hz
        truth_bp: Ground truth bandpass (complex) [n_freq] or [n_ant, n_pol, n_freq]
        casa_bp: CASA bandpass solution (complex)
        recovered_bp: Recovered bandpass (complex)
        antenna_idx: Antenna index to plot
        pol_idx: Polarization index (0=XX, 1=YY)
        output_file_path: Save to file if provided

    Returns:
        Bokeh figure layout
    """
    freq_ghz = freqs / 1e9

    # Extract data for specific antenna/pol
    def extract_bp(bp_data):
        if bp_data is None:
            return None, None
        if bp_data.ndim == 1:  # Just frequency
            amp = np.abs(bp_data)
            phase = np.angle(bp_data, deg=True)
        elif bp_data.ndim == 3:  # [n_ant, n_pol, n_freq]
            amp = np.abs(bp_data[antenna_idx, pol_idx, :])
            phase = np.angle(bp_data[antenna_idx, pol_idx, :], deg=True)
        else:  # [n_freq, ...]
            amp = np.abs(bp_data[:, antenna_idx])
            phase = np.angle(bp_data[:, antenna_idx], deg=True)
        return amp, phase

    truth_amp, truth_phase = extract_bp(truth_bp)
    casa_amp, casa_phase = extract_bp(casa_bp)
    rec_amp, rec_phase = extract_bp(recovered_bp)

    pol_name = ["XX", "YY"][pol_idx]

    # Amplitude plot
    p_amp = figure(
        title=f"Bandpass Amplitude - Antenna {antenna_idx} ({pol_name})",
        x_axis_label="Frequency (GHz)",
        y_axis_label="Amplitude",
        width=800,
        height=300,
    )

    if truth_amp is not None:
        p_amp.line(freq_ghz, truth_amp, legend_label="Truth", color="green", line_width=2)

    if casa_amp is not None:
        p_amp.line(
            freq_ghz, casa_amp, legend_label="CASA", color="red", line_dash="dashed", line_width=2
        )

    if rec_amp is not None:
        p_amp.scatter(
            freq_ghz,
            rec_amp,
            legend_label="Recovered",
            color="blue",
            size=4,
            alpha=0.6,
        )

    p_amp.legend.location = "top_right"

    # Phase plot
    p_phase = figure(
        title=f"Bandpass Phase - Antenna {antenna_idx} ({pol_name})",
        x_axis_label="Frequency (GHz)",
        y_axis_label="Phase (degrees)",
        width=800,
        height=300,
    )

    if truth_phase is not None:
        p_phase.line(
            freq_ghz, truth_phase, legend_label="Truth", color="green", line_width=2
        )

    if casa_phase is not None:
        p_phase.line(
            freq_ghz,
            casa_phase,
            legend_label="CASA",
            color="red",
            line_dash="dashed",
            line_width=2,
        )

    if rec_phase is not None:
        p_phase.scatter(
            freq_ghz,
            rec_phase,
            legend_label="Recovered",
            color="blue",
            size=4,
            alpha=0.6,
        )

    p_phase.legend.location = "top_right"

    layout = column(p_amp, p_phase)

    if output_file_path:
        output_file(output_file_path)
        save(layout)
        print(f"Bandpass plot saved to {output_file_path}")

    return layout


def plot_three_way_comparison(
    truth: np.ndarray,
    casa: np.ndarray,
    recovered: np.ndarray,
    x_label: str = "Antenna",
    y_label: str = "Value",
    title: str = "Comparison",
    truth_name: str = "Truth",
    casa_name: str = "CASA",
    recovered_name: str = "Recovered",
    output_file_path: Optional[str] = None,
) -> Tuple[figure, figure]:
    """Create three-way comparison plot with residuals.

    Args:
        truth: Ground truth values
        casa: CASA values
        recovered: Recovered values
        x_label: X-axis label
        y_label: Y-axis label
        title: Plot title
        truth_name: Legend label for truth
        casa_name: Legend label for CASA
        recovered_name: Legend label for recovered
        output_file_path: Save to file if provided

    Returns:
        Tuple of (comparison_plot, residual_plot)
    """
    n_points = len(truth)
    x_vals = list(range(n_points))

    # Main comparison
    p1 = figure(
        title=title,
        x_axis_label=x_label,
        y_axis_label=y_label,
        width=800,
        height=350,
    )

    p1.scatter(
        x_vals,
        truth.tolist(),
        size=10,
        color="green",
        legend_label=truth_name,
        alpha=0.7,
        marker="circle",
    )

    p1.scatter(
        x_vals,
        casa.tolist(),
        size=8,
        color="red",
        legend_label=casa_name,
        alpha=0.7,
        marker="square",
    )

    p1.scatter(
        x_vals,
        recovered.tolist(),
        size=8,
        color="blue",
        legend_label=recovered_name,
        alpha=0.7,
        marker="triangle",
    )

    p1.legend.location = "top_right"

    # Residuals
    casa_err = (casa - truth).tolist()
    rec_err = (recovered - truth).tolist()

    p2 = figure(
        title=f"{title} - Residuals from Truth",
        x_axis_label=x_label,
        y_axis_label=f"{y_label} Error",
        width=800,
        height=250,
    )

    p2.scatter(
        x_vals, casa_err, size=8, color="red", legend_label=f"{casa_name} Error", alpha=0.7
    )

    p2.scatter(
        x_vals,
        rec_err,
        size=8,
        color="blue",
        legend_label=f"{recovered_name} Error",
        alpha=0.7,
        marker="triangle",
    )

    # Zero line
    zero_line = Span(location=0, dimension="width", line_color="black", line_dash="dashed")
    p2.add_layout(zero_line)

    # RMS indicators
    casa_rms = np.sqrt(np.mean(np.array(casa_err) ** 2))
    rec_rms = np.sqrt(np.mean(np.array(rec_err) ** 2))

    p2.legend.location = "top_right"
    p2.title.text += f" | CASA RMS: {casa_rms:.4f}, Ours RMS: {rec_rms:.4f}"

    if output_file_path:
        layout = column(p1, p2)
        output_file(output_file_path)
        save(layout)
        print(f"Comparison plot saved to {output_file_path}")

    return p1, p2


def plot_leakage_dterms(
    d_terms: Dict[str, np.ndarray],
    antenna_idx: Optional[int] = None,
    output_file_path: Optional[str] = None,
) -> figure:
    """Plot D-term (leakage) amplitudes and phases.

    Args:
        d_terms: Dictionary with 'truth', 'casa', 'recovered' D-terms [n_ant, 2] for d_xy, d_yx
        antenna_idx: If specified, plot specific antenna. Otherwise plot all.
        output_file_path: Save to file if provided

    Returns:
        Bokeh figure layout
    """
    plots = []

    # D_xy amplitude
    p1 = figure(
        title="D_xy Amplitude",
        x_axis_label="Antenna",
        y_axis_label="|D_xy|",
        width=600,
        height=300,
    )

    for name, color in [("truth", "green"), ("casa", "red"), ("recovered", "blue")]:
        if name in d_terms and d_terms[name] is not None:
            d_xy = d_terms[name][:, 0]  # First polarization
            amps = np.abs(d_xy)
            antennas = list(range(len(amps)))
            p1.scatter(antennas, amps.tolist(), size=8, color=color, legend_label=name.title(), alpha=0.7)

    p1.legend.location = "top_right"
    plots.append(p1)

    # D_xy phase
    p2 = figure(
        title="D_xy Phase",
        x_axis_label="Antenna",
        y_axis_label="Phase (degrees)",
        width=600,
        height=300,
    )

    for name, color in [("truth", "green"), ("casa", "red"), ("recovered", "blue")]:
        if name in d_terms and d_terms[name] is not None:
            d_xy = d_terms[name][:, 0]
            phases = np.angle(d_xy, deg=True)
            antennas = list(range(len(phases)))
            p2.scatter(antennas, phases.tolist(), size=8, color=color, legend_label=name.title(), alpha=0.7)

    p2.legend.location = "top_right"
    plots.append(p2)

    layout = gridplot([plots], ncols=2)

    if output_file_path:
        output_file(output_file_path)
        save(layout)
        print(f"Leakage plot saved to {output_file_path}")

    return layout


def plot_time_series_gains(
    times: np.ndarray,
    gains: Dict[str, np.ndarray],
    antenna_idx: int = 0,
    pol_idx: int = 0,
    output_file_path: Optional[str] = None,
) -> figure:
    """Plot time-varying gains.

    Args:
        times: Time values (MJD or seconds)
        gains: Dictionary with 'truth', 'casa', 'recovered' gains [n_time, n_ant, n_pol]
        antenna_idx: Antenna to plot
        pol_idx: Polarization (0=XX, 1=YY)
        output_file_path: Save to file if provided

    Returns:
        Bokeh figure layout
    """
    pol_name = ["XX", "YY"][pol_idx]

    # Amplitude vs time
    p_amp = figure(
        title=f"Gain Amplitude vs Time - Ant {antenna_idx} ({pol_name})",
        x_axis_label="Time",
        y_axis_label="Amplitude",
        width=800,
        height=300,
    )

    for name, color in [("truth", "green"), ("casa", "red"), ("recovered", "blue")]:
        if name in gains and gains[name] is not None:
            g = gains[name][:, antenna_idx, pol_idx]
            amps = np.abs(g)
            p_amp.line(times, amps, legend_label=name.title(), color=color, line_width=2)

    p_amp.legend.location = "top_right"

    # Phase vs time
    p_phase = figure(
        title=f"Gain Phase vs Time - Ant {antenna_idx} ({pol_name})",
        x_axis_label="Time",
        y_axis_label="Phase (degrees)",
        width=800,
        height=300,
    )

    for name, color in [("truth", "green"), ("casa", "red"), ("recovered", "blue")]:
        if name in gains and gains[name] is not None:
            g = gains[name][:, antenna_idx, pol_idx]
            phases = np.angle(g, deg=True)
            p_phase.line(times, phases, legend_label=name.title(), color=color, line_width=2)

    p_phase.legend.location = "top_right"

    layout = column(p_amp, p_phase)

    if output_file_path:
        output_file(output_file_path)
        save(layout)
        print(f"Time series plot saved to {output_file_path}")

    return layout


def plot_error_histogram(
    errors: Dict[str, np.ndarray],
    title: str = "Error Distribution",
    x_label: str = "Error",
    bins: int = 30,
    output_file_path: Optional[str] = None,
) -> figure:
    """Plot histogram of errors.

    Args:
        errors: Dictionary with error arrays (e.g., {'CASA': err_casa, 'Recovered': err_ours})
        title: Plot title
        x_label: X-axis label
        bins: Number of histogram bins
        output_file_path: Save to file if provided

    Returns:
        Bokeh figure
    """
    p = figure(
        title=title,
        x_axis_label=x_label,
        y_axis_label="Count",
        width=600,
        height=400,
    )

    colors = Category10_10[:len(errors)]

    for (name, err_arr), color in zip(errors.items(), colors):
        hist, edges = np.histogram(err_arr, bins=bins)
        p.quad(
            top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color=color,
            line_color="white",
            alpha=0.6,
            legend_label=f"{name} (RMS={np.sqrt(np.mean(err_arr**2)):.4f})",
        )

    # Zero line
    zero_line = Span(location=0, dimension="height", line_color="black", line_dash="dashed", line_width=2)
    p.add_layout(zero_line)

    p.legend.location = "top_right"

    if output_file_path:
        output_file(output_file_path)
        save(p)
        print(f"Histogram saved to {output_file_path}")

    return p


def create_validation_dashboard(
    results_file: str,
    output_dir: str = "plots",
) -> None:
    """Create comprehensive validation dashboard from results file.

    Args:
        results_file: Path to .npz results file
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(results_file, allow_pickle=True)

    all_plots = []

    # Delay plots
    if all(k in data for k in ["truth_delays", "casa_delays", "recovered_delays"]):
        truth_ns = data["truth_delays"] * 1e9
        casa_ns = data["casa_delays"] * 1e9
        rec_ns = data["recovered_delays"] * 1e9

        p1, p2 = plot_three_way_comparison(
            truth_ns,
            casa_ns,
            rec_ns,
            x_label="Antenna",
            y_label="Delay (ns)",
            title="Delay Recovery",
            output_file_path=os.path.join(output_dir, "delays.html"),
        )
        all_plots.extend([p1, p2])

        # Error histogram
        casa_err = casa_ns - truth_ns
        rec_err = rec_ns - truth_ns
        hist_plot = plot_error_histogram(
            {"CASA": casa_err, "Recovered": rec_err},
            title="Delay Error Distribution",
            x_label="Error (ns)",
            output_file_path=os.path.join(output_dir, "delay_errors.html"),
        )
        all_plots.append(hist_plot)

    # Gain plots
    if all(k in data for k in ["truth_gains", "casa_gains", "recovered_gains"]):
        # Amplitude
        truth_amp = np.abs(data["truth_gains"][:, 0])
        casa_amp = np.abs(data["casa_gains"][:, 0])
        rec_amp = np.abs(data["recovered_gains"][:, 0])

        p3, p4 = plot_three_way_comparison(
            truth_amp,
            casa_amp,
            rec_amp,
            x_label="Antenna",
            y_label="Amplitude",
            title="Gain Amplitude (XX)",
            output_file_path=os.path.join(output_dir, "gain_amplitude.html"),
        )
        all_plots.extend([p3, p4])

        # Phase
        truth_phase = np.angle(data["truth_gains"][:, 0], deg=True)
        casa_phase = np.angle(data["casa_gains"][:, 0], deg=True)
        rec_phase = np.angle(data["recovered_gains"][:, 0], deg=True)

        p5, p6 = plot_three_way_comparison(
            truth_phase,
            casa_phase,
            rec_phase,
            x_label="Antenna",
            y_label="Phase (degrees)",
            title="Gain Phase (XX)",
            output_file_path=os.path.join(output_dir, "gain_phase.html"),
        )
        all_plots.extend([p5, p6])

    # Bandpass plots
    if "bandpass_truth" in data and "frequencies" in data:
        bp_plot = plot_bandpass_comparison(
            data["frequencies"],
            truth_bp=data.get("bandpass_truth"),
            casa_bp=data.get("bandpass_casa"),
            recovered_bp=data.get("bandpass_recovered"),
            antenna_idx=0,
            pol_idx=0,
            output_file_path=os.path.join(output_dir, "bandpass.html"),
        )
        all_plots.append(bp_plot)

    # Combined dashboard
    if all_plots:
        rows = []
        for i in range(0, len(all_plots), 2):
            if i + 1 < len(all_plots):
                rows.append([all_plots[i], all_plots[i + 1]])
            else:
                rows.append([all_plots[i], None])

        dashboard = gridplot(rows)
        output_path = os.path.join(output_dir, "validation_dashboard.html")
        output_file(output_path)
        save(dashboard)
        print(f"Validation dashboard saved to {output_path}")
    else:
        print("No data available for dashboard")
