"""Plotting utilities for calibration results.

Reads saved solver output and generates diagnostic plots.
Separate from solver to allow quick plot iteration without re-running inference.
"""

import os

import numpy as np
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, save


def plot_calibration_results(
    results_file: str,
    output_dir: str = "plots",
):
    """Generate diagnostic plots from saved calibration results.

    Args:
        results_file: Path to saved results (.npz file)
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    data = np.load(results_file, allow_pickle=True)

    plots = []

    # Delay plots
    if "delays_casa" in data and "delays_recovered" in data:
        casa_ns = data["delays_casa"] * 1e9
        recovered_ns = data["delays_recovered"] * 1e9
        n_ant = len(casa_ns)
        ant_idx = list(range(n_ant))

        # Comparison
        p1 = figure(
            title="Delay Comparison: CASA vs Recovered",
            x_axis_label="Antenna",
            y_axis_label="Delay (ns)",
            width=600,
            height=300,
        )
        p1.scatter(
            ant_idx,
            casa_ns.tolist(),
            size=8,
            color="red",
            legend_label="CASA",
            alpha=0.7,
        )
        p1.scatter(
            ant_idx,
            recovered_ns.tolist(),
            size=8,
            color="blue",
            legend_label="Recovered",
            marker="triangle",
            alpha=0.7,
        )
        p1.legend.location = "top_right"
        plots.append(p1)

        # Residuals
        residuals = (recovered_ns - casa_ns).tolist()
        p2 = figure(
            title="Delay Residuals (Recovered - CASA)",
            x_axis_label="Antenna",
            y_axis_label="Residual (ns)",
            width=600,
            height=300,
        )
        p2.scatter(ant_idx, residuals, size=8, color="purple", alpha=0.7)
        p2.line([0, n_ant - 1], [0, 0], color="black", line_dash="dashed")
        plots.append(p2)

    # Gain plots
    if "gains_casa" in data and "gains_recovered" in data:
        casa_gains = data["gains_casa"]
        recovered_gains = data["gains_recovered"]
        n_ant = len(casa_gains)
        ant_idx = list(range(n_ant))

        # Amplitude comparison (XX)
        casa_amp = np.abs(casa_gains[:, 0]).tolist()
        recovered_amp = np.abs(recovered_gains[:, 0]).tolist()

        p3 = figure(
            title="Gain Amplitude: CASA vs Recovered (XX)",
            x_axis_label="Antenna",
            y_axis_label="Amplitude",
            width=600,
            height=300,
        )
        p3.scatter(
            ant_idx, casa_amp, size=8, color="red", legend_label="CASA", alpha=0.7
        )
        p3.scatter(
            ant_idx,
            recovered_amp,
            size=8,
            color="blue",
            legend_label="Recovered",
            marker="triangle",
            alpha=0.7,
        )
        p3.legend.location = "top_right"
        plots.append(p3)

        # Amplitude residuals
        amp_residuals = [r - c for r, c in zip(recovered_amp, casa_amp)]
        p4 = figure(
            title="Gain Amplitude Residuals (XX)",
            x_axis_label="Antenna",
            y_axis_label="Residual",
            width=600,
            height=300,
        )
        p4.scatter(ant_idx, amp_residuals, size=8, color="purple", alpha=0.7)
        p4.line([0, n_ant - 1], [0, 0], color="black", line_dash="dashed")
        plots.append(p4)

    # Amplitude vs frequency (if available)
    if "freqs" in data and "amp_obs_freq" in data and "amp_model_freq" in data:
        freq_ghz = (data["freqs"] / 1e9).tolist()
        amp_obs = data["amp_obs_freq"].tolist()
        amp_model = data["amp_model_freq"].tolist()

        p5 = figure(
            title="Amplitude vs Frequency",
            x_axis_label="Frequency (GHz)",
            y_axis_label="Amplitude (Jy)",
            width=600,
            height=300,
        )
        p5.line(freq_ghz, amp_obs, legend_label="Observed", color="blue")
        p5.line(
            freq_ghz, amp_model, legend_label="Model", color="red", line_dash="dashed"
        )
        p5.legend.location = "top_right"
        plots.append(p5)

    # Arrange grid
    if len(plots) == 0:
        print("No data to plot")
        return

    rows = []
    for i in range(0, len(plots), 2):
        if i + 1 < len(plots):
            rows.append([plots[i], plots[i + 1]])
        else:
            rows.append([plots[i], None])

    layout = gridplot(rows)

    # Save
    output_path = os.path.join(output_dir, "calibration_plots.html")
    output_file(output_path)
    save(layout)
    print(f"Plots saved to {output_path}")


def save_calibration_results(
    output_file: str,
    delays_casa: np.ndarray = None,
    delays_recovered: np.ndarray = None,
    gains_casa: np.ndarray = None,
    gains_recovered: np.ndarray = None,
    freqs: np.ndarray = None,
    amp_obs_freq: np.ndarray = None,
    amp_model_freq: np.ndarray = None,
    **kwargs,
):
    """Save calibration results for later plotting.

    Args:
        output_file: Output .npz file
        delays_casa: CASA delay solution (seconds)
        delays_recovered: Recovered delays (seconds)
        gains_casa: CASA gain solution (complex)
        gains_recovered: Recovered gains (complex)
        freqs: Frequencies (Hz)
        amp_obs_freq: Observed amplitude vs frequency
        amp_model_freq: Model amplitude vs frequency
        **kwargs: Additional data to save
    """
    save_dict = {}

    if delays_casa is not None:
        save_dict["delays_casa"] = delays_casa
    if delays_recovered is not None:
        save_dict["delays_recovered"] = delays_recovered
    if gains_casa is not None:
        save_dict["gains_casa"] = gains_casa
    if gains_recovered is not None:
        save_dict["gains_recovered"] = gains_recovered
    if freqs is not None:
        save_dict["freqs"] = freqs
    if amp_obs_freq is not None:
        save_dict["amp_obs_freq"] = amp_obs_freq
    if amp_model_freq is not None:
        save_dict["amp_model_freq"] = amp_model_freq

    save_dict.update(kwargs)

    np.savez(output_file, **save_dict)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot calibration results")
    parser.add_argument("results_file", help="Path to saved results (.npz)")
    parser.add_argument("-o", "--output", default="plots", help="Output directory")

    args = parser.parse_args()
    plot_calibration_results(args.results_file, args.output)
