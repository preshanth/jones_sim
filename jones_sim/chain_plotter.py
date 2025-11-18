#!/usr/bin/env python3
"""Bayesian delay solver - PLOTTING MODULE.

This module:
- Loads saved trace from NetCDF file
- Loads metadata
- Creates all diagnostic plots
- No sampling - just visualization

Run after bayesian_delay_sampler.py
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pickle
from typing import Optional


class BayesianDelayPlotter:
    """Plotting for Bayesian delay solver results."""

    def __init__(self, trace_file: str):
        """Initialize plotter.

        Args:
            trace_file: Path to trace NetCDF file
        """
        self.trace_file = trace_file
        self.metadata_file = trace_file.replace(".nc", "_metadata.pkl")

        print(f"\n{'=' * 70}")
        print("BAYESIAN DELAY PLOTTER")
        print(f"{'=' * 70}")
        print(f"Loading trace from: {trace_file}")

        # Load trace
        self.trace = az.from_netcdf(trace_file)
        print("✓ Trace loaded")

        # Load metadata
        with open(self.metadata_file, "rb") as f:
            metadata = pickle.load(f)

        self.casa_delays = metadata["casa_delays"]
        self.casa_delays_std = metadata["casa_delays_std"]
        self.thermal_noise_sigma = metadata["thermal_noise_sigma"]
        self.prior_bound_ns = metadata["prior_bound_ns"]
        self.n_antennas = metadata["n_antennas"]
        self.observed_vis_real = metadata["observed_vis_real"]
        self.observed_vis_imag = metadata["observed_vis_imag"]
        self.model_vis_real = metadata["model_vis_real"]
        self.model_vis_imag = metadata["model_vis_imag"]
        self.frequencies = metadata["frequencies"]
        self.antenna1 = metadata["antenna1"]
        self.antenna2 = metadata["antenna2"]

        print("✓ Metadata loaded")
        print(f"  Antennas: {self.n_antennas}")
        print(f"  Visibilities: {len(self.frequencies):,}")
        print(f"  Thermal noise σ: {self.thermal_noise_sigma:.5f} Jy")

    def plot_data_diagnostics(self, output_file: str = "data_diagnostics.png"):
        """Plot data quality diagnostics."""
        print(f"\n{'=' * 70}")
        print("PLOTTING DATA DIAGNOSTICS")
        print(f"{'=' * 70}")

        # Compute visibility amplitudes and phases
        obs_amp = np.sqrt(self.observed_vis_real**2 + self.observed_vis_imag**2)
        obs_phase = np.arctan2(self.observed_vis_imag, self.observed_vis_real)

        model_amp = np.sqrt(self.model_vis_real**2 + self.model_vis_imag**2)
        model_phase = np.arctan2(self.model_vis_imag, self.model_vis_real)

        # Phase residuals
        phase_residual = obs_phase - model_phase
        phase_residual_wrapped = np.angle(np.exp(1j * phase_residual))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Amplitude vs frequency
        ax = axes[0, 0]
        for corr in range(4):
            ax.scatter(
                self.frequencies / 1e9,
                obs_amp[:, corr],
                alpha=0.3,
                s=1,
                label=f"Corr {corr}",
            )
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Amplitude (Jy)")
        ax.set_title("Observed Visibility Amplitudes")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Phase residuals vs frequency
        ax = axes[0, 1]
        for corr in range(4):
            ax.scatter(
                self.frequencies / 1e9,
                np.degrees(phase_residual_wrapped[:, corr]),
                alpha=0.3,
                s=1,
                label=f"Corr {corr}",
            )
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Phase Residual (deg)")
        ax.set_title("Phase Residuals vs Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Real vs Imag (complex plane)
        ax = axes[0, 2]
        for corr in range(4):
            ax.scatter(
                self.observed_vis_real[:, corr],
                self.observed_vis_imag[:, corr],
                alpha=0.3,
                s=1,
                label=f"Corr {corr}",
            )
        ax.set_xlabel("Real (Jy)")
        ax.set_ylabel("Imag (Jy)")
        ax.set_title("Complex Plane (Observed)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

        # 4. Histogram of phase residuals
        ax = axes[1, 0]
        ax.hist(
            np.degrees(phase_residual_wrapped.flatten()),
            bins=100,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel("Phase Residual (deg)")
        ax.set_ylabel("Count")
        ax.set_title("Phase Residual Distribution")
        ax.grid(True, alpha=0.3)

        # 5. Thermal noise check - real residuals
        ax = axes[1, 1]
        real_resid = self.observed_vis_real - self.model_vis_real
        ax.hist(
            real_resid.flatten(),
            bins=100,
            alpha=0.7,
            edgecolor="black",
        )
        ax.axvline(
            -self.thermal_noise_sigma,
            color="red",
            linestyle="--",
            label=f"±σ = {self.thermal_noise_sigma:.3f} Jy",
        )
        ax.axvline(self.thermal_noise_sigma, color="red", linestyle="--")
        ax.set_xlabel("Real Residual (Jy)")
        ax.set_ylabel("Count")
        ax.set_title("Real Visibility Residuals")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Thermal noise check - imag residuals
        ax = axes[1, 2]
        imag_resid = self.observed_vis_imag - self.model_vis_imag
        ax.hist(
            imag_resid.flatten(),
            bins=100,
            alpha=0.7,
            edgecolor="black",
        )
        ax.axvline(
            -self.thermal_noise_sigma,
            color="red",
            linestyle="--",
            label=f"±σ = {self.thermal_noise_sigma:.3f} Jy",
        )
        ax.axvline(self.thermal_noise_sigma, color="red", linestyle="--")
        ax.set_xlabel("Imag Residual (Jy)")
        ax.set_ylabel("Count")
        ax.set_title("Imag Visibility Residuals")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_traces(self, output_file: str = "trace_plot.png"):
        """Plot MCMC traces and diagnostics."""
        print(f"\n{'=' * 70}")
        print("PLOTTING TRACES")
        print(f"{'=' * 70}")

        # Trace plots for all free delays
        fig = az.plot_trace(
            self.trace,
            var_names=["delays_free"],
            compact=True,
            backend_kwargs={"figsize": (14, 2 * (self.n_antennas - 1))},
        )
        plt.suptitle("MCMC Traces for Free Delays", y=1.002)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_posterior_distributions(
        self, output_file: str = "posterior_distributions.png"
    ):
        """Plot posterior distributions for all antennas."""
        print(f"\n{'=' * 70}")
        print("PLOTTING POSTERIOR DISTRIBUTIONS")
        print(f"{'=' * 70}")

        delays_free_post = self.trace.posterior["delays_free"].values
        n_samples = delays_free_post.shape[0] * delays_free_post.shape[1]
        delays_free_flat = delays_free_post.reshape(n_samples, -1)

        fig, axes = plt.subplots(
            (self.n_antennas + 2) // 3,
            3,
            figsize=(15, 3 * ((self.n_antennas + 2) // 3)),
        )
        if self.n_antennas == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        casa_ns = self.casa_delays * 1e9

        for ant in range(self.n_antennas):
            ax = axes[ant]

            if ant == 0:
                # Fixed antenna
                ax.axvline(casa_ns[ant], color="red", linewidth=2, label="CASA (FIXED)")
                ax.set_xlim(casa_ns[ant] - 0.1, casa_ns[ant] + 0.1)
            else:
                # Sampled antenna
                delays_ns = delays_free_flat[:, ant - 1] * 1e9
                ax.hist(delays_ns, bins=50, alpha=0.7, edgecolor="black", density=True)
                ax.axvline(casa_ns[ant], color="red", linestyle="--", label="CASA")
                ax.axvline(
                    np.mean(delays_ns), color="blue", linestyle="-", label="Posterior"
                )

            ax.set_xlabel("Delay (ns)")
            ax.set_ylabel("Density")
            ax.set_title(f"Antenna {ant}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(self.n_antennas, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_autocorr(self, output_file: str = "autocorr_plot.png"):
        """Plot autocorrelation."""
        print(f"\n{'=' * 70}")
        print("PLOTTING AUTOCORRELATION")
        print(f"{'=' * 70}")

        try:
            fig = az.plot_autocorr(
                self.trace,
                var_names=["delays_free"],
                max_lag=100,
                backend_kwargs={"figsize": (14, 2 * (self.n_antennas - 1))},
            )
            plt.suptitle("Autocorrelation of MCMC Chains", y=1.002)
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"✓ Saved: {output_file}")
            plt.close()
        except Exception as e:
            print(f"⚠ Skipping autocorr plot: {e}")

    def plot_rank(self, output_file: str = "rank_plot.png"):
        """Plot rank plots (convergence diagnostic)."""
        print(f"\n{'=' * 70}")
        print("PLOTTING RANK PLOTS")
        print(f"{'=' * 70}")

        fig = az.plot_rank(
            self.trace,
            var_names=["delays_free"],
            backend_kwargs={"figsize": (14, 2 * (self.n_antennas - 1))},
        )
        plt.suptitle("Rank Plots (uniform = good convergence)", y=1.002)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_forest(self, output_file: str = "forest_plot.png"):
        """Plot forest plot (summary of all parameters)."""
        print(f"\n{'=' * 70}")
        print("PLOTTING FOREST PLOT")
        print(f"{'=' * 70}")

        fig = az.plot_forest(
            self.trace,
            var_names=["delays_free"],
            combined=True,
            figsize=(10, max(6, self.n_antennas * 0.5)),
        )
        plt.title("95% HDI for Free Delays")
        plt.xlabel("Delay (s)")
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_corner(self, output_file: str = "corner_plot.png"):
        """Plot corner plot."""
        print(f"\n{'=' * 70}")
        print("PLOTTING CORNER PLOT")
        print(f"{'=' * 70}")

        try:
            import corner
        except ImportError:
            print("⚠ corner package not installed. Skipping corner plot.")
            print("  Install with: pip install corner")
            return

        # Get samples
        delays_free_post = self.trace.posterior["delays_free"].values
        n_samples = delays_free_post.shape[0] * delays_free_post.shape[1]
        delays_free_flat = delays_free_post.reshape(n_samples, -1)

        # Convert to ns
        samples_ns = delays_free_flat * 1e9
        casa_ns = self.casa_delays[1:] * 1e9

        # Labels
        labels = [f"Ant {i + 1}" for i in range(self.n_antennas - 1)]

        # Corner plot
        fig = corner.corner(
            samples_ns,
            labels=labels,
            truths=casa_ns,
            truth_color="red",
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3f",
            title_kwargs={"fontsize": 10},
        )

        fig.suptitle(
            "Corner Plot: Posterior Delays (ns)\nRed lines = CASA values",
            y=0.995,
            fontsize=12,
        )

        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_residuals(self, output_file: str = "residual_diagnostics.png"):
        """Plot posterior predictive residuals - COMPLEX version."""
        print(f"\n{'=' * 70}")
        print("PLOTTING RESIDUALS")
        print(f"{'=' * 70}")

        # Get posterior mean delays
        delays_free_post = self.trace.posterior["delays_free"].values
        n_samples = delays_free_post.shape[0] * delays_free_post.shape[1]
        delays_free_flat = delays_free_post.reshape(n_samples, -1)
        delays_mean = np.zeros(self.n_antennas)
        delays_mean[0] = self.casa_delays[0]
        delays_mean[1:] = np.mean(delays_free_flat, axis=0)

        # Apply posterior delays to MODEL visibilities
        tau1 = delays_mean[self.antenna1]
        tau2 = delays_mean[self.antenna2]
        delay_phase = 2 * np.pi * (tau1 - tau2) * self.frequencies

        # Rotate model visibilities
        cos_delay = np.cos(-delay_phase)[:, None]
        sin_delay = np.sin(-delay_phase)[:, None]

        model_real_corrected = (
            cos_delay * self.model_vis_real - sin_delay * self.model_vis_imag
        )
        model_imag_corrected = (
            sin_delay * self.model_vis_real + cos_delay * self.model_vis_imag
        )

        # Residuals BEFORE (no delay correction)
        resid_real_before = self.observed_vis_real - self.model_vis_real
        resid_imag_before = self.observed_vis_imag - self.model_vis_imag

        # Residuals AFTER (with posterior delay correction)
        resid_real_after = self.observed_vis_real - model_real_corrected
        resid_imag_after = self.observed_vis_imag - model_imag_corrected

        # Compute phases for visualization
        obs_phase = np.arctan2(self.observed_vis_imag, self.observed_vis_real)
        model_phase_before = np.arctan2(self.model_vis_imag, self.model_vis_real)
        model_phase_after = np.arctan2(model_imag_corrected, model_real_corrected)

        phase_resid_before = np.angle(np.exp(1j * (obs_phase - model_phase_before)))
        phase_resid_after = np.angle(np.exp(1j * (obs_phase - model_phase_after)))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Real residuals before/after
        ax = axes[0, 0]
        ax.hist(
            resid_real_before.flatten(),
            bins=100,
            alpha=0.5,
            label="Before",
            edgecolor="black",
        )
        ax.hist(
            resid_real_after.flatten(),
            bins=100,
            alpha=0.5,
            label="After",
            edgecolor="black",
        )
        ax.axvline(-self.thermal_noise_sigma, color="red", linestyle="--", alpha=0.5)
        ax.axvline(self.thermal_noise_sigma, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Real Residual (Jy)")
        ax.set_ylabel("Count")
        ax.set_title("Real Residuals: Before vs After")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Imag residuals before/after
        ax = axes[0, 1]
        ax.hist(
            resid_imag_before.flatten(),
            bins=100,
            alpha=0.5,
            label="Before",
            edgecolor="black",
        )
        ax.hist(
            resid_imag_after.flatten(),
            bins=100,
            alpha=0.5,
            label="After",
            edgecolor="black",
        )
        ax.axvline(-self.thermal_noise_sigma, color="red", linestyle="--", alpha=0.5)
        ax.axvline(self.thermal_noise_sigma, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Imag Residual (Jy)")
        ax.set_ylabel("Count")
        ax.set_title("Imag Residuals: Before vs After")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Phase residuals before/after
        ax = axes[0, 2]
        ax.hist(
            np.degrees(phase_resid_before.flatten()),
            bins=100,
            alpha=0.5,
            label="Before",
            edgecolor="black",
        )
        ax.hist(
            np.degrees(phase_resid_after.flatten()),
            bins=100,
            alpha=0.5,
            label="After",
            edgecolor="black",
        )
        ax.set_xlabel("Phase Residual (deg)")
        ax.set_ylabel("Count")
        ax.set_title("Phase Residuals: Before vs After")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Real residuals vs frequency (after)
        ax = axes[1, 0]
        for corr in range(4):
            ax.scatter(
                self.frequencies / 1e9,
                resid_real_after[:, corr],
                alpha=0.3,
                s=1,
                label=f"Corr {corr}",
            )
        ax.axhline(0, color="black", linestyle="-", alpha=0.3)
        ax.axhline(self.thermal_noise_sigma, color="red", linestyle="--", alpha=0.5)
        ax.axhline(-self.thermal_noise_sigma, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Real Residual (Jy)")
        ax.set_title("Corrected Real Residuals vs Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Imag residuals vs frequency (after)
        ax = axes[1, 1]
        for corr in range(4):
            ax.scatter(
                self.frequencies / 1e9,
                resid_imag_after[:, corr],
                alpha=0.3,
                s=1,
                label=f"Corr {corr}",
            )
        ax.axhline(0, color="black", linestyle="-", alpha=0.3)
        ax.axhline(self.thermal_noise_sigma, color="red", linestyle="--", alpha=0.5)
        ax.axhline(-self.thermal_noise_sigma, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Imag Residual (Jy)")
        ax.set_title("Corrected Imag Residuals vs Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Scatter reduction summary
        ax = axes[1, 2]
        std_real_before = np.std(resid_real_before)
        std_real_after = np.std(resid_real_after)
        std_imag_before = np.std(resid_imag_before)
        std_imag_after = np.std(resid_imag_after)

        categories = ["Real", "Imag"]
        before_vals = [std_real_before, std_imag_before]
        after_vals = [std_real_after, std_imag_after]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width / 2, before_vals, width, label="Before", alpha=0.7)
        ax.bar(x + width / 2, after_vals, width, label="After", alpha=0.7)
        ax.axhline(
            self.thermal_noise_sigma,
            color="red",
            linestyle="--",
            label=f"Thermal σ = {self.thermal_noise_sigma:.3f} Jy",
        )
        ax.set_ylabel("Std (Jy)")
        ax.set_title("Residual Scatter Reduction")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

        # Print statistics
        print(f"\nResidual Statistics:")
        print(f"  Real std before:  {std_real_before:.5f} Jy")
        print(f"  Real std after:   {std_real_after:.5f} Jy")
        print(f"  Imag std before:  {std_imag_before:.5f} Jy")
        print(f"  Imag std after:   {std_imag_after:.5f} Jy")
        print(f"  Thermal noise σ:  {self.thermal_noise_sigma:.5f} Jy")
        print(
            f"  Improvement (real): {(1 - std_real_after / std_real_before) * 100:.1f}%"
        )
        print(
            f"  Improvement (imag): {(1 - std_imag_after / std_imag_before) * 100:.1f}%"
        )

    def print_summary(self):
        """Print convergence and delay summary."""
        print(f"\n{'=' * 70}")
        print("RESULTS SUMMARY")
        print(f"{'=' * 70}")

        # Convergence metrics
        summary = az.summary(self.trace, var_names=["delays_free"])
        max_rhat = summary["r_hat"].max()
        min_ess = summary["ess_bulk"].min()

        print("\nConvergence Metrics:")
        print(
            f"  max(r_hat): {max_rhat:.4f} {'✓ GOOD' if max_rhat < 1.01 else '✗ BAD'}"
        )
        print(
            f"  min(ess_bulk): {min_ess:.0f} {'✓ GOOD' if min_ess > 100 else '✗ BAD'}"
        )

        # Delay table
        delays_free_post = self.trace.posterior["delays_free"].values
        n_samples = delays_free_post.shape[0] * delays_free_post.shape[1]
        delays_free_flat = delays_free_post.reshape(n_samples, -1)

        delays_ns = np.zeros((n_samples, self.n_antennas))
        delays_ns[:, 0] = self.casa_delays[0] * 1e9
        delays_ns[:, 1:] = delays_free_flat * 1e9

        casa_ns = self.casa_delays * 1e9

        print(
            f"\n{'Ant':<5} {'Mean (ns)':<12} {'Std (ns)':<12} {'95% CI (ns)':<30} {'CASA (ns)':<12} {'Diff (ns)':<12}"
        )
        print("-" * 90)

        for ant in range(self.n_antennas):
            if ant == 0:
                mean = casa_ns[ant]
                std = 0.0
                ci_low = casa_ns[ant]
                ci_high = casa_ns[ant]
                casa = casa_ns[ant]
                diff = 0.0
            else:
                mean = np.mean(delays_ns[:, ant])
                std = np.std(delays_ns[:, ant])
                ci = np.percentile(delays_ns[:, ant], [2.5, 97.5])
                ci_low, ci_high = ci
                casa = casa_ns[ant]
                diff = mean - casa

            print(
                f"{ant:<5} {mean:>11.3f} {std:>11.3f} [{ci_low:>9.3f}, {ci_high:>9.3f}] {casa:>11.3f} {diff:>11.3f}"
            )

    def plot_all(self, prefix: str = ""):
        """Generate all plots.

        Args:
            prefix: Prefix for output filenames (default: "")
        """
        print(f"\n{'=' * 70}")
        print("GENERATING ALL PLOTS")
        print(f"{'=' * 70}")

        self.plot_data_diagnostics(f"{prefix}data_diagnostics.png")
        self.plot_traces(f"{prefix}trace_plot.png")
        self.plot_posterior_distributions(f"{prefix}posterior_distributions.png")
        self.plot_autocorr(f"{prefix}autocorr_plot.png")
        self.plot_rank(f"{prefix}rank_plot.png")
        self.plot_forest(f"{prefix}forest_plot.png")
        self.plot_corner(f"{prefix}corner_plot.png")
        self.plot_residuals(f"{prefix}residual_diagnostics.png")
        self.print_summary()

        print(f"\n{'=' * 70}")
        print("✓ ALL PLOTS COMPLETE!")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bayesian_delay_plotter.py <trace_file.nc> [options]")
        print("\nOptions:")
        print("  --prefix PREFIX: prefix for output files (default: none)")
        print("  --no-corner: skip corner plot (requires corner package)")
        print("\nExample:")
        print("  python bayesian_delay_plotter.py delay_trace.nc")
        print("  python bayesian_delay_plotter.py delay_trace.nc --prefix test_")
        sys.exit(1)

    trace_file = sys.argv[1]

    # Parse options
    prefix = ""
    skip_corner = False

    for i, arg in enumerate(sys.argv):
        if arg == "--prefix" and i + 1 < len(sys.argv):
            prefix = sys.argv[i + 1]
        elif arg == "--no-corner":
            skip_corner = True

    # Load and plot
    plotter = BayesianDelayPlotter(trace_file)

    if skip_corner:
        # Plot everything except corner
        plotter.plot_data_diagnostics(f"{prefix}data_diagnostics.png")
        plotter.plot_traces(f"{prefix}trace_plot.png")
        plotter.plot_posterior_distributions(f"{prefix}posterior_distributions.png")
        plotter.plot_autocorr(f"{prefix}autocorr_plot.png")
        plotter.plot_rank(f"{prefix}rank_plot.png")
        plotter.plot_forest(f"{prefix}forest_plot.png")
        plotter.plot_residuals(f"{prefix}residual_diagnostics.png")
        plotter.print_summary()
    else:
        # Plot everything
        plotter.plot_all(prefix)
