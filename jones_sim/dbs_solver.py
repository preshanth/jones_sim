#!/usr/bin/env python3
"""Bayesian delay solver - SAMPLING MODULE.

This module:
- Loads MS data and CASA delays
- Estimates thermal noise
- Builds PyMC model with complex Gaussian likelihood
- Runs MCMC sampling
- Saves trace to disk

NO PLOTTING - that's in bayesian_delay_plotter.py
"""

import logging
import pickle
from collections import defaultdict

import arviz as az
import casatools
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .casa_interface import MeasurementSetHandler

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None


class BayesianDelaySampler:
    """Bayesian delay solver - SAMPLING ONLY."""

    def __init__(self, ms_path: str, casa_cal_table: str):
        """Initialize.

        Args:
            ms_path: Path to MS
            casa_cal_table: Path to CASA K-table (REQUIRED)
        """
        self.ms_path = ms_path
        self.casa_cal_table = casa_cal_table
        self.ms_handler = MeasurementSetHandler(ms_path)

        # Data - COMPLEX VISIBILITIES
        self.observed_vis_real = None
        self.observed_vis_imag = None
        self.model_vis_real = None
        self.model_vis_imag = None
        self.frequencies = None
        self.antenna1 = None
        self.antenna2 = None
        self.n_antennas = None

        # CASA
        self.casa_delays = None
        self.casa_delays_std = None

        # Thermal noise
        self.thermal_noise_sigma = None

        # PyMC
        self.model = None
        self.trace = None
        self.use_numpyro = False
        self.use_blackjax = False
        self.prior_bound_ns = None

        print(f"\n{'=' * 70}")
        print("BAYESIAN DELAY SAMPLER")
        print(f"{'=' * 70}")
        print(f"MS: {ms_path}")
        print(f"CASA table: {casa_cal_table}")

    def load_data(self, spw: int = 0, field: int = 0):
        """Load MS data - ONE SPW, time-averaged COMPLEX visibilities.

        Args:
            spw: Spectral window
            field: Field ID
        """
        print(f"\n{'=' * 70}")
        print("LOADING DATA")
        print(f"{'=' * 70}")

        # Get summary
        summary = self.ms_handler.get_observation_summary()
        self.n_antennas = summary["n_antennas"]

        print(f"Antennas: {self.n_antennas}")
        print(f"SPWs: {summary['n_spw']}")

        # Read DATA
        data_dict = self.ms_handler.read_visibilities(field=field, spw=spw)

        data_array = data_dict["data"]  # (n_corr, n_chan, n_row)
        antenna1 = data_dict["antenna1"]
        antenna2 = data_dict["antenna2"]

        if "frequency" in data_dict:
            freqs = data_dict["frequency"]
        else:
            freqs = summary["frequency_info"][spw]["chan_freqs"]

        n_corr, n_chan, n_row = data_array.shape
        print(f"Data shape: {n_corr} corr × {n_chan} chan × {n_row} rows")
        print(f"Frequency range: {freqs[0] / 1e9:.3f} - {freqs[-1] / 1e9:.3f} GHz")

        # Read MODEL_DATA
        tb = casatools.table()
        tb.open(self.ms_path)
        model_array = tb.getcol("MODEL_DATA")
        tb.close()

        # Average over TIME for each (baseline, channel) pair
        print("\n⏱ Averaging over time...")

        # Create unique (ant1, ant2, chan) keys
        baseline_chan_dict = {}

        for row in range(n_row):
            for chan in range(n_chan):
                key = (antenna1[row], antenna2[row], chan)

                # Extract 4 correlations (COMPLEX!)
                obs_vis = data_array[:, chan, row]  # (4,) complex
                model_vis = model_array[:, chan, row]  # (4,) complex

                if key not in baseline_chan_dict:
                    baseline_chan_dict[key] = {
                        "obs_vis_sum": obs_vis.copy(),
                        "model_vis_sum": model_vis.copy(),
                        "count": 1,
                    }
                else:
                    baseline_chan_dict[key]["obs_vis_sum"] += obs_vis
                    baseline_chan_dict[key]["model_vis_sum"] += model_vis
                    baseline_chan_dict[key]["count"] += 1

        print(
            f"  Time samples per baseline/channel: ~{n_row // len(np.unique(list(zip(antenna1, antenna2)), axis=0))}"
        )
        print(f"  Unique (baseline, channel) pairs: {len(baseline_chan_dict):,}")

        # Extract averaged data - SPLIT INTO REAL AND IMAG
        obs_real = []
        obs_imag = []
        model_real = []
        model_imag = []
        freq_list = []
        ant1_list = []
        ant2_list = []

        for (ant1, ant2, chan), data in baseline_chan_dict.items():
            # Average complex visibilities
            obs_vis_avg = data["obs_vis_sum"] / data["count"]
            model_vis_avg = data["model_vis_sum"] / data["count"]

            # Split into real and imaginary
            obs_real.append(obs_vis_avg.real)  # (4,)
            obs_imag.append(obs_vis_avg.imag)  # (4,)
            model_real.append(model_vis_avg.real)
            model_imag.append(model_vis_avg.imag)

            # Metadata
            freq_list.append(freqs[chan])
            ant1_list.append(ant1)
            ant2_list.append(ant2)

        # Convert to arrays
        self.observed_vis_real = np.array(obs_real)  # (n_vis, 4)
        self.observed_vis_imag = np.array(obs_imag)  # (n_vis, 4)
        self.model_vis_real = np.array(model_real)
        self.model_vis_imag = np.array(model_imag)
        self.frequencies = np.array(freq_list)  # (n_vis,)
        self.antenna1 = np.array(ant1_list, dtype=int)
        self.antenna2 = np.array(ant2_list, dtype=int)

        n_vis = len(self.frequencies)
        total_data_points = n_vis * 4 * 2  # 4 corr × 2 (real+imag)
        print(
            f"✓ Time-averaged to {n_vis:,} visibilities ({total_data_points:,} data points)"
        )

    def read_casa_delays(self):
        """Read CASA delays - REQUIRED."""
        print(f"\n{'=' * 70}")
        print("READING CASA DELAYS")
        print(f"{'=' * 70}")

        tb = casatools.table()
        tb.open(self.casa_cal_table)

        fparam = tb.getcol("FPARAM")  # nanoseconds
        flags = tb.getcol("FLAG")
        antennas = tb.getcol("ANTENNA1")

        tb.close()

        print(f"FPARAM shape: {fparam.shape}")

        # Extract delays per antenna (in ns)
        casa_delays_ns = np.zeros(self.n_antennas)
        delay_counts = np.zeros(self.n_antennas)

        if fparam.ndim == 3:
            n_pol, n_chan, n_rows = fparam.shape
            for row in range(n_rows):
                ant = antennas[row]
                for pol in range(n_pol):
                    for chan in range(n_chan):
                        if not flags[pol, chan, row]:
                            casa_delays_ns[ant] += fparam[pol, chan, row]
                            delay_counts[ant] += 1
        elif fparam.ndim == 2:
            n_pol, n_rows = fparam.shape
            for row in range(n_rows):
                ant = antennas[row]
                for pol in range(n_pol):
                    if not flags[pol, row]:
                        casa_delays_ns[ant] += fparam[pol, row]
                        delay_counts[ant] += 1

        # Average
        casa_delays_ns /= delay_counts

        # Convert to seconds
        self.casa_delays = casa_delays_ns * 1e-9
        self.casa_delays_std = np.std(casa_delays_ns) * 1e-9

        print("\nCASA delays (ns):")
        for ant in range(self.n_antennas):
            print(f"  Ant {ant}: {casa_delays_ns[ant]:8.3f}")
        print(f"Std: {self.casa_delays_std * 1e9:.3f} ns")

    def estimate_thermal_noise_from_time_scatter(self):
        """Estimate thermal noise from time scatter per channel."""
        print(f"\n{'=' * 70}")
        print("ESTIMATING THERMAL NOISE FROM TIME SCATTER")
        print(f"{'=' * 70}")

        # Reuse your existing time-averaging dictionary, but now collect variance too
        ms = casatools.table()
        ms.open(self.ms_path)
        data = ms.getcol("DATA")
        flag = ms.getcol("FLAG")
        a1 = ms.getcol("ANTENNA1")
        a2 = ms.getcol("ANTENNA2")
        ms.close()

        stds_real = []
        stds_imag = []

        # Group by (ant1, ant2, chan, pol)
        samples = defaultdict(list)

        for i in range(len(a1)):
            for chan in range(data.shape[1]):
                for pol in range(4):
                    if not flag[pol, chan, i]:
                        key = (a1[i], a2[i], chan, pol)
                        samples[key].append(data[pol, chan, i])

        print(f"Processing {len(samples)} baseline-channel-pol groups...")

        for key, vis_list in samples.items():
            if len(vis_list) >= 6:  # need some statistics
                vis = np.array(vis_list)
                stds_real.append(np.std(vis.real))
                stds_imag.append(np.std(vis.imag))

        sigma_real = np.median(stds_real)
        sigma_imag = np.median(stds_imag)
        self.thermal_noise_sigma = (sigma_real + sigma_imag) / 2.0

        print(f"Groups with ≥6 samples: {len(stds_real)}")
        print(f"σ_real = {sigma_real:.5f} Jy,  σ_imag = {sigma_imag:.5f} Jy")
        print(f"Final thermal noise σ = {self.thermal_noise_sigma:.5f} Jy")
        print("→ This is the true, uncorruptible thermal noise!")

    def build_model(
        self,
        prior_bound_ns: float = 0.5,
        use_numpyro: bool = False,
        use_blackjax: bool = False,
    ):
        """Build PyMC model - COMPLEX GAUSSIAN LIKELIHOOD.

        CRITICAL FIX: Uses TruncatedNormal instead of Uniform for better convergence!

        Args:
            prior_bound_ns: Bounds around CASA delays in ns (default ±0.5 ns)
            use_numpyro: Use NumPyro sampler for better convergence
            use_blackjax: Use BlackJAX sampler for better convergence
        """
        print(f"\n{'=' * 70}")
        print("BUILDING MODEL - COMPLEX GAUSSIAN LIKELIHOOD")
        print(f"{'=' * 70}")

        print(f"CASA delays (s): {self.casa_delays}")
        print(f"CASA std (spread): {self.casa_delays_std * 1e9:.3f} ns")
        print(f"Prior bounds: CASA ± {prior_bound_ns:.3f} ns")
        print(f"Thermal noise σ: {self.thermal_noise_sigma:.5f} Jy")
        print(f"Frequency shape: {self.frequencies.shape}")

        # Bounds in seconds
        bound_sec = prior_bound_ns * 1e-9

        with pm.Model() as model:
            # FIX antenna 0 as reference, FREE antennas 1..N-1

            # CRITICAL FIX: Use TruncatedNormal instead of Uniform!
            # Uniform has zero gradient everywhere → bad for NUTS
            # TruncatedNormal has smooth gradients → guides sampler toward CASA values
            delays_free = pm.TruncatedNormal(
                "delays_free",
                mu=self.casa_delays[1:],  # Center at CASA values
                sigma=bound_sec / 3,  # ~99% mass within ±bound_sec
                lower=self.casa_delays[1:] - bound_sec,
                upper=self.casa_delays[1:] + bound_sec,
                shape=self.n_antennas - 1,
            )

            # Concatenate: antenna 0 is fixed at CASA value
            delays = pt.concatenate([pt.as_tensor([self.casa_delays[0]]), delays_free])

            # Forward model: Apply delays to MODEL visibilities
            # V_pred = V_model * exp(+1j * 2π * (τ1-τ2) * ν)
            tau1 = delays[self.antenna1]  # (n_vis,)
            tau2 = delays[self.antenna2]  # (n_vis,)

            # Delay phase shift
            delay_phase = 2 * np.pi * (tau1 - tau2) * self.frequencies  # (n_vis,)

            # Apply rotation to MODEL visibilities
            cos_delay = pt.cos(delay_phase)  # (n_vis,)
            sin_delay = pt.sin(delay_phase)  # (n_vis,)

            # Broadcast for 4 correlations: (n_vis,) -> (n_vis, 4)
            cos_delay_bc = cos_delay[:, None]
            sin_delay_bc = sin_delay[:, None]

            # Apply rotation: V_pred = V_model * exp(+i * delay_phase)
            # (re + i*im) * (cos + i*sin) = (re*cos - im*sin) + i*(im*cos + re*sin)
            model_real_corrected = (
                cos_delay_bc * self.model_vis_real - sin_delay_bc * self.model_vis_imag
            )
            model_imag_corrected = (
                cos_delay_bc * self.model_vis_imag + sin_delay_bc * self.model_vis_real
            )

            # Complex Gaussian likelihood = 2 independent Normal likelihoods
            pm.Normal(
                "vis_real",
                mu=model_real_corrected.flatten(),
                sigma=self.thermal_noise_sigma,
                observed=self.observed_vis_real.flatten(),
            )
            pm.Normal(
                "vis_imag",
                mu=model_imag_corrected.flatten(),
                sigma=self.thermal_noise_sigma,
                observed=self.observed_vis_imag.flatten(),
            )

        self.model = model
        self.prior_bound_ns = prior_bound_ns
        self.use_numpyro = use_numpyro
        self.use_blackjax = use_blackjax

        print(f"Free parameters: {self.n_antennas - 1} delays (antenna 0 FIXED)")
        print(f"Data points: {len(self.frequencies) * 4 * 2:,} (real + imag)")
        print(
            f"Prior: TruncatedNormal(CASA, σ={bound_sec / 3 * 1e9:.3f} ns, bounds=±{prior_bound_ns:.3f} ns)"
        )
        print(f"Likelihood: Complex Gaussian with σ={self.thermal_noise_sigma:.5f} Jy")

        if use_blackjax:
            print("Sampler: BlackJAX/NUTS")
        elif use_numpyro:
            print("Sampler: NumPyro/NUTS")
        else:
            print("Sampler: PyMC/NUTS")

    def sample(self, draws=2000, tune=3000, chains=4, target_accept=0.9):
        """Sample from the posterior.

        Args:
            draws: Number of draws per chain
            tune: Number of tuning steps
            chains: Number of chains
            target_accept: Target acceptance rate (0.85-0.95)
        """
        print(f"\n{'=' * 70}")
        print("SAMPLING")
        print(f"{'=' * 70}")
        print(f"Draws: {draws}, Tune: {tune}, Chains: {chains}")
        print(f"Target accept: {target_accept}")

        with self.model:
            # Initialize at CASA delays (center of prior)
            start = {"delays_free": self.casa_delays[1:]}

            if self.use_blackjax:
                print("Using BlackJAX sampler...")
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    nuts_sampler="blackjax",
                    target_accept=target_accept,
                    return_inferencedata=True,
                    initvals=start,
                    idata_kwargs={"log_likelihood": True},
                )
            elif self.use_numpyro:
                print("Using NumPyro sampler...")
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    nuts_sampler="numpyro",
                    target_accept=target_accept,
                    return_inferencedata=True,
                    initvals=start,
                    idata_kwargs={"log_likelihood": True},
                )
            else:
                print("Using PyMC sampler...")
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    return_inferencedata=True,
                    initvals=start,
                    idata_kwargs={"log_likelihood": True},
                )

        print("✓ Sampling complete")

    def save_trace(self, filename: str = "delay_trace.nc"):
        """Save trace to NetCDF file (ArviZ format).

        Args:
            filename: Output filename (default: delay_trace.nc)
        """
        if self.trace is None:
            raise ValueError("No trace to save! Run sample() first.")

        print(f"\n{'=' * 70}")
        print("SAVING TRACE")
        print(f"{'=' * 70}")
        print(f"File: {filename}")

        # Save trace
        self.trace.to_netcdf(filename)
        print("✓ Trace saved")

        # Also save metadata as pickle for easy loading
        metadata = {
            "ms_path": self.ms_path,
            "casa_cal_table": self.casa_cal_table,
            "casa_delays": self.casa_delays,
            "casa_delays_std": self.casa_delays_std,
            "thermal_noise_sigma": self.thermal_noise_sigma,
            "prior_bound_ns": self.prior_bound_ns,
            "n_antennas": self.n_antennas,
            "observed_vis_real": self.observed_vis_real,
            "observed_vis_imag": self.observed_vis_imag,
            "model_vis_real": self.model_vis_real,
            "model_vis_imag": self.model_vis_imag,
            "frequencies": self.frequencies,
            "antenna1": self.antenna1,
            "antenna2": self.antenna2,
        }

        metadata_file = filename.replace(".nc", "_metadata.pkl")
        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)
        print(f"✓ Metadata saved: {metadata_file}")

    def print_summary(self):
        """Print quick summary of results."""
        if self.trace is None:
            raise ValueError("No trace! Run sample() first.")

        print(f"\n{'=' * 70}")
        print("CONVERGENCE SUMMARY")
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

        if max_rhat > 1.01:
            print("\n⚠ WARNING: Chains have not converged!")
            print("  Try:")
            print("    1. Increase tune/draws")
            print("    2. Use --blackjax or --numpyro for better convergence")
            print("    3. Check data quality")
            print("    4. Relax prior_bound_ns if too tight")

        # Print delay table
        delays_free_post = self.trace.posterior["delays_free"].values
        n_samples = delays_free_post.shape[0] * delays_free_post.shape[1]
        delays_free_flat = delays_free_post.reshape(n_samples, -1)

        delays_ns = np.zeros((n_samples, self.n_antennas))
        delays_ns[:, 0] = self.casa_delays[0] * 1e9
        delays_ns[:, 1:] = delays_free_flat * 1e9

        casa_ns = self.casa_delays * 1e9

        print(
            f"\n{'Ant':<5} {'Mean (ns)':<12} {'Std (ns)':<12} {'CASA (ns)':<12} {'Diff (ns)':<12}"
        )
        print("-" * 60)

        for ant in range(self.n_antennas):
            if ant == 0:
                mean = casa_ns[ant]
                std = 0.0
                casa = casa_ns[ant]
                diff = 0.0
            else:
                mean = np.mean(delays_ns[:, ant])
                std = np.std(delays_ns[:, ant])
                casa = casa_ns[ant]
                diff = mean - casa

            print(f"{ant:<5} {mean:>11.3f} {std:>11.3f} {casa:>11.3f} {diff:>11.3f}")


def run(
    ms_path: str,
    casa_table: str,
    output_file: str = "delay_trace.nc",
    spw: int = 0,
    prior_bound_ns: float = 0.5,
    use_numpyro: bool = False,
    use_blackjax: bool = False,
    draws: int = 2000,
    tune: int = 3000,
    chains: int = 4,
    target_accept: float = 0.9,
):
    """Run the sampler and save results.

    Args:
        ms_path: Path to measurement set
        casa_table: Path to CASA calibration table
        output_file: Output file for trace (default: delay_trace.nc)
        spw: Spectral window (default 0)
        prior_bound_ns: Bounded walk around CASA delays in ns (default ±0.5 ns)
        use_numpyro: Use NumPyro sampler
        use_blackjax: Use BlackJAX sampler
        draws: Number of draws per chain (default 2000)
        tune: Number of tuning steps (default 3000)
        chains: Number of chains (default 4)
        target_accept: Target acceptance rate (default 0.9)
    """

    sampler = BayesianDelaySampler(ms_path, casa_table)

    sampler.load_data(spw=spw)
    sampler.read_casa_delays()
    sampler.estimate_thermal_noise_from_time_scatter()
    sampler.build_model(
        prior_bound_ns=prior_bound_ns,
        use_numpyro=use_numpyro,
        use_blackjax=use_blackjax,
    )
    sampler.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept)
    sampler.print_summary()
    sampler.save_trace(output_file)

    print(f"\n{'=' * 70}")
    print("✓ SAMPLING COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Trace saved to: {output_file}")
    print(f"Metadata saved to: {output_file.replace('.nc', '_metadata.pkl')}")
    print("\nNext step:")
    print(f"  python bayesian_delay_plotter.py {output_file}")

    return sampler


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python bayesian_delay_sampler.py <ms_path> <casa_table> [options]"
        )
        print("\nPositional arguments:")
        print("  output_file: output filename (default: delay_trace.nc)")
        print("  spw: spectral window (default 0)")
        print("  prior_bound_ns: bounded walk around CASA in ns (default ±0.5)")
        print("\nOptions:")
        print("  --numpyro: use NumPyro sampler (better convergence)")
        print("  --blackjax: use BlackJAX sampler (better convergence)")
        print("  --draws N: number of draws per chain (default 2000)")
        print("  --tune N: number of tuning steps (default 3000)")
        print("  --chains N: number of chains (default 4)")
        print("  --target-accept X: target acceptance rate (default 0.9)")
        print("\nExample:")
        print(
            "  python bayesian_delay_sampler.py test.ms delays.K --blackjax --draws 3000"
        )
        sys.exit(1)

    ms = sys.argv[1]
    cal = sys.argv[2]

    # Parse flags first to know which args to skip
    flag_values = set()
    use_numpyro = "--numpyro" in sys.argv
    use_blackjax = "--blackjax" in sys.argv

    draws = 2000
    tune = 3000
    chains = 4
    target_accept = 0.9

    for i, arg in enumerate(sys.argv):
        if arg == "--draws" and i + 1 < len(sys.argv):
            draws = int(sys.argv[i + 1])
            flag_values.add(sys.argv[i + 1])
        elif arg == "--tune" and i + 1 < len(sys.argv):
            tune = int(sys.argv[i + 1])
            flag_values.add(sys.argv[i + 1])
        elif arg == "--chains" and i + 1 < len(sys.argv):
            chains = int(sys.argv[i + 1])
            flag_values.add(sys.argv[i + 1])
        elif arg == "--target-accept" and i + 1 < len(sys.argv):
            target_accept = float(sys.argv[i + 1])
            flag_values.add(sys.argv[i + 1])

    # Now parse positional args, skipping flag values
    output = "delay_trace.nc"
    spw = 0
    prior_bound = 0.5
    pos_args = [
        arg
        for arg in sys.argv[3:]
        if not arg.startswith("--") and arg not in flag_values
    ]

    if len(pos_args) > 0:
        output = pos_args[0]
    if len(pos_args) > 1:
        spw = int(pos_args[1])
    if len(pos_args) > 2:
        prior_bound = float(pos_args[2])

    sampler = run(
        ms,
        cal,
        output,
        spw,
        prior_bound,
        use_numpyro,
        use_blackjax,
        draws,
        tune,
        chains,
        target_accept,
    )
