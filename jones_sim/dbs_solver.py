#!/usr/bin/env python3
"""Simple Bayesian delay solver - NO BULLSHIT VERSION.

- Reads ONE SPW
- ALL data (no max_vis)
- CASA delays required (no fallback)
- Free all antennas EXCEPT antenna 0 (reference)
- Wrapped phases
- Let it fail if something's wrong
"""

import numpy as np
import arviz as az
import pymc as pm
import pytensor.tensor as pt
import casatools

from casa_interface import MeasurementSetHandler


class SimpleDelaySolver:
    """Clean Bayesian delay solver."""

    def __init__(self, ms_path: str, casa_cal_table: str):
        """Initialize.

        Args:
            ms_path: Path to MS
            casa_cal_table: Path to CASA K-table (REQUIRED)
        """
        self.ms_path = ms_path
        self.casa_cal_table = casa_cal_table
        self.ms_handler = MeasurementSetHandler(ms_path)

        # Data
        self.observed_phases = None
        self.model_phases = None
        self.frequencies = None
        self.antenna1 = None
        self.antenna2 = None
        self.n_antennas = None

        # CASA
        self.casa_delays = None
        self.casa_delays_std = None

        # PyMC
        self.model = None
        self.trace = None

        print(f"\n{'=' * 70}")
        print(f"SIMPLE BAYESIAN DELAY SOLVER")
        print(f"{'=' * 70}")
        print(f"MS: {ms_path}")
        print(f"CASA table: {casa_cal_table}")

    def load_data(self, spw: int = 0, field: int = 0):
        """Load MS data - ONE SPW, time-averaged.

        Args:
            spw: Spectral window
            field: Field ID
        """
        print(f"\n{'=' * 70}")
        print(f"LOADING DATA")
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
        print(f"\n⏱ Averaging over time...")

        # Create unique (ant1, ant2, chan) keys
        baseline_chan_dict = {}

        for row in range(n_row):
            for chan in range(n_chan):
                key = (antenna1[row], antenna2[row], chan)

                # Extract 4 correlations
                obs_vis = data_array[:, chan, row]  # (4,)
                model_vis = model_array[:, chan, row]  # (4,)

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

        # Extract averaged data
        obs_phases = []
        model_phases = []
        freq_list = []
        ant1_list = []
        ant2_list = []

        for (ant1, ant2, chan), data in baseline_chan_dict.items():
            # Average complex visibilities
            obs_vis_avg = data["obs_vis_sum"] / data["count"]
            model_vis_avg = data["model_vis_sum"] / data["count"]

            # Phases
            obs_phases.append(np.angle(obs_vis_avg))
            model_phases.append(np.angle(model_vis_avg))

            # Metadata
            freq_list.append(freqs[chan])
            ant1_list.append(ant1)
            ant2_list.append(ant2)

        # Convert to arrays
        self.observed_phases = np.array(obs_phases)  # (n_vis, 4)
        self.model_phases = np.array(model_phases)
        self.frequencies = np.array(freq_list)  # (n_vis,)
        self.antenna1 = np.array(ant1_list, dtype=int)
        self.antenna2 = np.array(ant2_list, dtype=int)

        n_vis = len(self.frequencies)
        total_data_points = n_vis * 4
        print(
            f"✓ Time-averaged to {n_vis:,} visibilities ({total_data_points:,} data points)"
        )

    def read_casa_delays(self):
        """Read CASA delays - REQUIRED."""
        print(f"\n{'=' * 70}")
        print(f"READING CASA DELAYS")
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

        print(f"\nCASA delays (ns):")
        for ant in range(self.n_antennas):
            print(f"  Ant {ant}: {casa_delays_ns[ant]:8.3f}")
        print(f"Std: {self.casa_delays_std * 1e9:.3f} ns")

    def estimate_phase_noise(self):
        """Estimate phase noise from RAW residuals (no CASA correction)."""
        print(f"\n{'=' * 70}")
        print(f"ESTIMATING PHASE NOISE")
        print(f"{'=' * 70}")

        # RAW phase residuals (no CASA correction!)
        phase_residual = self.observed_phases - self.model_phases
        phase_residual_wrapped = np.angle(np.exp(1j * phase_residual))

        # Get unique frequencies
        unique_freqs = np.unique(self.frequencies)

        # Compute scatter per frequency channel
        stds_per_channel = []
        for freq in unique_freqs:
            mask = self.frequencies == freq
            # Scatter for this frequency (all correlations)
            std_chan = np.std(phase_residual_wrapped[mask])
            stds_per_channel.append(std_chan)

        # Median across channels (robust to outliers)
        self.phase_noise_std = np.median(stds_per_channel)

        print(f"Channels analyzed: {len(unique_freqs)}")
        print(
            f"Phase noise (RAW, median): {self.phase_noise_std:.6f} rad = {np.degrees(self.phase_noise_std):.3f} deg"
        )
        print(
            f"Min channel std: {np.min(stds_per_channel):.6f} rad = {np.degrees(np.min(stds_per_channel)):.3f} deg"
        )
        print(
            f"Max channel std: {np.max(stds_per_channel):.6f} rad = {np.degrees(np.max(stds_per_channel)):.3f} deg"
        )

    def build_model(self, prior_bound_ns: float = 0.5):
        """Build PyMC model - FIX antenna 0, free others, BOUNDED walk.

        Args:
            prior_bound_ns: Bounds around CASA delays in ns (default ±0.5 ns)
        """
        print(f"\n{'=' * 70}")
        print(f"BUILDING MODEL")
        print(f"{'=' * 70}")

        # Phase residuals (wrapped)
        phase_residual = self.observed_phases - self.model_phases
        phase_residual_wrapped = np.angle(np.exp(1j * phase_residual))

        print(f"CASA delays: {self.casa_delays}")
        print(f"CASA std (spread): {self.casa_delays_std * 1e9:.3f} ns")
        print(f"Prior bounds: CASA ± {prior_bound_ns:.3f} ns")
        print(f"Frequency shape: {self.frequencies.shape}")

        # Bounds in seconds
        bound_sec = prior_bound_ns * 1e-9

        with pm.Model() as model:
            # FIX antenna 0 as reference, FREE antennas 1..N-1
            # BOUNDED UNIFORM walk around CASA delays
            delays_free = pm.Uniform(
                "delays_free",
                lower=self.casa_delays[1:] - bound_sec,
                upper=self.casa_delays[1:] + bound_sec,
                shape=self.n_antennas - 1,
            )

            # Concatenate: antenna 0 is fixed at CASA value
            delays = pt.concatenate([pt.as_tensor([self.casa_delays[0]]), delays_free])

            # Forward model
            tau1 = delays[self.antenna1]
            tau2 = delays[self.antenna2]

            # Phase correction
            phase_correction = 2 * np.pi * (tau1 - tau2) * self.frequencies

            # WRAP the predicted phase!
            phase_correction_wrapped = pt.arctan2(
                pt.sin(phase_correction), pt.cos(phase_correction)
            )

            # Tile for 4 correlations
            predicted_phase = pt.tile(phase_correction_wrapped[:, None], (1, 4))

            # Likelihood
            pm.Normal(
                "phase_obs",
                mu=predicted_phase.flatten(),
                sigma=self.phase_noise_std,
                observed=phase_residual_wrapped.flatten(),
            )

        self.model = model
        self.prior_bound_ns = prior_bound_ns

        print(f"Free parameters: {self.n_antennas - 1} delays (antenna 0 FIXED)")
        print(f"Data points: {len(self.frequencies) * 4:,}")
        print(f"Prior: Uniform(CASA ± {prior_bound_ns:.3f} ns)")
        print(f"Likelihood: Normal(wrapped_phase, {self.phase_noise_std:.6f} rad)")

    def sample(self, draws=2000, tune=3000, chains=4):
        print(f"\n{'=' * 70}")
        print(f"SAMPLING")
        print(f"{'=' * 70}")
        print(f"Draws: {draws}, Tune: {tune}, Chains: {chains}")

        with self.model:
            # Initialize at CASA delays (center of bounded range)
            start = {"delays_free": self.casa_delays[1:]}

            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=0.95,  # Reduced from 0.995 for faster sampling
                return_inferencedata=True,
                initvals=start,
                idata_kwargs={"log_likelihood": True},
            )

        print(f"✓ Sampling complete")

    def summarize(self):
        """Print results."""
        print(f"\n{'=' * 70}")
        print(f"RESULTS")
        print(f"{'=' * 70}")

        delays_free_post = self.trace.posterior["delays_free"].values
        n_samples = delays_free_post.shape[0] * delays_free_post.shape[1]
        delays_free_flat = delays_free_post.reshape(n_samples, -1)

        # Reconstruct full delays: antenna 0 fixed, others sampled
        delays_ns = np.zeros((n_samples, self.n_antennas))
        delays_ns[:, 0] = self.casa_delays[0] * 1e9  # Fixed
        delays_ns[:, 1:] = delays_free_flat * 1e9  # Sampled

        casa_ns = self.casa_delays * 1e9

        print(
            f"\n{'Ant':<5} {'Mean':<10} {'Std':<10} {'95% CI':<25} {'CASA':<10} {'Diff':<10} {'Status'}"
        )
        print("-" * 95)

        for ant in range(self.n_antennas):
            if ant == 0:
                # Fixed antenna
                mean = casa_ns[ant]
                std = 0.0
                ci = [casa_ns[ant], casa_ns[ant]]
                casa = casa_ns[ant]
                diff = 0.0
                status = "FIXED (ref)"
            else:
                # Sampled antenna
                mean = np.mean(delays_ns[:, ant])
                std = np.std(delays_ns[:, ant])
                ci = np.percentile(delays_ns[:, ant], [2.5, 97.5])
                casa = casa_ns[ant]
                diff = mean - casa
                status = "free"

            print(
                f"{ant:<5} {mean:>9.3f} {std:>9.3f} [{ci[0]:>7.3f}, {ci[1]:>7.3f}] {casa:>9.3f} {diff:>9.3f}  {status}"
            )

        # Convergence
        summary = az.summary(self.trace, var_names=["delays_free"])
        max_rhat = summary["r_hat"].max()
        min_ess = summary["ess_bulk"].min()

        print(f"\nConvergence:")
        print(
            f"  max(r_hat): {max_rhat:.4f} {'✓ GOOD' if max_rhat < 1.01 else '✗ BAD'}"
        )
        print(
            f"  min(ess_bulk): {min_ess:.0f} {'✓ GOOD' if min_ess > 100 else '✗ BAD'}"
        )


def run(ms_path: str, casa_table: str, spw: int = 0, prior_bound_ns: float = 0.5):
    """Run the solver.

    Args:
        ms_path: Path to measurement set
        casa_table: Path to CASA calibration table
        spw: Spectral window (default 0)
        prior_bound_ns: Bounded walk around CASA delays in ns (default ±0.5 ns)
    """

    solver = SimpleDelaySolver(ms_path, casa_table)

    solver.load_data(spw=spw)
    solver.read_casa_delays()
    solver.estimate_phase_noise()
    solver.build_model(prior_bound_ns=prior_bound_ns)
    solver.sample()
    solver.summarize()

    return solver


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python simple_delay_solver.py <ms_path> <casa_table> [spw] [prior_bound_ns]"
        )
        print("  spw: spectral window (default 0)")
        print("  prior_bound_ns: bounded walk around CASA in ns (default ±0.5)")
        sys.exit(1)

    ms = sys.argv[1]
    cal = sys.argv[2]
    spw = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    prior_bound = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    solver = run(ms, cal, spw, prior_bound)
