#!/usr/bin/env python3
"""Unified Calibration Solver for K, G, B effects with solint support.

This module provides Bayesian calibration solving using NumPyro/JAX.
It replaces the previous BayesianDelaySampler with a more general solver.
"""

import logging
from typing import List

import arviz as az
import casatools
import numpy as np

from .casa_interface import MeasurementSetHandler
from .jax_config import configure_jax

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    numpyro = None


class CalibrationSolver:
    """Unified calibration solver for K, G, B effects."""

    def __init__(
        self, ms_path: str, max_cpu_fraction: float = 0.5, gpu_device: int = 0
    ):
        """Initialize solver.

        Args:
            ms_path: Path to measurement set
            max_cpu_fraction: Max fraction of CPU cores if GPU unavailable (default: 0.5)
            gpu_device: GPU device ID to use if available (default: 0)
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX and NumPyro required. Install with: pip install jax jaxlib numpyro"
            )

        # Configure JAX for GPU or CPU
        configure_jax(max_cpu_fraction=max_cpu_fraction, gpu_device=gpu_device)

        self.ms_path = ms_path
        self.ms_handler = MeasurementSetHandler(ms_path)

        # Data storage
        self.data = None
        self.frequencies = None
        self.n_antennas = None
        self.n_channels = None
        self.noise_sigma = None

        # Effects configuration
        self.effects = {}  # {name: {solint, calmode, prior_config, casa_values}}
        self.solint_groups = {}  # {name: [group_indices]}

        # Results
        self.mcmc = None
        self.trace = None

        logger.info(f"CalibrationSolver initialized for {ms_path}")

    def load_data(self, spw: str = "0", field: int = 0, solint: str = "inf"):
        """Load data with optional channel selection and time-averaging.

        Args:
            spw: Spectral window selection (CASA syntax, e.g., '0', '0:32', '0:30~34')
            field: Field ID
            solint: Solution interval - 'int' keeps all integrations, anything else time-averages
        """
        # Time-average for anything except solint='int'
        time_average = solint != "int"
        logger.info(
            f"Loading data: spw={spw}, field={field}, solint={solint}, time_average={time_average}"
        )

        # Get observation summary
        summary = self.ms_handler.get_observation_summary()
        self.n_antennas = summary["n_antennas"]

        # Parse spw to get base spw number for frequency info
        spw_base = int(spw.split(":")[0]) if ":" in spw else int(spw)
        spw_info = summary["frequency_info"][spw_base]

        # Read data using MeasurementSetHandler (handles selection)
        data_dict = self.ms_handler.read_visibilities(field=field, spw=spw)

        vis_obs = data_dict["data"]  # (n_pol, n_chan, n_row)
        antenna1 = data_dict["antenna1"]
        antenna2 = data_dict["antenna2"]
        time = data_dict["time"]
        flag = data_dict["flag"]

        # Parse channel selection from spw string
        all_freqs = np.array(spw_info["chan_freqs"])
        if ":" in spw:
            chan_spec = spw.split(":")[1]
            if "~" in chan_spec:
                chan_start, chan_end = map(int, chan_spec.split("~"))
                chan_slice = slice(chan_start, chan_end + 1)
            else:
                chan_idx = int(chan_spec)
                chan_slice = slice(chan_idx, chan_idx + 1)
            self.frequencies = all_freqs[chan_slice]
        else:
            chan_slice = slice(None)  # All channels
            self.frequencies = all_freqs

        # Get MODEL_DATA separately (not in standard read_visibilities)
        tb = casatools.table()
        tb.open(self.ms_path)

        all_model = tb.getcol("MODEL_DATA")
        # Apply same channel selection to MODEL_DATA
        vis_model = all_model[:, chan_slice, :]

        tb.close()

        # Get shapes
        n_pol_obs, n_chan_obs, n_row = vis_obs.shape
        n_pol_mod, n_chan_mod, _ = vis_model.shape

        logger.info(
            f"vis_obs shape: {vis_obs.shape}, vis_model shape: {vis_model.shape}"
        )

        # CASA msselect may not actually select channels - it depends on the MS structure
        # If vis_obs has all channels but we want selection, apply it here
        if ":" in spw and n_chan_obs > n_chan_mod:
            # vis_obs wasn't channel-selected by msselect, apply manually
            vis_obs = vis_obs[:, chan_slice, :]
            flag = flag[:, chan_slice, :]
            n_pol_obs, n_chan_obs, n_row = vis_obs.shape
            logger.info(f"Applied channel selection to vis_obs: {vis_obs.shape}")
        elif n_chan_obs != n_chan_mod:
            # Mismatch - use the smaller one
            n_chan_use = min(n_chan_obs, n_chan_mod)
            vis_obs = vis_obs[:, :n_chan_use, :]
            vis_model = vis_model[:, :n_chan_use, :]
            flag = flag[:, :n_chan_use, :]
            n_chan_obs = n_chan_use
            logger.warning(f"Channel mismatch, truncated to {n_chan_use}")

        n_pol = n_pol_obs
        n_chan = n_chan_obs
        self.n_channels = n_chan

        # Make sure frequencies array matches
        if len(self.frequencies) != n_chan:
            logger.info(
                f"Adjusting frequencies from {len(self.frequencies)} to {n_chan}"
            )
            self.frequencies = self.frequencies[:n_chan]

        logger.info(
            f"Final data: {n_row} rows, {n_chan} channels, {n_pol} polarizations"
        )

        if time_average:
            # Time-average per baseline - vectorized implementation
            logger.info("Time-averaging data...")

            # Filter out autocorrelations
            cross_mask = antenna1 != antenna2
            vis_obs = vis_obs[:, :, cross_mask]
            vis_model = vis_model[:, :, cross_mask]
            flag = flag[:, :, cross_mask]
            antenna1 = antenna1[cross_mask]
            antenna2 = antenna2[cross_mask]

            n_pol, n_chan, n_row = vis_obs.shape

            # Create baseline index: unique (ant1, ant2) pairs
            # Use Cantor pairing or simple encoding
            baseline_ids = antenna1 * self.n_antennas + antenna2
            unique_baselines, inverse_idx = np.unique(baseline_ids, return_inverse=True)
            n_baselines = len(unique_baselines)

            # Vectorized averaging using np.bincount
            vis_obs_avg = np.zeros((n_pol, n_chan, n_baselines), dtype=complex)
            vis_model_avg = np.zeros((n_pol, n_chan, n_baselines), dtype=complex)
            counts = np.zeros((n_chan, n_baselines), dtype=int)

            for pol in range(n_pol):
                for chan in range(n_chan):
                    # Mask flagged data
                    valid = ~flag[pol, chan, :]
                    if np.any(valid):
                        # Sum real and imaginary separately for bincount
                        obs_real = np.bincount(
                            inverse_idx[valid],
                            weights=vis_obs[pol, chan, valid].real,
                            minlength=n_baselines,
                        )
                        obs_imag = np.bincount(
                            inverse_idx[valid],
                            weights=vis_obs[pol, chan, valid].imag,
                            minlength=n_baselines,
                        )
                        mod_real = np.bincount(
                            inverse_idx[valid],
                            weights=vis_model[pol, chan, valid].real,
                            minlength=n_baselines,
                        )
                        mod_imag = np.bincount(
                            inverse_idx[valid],
                            weights=vis_model[pol, chan, valid].imag,
                            minlength=n_baselines,
                        )

                        vis_obs_avg[pol, chan, :] = obs_real + 1j * obs_imag
                        vis_model_avg[pol, chan, :] = mod_real + 1j * mod_imag

                        if pol == 0:  # Count once per channel
                            counts[chan, :] = np.bincount(
                                inverse_idx[valid], minlength=n_baselines
                            )

            # Average by counts
            counts = np.maximum(counts, 1)  # Avoid div by zero
            for pol in range(n_pol):
                vis_obs_avg[pol, :, :] /= counts
                vis_model_avg[pol, :, :] /= counts

            # Reconstruct antenna indices from baseline IDs
            ant1_avg = unique_baselines // self.n_antennas
            ant2_avg = unique_baselines % self.n_antennas

            self.data = {
                "vis_obs": vis_obs_avg,
                "vis_model": vis_model_avg,
                "antenna1": ant1_avg.astype(int),
                "antenna2": ant2_avg.astype(int),
                "time": np.zeros(n_baselines),  # Time-averaged, no meaningful time
                "flag": np.zeros((n_pol, n_chan, n_baselines), dtype=bool),
                "time_averaged": True,
            }

            logger.info(f"Time-averaged to {n_baselines} baselines")
        else:
            # Store raw data
            self.data = {
                "vis_obs": vis_obs,
                "vis_model": vis_model,
                "antenna1": antenna1,
                "antenna2": antenna2,
                "time": time,
                "flag": flag,
                "time_averaged": False,
            }

        n_pol, n_chan, n_row = self.data["vis_obs"].shape
        logger.info(
            f"Final data: {n_row} rows, {n_chan} channels, {n_pol} polarizations"
        )
        logger.info(f"Antennas: {self.n_antennas}")
        if len(self.frequencies) > 1:
            logger.info(
                f"Frequency range: {self.frequencies[0]/1e9:.3f} - {self.frequencies[-1]/1e9:.3f} GHz"
            )
        else:
            logger.info(f"Frequency: {self.frequencies[0]/1e9:.3f} GHz")

    def apply_corrections(self, **solutions):
        """Apply solved calibration corrections to vis_obs.

        This corrects the data with previously solved effects before
        solving for the next effect in the chain.

        Args:
            **solutions: Effect solutions keyed by name
                K: Delay solution in seconds, shape (n_antennas,)
                G: Gain solution, shape (n_antennas, 2) complex
                B: Bandpass solution, shape (n_antennas, n_chan, 2) complex
                etc.
        """
        if self.data is None:
            raise ValueError("Must call load_data() before apply_corrections()")

        from jones_sim.solvable_effects import get_effect

        vis_obs = self.data["vis_obs"]  # (n_pol, n_chan, n_row)
        ant1 = self.data["antenna1"]
        ant2 = self.data["antenna2"]
        freqs = self.frequencies
        n_pol = vis_obs.shape[0]

        for effect_name, solution in solutions.items():
            effect = get_effect(effect_name)
            vis_obs = effect.apply_inverse(
                vis_obs,
                np.array(solution),
                ant1,
                ant2,
                freqs,
                n_pol=n_pol,
            )
            logger.info(f"Applied {effect_name} correction")

        self.data["vis_obs"] = vis_obs

    def get_solution(self, effect_name: str):
        """Extract solved parameters for an effect.

        Args:
            effect_name: 'K', 'G', 'B', etc.

        Returns:
            Solution array (delays in seconds, gains as complex, etc.)
        """
        if self.trace is None:
            raise ValueError("No solution available. Run optimize() or sample() first.")

        if effect_name not in self.effects:
            raise ValueError(f"Effect {effect_name} not in solver")

        from jones_sim.solvable_effects import get_effect

        effect = get_effect(effect_name)
        casa_values = self.effects[effect_name]["casa_values"]
        return effect.extract_solution(self.trace, self.n_antennas, casa_values)

    def add_effect(
        self, name: str, solint: str = "inf", calmode: str = "ap", **prior_config
    ):
        """Add effect to solve.

        Args:
            name: Effect name ('K', 'G', 'B', 'D', etc.)
            solint: Solution interval ('inf', 'int', or time like '30s')
            calmode: Calibration mode ('p', 'a', 'ap')
            **prior_config: Effect-specific prior parameters
                K: prior_bound_ns (default 1.0)
                G: prior_std (default 0.1)
                B: prior_std (default 0.05)
                D: prior_std (default 0.01)
        """
        # Check if effect exists in registry
        from jones_sim.solvable_effects import EFFECT_REGISTRY

        if name not in EFFECT_REGISTRY:
            raise ValueError(
                f"Unknown effect: {name}. Available: {list(EFFECT_REGISTRY.keys())}"
            )

        # Default prior configs
        defaults = {
            "K": {"prior_bound_ns": 1.0},
            "G": {"prior_std": 0.1},
            "B": {"prior_std": 0.05},
            "D": {"prior_std": 0.01},
        }

        config = defaults.get(name, {})
        config.update(prior_config)
        config["solint"] = solint
        config["calmode"] = calmode
        config["casa_values"] = None  # Set by load_casa_solutions

        self.effects[name] = config
        logger.info(f"Added effect {name}: solint={solint}, calmode={calmode}")

    def load_casa_solutions(self, **tables):
        """Load CASA calibration tables as prior centers.

        Args:
            K: Path to K-table (delays)
            G: Path to G-table (gains)
            B: Path to B-table (bandpass)
        """
        tb = casatools.table()

        for effect_name, table_path in tables.items():
            if effect_name not in self.effects:
                logger.warning(
                    f"Effect {effect_name} not added, skipping table {table_path}"
                )
                continue

            logger.info(f"Loading CASA {effect_name} table: {table_path}")
            tb.open(table_path)

            if effect_name == "K":
                # Delay table: FPARAM in nanoseconds
                fparam = tb.getcol("FPARAM")
                antennas = tb.getcol("ANTENNA1")
                flags = tb.getcol("FLAG")

                # Extract delays per antenna
                delays_ns = np.zeros(self.n_antennas)
                counts = np.zeros(self.n_antennas)

                if fparam.ndim == 3:
                    n_pol, n_chan, n_rows = fparam.shape
                    for row in range(n_rows):
                        ant = antennas[row]
                        for pol in range(n_pol):
                            for chan in range(n_chan):
                                if not flags[pol, chan, row]:
                                    delays_ns[ant] += fparam[pol, chan, row]
                                    counts[ant] += 1
                else:
                    n_pol, n_rows = fparam.shape
                    for row in range(n_rows):
                        ant = antennas[row]
                        for pol in range(n_pol):
                            if not flags[pol, row]:
                                delays_ns[ant] += fparam[pol, row]
                                counts[ant] += 1

                counts[counts == 0] = 1  # Avoid division by zero
                delays_ns /= counts
                self.effects["K"]["casa_values"] = (
                    delays_ns * 1e-9
                )  # Convert to seconds

                logger.info(f"K delays loaded: std={np.std(delays_ns):.3f} ns")

            elif effect_name in ["G", "B"]:
                # Gain/bandpass table: CPARAM (complex)
                cparam = tb.getcol("CPARAM")
                antennas = tb.getcol("ANTENNA1")
                flags = tb.getcol("FLAG")

                # Shape: (n_pol, n_chan, n_rows) for B, (n_pol, 1, n_rows) for G
                n_pol, n_chan, n_rows = cparam.shape

                if effect_name == "G":
                    # Per-antenna gains
                    gains = np.ones((self.n_antennas, 2), dtype=complex)
                    for row in range(n_rows):
                        ant = antennas[row]
                        for pol in range(min(n_pol, 2)):
                            if not flags[pol, 0, row]:
                                gains[ant, pol] = cparam[pol, 0, row]
                    self.effects["G"]["casa_values"] = gains

                else:  # B
                    # Per-antenna, per-channel bandpass
                    bandpass = np.ones((self.n_antennas, n_chan, 2), dtype=complex)
                    for row in range(n_rows):
                        ant = antennas[row]
                        for pol in range(min(n_pol, 2)):
                            for chan in range(n_chan):
                                if not flags[pol, chan, row]:
                                    bandpass[ant, chan, pol] = cparam[pol, chan, row]
                    self.effects["B"]["casa_values"] = bandpass

                logger.info(f"{effect_name} solutions loaded")

            elif effect_name == "D":
                # D-term table: CPARAM (complex leakage)
                cparam = tb.getcol("CPARAM")
                antennas = tb.getcol("ANTENNA1")
                flags = tb.getcol("FLAG")

                # Shape: (n_pol, n_chan, n_rows) where n_pol=2 (Dx, Dy)
                n_pol, n_chan, n_rows = cparam.shape

                # Extract D-terms per antenna
                d_terms = np.zeros((self.n_antennas, 2), dtype=complex)
                for row in range(n_rows):
                    ant = antennas[row]
                    for pol in range(min(n_pol, 2)):
                        if not flags[pol, 0, row]:
                            d_terms[ant, pol] = cparam[pol, 0, row]

                self.effects["D"]["casa_values"] = d_terms
                logger.info(
                    f"D-terms loaded: mean amplitude={np.mean(np.abs(d_terms)):.4f}"
                )

            tb.close()

    def estimate_noise(self, default_sigma: float = 0.1):
        """Estimate thermal noise from time scatter or use default.

        Args:
            default_sigma: Default noise sigma in Jy if estimation fails
        """
        # Check if data is time-averaged
        if self.data.get("time_averaged", False):
            # Can't estimate from scatter - use default or calculate from SEFD
            # For VLA L-band: SEFD=420 Jy, 2 MHz channel, 2s integration
            # sigma = 420 / sqrt(2 * 2e6 * 2) = 0.148 Jy per visibility
            # After averaging ~150 samples: sigma_avg = 0.148 / sqrt(150) = 0.012 Jy
            # Use a reasonable default
            self.noise_sigma = default_sigma
            logger.info(
                f"Time-averaged data: using default noise sigma = {self.noise_sigma:.5f} Jy"
            )
            return

        logger.info("Estimating noise from time scatter")

        vis_obs = self.data["vis_obs"]
        ant1 = self.data["antenna1"]
        ant2 = self.data["antenna2"]
        # time = self.data["time"]
        flag = self.data["flag"]

        n_pol, n_chan, n_row = vis_obs.shape

        # Group by (ant1, ant2, chan, pol) and compute scatter
        from collections import defaultdict

        samples = defaultdict(list)

        for i in range(n_row):
            for chan in range(n_chan):
                for pol in range(n_pol):
                    if not flag[pol, chan, i]:
                        key = (ant1[i], ant2[i], chan, pol)
                        samples[key].append(vis_obs[pol, chan, i])

        stds = []
        for key, vis_list in samples.items():
            if len(vis_list) >= 3:
                vis = np.array(vis_list)
                stds.append(np.std(vis.real))
                stds.append(np.std(vis.imag))

        if len(stds) > 0:
            self.noise_sigma = np.median(stds)
        else:
            self.noise_sigma = default_sigma
            logger.warning(
                f"No valid samples for noise estimation, using default = {default_sigma}"
            )

        logger.info(f"Noise sigma: {self.noise_sigma:.5f} Jy")

    def _group_by_solint(self, solint: str) -> List[np.ndarray]:
        """Group row indices by solint.

        Args:
            solint: 'inf', 'int', or time string

        Returns:
            List of arrays, each with row indices for one interval
        """
        time = self.data["time"]
        unique_times = np.unique(time)

        if solint == "inf":
            return [np.arange(len(time))]

        elif solint == "int":
            groups = []
            for t in unique_times:
                groups.append(np.where(time == t)[0])
            return groups

        else:
            # Parse time string (e.g., '30s', '1min')
            if solint.endswith("s"):
                interval = float(solint[:-1])
            elif solint.endswith("min"):
                interval = float(solint[:-3]) * 60
            else:
                interval = float(solint)

            # Group by interval
            t0 = time.min()
            groups = []
            current_group = []
            current_end = t0 + interval

            sorted_idx = np.argsort(time)
            for idx in sorted_idx:
                if time[idx] < current_end:
                    current_group.append(idx)
                else:
                    if current_group:
                        groups.append(np.array(current_group))
                    current_group = [idx]
                    current_end = time[idx] + interval

            if current_group:
                groups.append(np.array(current_group))

            return groups

    def build_model(self):
        """Build NumPyro model for all effects.

        For time-averaged data (solint='inf'), builds a single model.
        For per-integration (solint='int'), this builds model for current interval only.
        Use sample_all_intervals() for looping over intervals.
        """
        if self.noise_sigma is None:
            self.estimate_noise()

        # Prepare JAX arrays - data is (n_pol, n_chan, n_row)
        vis_obs = self.data["vis_obs"]
        vis_model = self.data["vis_model"]
        ant1 = self.data["antenna1"]
        ant2 = self.data["antenna2"]

        n_pol, n_chan, n_row = vis_obs.shape

        # Convert to JAX and reshape to (n_row, n_chan, n_pol) for easier indexing
        vis_obs_jax = jnp.transpose(jnp.array(vis_obs), (2, 1, 0))
        vis_model_jax = jnp.transpose(jnp.array(vis_model), (2, 1, 0))
        ant1_jax = jnp.array(ant1)
        ant2_jax = jnp.array(ant2)
        freqs_jax = jnp.array(self.frequencies)

        n_ant = self.n_antennas
        noise_sigma = self.noise_sigma
        effects = self.effects

        logger.info(f"Building model: {n_row} rows, {n_chan} channels, {n_pol} pols")
        logger.info(f"Noise sigma: {noise_sigma:.5f} Jy")

        # Import effect classes
        from jones_sim.solvable_effects import get_effect

        # Create effect instances
        effect_instances = {}
        for name in effects:
            effect_instances[name] = get_effect(name)

        def model():
            # Sample parameters for each effect
            all_params = {}
            for name, cfg in effects.items():
                effect = effect_instances[name]
                casa_values = jnp.array(cfg["casa_values"])
                all_params[name] = effect.sample_params(
                    n_ant,
                    casa_values,
                    cfg,
                    freqs=freqs_jax,
                )

            # Choose likelihood based on effect type
            if len(effects) == 1 and "K" in effects:
                # Delay-only solving: unwrap phases and fit slope across frequency
                if n_chan > 1:
                    logger.info("Using unwrapped phase-slope likelihood for delays")

                    delays = all_params["K"]  # (n_ant,)

                    # Extract phases per baseline (only parallel hands: XX and YY)
                    ratio = vis_obs_jax / (vis_model_jax + 1e-10)  # (n_row, n_chan, n_pol)
                    phi_obs_all = jnp.angle(ratio)  # (n_row, n_chan, n_pol)

                    # Select only XX (pol=0) and YY (pol=3)
                    phi_obs = jnp.stack([phi_obs_all[:, :, 0], phi_obs_all[:, :, 3]], axis=2)  # (n_row, n_chan, 2)

                    # Unwrap phases across frequency axis
                    # Detect wraps: if phase jump > π, add ±2π correction
                    phi_diff = jnp.diff(phi_obs, axis=1)  # (n_row, n_chan-1, 2)

                    # Wrap detection: add 2π when jumping down, subtract 2π when jumping up
                    corrections = jnp.where(phi_diff > jnp.pi, -2*jnp.pi,
                                  jnp.where(phi_diff < -jnp.pi, 2*jnp.pi, 0.0))

                    # Cumulative unwrapping
                    cumulative_corrections = jnp.cumsum(corrections, axis=1)  # (n_row, n_chan-1, 2)

                    # Apply corrections (pad first channel with zeros since it's reference)
                    corrections_padded = jnp.concatenate([
                        jnp.zeros((n_row, 1, 2)),
                        cumulative_corrections
                    ], axis=1)  # (n_row, n_chan, 2)

                    phi_unwrapped = phi_obs + corrections_padded  # (n_row, n_chan, 2)

                    # Compute slope via least squares
                    freq_mean = jnp.mean(freqs_jax)
                    freq_centered = freqs_jax - freq_mean  # (n_chan,)
                    freq_var = jnp.sum(freq_centered**2)

                    phi_mean = jnp.mean(phi_unwrapped, axis=1, keepdims=True)  # (n_row, 1, 2)
                    phi_centered = phi_unwrapped - phi_mean  # (n_row, n_chan, 2)

                    cov = jnp.sum(freq_centered[None, :, None] * phi_centered, axis=1)  # (n_row, 2)
                    slope_obs = cov / freq_var  # (n_row, 2)

                    # Predicted slope from delays: dφ/dν = 2π(τ₁ - τ₂)
                    tau_diff = delays[ant1_jax] - delays[ant2_jax]  # (n_row,)
                    slope_pred = 2 * jnp.pi * tau_diff  # (n_row,)

                    # Estimate uncertainty in slope
                    amplitude_xx_yy = jnp.stack([
                        jnp.abs(vis_model_jax[:, :, 0]),
                        jnp.abs(vis_model_jax[:, :, 3])
                    ], axis=2)  # (n_row, n_chan, 2)

                    sigma_phase = noise_sigma / (amplitude_xx_yy + 1e-10)
                    sigma_phase_avg = jnp.mean(sigma_phase, axis=1)  # (n_row, 2)
                    sigma_slope = sigma_phase_avg / jnp.sqrt(n_chan * freq_var)

                    # Likelihood on slope
                    numpyro.sample(
                        "phase_slope",
                        dist.Normal(slope_pred[:, None], sigma_slope),
                        obs=slope_obs
                    )

                else:
                    # Single channel: fall back to cosine/sine likelihood
                    logger.info("Single channel: using cosine/sine likelihood for delays")

                    delays = all_params["K"]
                    ratio = vis_obs_jax / (vis_model_jax + 1e-10)
                    ratio_normalized = ratio / (jnp.abs(ratio) + 1e-10)
                    cos_obs = ratio_normalized.real
                    sin_obs = ratio_normalized.imag

                    tau_diff = delays[ant1_jax] - delays[ant2_jax]
                    phase_pred = 2 * jnp.pi * tau_diff[:, None] * freqs_jax[None, :]
                    cos_pred = jnp.cos(phase_pred)
                    sin_pred = jnp.sin(phase_pred)

                    amplitude = jnp.abs(vis_model_jax) + 1e-10
                    sigma_phase = noise_sigma / amplitude

                    numpyro.sample("cos_residuals", dist.Normal(cos_pred[:, :, None], sigma_phase), obs=cos_obs)
                    numpyro.sample("sin_residuals", dist.Normal(sin_pred[:, :, None], sigma_phase), obs=sin_obs)

            else:
                # Standard visibility likelihood for gains, bandpass, multi-effect
                # Forward model: apply effects to model visibilities
                # pred shape: (n_row, n_chan, n_pol)
                pred = vis_model_jax

                # Apply each effect in order
                for name in effects:
                    effect = effect_instances[name]
                    pred = effect.apply(
                        pred,
                        all_params[name],
                        ant1_jax,
                        ant2_jax,
                        freqs_jax,
                        n_pol=n_pol,
                    )

                # Likelihood - complex Gaussian as two independent normals
                numpyro.sample(
                    "obs_real",
                    dist.Normal(pred.real, noise_sigma),
                    obs=vis_obs_jax.real,
                )
                numpyro.sample(
                    "obs_imag",
                    dist.Normal(pred.imag, noise_sigma),
                    obs=vis_obs_jax.imag,
                )

        self.numpyro_model = model
        logger.info("Model built")

    def optimize(
        self, num_steps: int = 1000, learning_rate: float = 0.01, seed: int = 0, debug: bool = False, debug_file: str = None
    ):
        """Find MAP estimate using gradient descent.

        Use this instead of sample() when noise is too low for MCMC.

        Args:
            num_steps: Number of optimization steps
            learning_rate: Adam learning rate
            seed: Random seed
            debug: Print debug info during optimization
            debug_file: File path for debug output (default: delay_optimization_debug.txt)
        """
        import numpyro.optim as optim
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoDelta

        logger.info(f"Optimizing: steps={num_steps}, lr={learning_rate}")

        # Set up debug file
        debug_fp = None
        if 'K' in self.effects and debug:
            if debug_file is None:
                debug_file = "delay_optimization_debug.txt"
            debug_fp = open(debug_file, 'w')

            casa_delays_ns = self.effects['K']['casa_values'] * 1e9
            debug_fp.write(f"{'='*70}\n")
            debug_fp.write("INITIAL DELAYS (CASA)\n")
            debug_fp.write(f"{'='*70}\n")
            for ant in range(self.n_antennas):
                debug_fp.write(f"  Ant {ant:2d}: {casa_delays_ns[ant]:8.3f} ns\n")
            debug_fp.write("\n")
            debug_fp.flush()
            print(f"Debug output: {debug_file}")

        # Use delta guide for MAP estimation
        guide = AutoDelta(self.numpyro_model)
        optimizer = optim.Adam(learning_rate)
        svi = SVI(self.numpyro_model, guide, optimizer, loss=Trace_ELBO())

        # Custom update loop with progress tracking
        rng_key = jax.random.PRNGKey(seed)
        svi_state = svi.init(rng_key)
        losses = []

        for step in range(num_steps):
            svi_state, loss = svi.update(svi_state)
            losses.append(loss)

            # Write progress every 100 steps if debugging K effect
            if debug_fp is not None and step % 100 == 0:
                params = svi.get_params(svi_state)
                current_delays = guide.median(params)

                if 'delays_free' in current_delays:
                    delays_full = jnp.concatenate([
                        jnp.array([self.effects['K']['casa_values'][0]]),
                        current_delays['delays_free']
                    ])
                    delays_ns = np.array(delays_full) * 1e9

                    debug_fp.write(f"\nStep {step:4d}, Loss: {loss:.2f}\n")
                    # Write all antennas (not just problematic ones)
                    for ant in range(self.n_antennas):
                        casa_val = self.effects['K']['casa_values'][ant] * 1e9
                        diff = delays_ns[ant] - casa_val
                        marker = " ⚠" if abs(diff) > 0.1 else ""
                        debug_fp.write(f"  Ant {ant:2d}: {delays_ns[ant]:8.3f} ns (Δ={diff:+7.3f} ns from CASA){marker}\n")
                    debug_fp.flush()

        # Store losses
        class SVIResult:
            def __init__(self, losses):
                self.losses = losses
        svi_result = SVIResult(losses)

        # Extract MAP estimates
        params = svi.get_params(svi_state)
        self.map_estimates = guide.median(params)

        # Store as trace-like structure for compatibility
        self.trace = {
            k: v[None, :] if v.ndim == 1 else v[None, :, :]
            for k, v in self.map_estimates.items()
        }
        self.mcmc = None  # No MCMC object for MAP

        # Print final loss
        final_loss = svi_result.losses[-1]
        logger.info(f"Optimization complete. Final loss: {final_loss:.2f}")

        # Debug: Write final delays and compare to CASA
        if debug_fp is not None:
            debug_fp.write(f"\n{'='*70}\n")
            debug_fp.write("FINAL DELAYS vs CASA\n")
            debug_fp.write(f"{'='*70}\n")

            if 'delays_free' in self.map_estimates:
                delays_full = jnp.concatenate([
                    jnp.array([self.effects['K']['casa_values'][0]]),
                    self.map_estimates['delays_free']
                ])
                delays_ns = np.array(delays_full) * 1e9
                casa_delays_ns = self.effects['K']['casa_values'] * 1e9

                debug_fp.write(f"{'Ant':<5} {'Final (ns)':<12} {'CASA (ns)':<12} {'Diff (ns)':<12}\n")
                debug_fp.write("-" * 50 + "\n")
                for ant in range(self.n_antennas):
                    diff = delays_ns[ant] - casa_delays_ns[ant]
                    marker = " ⚠" if abs(diff) > 0.1 else ""
                    debug_fp.write(f"{ant:<5} {delays_ns[ant]:>11.3f} {casa_delays_ns[ant]:>11.3f} {diff:>+11.3f}{marker}\n")

            # Write loss progression summary
            debug_fp.write(f"\n{'='*70}\n")
            debug_fp.write("LOSS PROGRESSION\n")
            debug_fp.write(f"{'='*70}\n")
            debug_fp.write(f"Initial loss: {losses[0]:.2f}\n")
            debug_fp.write(f"Final loss:   {losses[-1]:.2f}\n")
            debug_fp.write(f"Improvement:  {losses[0] - losses[-1]:.2f}\n")

            debug_fp.close()
            print(f"✓ Debug output written to: {debug_file}")

        return self.map_estimates

    def sample(
        self,
        draws: int = 500,
        tune: int = 500,
        chains: int = 2,
        target_accept: float = 0.8,
        seed: int = 0,
    ):
        """Run NUTS sampling.

        Args:
            draws: Number of samples per chain
            tune: Number of warmup samples
            chains: Number of chains
            target_accept: Target acceptance probability
            seed: Random seed
        """
        logger.info(f"Sampling: draws={draws}, tune={tune}, chains={chains}")

        kernel = NUTS(self.numpyro_model, target_accept_prob=target_accept)
        self.mcmc = MCMC(
            kernel,
            num_warmup=tune,
            num_samples=draws,
            num_chains=chains,
        )

        self.mcmc.run(jax.random.PRNGKey(seed))
        self.trace = self.mcmc.get_samples()

        logger.info("Sampling complete")

    def get_arviz_data(self):
        """Convert to ArviZ InferenceData."""
        if self.mcmc is None:
            raise ValueError(
                "No MCMC results. Use sample() instead of optimize() for ArviZ data."
            )
        return az.from_numpyro(self.mcmc)

    def print_summary(self):
        """Print convergence and parameter summary."""
        if self.trace is None:
            raise ValueError("No results. Run sample() or optimize() first.")

        print(f"\n{'=' * 70}")
        print("CALIBRATION SOLVER SUMMARY")
        print(f"{'=' * 70}")

        is_map = self.mcmc is None

        if is_map:
            print("Method: MAP optimization")
        else:
            print("Method: MCMC sampling")
            # Get ArviZ data for diagnostics
            idata = self.get_arviz_data()

        # Check convergence for delays
        if "K" in self.effects:
            if not is_map:
                summary = az.summary(idata, var_names=["delays_free"])
                max_rhat = summary["r_hat"].max()
                min_ess = summary["ess_bulk"].min()

                print("\nDelay Convergence:")
                print(
                    f"  max(r_hat): {max_rhat:.3f} {'OK' if max_rhat < 1.01 else 'BAD'}"
                )
                print(f"  min(ESS): {min_ess:.0f} {'OK' if min_ess > 100 else 'LOW'}")

            # Print delay values
            delays_samples = self.trace["delays_free"]
            casa_delays = self.effects["K"]["casa_values"]

            if is_map:
                print(
                    f"\n{'Ant':<5} {'CASA (ns)':<12} {'MAP (ns)':<12} {'Diff (ns)':<12}"
                )
                print("-" * 45)
            else:
                print(
                    f"\n{'Ant':<5} {'CASA (ns)':<12} {'Mean (ns)':<12} {'Std (ns)':<12} {'Diff (ns)':<12}"
                )
                print("-" * 55)

            # Antenna 0 (fixed)
            if is_map:
                print(
                    f"{0:<5} {casa_delays[0]*1e9:>11.3f} {casa_delays[0]*1e9:>11.3f} {0:>11.3f}"
                )
            else:
                print(
                    f"{0:<5} {casa_delays[0]*1e9:>11.3f} {casa_delays[0]*1e9:>11.3f} {'(fixed)':>11} {0:>11.3f}"
                )

            # Other antennas
            for ant in range(1, self.n_antennas):
                casa = casa_delays[ant] * 1e9
                if is_map:
                    val = float(delays_samples[0, ant - 1]) * 1e9
                    diff = val - casa
                    print(f"{ant:<5} {casa:>11.3f} {val:>11.3f} {diff:>11.3f}")
                else:
                    mean = np.mean(delays_samples[:, ant - 1]) * 1e9
                    std = np.std(delays_samples[:, ant - 1]) * 1e9
                    diff = mean - casa
                    print(
                        f"{ant:<5} {casa:>11.3f} {mean:>11.3f} {std:>11.3f} {diff:>11.3f}"
                    )

        # Gains summary
        if "G" in self.effects:
            casa_gains = self.effects["G"]["casa_values"]
            if "gain_amp" in self.trace:
                amp_samples = self.trace["gain_amp"]
                if is_map:
                    print("\nGain Amplitudes (MAP):")
                    print(
                        f"  RMS diff from CASA: {np.sqrt(np.mean((amp_samples[0] - np.abs(casa_gains[1:]))**2)):.4f}"
                    )
                else:
                    print("\nGain Amplitudes:")
                    print(f"  Mean std: {np.mean(np.std(amp_samples, axis=0)):.4f}")

    def save_trace(self, filename: str):
        """Save results to NetCDF file.

        Args:
            filename: Output filename
        """
        if self.mcmc is not None:
            idata = self.get_arviz_data()
            idata.to_netcdf(filename)
        else:
            # For MAP, save as numpy
            np.savez(filename.replace(".nc", ".npz"), **self.trace)
        logger.info(f"Results saved to {filename}")

    # def plot_diagnostics(self, output_dir: str = "solver_output"):
    #     """Generate diagnostic dashboard with data and results plots.

    #     Args:
    #         output_dir: Directory to save plots (created if doesn't exist)
    #     """
    #     try:
    #         from bokeh.layouts import column, gridplot, row
    #         from bokeh.models import Title
    #         from bokeh.plotting import figure, output_file, save
    #     except ImportError:
    #         logger.warning("Bokeh not available for plotting")
    #         return

    #     import os

    #     # Create output directory
    #     os.makedirs(output_dir, exist_ok=True)

    #     vis_obs = self.data["vis_obs"]  # (n_pol, n_chan, n_row)
    #     vis_model = self.data["vis_model"]
    #     time = self.data["time"]
    #     is_time_avg = self.data.get("time_averaged", False)

    #     n_pol, n_chan, n_row = vis_obs.shape
    #     pol_idx = 0  # Use first polarization (XX or RR)

    #     plots = []

    #     # Plot 1: Amplitude per Baseline
    #     amp_per_baseline = np.mean(np.abs(vis_obs[pol_idx, :, :]), axis=0)
    #     p1 = figure(
    #         title="Amplitude per Baseline",
    #         x_axis_label="Baseline Index",
    #         y_axis_label="Amplitude (Jy)",
    #         width=600,
    #         height=300,
    #     )
    #     p1.scatter(
    #         list(range(n_row)),
    #         amp_per_baseline.tolist(),
    #         size=5,
    #         color="blue",
    #         alpha=0.5,
    #     )
    #     plots.append(p1)

    #     # Plot 2: Amplitude vs Frequency (if multiple channels)
    #     if n_chan > 1:
    #         amp_obs_freq = np.mean(np.abs(vis_obs[pol_idx, :, :]), axis=1)
    #         amp_model_freq = np.mean(np.abs(vis_model[pol_idx, :, :]), axis=1)
    #         freq_ghz = (self.frequencies / 1e9).tolist()

    #         p2 = figure(
    #             title="Amplitude vs Frequency",
    #             x_axis_label="Frequency (GHz)",
    #             y_axis_label="Amplitude (Jy)",
    #             width=600,
    #             height=300,
    #         )
    #         p2.line(
    #             freq_ghz, amp_obs_freq.tolist(), legend_label="Observed", color="blue"
    #         )
    #         p2.line(
    #             freq_ghz,
    #             amp_model_freq.tolist(),
    #             legend_label="Model",
    #             color="red",
    #             line_dash="dashed",
    #         )
    #         p2.legend.location = "top_right"
    #         plots.append(p2)
    #     else:
    #         # Single channel - show phase per baseline instead
    #         phase_per_baseline = np.angle(vis_obs[pol_idx, 0, :])
    #         p2 = figure(
    #             title=f"Phase per Baseline (chan 0, {self.frequencies[0]/1e9:.3f} GHz)",
    #             x_axis_label="Baseline Index",
    #             y_axis_label="Phase (rad)",
    #             width=600,
    #             height=300,
    #         )
    #         p2.scatter(
    #             list(range(n_row)),
    #             phase_per_baseline.tolist(),
    #             size=5,
    #             color="green",
    #             alpha=0.5,
    #         )
    #         plots.append(p2)

    #     # Plot 3: Delay comparison (if K effect solved)
    #     if "K" in self.effects and self.trace is not None:
    #         casa_delays = self.effects["K"]["casa_values"]
    #         delays_samples = self.trace["delays_free"]

    #         # Get recovered delays
    #         is_map = self.mcmc is None
    #         if is_map:
    #             recovered = np.concatenate([[casa_delays[0]], delays_samples[0, :]])
    #         else:
    #             recovered = np.concatenate(
    #                 [[casa_delays[0]], np.mean(delays_samples, axis=0)]
    #             )

    #         # Convert to ns
    #         casa_ns = (casa_delays * 1e9).tolist()
    #         recovered_ns = (recovered * 1e9).tolist()
    #         ant_idx = list(range(self.n_antennas))

    #         p3 = figure(
    #             title="Delay Comparison: CASA vs Recovered",
    #             x_axis_label="Antenna",
    #             y_axis_label="Delay (ns)",
    #             width=600,
    #             height=300,
    #         )
    #         p3.scatter(
    #             ant_idx, casa_ns, size=8, color="red", legend_label="CASA", alpha=0.7
    #         )
    #         p3.scatter(
    #             ant_idx,
    #             recovered_ns,
    #             size=8,
    #             color="blue",
    #             legend_label="Recovered",
    #             marker="triangle",
    #             alpha=0.7,
    #         )
    #         p3.legend.location = "top_right"
    #         plots.append(p3)

    #         # Plot 4: Delay residuals
    #         residuals = [(r - c) for r, c in zip(recovered_ns, casa_ns)]
    #         p4 = figure(
    #             title="Delay Residuals (Recovered - CASA)",
    #             x_axis_label="Antenna",
    #             y_axis_label="Residual (ns)",
    #             width=600,
    #             height=300,
    #         )
    #         p4.scatter(ant_idx, residuals, size=8, color="purple", alpha=0.7)
    #         p4.line([0, self.n_antennas - 1], [0, 0], color="black", line_dash="dashed")
    #         plots.append(p4)

    #     # Arrange in grid
    #     if len(plots) == 4:
    #         layout = gridplot([[plots[0], plots[1]], [plots[2], plots[3]]])
    #     elif len(plots) == 3:
    #         layout = gridplot([[plots[0], plots[1]], [plots[2], None]])
    #     else:
    #         layout = column(plots)

    #     # Save
    #     output_path = os.path.join(output_dir, "dashboard.html")
    #     output_file(output_path)
    #     save(layout)
    #     logger.info(f"Dashboard saved to {output_path}")
    #     print(f"Dashboard saved to {output_path}")


def run(
    ms_path: str,
    casa_k_table: str,
    output_file: str = "cal_trace.nc",
    spw: str = "0",
    solint: str = "inf",
    prior_bound_ns: float = 1.0,
    use_map: bool = False,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    plot: bool = False,
):
    """Run calibration solver for delays.

    Args:
        ms_path: Path to measurement set
        casa_k_table: Path to CASA K-table
        output_file: Output trace file
        spw: Spectral window selection (e.g., '0', '0:32' for single channel)
        solint: Solution interval ('inf', 'int', or time string)
        prior_bound_ns: Prior bound in ns
        use_map: Use MAP optimization instead of MCMC
        draws: MCMC draws
        tune: MCMC tuning steps
        chains: Number of chains
        plot: Generate diagnostic plots
    """
    solver = CalibrationSolver(ms_path)
    solver.load_data(spw=spw, solint=solint)
    solver.add_effect("K", solint=solint, prior_bound_ns=prior_bound_ns)
    solver.load_casa_solutions(K=casa_k_table)
    solver.build_model()

    if use_map:
        solver.optimize()
    else:
        solver.sample(draws=draws, tune=tune, chains=chains)

    solver.print_summary()
    solver.save_trace(output_file)

    if plot:
        solver.plot_diagnostics()

    return solver


if __name__ == "__main__":
    import argparse

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Calibration solver for delays")
    parser.add_argument("ms_path", help="Path to measurement set")
    parser.add_argument("casa_k_table", help="Path to CASA K-table")
    parser.add_argument("-o", "--output", default="cal_trace.nc", help="Output file")
    parser.add_argument("--spw", default="0", help="SPW selection (e.g., '0', '0:32')")
    parser.add_argument("--solint", default="inf", help="Solution interval")
    parser.add_argument(
        "--prior-bound", type=float, default=1.0, help="Prior bound (ns)"
    )
    parser.add_argument("--map", action="store_true", help="Use MAP instead of MCMC")
    parser.add_argument("--draws", type=int, default=500, help="MCMC draws")
    parser.add_argument("--tune", type=int, default=500, help="MCMC tuning steps")
    parser.add_argument("--chains", type=int, default=2, help="Number of chains")
    parser.add_argument("--plot", action="store_true", help="Generate diagnostic plots")

    args = parser.parse_args()

    run(
        args.ms_path,
        args.casa_k_table,
        args.output,
        args.spw,
        args.solint,
        args.prior_bound,
        args.map,
        args.draws,
        args.tune,
        args.chains,
        args.plot,
    )
