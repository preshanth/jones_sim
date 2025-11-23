"""SBI-based bandpass calibration solver.

This module provides Simulation-Based Inference for bandpass calibration,
designed to work with the same data format as CalibrationSolver.

Key features:
- Follows CalibrationSolver data format (time-averaged baselines)
- Uses JonesSimulator.predict_visibilities() for corruption
- VLA-realistic priors (L-band: SEFD=420 Jy)
- Returns full posterior with credible intervals

Fixed for prototype:
- 64 channels (configurable but network is trained for specific n_chan)
- 27 antennas (VLA D-config)
- Independent channel priors (smoothness models planned for future)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from sbi import utils as sbi_utils
from sbi.inference import SNPE

logger = logging.getLogger(__name__)

try:
    import jax.numpy as jnp
    from jones_sim.simulator import JonesSimulator
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None


# VLA instrument specs
VLA_SEFD = {
    'L': 420.0,   # Jy at 1.4 GHz
    'S': 370.0,   # Jy at 3 GHz
    'C': 310.0,   # Jy at 6 GHz
    'X': 250.0,   # Jy at 10 GHz
}

VLA_BANDPASS_PRIOR = {
    'amp_range': (0.5, 2.0),      # Realistic amplitude variation
    'phase_range': (-np.pi, np.pi),
    'smoothness': 'independent',   # For now; future: 'fourier', 'gp', 'ar'
}


class BandpassSBISimulator:
    """SBI simulator for bandpass calibration.

    Generates training data by:
    1. Sampling bandpass from VLA-realistic prior
    2. Generating 3C286 model visibilities
    3. Corrupting with predict_visibilities()
    4. Adding thermal noise (SEFD-based)
    5. Time-averaging per baseline (like CalibrationSolver)

    Args:
        n_antennas: Number of antennas (default: 27 for VLA)
        n_channels: Number of frequency channels (fixed for network)
        frequencies: Frequency array in Hz
        vla_band: VLA band for SEFD ('L', 'S', 'C', 'X')
        int_time: Integration time in seconds
        chan_width: Channel width in Hz
        source_flux: 3C286 flux in Jy (default: 15 Jy at L-band)
        prior_config: Override default prior configuration
    """

    def __init__(
        self,
        n_antennas: int = 27,
        n_channels: int = 64,
        frequencies: Optional[np.ndarray] = None,
        vla_band: str = 'L',
        int_time: float = 2.0,
        chan_width: float = 15.625e6,  # 15.625 MHz for 1 GHz BW / 64 chan
        source_flux: float = 15.0,
        prior_config: Optional[Dict[str, Any]] = None,
    ):
        if not JAX_AVAILABLE:
            raise ImportError("JAX required for SBI simulator")

        self.n_antennas = n_antennas
        self.n_channels = n_channels
        self.vla_band = vla_band
        self.int_time = int_time
        self.chan_width = chan_width
        self.source_flux = source_flux

        # Frequency setup
        if frequencies is None:
            # Default L-band: 1-2 GHz
            self.frequencies = np.linspace(1.0e9, 2.0e9, n_channels)
        else:
            self.frequencies = frequencies
            assert len(frequencies) == n_channels

        # Prior configuration
        self.prior_config = prior_config or VLA_BANDPASS_PRIOR.copy()

        # SEFD for noise calculation
        self.sefd = VLA_SEFD.get(vla_band, 420.0)

        # Setup baseline pairs
        self.ant1, self.ant2 = self._generate_baselines()
        self.n_baselines = len(self.ant1)

        # Initialize JonesSimulator for corruption
        self.jones_sim = JonesSimulator()

        logger.info(f"BandpassSBISimulator initialized:")
        logger.info(f"  Antennas: {n_antennas}")
        logger.info(f"  Channels: {n_channels}")
        logger.info(f"  Baselines: {self.n_baselines}")
        logger.info(f"  VLA band: {vla_band} (SEFD={self.sefd} Jy)")
        logger.info(f"  Freq range: {self.frequencies[0]/1e9:.3f}-{self.frequencies[-1]/1e9:.3f} GHz")

    def _generate_baselines(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate baseline pairs (cross-correlations only)."""
        ant1_list, ant2_list = [], []
        for i in range(self.n_antennas):
            for j in range(i + 1, self.n_antennas):
                ant1_list.append(i)
                ant2_list.append(j)
        return np.array(ant1_list), np.array(ant2_list)

    def get_param_dim(self) -> int:
        """Get dimensionality of parameter space.

        Bandpass: (n_antennas-1) × n_channels × 2 pols × 2 (amp, phase)
        Reference antenna (ant 0) fixed at 1+0j.
        """
        return (self.n_antennas - 1) * self.n_channels * 2 * 2

    def get_obs_dim(self) -> int:
        """Get dimensionality of observation space.

        Time-averaged visibilities: n_baselines × n_channels × 4 pols × 2 (real, imag)
        """
        return self.n_baselines * self.n_channels * 4 * 2

    def get_prior(self) -> torch.distributions.Distribution:
        """Get prior distribution over bandpass parameters.

        Independent priors per channel (for now).
        Future: Add smoothness models (Fourier, GP, AR).

        Returns:
            BoxUniform distribution over valid bandpass parameter space
        """
        param_dim = self.get_param_dim()
        n_params_per_pol = param_dim // 2  # Half for XX, half for YY

        # Amplitude: log-space, realistic VLA range
        amp_low_log = np.log(self.prior_config['amp_range'][0])
        amp_high_log = np.log(self.prior_config['amp_range'][1])
        amp_low = np.full(n_params_per_pol, amp_low_log)
        amp_high = np.full(n_params_per_pol, amp_high_log)

        # Phase: uniform over full range
        phase_low = np.full(n_params_per_pol, self.prior_config['phase_range'][0])
        phase_high = np.full(n_params_per_pol, self.prior_config['phase_range'][1])

        # Concatenate
        low = np.concatenate([amp_low, phase_low])
        high = np.concatenate([amp_high, phase_high])

        return sbi_utils.BoxUniform(
            low=torch.tensor(low, dtype=torch.float32),
            high=torch.tensor(high, dtype=torch.float32)
        )

    def flatten_bandpass(self, bandpass: np.ndarray) -> np.ndarray:
        """Convert bandpass array to flat parameter vector.

        Args:
            bandpass: Complex bandpass, shape (n_antennas, n_channels, 2)

        Returns:
            Flat vector: [log_amp_XX, log_amp_YY, phase_XX, phase_YY]
        """
        # Extract free antennas (skip reference)
        bp_free = bandpass[1:, :, :]  # (n_antennas-1, n_channels, 2)

        # Split into amplitude and phase
        log_amp = np.log(np.abs(bp_free))
        phase = np.angle(bp_free)

        # Flatten: [all XX amps, all YY amps, all XX phases, all YY phases]
        log_amp_xx = log_amp[:, :, 0].flatten()
        log_amp_yy = log_amp[:, :, 1].flatten()
        phase_xx = phase[:, :, 0].flatten()
        phase_yy = phase[:, :, 1].flatten()

        return np.concatenate([log_amp_xx, log_amp_yy, phase_xx, phase_yy])

    def unflatten_bandpass(self, flat_params: np.ndarray) -> np.ndarray:
        """Convert flat parameter vector to bandpass array.

        Args:
            flat_params: Flat vector from SBI

        Returns:
            Complex bandpass, shape (n_antennas, n_channels, 2)
        """
        n_free = self.n_antennas - 1
        n_params_per_pol = n_free * self.n_channels

        # Split into components
        log_amp_xx = flat_params[0:n_params_per_pol].reshape(n_free, self.n_channels)
        log_amp_yy = flat_params[n_params_per_pol:2*n_params_per_pol].reshape(n_free, self.n_channels)
        phase_xx = flat_params[2*n_params_per_pol:3*n_params_per_pol].reshape(n_free, self.n_channels)
        phase_yy = flat_params[3*n_params_per_pol:].reshape(n_free, self.n_channels)

        # Reconstruct complex bandpass for free antennas
        bp_free = np.zeros((n_free, self.n_channels, 2), dtype=complex)
        bp_free[:, :, 0] = np.exp(log_amp_xx) * np.exp(1j * phase_xx)
        bp_free[:, :, 1] = np.exp(log_amp_yy) * np.exp(1j * phase_yy)

        # Add reference antenna (all 1+0j)
        bp_ref = np.ones((1, self.n_channels, 2), dtype=complex)
        bandpass = np.concatenate([bp_ref, bp_free], axis=0)

        return bandpass

    def _generate_model_visibilities(self) -> np.ndarray:
        """Generate uncorrupted 3C286 model visibilities.

        Point source at phase center with specified flux.

        Returns:
            Model visibilities, shape (n_baselines, n_channels, 4)
        """
        # Point source: constant flux across baselines/channels
        # Shape: (n_baselines, n_channels, 4 correlations)
        model_vis = np.ones((self.n_baselines, self.n_channels, 4), dtype=complex)
        model_vis *= self.source_flux

        return model_vis

    def _add_thermal_noise(self, vis: np.ndarray) -> np.ndarray:
        """Add realistic thermal noise based on VLA SEFD.

        Uses radiometer equation:
        sigma = SEFD / sqrt(2 * bandwidth * int_time * n_pol)

        Args:
            vis: Visibilities, shape (n_baselines, n_channels, 4)

        Returns:
            Noisy visibilities
        """
        # Radiometer equation per visibility
        sigma = self.sefd / np.sqrt(2 * self.chan_width * self.int_time * 2)

        # Complex Gaussian noise
        sigma_complex = sigma / np.sqrt(2)
        noise = (np.random.normal(0, sigma_complex, vis.shape) +
                 1j * np.random.normal(0, sigma_complex, vis.shape))

        return vis + noise

    def simulate(self, flat_params: np.ndarray) -> np.ndarray:
        """Simulate time-averaged visibilities given bandpass parameters.

        This is the core simulation for SBI training:
        1. Unflatten parameters to bandpass
        2. Generate clean model visibilities
        3. Corrupt using predict_visibilities()
        4. Add thermal noise
        5. Format for SBI (flatten real/imag)

        Args:
            flat_params: Flat parameter vector from prior

        Returns:
            Flattened observation vector (real, then imag parts)
        """
        # 1. Convert to bandpass array
        bandpass = self.unflatten_bandpass(flat_params)

        # 2. Generate model visibilities
        model_vis = self._generate_model_visibilities()

        # 3. Corrupt using predict_visibilities
        params_dict = {'B': bandpass}

        corrupted_vis = self.jones_sim.predict_visibilities(
            model_vis=model_vis,
            frequencies=self.frequencies,
            antenna1=self.ant1,
            antenna2=self.ant2,
            params=params_dict,
            use_jax=True
        )

        # 4. Add thermal noise
        noisy_vis = self._add_thermal_noise(corrupted_vis)

        # 5. Flatten for SBI: [all real parts, all imag parts]
        vis_real = np.real(noisy_vis).flatten()
        vis_imag = np.imag(noisy_vis).flatten()

        return np.concatenate([vis_real, vis_imag])


class SBIBandpassSolver:
    """SBI-based bandpass calibration solver.

    Trains a neural density estimator to learn p(bandpass | visibilities).
    After training, provides fast inference with full posterior uncertainties.

    Example:
        >>> simulator = BandpassSBISimulator(n_antennas=27, n_channels=64)
        >>> solver = SBIBandpassSolver(simulator, n_rounds=2)
        >>> solver.train(n_simulations=10000)
        >>> samples, summary = solver.infer(observed_vis)
        >>> print(f"Bandpass = {summary['mean']} ± {summary['std']}")
    """

    def __init__(
        self,
        simulator: BandpassSBISimulator,
        n_rounds: int = 2,
        density_estimator: str = "maf",
        device: str = "cpu",
    ):
        """Initialize SBI solver.

        Args:
            simulator: BandpassSBISimulator instance
            n_rounds: Number of sequential training rounds (SNPE-C)
            density_estimator: Neural density estimator type
                - "maf": Masked Autoregressive Flow (recommended)
                - "nsf": Neural Spline Flow
                - "mdn": Mixture Density Network
            device: torch device ("cpu", "cuda", "mps")
        """
        self.simulator = simulator
        self.n_rounds = n_rounds
        self.density_estimator = density_estimator
        self.device = device

        # Get prior from simulator
        self.prior = simulator.get_prior()

        # Initialize inference
        self.inference = SNPE(
            prior=self.prior,
            density_estimator=density_estimator,
            device=device,
        )

        self.posterior = None
        self.training_history = []

        logger.info(f"SBIBandpassSolver initialized:")
        logger.info(f"  Rounds: {n_rounds}")
        logger.info(f"  Estimator: {density_estimator}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Param dim: {simulator.get_param_dim()}")
        logger.info(f"  Obs dim: {simulator.get_obs_dim()}")

    def train(
        self,
        n_simulations: int = 10000,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        show_progress_bars: bool = True,
    ) -> None:
        """Train the neural density estimator.

        Args:
            n_simulations: Number of simulations per round
            training_batch_size: Batch size for neural network training
            learning_rate: Learning rate for optimizer
            show_progress_bars: Show progress during training/simulation
        """
        for round_idx in range(self.n_rounds):
            logger.info(f"\n=== Training Round {round_idx + 1}/{self.n_rounds} ===")

            # Sample parameters from prior (or proposal for round > 0)
            if round_idx == 0:
                theta = self.prior.sample((n_simulations,))
            else:
                theta = self.posterior.sample((n_simulations,))

            # Run simulations
            logger.info(f"Running {n_simulations} simulations...")
            x = []
            for i in range(n_simulations):
                if show_progress_bars and i % 100 == 0:
                    logger.info(f"  {i}/{n_simulations}")
                params_np = theta[i].numpy()
                obs = self.simulator.simulate(params_np)
                x.append(obs)

            x = torch.tensor(np.array(x), dtype=torch.float32)

            # Append to inference
            self.inference = self.inference.append_simulations(theta, x)

            # Train
            logger.info("Training neural network...")
            density_estimator = self.inference.train(
                training_batch_size=training_batch_size,
                learning_rate=learning_rate,
                show_train_summary=show_progress_bars,
            )

            # Build posterior
            self.posterior = self.inference.build_posterior(density_estimator)

            self.training_history.append({
                "round": round_idx,
                "n_simulations": n_simulations,
            })

        logger.info("\n=== Training Complete ===")

    def infer(
        self,
        observed_vis: np.ndarray,
        num_samples: int = 10000,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform inference on observed visibilities.

        Args:
            observed_vis: Observed visibilities (flattened format from simulator)
            num_samples: Number of posterior samples

        Returns:
            samples: Posterior samples of bandpass params, shape (num_samples, param_dim)
            summary: Dictionary with statistics:
                - "mean": Posterior mean
                - "std": Posterior standard deviation
                - "median": Posterior median
                - "credible_interval_68": 68% CI (±1σ)
                - "credible_interval_95": 95% CI (±2σ)
        """
        if self.posterior is None:
            raise ValueError("Must train before inference. Call .train() first.")

        # Convert to torch
        x_obs = torch.tensor(observed_vis, dtype=torch.float32)

        # Sample from posterior
        logger.info(f"Sampling {num_samples} from posterior...")
        samples = self.posterior.sample((num_samples,), x=x_obs)
        samples_np = samples.numpy()

        # Compute summary statistics
        mean = np.mean(samples_np, axis=0)
        std = np.std(samples_np, axis=0)
        median = np.median(samples_np, axis=0)

        # Credible intervals
        lower_68 = np.percentile(samples_np, 16, axis=0)
        upper_68 = np.percentile(samples_np, 84, axis=0)
        lower_95 = np.percentile(samples_np, 2.5, axis=0)
        upper_95 = np.percentile(samples_np, 97.5, axis=0)

        summary = {
            "mean": mean,
            "std": std,
            "median": median,
            "credible_interval_68": (lower_68, upper_68),
            "credible_interval_95": (lower_95, upper_95),
        }

        logger.info("Inference complete")

        return samples_np, summary

    def save(self, path: Union[str, Path]) -> None:
        """Save trained posterior to disk."""
        if self.posterior is None:
            raise ValueError("No trained posterior to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "posterior": self.posterior,
                "inference": self.inference,
                "prior": self.prior,
                "training_history": self.training_history,
                "simulator_config": {
                    "n_antennas": self.simulator.n_antennas,
                    "n_channels": self.simulator.n_channels,
                    "frequencies": self.simulator.frequencies,
                    "vla_band": self.simulator.vla_band,
                }
            }, f)

        logger.info(f"Saved posterior to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load trained posterior from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.posterior = data["posterior"]
        self.inference = data["inference"]
        self.prior = data["prior"]
        self.training_history = data.get("training_history", [])

        logger.info(f"Loaded posterior from {path}")
