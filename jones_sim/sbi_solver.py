"""Simulation-Based Inference (SBI) solver for radio calibration.

This module provides a general framework for using neural posterior estimation
to solve for calibration parameters (gains, bandpass, delays, etc.) with full
uncertainty quantification.

Key Features:
- Works with any SolvableEffect from solvable_effects.py
- Trains neural density estimators for fast posterior inference
- Returns credible intervals (e.g., 1.21 ± 0.05)
- Supports both individual effect solving and joint inference
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch
from sbi import analysis as sbi_analysis
from sbi import utils as sbi_utils
from sbi.inference import NPE, SNPE

from .solvable_effects import SolvableEffect


class SBISimulator(ABC):
    """Base class for SBI simulators.

    A simulator generates observations (visibilities) from parameters.
    """

    @abstractmethod
    def simulate(self, params: np.ndarray) -> np.ndarray:
        """Generate observations from parameters.

        Args:
            params: Calibration parameters (e.g., bandpass gains)

        Returns:
            Simulated observations (e.g., visibilities)
        """
        pass

    @abstractmethod
    def get_prior(self) -> torch.distributions.Distribution:
        """Return prior distribution over parameters."""
        pass

    @abstractmethod
    def get_param_dim(self) -> int:
        """Return dimensionality of parameter space."""
        pass

    @abstractmethod
    def get_obs_dim(self) -> int:
        """Return dimensionality of observation space."""
        pass


class EffectSBISimulator(SBISimulator):
    """SBI simulator for a single calibration effect.

    This wraps a SolvableEffect and provides the interface needed for SBI.
    """

    def __init__(
        self,
        effect: SolvableEffect,
        visibility_model: Callable,
        n_antennas: int,
        n_channels: int = 1,
        n_baselines: Optional[int] = None,
        prior_config: Optional[Dict[str, Any]] = None,
        freqs: Optional[np.ndarray] = None,
        noise_std: float = 0.01,
    ):
        """Initialize effect-based simulator.

        Args:
            effect: SolvableEffect instance (e.g., BandpassEffect())
            visibility_model: Function that generates true visibilities
            n_antennas: Number of antennas
            n_channels: Number of frequency channels
            n_baselines: Number of baselines (default: n_antennas*(n_antennas-1)/2)
            prior_config: Configuration for effect priors
            freqs: Frequency array in Hz
            noise_std: Noise standard deviation
        """
        self.effect = effect
        self.visibility_model = visibility_model
        self.n_antennas = n_antennas
        self.n_channels = n_channels
        self.n_baselines = n_baselines or (n_antennas * (n_antennas - 1)) // 2
        self.prior_config = prior_config or effect.get_default_prior_config()
        self.freqs = freqs if freqs is not None else np.linspace(1e9, 2e9, n_channels)
        self.noise_std = noise_std

        # Setup antenna pairs
        ant1_list, ant2_list = [], []
        for i in range(n_antennas):
            for j in range(i + 1, n_antennas):
                ant1_list.append(i)
                ant2_list.append(j)
        self.ant1 = np.array(ant1_list)
        self.ant2 = np.array(ant2_list)

    def simulate(self, params: np.ndarray) -> np.ndarray:
        """Simulate visibilities given calibration parameters.

        Args:
            params: Effect parameters (shape depends on effect type)

        Returns:
            Complex visibilities flattened to real vector
        """
        # Generate true model visibilities
        true_vis = self.visibility_model(
            self.ant1, self.ant2, self.freqs, self.n_antennas
        )  # Shape: (n_baselines, n_channels, n_pol)

        # Apply calibration effect corruption
        corrupted_vis = self.effect.apply(
            jnp.array(true_vis),
            jnp.array(params),
            jnp.array(self.ant1),
            jnp.array(self.ant2),
            jnp.array(self.freqs),
        )

        # Add noise
        noise_real = np.random.normal(0, self.noise_std, corrupted_vis.shape)
        noise_imag = np.random.normal(0, self.noise_std, corrupted_vis.shape)
        noisy_vis = corrupted_vis + noise_real + 1j * noise_imag

        # Flatten to real vector: [real parts, imag parts]
        vis_real = np.real(noisy_vis).flatten()
        vis_imag = np.imag(noisy_vis).flatten()

        return np.concatenate([vis_real, vis_imag])

    def get_prior(self) -> torch.distributions.Distribution:
        """Return prior distribution over parameters.

        This creates a simple uniform prior. Override for custom priors.
        """
        # This is effect-specific and should be overridden
        # For now, return a placeholder
        param_dim = self.get_param_dim()
        return sbi_utils.BoxUniform(
            low=-torch.ones(param_dim),
            high=torch.ones(param_dim)
        )

    @abstractmethod
    def get_param_dim(self) -> int:
        """Return parameter dimensionality."""
        pass

    def get_obs_dim(self) -> int:
        """Return observation dimensionality."""
        # 2 * (real + imag) * n_baselines * n_channels * n_pol
        n_pol = 4  # Default to 4 polarizations
        return 2 * self.n_baselines * self.n_channels * n_pol


class BandpassSBISimulator(EffectSBISimulator):
    """SBI simulator specifically for bandpass calibration.

    Bandpass parameters: complex gains per antenna, channel, and polarization.
    Shape: (n_antennas, n_channels, n_pol)
    """

    def get_param_dim(self) -> int:
        """Bandpass has (n_antennas-1) * n_channels * 2 * 2 parameters.

        We exclude the reference antenna and parameterize as:
        - amplitude (log-space, 1 param)
        - phase (1 param)
        per channel and polarization.
        """
        n_pol = 2  # XX and YY
        # (n_antennas - 1) antennas * n_channels * n_pol * 2 (amp, phase)
        return (self.n_antennas - 1) * self.n_channels * n_pol * 2

    def get_prior(self) -> torch.distributions.Distribution:
        """Bandpass prior: amplitude ~1, phase ~0."""
        param_dim = self.get_param_dim()
        n_pol = 2
        n_params_per_pol = param_dim // 2

        # Amplitude prior: log-normal around 1.0, std 0.1
        # In log-space: N(0, 0.1)
        amp_low = np.log(0.7) * np.ones(n_params_per_pol)
        amp_high = np.log(1.5) * np.ones(n_params_per_pol)

        # Phase prior: uniform [-pi, pi]
        phase_low = -np.pi * np.ones(n_params_per_pol)
        phase_high = np.pi * np.ones(n_params_per_pol)

        low = np.concatenate([amp_low, phase_low])
        high = np.concatenate([amp_high, phase_high])

        return sbi_utils.BoxUniform(
            low=torch.tensor(low, dtype=torch.float32),
            high=torch.tensor(high, dtype=torch.float32)
        )

    def params_to_bandpass(self, params: np.ndarray) -> np.ndarray:
        """Convert flat parameter vector to bandpass array.

        Args:
            params: Flat parameter vector from SBI

        Returns:
            Bandpass array of shape (n_antennas, n_channels, 2)
        """
        n_pol = 2
        n_free_ants = self.n_antennas - 1

        # Split into amplitudes and phases
        mid = len(params) // 2
        log_amps = params[:mid].reshape(n_free_ants, self.n_channels, n_pol)
        phases = params[mid:].reshape(n_free_ants, self.n_channels, n_pol)

        # Convert to complex
        bp_free = np.exp(log_amps) * np.exp(1j * phases)

        # Add reference antenna (all 1s)
        bp_ref = np.ones((1, self.n_channels, n_pol), dtype=complex)
        bandpass = np.concatenate([bp_ref, bp_free], axis=0)

        return bandpass

    def simulate(self, params: np.ndarray) -> np.ndarray:
        """Simulate visibilities with bandpass corruption."""
        # Convert params to bandpass array
        bandpass = self.params_to_bandpass(params)

        # Generate true model visibilities
        true_vis = self.visibility_model(
            self.ant1, self.ant2, self.freqs, self.n_antennas
        )

        # Apply bandpass corruption
        corrupted_vis = self.effect.apply(
            jnp.array(true_vis),
            jnp.array(bandpass),
            jnp.array(self.ant1),
            jnp.array(self.ant2),
            jnp.array(self.freqs),
        )

        # Add noise
        noise_real = np.random.normal(0, self.noise_std, corrupted_vis.shape)
        noise_imag = np.random.normal(0, self.noise_std, corrupted_vis.shape)
        noisy_vis = np.array(corrupted_vis) + noise_real + 1j * noise_imag

        # Flatten to real vector
        vis_real = np.real(noisy_vis).flatten()
        vis_imag = np.imag(noisy_vis).flatten()

        return np.concatenate([vis_real, vis_imag])


class SBICalibrationSolver:
    """General SBI-based calibration solver.

    This class trains neural density estimators to learn posteriors
    p(calibration_params | visibilities) for fast inference with uncertainties.

    Example:
        >>> from jones_sim.solvable_effects import BandpassEffect
        >>> solver = SBICalibrationSolver(simulator, n_rounds=3)
        >>> solver.train(n_simulations=10000)
        >>> posterior = solver.infer(observed_vis)
        >>> samples = posterior.sample((1000,))
        >>> # Get credible interval
        >>> mean = samples.mean(dim=0)
        >>> std = samples.std(dim=0)
    """

    def __init__(
        self,
        simulator: SBISimulator,
        n_rounds: int = 1,
        density_estimator: str = "maf",
        device: str = "cpu",
    ):
        """Initialize SBI solver.

        Args:
            simulator: SBISimulator instance
            n_rounds: Number of sequential rounds (SNPE-C)
            density_estimator: Neural density estimator type
                - "maf": Masked Autoregressive Flow
                - "nsf": Neural Spline Flow
                - "mdn": Mixture Density Network
            device: torch device ("cpu", "cuda", "mps")
        """
        self.simulator = simulator
        self.n_rounds = n_rounds
        self.density_estimator = density_estimator
        self.device = device

        # Get prior
        self.prior = simulator.get_prior()

        # Initialize inference object
        self.inference = SNPE(
            prior=self.prior,
            density_estimator=density_estimator,
            device=device,
        )

        self.posterior = None
        self.training_history = []

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
            training_batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            show_progress_bars: Show progress during training
        """
        for round_idx in range(self.n_rounds):
            print(f"\n=== Training Round {round_idx + 1}/{self.n_rounds} ===")

            # Sample parameters from prior (or proposal for round > 0)
            if round_idx == 0:
                theta = self.prior.sample((n_simulations,))
            else:
                theta = self.posterior.sample((n_simulations,))

            # Run simulations
            print(f"Running {n_simulations} simulations...")
            x = []
            for i in range(n_simulations):
                if show_progress_bars and i % 100 == 0:
                    print(f"  {i}/{n_simulations}")
                params_np = theta[i].numpy()
                obs = self.simulator.simulate(params_np)
                x.append(obs)

            x = torch.tensor(np.array(x), dtype=torch.float32)

            # Append to inference
            self.inference = self.inference.append_simulations(theta, x)

            # Train
            print("Training neural network...")
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

        print("\n=== Training Complete ===")

    def infer(
        self,
        observed_data: np.ndarray,
        num_samples: int = 10000,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Perform inference on observed data.

        Args:
            observed_data: Observed visibilities (flattened real vector)
            num_samples: Number of posterior samples

        Returns:
            samples: Posterior samples, shape (num_samples, param_dim)
            summary: Dictionary with statistics:
                - "mean": Posterior mean
                - "std": Posterior standard deviation
                - "median": Posterior median
                - "credible_interval_68": 68% credible interval (±1σ)
                - "credible_interval_95": 95% credible interval (±2σ)
        """
        if self.posterior is None:
            raise ValueError("Must train before inference. Call .train() first.")

        # Convert to torch
        x_obs = torch.tensor(observed_data, dtype=torch.float32)

        # Sample from posterior
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

        return samples_np, summary

    def save(self, path: Union[str, Path]) -> None:
        """Save trained posterior to disk.

        Args:
            path: Path to save location
        """
        if self.posterior is None:
            raise ValueError("No trained posterior to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save using pickle
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "posterior": self.posterior,
                "inference": self.inference,
                "prior": self.prior,
                "training_history": self.training_history,
            }, f)

    def load(self, path: Union[str, Path]) -> None:
        """Load trained posterior from disk.

        Args:
            path: Path to saved posterior
        """
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.posterior = data["posterior"]
        self.inference = data["inference"]
        self.prior = data["prior"]
        self.training_history = data.get("training_history", [])


class GainSBISimulator(EffectSBISimulator):
    """SBI simulator for gain calibration.

    Gain parameters: complex gains per antenna and polarization.
    Shape: (n_antennas, 2) for XX and YY pols
    """

    def get_param_dim(self) -> int:
        """Gains have (n_antennas-1) * 2 * 2 parameters.

        2 polarizations, 2 params each (amp, phase), minus reference antenna.
        """
        n_pol = 2
        return (self.n_antennas - 1) * n_pol * 2

    def get_prior(self) -> torch.distributions.Distribution:
        """Gain prior: amplitude ~1, phase ~0."""
        param_dim = self.get_param_dim()
        n_params = param_dim // 2

        # Amplitude: log-normal around 1
        amp_low = np.log(0.5) * np.ones(n_params)
        amp_high = np.log(2.0) * np.ones(n_params)

        # Phase: uniform [-pi, pi]
        phase_low = -np.pi * np.ones(n_params)
        phase_high = np.pi * np.ones(n_params)

        low = np.concatenate([amp_low, phase_low])
        high = np.concatenate([amp_high, phase_high])

        return sbi_utils.BoxUniform(
            low=torch.tensor(low, dtype=torch.float32),
            high=torch.tensor(high, dtype=torch.float32)
        )

    def params_to_gains(self, params: np.ndarray) -> np.ndarray:
        """Convert flat parameters to gain array.

        Returns:
            Gains of shape (n_antennas, 2)
        """
        n_pol = 2
        mid = len(params) // 2
        log_amps = params[:mid].reshape(-1, n_pol)
        phases = params[mid:].reshape(-1, n_pol)

        gains_free = np.exp(log_amps) * np.exp(1j * phases)
        gains_ref = np.ones((1, n_pol), dtype=complex)

        return np.concatenate([gains_ref, gains_free], axis=0)

    def simulate(self, params: np.ndarray) -> np.ndarray:
        """Simulate visibilities with gain corruption."""
        gains = self.params_to_gains(params)

        true_vis = self.visibility_model(
            self.ant1, self.ant2, self.freqs, self.n_antennas
        )

        corrupted_vis = self.effect.apply(
            jnp.array(true_vis),
            jnp.array(gains),
            jnp.array(self.ant1),
            jnp.array(self.ant2),
            jnp.array(self.freqs),
        )

        noise_real = np.random.normal(0, self.noise_std, corrupted_vis.shape)
        noise_imag = np.random.normal(0, self.noise_std, corrupted_vis.shape)
        noisy_vis = np.array(corrupted_vis) + noise_real + 1j * noise_imag

        vis_real = np.real(noisy_vis).flatten()
        vis_imag = np.imag(noisy_vis).flatten()

        return np.concatenate([vis_real, vis_imag])
