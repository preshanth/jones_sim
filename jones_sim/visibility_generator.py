"""High-level visibility generator for Jones matrix simulations.

This module provides the main interface for generating corrupted visibilities
by combining source models, Jones matrix effects, and realistic noise.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

from .source_models import SourceModel
from .simulator import JonesSimulator


class VisibilityGenerator:
    """High-level interface for generating corrupted interferometric visibilities.

    Combines source models, Jones matrix corruption, and noise to produce
    realistic visibility datasets for calibration studies.
    """

    def __init__(self,
                 n_antennas: int = 4,
                 noise_std: float = 0.0,
                 random_seed: Optional[int] = None):
        """Initialize visibility generator.

        Args:
            n_antennas: Number of antennas in the array
            noise_std: Standard deviation for Gaussian noise (Jy)
            random_seed: Random seed for reproducible noise
        """
        self.n_antennas = n_antennas
        self.noise_std = noise_std
        self.random_seed = random_seed

        # Initialize Jones simulator
        self.jones_simulator = JonesSimulator()

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

    def add_jones_effect(self, name: str, effect_instance):
        """Add a Jones matrix effect to the simulation.

        Args:
            name: Effect name (should match JonesSimulator.effect_order)
            effect_instance: Instance of an effect class
        """
        self.jones_simulator.add_effect(name, effect_instance)

    def generate_baseline_data(self,
                              frequencies: np.ndarray,
                              times: np.ndarray,
                              exclude_autocorr: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate baseline coordinate arrays for all antenna pairs.

        Args:
            frequencies: Frequency array in Hz
            times: Time array in seconds
            exclude_autocorr: If True, exclude autocorrelations

        Returns:
            Tuple of (freq_grid, time_grid, ant1_ids, ant2_ids) for all baselines
        """
        # Generate all baseline pairs
        baselines = []
        for i in range(self.n_antennas):
            for j in range(self.n_antennas):
                if exclude_autocorr and i == j:
                    continue
                baselines.append((i, j))

        n_baselines = len(baselines)
        n_freq = len(frequencies)
        n_time = len(times)
        n_vis = n_baselines * n_freq * n_time

        # Create coordinate grids
        freq_grid = np.tile(np.repeat(frequencies, n_time), n_baselines)
        time_grid = np.tile(times, n_baselines * n_freq)

        # Create antenna ID arrays
        ant1_list = []
        ant2_list = []
        for _ in range(n_freq * n_time):
            for ant1, ant2 in baselines:
                ant1_list.append(ant1)
                ant2_list.append(ant2)

        ant1_ids = np.array(ant1_list)
        ant2_ids = np.array(ant2_list)

        return freq_grid, time_grid, ant1_ids, ant2_ids

    def add_noise(self, visibilities: np.ndarray, flags: Optional[np.ndarray] = None) -> np.ndarray:
        """Add Gaussian noise to visibilities, respecting flags.

        Uses Option B: σ_real = σ_imag = σ/√2 so total complex variance = σ².
        Noise is only added to unflagged data points.

        Args:
            visibilities: Complex visibility array
            flags: Optional boolean flag array (True = flagged/bad data)

        Returns:
            Visibilities with added noise (only to unflagged data)
        """
        if self.noise_std <= 0:
            return visibilities

        # Generate Gaussian noise for real and imaginary parts
        # σ_real = σ_imag = σ_total/√2 for complex variance = σ_total²
        component_std = self.noise_std / np.sqrt(2)

        noise_real = np.random.normal(0, component_std, visibilities.shape)
        noise_imag = np.random.normal(0, component_std, visibilities.shape)
        noise = noise_real + 1j * noise_imag

        # Only add noise to unflagged data
        if flags is not None:
            noise[flags] = 0.0  # Don't add noise to flagged data

        return visibilities + noise

    def generate_visibilities(self,
                             source: SourceModel,
                             frequencies: np.ndarray,
                             times: np.ndarray,
                             flags: Optional[np.ndarray] = None,
                             exclude_autocorr: bool = True,
                             add_noise: bool = True) -> Dict[str, np.ndarray]:
        """Generate corrupted visibilities for a given source.

        Args:
            source: Source model instance
            frequencies: Frequency array in Hz
            times: Time array in seconds
            flags: Optional flag array [N, 4] for [XX, XY, YX, YY] (True = flagged)
            exclude_autocorr: If True, exclude autocorrelations
            add_noise: If True, add Gaussian noise

        Returns:
            Dictionary containing:
                - 'visibilities': Corrupted visibility array [N, 4] for [XX, XY, YX, YY]
                - 'flags': Flag array [N, 4] (True = flagged/bad data)
                - 'frequencies': Frequency for each visibility
                - 'times': Time for each visibility
                - 'antenna1': First antenna ID for each visibility
                - 'antenna2': Second antenna ID for each visibility
                - 'ideal_visibilities': Uncorrupted visibilities (for comparison)
        """
        # Generate baseline coordinates
        freq_grid, time_grid, ant1_ids, ant2_ids = self.generate_baseline_data(
            frequencies, times, exclude_autocorr
        )

        n_vis = len(freq_grid)

        # Get ideal visibilities from source model
        ideal_correlations = source.linear_correlations()

        # Replicate for all visibilities (same source for all baselines/times/freqs)
        ideal_visibilities = np.tile(ideal_correlations, (n_vis, 1))

        # Generate or propagate flags
        if flags is None:
            # Create default flags (all unflagged)
            visibility_flags = np.zeros((n_vis, 4), dtype=bool)
        else:
            # Ensure flags match visibility array shape
            if flags.shape != (n_vis, 4):
                raise ValueError(f"Flag array shape {flags.shape} doesn't match expected {(n_vis, 4)}")
            visibility_flags = flags.copy()

        # Apply Jones matrix corruption
        corrupted_visibilities = self.jones_simulator.corrupt_visibilities(
            ideal_visibilities, freq_grid, time_grid, ant1_ids, ant2_ids
        )

        # Add noise if requested (respecting flags)
        if add_noise:
            corrupted_visibilities = self.add_noise(corrupted_visibilities, visibility_flags)

        return {
            'visibilities': corrupted_visibilities,
            'flags': visibility_flags,
            'frequencies': freq_grid,
            'times': time_grid,
            'antenna1': ant1_ids,
            'antenna2': ant2_ids,
            'ideal_visibilities': ideal_visibilities
        }

    def compare_effects(self,
                       source: SourceModel,
                       frequencies: np.ndarray,
                       times: np.ndarray,
                       effect_names: Optional[List[str]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate visibilities with individual effects to study systematic corruption.

        Args:
            source: Source model instance
            frequencies: Frequency array in Hz
            times: Time array in seconds
            effect_names: List of effect names to test individually. If None, uses all active effects.

        Returns:
            Dictionary with keys:
                - 'baseline': Visibilities with no corruption
                - 'all_effects': Visibilities with all effects combined
                - Individual effect names: Visibilities with only that effect
        """
        if effect_names is None:
            effect_names = self.jones_simulator.list_effects()

        results = {}

        # Store original effects
        original_effects = self.jones_simulator.effects.copy()

        try:
            # Baseline: no corruption
            self.jones_simulator.clear_effects()
            results['baseline'] = self.generate_visibilities(
                source, frequencies, times, add_noise=False
            )

            # Individual effects
            for effect_name in effect_names:
                if effect_name in original_effects:
                    self.jones_simulator.clear_effects()
                    self.jones_simulator.add_effect(effect_name, original_effects[effect_name])
                    results[effect_name] = self.generate_visibilities(
                        source, frequencies, times, add_noise=False
                    )

            # All effects combined
            self.jones_simulator.clear_effects()
            for name, effect in original_effects.items():
                self.jones_simulator.add_effect(name, effect)

            results['all_effects'] = self.generate_visibilities(
                source, frequencies, times, add_noise=False
            )

        finally:
            # Restore original effects
            self.jones_simulator.clear_effects()
            for name, effect in original_effects.items():
                self.jones_simulator.add_effect(name, effect)

        return results

    def get_corruption_summary(self,
                              source: SourceModel,
                              frequency: float = 1e9,
                              time: float = 0.0) -> Dict[str, np.ndarray]:
        """Get summary of Jones matrix corruption at specific frequency/time.

        Args:
            source: Source model instance
            frequency: Frequency in Hz
            time: Time in seconds

        Returns:
            Dictionary with corruption matrices and source information
        """
        # Get source properties
        I, Q, U, V = source.stokes_parameters()
        ideal_corr = source.linear_correlations()

        # Get Jones matrices for each antenna
        jones_matrices = {}
        for ant_id in range(self.n_antennas):
            jones_matrices[f'antenna_{ant_id}'] = self.jones_simulator.compute_jones_matrix(
                frequency, time, ant_id
            )

        return {
            'stokes_parameters': np.array([I, Q, U, V]),
            'ideal_correlations': ideal_corr,
            'jones_matrices': jones_matrices,
            'frequency': frequency,
            'time': time
        }


# Convenience functions for common simulation scenarios

def quick_unpolarized_sim(frequencies: np.ndarray,
                         times: np.ndarray,
                         jones_effects: Dict[str, object],
                         noise_std: float = 0.01) -> Dict[str, np.ndarray]:
    """Quick simulation of unpolarized source with specified Jones effects."""
    from .source_models import create_unpolarized_source

    generator = VisibilityGenerator(noise_std=noise_std)

    # Add Jones effects
    for name, effect in jones_effects.items():
        generator.add_jones_effect(name, effect)

    # Generate visibilities
    source = create_unpolarized_source(1.0)
    return generator.generate_visibilities(source, frequencies, times)


def quick_polarized_sim(frequencies: np.ndarray,
                       times: np.ndarray,
                       jones_effects: Dict[str, object],
                       pol_type: str = 'linear',
                       pol_fraction: float = 0.05,
                       noise_std: float = 0.01) -> Dict[str, np.ndarray]:
    """Quick simulation of polarized source with specified Jones effects."""
    from .source_models import create_linear_source, create_circular_source

    generator = VisibilityGenerator(noise_std=noise_std)

    # Add Jones effects
    for name, effect in jones_effects.items():
        generator.add_jones_effect(name, effect)

    # Create source
    if pol_type == 'linear':
        source = create_linear_source(1.0, pol_fraction * 100, 30.0)
    elif pol_type == 'circular':
        source = create_circular_source(1.0, pol_fraction * 100, 2.0)
    else:
        raise ValueError("pol_type must be 'linear' or 'circular'")

    return generator.generate_visibilities(source, frequencies, times)