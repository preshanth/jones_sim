"""Simple Jones matrix simulator coordinator."""

import numpy as np
from typing import List, Dict, Optional, Any


class JonesSimulator:
    """Simple coordinator for Jones matrix effects.

    Manages a collection of effect instances and applies them in standard order.
    """

    def __init__(self):
        self.effects: Dict[str, Any] = {}
        self.effect_order = [
            'parallactic',
            'leakage',
            'gains',
            'bandpass',
            'rotation_measure',
            'rl_delay',
            'crosshand_phase'
        ]

    def add_effect(self, name: str, effect_instance):
        """Add an effect instance to the simulator.

        Args:
            name: Effect name (should match effect_order)
            effect_instance: Instance of an effect class with jones_matrix() method
        """
        self.effects[name] = effect_instance

    def compute_jones_matrix(self, freq: float, time: float, antenna_id: int) -> np.ndarray:
        """Compute total Jones matrix by multiplying effects in order.

        Args:
            freq: Frequency in Hz
            time: Time in seconds
            antenna_id: Antenna index

        Returns:
            2x2 complex Jones matrix
        """
        result = np.eye(2, dtype=complex)

        for effect_name in self.effect_order:
            if effect_name in self.effects:
                effect_matrix = self.effects[effect_name].jones_matrix(freq, time, antenna_id)
                result = result @ effect_matrix

        return result

    def corrupt_visibilities(self,
                           ideal_visibilities: np.ndarray,
                           frequencies: np.ndarray,
                           times: np.ndarray,
                           antenna1_ids: np.ndarray,
                           antenna2_ids: np.ndarray) -> np.ndarray:
        """Apply Jones corruption to ideal visibilities.

        Args:
            ideal_visibilities: Shape (N, 4) for [XX, XY, YX, YY] correlations
            frequencies: Shape (N,) frequency for each visibility
            times: Shape (N,) time for each visibility
            antenna1_ids: Shape (N,) first antenna index
            antenna2_ids: Shape (N,) second antenna index

        Returns:
            Corrupted visibilities with same shape
        """
        n_vis = len(ideal_visibilities)
        corrupted = np.zeros_like(ideal_visibilities)

        for i in range(n_vis):
            freq = frequencies[i]
            time = times[i]
            ant1 = antenna1_ids[i]
            ant2 = antenna2_ids[i]

            # Get Jones matrices for both antennas
            J1 = self.compute_jones_matrix(freq, time, ant1)
            J2 = self.compute_jones_matrix(freq, time, ant2)

            # Apply Kronecker product: (J1 ⊗ J2†) * V_ideal
            mueller = np.kron(J1, J2.conj().T)
            corrupted[i] = mueller @ ideal_visibilities[i]

        return corrupted

    def list_effects(self) -> List[str]:
        """Get list of active effect names."""
        return list(self.effects.keys())

    def remove_effect(self, name: str):
        """Remove an effect from the simulator."""
        if name in self.effects:
            del self.effects[name]

    def clear_effects(self):
        """Remove all effects."""
        self.effects.clear()